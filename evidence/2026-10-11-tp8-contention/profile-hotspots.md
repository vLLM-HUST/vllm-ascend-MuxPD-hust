# TP8 host/device hotspot audit

The current fenced plugin's long P slices are the clearest decode-latency target. The old-plugin scheduler trace records a 12,000-token cohort advancing **21 layers in a 248.31 ms worker round trip**, then **19 layers in 207.80 ms**, while one D request is active. These are profiled, whole-step host measurements, including D and RPC work; they are not isolated P kernel times. They support measuring and bounding raw P slices, not a numerical unprofiled SLO guarantee.

Sources are `old-profile-trace-26680.jsonl` steps 30–31 and the 16 per-rank traces named and SHA256-pinned in [profile-hotspots.json](profile-hotspots.json). Reproduce with `python local-results/tp8-contention/profile-hotspots-audit.py` from the repository root. The audit imports the repository's device-task classifier and records its SHA256 too. No NPU operations or API requests are performed by this audit.

## Captured work and attribution

Every rank agrees on these counts:

| Complete exported events | Old plugin | Native |
| --- | ---: | ---: |
| CPU MoE custom-op scopes | 80 | 80 |
| CPU GDN custom-op scopes | 60 | 60 |
| Prefill-containing forward equivalents (40 MoE / 30 GDN layers each) | 2 | 2 |
| Actual `npu_fx_compiler inference` scopes | 1 | 2 |
| Actual `aclmdlRIExecuteAsync` D graph replays | 7 | 6 |
| Attention graph task-update begin/end pairs | 70 | 60 |
| Synchronous sampler event waits | 9 | 8 |
| AI Core / VECTOR / MIX device kernel rows | 16,750 | 15,655 |

The plugin executes its P128 forward through FX and splits P12000 into raw layer slices. Native has two FX scopes. An FX scope can contain many native custom-op host launches; it is not proof of one fused device graph. The native prefill-containing forward may include D work, while the plugin uses separate forwards. Thus identical client requests and eight profiler iterations do **not** imply identical forward or replay counts. Graph D operations are not represented by repeated CPU MoE/GDN scopes.

Plugin logs map P to physical stream 45 and D to 44. However, captured graph kernels also execute on **physical streams 0 and 2**: rank 0 has 4,395 and 6,700 compute rows there. Stream 44 alone contains only 254 compute rows. A lane-ID-only analysis would miss most graph execution. The audit consequently reports all physical compute streams and does not invent a per-kernel P/D mapping for the internal graph streams.

## Host observations

These are **inclusive**, potentially nested durations, and must not be added together. They include profiler overhead and host/device dependency waits.

| Rank 0 scope | Old plugin count / inclusive ms | Native count / inclusive ms |
| --- | ---: | ---: |
| `vllm::qwen_gdn_attention_core` | 60 / 300.921 | 60 / 341.806 |
| `ChunkGatedDeltaRuleFunction` (nested in GDN) | 60 / 202.487 | 60 / 203.149 |
| `vllm::moe_forward_shared` | 80 / 222.343 | 80 / 224.000 |
| `npu_fx_compiler inference` (enclosing scopes) | 1 / 360.709 | 2 / 785.341 |
| `Event::synchronize` | 9 / 111.826 | 8 / 161.474 |
| `aclrtSynchronizeStreamWithTimeout` | 49 / 0.257 | 48 / 0.222 |
| `aclmdlRIExecuteAsync` host API | 7 / 42.604 | 6 / 41.489 |
| `Event::record` | 285 / 25.770 | 256 / 22.745 |
| `Event::wait` | 280 / 19.439 | 248 / 17.287 |
| `aten::copy_` | 2,133 / 55.261 | 2,164 / 52.739 |
| Vocabulary `vllm::all_gather` | 9 / 3.284 | 8 / 2.930 |

Sampler event wait varies strongly by rank: old plugin 56.92–111.83 ms, native 52.27–161.47 ms. That is residual waiting on required output completion, not removable synchronization overhead. Stream synchronization is less than 0.3 ms total on every rank in both captures, so deleting the native graph replay barrier is not a supported high-impact optimization.

As a coarse count check only, event records per physical forward equivalent are 285/9 = 31.67 versus 256/8 = 32; waits are 31.11 versus 31. These are not equal-work latency normalizations, but do not show the previous eager shared-communication event-count explosion. Repeated input bookkeeping and copies exist in both paths; this capture does not attribute their total duration to plugin duplication.

## Device observations

The classifier includes AI_CORE, AI_VECTOR, AI_VECTOR_CORE and MIX compute tasks, excludes recognizable communication kernels, and **excludes the Communication track's enclosing HCCL/transport ranges**. This export has no separately classified hardware HCCL rows; their absence is not evidence that communication is absent or free.

| Eight-rank range | Old plugin | Native |
| --- | ---: | ---: |
| Compute active union | 294.59–299.89 ms | 293.96–296.35 ms |
| First-to-last classified compute kernel span | 1,032.66–1,033.81 ms | 1,099.22–1,100.09 ms |
| Compute union / that span | 28.50–29.02% | 26.74–26.94% |
| Compute overlap across physical streams | 0 on all ranks | 0 on all ranks |

These ratios are **profile-window coverage**, not AI Core occupancy, utilization of all silicon resources, or an unprofiled throughput comparison. Zero observed cross-stream compute overlap is consistent with the current fenced execution. It does not implement hardware resource partitioning.

Rank 0 plugin device compute includes 39.17 ms of ordinary matmul, 28.33 ms of chunk-GDN state updates, 26.07 ms of grouped matmul, and 24.98 ms of add/RMS norm kernel durations. These kernel categories reinforce that GDN/MoE computation is real work; their enclosing host scopes also include dispatch and waiting. The remaining profile span cannot be assigned entirely to Python, communication, or idle hardware from this summary.

## Changes justified by this evidence

1. **Measure and cap actual raw P slices first.** `DecoupledMuxPDNPUWorker._advance_lanes` submits D then P in fenced mode; `MuxPDNPUWorker.sample_tokens` subsequently restores D, and the backend orders it after P's tail. The D token is therefore delayed by that P slice. The new opt-in timing receipt records host submission separately from a P-stream event span, with actual layer delta, first-slice status, cohort tokens, and raw/compiled mode. It only reads elapsed device time after an existing successful sampler completion barrier, and adds no synchronize. Empty unfinished-P transactions remain host-only. This is timing evidence for later calibration, not kernel-time or SLO proof.
2. **Do not train raw slice cost from whole compiled P.** The captured plugin's P128 uses FX while P12000 uses layer hooks. The new receipt keeps these execution modes distinct. First slices also contain input preparation that subsequent slices do not.
3. **Keep D's required sampling completion and graph update ordering.** Native `_to_list` copies sampled IDs to pinned host memory then synchronizes its transfer event (`gpu_model_runner.py:7604–7617`). Native graph replay's stream barrier (`vllm_ascend/compilation/acl_graph.py:252–266`) protects mutable update state and is tiny in this trace.
4. **Treat sampling D before P as a separate experimental protocol.** It could avoid waiting for P before D sampling, but the engine cannot return that cached sample until P host submission finishes; the next fenced D also waits for the remaining P tail. It requires exactly-once sample caching, error/cancellation cleanup, unchanged P completion ACKs, and explicit latency tests. It is not free removal of P cost and is not included in the timing change.
5. **Do not prioritize deleting layer safety checks.** `LayerInterleaver.checkpoint` only switches greenlets/fences at the quota boundary; ordinary layers update a counter and validate the owning thread/coroutine. There is no evidence that these checks explain the hundreds of milliseconds in the P-containing rounds. The D-only direct path, same-stream event-wait avoidance, and buffer/metadata reuse are already present.

Graph plus true P/D overlap remains a separate architectural optimization. This bounded fenced path does not claim to be spatial MuxWise or to partition the hardware.

The event timing receipt is local to the responding worker rank, not a cross-rank maximum; no timing reduction collective is added. Its device interval follows the incoming P dependency and therefore excludes time waiting for prior D before that start event. It still includes host feeding gaps between the start and end events. Whole-step host round-trip measurements and the per-slice event span answer different questions.

## Capture limits

Both collections used eight automatic-stop worker iterations, reported `RECORD` stop warnings, and were parsed offline because the daemon workers cannot run the parser. Only complete exported `X` intervals are counted. The client stall around automatic profiler stop (including the old client's roughly 1.18 s tail gap) is not treated as inference latency. These captures are diagnostic and separate from root's unprofiled contention/performance runs. Re-profile only when a new implementation question requires it; enable custom profiling scopes at startup if native input preparation and sampler scopes are specifically needed.
