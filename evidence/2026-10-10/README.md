# Ascend integration evidence — 2026-10-10

Hardware: Ascend 910B2. Model: Qwen3.5-35B-A3B BF16, TP2 + EP,
16K context, max 8 sequences, eager, no speculative decoding.

Host source pins:

- vllm-hust: `d0f22d2bda562156e4dbf433ce645e1769b4f804`
- vllm-ascend-hust: `03766ac696fde5ab1980d80ca0b8543d3580c989`
- MuxWise reference: `eeac5148f456428a05823c330e1d40afb0ef2438` (`slo_config`)

These are synthetic integration experiments, not production performance claims.
No CUDA paper speedup or NVIDIA regression parameters are transferred to Ascend.

## Receipts

| File | Experiment |
|---|---|
| `original-smoke.json` | Original runtime configuration; all 9 API cases passed |
| `serial-smoke.json` | MuxPD serial scheduler; all 9 API cases passed |
| `original.json` | Original 2048-token budget, Mamba align, prefix cache enabled |
| `serial-default.json` | MuxPD serial, 16384-token budget, Mamba none, prefix cache disabled |
| `serial-trace.jsonl.gz` | Scheduler admissions/completions for serial smoke and benchmark |
| `control.json` | Native mixed batching with matched 16K budget / Mamba none / prefix cache off |
| `layered-fenced.json` / `layered-smoke.json` | Initial layered worker: benchmark and 9/9 API cases |
| `layered-trace.jsonl.gz` | Initial layered scheduler receipts |
| `decoupled-fenced-8.json` | Cross-iteration worker, 8 prefill layers per decode iteration |
| `decoupled-fenced-smoke.json` | Cross-iteration worker: 9/9 API cases |
| `decoupled-fenced-progress.json` | 12,000-token P while D independently produces output |
| `decoupled-fenced-cancel.json` | Cancel long P before its first output; D completes normally |
| `decoupled-fenced-trace.jsonl.gz` | Full prompt admission, four D steps before P completion, cancellation ACK |
| `decoupled-overlap-*` | Independent compute streams: benchmark, 9/9 smoke, long-P progress, cancellation and host trace |
| `decoupled-adaptive-*-smoke.json` | Fenced/overlap with deadline-aware quotas: 9/9 API cases each |
| `decoupled-adaptive-fenced.json` / `decoupled-adaptive-overlap.json` | Sequential benchmarks with learned deadline-aware layer quotas |
| `decoupled-adaptive-*-trace.jsonl.gz` | Strict 40-layer progress receipts and quota calibration/decisions |
| `native-vs-*.json` | Isolated and paired natural-language correctness comparison, two repetitions |

The API cases include greedy arithmetic, three concurrent Chinese SSE requests,
an exact 2305-token input, cancellation, health after cancellation, and subsequent
generation. The trace validates full remaining prefill and homogeneous P/D;
it does not prove kernel concurrency or prove KV reclamation after HTTP cancel.
Native allocator unit/integration tests check lifecycle behavior separately.

Each benchmark warms up input lengths 128, 512, 2048, 4096, then runs 8 staggered
requests with 64 generated tokens and two repetitions. Input shapes cycle through
128, 2048, 512, 4096 at 120 ms arrival intervals. `ignore_eos=true` fixes output
length. Repetitions reuse the same synthetic prompts, so the original cache-enabled
run can benefit from cache reuse; all settings are preserved in the receipts.

The original configuration measured 7.02 and 7.10 output tokens/s; the initial
serial configuration measured 33.10 and 33.32. **This is a configuration comparison,
not an isolated MuxPD speedup:** token budget, Mamba mode and caching differ, and
different pairs of the same NPU model were used. The initial serial experiment
also overlapped part of a separate server's startup on the shared host. A matched
control is required before attributing a gain to scheduling.

The matched control and plugin measurements below use the same 16K token budget,
Mamba `none`, disabled prefix caching, eager TP2+EP and synchronous scheduling.
The control and overlap modes ran on physical NPUs 4/5; serial and fenced modes
ran on 6/7. Throughput is
total generated tokens divided by total time across both runs; percentile values
are nearest-rank over the 16 request samples. This small experiment does not
establish statistical significance.

| Configuration | Output tokens/s | TTFT p95 (s) | Mean TPOT p95 (s) |
|---|---:|---:|---:|
| Native matched control | 34.16 | 0.680 | 0.223 |
| Serial MuxPD | 33.21 | 0.658 | 0.240 |
| Initial layered fenced | 32.05 | 1.207 | 0.244 |
| Decoupled fenced, 8 layers | 29.44 | 3.085 | 0.236 |
| Decoupled overlap, 8 layers | 28.52 | 2.959 | 0.244 |
| Adaptive fenced | 31.38 | 1.388 | 0.252 |
| Adaptive overlap | 28.99 | 1.285 | 0.278 |

The layered receipt predates the optimization that skips per-layer host switching
for a single active phase. The decoupled receipt includes that optimization, final
layer completion without an extra iteration, and full P completion when no D is
present. Its fixed quota increases TTFT on this short-prompt workload because
new P cohorts wait for the current cohort to finish. These data show an overhead,
not a MuxPD throughput gain. Decoupled benchmark warmup uses 8 output tokens per
shape; earlier receipts use 64. Timed requests remain 64 output tokens in all runs.

Adaptive quotas improve this workload's TTFT relative to fixed 8-layer quotas,
while the matched native control remains faster. Adaptive fenced throughput rises
about 6.6% versus fixed fenced, with a higher TPOT tail. The two adaptive modes
choose 20 or 40 layers after calibration; the benchmark cohorts complete within
one host iteration when their deadlines require it. Cross-iteration progress is
demonstrated separately by the fixed-quota long-P experiment. Quotas do not remove
the cost of complete prefill or guarantee either SLO. Both adaptive benchmarks
ran sequentially, after model startup and correctness clients had finished.

The fixed modes and adaptive overlap are individually stable across their two
synthetic benchmark repetitions. Adaptive fenced has different output hashes for
3/8 request indices between repetitions; their P cohort grouping also differs.
For example, request 3 joins requests 1/2 in one repetition and requests 4/5/6/7 in
the other. Output hashes are also not all equal between configurations. Batch
shape can affect greedy generation, but these observations do not establish the
cause or rule out a state bug. Dedicated isolated/paired correctness cases are
used separately; no claim of bitwise equivalence for arbitrary inputs is made.

For natural-language checks, each of decoupled fenced, decoupled overlap, adaptive
fenced and adaptive overlap passed 16/16 cases, with all 16 outputs exactly matching
the matched native baseline. Each endpoint also reproduced the same text between
its two repetitions and between isolated/paired execution. Cases cover arithmetic,
ordered counting, distinct identifier extraction and a 2735-token Chinese prompt
with a known terminal code. The long P is submitted only after D emits its first
content event and while D remains active. The first comparison contains the native
measurement; later reports validate and reuse it with a recorded SHA256. These
limited tests do not establish equivalence for arbitrary prompts or sampling modes.

At the benchmark's TTFT <= 2 s / mean TPOT <= 0.1 s SLO, both configurations had
zero request goodput; the matched control and both fenced worker modes also have
zero goodput at that SLO; overlap and adaptive modes also have zero goodput.
None meets the 100 ms decode target. Device overlap
remains unproven by these fenced traces.

## Reproduce

Use `scripts/smoke_serving.py` and `scripts/benchmark_serving.py` with the commands
in the repository README. For trace validation:

```bash
gzip -dc evidence/2026-10-10/serial-trace.jsonl.gz > /tmp/muxpd-serial-trace.jsonl
/root/workspace/.venv/bin/python scripts/check_trace.py /tmp/muxpd-serial-trace.jsonl
```

Raw verbose server logs are retained locally under `local-results/`; the receipts
contain synthetic inputs/outputs and timing observations, not user traffic.

## Device profile: independent P/D compute intervals

The decoupled overlap worker, with 8 P layers per iteration, was profiled on
Qwen3.5-35B-A3B BF16 with TP2+EP. A 128-token prompt established a 64-token decode
request; a separate 12,000-token prompt then entered the P lane. Worker logs map
P to CANN physical stream 45 and D to stream 44 on both ranks. These are runtime
physical IDs obtained with `aclrtStreamGetId`, not opaque handles or PyTorch IDs.

| Rank | P/D compute interval overlap | Compute active union | Overlap / active union | First-to-last compute span | Active union / span |
|---|---:|---:|---:|---:|---:|
| 0 | 59.260 ms | 549.976 ms | 10.78% | 2883.717 ms | 19.07% |
| 1 | 50.477 ms | 550.915 ms | 9.16% | 2890.926 ms | 19.06% |

Thus actual device P/D compute task intervals overlap in this capture. The
calculation includes AI_CORE, AI_VECTOR_CORE and mixed AIC/AIV tasks in the
`Ascend Hardware` track. It excludes CPU/CANN host ranges, event waits, copies,
the `Communication` track and identifiable HCCL tasks. In particular, 7,623
communication operation/transport ranges per rank are excluded. Each rank has
16,115 classified compute tasks; their timestamps, durations, streams, task
types and names match `kernel_details.csv` exactly. An independent per-lane
interval-union/intersection calculation matches the analyzer results.

The active/span values describe this profiled window and include profiler
overhead; they are not production utilization. This experiment does not measure
simultaneous per-core occupancy or hardware resource partitioning, and the
matched serving benchmark has not demonstrated a speedup.

The profiler stopped at its configured 8 worker steps and warned that stopping
in RECORD state may leave incomplete parsed data. Offline parsing completed for
both ranks. All exported compute intervals are complete `X` events within the
device `PROFILING_ENABLE` / `PROFILING_DISABLE` markers. This checks the exported
sample, but does not establish completeness of unrecorded work or the full
64-token workload.

- `overlap-profile-independent.json`: independent verification, full trace SHA256,
  physical stream log lines, capture bounds and reproduction commands.
- `overlap-profile-analyzer-rank0.json` and `overlap-profile-analyzer-rank1.json`:
  `scripts/analyze_npu_trace.py` results with explicit per-rank lane maps.
- `overlap-profile-excerpt.json`: approximately 18 KB of original device metadata
  and selected overlapping compute intervals. Selection illustrates overlap;
  it is not a representative utilization sample.

The complete traces are approximately 104 MB per rank and remain local under
`local-results/overlap-profile/`. No full device trace is included in the repository.
