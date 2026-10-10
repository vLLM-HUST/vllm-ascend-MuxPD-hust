# Compiled full-prefill and decode graph runtime evidence

The exported TP4 window confirms **seven decode graph replay API records and one actual compiled FX prefill runtime entry on every rank**. The FX entry is the initial 128-token prefill of the decode probe. The concurrent 12,000-token prefill uses five raw layer slices (8, 16, 24, 32, 40 layers) in this capture.

| TP rank | `aclmdlRIExecuteAsync` records | `npu_fx_compiler inference` records | FX host range (ms) | Enclosed MoE / GDN / full attention / GEMM calls | Matched scheduler step |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 7 | 1 | 371.106 | 40 / 30 / 10 / 270 | 1433 |
| 1 | 7 | 1 | 377.369 | 40 / 30 / 10 / 270 | 1433 |
| 2 | 7 | 1 | 365.403 | 40 / 30 / 10 / 270 | 1433 |
| 3 | 7 | 1 | 380.948 | 40 / 30 / 10 / 270 | 1433 |

`npu_fx_compiler inference` is a `ph=X`, `cat=cpu_op` event on each worker's Python/CPU main thread. It is not a Python-function category. The installed `torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py:503-517` opens this range inside `_CompiledFxGraph.__call__` around `self.run_kernel(*args, **kwargs)`. The JSON retains the source hash and excerpt, complete marker events, and the first/last enclosed model-operation events. The marker therefore demonstrates actual entry into the compiled FX runtime wrapper, beyond merely enabling compilation for a call. Each range contains all 40 decoder layers' model operations and completes before the first D replay.

For request association, the audit subtracts `profiler_info.end_info.collectionTimeEnd - MonotonicTimeEnd` from each absolute trace timestamp, then checks containment in the scheduler's monotonic schedule/complete interval. Every FX range is wholly inside step 1433, cohort 98: `cmpl-muxpd-decode-probe-0-b0dfdfdf`, 128 scheduled tokens, `full_forward_return`, and 40 completed layers. This association is inferred from clock-aligned containment and scheduling controls; the FX event itself does not contain a request ID. The subsequent 12,000-token request, cohort 99, appears in steps 1435–1439 with `layer_hooks` receipts and no additional FX runtime marker.

The lifetime `prefill_compiled_calls=72` controller counter records permitted compilation attempts, including startup profiling; it is **not** a measured count of actual FX entry hits. This bounded trace establishes one full compiled prefill invocation across the four cooperating TP ranks, not 72 hits or four independent requests. Graph replay CANN API events also demonstrate runtime submission, not result correctness by themselves; correctness checks and client output are separate evidence.

All durations in the table are host range durations measured under profiling, not NPU kernel duration, occupancy, hardware partitioning, or an unprofiled latency comparison. The capture auto-stopped after eight worker iterations and emitted RECORD-state warnings on all ranks. Conclusions apply only to complete exported events in this window; they are not a whole-service census.

The initial audit regex missed both the word `compiler` and underscore-delimited `_fx_`. That filtering error has been corrected. The updated audit checks compilation-related names across host tracks, including `cpu_op`, and asserts the four-rank counts and enclosed layer-operation counts above.

Reproduce from the repository root (standard Python only; no torch import or device access):

```bash
python local-results/perf-tp4/graph-compiled-profile-proof-audit.py
```

`graph-compiled-profile-proof.json` contains per-trace SHA-256/byte size, profiler-info hashes, source-manifest and client-result hashes, clock calibration, raw retained events, scheduler receipts, controller snapshots, stream mapping log excerpts, profiling boundaries, and the audit script SHA-256. Full traces remain under `local-results/perf-tp4/graph-compiled-profile/` and are not intended for Git.
