# Isolated versus shared HCCL groups: bounded TP4 profile audit

Both profiles are diagnostic 128-token decode + 12,000-token prefill probes with eight-step automatic profiler stopping; they are not the timed throughput benchmark. The isolated profile used fixed quota 8, separate P/D groups, and a contiguous P slice before complete D submission. It predates the D-only fast path and compiled-prefill experiments. The old shared profile also contains the old event gates and per-layer host alternation, so this comparison does not isolate communicator identity as the only changed variable.

Each rank is analysed independently. Forward equivalents are inferred from matching MoE/40, GDN/30, and logits all-gather counts, not direct whole-model timing ranges. Raw work counts are retained below. The shared values are read from the existing hashed summary without reparsing its large traces; isolated trace intervals are checked independently against kernel_details.csv, including stream assignment and exact nanosecond unions.

| Mode | Rank | Forwards | MoE / GDN | Compute kernels | Compute union ms | Kernel window ms | Active/window | P/D overlap ms | Overlap/compute union |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_overlap8 | 0 | 9 | 360 / 270 | 16086 | 369.224 | 2915.699 | 12.66% | 15.035 | 4.07% |
| shared_overlap8 | 1 | 9 | 360 / 270 | 16086 | 370.905 | 2917.378 | 12.71% | 16.660 | 4.49% |
| shared_overlap8 | 2 | 9 | 360 / 270 | 16086 | 372.634 | 2916.585 | 12.78% | 10.649 | 2.86% |
| shared_overlap8 | 3 | 9 | 360 / 270 | 16086 | 370.572 | 2916.235 | 12.71% | 15.779 | 4.26% |
| isolated_overlap8 | 0 | 9 | 360 / 270 | 16086 | 368.382 | 2657.284 | 13.86% | 9.703 | 2.63% |
| isolated_overlap8 | 1 | 9 | 360 / 270 | 16086 | 370.977 | 2657.672 | 13.96% | 11.259 | 3.03% |
| isolated_overlap8 | 2 | 9 | 360 / 270 | 16086 | 370.971 | 2657.834 | 13.96% | 12.246 | 3.30% |
| isolated_overlap8 | 3 | 9 | 360 / 270 | 16086 | 370.478 | 2657.824 | 13.94% | 11.400 | 3.08% |

Normalized counts per forward equivalent:

| Mode | Rank | All-reduce | Python record / wait | CANN record / wait |
|---|---:|---:|---:|---:|
| shared_overlap8 | 0 | 81.000 | 215.667 / 205.444 | 462.333 / 369.444 |
| shared_overlap8 | 1 | 81.000 | 215.667 / 205.444 | 462.333 / 369.444 |
| shared_overlap8 | 2 | 81.000 | 215.667 / 205.444 | 462.333 / 369.444 |
| shared_overlap8 | 3 | 81.000 | 215.667 / 205.444 | 462.333 / 369.444 |
| isolated_overlap8 | 0 | 81.000 | 125.333 / 123.444 | 372.000 / 287.444 |
| isolated_overlap8 | 1 | 81.000 | 125.333 / 123.444 | 372.000 / 287.444 |
| isolated_overlap8 | 2 | 81.000 | 125.333 / 123.444 | 372.000 / 287.444 |
| isolated_overlap8 | 3 | 81.000 | 125.333 / 123.444 | 372.000 / 287.444 |

Communication-related intervals per forward equivalent (microseconds):

| Mode | Rank | All-reduce CPU inclusive / exclusive | HCOM all-reduce host API inclusive | Communication-track union | Hardware HCCL-classified tasks |
|---|---:|---:|---:|---:|---:|
| shared_overlap8 | 0 | 34356.4 / 15248.9 | 26622.8 | 179758.4 | 0 |
| shared_overlap8 | 1 | 35441.1 / 15671.4 | 25665.2 | 104627.7 | 0 |
| shared_overlap8 | 2 | 34550.6 / 15205.8 | 24278.8 | 172633.1 | 0 |
| shared_overlap8 | 3 | 34584.2 / 15276.2 | 23456.3 | 196451.3 | 0 |
| isolated_overlap8 | 0 | 13150.7 / 5815.5 | 27042.2 | 169981.4 | 0 |
| isolated_overlap8 | 1 | 13202.9 / 5765.9 | 23175.3 | 91054.5 | 0 |
| isolated_overlap8 | 2 | 13053.0 / 5669.4 | 24589.3 | 79489.4 | 0 |
| isolated_overlap8 | 3 | 13227.3 / 5723.0 | 24700.2 | 91938.1 | 0 |

The exported all-reduce count remains 81 per forward. Isolated execution reduces Python event records by 41.9% and waits by 39.9% per forward relative to the old shared implementation. This is consistent with removing per-collective P/D gates and per-layer host switching; it does not measure the end-to-end benefit of either change in isolation.

The same 16,086 selected kernels per rank cover essentially unchanged model-compute time (about 368–373 ms). The isolated kernel windows are about 2.657 s versus 2.916 s for shared, while isolated overlap is 2.63–3.30% versus shared 2.86–4.49%. Thus this capture shows fewer synchronization calls and a shorter submission window, not an across-rank increase in compute overlap. Rank 0 all-reduce CPU inclusive time falls from 34.36 to 13.15 ms per forward, whereas its HCOM host API inclusive time remains about 26.62 versus 27.04 ms. These differently nested/threaded intervals suggest less wrapper/gate work; they do not establish lower physical link latency or a quantitative causal attribution of throughput.

The compute numerator includes only explicit AI_CORE/AI_VECTOR/MIX hardware tasks, excluding recognizable HCCL/communication names. P/D overlap is the intersection of each lane's union on verified physical streams 45/44. Communication-track spans, HCCL host wrappers, event waits and DMA are excluded from model compute. Compute active/window is interval coverage under profiler overhead, not AICore occupancy or hardware resource partitioning.

CPU wrapper inclusive times contain nested work and cannot be summed across operations; exclusive times subtract contained cpu_op intervals on the same thread and still include uninstrumented time. HCOM API duration is host time. Communication-track union includes waiting and transport spans, not useful communication compute. Zero HCCL-classified hardware tasks does not mean no communication: SDMA, Notify Wait and Write Value tasks are retained separately in JSON. The different profile windows and workload timing prohibit interpreting communication-track union as a matched latency improvement.

All four isolated ranks emitted a RECORD-state warning on automatic stop. Every selected compute kernel lies between one exported PROFILING_ENABLE and PROFILING_DISABLE marker, and layer/logits work counts agree, but these checks do not prove the entire request lifetime or every profiler record was captured. Offline parsing completed after the worker-daemon parser refusal. Absolute timings are perturbed by instrumentation and are not substituted for the separate unprofiled benchmarks.

Reproduce: `python local-results/perf-tp4/isolated-vs-shared-profile-audit.py --ranks 0 1 2 3`. JSON includes trace/CSV/metadata/helper/summary hashes, the isolated frozen source manifest hash, server-log excerpts with line numbers, every rank's raw counts and normalized metrics. No raw trace is copied into the report.
