# Native versus shared-communicator overlap8: bounded profile audit

These traces describe one diagnostic 128-token decode plus 12,000-token prefill probe per mode, not the timed eight-request benchmark. Collection was limited to eight engine iterations and does not represent the full request lifetime. Native exported eight model-forward equivalents and shared overlap exported nine; raw totals therefore include different executed work. Each rank and capture has its own timeline. No cross-rank or cross-capture intervals are merged.

No whole-network or decoder-layer Python forward ranges were exported. The counts below infer full-forward equivalents from the independently agreeing MoE (40 per forward), GDN (30 per forward), and logits all-gather counts. A MoE marker is one component per decoder layer, not a whole-layer timing range. Whole-forward elapsed time cannot be recovered from these marker durations.

| Mode | Rank | Forward equivalents | MoE / GDN calls | All-reduce CPU calls | Event record / wait | Device compute union ms | Kernel window ms | Compute active/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0 | 8 | 320 / 240 | 648 | 976 / 968 | 371.456 | 2681.356 | 13.85% |
| native | 1 | 8 | 320 / 240 | 648 | 976 / 968 | 372.530 | 2681.479 | 13.89% |
| native | 2 | 8 | 320 / 240 | 648 | 976 / 968 | 371.430 | 2680.627 | 13.86% |
| native | 3 | 8 | 320 / 240 | 648 | 976 / 968 | 373.606 | 2680.853 | 13.94% |
| shared_overlap8 | 0 | 9 | 360 / 270 | 729 | 1941 / 1849 | 369.224 | 2915.699 | 12.66% |
| shared_overlap8 | 1 | 9 | 360 / 270 | 729 | 1941 / 1849 | 370.905 | 2917.378 | 12.71% |
| shared_overlap8 | 2 | 9 | 360 / 270 | 729 | 1941 / 1849 | 372.634 | 2916.585 | 12.78% |
| shared_overlap8 | 3 | 9 | 360 / 270 | 729 | 1941 / 1849 | 370.572 | 2916.235 | 12.71% |

Per-forward-equivalent event counts separate the extra forward from synchronization overhead:

| Mode | Rank | All-reduce | Python record | Python wait | CANN record | CANN stream wait |
|---|---:|---:|---:|---:|---:|---:|
| native | 0 | 81.000 | 122.000 | 121.000 | 368.750 | 285.000 |
| native | 1 | 81.000 | 122.000 | 121.000 | 368.750 | 285.000 |
| native | 2 | 81.000 | 122.000 | 121.000 | 368.750 | 285.000 |
| native | 3 | 81.000 | 122.000 | 121.000 | 368.750 | 285.000 |
| shared_overlap8 | 0 | 81.000 | 215.667 | 205.444 | 462.333 | 369.444 |
| shared_overlap8 | 1 | 81.000 | 215.667 | 205.444 | 462.333 | 369.444 |
| shared_overlap8 | 2 | 81.000 | 215.667 | 205.444 | 462.333 | 369.444 |
| shared_overlap8 | 3 | 81.000 | 215.667 | 205.444 | 462.333 | 369.444 |

CPU operation timing for rank 0 (milliseconds). Inclusive values include nested CPU operations and must not be added across rows. Exclusive values subtract the union of strictly contained cpu_op intervals on the same thread; they are not device kernel time or a measurement of useful host execution. Suspended greenlet time and uninstrumented work remain included. Coincident/crossing ranges are reported separately in JSON.

| Mode | Operation | Calls | Inclusive sum ms | Exclusive of contained CPU ops sum ms |
|---|---|---:|---:|---:|
| native | `vllm::moe_forward_shared` | 320 | 944.476 | 425.696 |
| native | `vllm::qwen_gdn_attention_core` | 240 | 653.106 | 245.678 |
| native | `vllm::all_reduce` | 648 | 108.937 | 40.487 |
| native | `vllm::all_gather` | 8 | 2.724 | 1.203 |
| native | `Event::record` | 976 | 86.888 | 72.440 |
| native | `Event::wait` | 968 | 63.687 | 50.016 |
| native | `Event::synchronize` | 8 | 148.097 | 148.097 |
| shared_overlap8 | `vllm::moe_forward_shared` | 360 | 987.519 | 506.955 |
| shared_overlap8 | `vllm::qwen_gdn_attention_core` | 270 | 623.990 | 257.347 |
| shared_overlap8 | `vllm::all_reduce` | 729 | 309.207 | 137.241 |
| shared_overlap8 | `vllm::all_gather` | 9 | 5.277 | 2.569 |
| shared_overlap8 | `Event::record` | 1941 | 156.017 | 135.354 |
| shared_overlap8 | `Event::wait` | 1849 | 131.580 | 105.940 |
| shared_overlap8 | `Event::synchronize` | 9 | 29.678 | 29.678 |

`device_compute.active_union_us` unions only positive-duration hardware tasks with an explicit AI_CORE/AI_VECTOR/MIX task type, excluding communication names. The kernel window is first selected kernel start to last selected kernel end. It excludes waits, host launch gaps, and communication work from its numerator. This active fraction is a timeline-coverage statistic, not measured AICore occupancy. HCCL hardware tasks and the broader Communication track spans are reported separately; they are not added to compute or interpreted as useful HCCL computation.

No hardware task in these exports has an explicit recognizable HCCL kernel name/type. The resulting zero classified HCCL-task count does not mean zero communication: SDMA, Notify Wait, and Write Value tasks remain visible separately, as do Communication track spans. Their entire duration can contain waiting and cannot be treated as useful communication compute time.

All four ranks agree on counts. After normalizing for the extra exported forward, shared overlap still performs 76.8% more Python event records and 69.8% more Python event waits per forward-equivalent. The all-reduce call count remains exactly 81 per forward in both modes. This supports auditing synchronization overhead; it does not isolate its causal contribution to end-to-end latency.

Python event wrappers, AscendCL APIs, and hardware record/wait tasks represent different layers of the same synchronization and must not be summed. CPU all-reduce duration measures the host wrapper and cannot establish device communication latency. An overlap trace can include P/D kernels on physical streams 45/44 concurrently; this says nothing by itself about application throughput. The prior independent audit supplies that lane mapping.

The source SHA256, profiler metadata SHA256, per-operation counts/timings, per-stream unions, and reproduction script SHA256 are in `native-vs-shared-profile.json`. Reproduce with `python local-results/perf-tp4/native-vs-shared-profile-audit.py --ranks 0 1 2 3`. The JSON preserves every rank separately; no inclusive totals are summed across operation names or devices.
