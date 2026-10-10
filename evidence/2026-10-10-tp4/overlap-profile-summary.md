# TP4 overlap device-profile audit

All four ranks show overlapping prefill/decode **device compute task intervals**
in this bounded capture. The intersection is 2.86–4.49% of each rank's compute
interval union. This establishes observed cross-stream task overlap; it does
not establish simultaneous per-core occupancy, hardware resource partitioning,
production utilization, or a serving speedup.

The workload used Qwen3.5-35B-A3B BF16, Ascend 910B2/A2, TP4 + EP4, eager V1,
decoupled overlap, and `layers_per_slice=8`. A 128-token prompt requested 64
decode tokens; after its first content event, a second request submitted a
12,000-token prompt and requested 8 tokens. The complete client workload is
recorded in [the client result](overlap-profile-client.json); the device profile
captures only its first eight worker steps.

| Rank / trace device | Compute union (ms) | P/D intersection (ms) | Intersection / union | First-to-last compute span (ms) | Compute union / span |
|---|---:|---:|---:|---:|---:|
| 0 | 369.22418 | 15.03470 | 4.072% | 2915.69926 | 12.663% |
| 1 | 370.90522 | 16.65962 | 4.492% | 2917.37808 | 12.714% |
| 2 | 372.63434 | 10.64874 | 2.858% | 2916.58484 | 12.776% |
| 3 | 370.57212 | 15.77892 | 4.258% | 2916.23504 | 12.707% |

Each row is measured independently on one device. No cross-device timestamps
are combined. The last column is activity within this profiled window, which
includes profiler overhead; it is not a hardware utilization counter or a
production utilization estimate. The denominator of the overlap percentage is
the compute union, not the full capture duration or the serving workload time.

## Attribution and independent verification

The worker startup log records physical stream 45 for prefill, 44 for decode,
and 47 for the caller on **each** rank. These are physical IDs obtained by the
worker, not opaque NPU stream handles or PyTorch stream IDs. Trace attribution
uses `args["Physic Stream Id"]` and device process labels `NPU 0` through `NPU 3`.
The four [lane maps](overlap-profile-lane-map-rank0.json) are separate inputs to
the analyzer; the complete twelve-line startup mapping is preserved in the
[independent report](overlap-profile-independent.json).

The included task types are `AI_CORE`, `AI_VECTOR_CORE`, `AI_VECTOR`, `MIX_AIC`,
and `MIX_AIV`, restricted to recognized device-task processes. Python, CANN
host tracks, profiler-generated Overlap Analysis ranges, and Communication
tracks do not count as computation. Recognizable HCCL operation names are also
excluded. This removes 15,489 Communication-track ranges on rank 0 and 10,953
on each other rank; those wrappers and nested transport intervals would
otherwise overstate overlap.

Each rank contains 16,086 selected compute intervals: 5,400 on prefill and
10,686 on decode. The type counts are 11,907 `AI_VECTOR_CORE`, 2,439 `AI_CORE`,
1,380 `MIX_AIC`, and 360 `MIX_AIV`. Every selected trace interval matches an
exported `kernel_details.csv` row by kernel name, task type, start, end, and
stream ID, including multiplicity. There are zero unmatched rows in either
direction on all four ranks.

The separate [audit script](overlap-profile-audit.py) reads the CSV intervals,
converts timestamps with `Decimal` to integer nanoseconds, forms each lane's
interval union, and intersects the two unions with a two-pointer algorithm.
The results exactly match both a direct trace calculation and the repository's
`scripts/analyze_npu_trace.py` output for every rank. The script imports only
standard-library modules and launches no inference or NPU work.

As an additional check, limiting both lanes to `AI_CORE`/`MIX_AIC` still gives
positive intersections: 4.21464, 4.98844, 1.53786, and 5.76636 ms for ranks
0–3. These subtype intersections are not additive with the vector intersections.
The [raw excerpt](overlap-profile-excerpt.json) retains metadata and six selected
overlapping pairs per rank, including MatMul/GroupedMatmul/FusedInferAttention
and vector operations. These largest-pair examples are selected for inspection,
not as a representative utilization sample.

## Capture limits and artifacts

Profiling started at approximately 03:05:42 UTC on 2026-10-10 and automatically
stopped at approximately 03:05:46 after `max_iterations=8`. All four workers
warned that stopping in `RECORD` could produce incomplete parsed data. Every
selected compute interval is inside its device's `PROFILING_ENABLE` and
`PROFILING_DISABLE` markers, with no interval crossing either boundary. This
boundary check does **not** prove that all runtime work was captured. The
64-token client workload continued after capture stopped.

The worker daemon could not parse its profile in-process. A separate non-daemon
process ran `torch_npu.profiler.profiler.analyse`; its log records successful
completion for all four ranks at 03:06:47–48 UTC. Hardware measurements under
profiling must remain separate from the serving benchmark results.

Compact archived evidence:

- [Independent report](overlap-profile-independent.json): per-rank exact values,
  capture boundaries, raw trace/CSV SHA256 hashes, source paths, stream and stop
  logs, exclusions, and reproduction commands.
- Analyzer outputs: [rank 0](overlap-profile-analyzer-rank0.json),
  [rank 1](overlap-profile-analyzer-rank1.json),
  [rank 2](overlap-profile-analyzer-rank2.json),
  [rank 3](overlap-profile-analyzer-rank3.json).
- Lane maps: [rank 0](overlap-profile-lane-map-rank0.json),
  [rank 1](overlap-profile-lane-map-rank1.json),
  [rank 2](overlap-profile-lane-map-rank2.json),
  [rank 3](overlap-profile-lane-map-rank3.json).
- [Raw event excerpt](overlap-profile-excerpt.json) and
  [reproducible independent audit](overlap-profile-audit.py).

The four complete traces total 438,950,245 bytes and remain under ignored
`local-results/tp4/overlap-profile/`; they are not included in Git. With those
local exports available, reproduce the compact analysis from the repository
root using:

```bash
/root/workspace/.venv/bin/python evidence/2026-10-10-tp4/overlap-profile-audit.py
```
