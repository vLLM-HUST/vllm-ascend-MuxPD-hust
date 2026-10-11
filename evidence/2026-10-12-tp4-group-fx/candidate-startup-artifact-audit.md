# TP4 group-FX startup artifact audit

Stage: `tp4-groupfx-candidate-b-diagnostic-v1`. Runtime snapshot:
`f09a100ade651936b3efeaf045ec5ce63d0b38fe`. This is a read-only audit of
generated text and startup logs; no Torch import, unpickling, inference request,
device call, compilation or test suite was performed by the auditor.

Reproduce with `python3 local-results/tp4-prefill-group-fx/audit_groupfx_artifacts.py`.
`candidate-startup-artifact-audit.json` records all 44 file SHA256 values, byte
sizes, absolute layer prefixes, operation counts, selected source hashes, startup
snapshots with source byte offsets and line hashes, and the bounded startup-log
prefix SHA256. Serving lines after the final startup snapshot are excluded.

## Observed startup structure

All eight checks passed. On each of four ranks, ten P graph files have exactly the
expected absolute prefixes: group 0 uses layers 0–3, group 1 uses 4–7, through group
9 using 36–39. Every group has three GDN custom-op calls, one full-attention call,
four MoE calls, four explicit TP all-reduces and four conditional MoE TP reductions.
The separate D graph on each rank covers all 40 layers, with ten times these counts.

The P files are under `*-cache/muxpd-prefill-groups-v1/prefill-group-XX/rank_R_0/`;
the original D backbone is under `*-cache/compile/rank_R_0/`. The rank-0 log records
ten distinct P AOT save paths at 16:24:14–16:24:57 UTC, then a separate D AOT save
path at 16:25:43. All four ranks report ten distinct entry names
`PrefillGroup00.forward` through `PrefillGroup09.forward`, each successfully returned
once during startup. All four D snapshots report four eager warmups, four FULL
capture forwards, four captured entries, and no failure.

Native `gpu_model_runner.py:6729–6750` runs each NONE-mode warmup before its FULL
capture. This agrees with independent D compilation after group profiling; the P
startup does not substitute for D backbone preparation.

The `code_hash` values in `cache_key_factors.json` are **not all unique**: groups
1–9 share a hash because their code structure is alike. Cache directory isolation,
distinct entry identities/AOT save paths and differing absolute layer prefixes are
the observed isolation evidence. A claim of ten unique code hashes would be false.

## Communication and execution boundary

The graph text contains `torch.ops.vllm.all_reduce(..., group_name='tp:0')`.
This is not a literal HCCL handle: native `parallel_state.py:130–167` resolves the
registered coordinator at call time, whose device communicator is wrapped by
`communicators.py:45–65` to route P to its sibling group. MoE remains an opaque
`moe_forward_shared(..., 'from_forward_context', ...)` call; native
`moe_runner.py:73–88,152–166` uses the current lane's forward context and advances
its MoE layer index. `overlap.py:251–289` saves and restores that context.

The selected backend has `enable_npugraph_ex=false` and `enable_static_kernel=false`.
Native Ascend `compiler_interface.py:47–68,282–301` returns the fusion-pass FX/AOT
callable. **P group execution is not P ACL graph replay.** Native
`backends.py:1251–1262` emits readable split-GraphModule text; these files are not
an inspection of all final kernels or opaque communication implementations.

This evidence does not independently prove all opaque HCCL operations selected
the P communicator, serving numerical equivalence, every dynamic kernel being
warm, physical device overlap, cancellation coverage, or a performance gain.
Startup GDN with absent attention metadata can bypass real kernels. Those claims
require the separately registered serving/lifecycle and, where necessary, device
trace evidence.
