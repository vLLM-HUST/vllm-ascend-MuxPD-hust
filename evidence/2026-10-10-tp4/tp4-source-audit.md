# TP4 + EP4 source audit — 2026-10-10

The inspected execution paths contain no identified TP2-only restriction that
blocks the planned TP4 experiment. This is a static audit, not evidence that
TP4 inference is correct, deadlock-free, faster, or overlapping on the device.
Those claims require the separate hardware results in this evidence directory.
The audit used file reads and `git show`; it did not initialize an NPU, send
inference requests, or change either engine repository.

## Scope and immutable source pins

| Component | Inspected commit |
|---|---|
| MuxPD | [`784236edf7bc99ff71d8df7b9b93168164a0f72a`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/tree/784236edf7bc99ff71d8df7b9b93168164a0f72a) |
| vLLM | [`d0f22d2bda562156e4dbf433ce645e1769b4f804`](https://github.com/vLLM-HUST/vllm-hust/tree/d0f22d2bda562156e4dbf433ce645e1769b4f804) |
| vLLM Ascend | [`03766ac696fde5ab1980d80ca0b8543d3580c989`](https://github.com/vLLM-HUST/vllm-ascend-hust/tree/03766ac696fde5ab1980d80ca0b8543d3580c989) |

The scope is Qwen3.5-35B-A3B, Ascend 910B2/A2, BF16, eager V1 execution,
synchronous scheduling, TP4 + EP4, and DP/PP/DCP/PCP all equal to 1. Prefix
caching is disabled and Mamba cache mode is `none`. The plugin explicitly
checks these parallel/cache restrictions and its narrower model/runtime
requirements. This audit does not extend support to other configurations.
Sources: [engine validation](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/config.py#L103-L128),
[worker validation](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/worker.py#L175-L200).

## Communication and rank groups

With these dimensions, native TP and EP groups each contain ranks `[0,1,2,3]`.
They are separately constructed process groups: EP spans DP × PCP × TP.
MuxPD installs gates on the actual TP and EP device communicators, keys ordering
by the actual `device_group`, and wraps `all_reduce`, `all_gather`, and
`reduce_scatter`. No two-rank ordering is embedded in the gate. Correctness
still depends on all ranks submitting the same collective sequence; inspecting
the local gate cannot prove distributed progress under every failure.
Sources: [TP construction](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/distributed/parallel_state.py#L1779-L1811),
[EP construction](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/distributed/parallel_state.py#L1889-L1915),
[gate installation](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/worker.py#L277-L289),
[gate implementation](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/overlap.py#L56-L123).

A2 selects the `ALLGATHER` MoE strategy at both TP2 and TP4. MC2 requires
EP world size ≥16, at most 24 experts per device, and a token-capacity check.
EP4 has 256/4 = 64 experts per device and fails the first two conditions.
Consequently P/D token-count differences do not switch either lane to MC2 in
this experiment. [A2 strategy selection](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ascend_forward_context.py#L289-L301).

The strategy name does **not** imply an EP all-gather for every MoE layer here.
With FlashComm off, preparation uses the DP path. Its gathers are conditional
on DP or PCP >1, so neither occurs with the audited configuration. Tokens are
already replicated across TP ranks; each EP rank computes its local experts,
and the combined routed/shared output is reduced through TP. Native MoE
sequence parallelism also requires DP >1 and is therefore false here.
Sources: [prepare dispatch](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/prepare_finalize.py#L348-L366),
[conditional gathers](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/prepare_finalize.py#L423-L477),
[final reduction](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/fused_moe.py#L597-L615),
[native sequence-parallel condition](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/config/parallel.py#L641-L656).

Keep FlashComm, shared-expert multistream, EPLB, and fine-grained TP disabled.
The worker rejects the first three, TP ≥16, and overlap with MLP/o-projection
fine-grained TP. All fine-grained TP fields default to zero; this audit assumes
those defaults for embedding and LM-head as well, rather than claiming the
worker guard covers every possible alternative communicator.
Sources: [worker gates](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/worker.py#L235-L255),
[fine-grained defaults](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ascend_config.py#L424-L429).

## Model partition changes

The following values were read from the local model's `config.json`, SHA256
`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`:
40 layers, 16 attention Q heads, 2 KV heads, 256 routed experts with top-8
routing, and GDN key/value head counts 16/32 with head dimensions 128/128.
The table applies the native partition formulas to that metadata.

| Per-rank quantity | TP2 + EP2 | TP4 + EP4 | Practical consequence |
|---|---:|---:|---|
| Full-attention Q heads | 8 | 4 | Smaller local attention computation |
| Full-attention KV heads | 1 | 1 | TP4 replicates two global KV heads across four ranks; this cache component does not halve per rank |
| GDN key/value heads | 8 / 16 | 4 / 8 | Smaller local GDN projections and recurrent states |
| Routed experts | 128 | 64 | Smaller local expert weights and a four-rank reduction |
| Each GDN `b` / `a` projection width | 16 | 8 | Narrower BF16 linear outputs require hardware correctness/performance checks |

Native attention explicitly permits KV-head replication when TP exceeds the
KV-head count. Expert ownership uses actual EP size and rank. GDN state shapes
divide by actual TP size. All relevant dimensions above divide four.
Sources: [attention partition](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/models/qwen3_next.py#L237-L253),
[expert ownership](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/models/qwen3_next.py#L109-L136),
[GDN state shapes](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/layers/mamba/mamba_utils.py#L212-L234).

The GDN BA-projection workaround for Marlin's minimum width is CUDA/quantization
specific and does not apply to this BF16 NPU path. Ascend unquantized linear
ultimately calls `torch.nn.functional.linear`; no TP4 rejection appears there.
Sources: [BA partition/workaround](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L595-L642),
[Ascend BF16 linear](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/linear.py#L53-L106).

The AscendC recurrent-GDN tiler accepts the TP4 values `nk=4`, `nv=8`,
`dk=dv=128`: it checks dimension upper bounds and that `nv` is a multiple of
`nk`. Its dynamic AIV launch count is `min(batch × nv, available AIV cores)`.
For a single decode request that is eight task units at TP4 versus sixteen at
TP2. This may change utilization and latency; it is native operator tiling,
not hardware resource partitioning introduced by MuxPD.
[Shape checks and dynamic launch count](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp#L290-L332).

## Shared state and remaining validation

The second runner constructs its own runner state but attaches the owner's
model, request registry, Mamba state-index map, and native KV tensors within
the same rank. It does not create cross-rank storage aliases or infer cache
shapes from a constant TP size. TP4's replicated KV heads remain separate
per-rank allocations. GDN prefill gathers/scatters state by prefill indices;
decode passes its state-index tensor to the recurrent operator.
Sources: [shared runner](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/worker.py#L69-L100),
[GDN state access](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L407-L444).

Shared request metadata still requires disjoint P/D ownership and pinned
prefill blocks. The worker checks cohort identity and rejects retirement or
preemption of active prefill requests. Full-cohort cancellation synchronizes
the prefill stream before ACK; invalidation removes cached input rows from
both runners. These protections are independent of TP size. This audit found
no TP4-specific alias mechanism, but it does not establish that every vendor
kernel or hidden workspace is safe under concurrent streams.
Sources: [cohort/cancellation guards](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/decoupled_worker.py#L100-L144),
[both-lane invalidation](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/src/vllm_muxpd/worker.py#L329-L344).

A source search plus review of the inspected execution paths found no
`TP == 2` or two-rank loop assumption in the plugin's `src` tree. At the pinned
implementation commit, the launcher does default to `--tensor-parallel-size 2`;
its final `"$@"` permits an explicit trailing `--tensor-parallel-size 4` override.
Use the TP4 experiment's documented launcher configuration and verify the
resolved size in startup logs. [Pinned launcher](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/784236edf7bc99ff71d8df7b9b93168164a0f72a/scripts/serve-qwen35.sh#L7-L19).

Hardware validation should compare against a **TP4 native matched control**,
check isolated and concurrent greedy outputs, long-prefill progress, cancellation
and KV reuse, and four-rank HCCL progress in both fenced and overlap modes.
Performance must be measured independently of correctness checks and profiling.
Any overlap claim requires per-rank device-compute intervals and fresh lane-to-
physical-stream mappings; TP2 stream IDs or HCCL wrapper spans are insufficient.
