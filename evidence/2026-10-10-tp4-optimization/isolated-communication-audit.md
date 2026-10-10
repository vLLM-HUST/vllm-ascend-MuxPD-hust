# Isolated P/D communication: ownership and mutable-state audit

Date: 2026-10-10. Static source review and CPU simulations only; this document
does not establish TP4 HCCL correctness, kernel concurrency, or performance.

Scope: Ascend A2, Qwen3.5-35B-A3B BF16, V1, eager, TP4 + EP4, DP/PP/PCP = 1,
text-only synchronous sampling, no speculative decoding, prefix caching, LoRA,
quantization, EPLB, FlashComm, fine-grained TP, or shared-expert side stream.
The plugin worktree is `perf/tp4-execution-overhead`; native revisions are
vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804` and
vllm-ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`.

## Decision

No additional *consumed, per-forward shared tensor scratch* was identified in
the scoped eager GDN, full-attention, or AllGather MoE path beyond the intentionally
shared KV/recurrent cache and already isolated workspace manager. The stronger
claim that the model has no mutable shared state would be false: native code
writes a shared MC2 mask even when its selected AllGather path does not consume
that mask, and rotary embedding rebinds temporary tensor attributes. The reasons
those sites do not currently invalidate this execution mode are listed below.

Submitting the entire P quota before the D network still switches only at a
decoder post-hook, after all Python code for a layer has returned. It does not
introduce host reentry inside a MoE prepare/finalize pair, rotary op, or attention
function. Device operations remain asynchronous; the analysis therefore also
distinguishes tensor rebinding from in-place writes to reused storage.

## Source findings

| Area | Concrete source evidence | Meaning for independent streams |
| --- | --- | --- |
| Ascend extra forward state | [`_ExtraForwardContextProxy`, lines 406–463](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ascend_forward_context.py#L406) delegates reads/writes to `get_forward_context()` | Saving/restoring the actual vLLM forward-context object also restores `moe_comm_method`, layer index, token counts, capture flag, and other proxy fields. The proxy is not a second independent mutable context that the backend forgot to restore. |
| GDN temporary tensors | [`AscendGatedDeltaNetAttention.forward`, lines 74–152](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L74) allocates local projections and `core_attn_out`; `_forward_core` reads metadata from the current context | No `self` scratch-tensor assignment occurs in this implementation. Each lane obtains its own activations and metadata. |
| GDN recurrent state | [lines 178–180 and 394–422](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L394) select shared state by `prefill_state_indices`, create a local initial-state tensor, then write final state to those indices. Decode passes separate metadata indices to its fused state update. | Shared recurrent storage is intentional. Safety depends on disjoint request slots and retaining the entire pending cohort's allocation until completion/cancellation ACK. Full cancellation explicitly synchronizes the P stream before ACK. |
| Full attention cache references | [`forward`, lines 1634–1667](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/attention/attention_v1.py#L1634), and `reshape_and_cache`, lines 1556–1579 | `self.key_cache/value_cache` lazily bind the same intentionally shared cache, not a batch-sized scratch allocation. Writes use lane metadata's slot mapping. Both runners attach exactly the same cache tensors. |
| Full attention outputs/workspace | [`forward_fused_infer_attention`, lines 1300–1401](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/attention/attention_v1.py#L1300), [`forward_paged_attention`, lines 1483–1502](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/attention/attention_v1.py#L1483) | Eager FIA returns local output; eager PA writes the caller-supplied lane output. The global GraphParams workspace/handle path is selected on capture and is outside this eager review. A decode-graph change needs its own review. |
| MoE singleton | [`_MoECommMethods`, lines 51–65](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/moe_comm_method.py#L51) and [`prepare/fused_experts/finalize`, lines 636–709](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/fused_moe.py#L636) | The communication helper object is shared. Its scalar preparation fields are consumed before the complete decoder layer returns, so another lane cannot alter them mid-call. This conclusion would fail for an intra-MoE greenlet yield. |
| MoE tensor/event lifetime | [`TokenDispatcherWithAllGather`, lines 356–464](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L356), [`fused_experts`, lines 149–185](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/fused_moe/moe_comm_method.py#L149) | Sorted tokens, indices, routing weights, and communication events are returned in per-call metadata/results rather than overwritten in model attributes. EPLB's in-place `moe_load` writes and shared-expert side-stream operation are disabled by the worker contract. |
| Reserved MC2 mask: real shared writes | [`set_ascend_forward_context`, lines 222–231](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ascend_forward_context.py#L222), allocation at lines 273–282 | Both lanes can enqueue writes into `_reserved_mc2_mask`. This is shared mutable device storage. In the scoped A2 TP4 case the selector returns ALLGATHER (lines 294–301), and AllGather preparation returns `mc2_mask=None` (prepare_finalize.py lines 419/476); downstream current kernels therefore do not consume the raced buffer. Do not enable MC2 without making this storage lane-local or adding dependencies. Removing unused writes is a separate optimization, not included here. |
| Rotary tensor attributes | [`AscendMRotaryEmbedding.forward_triton`, lines 487–516](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/rotary_embedding.py#L487) resets `self.cos/sin`, then binds newly allocated tensors | Rebinding Python references does not overwrite P's tensor storage. Ops are launched before the post-layer yield; allocations/freeing follow their originating stream. The cache dtype/device conversion is lazy; native startup warmup must finish before serving mixed traffic, as it does in the current startup path. This is not proof for a different allocator or an intra-op yield. |
| Cached request metadata and sampler | [`sample_tokens` request update, lines 2576–2606](https://github.com/vLLM-HUST/vllm-ascend/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py#L2576) iterates that runner's input-batch IDs; [`_update_states_after_model_execute`, lines 1527–1539](https://github.com/vLLM-HUST/vllm/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/worker/gpu_model_runner.py#L1527) returns immediately without speculative decoding | The request dictionary is shared, but the worker rejects IDs scheduled in both lanes and pending-P lifecycle changes before ACK. Sampling mutates only that lane's request IDs. P is sampled once only when its full forward completes; D sampling keeps its native device-to-host completion synchronization. |

## Communicator ownership

`init_model_parallel_group` returns a new coordinator and does not replace global
TP/EP state ([native parallel_state.py:1298](https://github.com/vLLM-HUST/vllm/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/distributed/parallel_state.py#L1298)).
The worker clones membership, local rank and backend in deterministic TP then EP
order, using `tp:muxpd_prefill` and `ep:muxpd_prefill` names. It warms each returned
P communicator using a small all-reduce on P's stream and synchronizes before
installing dispatch wrappers. This occurs before memory profiling finishes, so
materialized HCCL allocations can be included in the non-Torch memory accounting.
Shape-specific subsequent library allocations still require real startup checks.

Routing targets `device_communicator.all_reduce`, `all_gather`, `reduce_scatter`.
Only P calls use siblings; caller/native warmup and D use saved native methods.
No cross-lane collective gate is installed in isolated mode. New-group aliases
are deduplicated by native device-group identity. The constrained AllGather
path is essential: arbitrary direct `dist` operations or MC2's cached HCCL names
would bypass these three routes.

Cleanup synchronizes both lane streams, restores only owned method wrappers,
then destroys only owned P coordinators in reverse creation order. The native
coordinators remain intact. Failed synchronization retains ownership and avoids
destruction. If native group construction itself raises before returning a
coordinator, Python has no returned object to destroy; rank/process shutdown is
still needed to resolve a partially failed distributed initialization.

## CPU verification completed

Command, run in the plugin checkout without any NPU initialization or API load:

```bash
source /root/workspace/runtime/env.sh
/root/workspace/.venv/bin/python -m pytest \
  tests/test_worker_communication.py tests/test_decoupled_worker.py \
  tests/test_lanes.py tests/test_overlap.py tests/test_communicators.py -q
```

Result: **73 passed, 1 skipped, 22.44 s**. The skip is an overlap-specific assertion
for the fenced parameter. Ruff passed on the worker modules and their tests.

Coverage includes same-membership TP/EP creation and aliases; warmup order;
caller/D/P routes; no change to global coordinators; shared-mode compatibility;
partial initialization rollback; synchronization failure retaining ownership;
destroy failure and retry; P-first full-quota submission and complete D forward;
short-D/long-P progression; last-layer retirement; whole/partial/unstarted
cohort cancellation; pause; no-D completion; progress receipts; single P sample;
and shutdown of a suspended P before communicator destruction.

## Remaining TP4 validation

1. Startup: all four ranks must report matching original/sibling group membership
   and deterministic unique names, finish HCCL warmup, and reach health without
   additional-memory exhaustion. Record KV capacity as sibling communicators
   may reduce available cache memory. Include the first mixed request after health.
2. Correctness: run the existing deterministic/natural tests plus a retained long
   P with sustained D; verify D advances before P completion, complete P progress
   is exactly the layer count, and completed P samples once. Test a short P that
   finishes in its admission iteration and changing slice quotas.
3. Lifetime: full cancellation while P is suspended, partial cancellation followed
   by survivor completion, and repeated abort/reuse cycles with new requests to
   exercise recycled attention and recurrent slots. Confirm no ACK before P stream
   completion and no stale sample from a cancelled request.
4. Communication: inspect all-rank logs for HCCL order/timeouts and capture a short
   device trace with physical stream mapping. Demonstrate that P and D actually
   use separate communicator launches; TP4 CPU tests cannot prove this. A counter
   alone is dispatch evidence, not kernel concurrency evidence.
5. Performance: compare the same prompts, concurrency, output lengths and server
   configuration with matched native and prior plugin modes. Keep profiler runs
   separate. Independent groups remove the artificial cross-lane collective
   chain but do not reserve hardware resources or guarantee net acceleration.
6. Shutdown/restart: stop with a pending P and then relaunch, checking that custom
   group cleanup does not damage native groups or strand distributed peers.

The prospective decode graph path is explicitly excluded from this conclusion:
global graph attention metadata/workspaces, capture-time lane routing, and replay
input lifetimes must be reviewed when that change is integrated.
