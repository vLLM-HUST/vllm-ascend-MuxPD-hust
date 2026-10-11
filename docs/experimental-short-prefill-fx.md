# Experimental short-prefill FX with decode graph overlap

This prototype is disabled by default. It is not a per-layer compiler and does
not capture ACL graphs for prefill. Its purpose is to reduce Python submission
work for a complete short P while a separately captured D graph is in flight.
No model-level throughput or correctness improvement is established by the code
or by the CPU contract tests.

The additional policy option is `compile_short_prefill_tokens` (default `0`).
An initial diagnostic can use `512`, together with `execution="decoupled"`,
`decode_graph=true`, `stream_mode="overlap"`, and
`overlap_communication="isolated"`. `compile_full_prefill` must remain false.
The threshold applies to the **sum of scheduled P tokens in the fresh cohort**;
it does not include scheduled D tokens. A qualifying cohort receives an explicit
whole-model layer quota and must complete in that worker transaction. It cannot
suspend halfway through compiled code. Long P, resumed P, pause/cancel controls,
and any worker transaction without a complete quota use the existing raw path.

This policy changes short-P quota as well as execution. An FX-versus-raw ablation
must give both variants the same complete short-P quota; comparing FX/full40 to
raw/q8 does not isolate compilation. Small token count is not a D latency bound.
The existing D-first submission and native sampler completion boundaries remain.

Startup requires native AOT compilation, npugraph_ex, and static-kernel
compilation disabled. Native Ascend's pinned compiler sets `force_eager=True`;
npugraph_ex then calls its generated FX function without entering ACL capture.
The owner profile and finite prewarm execute in the P stream, P workspace and P
communicator route. The native profile produces the dynamic whole-model AOT
callable; no new compiler backend or weight copy is installed. Preparation:

- Requires dynamic `input_ids` and `positions` dimensions and a callable
  `aot_compiled_fn` already present after the native profile.
- Executes every exact `compile_size` and one non-exact representative of every
  reachable native compile range below the threshold (at most 64 entries).
  This also materializes npugraph_ex's lazily generated Python callable.
- Observes successful returns through that exact AOT callable. In serving, its
  replacement, a padded token count beyond the prepared limit, or a missing AOT
  return fails closed; it is never silently counted as a compiled hit.

The extra startup warmups end with a P-stream synchronization before returning
to native memory sizing and D capture. This is not added to the serving path.

This prevents a serving request from initiating a new **top-level Torch/FX**
compilation through the decorator. The native opaque GDN/MoE/Triton operators
retain their own first-use kernel initialization behavior. Their CPU work is
not eliminated merely by compiling the outer model. The successful-call counter
in the controller snapshot includes explicitly listed startup warmup calls.

With policy `trace_path` enabled, each short compiled P writes a begin/end
`MuxPD short FX diagnostic` JSON record to the worker log. It includes rank,
cohort/request IDs, aggregate P tokens, the actual AOT successful-return delta,
and P/D/caller router-call deltas for that one native P execute call. It records
no prompt or generated tokens and adds no device synchronization. Trace-off
does not gather these diagnostic snapshots. A successful end should have AOT
delta 1; TP8 should also show P router calls and zero D/caller deltas. This is
host-call evidence, not a device-completion ACK, and cannot by itself exclude
communication that bypasses the router; source review and device traces remain
necessary.

The shared model's AOT callable gains an instance observer, restored on shutdown;
its D capture calls execute the original callable and are not counted as P.
Captured D replay bypasses this Python observer. No global TP/EP coordinator or
GraphParams registry is replaced. P runs with NONE / capturing=False and uses
its existing workspace. In the pinned generated graph, `vllm.all_reduce` remains
an opaque custom op; its coordinator calls the lane router at runtime. GDN and
MoE similarly resolve current forward metadata at runtime.

Relevant pinned native source contracts:

- `vllm_ascend/compilation/compiler_interface.py:114` sets force-eager options.
- `torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py:821` dispatches
  force-eager execution before graph capture/cache mutation.
- `vllm/compilation/decorators.py:518` calls an existing AOT artifact directly.
- `vllm/compilation/piecewise_backend.py:358` selects an already compiled entry;
  it does not compile new ranges in serving.
- `vllm/distributed/parallel_state.py:130` dispatches the collective custom op
  back to its coordinator and then the installed communicator method.

Validation still required before any performance claim:

1. CPU protocol tests: limits and range coverage; startup scope restoration on
   failure; actual-return accounting; D capture unchanged; fresh versus resumed
   P; full quota, aggregate token boundary, cancellation and late-abort KV pins.
2. Real TP8 startup: no unexpected top-level compilation after ready; all ranks
   show P/D distinct PGs and stream mapping, D capture/replay succeeds, every
   selected prewarm entry returns through AOT.
3. Real GDN and full-attention layers: short aggregate sizes 1/128/512, threshold
   boundary and 513 raw fallback; disjoint state/KV reuse; varied D1/D4/D7 with
   max_num_seqs=8. D8 plus P needs a separately labelled max_num_seqs=16 run.
4. Natural exact comparison, existing native variance-aware synthetic/state
   reuse comparison, partial-cohort cancellation, long-P progress, and return
   to D-only graph execution. CPU fakes cannot prove these device properties.
5. Profile actual FX execution, D ACL replay, group/stream routing and kernel
   intervals. Compare elapsed windows and useful output, not overlap percentage
   alone. Only after these checks, run clean performance tests with an identical
   workload and quota-controlled raw comparison.

Native graph globals remain process-owned. Reinitializing a second model engine
inside the same worker process is unsupported; process exit releases them.
