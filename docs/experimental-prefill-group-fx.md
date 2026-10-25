# Experimental four-layer prefill FX groups

This independent, default-off prototype starts from plugin `9e740ba`. TP4
validation at runtime `f09a100` now covers actual compilation on four ranks,
finite serving, and one intermediate cancellation followed by reuse. It remains
experimental: synthetic cross-role token sequences match in only 9/19 cases,
including early D differences tracked in
[Issue #11](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11).
The [validation report](validation-tp4-prefill-group-fx.md) records the evidence
and its limits. No group-FX throughput or Goodput benefit has been established.
This does not include the short-prefill PIECEWISE or pipelined-sampler experiments.

## Current runtime integration

The branch now includes main's native KV-reclamation compatibility, process-owned
ECPA output receipts and retryable shutdown. Closing P group handles happens only
after P/D/update stream joins, before closing the D controller; a failed group
close retains its handle and prevents native shared-model cleanup. These source
changes do not establish a numerical fix or requalify the historical device
results for this newer source tree.

The installed recipe accepts an explicit `--experimental-prefill-groups` flag on
`configuration`, `plan` and `serve`. Without it, the existing raw-P configuration
and launch command are unchanged. For example, after installing the candidate
build in the separately qualified native environment:

```sh
vllm-muxpd configuration --experimental-prefill-groups \
  --output muxpd-group-configuration.json
vllm-hust-ext extension configure org.vllm-hust.muxpd \
  --file muxpd-group-configuration.json
vllm-hust-ext extension enable org.vllm-hust.muxpd
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 --experimental-prefill-groups --dry-run
```

Configuration and dry-run do not launch a model. An actual launch additionally
requires explicit device selection and the normal `serve` call without dry-run.
Serve rejects a saved raw configuration when the group flag is selected and vice
versa. The flag projects `compile_prefill_groups=true` and selects the required
fusion backend with `enable_npugraph_ex=false` and `enable_static_kernel=false`;
the worker still checks the complete group recipe and native AOT requirement.
This remains an unqualified experiment, not a change to the alpha release's
published hardware qualification.

A matched raw-P control must use the same fusion-backend settings explicitly in
its host configuration. The normal raw-P launch recipe retains its prior native
backend default, so comparing that recipe directly with the group flag would
also change D compilation and cannot isolate the effect of grouping P.

## Execution and restrictions

With `compile_prefill_groups=true`, the P runner executes the existing embedding,
ten consecutive four-layer group functions, and the existing final norm. Each
group references the four actual loaded layers, preserving HUST's layer forward,
GDN/full-attention/MoE implementations and weight/KV identities. This is FX
execution in `NONE` mode, **not P ACL graph capture/replay**.

The only accepted combination is decoupled execution, D FULL graph, overlap with
isolated P/D communication, fixed `layers_per_slice=4`, zero token-work budget,
and no adaptive, decode-budget guard, prefill-first or whole-prefill compilation.
The existing worker restrictions still apply: fixed Qwen3.5 path, PP1, no SP,
LoRA, quantization, speculative/structured outputs, request logprobs or unsupported
mutable model extensions. The prototype requires exactly 40 layers and no
auxiliary hidden-state outputs. Foreign decoder hooks are rejected.

The experimental policy settings are:

```json
{
  "execution": "decoupled",
  "stream_mode": "overlap",
  "overlap_communication": "isolated",
  "decode_graph": true,
  "compile_prefill_groups": true,
  "layers_per_slice": 4,
  "prefill_layer_token_budget": 0,
  "adaptive_layers": false,
  "prefill_first": false,
  "decode_budget_guard": false,
  "compile_full_prefill": false
}
```

Keep the canonical `vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler` and
`vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker`. Use the existing supported
D `FULL_DECODE_ONLY` compilation configuration and D capture gears. The Ascend
additional configuration must explicitly set `enable_npugraph_ex=false` and
`enable_static_kernel=false` within `ascend_compilation_config`; native AOT must
be enabled. Changing this global compiler backend also affects D compilation.
A comparison must keep backend, q4, token budget, OMP settings and sampling equal;
it cannot attribute a bundled backend change entirely to grouped P execution.

## Compilation identities and ownership

Ten source-defined classes have distinct forward code objects, qualified names
and source lines. Native AOT keys use these identities; ten instances of one
class would be unsafe with the native guard-dropping wrapper. Each group receives
its own compilation-config copy and cache namespace. The static forward-context
mapping is copied shallowly so it continues to reference the real attention
layers; live modules and weight tensors are not deep-copied.

Startup profiles all groups on the P stream with the P workspace/communicator
routing, checks each actual `aot_compiled_fn` exists, and synchronizes the P
stream before profiling returns. Serving calls these sealed handles directly,
without passing through `support_torch_compile`'s `skip_compiled` shortcut. The
outer P context remains `NONE`, `capturing=False`, `skip_compiled=True`; group
compiler-config scopes end before any greenlet switch. Missing/replaced handles,
out-of-range token shapes, lane ownership changes and unsupported kwargs fail
closed. The single dynamic range ends at `max_num_batched_tokens`; native's
prepared range dispatch remains responsible for the generated backend. Fixed
handle identity excludes re-entering the decorator's compilation decision; it
does not prove every lower-level kernel, autotuning or shape-dependent path is
already warm. Startup cannot prove every opaque GDN/MoE/Triton path is free of
first-use work.

Only the P runner's model-call reference is temporarily proxied. D retains its
original shared model/outer FULL graph wrapper and capture/replay path. Native
`NPUModelRunner._model_forward` remains responsible for its own tail. Startup
scopes wrap P profile/dummy calls individually, never the entire worker warmup,
so D capture retains the native D process group. Shutdown joins P/D/update
streams before releasing group handles and restoring P instance methods.

This base has no `_decode_callable`/temporary `do_not_compile` bypass for D.
The original backbone need not have an AOT handle after grouped P profiling.
Ascend `platform.py:489` forces one graph warmup; native
`GPUModelRunner._warmup_and_capture:6727-6749` executes D in NONE mode before
FULL capture. The D ACL wrapper calls the original model in that NONE warmup,
so the original backbone compiles before capture, as it did with an ordinary
uncompiled P profile. This startup dependency must be retained and verified on
device; group compilation is not evidence that the D backbone is prepared.

## Progress, cancellation and diagnostics

A group calls the actual layer `.forward` methods, bypassing decoder module
post-hooks; otherwise a greenlet could switch inside compilation or count each
layer twice. Only a successfully returned group advances four layers. The new
`group_forward_return` receipt is validated by both scheduler and trace checker:
40 total layers, multiples of four, the existing expected quota and the same
cohort/ACK identities. The last group's progress does not retire KV: final norm,
native execute state and normal sampling must still finish before completion.

Full/partial cancellation continues to use the existing suspended-cohort state
and stream join before ACK. Cancellation only takes effect at group boundaries;
submitted device work cannot be cancelled by discarding a Python frame. A failed
group does not earn progress. Unwinding a paused coroutine restores the P model
reference without poisoning later cohorts; actual forward exceptions poison the
executor and require process restart. Same-process reinitialization is unsupported,
consistent with the native global GraphParams lifecycle.

Startup logs list actual entry names, cache directories and successful group
return counts (including startup). With policy `trace_path` enabled, each serving
group return additionally logs rank, cohort, group, cumulative layers and P token
count. These prove host call/return accounting, not device completion or absence
of baked communication. Device diagnostics must check all ranks, actual P/D
routing and outputs independently.

## Validation boundary and required next step

CPU tests use injected compiler handles and fake streams. The pinned-native
class fixture replaces `TorchCompileWithNoGuardsWrapper.__init__` before group
construction: it checks real decorated class identities and loaded-layer sharing
without invoking `torch.compile` or allocating device tensors. Other fixtures
exercise actual worker startup/teardown/progress methods, scheduler receipts,
cooperative cancellation, config restoration and missing/changed handles.

The completed TP4 diagnostic verifies ten AOT group startup returns per rank,
correct absolute GDN/attention layer prefixes in all 40 P artifacts, independent
D backbone compilation and four captured D entries per rank, P-only token shapes
1/128/129/16383, and a four-D/12K-P cancellation with ACK at 8/40 layers followed
by reuse. All 16 natural-language text/answer comparisons pass. These observations
do not resolve the early synthetic D differences or prove opaque communication
routing, kernel overlap, or every dynamic path being warm.

Further gates remain: same-input numerical/state diagnosis; direct device traces;
broader D1/4/7 and first/intermediate/final or partial-cohort cancellation coverage;
hardware KV pressure; and clean startup memory/compile-latency measurements.
The observed early fourth-token divergence precedes the later 12K request and
maps by the synchronous scheduling protocol to D1 plus pending P384. It must not
be dismissed as the previously observed P128 repeat variance. No clean performance
comparison followed this diagnostic. Only after the correctness boundary and
required device checks are resolved should a matched q4/backend benchmark assess
throughput/Goodput.
