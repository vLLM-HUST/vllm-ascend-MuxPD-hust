# Experimental four-layer prefill FX groups

This independent, default-off prototype started from plugin `9e740ba`. Historical
TP4 validation at runtime `f09a100` covered actual compilation on four ranks,
finite serving, and one intermediate cancellation followed by reuse. It remains
experimental: that historical synthetic cross-role comparison matched only 9/19 cases,
including early D differences recorded in
[Issue #11](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11).
The [validation report](validation-tp4-prefill-group-fx.md) records the evidence
and its limits; the original failures are unchanged.
This does not include the short-prefill PIECEWISE or pipelined-sampler experiments.

The [follow-up investigation](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025)
adds same-card fixed-input controls, ordinary-worker replays and a separate
source98 deployment bridge. [Issue #11 was closed within that bounded opt-in scope](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11#issuecomment-6405978497).
This does not identify a unique cause of the old discrepancy, fix default
nondeterminism, prove hidden/KV-state identity or establish arbitrary-input
equivalence. Current quantitative results and tradeoffs are maintained only in
the [canonical performance/lifecycle report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025/performance-lifecycle-v4).

## Current runtime integration

The branch now includes main's native KV-reclamation compatibility, process-owned
ECPA output receipts and retryable shutdown. Closing P group handles happens only
after P/D/update stream joins, before closing the D controller; a failed group
close retains its handle and prevents native shared-model cleanup. These source
changes do not requalify the historical device results for a newer source tree.
The later [source98 ordinary-worker and Manager bridges](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025/device-evidence-v3)
have their own source and configuration identities. This documentation retains
the runtime bytes of `98eac12dc2595a8742a81b58e96b899fed905194`, the actual
source used by those bridges; a later documentation commit is not a new tested
runtime identity or a newly qualified published wheel.

The installed recipe accepts an explicit `--experimental-prefill-groups` flag on
`configuration`, `plan` and `serve`. Without it, the existing raw-P configuration
and launch command are unchanged. For example, after installing the candidate
build in the separately qualified native environment:

```sh
vllm-muxpd configuration --experimental-prefill-groups \
  --deterministic-collectives \
  --output muxpd-group-configuration.json
vllm-hust-ext extension configure org.vllm-hust.muxpd \
  --file muxpd-group-configuration.json
vllm-hust-ext extension enable org.vllm-hust.muxpd
vllm-muxpd plan --model /path/to/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 --experimental-prefill-groups --deterministic-collectives
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 --experimental-prefill-groups --deterministic-collectives --dry-run
```

Configuration and dry-run do not launch a model. An actual launch additionally
requires explicit device selection and the normal `serve` call without dry-run.
This example explicitly selects the collective profile used by the bounded
numerical follow-up. Grouping P alone does not enable it. Keep both flags
consistent across saved configuration, plan and serve; neither is a default.
Serve rejects a saved raw configuration when the group flag is selected and vice
versa. The flag projects `compile_prefill_groups=true` and selects the required
fusion backend with `enable_npugraph_ex=false` and `enable_static_kernel=false`;
the worker still checks the complete group recipe and native AOT requirement.
This remains an experimental, default-off path with bounded source-specific
device evidence, not a change to the alpha release's published qualification or
a general output-quality guarantee. The collective option's environment and
conflict checks are described in [deployment](ecpa-deployment.md#实验性确定性通信配置).

A matched raw-P control must use the same fusion-backend settings explicitly in
its host configuration. The normal raw-P launch recipe retains its prior native
backend default, so comparing that recipe directly with the group flag would
also change D compilation and cannot isolate the effect of grouping P.
The Manager bridge establishes activation and serving continuity; its raw/group
presets use different backends and its API responses do not contain token IDs.
It is not the matched-fusion numerical or performance comparison.

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
Stream resource quotas remain disabled in this fixed recipe. The native V1
runner enables internal formats, which does not satisfy the quota API's format
precondition. Separate P/D execution is not the paper's Cube/Vector spatial
partitioning, and host progress does not demonstrate physical kernel overlap.

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

## Historical validation boundary and current follow-up

CPU tests use injected compiler handles and fake streams. The pinned-native
class fixture replaces `TorchCompileWithNoGuardsWrapper.__init__` before group
construction: it checks real decorated class identities and loaded-layer sharing
without invoking `torch.compile` or allocating device tensors. Other fixtures
exercise actual worker startup/teardown/progress methods, scheduler receipts,
cooperative cancellation, config restoration and missing/changed handles.

The historical `f09a100` TP4 diagnostic verified ten AOT group startup returns per rank,
correct absolute GDN/attention layer prefixes in all 40 P artifacts, independent
D backbone compilation and four captured D entries per rank, P-only token shapes
1/128/129/16383, and a four-D/12K-P cancellation with ACK at 8/40 layers followed
by reuse. All 16 natural-language text/answer comparisons pass. These observations
do not resolve the early synthetic D differences or prove opaque communication
routing, kernel overlap, or every dynamic path being warm.

That early fourth-token divergence preceded the later 12K request and mapped by
the synchronous scheduling protocol to D1 plus pending P384. It cannot be
dismissed as the earlier P128 repeat variance. The later fixed-input numerical
investigation includes this admission pattern, with its capture limits and
source boundaries recorded in the canonical report; the old failure has not
been relabeled as a pass.

Completed follow-up includes ordinary-worker comparisons, source98 activation
and serving bridges, bounded cancellation/ACK/block-release/reuse observations,
and real allocation/preemption pressure under the declared pool geometries.
The [lifecycle and matched synchronous performance report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025/performance-lifecycle-v4)
retains the original small-pool failure and subsequent explicit headroom
treatment. These runs do not cover every D shape, cancellation point, pool size
or dynamic compiler path, and do not supply clean startup-memory or compile-time
measurements.

The [completed current assessment](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ebddb585390c333ac1f8c6ea40cfeead3c4321af/reports/muxpd-followup-20261025/quality-reference-v5)
includes the full paired task-quality comparison, separate practical native-default
reference and actual mixed-work profiles. On the fixed f7 task set, group preserves
raw's complete outputs, including its native-relative answer regressions. Both
fail the prespecified supplementary answer-nonregression criterion: a higher
total correct count cannot offset errors on cases both native controls answered
correctly. The original strict no-truncation failure also remains unchanged.
[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) retains
this known quality limitation.

The measured benefit is a shorter worst decode pause during long-prefill ingress.
The group/raw throughput increment is modest; group is substantially slower than
the separate native default-async/default-backend reference, with worse TTFT and
ordinary gap P99. Native/plugin mixed token trajectories also differ. Matched
profiles confirm actual P/D work but show no reduction in exported kernel-launch
counts, so fewer kernel launches cannot explain the observed timing increment.
[PR #10](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/10) is an optional
experimental increment with these inherited limits, not a native-relative quality
pass or a new qualification of the published wheel.
Timing runs, full-logit diagnostics, profiler traces and HTTP Goodput have
different scopes and cannot substitute for one another; none establishes
universal speedup, quality or deterministic outputs.
