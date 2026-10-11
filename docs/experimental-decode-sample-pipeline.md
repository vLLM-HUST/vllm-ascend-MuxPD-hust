# Experimental early greedy decode sampling

`decode_sample_pipeline` is disabled by default. It is permitted only with
`execution="decoupled"`, `decode_graph=true`, `stream_mode="overlap"`, isolated
communication, and the existing synchronous, BF16 Qwen3.5 worker restrictions.
This candidate does not implement async engine scheduling or early output/KV
acknowledgement. It has no measured model benefit yet.

On a D+P transaction, native D `execute_model` already submits both graph replay
and logits. The adapter restores the complete D lane and submits the native
greedy `_sample`, then a nonblocking copy into an additional D-owned pinned CPU
buffer and a private completion event. It next submits the existing P slice.
No CPU output is read and no request token history advances in this early phase.

At the original worker `sample_tokens` boundary, native `sample_tokens` runs
unchanged. Its `_sample` invocation consumes the already-submitted result once;
its `_to_list` invocation waits on the private copy event and reads those bytes.
All remaining native code still runs: clearing `ExecuteModelState`, recording
the native `sampling_done_event`, bookkeeping, token history, output creation,
and hybrid feedback on the native global stream. The private early copy event
is **not** the native hybrid-feedback event. Required feedback was not removed
or moved ahead of P. Native `transfer_event` is otherwise used only inside
`_to_list` in the pinned source.

The adapter leaves D-only, P-only, and cancelled/paused-P-with-no-P-submission
transactions on their previous path. It submits early only for an entire D
batch whose sampling metadata is all-greedy. Random/mixed sampling, penalties,
bad words, allowed-token masks, thinking-budget holders, async feedback,
discarded samples, logprobs, grammar and non-native logits processors take the
original path before any sample is submitted. Global custom logits-processor
configuration is rejected when the feature is enabled; entry-point processors
are also checked by exact native builtin type at runtime.

The supported native builtin list is exactly MinTokens, LogitBias and MinP
(`vllm/v1/sample/logits_processor/__init__.py:49-53`). The latter is skipped by
the all-greedy native sampler; native builtins mutate logits, not token history,
when applied. `sampled_token_ids_cpu` starts as None and is populated only by
the native async-output path, so the normal synchronous decode path can qualify.

Only one ticket may exist. Its original execute state, sampling metadata,
request ordering/mapping and sample tensor identity must remain intact until
native retirement. Duplicate consumption fails closed. Once any device work
has been submitted, an exception retains the ticket/buffer until a successful
D-stream shutdown join; a failed join preserves ownership for a cleanup retry.
The adapter never retries native sampling after such a failure. It restores only
its own instance method replacements, leaving unrelated replacements untouched.

With policy `trace_path` enabled, `MuxPD decode sampling diagnostic` JSON records
show rank, prefill cohort, submission number, D request IDs, enqueue/retire
phase, fallback reason and cumulative counts. An enqueue record must have
`private_copy_waited=false` and `native_retirement_completed=false`; its matching
retire record has both true. Trace-off does not build these records. Hardware
diagnostics must demonstrate `enqueued>0` and matching retirements on all ranks;
config acceptance or a fallback-only successful run does not exercise this path.

The initial narrow CPU run passed 32 adapter/configuration/fake copy-event
tests and skipped 171 import-gated tests. In a later authorized runtime-import
window, the complete plugin suite passed 702 tests with six mode-specific
skips. That run includes actual worker scope/shutdown fixtures and a fixture
calling the pinned native `_sample`, `_bookkeeping_sync`, and `sample_tokens`
bodies with CPU fakes. The native fixture verifies that bookkeeping, the native
sampling-done event and hybrid feedback retain their original retirement order.
It follows the native worker import order to avoid an upstream runner-first
circular import. No NPU tensors or model/runtime initialization were used.
These CPU results and two independent source reviews do not replace TP8
correctness, changing D batch/state reuse, cancellation, and device-trace
validation. Hardware performance remains unmeasured.

Expected benefit is limited to hiding some sampling/D2H tail under P submission.
The additional sampling launch itself delays the start of P, and no token is
returned before the original retirement boundary. This does not remove P host
spans, prove lower ITL, or guarantee greater useful throughput.
