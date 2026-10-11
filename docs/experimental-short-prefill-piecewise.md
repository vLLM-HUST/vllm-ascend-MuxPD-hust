# Experimental whole-short-prefill PIECEWISE graphs

This independent candidate remains **disabled by default**. The TP8 finite
126-request diagnosis of runtime `b95372ee9c5f5f568c6d0899724ecd66a48a527b`
passed its host graph/path and lifecycle contract, and natural-language text
matched the reference in 16/16 cases. A repeated synthetic P128 request diverged
at generated-token index 4; matched-input numerical diagnosis is still needed.
There is no candidate device profile or performance result. It is based on
plugin `9e740ba` and excludes the separate short-FX and early sampling experiments.

The proposed setting is `piecewise_short_prefill_tokens: 512` with
`execution: decoupled`, `decode_graph: true`, `stream_mode: overlap`, and
`overlap_communication: isolated`. `compile_full_prefill` must remain false.
Use the native V1 `VLLM_COMPILE` / `FULL_AND_PIECEWISE` configuration with an
explicit sorted gear list ending at 512 and containing `max_num_seqs`, for
example `[1,2,4,8,128,136,256,512]` for eight request slots. These eight gears were
used in the finite diagnosis; P129 pads to 136. At most 16 gears are
accepted. These are experimental preparation parameters, not recommendations.
The native fusion-pass backend must be selected; `enable_npugraph_ex`, static
kernels, shared `cudagraph_copy_inputs`, and Inductor graph partitioning are
rejected. All existing worker model, dtype, synchronous scheduling, and
communication restrictions still apply.

A fresh P cohort is eligible only when its **aggregate full prompt token count**
is within the threshold. The scheduler explicitly requests all decoder layers
and completion for that transaction. The worker never calls a whole graph while
reporting a partial layer quota. Existing, paused, or cancelled cohorts use the
raw hook path; larger prompts retain the existing layer quota. No hook/yield is
inserted inside a compiled graph. Full-forward receipts still require every
pending request to complete together; late aborts retain the existing ACK and
KV ownership checks. Sampling and frontend delivery remain at the original
worker boundary. A whole short P can still delay D sampling.

## Ownership and preparation

Weights and disjoint-request KV remain shared. P and D retain separate native
runners, input buffers, attention builders, streams, workspace managers, and
isolated P TP/EP communicators. This candidate adds graph ownership as follows:

1. Owner profiling runs in the real P stream/workspace/router scope. It compiles
   the shared model in NONE mode and discovers newly created native PIECEWISE
   wrappers. Before any capture these wrappers receive a fresh P graph pool.
2. Each runner's dispatcher keeps its own gear set: P keeps only nonuniform
   PIECEWISE descriptors up to the threshold; D keeps only uniform FULL
   descriptors up to `max_num_seqs`. P with one token cannot select D FULL.
3. After KV initialization, P's native capture runs first in the P scope. Every
   owned wrapper must have every prepared gear. The helper restores the previous
   `capturing` flag after each native subgraph returns, so uncaptured GDN and
   attention execute outside D's full-graph update path. A snapshot checks that
   P capture did not mutate D's process-global GraphParams.
4. D's existing outer FULL capture then runs on D's stream/workspace/native
   communicator and global graph pool. Internal PIECEWISE wrappers see mode
   FULL and invoke their underlying runnable; they do not replay P's entries.
   The two native graph memory results are summed rather than assuming the
   pools overlap. Startup scopes synchronize P before returning.
5. During a selected short P, every owned wrapper must replay exactly once.
   Missing entries, changed pools or tensor input addresses, wrong lanes, and
   unowned wrappers fail closed before native lazy capture. The successful
   whole-forward return is distinct from device completion and sampler ACK.
6. Shutdown joins the existing P/D/update streams before releasing graph
   references. The helper removes only its own wrappers' entries and its own
   method guard; it does not clear D's global GraphParams or other owners' graph
   state. In-process model reinitialization remains unsupported.

Preparation does not prove every opaque GDN/MoE/Triton kernel is warm. Those
operators may still perform first-use work. PIECEWISE deliberately excludes
attention/GDN custom operators and therefore cannot eliminate their host calls
or required device waits. Native TP8 diagnosis already observed successful
P128 PIECEWISE replays with a long host envelope; replay count alone is not a
speedup claim.

## CPU validation record

No test allocates NPU tensors or starts a server. Tests cover ownership/foreign
wrappers, bounded preparation, P/D capture order and separate pools, native
single-token dispatcher behavior, changed input-address rejection, missing
entry rejection before lazy capture, capture failure, caller restoration,
startup synchronization, short aggregate threshold, fresh/full quota,
raw continuation/cancellation, and strict trace receipts.

- Pure helper/controller/existing graph/trace checks: **154 passed, 1 skipped**.
- Pinned native imported affected suite: **504 passed, 2 skipped, 4 failed**.
  All four failures came from the existing fake worker omitting the real
  `NPUWorker.vllm_config` attribute used by the new configuration hook.
- After correcting that fixture and adding two startup scope tests, the whole
  worker graph file passed: **26 passed**. This is a targeted rerun, not a claim
  that the entire affected suite was rerun after the fixture change.
- Final pure helper/controller/graph/trace run, including close cleanup and
  bounded pool manifests: **156 passed, 1 skipped** in 0.71 seconds.
- Final pinned native affected suite after the fixture correction, including
  trace on/off and failed-forward diagnostic checks: **516 passed, 2 skipped**
  in 27.10 seconds. This covers the helper, controller, decode graph, worker
  graph, decoupled worker, scheduler, trace checker, and policy test files.
- Ruff on every changed Python file and `git diff --check`: passed.
  All CPU checks terminated. The separate device diagnosis is summarized below.

The final native command sourced `/root/workspace/runtime/env.sh`, prepended
this worktree's `src` directory to `PYTHONPATH`, and invoked the pinned venv's
`python -m pytest` on the eight affected files above. Tests import the native
runtime but use fake streams/tensors for worker execution. They do not start a
server, allocate NPU tensors, or establish real HCCL/ACL correctness.

The native dispatcher test invokes the pinned `CudagraphDispatcher` with CPU
configuration objects. It confirms P1 and P129 select PIECEWISE, D1 selects
FULL, D129 and P513 fall back to NONE, and each lane captures only its own
bounded descriptor list. Fake controller tests do not prove real ACL capture,
static addresses, HCCL behavior, or concurrent model correctness.

## Finite TP8 diagnosis and remaining numerical gate

The stage `short-piecewise-fixed4-omp1-diagnostic` completed its registered
126 requests: 80 core requests, natural16, and state-reuse30. The independent
`short-piecewise-fixed4-omp1-diagnostic-independent-proof.json` reports a passed
host graph/path and lifecycle contract on all eight ranks. Its
`physical_device_profile_complete` is false: successful replay calls and host
observations do not prove kernel completion, physical overlap or performance.
The core includes 78 complete responses and two planned client cancellations;
cancelled streams are not counted as complete generation.

Natural-language text matched the existing baseline in 16/16 cases and passed
16/16 oracles. This is a text comparison, not a token-ID equivalence proof.
The independent numerical review then identified a separate unresolved result:
two sequential P128-only requests with identical prompt IDs and sampling fields
(apart from request ID) produced different eight-token outputs beginning at
zero-based index 4. Both had complete prompt-ID echoes, usage, finish and DONE.
All eight ranks observed the intended gear128 replay with stable entry/address
metadata, but those checks do not inspect KV/GDN state or tensor contents.

The older native P128 repeats were stable, but used a **different prompt token
array** and generated 16 rather than eight tokens. They are not a matched
reference and cannot establish a candidate regression or explain it away.
This P128 result must not be folded into the already known native one-/two-token
instability. A separately registered matched-payload numerical comparison is
required before advancing the candidate. No cause—numerical variation, graph
replay, or cache/state aliasing—has been established.

The retained numerical records are
`short-piecewise-fixed4-omp1-diagnostic-numeric-review.{json,md}` and
`short-piecewise-fixed4-omp1-diagnostic-p128-supplement.{json,md}` under
`local-results/tp8-goodput`. Their source reports and raw streams remain intact.
Host/path proof PASS does not close this numerical gate or justify timing.

## Device evidence requirements before timing

Use a separate immutable source snapshot and fresh worker processes. First
verify all ranks' startup captures, private P pool/gear manifests, D FULL replay,
and P PIECEWISE replay on the intended physical streams. The trace-only marker
`MuxPD short PIECEWISE diagnostic` records rank, cohort, ordered request IDs,
aggregate P tokens, and before/after aggregate capture/replay counts. The
snapshot also records P and D pool identifiers, an actual pool equality check,
and each P wrapper's mode and entry gears. TorchNPU's two-integer native handle
is recorded by value; a nonstandard pool type is explicitly labelled as Python
object identity only, which cannot independently establish different device
pools. Manifests are bounded to 256 wrappers and 16 entries each; truncation is
explicit and invalidates a claim that every wrapper/gear was checked. These
host-only snapshots read no tensor contents and enqueue no device operations.
A successful host return must add one whole forward and aggregate replays equal
to the wrapper count while capture count stays fixed. Exactly one replay per
wrapper is enforced by the pinned serving guard; the aggregate marker alone
does not independently establish that distribution. These counts do not establish
device completion, exclude all baked communicator handles, or demonstrate overlap;
correlate them with source and device traces.

Then compare natural text and raw generated token IDs against matched native
and raw-plugin controls, respecting the known native synthetic instability.
Exercise GDN and full-attention layers, P sizes 1/128/129/256/511/512/513,
changing D batch sizes 1/4/7, aggregate 2x256 and over-threshold cohorts, repeated
gears, cancelled/late-aborted cohorts, preemption/KV reuse, and long raw P
continuations. Eight active D requests leave no P slot at `max_num_seqs=8`;
D8+P needs a separately declared larger-sequence stage.

For attribution, a future raw control must hold full-layer quota, scheduler
budget, compiler backend, and gear configuration constant. This candidate does
not add that control switch: disabling the option keeps the existing
FULL_DECODE_ONLY restriction. A direct comparison against the existing
npugraph-ex path therefore changes more than P capture. Private pools can also
increase reserved graph memory; startup may fail if the remaining device memory
or stream resources are insufficient. Only measure throughput/Goodput after
transport and lifecycle checks;
keep all failures and exclude diagnostic profiling from timing. No benefit is
assumed from increasing the percentage of overlapping device intervals alone.


## Source-only baseline note

A stronger native comparison can retain a 16K scheduler token budget while
limiting PIECEWISE capture to short gears ending at 512. In the pinned
`vllm/v1/cudagraph_dispatcher.py`, `dispatch()` returns NONE whenever
`num_tokens > max_cudagraph_capture_size` (around lines 270–281), independently
of the scheduler's token admission budget. Thus a native 16K-budget,
FULL_AND_PIECEWISE configuration can in principle replay short P and D while
executing larger P in NONE mode. The separate native 16K/cap512 stage now
provides a reference, but its original
P128 synthetic inputs differ from this candidate's inputs as described above.
A slower native 512-token-budget service is not sufficient as the sole
performance baseline. Native admission/chunking, asynchronous scheduling, and
all other launch settings must be declared and kept distinct in comparisons.


## Finite launch settings and original planning outline

The registered 126-request controller expanded this original outline. Its
retained plan/report, rather than the table below, defines the exact executed
case inventory. No additional requests or profile run are authorized by this
document.

Start from a new immutable snapshot of this branch, not the early-sampling or
short-FX source. Relative to the existing isolated-overlap graph launcher:

- Keep TP8+EP, BF16, language-model-only, 16K model/token admission budgets,
  eight request slots, synchronous scheduling, no chunked prefill, no prefix
  cache, `mamba_cache_mode=none`, and `OMP_NUM_THREADS=1`.
- Keep the plugin decoupled scheduler/worker and isolated overlap. Select
  `layers_per_slice=4`, `prefill_layer_token_budget=0`, `adaptive_layers=false`,
  `prefill_first=false`, `decode_budget_guard=false`, `compile_full_prefill=false`,
  and `piecewise_short_prefill_tokens=512`. Omit `decode_sample_pipeline` and
  `compile_short_prefill_tokens`: those separate branch fields are unsupported.
- Replace FULL_DECODE_ONLY with `--no-enforce-eager --compilation-config
  '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes":[1,2,4,8,128,136,256,512],
  "max_cudagraph_capture_size":512,"cudagraph_copy_inputs":false,
  "use_inductor_graph_partition":false,
  "pass_config":{"enable_sp":false,"fuse_gemm_comms":false}}'`.
  Explicitly set `--additional-config
  '{"ascend_compilation_config":{"enable_npugraph_ex":false,
  "enable_static_kernel":false}}'`. Preserve the other validated launch flags.
- Use a new port/source/output directory and trace path; pin the producer and
  candidate trace checker separately. Do not use the stock eager wrapper
  script without replacing its eager setting. None of these notes launches a
  process or authorizes this subagent to access the service.

A small initial matrix can stop at the first transport/lifecycle failure:

| Finite case | Required observation |
| --- | --- |
| Startup once, all eight ranks | P pool handle differs from D; complete manifests; all P wrappers PIECEWISE with eight gears, D FULL with 1/2/4/8 only; global params unchanged during P capture. |
| D-only batches 1, 4, 7, four generated tokens each | After their initialization prefills finish, pure-D intervals replay existing FULL entries; P counters remain unchanged during those intervals. |
| P128-only twice | Existing P entries replay without additional capture; no D graph entry is used for P. |
| P128, P129, P512, each with D1 and D4 | Six fixed cases; P129 pads to P136; each P whole return adds one replay per wrapper, no captures; D completion/ACK remains native. |
| P512 with D7 | One near-capacity slot case; no D8+P claim under eight request slots. |
| P513 with D4; P12K with D4 | Two raw controls; no P graph replay, layer-hook progress and full eventual ACK; 12K uses q4 when D is selected. |
| Two P256 with D4; two P257 with D4 | Two bounded admission attempts. Actual trace must establish a single aggregate 512/514-token cohort; otherwise report the co-admission coverage as missing, without unregistered retries. |
| One long-P cancellation and one short-P abort attempt | Long P only frees KV after stop/fence ACK. Short P abort timing must be classified by actual scheduler/worker trace; an abort delivered after completion is not evidence of in-flight cancellation. |
| Existing natural16 and bounded state-reuse helper | Preserve raw generated IDs and known native short-prompt instability; do not turn transport success into a semantic correctness claim. |

Only after these checks, collect one registered short device profile covering
P128/P129/P512 plus D activity. Match rank/cohort and actual physical stream
mapping, distinguish graph replay API success from kernel completion, and check
P versus D compute intervals and communication ownership. A finite diagnostic
is not a throughput measurement; no repeated capacity sweep is implied here.
