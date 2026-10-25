# Fixed-token numerical diagnostic

This is an explicitly selected diagnostic, not normal plugin activation or a
performance preset. It preserves the native greedy call and returned object,
then saves bounded full-vocabulary logits after native synchronous bookkeeping.
Cloning, D2H, sorting and file writes add overhead and may perturb execution
timing. Results must not be used as clean latency or throughput measurements.

The first frozen plan has 15 requests: three fully drained single-P128 phases,
then three fully drained four-P128 phases. Each request produces exactly eight
tokens at temperature 0, seed 42, without EOS termination. Warm repetitions share
one model instance. Separate invocations with new output directories are required
for fresh-process controls. Do not present the later warm phase as a cold start.

`LLM.enqueue()` alone is insufficient to freeze arrivals when its coordinator is
a background process. The driver sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` before
importing vLLM and checks the actual `InprocClient`. TP4 workers still use their
configured multiprocessing executor. It adds every phase request before calling
the explicit engine step loop and fully drains before the next phase. Worker
receipts remain authoritative for the actual batch order and shape.

```sh
python scripts/probe_fixed_token_numerics.py plan --output /new/path/plan.json
python scripts/probe_fixed_token_numerics.py run \
  --plan /new/path/plan.json --plan-sha256 PRINTED_PLAN_SHA \
  --engine-args /frozen/path/native-engine-args.json --role native \
  --output-dir /new/path/native-fresh-0
```

Repeat with `--role raw` and a matched raw-P engine configuration, then repeat
both configurations in fresh processes on the same cards. The external root
controller owns device selection, host/model/source identities and process
cleanup. This diagnostic never edits native source or installed wheels.

Engine JSON must contain TP4, synchronous scheduling, prefix caching off,
`max_num_seqs=8`, and `max_model_len=max_num_batched_tokens=16384`. It must omit
`worker_cls`; the driver selects `NumericalNativeNPUWorker` or
`NumericalMuxPDNPUWorker`. Raw mode additionally requires chunked prefill off and
the normal raw-P scheduler/policy settings. Remaining model, EP, compilation,
dtype and host options must be matched and retained by the controller. Do not
enable `VLLM_BATCH_INVARIANT`, HCCL deterministic overrides or new fusion options
in only one endpoint. The worker records their observed environment/torch state.

All ranks record schedule and sampler row order, exact prompt/output prefix,
input-buffer tokens and request KV block IDs. Rank 0 additionally stores the
first eight complete logit vectors per request as non-pickle `.npy` files with
size/SHA256 bindings. The native copied token IDs are recorded separately from a
CPU argmax inspection; the latter never drives generation. Post-native-warmup
Triton `constant_args` metadata is captured on every rank without reading tensor
contents. Integer table IDs alone do not establish equal compiler semantics.

On the grouped-prefill integration branch, `--role group` requires
`compile_prefill_groups=true` and both `enable_npugraph_ex=false` and
`enable_static_kernel=false` under `additional_config.ascend_compilation_config`.
The receipt binds this role to the actual policy and backend settings; a raw/group
label mismatch is rejected before model initialization. Use the same combined
source commit and observer for the matched `--role raw` comparison, with the same
explicit backend settings and only the group policy flag changed. The released
raw default backend remains a separate comparison.

The grouped diagnostic observes at most 128 serving group-return events per
rank, before the original callback can switch lanes. It records the host MoE
index, ordered layer names, active lane, cohort and scheduler metadata, without
tensor reads or synchronization. Startup returns without a serving schedule are
omitted. For the qualified 40-layer model, expected post-group indices are
4, 8, ..., 40; this expectation must be checked against observations. No index is
modified, and these host values do not prove KV contents or kernel equivalence.

Acceptance must compare identical prompts, actual batching, output prefixes and
state histories. Start with native-versus-native warm and fresh controls. At the
first cross-mode token divergence, compare logits only if the histories are
still identical; later autoregressive logits generally refer to different
inputs. A small argmax margin or within-native variability alone is not a waiver
for plugin-induced differences. Use measured native-control distributions to
justify any later BF16 tolerance and separately assess task quality. The driver
`passed` flag means collection plus shutdown completed, not numerical acceptance.
