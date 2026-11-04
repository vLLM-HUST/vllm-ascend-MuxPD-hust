# Experimental asynchronous decode queue

`async_decode_pipeline` is disabled by default. It restores the native
asynchronous queue for transactions containing decode requests alone, while
prefill continues through the existing independent P runner. The default
synchronous scheduler, worker and Manager deployment profile are unchanged.

## Start an ordinary API server

Run this from the repository root on the existing server. It uses the current
checkout and the existing inference environment without replacing native
dependencies. Confirm that the selected logical devices are available before
starting; logical IDs are not physical card IDs.

```bash
source /root/workspace/runtime/env.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export HCCL_DETERMINISTIC=true
export VLLM_BATCH_INVARIANT=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1

/root/workspace/.muxpd-ecpa-runtime/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /root/workspace/models/Qwen3.5-35B-A3B \
  --served-model-name Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 8002 \
  --api-server-count 1 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 --language-model-only \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --max-num-seqs 8 --gpu-memory-utilization 0.85 \
  --no-enable-chunked-prefill --async-scheduling \
  --no-enable-prefix-caching --mamba-cache-mode none \
  --no-numa-bind --seed 42 --stream-interval 1 --disable-log-stats \
  --scheduler-cls vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker \
  --batch-admission-policy vllm_muxpd.policy.MuxPDPolicy \
  --batch-admission-policy-config \
  '{"execution":"decoupled","async_decode_pipeline":true,"stream_mode":"overlap","overlap_communication":"isolated","decode_graph":true,"compile_prefill_groups":true,"layers_per_slice":4,"adaptive_layers":false,"prefill_first":false,"compile_full_prefill":false,"prefill_layer_token_budget":0,"decode_budget_guard":false,"require_hccl_deterministic":true}' \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --additional-config \
  '{"enable_cpu_binding":false,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}'
```

This is a direct OpenAI API server launch using the explicit experimental
classes. It does not use `vllm-muxpd serve` or Manager projection. Existing
Manager-generated commands contain `--no-async-scheduling` and select the
synchronous pair; those commands do not enable this experiment. Keep the
literal HCCL setting and the policy requirement together. They validate an
environment profile and do not guarantee identical model answers.

After the server reports readiness, an ordinary streaming request is:

```bash
curl -N http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Explain tensor parallelism in one sentence."}],"temperature":0,"max_tokens":64,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

The recipe uses the full prompt budget, fixed four-layer P groups, independent
D graphs and isolated P communication groups. Prefix caching and Mamba aligned
prefill splitting are disabled. Grouped P still requires the fusion backend
with npugraph-ex and static kernels disabled. Adaptive quotas, prefill-first,
full-prefill compilation and the decode budget guard are excluded. The existing
text-only, BF16, Qwen3.5 and native-source compatibility restrictions apply.

## Queue and lifecycle boundaries

Only pure decode transactions can have two batches in flight. Any waiting or
skipped request stops queue filling, even when that request cannot yet acquire
KV. Prefill admission, continuation and cancellation run after all queued
decode outputs retire, with one such transaction in flight. A persistently
nonempty waiting queue can therefore remove most of the pipeline opportunity;
the change does not establish a capacity benefit for saturated arrivals.

A zero-token P continuation or cancellation still returns its receipt from the
execute RPC, because the native queued engine does not issue a separate sample
RPC for that step. D uses native asynchronous sampling, its private output-copy
stream and device token feedback. Its deferred output owns a transaction-local
request order and immutable empty-prefill receipt. P sampling remains
synchronous; mixed steps materialize D before returning the combined receipt.

D placeholders extend admission snapshots but are not added for pending P.
Explicit full histories cover both the first D after P and a preempted D
returning to P. Aborted or preempted D requests retain their KV blocks until
their final queued output retires. Reusing an ID with an outstanding D output,
keep-pause with queued work and resetting running requests before a drain are
rejected. Required caller/decode stream dependencies remain in place.

## Evidence and interpretation

The scheduling campaign's configurations, repeated comparisons, measurements
and raw receipts belong in the
[canonical benchmark report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8b5a7d57c94bb34b1756ee7c867dbe72094bcd9c/reports/muxpd-scheduling-20261026/).
The report includes repeated TP4 performance comparisons, hardware lifecycle
checks, and verified public raw evidence. Use its source and configuration
identities. Earlier source98 measurements and historical HTTP Goodput results do
not describe this asynchronous candidate.

Reducing the worst streaming pause does not imply that the usual tail improves.
P99 inter-token gaps and first-token waiting time must be assessed alongside
throughput and maximum pauses. P still waits for queue draining and complete
layer groups, so its first-token timing can worsen. The finite offline actor
comparison does not establish HTTP Goodput, sustained capacity, arbitrary
arrival-pattern gains or physical device overlap.

The native-relative answer regressions tracked by
[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)
remain unresolved. Faster scheduling and selected matching token sequences
do not constitute a task-quality pass.

CPU tests cover FIFO retirement, placeholders, history handoffs, native queued
engine dispatch, cancellation, preemption and block ownership. These tests
cannot replace hardware and answer-quality validation. Manager qualification
is also outstanding: its manifest and deployment profile are unchanged, and
pure-D deferred outputs do not yet emit the worker runtime-effective event
required by that qualification. This change does not release a newly qualified package.
