#!/usr/bin/env bash
# Opt-in quality-mitigation preset: the MuxWise-aligned preset plus exact per-row
# decode attention and one prompt per prefill forward. Each request then computes
# what it computes alone, independent of co-batched requests. Large throughput and
# latency cost; see docs/decode-attention-layout.md. Not a default or performance preset.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$script_dir/serve-qwen35.sh" \
  --async-scheduling \
  --scheduler-cls vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker \
  --no-enforce-eager --no-numa-bind \
  --additional-config '{"enable_cpu_binding":false}' \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --batch-admission-policy-config \
  '{"execution":"decoupled","async_decode_pipeline":true,"stream_mode":"overlap","overlap_communication":"isolated","decode_graph":true,"layers_per_slice":4,"decode_attention_layout":"row","max_prefill_requests":1}' \
  "$@"
