#!/usr/bin/env bash
# Independent full-prompt prefill and captured decode, tuned on pinned TP4.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$script_dir/serve-qwen35.sh" \
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker \
  --no-enforce-eager \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"fenced","layers_per_slice":8,"adaptive_layers":true,"prefill_first":true,"decode_graph":true,"compile_full_prefill":true,"prefill_layer_token_budget":262144}' \
  "$@"
