#!/usr/bin/env bash
# Opt-in interactive-latency preset; see docs/performance-tp4-goodput.md.
# Raw prefill in four-layer slices, with independent captured decode batches.
# Select devices, CPU affinity and cache directories for your deployment.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MUXPD_TP_SIZE="${MUXPD_TP_SIZE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V2_MODEL_RUNNER=0
# The base supplies synchronous scheduling, max sequences 8, and full 16K budget.
# User arguments remain last and can override any command-line preset.
exec "$script_dir/serve-qwen35.sh" \
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker \
  --data-parallel-size 1 --pipeline-parallel-size 1 --api-server-count 1 \
  --no-enforce-eager --no-numa-bind \
  --additional-config '{"enable_cpu_binding":false}' \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"overlap","layers_per_slice":4,"adaptive_layers":false,"prefill_first":false,"decode_graph":true,"compile_full_prefill":false,"prefill_layer_token_budget":0,"decode_budget_guard":false,"overlap_communication":"isolated"}' \
  "$@"
