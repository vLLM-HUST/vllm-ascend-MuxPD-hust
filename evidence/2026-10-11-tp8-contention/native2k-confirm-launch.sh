#!/usr/bin/env bash
set -eo pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /root/workspace/runtime/env.sh
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH="/root/workspace/vllm-ascend-MuxPD-hust/src:$PYTHONPATH"
exec /root/workspace/.venv/bin/python -m vllm.entrypoints.openai.api_server --model /root/workspace/models/Qwen3.5-35B-A3B --served-model-name Qwen3.5-35B-A3B --host 127.0.0.1 --port 8000 --tensor-parallel-size 8 --enable-expert-parallel --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85 --language-model-only --no-enable-prefix-caching --mamba-cache-mode none --no-async-scheduling --no-enforce-eager --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' --enable-chunked-prefill
