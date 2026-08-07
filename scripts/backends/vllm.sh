#!/usr/bin/env bash

set -euo pipefail

vllm=$VLLM_ENV/bin/vllm
require_file "$vllm"
[[ -d "$MODEL" ]] || die "model directory not found: $MODEL"

args=(
  serve "$MODEL"
  --served-model-name "$MODEL_ALIAS"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ "${PIPELINE_PARALLEL_SIZE:-1}" != 1 ]]; then
  args+=(--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  args+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" != 0 ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ "${ENFORCE_EAGER:-0}" != 0 ]]; then
  args+=(--enforce-eager)
fi

exec "$vllm" "${args[@]}" "$@"
