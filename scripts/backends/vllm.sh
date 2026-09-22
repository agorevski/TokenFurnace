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
# Preserve the runtime default when unset; zero must explicitly disable APC.
# Hybrid Mamba/GatedDeltaNet models also need a supported cache mode.
if [[ "${ENABLE_PREFIX_CACHING:-}" == 0 ]]; then
  args+=(--no-enable-prefix-caching)
elif [[ -n "${ENABLE_PREFIX_CACHING:-}" ]]; then
  args+=(--enable-prefix-caching)
  if [[ -n "${MAMBA_CACHE_MODE:-}" ]]; then
    args+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
  fi
  if [[ -n "${PREFIX_CACHING_HASH_ALGO:-}" ]]; then
    args+=(--prefix-caching-hash-algo "$PREFIX_CACHING_HASH_ALGO")
  fi
fi
if [[ -n "${BLOCK_SIZE:-}" ]]; then
  args+=(--block-size "$BLOCK_SIZE")
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  args+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" != 0 ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" != 0 ]]; then
  args+=(--language-model-only)
fi
if [[ "${ENABLE_EXPERT_PARALLEL:-0}" != 0 ]]; then
  args+=(--enable-expert-parallel)
fi
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi
if [[ -n "${GENERATION_CONFIG:-}" ]]; then
  args+=(--generation-config "$GENERATION_CONFIG")
fi
if [[ -n "${OVERRIDE_GENERATION_CONFIG:-}" ]]; then
  args+=(--override-generation-config "$OVERRIDE_GENERATION_CONFIG")
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
fi
if [[ "${ENFORCE_EAGER:-0}" != 0 ]]; then
  args+=(--enforce-eager)
fi

exec "$vllm" "${args[@]}" "$@"
