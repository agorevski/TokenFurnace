#!/usr/bin/env bash

set -euo pipefail

server=$BUILD_DIR/bin/llama-server
require_file "$server"
require_file "$MODEL"
add_runtime_libraries

export CUDA_SCALE_LAUNCH_QUEUES=${CUDA_SCALE_LAUNCH_QUEUES:-4x}

args=(
  --model "$MODEL"
  --alias "$MODEL_ALIAS"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --n-gpu-layers 99
  --split-mode "${SPLIT_MODE:-layer}"
  --tensor-split "${TENSOR_SPLIT:-1,1,1,1}"
  --fit on
  --fit-target 2048
  --flash-attn "${FLASH_ATTN:-on}"
  --jinja
  --temp 1.0
  --top-p 0.95
  --min-p 0.01
  --slot-save-path "$MODEL_DIR/slot-cache"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
)

if [[ -n "${CACHE_TYPE_K:-}" ]]; then
  args+=(--cache-type-k "$CACHE_TYPE_K")
fi
if [[ -n "${CACHE_TYPE_V:-}" ]]; then
  args+=(--cache-type-v "$CACHE_TYPE_V")
fi

# Explicit prompt/prefix caching knobs (b10298). llama-server already defaults
# to --cache-prompt on, --cache-ram 8192 MiB, and --cache-idle-slots on, so
# these are only appended when a profile opts in, keeping existing targets
# (e.g. DeepSeek) on the stock defaults. Prefix matching is exact: any change to
# the system prompt, tool schemas, timestamps, or message ordering invalidates
# the cached prefix.
if [[ "${CACHE_PROMPT:-1}" == 0 ]]; then
  args+=(--no-cache-prompt)
fi
if [[ -n "${CACHE_RAM_MIB:-}" ]]; then
  args+=(--cache-ram "$CACHE_RAM_MIB")
fi
if [[ "${CACHE_IDLE_SLOTS:-1}" == 0 ]]; then
  args+=(--no-cache-idle-slots)
fi
if [[ -n "${CACHE_REUSE:-}" ]]; then
  args+=(--cache-reuse "$CACHE_REUSE")
fi
if [[ -n "${SLOT_PROMPT_SIMILARITY:-}" ]]; then
  args+=(--slot-prompt-similarity "$SLOT_PROMPT_SIMILARITY")
fi
if [[ "${ENABLE_METRICS:-0}" != 0 ]]; then
  args+=(--metrics)
fi

if [[ "${DSPARK:-0}" != 0 ]]; then
  require_file "$DRAFT_MODEL"
  args+=(
    --spec-draft-model "$DRAFT_MODEL"
    --spec-type draft-dspark
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
    --spec-draft-ngl 99
  )
fi

mkdir -p "$MODEL_DIR/slot-cache"
exec "$server" "${args[@]}" "$@"
