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
  --split-mode layer
  --tensor-split 1,1,1,1
  --fit on
  --fit-target 2048
  --flash-attn on
  --jinja
  --temp 1.0
  --top-p 0.95
  --min-p 0.01
  --slot-save-path "$MODEL_DIR/slot-cache"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
)

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
