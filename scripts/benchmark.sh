#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: benchmark.sh [q4|q8|iq3]

Environment:
  PP_TOKENS     Prompt tokens, default 8192
  TG_TOKENS     Generated tokens, default 128
  REPETITIONS   Repetitions, default 2
  BATCH_SIZE    Logical batch, profile default
  UBATCH_SIZE   Physical batch, profile default
EOF
}

case "${1:-q4}" in
  -h|--help) usage; exit 0 ;;
  q4)
    model=$Q4_MODEL
    batch_size=${BATCH_SIZE:-8192}
    ubatch_size=${UBATCH_SIZE:-2048}
    ;;
  q8)
    model=$Q8_MODEL
    batch_size=${BATCH_SIZE:-2048}
    ubatch_size=${UBATCH_SIZE:-512}
    ;;
  iq3)
    model=$IQ3_MODEL
    batch_size=${BATCH_SIZE:-8192}
    ubatch_size=${UBATCH_SIZE:-2048}
    ;;
  *) usage >&2; exit 2 ;;
esac

bench=$BUILD_DIR/bin/llama-bench
require_file "$bench"
require_file "$model"
add_runtime_libraries
export CUDA_SCALE_LAUNCH_QUEUES=${CUDA_SCALE_LAUNCH_QUEUES:-4x}

exec "$bench" \
  --model "$model" \
  --n-prompt "${PP_TOKENS:-8192}" \
  --n-gen "${TG_TOKENS:-128}" \
  --repetitions "${REPETITIONS:-2}" \
  --n-gpu-layers 99 \
  --flash-attn on \
  --batch-size "$batch_size" \
  --ubatch-size "$ubatch_size" \
  --split-mode layer \
  --tensor-split 1/1/1/1 \
  --fit-target 2048 \
  --output md \
  --progress
