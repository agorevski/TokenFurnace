#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

if [[ -f "$PROJECT_DIR/config.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/config.env"
  set +a
fi

MODEL_DIR=${MODEL_DIR:-/home/algore/models/dsv4-0731}
LLAMA_DIR=${LLAMA_DIR:-/home/algore/llama.cpp-dspark-current}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-peer2048}
MODEL_ALIAS=${MODEL_ALIAS:-deepseek-v4-flash}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8090}

Q4_MODEL=${Q4_MODEL:-$MODEL_DIR/UD-IQ4_XS/DeepSeek-V4-Flash-0731-UD-IQ4_XS-00001-of-00004.gguf}
Q8_MODEL=${Q8_MODEL:-$MODEL_DIR/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}
IQ3_MODEL=${IQ3_MODEL:-$MODEL_DIR/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf}
DRAFT_MODEL=${DRAFT_MODEL:-$MODEL_DIR/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

add_runtime_libraries() {
  local paths=()
  local nccl_dir

  if [[ -d /home/algore/miniconda3/lib ]]; then
    paths+=(/home/algore/miniconda3/lib)
  fi

  nccl_dir=$(find /home/algore/miniconda3/lib/python*/site-packages/nvidia/nccl/lib \
    -maxdepth 0 -type d -print -quit 2>/dev/null || true)
  if [[ -n "$nccl_dir" ]]; then
    paths=("$nccl_dir" "${paths[@]}")
  fi

  if ((${#paths[@]})); then
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$(IFS=:; printf '%s' "${paths[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  fi
}

health_url() {
  printf 'http://%s:%s/health' "$SERVER_HOST" "$SERVER_PORT"
}
