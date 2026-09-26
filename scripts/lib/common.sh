#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

if [[ -f "$PROJECT_DIR/config.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/config.env"
  set +a
fi

HARDWARE_PROFILE=${HARDWARE_PROFILE:-4x-rtx8000}
SERVER_HOST=${SERVER_HOST:-0.0.0.0}

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

load_env_file() {
  local path=$1
  require_file "$path"
  set -a
  # shellcheck disable=SC1090
  source "$path"
  set +a
}

load_hardware() {
  load_env_file "$PROJECT_DIR/hardware/$HARDWARE_PROFILE/profile.env"
}

load_target() {
  local target=$1
  local profile=${2:-}
  local target_dir=$PROJECT_DIR/targets/$target

  load_hardware
  load_env_file "$target_dir/target.env"
  profile=${profile:-$DEFAULT_PROFILE}
  load_env_file "$target_dir/profiles/$profile.env"

  TARGET=$target
  PROFILE=$profile
  TARGET_DIR=$target_dir
  export TARGET PROFILE TARGET_DIR
}

require_positive_integer() {
  local name=$1
  local value=${!name:-}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
    die "$name must be a positive integer (got: ${value:-unset})"
}

load_benchmark_baseline() {
  [[ -n "${TARGET_DIR:-}" ]] ||
    die "load_target must be called before load_benchmark_baseline"

  BENCHMARK_BASELINE=${BASELINE:-$TARGET_DIR/benchmark.env}
  load_env_file "$BENCHMARK_BASELINE"

  local name
  for name in \
    BENCHMARK_CONTEXT_SIZE \
    BENCHMARK_BATCH_SIZE \
    BENCHMARK_UBATCH_SIZE \
    BENCHMARK_PREFILL_TOKENS_SHORT \
    BENCHMARK_PREFILL_TOKENS_LONG \
    BENCHMARK_OUTPUT_TOKENS \
    BENCHMARK_REPETITIONS \
    BENCHMARK_API_PROMPT_TOKENS \
    BENCHMARK_API_OUTPUT_TOKENS \
    BENCHMARK_API_PREFILL_OUTPUT_TOKENS \
    BENCHMARK_API_CONCURRENCY \
    BENCHMARK_API_REQUESTS; do
    require_positive_integer "$name"
  done

  : "${BENCHMARK_API_PROMPT_FILE:?BENCHMARK_API_PROMPT_FILE is required}"
  : "${BENCHMARK_CACHE_STATE:?BENCHMARK_CACHE_STATE is required}"
  case "$BENCHMARK_CACHE_STATE" in
    cold|warm) ;;
    *) die "BENCHMARK_CACHE_STATE must be cold or warm" ;;
  esac

  if [[ "$BENCHMARK_API_PROMPT_FILE" != /* ]]; then
    BENCHMARK_API_PROMPT_FILE=$PROJECT_DIR/$BENCHMARK_API_PROMPT_FILE
  fi
  require_file "$BENCHMARK_API_PROMPT_FILE"

  if [[ -n "${BENCHMARK_API_PROMPT_SHA256:-}" ]]; then
    local actual_sha256
    actual_sha256=$(sha256sum "$BENCHMARK_API_PROMPT_FILE")
    actual_sha256=${actual_sha256%% *}
    [[ "$actual_sha256" == "$BENCHMARK_API_PROMPT_SHA256" ]] ||
      die "benchmark prompt checksum mismatch: expected $BENCHMARK_API_PROMPT_SHA256, got $actual_sha256"
  fi

  BENCHMARK_TEMPERATURE=${BENCHMARK_TEMPERATURE:-0}
  export BENCHMARK_BASELINE BENCHMARK_API_PROMPT_FILE BENCHMARK_TEMPERATURE
}

require_benchmark_value() {
  local label=$1
  local actual=${2:-}
  local expected=$3
  [[ -n "$actual" ]] || die "$label is not configured for backend $BACKEND"
  [[ "$actual" == "$expected" ]] ||
    die "$label=$actual does not match baseline value $expected"
}

validate_benchmark_profile() {
  case "$BACKEND" in
    llama.cpp)
      require_benchmark_value CTX_SIZE "${CTX_SIZE:-}" "$BENCHMARK_CONTEXT_SIZE"
      require_benchmark_value BATCH_SIZE "${BATCH_SIZE:-}" "$BENCHMARK_BATCH_SIZE"
      require_benchmark_value UBATCH_SIZE "${UBATCH_SIZE:-}" "$BENCHMARK_UBATCH_SIZE"
      if [[ "$BENCHMARK_CACHE_STATE" == cold ]]; then
        require_benchmark_value CACHE_PROMPT "${CACHE_PROMPT:-}" 0
      else
        require_benchmark_value CACHE_PROMPT "${CACHE_PROMPT:-}" 1
      fi
      ;;
    vllm)
      require_benchmark_value MAX_MODEL_LEN "${MAX_MODEL_LEN:-}" "$BENCHMARK_CONTEXT_SIZE"
      require_benchmark_value MAX_NUM_BATCHED_TOKENS \
        "${MAX_NUM_BATCHED_TOKENS:-}" "$BENCHMARK_BATCH_SIZE"
      if [[ "$BENCHMARK_CACHE_STATE" == cold ]]; then
        require_benchmark_value ENABLE_PREFIX_CACHING \
          "${ENABLE_PREFIX_CACHING:-}" 0
      else
        require_benchmark_value ENABLE_PREFIX_CACHING \
          "${ENABLE_PREFIX_CACHING:-}" 1
      fi
      printf 'warning: vLLM has no direct UBATCH_SIZE equivalent; microbatch is unmapped\n' >&2
      ;;
    ninfer)
      require_benchmark_value MAX_CONTEXT "${MAX_CONTEXT:-}" "$BENCHMARK_CONTEXT_SIZE"
      printf 'warning: NInfer batch and microbatch controls are unmapped\n' >&2
      ;;
    *)
      die "benchmark baseline validation does not support backend: $BACKEND"
      ;;
  esac
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
