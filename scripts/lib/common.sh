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
SERVER_HOST=${SERVER_HOST:-127.0.0.1}

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
