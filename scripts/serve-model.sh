#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: serve-model.sh TARGET [PROFILE] [-- extra backend arguments]

Examples:
  serve-model.sh deepseek-v4-flash-0731 q4-balanced
  serve-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
  serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
  serve-model.sh qwen3.6-35b-a3b llama-cpp-q4km-mtp-1gpu
  serve-model.sh qwen3.8-27b llama-cpp-q4km-1gpu
EOF
}

case "${1:-}" in
  -h|--help|"") usage; exit 0 ;;
esac

target=$1
shift
profile=
if (($#)) && [[ "$1" != -- && "$1" != -* ]]; then
  profile=$1
  shift
fi
[[ "${1:-}" != -- ]] || shift

load_target "$target" "$profile"
printf 'Serving target=%s profile=%s backend=%s on http://%s:%s\n' \
  "$TARGET" "$PROFILE" "$BACKEND" "$SERVER_HOST" "$SERVER_PORT"

profile_file=$TARGET_DIR/profiles/$PROFILE.env
printf 'Profile data (%s):\n' "$profile_file"
while IFS= read -r line; do
  if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]]; then
    name=${BASH_REMATCH[1]}
    printf '  %s=%q\n' "$name" "${!name}"
  fi
done < "$profile_file"

case "$BACKEND" in
  llama.cpp) source "$SCRIPT_DIR/backends/llama-cpp.sh" ;;
  ninfer) source "$SCRIPT_DIR/backends/ninfer.sh" ;;
  vllm) source "$SCRIPT_DIR/backends/vllm.sh" ;;
  vllm-docker) source "$SCRIPT_DIR/backends/vllm-docker.sh" ;;
  *) die "unsupported backend: $BACKEND" ;;
esac
