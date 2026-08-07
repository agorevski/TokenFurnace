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
  serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp4
EOF
}

case "${1:-}" in
  -h|--help|"") usage; exit 0 ;;
esac

target=$1
profile=${2:-}
shift
[[ -z "$profile" ]] || shift
[[ "${1:-}" != -- ]] || shift

load_target "$target" "$profile"
printf 'Serving target=%s profile=%s backend=%s on http://%s:%s\n' \
  "$TARGET" "$PROFILE" "$BACKEND" "$SERVER_HOST" "$SERVER_PORT"

case "$BACKEND" in
  llama.cpp) source "$SCRIPT_DIR/backends/llama-cpp.sh" ;;
  vllm) source "$SCRIPT_DIR/backends/vllm.sh" ;;
  *) die "unsupported backend: $BACKEND" ;;
esac
