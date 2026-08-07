#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  printf 'Usage: download-model.sh TARGET [PROFILE]\n'
}

case "${1:-}" in
  -h|--help|"") usage; exit 0 ;;
esac

load_target "$1" "${2:-}"
require_command hf
mkdir -p "$MODEL_DIR"

args=(download "$HF_REPO" --local-dir "$MODEL_DIR")
if [[ -n "${DOWNLOAD_PATTERNS:-}" ]]; then
  IFS='|' read -r -a patterns <<< "$DOWNLOAD_PATTERNS"
  for pattern in "${patterns[@]}"; do
    args+=(--include "$pattern")
  done
fi

printf 'Downloading %s for target=%s profile=%s to %s\n' \
  "$HF_REPO" "$TARGET" "$PROFILE" "$MODEL_DIR"
df -h "$MODEL_DIR" | tail -1
exec hf "${args[@]}"
