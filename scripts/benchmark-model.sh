#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

case "${1:-}" in
  -h|--help|"")
    printf 'Usage: benchmark-model.sh TARGET [PROFILE] [-- benchmark-api.py arguments]\n'
    exit 0
    ;;
esac

target=$1
profile=${2:-}
shift
[[ -z "$profile" ]] || shift
[[ "${1:-}" != -- ]] || shift
load_target "$target" "$profile"

exec python3 "$SCRIPT_DIR/benchmark-api.py" \
  --base-url "http://$SERVER_HOST:$SERVER_PORT" \
  --model "$MODEL_ALIAS" "$@"
