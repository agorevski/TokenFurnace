#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

case "${1:-}" in
  -h|--help|"")
    cat <<'USAGE'
Usage: benchmark-model.sh TARGET [PROFILE] [-- benchmark-api.py arguments]

Examples:
  # Single-request decode latency (default)
  benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
  # Prefill-oriented run: use a long prompt and generate one token
  benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km -- --prompt-file PROMPT.txt --max-tokens 1
  # Aggregate serving throughput under concurrency
  benchmark-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2 -- --concurrency 16 --requests 64
USAGE
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
