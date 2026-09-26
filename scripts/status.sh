#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

target=${1:-deepseek-v4-flash-0731}
profile=${2:-}
load_target "$target" "$profile"

printf '=== target: %s / %s ===\n' "$TARGET" "$PROFILE"
if curl -sf --max-time 3 "$(health_url)" >/dev/null; then
  printf 'healthy: http://%s:%s\n' "$SERVER_HOST" "$SERVER_PORT"
  if [[ "$BACKEND" == llama.cpp ]]; then
    printf '%s\n' '=== slots ==='
    curl -s --max-time 5 "http://$SERVER_HOST:$SERVER_PORT/slots" | python3 -m json.tool
  else
    printf '%s\n' '=== models ==='
    curl -s --max-time 5 "http://$SERVER_HOST:$SERVER_PORT/v1/models" | python3 -m json.tool
  fi
else
  printf 'not responding: http://%s:%s\n' "$SERVER_HOST" "$SERVER_PORT"
fi

printf '%s\n' '=== process ==='
pgrep -af 'llama-server|ninfer-serve|vllm serve|vllm.entrypoints' || true

printf '%s\n' '=== GPUs ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader
