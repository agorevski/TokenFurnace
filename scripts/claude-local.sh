#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

target=${1:-deepseek-v4-flash-0731}
profile=${2:-}
if (($#)); then shift; fi
if (($#)); then shift; fi
load_target "$target" "$profile"
[[ "$BACKEND" == llama.cpp ]] \
  || die "direct Anthropic compatibility is currently configured only for llama.cpp targets"

require_command curl
require_command claude

base_url=${ANTHROPIC_BASE_URL:-http://$SERVER_HOST:$SERVER_PORT}
curl -sf --max-time 3 "$base_url/health" >/dev/null \
  || die "model server is not healthy at $base_url"

export ANTHROPIC_BASE_URL=$base_url
export ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN:-dummy}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-dummy}
export ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-$MODEL_ALIAS}
export ANTHROPIC_SMALL_FAST_MODEL=${ANTHROPIC_SMALL_FAST_MODEL:-$MODEL_ALIAS}
# CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 suppresses background/telemetry
# requests. With PARALLEL=1 the llama.cpp server has a single KV slot, so an
# unrelated request between coding turns can evict the cached system-prompt
# prefix; disabling that traffic keeps the prefix cache warm for the session.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}

exec claude "$@"
