#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_command curl
require_command claude

base_url=${ANTHROPIC_BASE_URL:-http://$SERVER_HOST:$SERVER_PORT}
curl -sf --max-time 3 "$base_url/health" >/dev/null \
  || die "llama-server is not healthy at $base_url; run scripts/serve-q4.sh"

export ANTHROPIC_BASE_URL=$base_url
export ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN:-dummy}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-dummy}
export ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-$MODEL_ALIAS}
export ANTHROPIC_SMALL_FAST_MODEL=${ANTHROPIC_SMALL_FAST_MODEL:-$MODEL_ALIAS}
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}

exec claude "$@"
