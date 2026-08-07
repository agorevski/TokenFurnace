#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_command curl
require_command copilot

provider_url=${COPILOT_PROVIDER_BASE_URL:-http://$SERVER_HOST:$SERVER_PORT/v1}
curl -sf --max-time 3 "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null \
  || die "llama-server is not healthy at http://$SERVER_HOST:$SERVER_PORT; run scripts/serve-q4.sh"

export COPILOT_ENABLE_ALT_PROVIDERS=true
export COPILOT_PROVIDER_BASE_URL=$provider_url
export COPILOT_PROVIDER_TYPE=${COPILOT_PROVIDER_TYPE:-openai}
export COPILOT_PROVIDER_API_KEY=${COPILOT_PROVIDER_API_KEY:-dummy}
export COPILOT_PROVIDER_WIRE_API=${COPILOT_PROVIDER_WIRE_API:-completions}
export COPILOT_MODEL=${COPILOT_MODEL:-$MODEL_ALIAS}
export COPILOT_PROVIDER_MODEL_ID=${COPILOT_PROVIDER_MODEL_ID:-gpt-4.1}
export COPILOT_PROVIDER_WIRE_MODEL=${COPILOT_PROVIDER_WIRE_MODEL:-$MODEL_ALIAS}
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS=${COPILOT_PROVIDER_MAX_PROMPT_TOKENS:-65536}
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=${COPILOT_PROVIDER_MAX_OUTPUT_TOKENS:-8192}
export COPILOT_OFFLINE=${COPILOT_OFFLINE:-true}

exec copilot "$@"
