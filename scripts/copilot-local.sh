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

require_command curl
require_command copilot

provider_url=${COPILOT_PROVIDER_BASE_URL:-http://$SERVER_HOST:$SERVER_PORT/v1}
curl -sf --max-time 3 "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null \
  || die "model server is not healthy at http://$SERVER_HOST:$SERVER_PORT"

export COPILOT_ENABLE_ALT_PROVIDERS=true
export COPILOT_PROVIDER_BASE_URL=$provider_url
export COPILOT_PROVIDER_TYPE=${COPILOT_PROVIDER_TYPE:-openai}
export COPILOT_PROVIDER_API_KEY=${COPILOT_PROVIDER_API_KEY:-dummy}
# COPILOT_PROVIDER_WIRE_API accepts only "completions" or "responses" (verified
# against the installed CLI 1.0.79-6; an unsupported value is rejected as
# "Invalid wire API format"). "chat" is NOT a valid value, so it is left as
# "completions"; text completions still share byte-identical prompt prefixes for
# server-side prefix caching.
export COPILOT_PROVIDER_WIRE_API=${COPILOT_PROVIDER_WIRE_API:-completions}
export COPILOT_MODEL=${COPILOT_MODEL:-$MODEL_ALIAS}
export COPILOT_PROVIDER_MODEL_ID=${COPILOT_PROVIDER_MODEL_ID:-gpt-4.1}
export COPILOT_PROVIDER_WIRE_MODEL=${COPILOT_PROVIDER_WIRE_MODEL:-$MODEL_ALIAS}
# Derive the max prompt-token budget from the loaded context window instead of a
# hardcoded 65536: vLLM profiles expose MAX_MODEL_LEN (e.g. 32768) and llama.cpp
# profiles expose CTX_SIZE. A prompt budget larger than the served context would
# let the agent build prompts the server must truncate, which also breaks exact
# prefix caching. Reserve the output budget so prompt + output fits the window.
copilot_ctx=${MAX_MODEL_LEN:-${CTX_SIZE:-65536}}
copilot_out=${COPILOT_PROVIDER_MAX_OUTPUT_TOKENS:-8192}
copilot_prompt_default=$((copilot_ctx - copilot_out))
if ((copilot_prompt_default < 1024)); then
  copilot_prompt_default=$copilot_ctx
fi
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS=${COPILOT_PROVIDER_MAX_PROMPT_TOKENS:-$copilot_prompt_default}
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=$copilot_out
export COPILOT_OFFLINE=${COPILOT_OFFLINE:-true}

exec copilot "$@"
