#!/usr/bin/env bash

set -euo pipefail

server=$NINFER_BUILD_DIR/apps/ninfer-serve
require_file "$server"
require_file "$MODEL"
require_file "$NINFER_ENV/lib/libcudart.so"
[[ "${KV_DTYPE:-int8}" == int8 ]] ||
  die "this patched SM75 NInfer build supports only KV_DTYPE=int8"

export LD_LIBRARY_PATH="$NINFER_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

args=(
  "$MODEL"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --model-id "$MODEL_ALIAS"
  --max-context "$MAX_CONTEXT"
  --kv-capacity "${KV_CAPACITY:-auto}"
  --max-concurrency "${MAX_CONCURRENCY:-1}"
  --prefill-chunk "${PREFILL_CHUNK:-1024}"
  --device "${DEVICE:-0}"
  --kv-dtype "${KV_DTYPE:-int8}"
  --default-max-tokens "${DEFAULT_MAX_TOKENS:-4096}"
)

if [[ -n "${REQUEST_LOG_JSONL:-}" ]]; then
  mkdir -p "$(dirname -- "$REQUEST_LOG_JSONL")"
  args+=(--request-log-jsonl "$REQUEST_LOG_JSONL")
fi
if [[ "${NO_PREFIX_REUSE:-0}" != 0 ]]; then
  args+=(--no-prefix-reuse)
fi
if [[ "${NO_THINKING:-0}" != 0 ]]; then
  args+=(--no-thinking)
fi
if [[ "${GREEDY:-0}" != 0 ]]; then
  args+=(--greedy)
fi
if [[ -n "${SPEC_BACKEND:-}" ]]; then
  args+=(--spec "$SPEC_BACKEND" --draft-tokens "$DRAFT_TOKENS")
  if [[ "${LM_HEAD_DRAFT:-0}" != 0 ]]; then
    args+=(--lm-head-draft)
  fi
fi

exec "$server" "${args[@]}" "$@"
