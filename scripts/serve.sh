#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: serve.sh [options] [-- extra llama-server arguments]

Options:
  --profile q4|q8|iq3   Model profile, default q4
  --host HOST           Bind host, default 127.0.0.1
  --port PORT           Bind port, default 8090
  --ctx-size N          Total context, default 65536
  --parallel N          Server slots, default 1
  --no-dspark           Disable speculative decoding
  -h, --help            Show help

Environment overrides:
  MODEL, DRAFT_MODEL, BATCH_SIZE, UBATCH_SIZE, SPEC_DRAFT_N_MAX
EOF
}

profile=${PROFILE:-q4}
ctx_size=${CTX_SIZE:-65536}
parallel=${PARALLEL:-1}
dspark=${DSPARK:-1}
extra=()

while (($#)); do
  case "$1" in
    --profile) profile=$2; shift 2 ;;
    --host) SERVER_HOST=$2; shift 2 ;;
    --port) SERVER_PORT=$2; shift 2 ;;
    --ctx-size) ctx_size=$2; shift 2 ;;
    --parallel) parallel=$2; shift 2 ;;
    --no-dspark) dspark=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

case "$profile" in
  q4)
    model=${MODEL:-$Q4_MODEL}
    batch_size=${BATCH_SIZE:-${Q4_BATCH_SIZE:-8192}}
    ubatch_size=${UBATCH_SIZE:-${Q4_UBATCH_SIZE:-2048}}
    ;;
  q8)
    model=${MODEL:-$Q8_MODEL}
    batch_size=${BATCH_SIZE:-${Q8_BATCH_SIZE:-2048}}
    ubatch_size=${UBATCH_SIZE:-${Q8_UBATCH_SIZE:-512}}
    ;;
  iq3)
    model=${MODEL:-$IQ3_MODEL}
    batch_size=${BATCH_SIZE:-8192}
    ubatch_size=${UBATCH_SIZE:-2048}
    ;;
  *) die "unknown profile: $profile" ;;
esac

server=$BUILD_DIR/bin/llama-server
require_file "$server"
require_file "$model"
add_runtime_libraries

export CUDA_SCALE_LAUNCH_QUEUES=${CUDA_SCALE_LAUNCH_QUEUES:-4x}

args=(
  --model "$model"
  --alias "$MODEL_ALIAS"
  --ctx-size "$ctx_size"
  --parallel "$parallel"
  --batch-size "$batch_size"
  --ubatch-size "$ubatch_size"
  --n-gpu-layers 99
  --split-mode layer
  --tensor-split 1,1,1,1
  --fit on
  --fit-target 2048
  --flash-attn on
  --jinja
  --temp 1.0
  --top-p 0.95
  --min-p 0.01
  --slot-save-path "$MODEL_DIR/slot-cache"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
)

if [[ "$dspark" != 0 ]]; then
  require_file "$DRAFT_MODEL"
  args+=(
    --spec-draft-model "$DRAFT_MODEL"
    --spec-type draft-dspark
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"
    --spec-draft-ngl 99
  )
fi

mkdir -p "$MODEL_DIR/slot-cache"
printf 'Serving profile=%s model=%s on http://%s:%s\n' "$profile" "$model" "$SERVER_HOST" "$SERVER_PORT"
exec "$server" "${args[@]}" "${extra[@]}"
