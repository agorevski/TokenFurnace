#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

case "${1:-}" in
  -h|--help|"")
    cat <<'USAGE'
Usage: benchmark-native.sh TARGET [PROFILE] [-- extra benchmark arguments]

Runs a repeatable native backend sweep: 512- and 4096-token prompt processing
plus 128-token generation, with five repetitions and JSON output.
USAGE
    exit 0
    ;;
esac

target=$1
shift
profile=
if (($#)) && [[ "$1" != -- && "$1" != -* ]]; then
  profile=$1
  shift
fi
[[ "${1:-}" != -- ]] || shift
load_target "$target" "$profile"

case "$BACKEND" in
  llama.cpp)
    benchmark=$BUILD_DIR/bin/llama-bench
    require_file "$benchmark"
    require_file "$MODEL"
    add_runtime_libraries
    bench_tensor_split=${TENSOR_SPLIT:-1,1,1,1}
    # llama-server uses comma-separated fractions, while llama-bench uses slashes;
    # commas select multiple benchmark variants and silently duplicate the sweep.
    bench_tensor_split=${bench_tensor_split//,/\/}

    if [[ -n "${CUDA_SCALE_LAUNCH_QUEUES+x}" ]]; then
      if [[ -n "$CUDA_SCALE_LAUNCH_QUEUES" ]]; then
        export CUDA_SCALE_LAUNCH_QUEUES
      else
        unset CUDA_SCALE_LAUNCH_QUEUES
      fi
    fi

    args=(
      -m "$MODEL"
      -ngl 99
      -sm "${SPLIT_MODE:-layer}"
      -ts "$bench_tensor_split"
      -fa "${FLASH_ATTN:-on}"
      -b "${BATCH_SIZE:-2048}"
      -ub "${UBATCH_SIZE:-512}"
      -r 5
      -p 0
      -n 0
      -pg 512,0
      -pg 4096,0
      -pg 0,128
      -o json
    )
    ;;
  ninfer)
    benchmark=$NINFER_BUILD_DIR/bench/ninfer_bench
    require_file "$benchmark"
    require_file "$MODEL"
    [[ "${KV_DTYPE:-int8}" == int8 ]] ||
      die "this patched SM75 NInfer build supports only KV_DTYPE=int8"
    export LD_LIBRARY_PATH="$NINFER_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    args=(
      --weights "$MODEL"
      --device "${DEVICE:-0}"
      --kv-dtype "${KV_DTYPE:-int8}"
      --max-ctx "$MAX_CONTEXT"
      --prefill-chunk "${PREFILL_CHUNK:-1024}"
      --n-prompt 512,4096
      --n-gen 128
      --repetitions 5
      --warmup 1
      --output json
    )
    if [[ -n "${SPEC_BACKEND:-}" ]]; then
      [[ "$SPEC_BACKEND" == mtp ]] ||
        die "NInfer native benchmark supports only MTP speculation"
      args+=(--mtp-draft-tokens "$DRAFT_TOKENS")
      if [[ "${LM_HEAD_DRAFT:-0}" != 0 ]]; then
        args+=(--lm-head-draft)
      fi
    fi
    ;;
  *)
    die "native benchmark does not support backend: $BACKEND"
    ;;
esac

exec "$benchmark" "${args[@]}" "$@"
