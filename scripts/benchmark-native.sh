#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

case "${1:-}" in
  -h|--help|"")
    cat <<'USAGE'
Usage: benchmark-native.sh TARGET [PROFILE] [-- extra llama-bench arguments]

Runs a repeatable llama.cpp kernel sweep: 512- and 4096-token prompt
processing plus 128-token generation, with five repetitions and JSON output.
The selected profile must use the llama.cpp backend.
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

[[ "$BACKEND" == llama.cpp ]] ||
  die "native benchmark requires a llama.cpp profile (got $BACKEND)"
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

exec "$benchmark" "${args[@]}" "$@"
