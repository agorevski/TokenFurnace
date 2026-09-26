#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

case "${1:-}" in
  -h|--help|"")
    cat <<'USAGE'
Usage: benchmark-config.sh TARGET [PROFILE]

Validates a target profile against its selected benchmark baseline and prints
the resolved comparison manifest in shell-compatible key=value form.
USAGE
    exit 0
    ;;
esac

target=$1
profile=${2:-}
load_target "$target" "$profile"
load_benchmark_baseline
validate_benchmark_profile

case "$BACKEND" in
  llama.cpp)
    resolved_context=$CTX_SIZE
    resolved_batch=$BATCH_SIZE
    resolved_ubatch=$UBATCH_SIZE
    ;;
  vllm)
    resolved_context=$MAX_MODEL_LEN
    resolved_batch=$MAX_NUM_BATCHED_TOKENS
    resolved_ubatch=unmapped
    ;;
  ninfer)
    resolved_context=$MAX_CONTEXT
    resolved_batch=unmapped
    resolved_ubatch=unmapped
    ;;
esac

baseline_sha256=$(sha256sum "$BENCHMARK_BASELINE")
baseline_sha256=${baseline_sha256%% *}
prompt_sha256=$(sha256sum "$BENCHMARK_API_PROMPT_FILE")
prompt_sha256=${prompt_sha256%% *}

print_value() {
  printf '%s=%q\n' "$1" "$2"
}

print_value TARGET "$TARGET"
print_value PROFILE "$PROFILE"
print_value BACKEND "$BACKEND"
print_value HF_REPO "${HF_REPO:-}"
print_value MODEL "${MODEL:-}"
print_value BENCHMARK_BASELINE "$BENCHMARK_BASELINE"
print_value BENCHMARK_BASELINE_SHA256 "$baseline_sha256"
print_value RESOLVED_CONTEXT_SIZE "$resolved_context"
print_value RESOLVED_BATCH_SIZE "$resolved_batch"
print_value RESOLVED_UBATCH_SIZE "$resolved_ubatch"
print_value BENCHMARK_PREFILL_TOKENS_SHORT "$BENCHMARK_PREFILL_TOKENS_SHORT"
print_value BENCHMARK_PREFILL_TOKENS_LONG "$BENCHMARK_PREFILL_TOKENS_LONG"
print_value BENCHMARK_OUTPUT_TOKENS "$BENCHMARK_OUTPUT_TOKENS"
print_value BENCHMARK_REPETITIONS "$BENCHMARK_REPETITIONS"
print_value BENCHMARK_API_PROMPT_FILE "$BENCHMARK_API_PROMPT_FILE"
print_value BENCHMARK_API_PROMPT_SHA256 "$prompt_sha256"
print_value BENCHMARK_API_PROMPT_TOKENS "$BENCHMARK_API_PROMPT_TOKENS"
print_value BENCHMARK_API_OUTPUT_TOKENS "$BENCHMARK_API_OUTPUT_TOKENS"
print_value BENCHMARK_API_PREFILL_OUTPUT_TOKENS \
  "$BENCHMARK_API_PREFILL_OUTPUT_TOKENS"
print_value BENCHMARK_API_CONCURRENCY "$BENCHMARK_API_CONCURRENCY"
print_value BENCHMARK_API_REQUESTS "$BENCHMARK_API_REQUESTS"
print_value BENCHMARK_TEMPERATURE "$BENCHMARK_TEMPERATURE"
print_value BENCHMARK_CACHE_STATE "$BENCHMARK_CACHE_STATE"
