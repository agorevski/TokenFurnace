#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: download-models.sh [q4|q8|iq3|draft|all]

q4     Download UD-IQ4_XS and the Q8_0 DSpark drafter (default)
q8     Download lossless UD-Q8_K_XL and the Q8_0 DSpark drafter
iq3    Download UD-IQ3_XXS and the Q8_0 DSpark drafter
draft  Download only the DSpark drafter
all    Download Q4, Q8, IQ3, and the DSpark drafter

Environment:
  MODEL_DIR  Destination directory
  HF_REPO    Hugging Face repository
EOF
}

case "${1:-q4}" in
  -h|--help)
    usage
    exit 0
    ;;
  q4) patterns=("*UD-IQ4_XS*" "*dspark-DeepSeek-V4-Flash-0731-Q8_0*") ;;
  q8) patterns=("*UD-Q8_K_XL*" "*dspark-DeepSeek-V4-Flash-0731-Q8_0*") ;;
  iq3) patterns=("*UD-IQ3_XXS*" "*dspark-DeepSeek-V4-Flash-0731-Q8_0*") ;;
  draft) patterns=("*dspark-DeepSeek-V4-Flash-0731-Q8_0*") ;;
  all) patterns=("*UD-IQ4_XS*" "*UD-Q8_K_XL*" "*UD-IQ3_XXS*" "*dspark-DeepSeek-V4-Flash-0731-Q8_0*") ;;
  *) usage >&2; exit 2 ;;
esac

require_command hf
HF_REPO=${HF_REPO:-unsloth/DeepSeek-V4-Flash-0731-GGUF}
mkdir -p "$MODEL_DIR"

args=(download "$HF_REPO" --local-dir "$MODEL_DIR")
for pattern in "${patterns[@]}"; do
  args+=(--include "$pattern")
done

printf 'Repository: %s\nDestination: %s\n' "$HF_REPO" "$MODEL_DIR"
df -h "$MODEL_DIR" | tail -1
hf "${args[@]}"
