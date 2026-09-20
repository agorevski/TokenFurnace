#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TARGET=qwen3.8-27b
PROFILE=${1:-llama-cpp-q4km-2gpu-tensor}

usage() {
  cat <<'EOF'
Usage: setup-qwen3.8-27b.sh [PROFILE]

Builds the CUDA/NCCL llama.cpp runtime and downloads Qwen3.8-27B for the
selected profile. The default is the measured Q4_K_M tensor-parallel profile
on NVLink-connected GPUs 2 and 3.
EOF
}

case "$PROFILE" in
  -h|--help) usage; exit 0 ;;
esac

load_target "$TARGET" "$PROFILE"
[[ "$BACKEND" == llama.cpp ]] \
  || die "setup-qwen3.8-27b.sh supports llama.cpp profiles; got backend=$BACKEND"

for command in nvidia-smi nvcc git cmake hf; do
  require_command "$command"
done

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
IFS=',' read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
((gpu_count >= ${#selected_gpus[@]})) \
  || die "profile requires ${#selected_gpus[@]} GPUs, but nvidia-smi found $gpu_count"

printf 'Configuring %s with profile=%s on CUDA devices %s\n' \
  "$TARGET_NAME" "$PROFILE" "$CUDA_VISIBLE_DEVICES"
"$SCRIPT_DIR/build-llama-cpp.sh"
"$SCRIPT_DIR/download-model.sh" "$TARGET" "$PROFILE"

require_file "$BUILD_DIR/bin/llama-server"
require_file "$MODEL"
mkdir -p "$MODEL_DIR/slot-cache"

cat <<EOF

Setup complete.

Start the OpenAI-compatible server:
  ./scripts/serve-model.sh $TARGET $PROFILE

Check it from another terminal:
  ./scripts/status.sh $TARGET $PROFILE

Endpoint: http://$SERVER_HOST:$SERVER_PORT/v1
Model:    $MODEL_ALIAS
EOF
