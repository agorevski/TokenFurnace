#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TARGET=ternary-bonsai-2-27b
PROFILE=${1:-}
PRISM_RELEASE=prism-b10709-9a9394a
PRISM_COMMIT=9a9394a89
PRISM_ARCHIVE=llama-prism-b10709-9a9394a-bin-linux-cuda-12.4-x64.tar.gz
PRISM_SHA256=f542fdcc818562359e947db65e0b11c4658dd5ca3bd240490448252e817d8e7a
PRISM_URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/$PRISM_RELEASE/$PRISM_ARCHIVE"

usage() {
  cat <<'EOF'
Usage: setup-ternary-bonsai-2-27b.sh [PROFILE]

Installs the pinned PrismML llama.cpp CUDA 12.4 release and downloads the
selected Bonsai 2 27B GGUF. The default profile is the measured PQ2_0 winner
on one RTX 8000. Available profiles:

  llama-cpp-pq2-1gpu   Faster measured prefill and decode; default
  llama-cpp-ptq1-1gpu  Smaller 5.95 GB artifact
EOF
}

case "$PROFILE" in
  -h|--help) usage; exit 0 ;;
esac

load_target "$TARGET" "$PROFILE"
[[ "$BACKEND" == llama.cpp ]] \
  || die "setup-ternary-bonsai-2-27b.sh requires a llama.cpp profile; got backend=$BACKEND"

for command in curl hf nvidia-smi sha256sum tar; do
  require_command "$command"
done

IFS=',' read -ra selected_gpus <<<"$CUDA_VISIBLE_DEVICES"
for gpu in "${selected_gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid CUDA device in profile: $gpu"
  nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null \
    || die "profile requests unavailable CUDA device $gpu"
done

runtime=$BUILD_DIR/bin/llama-server
get_runtime_version() {
  add_runtime_libraries
  "$runtime" --version 2>&1
}
runtime_version=$(get_runtime_version || true)
if [[ "$runtime_version" != *"commit $PRISM_COMMIT"* ]]; then
  tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/tokenfurnace-bonsai2.XXXXXXXX")
  cleanup() {
    if [[ -n "${tmp_dir:-}" && "$tmp_dir" == "${TMPDIR:-/tmp}"/tokenfurnace-bonsai2.* ]]; then
      rm -rf -- "$tmp_dir"
    fi
  }
  trap cleanup EXIT

  archive=$tmp_dir/$PRISM_ARCHIVE
  printf 'Downloading PrismML llama.cpp %s\n' "$PRISM_RELEASE"
  curl -fL --retry 3 --output "$archive" "$PRISM_URL"
  printf '%s  %s\n' "$PRISM_SHA256" "$archive" | sha256sum --check -

  mkdir -p "$BUILD_DIR/bin"
  tar -xzf "$archive" -C "$BUILD_DIR/bin" --strip-components=1
  runtime_version=$(get_runtime_version)
fi

[[ "$runtime_version" == *"commit $PRISM_COMMIT"* ]] \
  || die "installed llama.cpp does not report required commit $PRISM_COMMIT"
printf '%s\n' "$runtime_version"

"$SCRIPT_DIR/download-model.sh" "$TARGET" "$PROFILE"

require_file "$runtime"
require_file "$BUILD_DIR/bin/llama-bench"
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
GGUF:     $MODEL
EOF
