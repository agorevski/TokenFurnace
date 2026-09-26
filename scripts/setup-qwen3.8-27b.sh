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

Builds the selected llama.cpp or NInfer runtime and downloads Qwen3.8-27B for
the selected profile. The default is the measured Q4_K_M tensor-parallel
profile on NVLink-connected GPUs 2 and 3.
EOF
}

case "$PROFILE" in
  -h|--help) usage; exit 0 ;;
esac

load_target "$TARGET" "$PROFILE"

for command in nvidia-smi git hf; do
  require_command "$command"
done

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
IFS=',' read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
((gpu_count >= ${#selected_gpus[@]})) \
  || die "profile requires ${#selected_gpus[@]} GPUs, but nvidia-smi found $gpu_count"

printf 'Configuring %s with profile=%s on CUDA devices %s\n' \
  "$TARGET_NAME" "$PROFILE" "$CUDA_VISIBLE_DEVICES"
case "$BACKEND" in
  llama.cpp)
    require_command nvcc
    require_command cmake
    "$SCRIPT_DIR/build-llama-cpp.sh"
    ;;
  ninfer)
    "$SCRIPT_DIR/build-ninfer.sh"
    ;;
  *)
    die "setup-qwen3.8-27b.sh does not build backend=$BACKEND"
    ;;
esac
"$SCRIPT_DIR/download-model.sh" "$TARGET" "$PROFILE"

case "$BACKEND" in
  llama.cpp) require_file "$BUILD_DIR/bin/llama-server" ;;
  ninfer)
    require_file "$NINFER_BUILD_DIR/apps/ninfer-serve"
    require_file "$MODEL"
    require_file "$MODEL_DIR/artifact-manifest.json"
    require_command python3
    [[ "${KV_DTYPE:-int8}" == int8 ]] ||
      die "this patched SM75 NInfer build supports only KV_DTYPE=int8"
    python3 - "$MODEL" "$MODEL_DIR/artifact-manifest.json" \
      "$NINFER_CONTAINER_VERSION" "$NINFER_MIN_RUNTIME_REVISION" \
      "$MODEL_SIZE_BYTES" "$MODEL_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
expected_version = int(sys.argv[3])
expected_minimum_revision = sys.argv[4]
expected_size = int(sys.argv[5])
expected_sha256 = sys.argv[6]

with model.open("rb") as stream:
    header = stream.read(8)
expected_header = b"NINFER\0" + bytes([expected_version])
if header != expected_header:
    raise SystemExit(
        f"incompatible NInfer container header {header!r}; expected {expected_header!r}"
    )

manifest = json.loads(manifest_path.read_text())
artifact = manifest.get("artifact", {})
if artifact.get("container_version") != expected_version:
    raise SystemExit(
        "artifact manifest container_version does not match the pinned v2 profile"
    )
if artifact.get("filename") != model.name:
    raise SystemExit("artifact manifest filename does not match downloaded model")
runtime = manifest.get("runtime", {})
if runtime.get("minimum_revision") != expected_minimum_revision:
    raise SystemExit("artifact minimum runtime revision does not match the pinned profile")
if model.stat().st_size != expected_size or artifact.get("bytes") != expected_size:
    raise SystemExit("downloaded NInfer artifact size does not match the pinned profile")

digest = hashlib.sha256()
with model.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256 or artifact.get("sha256") != expected_sha256:
    raise SystemExit("downloaded NInfer artifact SHA-256 does not match the pinned profile")
PY
    ;;
esac
require_file "$MODEL"
if [[ "$BACKEND" == llama.cpp ]]; then
  mkdir -p "$MODEL_DIR/slot-cache"
fi

cat <<EOF

Setup complete.

Start the OpenAI-compatible server:
  ./scripts/serve-model.sh $TARGET $PROFILE

Check it from another terminal:
  ./scripts/status.sh $TARGET $PROFILE

Endpoint: http://$SERVER_HOST:$SERVER_PORT/v1
Model:    $MODEL_ALIAS
EOF
