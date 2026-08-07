#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: build-llama-cpp.sh [--clean]

Builds current llama.cpp for four RTX 8000 GPUs:
  CUDA architecture 75
  CUDA graphs and Flash Attention
  NCCL when installed
  peer access through batch size 2048

Environment:
  LLAMA_DIR     Source checkout
  BUILD_DIR     CMake build directory
  CUDA_ROOT     CUDA toolkit root
  CUDA_ARCH     CUDA architecture, default 75
  BUILD_JOBS    Parallel build jobs
EOF
}

clean=0
case "${1:-}" in
  "") ;;
  --clean) clean=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

for cmd in git cmake gcc g++ nproc; do
  require_command "$cmd"
done

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" pull --ff-only
fi

if ((clean)); then
  [[ "$BUILD_DIR" == "$LLAMA_DIR"/* && "$BUILD_DIR" != "$LLAMA_DIR" ]] \
    || die "refusing to clean build directory outside LLAMA_DIR: $BUILD_DIR"
  rm -rf -- "$BUILD_DIR"
fi

CUDA_ROOT=${CUDA_ROOT:-}
if [[ -z "$CUDA_ROOT" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT=$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)
fi
[[ -n "$CUDA_ROOT" ]] || die "CUDA_ROOT is unset and nvcc was not found"

CUDA_ARCH=${CUDA_ARCH:-75}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
cmake_args=(
  -S "$LLAMA_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_CUDA=ON
  -DGGML_CUDA_NCCL=ON
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=2048
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  -DCUDAToolkit_ROOT="$CUDA_ROOT"
  -DCMAKE_C_COMPILER=/usr/bin/gcc
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
)

for pair in \
  "CUDA_cudart_LIBRARY=$CUDA_ROOT/lib/libcudart.so.12" \
  "CUDA_cublas_LIBRARY=$CUDA_ROOT/lib/libcublas.so.12" \
  "CUDA_cublasLt_LIBRARY=$CUDA_ROOT/lib/libcublasLt.so.12"; do
  key=${pair%%=*}
  value=${pair#*=}
  if [[ -e "$value" ]]; then
    cmake_args+=("-D$key=$value")
  fi
done

nccl_root=$(find /home/algore/miniconda3/lib/python*/site-packages/nvidia/nccl \
  -maxdepth 0 -type d -print -quit 2>/dev/null || true)
if [[ -n "$nccl_root" ]]; then
  cmake_args+=(
    "-DNCCL_INCLUDE_DIR=$nccl_root/include"
    "-DNCCL_LIBRARY=$nccl_root/lib/libnccl.so.2"
  )
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config Release -j "$BUILD_JOBS" \
  --target llama-server llama-cli llama-bench llama-gguf-split

"$BUILD_DIR/bin/llama-server" --version
