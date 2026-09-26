#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: build-ninfer.sh [--update]

Builds the Turing NInfer fork for CUDA architecture 75, including the CLI,
OpenAI-compatible server, and native benchmark.

Environment:
  NINFER_DIR        Source checkout
  NINFER_BUILD_DIR  CMake build directory
  NINFER_ENV        Conda environment containing CUDA >= 12.8 and dependencies
  CUDA_ARCH         CUDA architecture, default 75
  BUILD_JOBS        Parallel build jobs

--update fetches the pinned commit without merging an upstream branch. It
checks out that commit only when the existing checkout is clean.
EOF
}

update=0
while (($#)); do
  case "$1" in
    --update) update=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

: "${NINFER_DIR:=/home/algore/ninfer-2080ti-22g}"
: "${NINFER_BUILD_DIR:=$NINFER_DIR/build}"
: "${NINFER_ENV:=/home/algore/.conda/envs/ninfer-build}"
: "${NINFER_COMMIT:=6460bf03a86f1288dc7b240d70c30887199099af}"

for path in "$NINFER_ENV/bin/cmake" "$NINFER_ENV/bin/ninja" "$NINFER_ENV/bin/nvcc"; do
  require_file "$path"
done
require_command git
require_command nproc

if [[ ! -d "$NINFER_DIR/.git" ]]; then
  git clone --no-checkout https://github.com/mr-september/ninfer-2080ti-22g.git "$NINFER_DIR"
  git -C "$NINFER_DIR" checkout --detach "$NINFER_COMMIT"
elif ((update)); then
  git -C "$NINFER_DIR" fetch origin "$NINFER_COMMIT"
  if [[ "$(git -C "$NINFER_DIR" rev-parse HEAD)" != "$NINFER_COMMIT" ]]; then
    [[ -z "$(git -C "$NINFER_DIR" status --porcelain)" ]] ||
      die "refusing to move a dirty NInfer checkout to pinned commit $NINFER_COMMIT"
    git -C "$NINFER_DIR" checkout --detach "$NINFER_COMMIT"
  fi
fi
[[ "$(git -C "$NINFER_DIR" rev-parse HEAD)" == "$NINFER_COMMIT" ]] ||
  die "NInfer checkout must be at pinned commit $NINFER_COMMIT"

patch_file=$PROJECT_DIR/patches/ninfer-sm75-int8-kv-only.patch
if git -C "$NINFER_DIR" apply --check "$patch_file" 2>/dev/null; then
  git -C "$NINFER_DIR" apply "$patch_file"
elif ! git -C "$NINFER_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
  die "NInfer SM75 INT8-KV patch is neither applicable nor already applied"
fi

export PATH="$NINFER_ENV/bin:$PATH"
export PKG_CONFIG_PATH="$NINFER_ENV/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$NINFER_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset CC CC_FOR_BUILD CXX CXX_FOR_BUILD CFLAGS CXXFLAGS LDFLAGS
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT

"$NINFER_ENV/bin/cmake" \
  -S "$NINFER_DIR" \
  -B "$NINFER_BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" \
  -DCMAKE_CUDA_COMPILER="$NINFER_ENV/bin/nvcc" \
  -DCMAKE_PREFIX_PATH="$NINFER_ENV" \
  -DNINFER_SM75_INT8_KV_ONLY=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
"$NINFER_ENV/bin/cmake" --build "$NINFER_BUILD_DIR" \
  --parallel "${BUILD_JOBS:-$(nproc)}" \
  --target ninfer ninfer-serve ninfer_bench

"$NINFER_BUILD_DIR/apps/ninfer" --help >/dev/null
"$NINFER_BUILD_DIR/apps/ninfer-serve" --help >/dev/null
"$NINFER_BUILD_DIR/bench/ninfer_bench" --help >/dev/null
printf 'Built NInfer commit %s for sm_%s\n' \
  "$(git -C "$NINFER_DIR" rev-parse HEAD)" "${CUDA_ARCH:-75}"
