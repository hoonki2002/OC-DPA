#!/usr/bin/env bash
# Step 0 — Build DPA-MoE-enabled llama.cpp.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

: "${BUILD_DIR:=third_party/llama.cpp/build-dpa-moe}"
: "${BUILD_TYPE:=Release}"
: "${CUDA_ROOT:=}"
: "${CUDA_ARCHITECTURES:=80}"
: "${BUILD_TARGETS:=llama-simple dpa-moe-gate-trace llama-perplexity llama-server llama-quantize}"
: "${BUILD_JOBS:=$(nproc)}"
: "${LLAMA_BUILD_SERVER:=ON}"
: "${FORCE_CONFIGURE:=0}"

if [ -z "$CUDA_ROOT" ]; then
    if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
        CUDA_ROOT=/usr/local/cuda-12.8
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_ROOT=/usr/local/cuda
    else
        echo "[0_build] ERROR: no CUDA nvcc found. Set CUDA_ROOT=/path/to/cuda." >&2
        exit 2
    fi
fi

CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
[ -x "$CUDA_COMPILER" ] || {
    echo "[0_build] ERROR: CUDA compiler not executable: $CUDA_COMPILER" >&2
    exit 2
}

echo "==== 0_build llama.cpp ===="
echo "build dir       : $BUILD_DIR"
echo "build type      : $BUILD_TYPE"
echo "cuda root       : $CUDA_ROOT"
echo "cuda compiler   : $CUDA_COMPILER"
echo "cuda arch       : $CUDA_ARCHITECTURES"
echo "targets         : $BUILD_TARGETS"
echo "jobs            : $BUILD_JOBS"
echo

if [ "$FORCE_CONFIGURE" = "1" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cmake -S third_party/llama.cpp -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGML_CUDA=ON \
        -DDPA_MOE_EXT=ON \
        -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
        -DCUDAToolkit_ROOT="$CUDA_ROOT" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER="$LLAMA_BUILD_SERVER"
else
    echo "[0_build] reuse existing configure: $BUILD_DIR/CMakeCache.txt"
fi

for target in $BUILD_TARGETS; do
    echo "==== build target: $target ===="
    cmake --build "$BUILD_DIR" --target "$target" -j "$BUILD_JOBS"
done

echo "[0_build] done"
find "$BUILD_DIR/bin" -maxdepth 1 -type f -perm -111 \
    \( -name 'llama-*' -o -name 'dpa-moe-*' \) -printf '  %p\n' | sort
