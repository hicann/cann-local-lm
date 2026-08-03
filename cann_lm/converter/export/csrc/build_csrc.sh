#!/bin/bash
set -euo pipefail
#=============================================================================
# build_csrc.sh — Build the CANN-LM C++ backend (copied from
#                 torch-delegate-harmony/csrc/).
#
# Environment:
#   ASCEND_HOME_PATH  — CANN installation root (default: /home/developer/Ascend/cann-9.1.0)
#   TORCH_DELEGATE_HOME — torch-delegate-harmony project root
#
# Usage:
#   ./build_csrc.sh                   # build release
#   ./build_csrc.sh --debug           # build debug
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSRC_DIR="$SCRIPT_DIR"

# ---------- defaults ----------
BUILD_TYPE="${2:-Release}"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
CANN_PATH="${CANN_PATH:-/home/developer/Ascend/cann-9.1.0}"
BUILD_DIR="${CSRC_DIR}/build"

# ---------- parse args ----------
if [[ "${1:-}" == "--debug" ]]; then
    BUILD_TYPE="Debug"
fi

# ---------- find torch ----------
VENV_PYTHON="${TORCH_DELEGATE_HOME}/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: .venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

TORCH_DIR="$($VENV_PYTHON -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake", "Torch"))' 2>/dev/null || true)"

if [[ ! -d "$TORCH_DIR" ]]; then
    echo "ERROR: cannot find TorchConfig.cmake" >&2
    exit 1
fi

# ---------- cmake modules ----------
CMAKE_MODULE_DIR="${TORCH_DELEGATE_HOME}/cmake/modules"
CMAKE_THIRD_PARTY_DIR="${TORCH_DELEGATE_HOME}/output/third_party"

# ---------- print config ----------
echo "=== Build Config ==="
echo "  BUILD_TYPE:       $BUILD_TYPE"
echo "  CSRC_DIR:         $CSRC_DIR"
echo "  BUILD_DIR:        $BUILD_DIR"
echo "  TORCH_DIR:        $TORCH_DIR"
echo "  CANN_PATH: $CANN_PATH"
echo "  CMAKE_MODULE_DIR: $CMAKE_MODULE_DIR"
echo ""

# ---------- cmake configure ----------
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$CSRC_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_MODULE_PATH="$CMAKE_MODULE_DIR" \
    -DTorch_DIR="$TORCH_DIR" \
    -DASCEND_INSTALL_PATH="${CANN_PATH}/aarch64-linux" \
    -Dnlohmann_json_DIR="${CMAKE_THIRD_PARTY_DIR}/json/share/cmake/nlohmann_json"

# Note: ASCEND_INSTALL_PATH points to the CANN arch-specific dir that
#       contains include/ and lib64/ subdirectories.

# ---------- build ----------
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "=== Build complete ==="
echo "  Library: $(find "$BUILD_DIR" -name "*.so" -exec ls -lh {} \; 2>/dev/null | head -5)"
echo "  Build dir: $BUILD_DIR"
