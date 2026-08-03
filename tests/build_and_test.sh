#!/bin/bash
set -euo pipefail
#=============================================================================
# build_and_test.sh — Build and run all C++ tests
#
# Prerequisites:
#   - ASCEND_HOME_PATH must be set (or CANN at /home/developer/Ascend/cann-9.1.0)
#   - build_csrc.sh must have been run first (produces libtorch_delegate_backend.so)
#
# Usage:
#   ./build_and_test.sh              # build + run all tests
#   ./build_and_test.sh --debug      # debug build
#   ./build_and_test.sh --list       # list available test targets
#   ./build_and_test.sh -- <filter>  # run specific test via ctest -R
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/c++/epair_archive"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CSRC_DIR="$PROJECT_DIR/cann_lm/converter/export/csrc"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
CANN_PATH="${CANN_PATH:-/home/developer/Ascend/cann-9.1.0}"
BUILD_TYPE="${2:-Release}"
BUILD_DIR="${SCRIPT_DIR}/build"

if [[ "${1:-}" == "--debug" ]]; then
    BUILD_TYPE="Debug"
fi

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

CMAKE_MODULE_DIR="${TORCH_DELEGATE_HOME}/cmake/modules"
CMAKE_THIRD_PARTY_DIR="${TORCH_DELEGATE_HOME}/output/third_party"

echo "=== Build Config ==="
echo "  TEST_DIR:         $TEST_DIR"
echo "  BUILD_TYPE:       $BUILD_TYPE"
echo "  BUILD_DIR:        $BUILD_DIR"
echo "  CSRC_DIR:         $CSRC_DIR"
echo "  TORCH_DIR:        $TORCH_DIR"
echo "  CANN_PATH:        $CANN_PATH"
echo "  CMAKE_MODULE_DIR: $CMAKE_MODULE_DIR"
echo ""

# Generate fixtures first
echo "=== Generating fixtures ==="
"$VENV_PYTHON" "$TEST_DIR/gen_fixtures.py" --output-dir "$BUILD_DIR/fixtures"

# Configure and build
mkdir -p "$BUILD_DIR"
cmake -S "$TEST_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_MODULE_PATH="$CMAKE_MODULE_DIR" \
    -DASCEND_INSTALL_PATH="${CANN_PATH}/aarch64-linux" \
    -DTorch_DIR="$TORCH_DIR" \
    -Dnlohmann_json_DIR="${CMAKE_THIRD_PARTY_DIR}/json/share/cmake/nlohmann_json" \
    -DPython3_EXECUTABLE="$VENV_PYTHON"

cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "=== Running tests ==="
export LD_LIBRARY_PATH="${CANN_PATH}/aarch64-linux/lib64:${CANN_PATH}/aarch64-linux/devlib:$LD_LIBRARY_PATH"

if [[ "${1:-}" == "--list" ]]; then
    cmake --build "$BUILD_DIR" --target help 2>/dev/null | grep -E '^\.\./test_' || true
elif [[ "$#" -ge 2 && "${1:-}" == "--" ]]; then
    shift
    ctest --test-dir "$BUILD_DIR" -R "$@" --output-on-failure -V
else
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo ""
echo "=== All tests completed ==="
