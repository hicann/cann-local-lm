#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_cpp.sh — CANN-LM C++ 统一构建入口 (顶层 CMakeLists, 单 build 树)
#
# 产物 (默认全开):
#   - libtorch_delegate_backend.so   (cann_lm/converter/export/csrc, backend lib)
#   - cann_lm_compile                (csrc/tools/compile_main.cc, epair→OM 编译工具)
#   - test_*                          (tests/c++/epair_archive, C++ tests)
#
# 前置: source scripts/set_env_compile.sh   (设 ASCEND_HOME_PATH / LD_LIBRARY_PATH)
#
# 用法:
#   bash scripts/cann_lm_compile/build_cpp.sh                       # 全量 release
#   bash scripts/cann_lm_compile/build_cpp.sh --asan                # ASAN (tests 强制, 兑现 AGENTS.md)
#   bash scripts/cann_lm_compile/build_cpp.sh --debug              # Debug
#   bash scripts/cann_lm_compile/build_cpp.sh --no-tests --no-om    # 仅 backend lib
#   bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend
#
# 环境覆盖: BUILD_DIR / TORCH_DELEGATE_HOME
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"

# third_party (nlohmann_json + GTest 预构建依赖; csrc/tests 不依赖 torch, 无需 venv)
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"

# ── 校验 env ──
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH 未设. 先 source scripts/set_env_compile.sh" >&2
    exit 1
fi

# ── 派生路径 (CANN + third_party; csrc 不依赖 torch, 无需 venv) ──
THIRD_PARTY="${TORCH_DELEGATE_HOME}/output/third_party"
ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/aarch64-linux"

# ── 参数 (每次显式传 -D 值, 覆盖 cache, 避免 sticky: 如 --no-tests 一次后 BUILD_TESTS 黏 OFF) ──
BUILD_TYPE="Release"
ASAN_VAL="OFF"
BUILD_TESTS_VAL="ON"
BUILD_COMPILE_TOOL_VAL="ON"
BUILD_BACKEND_VAL="ON"
TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      BUILD_TYPE="Debug"; shift;;
        --asan)       ASAN_VAL="ON"; shift;;
        --no-tests)        BUILD_TESTS_VAL="OFF"; shift;;
        --no-compile-tool) BUILD_COMPILE_TOOL_VAL="OFF"; shift;;
        --no-om)           BUILD_COMPILE_TOOL_VAL="OFF"; shift;;   # 别名 (旧 --no-om)
        --no-backend)      BUILD_BACKEND_VAL="OFF"; shift;;
        --target)     TARGET="$2"; shift 2;;
        -h|--help)    grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
        *) echo "unknown arg: $1 (see --help)" >&2; exit 1;;
    esac
done
JOBS="$(nproc 2>/dev/null || echo 4)"

echo "=== Build Config (顶层统一 build 树, 不依赖 torch) ==="
echo "  REPO:                $REPO"
echo "  BUILD_DIR:           $BUILD_DIR"
echo "  BUILD_TYPE:          $BUILD_TYPE"
echo "  ASCEND_INSTALL_PATH: $ASCEND_INSTALL_PATH"
echo "  third_party:         ${THIRD_PARTY} (nlohmann_json + GTest)"
echo "  ASAN:                $ASAN_VAL"
echo "  JOBS:                $JOBS"
echo ""

mkdir -p "$BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH" \
    -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
    -DGTest_DIR="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
    -DBUILD_BACKEND=$BUILD_BACKEND_VAL \
    -DBUILD_COMPILE_TOOL=$BUILD_COMPILE_TOOL_VAL \
    -DBUILD_TESTS=$BUILD_TESTS_VAL \
    -DENABLE_ASAN=$ASAN_VAL

if [[ -n "$TARGET" ]]; then
    cmake --build "$BUILD_DIR" -j"$JOBS" --target "$TARGET"
else
    cmake --build "$BUILD_DIR" -j"$JOBS"
fi

echo ""
echo "=== Build complete ==="
echo "  build dir: $BUILD_DIR"
ls -lh "$BUILD_DIR"/*.so "$BUILD_DIR"/cann_lm_compile "$BUILD_DIR"/export_and_load_example 2>/dev/null | awk '{print "  "$5, $9}' || true
