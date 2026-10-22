#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_cpp.sh — CANN-LM C++ 统一构建入口 (顶层 CMakeLists, 单 build 树)
#
# 产物 (默认: backend + compile tool, 不含测试):
#   - libtorch_delegate_backend.so   (cann_lm/converter/export/csrc, backend lib)
#   - cann_lm_compile                (csrc/tools/compile_main.cc, epair→OM 编译工具)
# 测试构建独立: bash scripts/cann_lm_compile/build_test.sh (ASAN, 独立 build_tests 树 + ctest)
#
# 前置: source scripts/set_env.sh   (设 ASCEND_HOME_PATH / LD_LIBRARY_PATH / SOC / GE_*)
#
# 用法:
#   bash scripts/cann_lm_compile/build_cpp.sh                       # release .so + cann_lm_compile
#   bash scripts/cann_lm_compile/build_cpp.sh --debug              # Debug
#   bash scripts/cann_lm_compile/build_cpp.sh --tests             # 含测试 (一般用 build_test.sh)
#   bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend
#
# 环境覆盖: BUILD_DIR / THIRD_PARTY
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"

# third_party (nlohmann_json + GTest 预构建依赖; csrc/tests 不依赖 torch, 无需 venv)
# 自带三方库 (bash scripts/build.sh third_party 构建), 不再依赖 torch-delegate
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"

# ── 校验 env ──
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH 未设. 先 source scripts/set_env.sh" >&2
    exit 1
fi

# ── 派生路径 (CANN + third_party; csrc 不依赖 torch, 无需 venv) ──
# THIRD_PARTY 已在头部设 (自带 output/third_party); 缺失则提示先跑 build.sh third_party
if [[ ! -d "${THIRD_PARTY}/json" ]]; then
    echo "ERROR: third_party 缺失: ${THIRD_PARTY}/json" >&2
    echo "  先构建: bash scripts/build.sh third_party" >&2
    exit 1
fi
ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/$(uname -m)-linux"

# ── 参数 (每次显式传 -D 值, 覆盖 cache, 避免 sticky: 如 --no-tests 一次后 BUILD_TESTS 黏 OFF) ──
BUILD_TYPE="Release"
ASAN_VAL="OFF"
COVERAGE_VAL="OFF"
BUILD_TESTS_VAL="OFF"
BUILD_COMPILE_TOOL_VAL="ON"
BUILD_BACKEND_VAL="ON"
TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      BUILD_TYPE="Debug"; shift;;
        --asan)       ASAN_VAL="ON"; shift;;
        --coverage)   COVERAGE_VAL="ON"; shift;;   # gcov 覆盖率 (独立 build 目录, 与 ASAN 互斥)
        --tests)           BUILD_TESTS_VAL="ON"; shift;;     # 含测试 (一般用 build_test.sh)
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
echo "  COVERAGE:            $COVERAGE_VAL"
echo "  JOBS:                $JOBS"
echo ""

mkdir -p "$BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH" \
    -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
    -DGTest_DIR="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
    -DBUILD_ENGINE=OFF \
    -DBUILD_CONVERTER=ON \
    -DBUILD_BACKEND=$BUILD_BACKEND_VAL \
    -DBUILD_COMPILE_TOOL=$BUILD_COMPILE_TOOL_VAL \
    -DBUILD_TESTS=$BUILD_TESTS_VAL \
    -DENABLE_ASAN=$ASAN_VAL \
    -DENABLE_COVERAGE=$COVERAGE_VAL

if [[ -n "$TARGET" ]]; then
    cmake --build "$BUILD_DIR" -j"$JOBS" --target "$TARGET"
else
    cmake --build "$BUILD_DIR" -j"$JOBS"
fi

echo ""
echo "=== Build complete ==="
echo "  build dir: $BUILD_DIR"
ls -lh "$BUILD_DIR"/*.so "$BUILD_DIR"/cann_lm_compile "$BUILD_DIR"/export_and_load_example 2>/dev/null | awk '{print "  "$5, $9}' || true
