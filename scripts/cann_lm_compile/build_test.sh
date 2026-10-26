#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_test.sh — converter C++ UT 构建 + 运行 (ASAN, 独立 build_tests/ 树)
#
# BUILD_CONVERTER=ON + BUILD_ENGINE=OFF + ctest
# converter UT (tests/cpp/epair_archive/) 依赖 CANN lib64 + nlohmann_json + GTest
#
# 产物: build_tests/ 下 converter .so + test_*
#
# 前置: source scripts/cann_lm_compile/set_env.sh; third_party 就绪
#
# 用法:
#   bash scripts/cann_lm_compile/build_test.sh             # ASAN 构建 + 跑测试
#   bash scripts/cann_lm_compile/build_test.sh --no-asan   # 不开 ASAN
#   bash scripts/cann_lm_compile/build_test.sh --skip-run  # 仅构建不跑
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build_tests/converter}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH 未设. 先 source scripts/cann_lm_compile/set_env.sh" >&2; exit 1
fi
if [[ ! -d "${THIRD_PARTY}/json" ]]; then
    echo "ERROR: third_party 缺失: ${THIRD_PARTY}/json (先 bash build.sh third_party)" >&2; exit 1
fi

ASAN_VAL="ON"; RUN_TESTS="YES"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-asan)   ASAN_VAL="OFF"; shift;;
        --skip-run)  RUN_TESTS="NO"; shift;;
        -h|--help)   grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
        *) echo "unknown arg: $1 (see --help)" >&2; exit 1;;
    esac
done
JOBS="$(nproc 2>/dev/null || echo 4)"

echo "=== Converter Test Build (BUILD_CONVERTER=ON, ASAN=$ASAN_VAL) ==="
echo "  BUILD_DIR:   $BUILD_DIR"
echo "  third_party: ${THIRD_PARTY}"
echo "  JOBS:        $JOBS"
echo ""

mkdir -p "$BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/$(uname -m)-linux" \
    -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
    -DGTest_dir="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
    -DBUILD_ENGINE=OFF \
    -DBUILD_CONVERTER=ON \
    -DBUILD_BACKEND=ON \
    -DBUILD_COMPILE_TOOL=OFF \
    -DBUILD_TESTS=ON \
    -DENABLE_ASAN=$ASAN_VAL

cmake --build "$BUILD_DIR" -j"$JOBS"

if [[ "$RUN_TESTS" == "YES" ]]; then
    echo ""
    echo "=== Run converter tests (ctest) ==="
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo ""
echo "=== Converter test build complete ==="
echo "  build dir: $BUILD_DIR"
