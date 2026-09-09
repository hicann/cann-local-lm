#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_test.sh — CANN-LM C++ 测试构建 + 运行 (ASAN, 独立 build_tests 树)
#
# 镜像 torch-delegate build_test.sh: 独立 build 目录 (ASAN 插桩, 与 release build/ 隔离),
#   BUILD_BACKEND=ON (产 ASAN .so 供测试链) + BUILD_TESTS=ON + ENABLE_ASAN=ON, 构建 + ctest.
#   static-libasan (DelegateCommonFlags) 免 LD_PRELOAD, ctest 可直接跑 (兑现 AGENTS.md ASAN 强制).
#
# 产物: build_tests/ 下 ASAN 版 libtorch_delegate_backend.so + test_* (+ .epair fixtures)
#
# 前置: source scripts/set_env.sh (CANN env); third_party 就绪 (bash scripts/build.sh third_party)
#
# 用法:
#   bash scripts/cann_lm_compile/build_test.sh             # ASAN 构建 + 跑测试 (默认)
#   bash scripts/cann_lm_compile/build_test.sh --no-asan   # 不开 ASAN
#   bash scripts/cann_lm_compile/build_test.sh --skip-run  # 仅构建不跑
#
# 环境覆盖: BUILD_DIR / THIRD_PARTY
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build_tests}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH 未设. 先 source scripts/set_env.sh" >&2; exit 1
fi
if [[ ! -d "${THIRD_PARTY}/json" ]]; then
    echo "ERROR: third_party 缺失: ${THIRD_PARTY}/json (先 bash scripts/build.sh third_party)" >&2; exit 1
fi
ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/aarch64-linux"

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

echo "=== Test Build Config (独立 build_tests 树, ASAN, 与 release build/ 隔离) ==="
echo "  BUILD_DIR:           $BUILD_DIR"
echo "  ASCEND_INSTALL_PATH: $ASCEND_INSTALL_PATH"
echo "  third_party:         ${THIRD_PARTY} (nlohmann_json + GTest)"
echo "  ASAN:                $ASAN_VAL (static-libasan, 免 LD_PRELOAD)"
echo "  JOBS:                $JOBS"
echo ""

mkdir -p "$BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH" \
    -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
    -DGTest_DIR="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
    -DBUILD_BACKEND=ON \
    -DBUILD_COMPILE_TOOL=OFF \
    -DBUILD_TESTS=ON \
    -DENABLE_ASAN=$ASAN_VAL

cmake --build "$BUILD_DIR" -j"$JOBS"

if [[ "$RUN_TESTS" == "YES" ]]; then
    echo ""
    echo "=== Run tests (ctest) ==="
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo ""
echo "=== Test build complete ==="
echo "  build dir: $BUILD_DIR"
find "$BUILD_DIR" -maxdepth 4 -name 'test_*' -type f -executable 2>/dev/null | head | sed 's/^/  /' || true
