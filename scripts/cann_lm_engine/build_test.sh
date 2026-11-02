#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_test.sh — engine C++ UT 构建 + 运行 (ASAN, 独立 build_tests/ 树)
#
# BUILD_ENGINE=ON + BUILD_CONVERTER=OFF + stub_ndk 自动编译 + ctest
# engine UT 不依赖 CANN（用 stub NDK .so），但 cmake 需要 CANN env（头文件路径）
#
# 产物: build_tests/ 下 engine .so + ut_* + bbit
#
# 前置: source scripts/cann_lm_compile/set_env.sh; third_party 就绪
#
# 用法:
#   bash scripts/cann_lm_engine/build_test.sh             # ASAN 构建 + 跑测试
#   bash scripts/cann_lm_engine/build_test.sh --no-asan   # 不开 ASAN
#   bash scripts/cann_lm_engine/build_test.sh --skip-run  # 仅构建不跑
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build_tests/engine}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"

# ── 镜像模式: 三方库走 lib_cache 扁平布局 (modules Find 模块直连), 不需 output/third_party ──
MIRROR_THIRD_PARTY_VAL="${MIRROR_THIRD_PARTY:-OFF}"
CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH:-}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH 未设. 先 source scripts/cann_lm_compile/set_env.sh" >&2; exit 1
fi
# 镜像下校验 lib_cache/json; 非镜像校验 output/third_party/{json,gtest}
if [[ "$MIRROR_THIRD_PARTY_VAL" == "ON" ]]; then
    if [[ ! -d "${CANN_3RD_LIB_PATH}/lib_cache/json/include/nlohmann" ]]; then
        echo "ERROR: 镜像三方库缺失: ${CANN_3RD_LIB_PATH}/lib_cache/json/include/nlohmann" >&2; exit 1
    fi
else
    if [[ ! -d "${THIRD_PARTY}/json" ]]; then
        echo "ERROR: third_party 缺失: ${THIRD_PARTY}/json (先 bash build.sh third_party)" >&2; exit 1
    fi
    if [[ ! -d "${THIRD_PARTY}/gtest" ]]; then
        echo "ERROR: third_party 缺失: ${THIRD_PARTY}/gtest (先 bash build.sh third_party)" >&2; exit 1
    fi
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

echo "=== Engine Test Build (BUILD_ENGINE=ON, ASAN=$ASAN_VAL) ==="
echo "  BUILD_DIR:   $BUILD_DIR"
if [[ "$MIRROR_THIRD_PARTY_VAL" == "ON" ]]; then
    echo "  MIRROR:      ON (lib_cache 直连, CANN_3RD_LIB_PATH=$CANN_3RD_LIB_PATH)"
else
    echo "  third_party: ${THIRD_PARTY}"
fi
echo "  JOBS:        $JOBS"
echo ""

mkdir -p "$BUILD_DIR"
if [[ "$MIRROR_THIRD_PARTY_VAL" == "ON" ]]; then
    # 镜像内: 三方库走 modules Find 模块直连 lib_cache, 不传 nlohmann_json_DIR/GTest_DIR
    cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/$(uname -m)-linux" \
        -DMIRROR_THIRD_PARTY=ON \
        -DCANN_3RD_LIB_PATH="$CANN_3RD_LIB_PATH" \
        -DBUILD_ENGINE=ON \
        -DBUILD_CONVERTER=OFF \
        -DBUILD_BACKEND=ON \
        -DBUILD_COMPILE_TOOL=OFF \
        -DBUILD_TESTS=ON \
        -DENABLE_ASAN=$ASAN_VAL
else
    cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}/$(uname -m)-linux" \
        -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
        -DGTest_dir="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
        -DBUILD_ENGINE=ON \
        -DBUILD_CONVERTER=OFF \
        -DBUILD_BACKEND=ON \
        -DBUILD_COMPILE_TOOL=OFF \
        -DBUILD_TESTS=ON \
        -DENABLE_ASAN=$ASAN_VAL
fi

cmake --build "$BUILD_DIR" -j"$JOBS"

# 编译 stub NDK .so (NDKProxy dlopen 需要)
ENGINE_DIR="$BUILD_DIR/cann_lm/engine"
if [[ -d "$ENGINE_DIR" ]]; then
    _EXT="$REPO/cann_lm/engine/external_cann"
    _COMPAT="$REPO/cann_lm/engine/hiai_acl/src/ohos_compat.h"
    for stub_src in stub_neural_network_core stub_hiai_foundation; do
        stub_so="$ENGINE_DIR/lib${stub_src#stub_}.so"
        if [[ ! -f "$stub_so" ]]; then
            g++ -std=c++17 -fPIC -shared -I"$_EXT" -include"$_COMPAT" \
                "$REPO/tests/cpp/engine/stub_ndk/${stub_src}.cpp" \
                -o "$stub_so" -lpthread 2>/dev/null && echo "  ✓ stub: $(basename "$stub_so")"
        fi
    done
fi

if [[ "$RUN_TESTS" == "YES" ]]; then
    echo ""
    echo "=== Run engine tests (ctest) ==="
    [[ -d "$ENGINE_DIR" ]] && export LD_LIBRARY_PATH="$ENGINE_DIR:${LD_LIBRARY_PATH:-}"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo ""
echo "=== Engine test build complete ==="
echo "  build dir: $BUILD_DIR"
