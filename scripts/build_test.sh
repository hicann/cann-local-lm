#!/usr/bin/env bash
set -uo pipefail
#=============================================================================
# build_test.sh — cann-device-lm 顶层 C++ + Python 测试构建 + 运行入口
#
# 分发到子模块:
#   1. scripts/cann_lm_engine/build_test.sh   — engine C++ UT (BUILD_ENGINE=ON + stub_ndk + ctest)
#   2. scripts/cann_lm_compile/build_test.sh  — converter C++ UT (BUILD_CONVERTER=ON + ctest)
#   3. Python UT (pytest, tests/python/)
#
# 前置: source scripts/cann_lm_compile/set_env.sh (CANN env); third_party 就绪
#
# 用法:
#   bash scripts/build_test.sh                # 构建 + 跑全部 C++ + Python 测试
#   bash scripts/build_test.sh --no-asan      # 不开 ASAN
#   bash scripts/build_test.sh --skip-run     # 仅构建不跑
#   bash scripts/build_test.sh --c-only       # 只跑 C++ 测试（跳过 Python）
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# 解析参数
C_ONLY=0; PASS_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --c-only)  C_ONLY=1; shift;;
        *)         PASS_ARGS="$PASS_ARGS $1"; shift;;
    esac
done

FAIL=0

echo "=== [1/3] engine C++ UT ==="
if bash "$SCRIPT_DIR/cann_lm_engine/build_test.sh" $PASS_ARGS; then
    echo "✓ engine UT: PASS"
else
    echo "✗ engine UT: FAIL"; FAIL=1
fi

echo ""
echo "=== [2/3] converter C++ UT ==="
if bash "$SCRIPT_DIR/cann_lm_compile/build_test.sh" $PASS_ARGS; then
    echo "✓ converter UT: PASS"
else
    echo "✗ converter UT: FAIL"; FAIL=1
fi

if [[ $C_ONLY -eq 0 ]]; then
    echo ""
    echo "=== [3/3] Python UT (pytest) ==="
    PY_FAIL=0
    SUITES=(
        "tests/python/converter"
        "tests/python/llm_models"
        "tests/python --ignore=tests/python/converter --ignore=tests/python/llm_models --ignore=tests/python/dopt_lm"
        "tests/python/dopt_lm"
    )
    for suite in "${SUITES[@]}"; do
        dir="${suite%% *}"
        if [[ -d "$REPO/$dir" ]]; then
            echo "  --- pytest $dir ---"
            if ( cd "$REPO" && timeout 600 python3 -m pytest $suite -q ); then
                echo "  ✓ $dir: PASS"
            else
                echo "  ✗ $dir: FAIL"; PY_FAIL=1
            fi
        fi
    done
    [[ $PY_FAIL -eq 0 ]] && echo "Python UT: 全部通过" || { echo "Python UT: 有失败"; FAIL=1; }
fi

echo ""
echo "=== 全部测试完成 ==="
if [[ $FAIL -eq 0 ]]; then
    echo "✓ 全部通过"
    exit 0
else
    echo "✗ 有失败"
    exit 1
fi
