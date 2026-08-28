#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_wheel.sh — 打包 cann-lm whl (converter + models + dopt_lm)
#
# cann_lm 是 PEP 420 namespace 包: 本 whl 装 converter+models+dopt_lm;
# cann-lm-engine whl (scripts/cann_lm_engine/) 装 cann_lm.engine, 独立 distribution.
# 源码在 cann_lm/ (pyproject 的 package-dir ../../cann_lm 指向).
# torchair/torch-delegate 不在 PyPI, 需源码/whl 预装 (见 AGENTS.md ① converter env), 故 --no-deps.
#
# 用法: bash scripts/cann_lm_converter/build_wheel.sh
# 产物: $REPO/dist/cann_lm-*.whl
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST="$REPO/dist"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
VENV="${VENV:-${TORCH_DELEGATE_HOME}/.venv}"
PIP="$VENV/bin/pip"

mkdir -p "$DIST"
echo "### build cann-lm (converter + models + dopt_lm)"
"$PIP" wheel "$SCRIPT_DIR" --no-deps -w "$DIST" 2>&1 | grep -E "Built|Created|error|Error" | tail -3
echo ""
echo "=== dist/ ==="
ls -la "$DIST"/*.whl 2>/dev/null | awk '{print "  "$5, $9}' || echo "  (无 whl)"
