#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_wheel.sh — 打包 cann-lm-engine whl (cann_lm.engine, 独立 distribution)
#
# cann_lm 是 PEP 420 namespace 包; 本 whl 只装 cann_lm.engine (api/backend/core/
# infra/pipeline), 源码在 cann_lm/engine/ (pyproject package-dir ../../cann_lm/engine).
# 与 cann-lm whl 共享 cann_lm 命名空间, 互不冲突.
#
# 用法: bash scripts/cann_lm_engine/build_wheel.sh
# 产物: $REPO/dist/cann_lm_engine-*.whl
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST="$REPO/dist"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
VENV="${VENV:-${TORCH_DELEGATE_HOME}/.venv}"
PIP="$VENV/bin/pip"

mkdir -p "$DIST"
echo "### build cann-lm-engine (cann_lm.engine)"
"$PIP" wheel "$SCRIPT_DIR" --no-deps -w "$DIST" 2>&1 | grep -E "Built|Created|error|Error" | tail -3
echo ""
echo "=== dist/ ==="
ls -la "$DIST"/*.whl 2>/dev/null | awk '{print "  "$5, $9}' || echo "  (无 whl)"
