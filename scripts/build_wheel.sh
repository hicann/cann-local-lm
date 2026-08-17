#!/usr/bin/env bash
#
# build_wheel.sh — 构建 cann-lm 系列 whl (cann-lm converter+models + cann-lm-engine)
#
# pyproject 都在 scripts/ 下 (源码在 cann_lm/):
#   scripts/cann_lm_converter/pyproject.toml -> cann-lm (converter + models + dopt_lm)
#   scripts/cann_lm_engine/pyproject.toml  -> cann-lm-engine (cann_lm.engine, 独立 distribution)
# cann_lm 是 PEP 420 namespace 包, 两 whl 共享 cann_lm 命名空间。
#
# 用法:
#   bash scripts/build_wheel.sh             # all: cann-lm + cann-lm-engine
#   bash scripts/build_wheel.sh engine      # 仅 cann-lm-engine
#   bash scripts/build_wheel.sh cann_lm     # 仅 cann-lm
# 产物: $REPO/dist/*.whl
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$REPO/dist"
VENV="${VENV:-/mnt/workspace/torch-delegate-harmony/.venv}"
PIP="$VENV/bin/pip"

mkdir -p "$DIST"

build_cann_lm_engine() {
  echo "### [1/2] build cann-lm-engine (cann_lm.engine)"
  "$PIP" wheel "$SCRIPT_DIR/cann_lm_engine" --no-deps -w "$DIST" 2>&1 | grep -E "Built|Created|error|Error" | tail -3
}

build_cann_lm() {
  echo "### [2/2] build cann-lm (converter + models + dopt_lm)"
  "$PIP" wheel "$SCRIPT_DIR/cann_lm_converter" --no-deps -w "$DIST" 2>&1 | grep -E "Built|Created|error|Error" | tail -3
}

case "${1:-all}" in
  engine)   build_cann_lm_engine ;;
  cann_lm)  build_cann_lm ;;
  all)      build_cann_lm_engine; build_cann_lm ;;
  *) echo "用法: bash scripts/build_wheel.sh {engine|cann_lm|all}"; exit 1 ;;
esac

echo ""
echo "=== dist/ ==="
ls -la "$DIST"/*.whl 2>/dev/null | awk '{print "  "$5, $9}' || echo "  (无 whl)"
