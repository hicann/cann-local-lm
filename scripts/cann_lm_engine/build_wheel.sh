#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_wheel.sh — 打包 cann-lm-engine whl (engine python ctypes 绑定)
#
# 源头: cann_lm/engine/python/(flat-layout 项目根, pyproject.toml 在此, 包 cann_lm/
# 居其下)。本 whl 纯 python(ctypes 直通 C ABI, 零运行时依赖), 不含 C++ .so —
# .so 由消费者经 CANN_LM_LIB env 提供(host 用 build_host2 的 libcann_lm_engine.so,
# 设备用 deploy_ohos/ 的 OHOS aarch64 .so; OHOS .so 平台 tag 不入 wheel)。
#
# 与 cann-lm converter wheel 共享 cann_lm 命名空间: 本 wheel 的 cann_lm/__init__.py
# 含 pkgutil.extend_path, 两 wheel 同 env 不互隐(cann_lm.Engine 与 cann_lm.converter 并存)。
#
# 用法: bash scripts/cann_lm_engine/build_wheel.sh
# 产物: $REPO/dist/cann_lm_engine-*.whl
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_SRC="$REPO/cann_lm/engine/python"
PYPROJECT="$REPO/scripts/cann_lm_engine/pyproject.toml"
DIST="$REPO/dist"
mkdir -p "$DIST"

# PIP 选择: 优先 P1 venv 的 pip(converter wheel 需要 torch venv); engine wheel 纯 python,
# venv 不在则回退系统 python3 -m pip(本机无 deps/torch-delegate/.venv 时逃生)。
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-${REPO}/deps/torch-delegate}"
VENV="${VENV:-${TORCH_DELEGATE_HOME}/.venv}"
if [[ -x "$VENV/bin/pip" ]]; then
  PIP="$VENV/bin/pip"
else
  PIP="python3 -m pip"
fi

echo "### build cann-lm-engine (engine python ctypes 绑定)"
echo "  源: $PY_SRC"
echo "  pip: $PIP"
(cd "$REPO/scripts/cann_lm_engine" && $PIP wheel . --no-deps -w "$DIST") 2>&1 | grep -E "Built|Created|error|Error" | tail -3
rm -rf "$REPO/scripts/cann_lm_engine/cann_lm_engine.egg-info"  # 清理构建中间产物
echo ""
echo "=== dist/ ==="
ls -la "$DIST"/cann_lm_engine-*.whl 2>/dev/null | awk '{print "  "$5, $9}' \
  || { echo "  ✗ 无 cann_lm_engine-*.whl"; exit 1; }
