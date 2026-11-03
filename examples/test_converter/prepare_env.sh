#!/usr/bin/env bash
# prepare_env.sh — 一次性: 装 converter PyPI 依赖进 P1 venv (deps --no-deps 漏的)
#
# cann_local_lm whl 用 --no-deps 装 (torchair/torch_delegate 不在 PyPI), 但 converter 加载权重/config
# 需 safetensors/pyyaml/packaging (PyPI 依赖). 本脚本一次性补装进 P1 venv.
#
# 用法: bash examples/test_converter/prepare_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-${PROJECT_ROOT}/deps/torch-delegate}"
VENV="${TORCH_DELEGATE_HOME}/.venv"
PIP="${VENV}/bin/pip"

[[ -x "${VENV}/bin/python" ]] || { echo "✗ P1 venv 不存在: ${VENV} (先 bash scripts/build.sh deps)"; exit 1; }

echo "=== prepare_env: 装 converter PyPI 依赖进 P1 venv ==="
echo "  venv: ${VENV}"
echo "  python: $(${VENV}/bin/python --version 2>&1)"

# 检查已装则跳 (幂等)
NEED_INSTALL=0
  for pkg in safetensors pyyaml packaging transformers; do
    if ! "${VENV}/bin/python" -c "import ${pkg//-/_}" 2>/dev/null; then
      NEED_INSTALL=1; echo "  缺: ${pkg}"
    fi
  done

if [[ $NEED_INSTALL -eq 0 ]]; then
  echo "  ✓ converter PyPI 依赖已装, 跳过"
  exit 0
fi

echo "  安装缺失依赖..."
"${PIP}" install safetensors pyyaml packaging transformers

echo ""
echo "=== prepare_env 完成 ==="
echo "  验证: ${VENV}/bin/python -c 'import safetensors,yaml,packaging,transformers; print(\"OK\")'"
