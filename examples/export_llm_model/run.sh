#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
CSRC_DIR="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
VENV_DIR="${VENV_DIR:-/mnt/workspace/torchair_lazy_venv}"
PYTHON="${VENV_DIR}/bin/python3"

source /home/developer/Ascend/cann/set_env.sh

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

_TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "========================================"
echo " Export LLM Model: qwen3-0.6B"
echo "========================================"

cd "${ROOT_DIR}"
rm -rf epair_model epair_model.epair input_0.bin

"${PYTHON}" export_llm_model_epair.py

echo ""
echo "========================================"
echo " Done"
echo "========================================"
