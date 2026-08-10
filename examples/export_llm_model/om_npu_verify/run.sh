#!/usr/bin/env bash
#
# run.sh — Export gemma4/qwen3/qwen3.5 LLM models to epair (cann_lm definitions)
#
# 用法:
#   bash run.sh                  # 导出全部三模型
#   bash run.sh "qwen3 qwen3_5"  # 导出指定模型
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
VENV_DIR="${VENV_DIR:-/mnt/workspace/torch-delegate-harmony/.venv}"
PYTHON="${VENV_DIR}/bin/python3"

source /home/developer/Ascend/cann/set_env.sh

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
_TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# model -> HF path 映射 (本地已下载)
declare -A HF_PATHS=(
  [gemma4]="/mnt/workspace/data/models/gemma-4-E2B-it"
  [qwen3]="/mnt/workspace/data/models/qwen3-0.6B"
  [qwen3_5]="/mnt/workspace/data/models/Qwen3.5-0.8B"
)

MODELS="${1:-gemma4 qwen3 qwen3_5}"

cd "${ROOT_DIR}"

for m in ${MODELS}; do
  [[ -n "${HF_PATHS[$m]:-}" ]] || { echo "[✗] unknown model '${m}', available: ${!HF_PATHS[*]}"; exit 1; }
  echo ""
  echo "========================================"
  echo " Export LLM Model: ${m}  (${HF_PATHS[$m]})"
  echo "========================================"
  out="${ROOT_DIR}/epair_${m}"
  rm -rf "${out}" "${out}.pt2" "${ROOT_DIR}/input_0_epair_${m}.bin"
  "${PYTHON}" "${ROOT_DIR}/export_llm_model_epair.py" \
    --model "${m}" \
    --hf-path "${HF_PATHS[$m]}" \
    --seq-len 8 \
    --output-dir "${out}"
done

echo ""
echo "========================================"
echo " Done: ${MODELS}"
echo "========================================"
