#!/usr/bin/env bash
#
# run.sh — 用 cann_lm_converter.py CLI 把 gemma4/qwen3/qwen3.5 导出为 epair
#
# 用法:
#   bash run.sh                          # 导出全部三模型 (single 模式)
#   bash run.sh "qwen3 qwen3_5"          # 导出指定模型
#   MODE=prefill-decode bash run.sh gemma4   # prefill-decode 模式 (prefill+decode 两图)
#   DTYPE=float16 bash run.sh            # 指定 dtype (默认 bfloat16, 三模型原生)
#   SEQ_LEN=128 bash run.sh              # 指定 seq_len (默认 8, 快测)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../../.." && pwd)"
VENV_DIR="${VENV_DIR:-/mnt/workspace/torch-delegate-harmony/.venv}"
PYTHON="${VENV_DIR}/bin/python3"
CONVERTER="${PROJECT_ROOT}/cann_lm/converter/cann_lm_converter.py"

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

MODE="${MODE:-single}"
DTYPE="${DTYPE:-bfloat16}"
SEQ_LEN="${SEQ_LEN:-8}"
NUM_LAYERS="${NUM_LAYERS:-}"
MODELS="${1:-gemma4 qwen3 qwen3_5}"

cd "${ROOT_DIR}"

for m in ${MODELS}; do
  [[ -n "${HF_PATHS[$m]:-}" ]] || { echo "[✗] unknown model '${m}', available: ${!HF_PATHS[*]}"; exit 1; }
  echo ""
  echo "========================================"
  echo " cann_lm_converter.py: ${m}"
  echo "   HF:    ${HF_PATHS[$m]}"
  echo "   mode:  ${MODE}  dtype: ${DTYPE}  seq_len: ${SEQ_LEN}${NUM_LAYERS:+  layers: ${NUM_LAYERS}}"
  echo "========================================"
  out="${ROOT_DIR}/epair_${m}"
  rm -rf "${out}"
  args=(--hf-path "${HF_PATHS[$m]}" --output-dir "${out}" --dtype "${DTYPE}")
  if [[ "${MODE}" == "prefill-decode" ]]; then
    args+=(--mode prefill-decode --seq-len "${SEQ_LEN}")
  else
    args+=(--seq-len "${SEQ_LEN}")
  fi
  [[ -z "${NUM_LAYERS}" ]] || args+=(--num-layers "${NUM_LAYERS}")
  "${PYTHON}" "${CONVERTER}" "${args[@]}"
done

echo ""
echo "========================================"
echo " Export done: ${MODELS} (mode=${MODE})"
echo "========================================"

# 验证产物
echo ""
"${PYTHON}" "${ROOT_DIR}/verify.py" ${MODELS}
