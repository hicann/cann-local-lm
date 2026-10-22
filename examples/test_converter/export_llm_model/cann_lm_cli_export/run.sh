#!/usr/bin/env bash
#
# run.sh — 用 cann_lm_converter.py CLI 把 gemma4/qwen3.5 导出为 epair
#
# 用法:
#   bash run.sh                          # 导出全部两模型 (single 模式)
#   bash run.sh "qwen3_5"               # 导出指定模型
#   MODE=prefill-decode bash run.sh gemma4   # prefill-decode 模式 (prefill+decode 两图)
#   DTYPE=float16 bash run.sh            # 指定 dtype (默认空=按 target-soc 派生: Ascend→bf16, Kirin→fp16)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../../.." && pwd)"
CONVERTER="${PROJECT_ROOT}/cann_lm/converter/cann_lm_converter.py"

source "$PROJECT_ROOT/scripts/set_env_export.sh"

PYTHON="${CONVERTER_PYTHON}"

# model -> HF path 映射 (由调用方 export 必填; 无机器默认)
declare -A HF_PATHS=(
  [gemma4]="${GEMMA4_HF_PATH:?请 export GEMMA4_HF_PATH=<gemma4 model dir>}"
  [qwen3_5]="${QWEN3_5_HF_PATH:?请 export QWEN3_5_HF_PATH=<qwen3.5 model dir>}"
)

MODE="${MODE:-single}"
DTYPE="${DTYPE:-}"   # 空=converter 按 --target-soc 派生 (Ascend→bf16, Kirin→fp16); float16/float32 覆盖
SEQ_LEN="${SEQ_LEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-}"
MODELS="${1:-gemma4 qwen3_5}"

cd "${ROOT_DIR}"

for m in ${MODELS}; do
  [[ -n "${HF_PATHS[$m]:-}" ]] || { echo "[✗] unknown model '${m}', available: ${!HF_PATHS[*]}"; exit 1; }
  echo ""
  echo "========================================"
  echo " cann_lm_converter.py: ${m}"
  echo "   HF:    ${HF_PATHS[$m]}"
  echo "   mode:  ${MODE}  dtype: ${DTYPE}${NUM_LAYERS:+  layers: ${NUM_LAYERS}}"
  echo "========================================"
  out="${ROOT_DIR}/epair_${m}"
  rm -rf "${out}"
  args=(--hf-path "${HF_PATHS[$m]}" --output-dir "${out}")
  [[ -n "${DTYPE:-}" ]] && args+=(--dtype "${DTYPE}")
  if [[ "${MODE}" == "prefill-decode" ]]; then
    args+=(--mode prefill-decode)
  else
    args+=(--mode single)
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
