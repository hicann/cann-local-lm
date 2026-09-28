#!/usr/bin/env bash
# dopt_op_benchmark — reference-operator (QLinear fakequant) golden + comparison harness.
# Usage:
#   bash run.sh gen        [--model-path X --quant-dir Y --output-dir Z --device cpu --operators sample]
#   bash run.sh compare --op-name model.layers.0.self_attn.q_proj [--model-path X --quant-dir Y --golden-dir Z --device cpu]
# Paths default to env vars: GEMMA4_MODEL_PATH, DOPT_QUANT_DIR, DOPT_PYTHON, DEVICE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # CANN-LM_5883
PYTHON_BIN="${DOPT_PYTHON:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/cann_lm/converter/quant:${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Auto-detect broken torchaudio (torch/torchaudio version mismatch in some system pythons);
# fall back to the bundled stub shim so transformers can load audio-capable models (e.g. gemma4)
# without real audio I/O (GPTQ text path does not need it).
if ! "${PYTHON_BIN}" -c "import torchaudio" >/dev/null 2>&1; then
  export PYTHONPATH="${SCRIPT_DIR}/shims:${PYTHONPATH}"
  echo "[run.sh] torchaudio import failed in this env; using bundled stub shim (${SCRIPT_DIR}/shims)"
fi

CMD="${1:-}"; shift || true
case "${CMD}" in
  gen)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/gen_golden.py" "$@"
    ;;
  compare)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/compare.py" "$@"
    ;;
  selfcheck)
    # Tier 0 bedrock: run hand-computed fixtures against the math-spec reference.
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/ref_w4a8.py" "$@"
    ;;
  verify-materialize)
    # ② materialize (quant_swap) 对齐 Tier 0: q bit-exact + pack/transdata round-trip.
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_materialize.py" "$@"
    ;;
  verify-quant-matmul)
    # ③ cann_lm::dynamic_quant + quant_matmul + QuantLinear 对齐 Tier 0.
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_quant_matmul.py" "$@"
    ;;
  *)
    echo "usage: $0 {gen|compare|selfcheck} [args...]" >&2
    exit 1
    ;;
esac
