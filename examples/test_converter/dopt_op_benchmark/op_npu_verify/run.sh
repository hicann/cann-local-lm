#!/usr/bin/env bash
# 算子级 NPU 验证 — 编排骨 3 env (① export → ② OM 编译 → ③ NPU 跑 → CPU compare)
# 用法: bash run.sh <op_name> [target_soc]   (target_soc: Ascend910B / KirinX90)
# 例:   bash run.sh model.layers.0.self_attn.q_proj Ascend910B
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OP_NAME="${1:?usage: run.sh <op_name> [target_soc]}"
TARGET_SOC="${2:-Ascend910B}"
EPAIR_DIR="${SCRIPT_DIR}/epair_out_$(echo "${TARGET_SOC}" | tr A-Z a-z)"
CONV_PY="${CONV_PY:-${PROJECT_ROOT}/deps/torch-delegate/.venv/bin/python}"
NPU_PY="${NPU_PY:-/opt/buildtools/python-3.12.9/bin/python3}"

echo "=== [1/4] export epair (converter env ①) ==="
source "${PROJECT_ROOT}/scripts/set_env_export.sh"
"${CONV_PY}" "${SCRIPT_DIR}/build_epair.py" --op-name "${OP_NAME}" \
    --quant-dir "${DOPT_QUANT_DIR:?set DOPT_QUANT_DIR}" --output-dir "${EPAIR_DIR}" --target-soc "${TARGET_SOC}"

echo "=== [2/4] compile OM (CANN env ②) ==="
source "${PROJECT_ROOT}/scripts/set_env_cann.sh"
bash "${SCRIPT_DIR}/compile_om.sh" "${EPAIR_DIR}"

echo "=== [3/4] run NPU (env ③, system python + torch_npu) ==="
source "${PROJECT_ROOT}/scripts/set_env_cann.sh" --npu
"${NPU_PY}" "${SCRIPT_DIR}/run_npu.py" --om "${EPAIR_DIR}/op_model.om" --op-name "${OP_NAME}"

echo "=== [4/4] compare (CPU, 余弦) ==="
"${NPU_PY}" "${SCRIPT_DIR}/verify_op.py" --op-name "${OP_NAME}" \
    --golden-dir "${BENCH_DIR}/golden" --npu-output "${SCRIPT_DIR}/npu_output.pt"
