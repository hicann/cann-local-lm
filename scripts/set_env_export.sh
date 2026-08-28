#!/bin/bash
# set_env_export.sh — ① converter env (导出 epair + 量化)
#   CPU venv + torchair stubs (不装 torch_npu, 走 stub 路径); 用于 torch.export + torchair GE 转换
#
# 用法: source scripts/set_env_export.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh"   # 公共部分 (CANN/PYTHONPATH/通用变量)

export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # GE 详细日志 (定位 not registered 等)
# stubs 必须在 CANN lib64 前 (否则 aclrtSetDevice → 107001, 无设备)
export LD_LIBRARY_PATH="${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export CONVERTER_PYTHON="${TORCH_DELEGATE_HOME}/.venv/bin/python"

echo "[set_env] ① export (converter venv, CPU + torchair stubs, 不装 torch_npu)"
echo "  venv python : ${CONVERTER_PYTHON}"
echo "  dopt python : ${DOPT_PYTHON} (subprocess 用, --quant-method gptq 触发)"
echo "  stubs       : ${TORCHAIR_ROOT}/build/stubs"
