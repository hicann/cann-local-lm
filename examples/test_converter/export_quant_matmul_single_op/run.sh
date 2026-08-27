#!/usr/bin/env bash
#
# run.sh — 单算子 QuantBatchMatmulV4 W4A8 OM 编译测试
#
# 流程: Python 导出 (CPU stub) → C++ EpairModelLoader + aclgrphBuildModel (NPU) → .om
# 复用 examples/test_converter/export_and_load_epair 的 main.cpp + reg_register.h (已注册 QuantBatchMatmulV4)
#
# 前置: source /home/developer/Ascend/cann/set_env.sh
# 用法:  bash run.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
TORCHAIR_FULL="${TORCHAIR_FULL:-/mnt/workspace/torchair}"
PYTHON="${TORCH_DELEGATE_HOME}/.venv/bin/python3"
EPAIR_DIR="${ROOT_DIR}/epair_model"
# 复用 export_and_load_epair 的 C++ 二进制 (main.cpp + reg_register.h 含 QuantBatchMatmulV4)
LOAD_EXAMPLE_DIR="${PROJECT_ROOT}/examples/test_converter/export_and_load_epair"
LOAD_BINARY="${LOAD_EXAMPLE_DIR}/build/export_and_load_example"
CSRC_DIR="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${CSRC_DIR}/build/libtorch_delegate_backend.so"

phase() { echo ""; echo "==== $1 ===="; }
ok()   { echo "  [✓] $1"; }
die()  { echo "  [✗] $1"; exit 1; }

# ── 1. 环境检查 ──
phase "环境检查"
[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
[[ -f "${BACKEND_LIB}" ]] || die "backend lib 不存在: ${BACKEND_LIB} (先 cd ${CSRC_DIR} && bash build_csrc.sh)"
ok "backend lib"

# ── 2. Python 导出单算子 → epair (CPU stub) ──
phase "Python 导出 QuantMatmulSingleOp → epair"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export LD_LIBRARY_PATH="${TORCHAIR_FULL}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
"${PYTHON}" "${ROOT_DIR}/export_and_save_epair.py"
[[ -f "${EPAIR_DIR}/ep_json_with_ascend_ir.json" ]] || die "ep_json 未生成"
ok "epair: ${EPAIR_DIR}"

# ── 3. 构建 C++ 二进制 (复用 export_and_load_epair) ──
phase "构建 C++ 二进制 (export_and_load_example)"
TORCH_PKG_DIR="$("${PYTHON}" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)" \
  || die "无法获取 torch 路径"
cmake -S "${LOAD_EXAMPLE_DIR}" -B "${LOAD_EXAMPLE_DIR}/build" \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" -DTORCH_PKG_DIR="${TORCH_PKG_DIR}" >/dev/null
cmake --build "${LOAD_EXAMPLE_DIR}/build" -j"$(nproc)" >/dev/null || die "C++ 编译失败"
ok "binary: ${LOAD_BINARY}"

# ── 4. C++ 加载 + OM 编译 (NPU) ──
phase "C++ OM 编译 (EpairModelLoader → GetGEGraph → aclgrphBuildModel)"
_PY_LIB="$("${PYTHON}" -c 'import sys;print(sys.base_prefix)')/lib"
_BACKEND_STUB_LIB="${CSRC_DIR}/build/stub_lib"
export LD_LIBRARY_PATH="${_PY_LIB}:${TORCHAIR_FULL}/build/stubs:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib:${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:${_BACKEND_STUB_LIB:+${_BACKEND_STUB_LIB}}:${LD_LIBRARY_PATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# NPU 上跑 OM 编译; CPU-only 时设 SKIP_OM_BUILD=1 只做 GetGEGraph
export SKIP_OM_BUILD="${SKIP_OM_BUILD:-0}"

"${LOAD_BINARY}" \
  "${EPAIR_DIR}/ep_json_with_ascend_ir.json" \
  "${EPAIR_DIR}/weight_info.json" \
  "quant_matmul_single_op"

echo ""
echo "==== 单算子 OM 编译测试完成 ===="
echo "产物: quant_matmul_single_op.om (若 OM 编译成功)"
