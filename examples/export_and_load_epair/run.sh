#!/usr/bin/env bash
#
# run.sh — CANN-LM epair 导出与加载一键式验证脚本
#
# 前置条件：
#   1. source /home/developer/Ascend/cann/set_env.sh
#   2. venv（由 torch-delegate-harmony 创建）：/mnt/workspace/torch-delegate-harmony/.venv/
#   3. torch_delegate whl 已安装到 venv 中
#
# 用法：
#   bash run.sh
#
# 一键执行命令（含 CANN 环境初始化）：
#   source /home/developer/Ascend/cann/set_env.sh && \
#     export TORCH_DELEGATE_HOME="/mnt/workspace/torch-delegate-harmony" && \
#     bash /mnt/workspace/project/CANN-LM_5883/examples/export_and_load_example/run.sh
# 日志打屏export ASCEND_SLOG_PRINT_TO_STDOUT=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
PHASE=0
TOTAL=5

phase() {
  PHASE=$((PHASE + 1))
  echo ""
  echo "========================================"
  echo " [${PHASE}/${TOTAL}] $1"
  echo "========================================"
}

check() {
  echo "  [✓] $1"
}

fail() {
  echo "  [✗] $1"
  exit 1
}

# =========================================================
# Phase 1: 检查环境
# =========================================================
phase "检查环境"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  fail "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
fi
check "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

_VENV_PY="${TORCH_DELEGATE_HOME}/.venv/bin/python3"
if [[ -x "${_VENV_PY}" ]]; then
  PYTHON="${_VENV_PY}"
else
  # 尝试从 PATH 找 python3
  PYTHON="python3"
fi
if ! command -v "${PYTHON}" &> /dev/null; then
  fail "python3 未找到"
fi
check "python3 $("${PYTHON}" --version 2>&1 | awk '{print $2}')"

if ! command -v cmake &> /dev/null; then
  fail "cmake 未找到"
fi
check "cmake $(cmake --version | head -1 | awk '{print $3}')"

CSRC_DIR="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${CSRC_DIR}/build/libtorch_delegate_backend.so"
if [[ ! -f "${BACKEND_LIB}" ]]; then
  echo "  [*] 后端库未构建，将在 Phase 2 中编译"
else
  check "libtorch_delegate_backend.so (prebuilt)"
fi

_TORCHAIR_STUBS="$(realpath "${TORCH_DELEGATE_HOME}/../torchair/build/stubs" 2>/dev/null || true)"
if [[ -d "${_TORCHAIR_STUBS:-}" ]]; then
  check "torchair stubs: ${_TORCHAIR_STUBS}"
fi

# =========================================================
# Phase 2: 编译 C++ 后端库
# =========================================================
phase "编译 CANN-LM 后端库"

if [[ ! -f "${BACKEND_LIB}" ]]; then
  echo "  Building backend library from ${CSRC_DIR}..."
  cd "${CSRC_DIR}"
  bash build_csrc.sh
  check "libtorch_delegate_backend.so built"
else
  check "已存在，跳过"
fi

# =========================================================
# Phase 3: 导出示例模型
# =========================================================
phase "导出示例模型 (epair 目录)"

# 设置 PYTHONPATH 使 cann_lm 包可导入
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 禁用 fallback codegen（以避免依赖 torch_npu）
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# torchair stubs（用于无 NPU 硬件时 Python 侧初始化通过）
if [[ -d "${_TORCHAIR_STUBS:-}" ]]; then
  export LD_LIBRARY_PATH="${_TORCHAIR_STUBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${ROOT_DIR}"
"${PYTHON}" "${ROOT_DIR}/export_and_save_epair.py"

EPDIR="${ROOT_DIR}/epair_model"
if [[ -f "${EPDIR}/ep_json_with_ascend_ir.json" ]]; then
  check "epair_model 目录已生成 (含 ascend IR 标注)"
else
  fail "epair_model 未正确生成"
fi

# =========================================================
# Phase 4: 编译 C++ 示例二进制
# =========================================================
phase "编译 C++ 示例"

_TORCH_PKG_DIR="$("${PYTHON}" -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 TORCH_PKG_DIR"

cd "${ROOT_DIR}"
rm -rf build
cmake -S . -B build \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" \
  -DTORCH_PKG_DIR="${_TORCH_PKG_DIR}"
cmake --build build -j"$(nproc)"

if [[ -f "${ROOT_DIR}/build/export_and_load_example" ]]; then
  check "build/export_and_load_example"
else
  fail "编译失败"
fi

# =========================================================
# Phase 5: 验证
# =========================================================
phase "Delegate 加载验证"

cd "${ROOT_DIR}"
# C++ 二进制运行时需要 CANN + torch 库
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${_TORCH_PKG_DIR}/lib:${LD_LIBRARY_PATH:-}"
_TORCH_LIBS="$(dirname "${_TORCH_PKG_DIR}")/torch.libs"
if [[ -d "${_TORCH_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIBS}:${LD_LIBRARY_PATH}"
fi
if [[ -d "${_TORCHAIR_STUBS:-}" ]]; then
  export LD_LIBRARY_PATH="${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH}"
fi

"${PYTHON}" "${ROOT_DIR}/compare_single_op_vs_delegate.py"

echo ""
echo "========================================"
echo " 验证通过"
echo "========================================"
