#!/usr/bin/env bash
#
# run.sh — CANN-LM epair 导出 + C++ 加载（main.cpp 场景）一键式验证
#
# 支持不同模型：bash run.sh [simple|addrmsnorm] （默认 simple）
#
# 前置：source /home/developer/Ascend/cann-9.1.0/set_env.sh
# 用法：  bash run.sh addrmsnorm
#
set -euo pipefail

MODEL="${1:-simple}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
TORCHAIR_FULL="${TORCHAIR_FULL:-/mnt/workspace/torchair-full}"
# torch_delegate + torchair 装在系统 python3；.venv 只用于 C++ 链接的 torch 路径
PYTHON="python3"
_VENV_PY="${TORCH_DELEGATE_HOME}/.venv/bin/python3"   # 仅取 TORCH_PKG_DIR（匹配 backend 构建）
CSRC_DIR="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${CSRC_DIR}/build/libtorch_delegate_backend.so"
PHASE=0

phase() { PHASE=$((PHASE + 1)); echo ""; echo "==== [${PHASE}] $1 ===="; }
ok()   { echo "  [✓] $1"; }
die()  { echo "  [✗] $1"; exit 1; }

# ── 模型注册：每个模型给出导出命令 + epair 目录 ──
case "${MODEL}" in
  simple)
    EXPORT_CMD=("${PYTHON}" "${ROOT_DIR}/export_and_save_epair.py")
    COMPILE_CMD=()
    EPAIR_DIR="${ROOT_DIR}/epair_model"
    ;;
  addrmsnorm)
    GEN="${ROOT_DIR}/export_addrmsnorm_model/generate"
    EXPORT_CMD=("${PYTHON}" "${ROOT_DIR}/export_addrmsnorm_model/export_addrmsnorm_model.py" \
                  export --output "${GEN}/addrmsnorm.pt2" --device cpu)
    COMPILE_CMD=("${PYTHON}" "${ROOT_DIR}/export_addrmsnorm_model/export_addrmsnorm_model.py" \
                  compile --ep-path "${GEN}/addrmsnorm.pt2" --output-dir "${GEN}/addrmsnorm_epair")
    EPAIR_DIR="${GEN}/addrmsnorm_epair"
    ;;
  *)
    die "unknown model '${MODEL}', supported: simple | addrmsnorm"
    ;;
esac

# =========================================================
phase "环境检查 (${MODEL})"
# =========================================================
[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
command -v cmake &>/dev/null || die "cmake 未找到"
ok "cmake $(cmake --version | head -1 | awk '{print $3}')"
[[ -f "${BACKEND_LIB}" ]] && ok "backend lib (prebuilt)" || echo "  [*] backend 待构建"

# =========================================================
phase "编译 CANN-LM C++ 后端库 (libtorch_delegate_backend.so)"
# =========================================================
if [[ ! -f "${BACKEND_LIB}" ]]; then
  (cd "${CSRC_DIR}" && bash build_csrc.sh) || die "build_csrc.sh 失败"
fi
ok "${BACKEND_LIB##*/}"

# =========================================================
phase "导出模型 → epair 目录"
# =========================================================
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export LD_LIBRARY_PATH="${TORCHAIR_FULL}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

"${EXPORT_CMD[@]}"
if [[ ${#COMPILE_CMD[@]} -gt 0 ]]; then
  "${COMPILE_CMD[@]}"
fi
[[ -f "${EPAIR_DIR}/ep_json_with_ascend_ir.json" ]] || die "epair 未生成: ${EPAIR_DIR}"
ok "epair: ${EPAIR_DIR}"

# =========================================================
phase "编译 C++ 示例 (export_and_load_example)"
# =========================================================
TORCH_PKG_DIR="$("${_VENV_PY}" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)" || die "无法获取 torch 路径"
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" -DTORCH_PKG_DIR="${TORCH_PKG_DIR}" >/dev/null
cmake --build "${ROOT_DIR}/build" -j"$(nproc)" >/dev/null || die "C++ 编译失败"
ok "build/export_and_load_example"

# =========================================================
phase "C++ 加载验证 (main.cpp 场景: load→GetGEGraph→DumpToFile→编译→.om)"
# =========================================================
_PY_LIB="$("${PYTHON}" -c 'import sys;print(sys.base_prefix)')/lib"
# libascend_hal.so 在 CANN devlib/ 或 csrc/build/stub_lib/（build_csrc.sh 放的 stub），
# 不在 lib64/。缺它则 libtorch_delegate_backend.so 加载失败 → EpairModelLoader 符号不可用。
_BACKEND_STUB_LIB="${CSRC_DIR}/build/stub_lib"
export LD_LIBRARY_PATH="${_PY_LIB}:${TORCHAIR_FULL}/build/stubs:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib:${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:${_BACKEND_STUB_LIB:+${_BACKEND_STUB_LIB}}:${LD_LIBRARY_PATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 打开 GE 详细日志（定位 not registered 等）

"${ROOT_DIR}/build/export_and_load_example" \
  "${EPAIR_DIR}/ep_json_with_ascend_ir.json" \
  "${EPAIR_DIR}/weight_info.json" \
  "${MODEL}"

echo ""
echo "==== 验证通过 (${MODEL}) ===="
