#!/usr/bin/env bash
#
# run.sh — CANN-LM epair 导出 + C++ 加载（main.cpp 场景）一键式验证
#
# 支持不同模型：bash run.sh [simple|addrmsnorm] （默认 simple）
#
# 前置：source /home/developer/Ascend/cann/set_env.sh
# 用法：  bash run.sh [simple|addrmsnorm]   (默认 simple)
#         SKIP_OM_BUILD=0 bash run.sh simple   # 编译 OM (CPU 侧, 需算子对 SOC_VERSION 存在; 默认 1=只 load+GetGEGraph+ONNX)
#
set -euo pipefail

MODEL="${1:-simple}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
TORCHAIR_FULL="${TORCHAIR_FULL:-/mnt/workspace/torchair}"
# torch_delegate + torchair 装在 torch-delegate venv (torch 2.9.0+cpu)
PYTHON="${TORCH_DELEGATE_HOME}/.venv/bin/python3"
_VENV_PY="${PYTHON}"   # 取 TORCH_PKG_DIR（匹配 backend 构建）
CSRC_DIR="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${PROJECT_ROOT}/build/libtorch_delegate_backend.so"
STUBS_LIB="${TORCHAIR_FULL}/build/stubs"
PHASE=0

phase() { PHASE=$((PHASE + 1)); echo ""; echo "==== [${PHASE}] $1 ===="; }
ok()   { echo "  [✓] $1"; }
warn() { echo "  [!] $1"; }
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
# --- CANN ---
[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "ASCEND_HOME_PATH 未设置，请先 source /home/developer/Ascend/cann/set_env.sh"
ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
[[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" ]] || warn "libascendcl.so 未找到 (CANN lib64?)"
# --- NPU (信息项; OM 编译不需 NPU, 仅 OM 执行/出字才需要) ---
if command -v npu-smi &>/dev/null; then
  _NPU=$(npu-smi info -l 2>/dev/null | grep -oE 'Ascend[0-9]+' | head -1 || true)
  ok "NPU: ${_NPU:-可见} (本例仅编译 OM 不执行; NPU 非必需)"
else
  warn "npu-smi 不可用 (无 NPU); OM 编译仍可跑 (CPU 侧, 需算子在 CANN 中)"
fi
# --- SOC_VERSION (OM 编译关键: 需算子对该 Soc 存在; 默认本机 Ascend910_9362) ---
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
ok "SOC_VERSION=${SOC_VERSION} (OM 编译用; 错误 Soc 会因算子缺失失败)"
# --- cmake ---
command -v cmake &>/dev/null || die "cmake 未找到"
ok "cmake $(cmake --version | head -1 | awk '{print $3}')"
# --- venv python ---
[[ -x "${PYTHON}" ]] || die "venv python 不存在: ${PYTHON} (检查 TORCH_DELEGATE_HOME=${TORCH_DELEGATE_HOME})"
ok "venv python: ${PYTHON}"
# --- venv 内 torch / torchair / torch_delegate ---
"${PYTHON}" - <<'PY' 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torchair", "torch_delegate"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m} {'importable' if u.find_spec(m) else 'MISSING'}")
PY
# --- stubs (converter CPU 环境需要) ---
[[ -d "${STUBS_LIB}" ]] && ok "stubs: ${STUBS_LIB}" || warn "stubs 目录缺失: ${STUBS_LIB} (converter export 会失败)"
# --- backend lib (C++ 加载需要) ---
[[ -f "${BACKEND_LIB}" ]] && ok "backend lib (prebuilt): ${BACKEND_LIB##*/}" || warn "backend 待构建 (本脚本会自动 build_cpp.sh)"
# --- TORCH_PKG_DIR (C++ 链接 torch) ---
TORCH_PKG_DIR="$("${_VENV_PY}" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)" || die "无法从 venv 获取 torch 路径"
ok "TORCH_PKG_DIR=${TORCH_PKG_DIR}"

# =========================================================
phase "编译 CANN-LM C++ 后端库 (libtorch_delegate_backend.so)"
# =========================================================
if [[ ! -f "${BACKEND_LIB}" ]]; then
  bash "${PROJECT_ROOT}/scripts/cann_lm_compile/build_cpp.sh" --target torch_delegate_backend --no-tests --no-om || die "build_cpp.sh 失败"
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
# libascend_hal.so 在 CANN devlib/ 或 build/stub_lib/（build_cpp.sh 放的 stub），
# 不在 lib64/。缺它则 libtorch_delegate_backend.so 加载失败 → EpairModelLoader 符号不可用。
_BACKEND_STUB_LIB="${PROJECT_ROOT}/build/stub_lib"
export LD_LIBRARY_PATH="${_PY_LIB}:${TORCHAIR_FULL}/build/stubs:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib:${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:${_BACKEND_STUB_LIB:+${_BACKEND_STUB_LIB}}:${LD_LIBRARY_PATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 打开 GE 详细日志（定位 not registered 等）
# SKIP_OM_BUILD 值判断: 默认 1=只 load+GetGEGraph+ONNX (本例主用途); 设 0=编译 OM (CPU 侧, 需算子对 SOC_VERSION 存在, 不需 NPU)
export SKIP_OM_BUILD="${SKIP_OM_BUILD:-1}"

"${ROOT_DIR}/build/export_and_load_example" \
  "${EPAIR_DIR}/ep_json_with_ascend_ir.json" \
  "${EPAIR_DIR}/weight_info.json" \
  "${MODEL}"

echo ""
echo "==== 验证通过 (${MODEL}) ===="
