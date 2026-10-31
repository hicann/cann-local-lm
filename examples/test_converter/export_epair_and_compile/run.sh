#!/usr/bin/env bash
#
# run.sh — epair 导出 + OM 编译 (cann_lm_compile) 一键式验证
#
# 前置: .run 包已安装到 /opt/cann-device-lm/, cann_lm whl 已装进 P1 venv
#
# 流程: [1] 环境检查 → [2] 导出 epair → [3] cann_lm_compile 编译 OM
#
# 用法:  bash run.sh [simple|addrmsnorm]   (默认 simple)
#
set -euo pipefail

MODEL="${1:-simple}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 环境设置 (从已安装的 .run 包) ──
CANN_DEVICE_LM="${CANN_DEVICE_LM:-/opt/cann-device-lm}"
source "$HOME/Ascend/cann/set_env.sh"
source "${CANN_DEVICE_LM}/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CONVERTER_PYTHON="${CONVERTER_PYTHON:-/path/to/venv/bin/python}"

PYTHON="${CONVERTER_PYTHON}"
COMPILE_BIN="${CANN_DEVICE_LM}/cann_lm_compile"

phase() { echo ""; echo "==== $1 ===="; }
ok()   { echo "  [✓] $1"; }
warn() { echo "  [!] $1"; }
die()  { echo "  [✗] $1"; exit 1; }

# ── 模型注册 ──
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
ok "CANN ${ASCEND_HOME_PATH}"
ok "SOC_VERSION=${SOC_VERSION}"
[[ -x "${PYTHON}" ]] || die "CONVERTER_PYTHON 不存在: ${PYTHON} (export CONVERTER_PYTHON=/path/to/venv/bin/python)"
ok "python: ${PYTHON}"
"${PYTHON}" - <<'PY' 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torchair", "torch_delegate", "cann_lm"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m} {'importable' if u.find_spec(m) else 'MISSING'}")
PY
[[ -x "${COMPILE_BIN}" ]] || die "cann_lm_compile 不存在: ${COMPILE_BIN} (先装 .run 包)"
ok "cann_lm_compile: ${COMPILE_BIN}"

# =========================================================
phase "导出模型 → epair"
# =========================================================
"${EXPORT_CMD[@]}"
if [[ ${#COMPILE_CMD[@]} -gt 0 ]]; then
  "${COMPILE_CMD[@]}"
fi
[[ -f "${EPAIR_DIR}/ep_json_with_ascend_ir.json" ]] || die "epair 未生成: ${EPAIR_DIR}"
ok "epair: ${EPAIR_DIR}"

# =========================================================
phase "OM 编译 (cann_lm_compile: epair → OM)"
# =========================================================
# cann_lm_compile 用 quick_exit 规避 GE 库静态析构, 正常退出码 0
"${COMPILE_BIN}" \
  "${EPAIR_DIR}/ep_json_with_ascend_ir.json" \
  "${EPAIR_DIR}/weight_info.json" \
  "${EPAIR_DIR}/${MODEL}" 2>&1 | grep -vE "^\[INFO\]" || true

OM_FILE="${EPAIR_DIR}/${MODEL}.om"
if [[ -f "${OM_FILE}" ]]; then
  ok "${MODEL}.om ($(ls -lh "${OM_FILE}" | awk '{print $5}'))"
else
  die "${OM_FILE} 未产出 (见上方日志)"
fi

echo ""
echo "==== 验证通过 (${MODEL}) ===="
