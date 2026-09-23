#!/usr/bin/env bash
#
# run.sh — epair 导出 + OM 编译 (cann_lm_compile) 一键式验证
#
# 流程: [1] 环境检查 → [2] 导出 epair → [3] cann_lm_compile 编译 OM
# OM 编译统一走 cann_lm_compile (csrc/tools/compile_main.cc), 不再用 main.cpp
#
# 用法:  bash run.sh [simple|addrmsnorm]   (默认 simple)
#
set -euo pipefail

MODEL="${1:-simple}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/../common/check_env.sh"   # NPU 探测+CANN 自动 source+路径派生+ensure 依赖+LD
PYTHON="${CONVERTER_PYTHON}"
COMPILE_BIN="${PROJECT_ROOT}/build/cann_lm_compile"
PHASE=0

phase() { PHASE=$((PHASE + 1)); echo ""; echo "==== [${PHASE}] $1 ===="; }
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
phase "环境检查 (${MODEL}) [NPU_MODE=${NPU_MODE}]"
# =========================================================
ok "CANN ${ASCEND_HOME_PATH}"
if [[ "$NPU_MODE" == "1" ]]; then ok "NPU 可见"; else warn "无 NPU (CPU 交叉编译, lite 模式)"; fi
ok "SOC_VERSION=${SOC_VERSION}"
[[ -x "${PYTHON}" ]] || die "venv python 不存在: ${PYTHON}"
ok "venv python: ${PYTHON}"
"${PYTHON}" - <<'PY' 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torchair", "torch_delegate"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m} {'importable' if u.find_spec(m) else 'MISSING'}")
PY
ok "backend lib: ${BACKEND_LIB##*/}"
[[ -x "${COMPILE_BIN}" ]] || die "cann_lm_compile 未构建: ${COMPILE_BIN} (跑 bash scripts/build.sh cpp)"
ok "cann_lm_compile: ${COMPILE_BIN##*/}"

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
# cann_lm_compile 退出时 GE teardown 可能 abort, 用 || true 容错, 以 .om 存在判定成功
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
