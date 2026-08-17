#!/usr/bin/env bash
# check_env.sh — om_npu_verify 用例环境检查
#
# 覆盖三套子流程所需环境：
#   1. 导出 epair        — converter venv (CPU + torchair stubs)
#   2. OM 编译           — 真 CANN (CPU 侧, 不需 NPU) + cmake + ../build/export_llm_model
#   3. OM 运行/出字       — 系统 Python + torch_npu + acl + NPU
#
# 用法（先 source CANN 环境）:
#   source /home/developer/Ascend/cann/set_env.sh
#   bash check_env.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"          # CANN-LM_5883 (om_npu_verify 上 4 级)
EXPORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                # examples/export_llm_model

# 可被环境变量覆盖
VENV="${VENV:-/mnt/workspace/torch-delegate-harmony/.venv}"
TORCHAIR_FULL="${TORCHAIR_FULL:-/mnt/workspace/torchair}"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
SYS_PY="${SYS_PY:-/opt/buildtools/Python-3.11.4/bin/python3}"
CANN_PY="${CANN_PY:-${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.2.0}/python/site-packages}"
CSRC="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${CSRC}/build/libtorch_delegate_backend.so"
BUILD_BIN="${EXPORT_DIR}/build/export_llm_model"
STUBS="${TORCHAIR_FULL}/build/stubs"
HF_GEMMA4="${HF_GEMMA4:-/mnt/workspace/data/models/gemma-4-E2B-it}"
HF_QWEN3="${HF_QWEN3:-/mnt/workspace/data/models/qwen3-0.6B}"
HF_QWEN35="${HF_QWEN35:-/mnt/workspace/data/models/Qwen3.5-0.8B}"

pass=0; warn=0; fail=0
ok()   { echo "  [✓] $1"; pass=$((pass+1)); }
wn()   { echo "  [!] $1"; warn=$((warn+1)); }
ng()   { echo "  [✗] $1"; fail=$((fail+1)); }
hdr()  { echo ""; echo "== $1 =="; }

echo "################ om_npu_verify 环境检查 ################"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  EXPORT_DIR=${EXPORT_DIR}"

# ───────────── 公共 ─────────────
hdr "公共: CANN"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"; else ng "ASCEND_HOME_PATH 未设置 (先 source /home/developer/Ascend/cann/set_env.sh)"; fi
[[ -f "${ASCEND_HOME_PATH:-}/lib64/libascendcl.so" ]] && ok "CANN lib64/libascendcl.so" || ng "libascendcl.so 缺失"

hdr "公共: NPU (OM 运行/出字需要; OM 编译不需要)"
if command -v npu-smi &>/dev/null; then
  _npu=$(npu-smi info 2>/dev/null | grep -oE 'Ascend[0-9]+' | head -1 || true)
  _cnt=$(npu-smi info 2>/dev/null | grep -cE 'Ascend910[[:space:]]+\|' || true)
  ok "NPU: ${_npu:-可见} (芯片数 ${_cnt:-?})"
else
  wn "npu-smi 不可用 (无 NPU → OM 运行/出字不可跑; OM 编译仍可)"
fi

# ───────────── Flow 1: 导出 epair (converter venv + stubs) ─────────────
hdr "Flow 1: 导出 epair (converter venv + torchair stubs, CPU)"
if [[ -x "${VENV}/bin/python3" ]]; then ok "venv python: ${VENV}/bin/python3"; else ng "venv python 缺失 (${VENV})"; fi
"${VENV}/bin/python3" - <<'PY' 2>/dev/null
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torchair","torch_delegate"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m}")
PY
[[ -d "${STUBS}" ]] && ok "stubs: ${STUBS}" || ng "stubs 目录缺失 (${STUBS}; converter export 会失败)"

# ───────────── Flow 2: OM 编译 (真 CANN + cmake + C++ binary) ─────────────
hdr "Flow 2: OM 编译 (真 CANN, CPU 侧, 不需 NPU; 需算子在 CANN 对 Soc 存在)"
command -v cmake &>/dev/null && ok "cmake $(cmake --version | head -1 | awk '{print $3}')" || ng "cmake 未找到"
[[ -f "${BACKEND_LIB}" ]] && ok "backend lib: ${BACKEND_LIB##*/}" || ng "backend lib 缺失 (cd ${CSRC} && bash build_csrc.sh)"
_vtorch="$("${VENV}/bin/python3" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
[[ -n "${_vtorch}" ]] && ok "TORCH_PKG_DIR=${_vtorch}" || ng "无法从 venv 取 torch 路径"
if [[ -x "${BUILD_BIN}" ]]; then ok "C++ binary: ${BUILD_BIN}"; else wn "C++ binary 未构建 (cmake -S ${EXPORT_DIR} -B ${BUILD_BIN%/*}/ && cmake --build)"; fi
if [[ -n "${SOC_VERSION:-}" ]]; then ok "SOC_VERSION=${SOC_VERSION}"; else wn "SOC_VERSION 未设 (默认 Ascend910_9362; 错误 Soc → 算子缺失失败)"; fi
[[ -f /tmp/fusion_autoub_off.cfg ]] && ok "AUB off cfg: /tmp/fusion_autoub_off.cfg" || wn "fusion_autoub_off.cfg 缺失 (OM 编译需关 AUB)"
echo "  epair 产物 (OM 编译输入):"
for d in epair_gemma4 epair_gemma4_pd epair_qwen3_5_emb; do
  _j="${SCRIPT_DIR}/${d}"
  if [[ -d "${_j}" ]]; then
    _n=$(find "${_j}" -maxdepth 1 -name '*ep_json_with_ascend_ir.json' | wc -l)
    _w=$(find "${_j}" -maxdepth 1 -name 'weight_info.json' | wc -l)
    [[ ${_n} -gt 0 && ${_w} -gt 0 ]] && ok "${d}/ (ep_json×${_n}, weight_info ✓)" || wn "${d}/ 不完整"
  else
    wn "${d}/ 不存在"
  fi
done

# ───────────── Flow 3: OM 运行/出字 (系统 Python + torch_npu + acl + NPU) ─────────────
hdr "Flow 3: OM 运行/出字 (系统 Python + torch_npu + acl + NPU)"
if [[ -x "${SYS_PY}" ]]; then ok "系统 python: ${SYS_PY}"; else ng "系统 python 缺失 (${SYS_PY})"; fi
"${SYS_PY}" - <<'PY' 2>/dev/null
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torch_npu","acl"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m}")
PY
[[ -d "${CANN_PY}" ]] && ok "CANN python site-packages: ${CANN_PY}" || ng "CANN_PY 缺失 (${CANN_PY})"
echo "  OM 文件 (OM 运行/出字输入):"
for om in /tmp/opencode/gemma4_calib/gemma4_calib_noaub.om /tmp/opencode/gemma4_pd/prefill.om /tmp/opencode/gemma4_pd/decode.om /tmp/opencode/qwen3_calib/qwen3_full.om; do
  [[ -f "${om}" ]] && ok "${om}" || wn "${om##*/} 缺失"
done
echo "  HF 模型 (eager 对比 + host embed gather 用):"
for m in "${HF_GEMMA4}/config.json" "${HF_QWEN3}/config.json" "${HF_QWEN35}/config.json"; do
  [[ -f "${m}" ]] && ok "$(dirname ${m})" || wn "$(dirname ${m}) 缺失"
done

# ───────────── run 脚本 ─────────────
hdr "run 脚本"
echo "  $(ls ${SCRIPT_DIR}/run_*.py 2>/dev/null | wc -l) 个: $(ls ${SCRIPT_DIR}/run_*.py 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

# ───────────── 汇总 ─────────────
echo ""
echo "################ 汇总 ################"
echo "  ✓ pass=${pass}  [!] warn=${warn}  [✗] fail=${fail}"
[[ ${fail} -eq 0 ]] && echo "  → 无硬缺项 (warn 可按子流程忽略)" || echo "  → 有 ${fail} 项硬缺, 按需补齐"
echo "  (Flow1 导出 / Flow2 OM编译 / Flow3 OM运行出字 各自只需对应检查项通过即可)"
