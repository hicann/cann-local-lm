#!/usr/bin/env bash
# check_env.sh — om_npu_verify 用例环境检查
#
# 覆盖三套子流程所需环境：
#   1. 导出 epair        — converter venv (CPU + torchair stubs)
#   2. OM 编译           — 真 CANN (CPU 侧, 不需 NPU) + cmake + build/cann_lm_compile
#   3. OM 运行/出字       — 系统 Python + torch_npu + acl + NPU
#
# 环境前置：本脚本 source ① converter 基底 (set_env_export.sh) 拿 CONVERTER_PYTHON /
#   TORCH_DELEGATE_HOME / stubs / PYTHONPATH / DOPT_PYTHON; ②③ 真 CANN 基底与 ① 的
#   LD 互斥 (stubs-first ↔ torch_npu), 运行 ②③ 时另 source set_env_cann.sh, 此处仅检查
#   共用项. HF 模型路径由调用方必填 export (增量, 不与基底冲突).
#
# 用法:
#   export GEMMA4_HF_PATH=<gemma4 model dir>
#   export QWEN3_HF_PATH=<qwen3 model dir>
#   export QWEN3_5_HF_PATH=<qwen3.5 model dir>
#   bash check_env.sh
#
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SELF_DIR}/../../../.." && pwd)"   # CANN-LM_5883
EXPORT_DIR="$(cd "${SELF_DIR}/.." && pwd)"               # examples/test_converter/export_llm_model

# source ① converter 基底 (设 CONVERTER_PYTHON / TORCH_DELEGATE_HOME / stubs LD / PYTHONPATH / DOPT_PYTHON)
# 注: 基底脚本内部也定义 SCRIPT_DIR/PROJECT_ROOT, 故本脚本用 SELF_DIR 避免被覆盖
source "${PROJECT_ROOT}/scripts/set_env_export.sh" >/dev/null 2>&1 || true

# 基底提供的变量 (不再自派生 LD_LIBRARY_PATH / PYTHONPATH)
CSRC="${PROJECT_ROOT}/cann_lm/converter/export/csrc"
BACKEND_LIB="${PROJECT_ROOT}/build/libtorch_delegate_backend.so"
BUILD_BIN="${PROJECT_ROOT}/build/cann_lm_compile"
HF_GEMMA4="${GEMMA4_HF_PATH:-}"
HF_QWEN3="${QWEN3_HF_PATH:-}"
HF_QWEN35="${QWEN3_5_HF_PATH:-}"

pass=0; warn=0; fail=0
ok()   { echo "  [✓] $1"; pass=$((pass+1)); }
wn()   { echo "  [!] $1"; warn=$((warn+1)); }
ng()   { echo "  [✗] $1"; fail=$((fail+1)); }
hdr()  { echo ""; echo "== $1 =="; }

echo "################ om_npu_verify 环境检查 ################"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  EXPORT_DIR=${EXPORT_DIR}"

# ───────────── 公共: CANN (基底已 source) ─────────────
hdr "公共: CANN (① 基底 set_env_export.sh 已 source)"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"; else ng "ASCEND_HOME_PATH 未设置 (先 source CANN set_env.sh 或检查 set_env_export.sh)"; fi
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
hdr "Flow 1: 导出 epair (① 基底: CONVERTER_PYTHON + torchair stubs, CPU)"
if [[ -x "${CONVERTER_PYTHON:-}" ]]; then ok "CONVERTER_PYTHON: ${CONVERTER_PYTHON}"; else ng "CONVERTER_PYTHON 缺失 (set_env_export.sh 未正确 source)"; fi
"${CONVERTER_PYTHON:-python3}" - <<'PY' 2>/dev/null
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torchair","torch_delegate"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m}")
PY
[[ -d "${STUBS_DIR:-${TORCHAIR_ROOT:-}/build/stubs}" ]] && ok "stubs: ${STUBS_DIR:-${TORCHAIR_ROOT:-}/build/stubs}" || ng "stubs 目录缺失 (converter export 会失败)"

# ───────────── Flow 2: OM 编译 (真 CANN + cmake + C++ binary) ─────────────
# 注: ② 真 CANN 基底 (set_env_cann.sh) 与 ① LD 互斥, 运行 ② 时另 source; 此处检查共用项
hdr "Flow 2: OM 编译 (真 CANN, CPU 侧, 不需 NPU; 需算子在 CANN 对 Soc 存在)"
command -v cmake &>/dev/null && ok "cmake $(cmake --version | head -1 | awk '{print $3}')" || ng "cmake 未找到"
[[ -f "${BACKEND_LIB}" ]] && ok "backend lib: ${BACKEND_LIB##*/}" || ng "backend lib 缺失 (bash scripts/build.sh cpp --asan)"
_vtorch="$("${CONVERTER_PYTHON:-python3}" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
[[ -n "${_vtorch}" ]] && ok "TORCH_PKG_DIR=${_vtorch}" || ng "无法从 venv 取 torch 路径"
if [[ -x "${BUILD_BIN}" ]]; then ok "C++ binary: ${BUILD_BIN}"; else wn "C++ binary 未构建 (bash scripts/build.sh cpp)"; fi
if [[ -n "${SOC_VERSION:-}" ]]; then ok "SOC_VERSION=${SOC_VERSION}"; else wn "SOC_VERSION 未设 (② source set_env_cann.sh 后默认 Ascend910_9362; 错误 Soc → 算子缺失失败)"; fi
echo "  epair 产物 (OM 编译输入):"
for d in epair_gemma4 epair_gemma4_pd epair_qwen3_5_emb; do
  _j="${SELF_DIR}/${d}"
  if [[ -d "${_j}" ]]; then
    _n=$(find "${_j}" -maxdepth 1 -name '*ep_json_with_ascend_ir.json' | wc -l)
    _w=$(find "${_j}" -maxdepth 1 -name 'weight_info.json' | wc -l)
    [[ ${_n} -gt 0 && ${_w} -gt 0 ]] && ok "${d}/ (ep_json×${_n}, weight_info ✓)" || wn "${d}/ 不完整"
  else
    wn "${d}/ 不存在"
  fi
done

# ───────────── Flow 3: OM 运行/出字 (系统 Python + torch_npu + acl + NPU) ─────────────
hdr "Flow 3: OM 运行/出字 (基底 DOPT_PYTHON + torch_npu + acl + NPU)"
if [[ -x "${DOPT_PYTHON:-}" ]]; then ok "DOPT_PYTHON: ${DOPT_PYTHON}"; else ng "DOPT_PYTHON 缺失 (set_env_export.sh 未正确 source)"; fi
"${DOPT_PYTHON:-python3}" - <<'PY' 2>/dev/null
import torch, importlib.util as u
print(f"[✓] torch {torch.__version__}")
for m in ("torch_npu","acl"):
    print(f"[{'✓' if u.find_spec(m) else '✗'}] {m}")
PY
CANN_PY="${ASCEND_HOME_PATH:-}/python/site-packages"
[[ -d "${CANN_PY}" ]] && ok "CANN python site-packages: ${CANN_PY}" || ng "CANN_PY 缺失 (${CANN_PY})"
echo "  OM 文件 (OM 运行/出字输入; 由 run_e2e_pipeline.sh 或 run_*.py 经 env 指定):"
echo "    PREFILL_OM / DECODE_OM / GEMMA4_OM / QWEN3_OM / QWEN35_OM env 变量"
echo "  HF 模型 (eager 对比 + host embed gather 用; 必填 env):"
for kv in "GEMMA4_HF_PATH:${HF_GEMMA4}" "QWEN3_HF_PATH:${HF_QWEN3}" "QWEN3_5_HF_PATH:${HF_QWEN35}"; do
  _n="${kv%%:*}"; _v="${kv#*:}"
  [[ -n "${_v}" && -f "${_v}/config.json" ]] && ok "${_n}=${_v}" || wn "${_n} 未设或缺失 (export ${_n}=<model dir>)"
done

# ───────────── run 脚本 ─────────────
hdr "run 脚本"
echo "  $(ls ${SELF_DIR}/run_*.py 2>/dev/null | wc -l) 个: $(ls ${SELF_DIR}/run_*.py 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

# ───────────── 汇总 ─────────────
echo ""
echo "################ 汇总 ################"
echo "  ✓ pass=${pass}  [!] warn=${warn}  [✗] fail=${fail}"
[[ ${fail} -eq 0 ]] && echo "  → 无硬缺项 (warn 可按子流程忽略)" || echo "  → 有 ${fail} 项硬缺, 按需补齐"
echo "  (Flow1 导出 / Flow2 OM编译 / Flow3 OM运行出字 各自只需对应检查项通过即可)"
