#!/usr/bin/env bash
# check_env.sh — test_converter 共享环境探测 + 分叉 (NPU/CPU 双模)
#
# 功能: NPU 探测→NPU_MODE; CANN 自动探测+兜底 source; 路径自定位派生;
#       ensure 共享依赖 (backend/third_party/stubs); converter LD 分叉 (NPU 真 CANN / CPU stubs-first)
#
# 用法 (各用例 run.sh 顶部):
#   source "$(dirname "$0")/../common/check_env.sh"
#   # 之后: $NPU_MODE, $CONVERTER_PYTHON, $STUBS_DIR, $BACKEND_LIB, $PROJECT_ROOT, LD 已设
#
set -uo pipefail
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_COMMON_DIR}/../../.." && pwd)}"

# ── NPU 探测 (npu-smi 依赖 driver lib; CANN set_env 在第 26 行才 source, 此处先前置 driver lib 让 npu-smi 可执行) ──
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
if command -v npu-smi &>/dev/null && npu-smi info &>/dev/null 2>&1; then
  export NPU_MODE=1
  _NPU_NAME="$(npu-smi info -l 2>/dev/null | grep -oE 'Ascend[0-9]+' | head -1 || true)"
  echo "[env] NPU_MODE=1 (NPU 可见: ${_NPU_NAME:- Ascend}) → converter 真 CANN, ③ 可跑"
else
  export NPU_MODE=0
  echo "[env] NPU_MODE=0 (无 NPU) → converter stubs-first, ③ 跳过"
fi

# ── CANN 自动探测 + 兜底 source (ASCEND_HOME_PATH 未设时遍历候选) ──
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for _c in /home/developer/Ascend/cann /usr/local/Ascend/cann "${HOME}/Ascend/cann"; do
    if [[ -f "${_c}/set_env.sh" ]]; then
      echo "[env] source CANN: ${_c}/set_env.sh"
      set +u; source "${_c}/set_env.sh"; set -u   # CANN set_env.sh 用 $VAR 无默认, 临时关 -u 防 unbound
      break
    fi
  done
fi
[[ -n "${ASCEND_HOME_PATH:-}" ]] || { echo "[env] ✗ ASCEND_HOME_PATH 未设 (CANN 未安装?)"; return 1 2>/dev/null || exit 1; }
echo "[env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

# ── 路径自定位派生 (与 build.conf 一致, 不硬编码) ──
export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-${PROJECT_ROOT}/deps/torch-delegate}"
export TORCHAIR_ROOT="${TORCHAIR_ROOT:-${PROJECT_ROOT}/deps/torchair}"
export CONVERTER_PYTHON="${CONVERTER_PYTHON:-${TORCH_DELEGATE_HOME}/.venv/bin/python}"
export STUBS_DIR="${TORCHAIR_ROOT}/build/stubs"
export BACKEND_LIB="${PROJECT_ROOT}/build/libtorch_delegate_backend.so"
export DOPT_PYTHON="${DOPT_PYTHON:-}"  # 装 torch_npu 的系统 python; 由 scripts/cann_lm_compile/_env_common.sh 自动探测 (source set_env_cann.sh --npu 后已设)
export SOC_VERSION="${SOC_VERSION:-$([[ $NPU_MODE == 1 ]] && echo Ascend910_9362 || echo kirinx90)}"

# ── ensure 共享依赖 (幂等, 缺失自动建) ──
source "${_COMMON_DIR}/ensure_backend.sh" && ensure_backend || return 1 2>/dev/null || exit 1
source "${_COMMON_DIR}/ensure_third_party.sh" && ensure_third_party || return 1 2>/dev/null || exit 1

# ── converter 运行 env (通用, 各用例复用) ──
# lite_export_context (LiteConfig) 绕过 aclrtSetDevice (Python 级 fake torch_npu + _get_device_id=lambda:0),
# 不需 C 级 stubs; torchair _torchair.so 闭包 (libge_runner/libascendcl/libhccl/libruntime_common) 均 NEEDED
# liberror_manager, 不需 devlib 的 libascend_hal.so (后者只被 DVPP/dump/ml 库引用).
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── LD: 真 CANN lib64 优先 (liberror_manager 取 host 版, 避免 devlib/aarch64 device 版干扰致
#   import torchair 失败; cann set_env.sh 先设 devlib:lib64, 此处强制 lib64 到最前覆盖). lite 模式无 stubs. ──
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# A5 修复: source CANN 后 libpython 可能被解析为系统旧版致 _ctypes 符号缺失
# (_PyErr_SetLocaleString); 前置 converter python 的 LIBDIR (含正确 libpython), 幂等 prepend
if [[ -x "${CONVERTER_PYTHON:-}" ]]; then
  _PY_LIBDIR="$("${CONVERTER_PYTHON}" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')" 2>/dev/null || true)"
  if [[ -n "${_PY_LIBDIR}" && -d "${_PY_LIBDIR}" ]]; then
    export LD_LIBRARY_PATH="${_PY_LIBDIR}:${LD_LIBRARY_PATH}"
    echo "[env] libpython 前置: ${_PY_LIBDIR} (A5 修复 _ctypes 符号)"
  fi
  unset _PY_LIBDIR
fi

unset _COMMON_DIR _NPU_NAME _c
echo "[env] 就绪 (NPU_MODE=$NPU_MODE, SOC=$SOC_VERSION, lite_export_context 无 stubs)"
