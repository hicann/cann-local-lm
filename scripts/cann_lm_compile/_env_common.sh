# _env_common.sh — set_env 公共部分 (由 set_env.sh / set_env_cann.sh source, 不直接 source)
#
# 提供: source build.conf (版本 + 路径默认) + 校验 ASCEND_HOME_PATH (外部预置, 不自动 source) +
#        TORCH_DEVICE_BACKEND_AUTOLOAD=0 + DOPT_PYTHON (export)
# PYTHONPATH 不在此设: set_env.sh 反查 cann_lm 装位置注入 (自定位, 不硬编码 repo 根)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 版本 + 路径默认值 (build.conf 自定位 PROJECT_ROOT; 环境变量优先; 上游仓路径由 PROJECT_ROOT/deps 派生)
source "${SCRIPT_DIR}/build.conf"

# CANN: 不自动 source; 要求外部预置 ASCEND_HOME_PATH (对齐 torch-delegate 模式)
# 用户须先 source CANN set_env.sh (本机已装 cann-toolkit, 通常 0828); converter+compile 复用一套
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[set_env] 错误: ASCEND_HOME_PATH 未设。请先 source CANN 环境:" >&2
    echo "    source ${CANN_SET_ENV}  (或 source /path/to/Ascend/cann/set_env.sh)" >&2
    echo "  之后: source scripts/cann_lm_compile/set_env.sh (converter+compile 统一 env)" >&2
    return 1 2>/dev/null || exit 1
fi

# 阻止 torch autoload npu backend (converter venv 不装 torch_npu; NPU 由 torch_npu 显式 import)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# dopt subprocess 用系统 Python: 自动探测装了 torch_npu 的 python (③ NPU 阶段需).
# 优先级: 已 export 的 DOPT_PYTHON → 候选路径 + PATH python3 中能 import torch_npu 的.
# 换机器无需改: 只要新机有装 torch_npu 的 python 在候选路径或 PATH 即自动命中;
# converter P1 venv 的 python 无 torch_npu, 会被自动跳过不会误选.
_resolve_dopt_python() {
    local p plib
    # LD 构造: 候选 python 的 <prefix>/lib (libpython 在此; 不加会加载系统旧版 → site-packages 计算错 → import torch_npu 失败) + CANN lib64 (torch_npu .so 依赖) + 原 LD
    # ① 已显式设值且能 import torch_npu → 直接用 (用户覆盖最高优先)
    if [[ -n "${DOPT_PYTHON:-}" && -x "${DOPT_PYTHON}" ]]; then
        plib="$(dirname "$(dirname "${DOPT_PYTHON}")")/lib"; [[ -d "$plib" ]] || plib=""
        if LD_LIBRARY_PATH="${plib:+${plib}:}${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
           "${DOPT_PYTHON}" -c "import torch_npu" 2>/dev/null; then
            [[ -n "$plib" ]] && export LD_LIBRARY_PATH="${plib}:${LD_LIBRARY_PATH:-}"
            return 0
        fi
    fi
    # ② 候选探测: 常见路径 + PATH 里的 python3, 取第一个能 import torch_npu 的
    local candidates=(
        /opt/buildtools/python-3.12.9/bin/python3
        /opt/buildtools/Python-3.11.4/bin/python3
        /usr/local/bin/python3
        /usr/bin/python3
    )
    local in_path; in_path="$(command -v python3 2>/dev/null || true)"
    [[ -n "$in_path" ]] && candidates+=("$in_path")
    for p in "${candidates[@]}"; do
        [[ -x "$p" ]] || continue
        plib="$(dirname "$(dirname "$p")")/lib"; [[ -d "$plib" ]] || plib=""
        if LD_LIBRARY_PATH="${plib:+${plib}:}${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
           "$p" -c "import torch_npu" 2>/dev/null; then
            export DOPT_PYTHON="$p"
            [[ -n "$plib" ]] && export LD_LIBRARY_PATH="${plib}:${LD_LIBRARY_PATH:-}"
            return 0
        fi
    done
    return 1  # 未找到装 torch_npu 的 python (仅 ③ NPU 阶段需要; ①② 不阻塞)
}
if ! _resolve_dopt_python; then
    echo "[_env_common] DOPT_PYTHON: 未探测到装 torch_npu 的 python (仅 ③ NPU 阶段需要, ①② 不受影响)" >&2
fi

set +u
