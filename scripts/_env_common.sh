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
    echo "  之后: source scripts/set_env.sh (converter+compile 统一 env)" >&2
    return 1 2>/dev/null || exit 1
fi

# 阻止 torch autoload npu backend (converter venv 不装 torch_npu; NPU 由 torch_npu 显式 import)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# dopt subprocess 用系统 Python 路径
export DOPT_PYTHON="${DOPT_PYTHON}"

set +u
