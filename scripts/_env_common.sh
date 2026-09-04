# _env_common.sh — set_env 公共部分 (由各 set_env_<env>.sh source, 不直接 source)
#
# 提供: source build.conf (版本 + 路径默认) + source CANN set_env.sh (若 ASCEND_HOME_PATH 未设) +
#        PYTHONPATH + TORCH_DEVICE_BACKEND_AUTOLOAD=0 + DOPT_PYTHON (export)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 版本 + 路径默认值 (单一来源 build.conf; 环境变量优先)
source "${SCRIPT_DIR}/build.conf"

# source CANN set_env.sh (若 ASCEND_HOME_PATH 未设)
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ ! -r "${CANN_SET_ENV}" ]]; then
        echo "[set_env] 错误: CANN set_env.sh 不存在或不可读: ${CANN_SET_ENV}" >&2
        echo "  用 CANN_SET_ENV=... 覆盖" >&2
        return 1 2>/dev/null || exit 1
    fi
    source "${CANN_SET_ENV}"
fi

# 项目根加 PYTHONPATH (converter / NPU 都需)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 阻止 torch autoload npu backend (converter venv 不装 torch_npu; NPU 由 torch_npu 显式 import)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# dopt subprocess 用系统 Python 路径
export DOPT_PYTHON="${DOPT_PYTHON}"

set +u
