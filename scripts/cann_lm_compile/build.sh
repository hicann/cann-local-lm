#!/usr/bin/env bash
# =============================================================================
# build.sh — cann_lm_compile 前置环境准备 (+ 构建)
#
# 职责: 校验前置依赖 (CANN / converter venv / torch cmake / third_party) →
#       source set_env.sh → (bash 执行时) 跑 build_cpp.sh
#
# 用法:
#   bash scripts/cann_lm_compile/build.sh              # 校验 + env + 构建 (全树)
#   bash scripts/cann_lm_compile/build.sh --no-tests   # 校验 + env + 构建 (跳 tests)
#   bash scripts/cann_lm_compile/build.sh --asan       # 校验 + env + ASAN 构建
#   bash scripts/cann_lm_compile/build.sh --check      # 仅校验前置, 不 source 不构建
#   source scripts/cann_lm_compile/build.sh            # 校验 + env (持久到当前 shell), 不构建
#   (其余参数透传 build_cpp.sh: --debug / --target ... / --no-compile-tool ...)
#
# 路径默认值可由环境变量覆盖: CANN_SET_ENV / THIRD_PARTY
# =============================================================================

# 仅 bash 执行 (非 source) 时启用 set -e; source 模式留宽松避免污染交互 shell
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then set -euo pipefail; else set -u; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── 默认路径 (可覆盖) ──
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann/set_env.sh}"
THIRD_PARTY="${THIRD_PARTY:-${REPO}/output/third_party}"

# ── 1. 前置校验 ──
echo "=== [1/3] 前置环境校验 ==="
FAIL=0
ok() { echo "  ✓ $1"; }
ng() { echo "  ✗ $1" >&2; FAIL=1; }

if [[ -r "$CANN_SET_ENV" ]]; then ok "CANN set_env.sh : $CANN_SET_ENV"; else ng "CANN set_env.sh 不可读: $CANN_SET_ENV (用 CANN_SET_ENV=... 覆盖)"; fi

if [[ -d "${THIRD_PARTY}/json" ]]; then ok "nlohmann_json: ${THIRD_PARTY}/json"; else ng "third_party/json 缺失 (${THIRD_PARTY}/json)"; fi
if [[ -d "${THIRD_PARTY}/gtest" ]]; then ok "GTest        : ${THIRD_PARTY}/gtest"; else ng "third_party/gtest 缺失 (${THIRD_PARTY}/gtest)"; fi

if [[ "$FAIL" -eq 1 ]]; then
  echo "" >&2
  echo "前置校验失败, 中止。修复上述 ✗ 项后重试。" >&2
  echo "  CANN 未装: 见 AGENTS.md「环境问题」(下载 cann run 包安装)" >&2
  echo "  third_party 缺失: bash scripts/build.sh third_party (nlohmann_json + GTest)" >&2
  return 1 2>/dev/null || exit 1
fi

# ── 2. source compile env ──
echo ""
echo "=== [2/3] source set_env.sh ==="
source "$REPO/scripts/set_env.sh"

# ── 3. 构建 ──
CHECK_ONLY=0
BUILD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift;;
    *) BUILD_ARGS+=("$1"); shift;;
  esac
done

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo ""
  echo "=== [3/3] --check: 仅校验, 跳过 source/build (注: 上方 [2/3] 已 source)" >&2
  return 0 2>/dev/null || exit 0
fi

echo ""
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # bash 执行: 构建
  echo "=== [3/3] 构建 (build_cpp.sh ${BUILD_ARGS[*]:-}) ==="
  bash "$REPO/scripts/cann_lm_compile/build_cpp.sh" "${BUILD_ARGS[@]}"
else
  # source 模式: env 已就绪, 不构建
  echo "=== [3/3] source 模式: env 已就绪 (持久)。构建请跑: bash scripts/cann_lm_compile/build_cpp.sh ==="
fi
