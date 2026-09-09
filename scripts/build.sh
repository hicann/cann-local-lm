#!/usr/bin/env bash
# =============================================================================
# build.sh — CANN-LM 顶层 end-to-end 构建入口
#
# subcommand:
#   check         校验基底 (CANN / P1 venv / torchair / torch_delegate / cann_lm / third_party), 不构建
#   deps          装 cann_lm/engine wheels 进 P1 venv (torchair+torch-delegate 为前置条件, 见 docs/build_guide.md)
#   third_party   自带 C++ 三方库 (nlohmann_json + GTest + makeself) → output/third_party [不需 env]
#   cpp [flags]   delegate build_cpp.sh (.so + cann_lm_compile, 不含测试)
#   tests [flags] delegate build_test.sh (ASAN, 独立 build_tests/ 树 + ctest)
#   wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
#   package       delegate package_run.sh (打 .run 自解压包)
#   all           deps → third_party → cpp → wheel
#
# 前置条件 (不本仓构建, 见 docs/build_guide.md):
#   - CANN toolkit (source 其 set_env.sh 设 ASCEND_HOME_PATH)
#   - torchair (TORCHAIR_ROOT) + torch-delegate (TORCH_DELEGATE_HOME, 含 .venv [torch+torchair+torch_delegate])
# converter (①) + compile (②) 复用一套 env: source scripts/set_env.sh (真 CANN, 无 stubs/无 0715)
#   check / deps / cpp / tests / package 均 source set_env.sh (需先 source CANN set_env.sh 设 ASCEND_HOME_PATH)
#   ③ NPU 运行: source scripts/set_env_cann.sh --npu
#   wheel / third_party 不需 env
# OM 编译 (epair→OM) 属测试/示例, 不在构建流程: 直接跑 examples/test_converter/export_llm_model/run_compile.sh
# 跨仓路径由 build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME) 提供, 可被环境变量覆盖.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${SCRIPT_DIR}/build.conf"   # build.conf 自定位 PROJECT_ROOT

_log()  { printf '\n▶ %s\n' "$*"; }
_step() { printf '  • %s\n' "$*"; }
_ok()   { printf '  ✓ %s\n' "$*"; }
_ng()   { printf '  ✗ %s\n' "$*" >&2; }
_skip() { printf '  ○ %s\n' "$*"; }

src_env()   { source "${SCRIPT_DIR}/set_env.sh"; }          # converter+compile 统一 env (①②)
src_npu()   { source "${SCRIPT_DIR}/set_env_cann.sh" --npu; }  # ③ NPU runtime
_have()     { ( cd /tmp && "${CONVERTER_PYTHON}" -c "import $1" 2>/dev/null ); }  # 中性 CWD, 避免 CWD 源码树误报已装

# ── check ─────────────────────────────────────────────────────────────────────
cmd_check() {
  _log "基底校验 (不构建)"
  # converter+compile 复用一套 (真 CANN, 无 stubs); 前置须 source CANN set_env.sh
  src_env >/dev/null 2>&1
  local fail=0
  set +e
  # ── 必需 (converter+compile ①②) ──
  if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" ]]; then _ok "CANN ${ASCEND_HOME_PATH}"; else _ng "CANN libascendcl.so (source CANN set_env)"; fail=1; fi
  if [[ -x "${CONVERTER_PYTHON}" ]]; then _ok "P1 venv ${CONVERTER_PYTHON}"; else _ng "P1 venv python (${CONVERTER_PYTHON})"; fail=1; fi
  if _have torch; then _ok "torch ($("${CONVERTER_PYTHON}" -c 'import torch;print(torch.__version__)' 2>/dev/null))"; else _ng "torch (前置: 见 docs/build_guide.md 安装 torch-delegate)"; fail=1; fi
  if _have torchair;       then _ok "torchair";       else _ng "torchair (前置: 见 docs/build_guide.md)";        fail=1; fi
  if _have torch_delegate; then _ok "torch_delegate"; else _ng "torch_delegate (前置: 见 docs/build_guide.md)";  fail=1; fi
  if _have cann_lm;        then _ok "cann_lm";        else _ng "cann_lm (bash build.sh wheel)";        fail=1; fi
  if [[ -d "${REPO}/output/third_party/json" ]]; then _ok "third_party json"; else _ng "third_party (bash build.sh third_party)"; fail=1; fi
  # ── 可选 (仅量化需 NPU 加速时需要, converter/compile 不依赖) ──
  echo "  可选 (仅量化 NPU 加速):"
  if [[ -x "${DOPT_PYTHON:-}" ]]; then _ok "P2 ${DOPT_PYTHON} (量化 NPU 加速用)"; else _ng "P2 python ${DOPT_PYTHON:-<未设>} (可选, 仅量化 NPU 加速时需; 见 docs/build_guide.md)"; fi
  if [[ -x "${DOPT_PYTHON:-}" ]] && "${DOPT_PYTHON}" -c "import torch_npu" 2>/dev/null; then _ok "torch_npu (P2, 量化 NPU 加速用)"; else _ng "torch_npu (可选, 仅量化需 NPU 加速时装; converter/compile 不需)"; fi
  set -e
  echo ""
  if [[ $fail -eq 0 ]]; then _ok "基底齐 (converter+compile ①② 必需项就绪; ③ NPU 量化加速为可选)"; else _ng "缺 ${fail} 项必需项 (按提示补)"; return 1; fi
}

# ── deps (装 cann_lm wheels 进 P1 venv; torchair/torch-delegate 为前置条件, 见 docs/build_guide.md) ──
cmd_deps() {
  _log "deps → 装 cann_lm/engine wheels 进 P1 venv (torchair+torch-delegate 须前置就绪)"
  src_env >/dev/null 2>&1   # 设 CONVERTER_PYTHON 等; 需先 source CANN set_env.sh

  # 前置条件校验: P1 venv + torchair + torch_delegate 须就绪
  # 兼容两种来源 (见 docs/build_guide.md): 源码装 (TORCHAIR_ROOT/TORCH_DELEGATE_HOME 有源码根) 或 whl 装 (仅 venv + 预编 whl)
  # 本仓不构建它们; 只查 CONVERTER_PYTHON 可执行 + torchair/torch_delegate 已装 venv
  local fail=0
  if [[ ! -x "${CONVERTER_PYTHON}" ]]; then _ng "P1 venv python 不可执行: ${CONVERTER_PYTHON} (前置: 见 docs/build_guide.md 装上游)"; fail=$((fail+1)); fi
  if ! _have torchair;        then _ng "torchair 未装 venv (前置: 见 docs/build_guide.md)"; fail=$((fail+1)); fi
  if ! _have torch_delegate;  then _ng "torch_delegate 未装 venv (前置: 见 docs/build_guide.md)"; fail=$((fail+1)); fi
  [[ $fail -eq 0 ]] || { _ng "前置条件缺 ${fail} 项, 中止 (补齐后重跑)"; return 1; }

  # 装 cann_lm + cann_lm_engine wheels 进 P1 venv (幂等: 已装则跳)
  if _have cann_lm; then _skip "cann_lm (已装 → 复用)"; else
    _step "build + install cann_lm / cann_lm_engine wheels"
    bash "${REPO}/scripts/cann_lm_converter/build_wheel.sh"
    bash "${REPO}/scripts/cann_lm_engine/build_wheel.sh"
    local cw; cw="$(ls "${REPO}"/dist/cann_lm-*.whl 2>/dev/null | grep -v engine | head -1 || true)"
    local ew; ew="$(ls "${REPO}"/dist/cann_lm_engine-*.whl 2>/dev/null | head -1 || true)"
    [[ -f "$cw" && -f "$ew" ]] || { _ng "cann_lm whl 未产出 (${REPO}/dist/)"; return 1; }
    "${CONVERTER_PYTHON}" -m pip install --force-reinstall --no-deps "$cw" "$ew"
    _ok "cann_lm + cann_lm_engine installed"
  fi
  echo ""
  _ok "converter+compile env 就绪: source scripts/set_env.sh"
}

# ── third_party (C++ 三方库, 不需 env) ──────────────────────────────────────────
cmd_third_party() {
  _log "third_party build (自带 nlohmann_json + GTest + makeself)"
  bash "${REPO}/scripts/cann_lm_compile/build_third_party.sh" "$@"
  _ok "third_party → ${REPO}/output/third_party"
}

# ── cpp (②) ──────────────────────────────────────────────────────────────────
cmd_cpp() {
  _log "C++ build (.so + cann_lm_compile, 不含测试; delegate build_cpp.sh)"
  src_env >/dev/null 2>&1
  bash "${REPO}/scripts/cann_lm_compile/build_cpp.sh" "$@"
  _ok "C++ build done (build/)"
}

# ── tests (②, ASAN, 独立 build_tests 树) ─────────────────────────────────────
cmd_tests() {
  _log "C++ 测试构建 + 运行 (ASAN, 独立 build_tests/, delegate build_test.sh)"
  src_env >/dev/null 2>&1
  bash "${REPO}/scripts/cann_lm_compile/build_test.sh" "$@"
  _ok "tests done (build_tests/)"
}

# ── wheel (不需 env) ──────────────────────────────────────────────────────────
cmd_wheel() {
  _log "wheel build (P1 venv pip, 不需 CANN)"
  bash "${REPO}/scripts/cann_lm_converter/build_wheel.sh"
  bash "${REPO}/scripts/cann_lm_engine/build_wheel.sh"
  _ok "wheels → ${REPO}/dist/"
}

# ── package (②) ───────────────────────────────────────────────────────────────
cmd_package() {
  _log "package .run (②, delegate package_run.sh)"
  src_env >/dev/null 2>&1
  bash "${REPO}/scripts/cann_lm_compile/package_run.sh" "$@"
}

# ── all ───────────────────────────────────────────────────────────────────────
cmd_all() {
  _log "all: deps → third_party → cpp → wheel"
  cmd_deps
  cmd_third_party
  cmd_cpp
  cmd_wheel
}

usage() {
  cat <<'EOF'
build.sh — CANN-LM 顶层 end-to-end 构建入口

subcommand:
  check         校验基底 (CANN / P1 venv / torchair / torch_delegate / cann_lm / third_party), 不构建
  deps          装 cann_lm/engine wheels 进 P1 venv (torchair+torch-delegate 为前置条件, 见 docs/build_guide.md)
  third_party   自带 C++ 三方库 (nlohmann_json + GTest + makeself) → output/third_party
  cpp [flags]   delegate build_cpp.sh (.so + cann_lm_compile, 不含测试)
  tests [flags] delegate build_test.sh (ASAN, 独立 build_tests/ + ctest)
  wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
  package       delegate package_run.sh (打 .run 自解压包)
  all           deps → third_party → cpp → wheel

env 映射: check·deps·cpp·tests·package→②cann / ①export 仅供日常 source 跑 converter / wheel·third_party 无 env
路径默认值在 scripts/build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME 等, 可被环境变量覆盖)
OM 编译 (epair→OM) 属测试/示例, 不在构建流程: 直接跑 examples/test_converter/export_llm_model/run_compile.sh

用法: bash scripts/build.sh <subcommand> [args]
EOF
}

case "${1:-help}" in
  check)        shift; cmd_check "$@";;
  deps)         shift; cmd_deps "$@";;
  third_party)  shift; cmd_third_party "$@";;
  cpp)          shift; cmd_cpp "$@";;
  tests)        shift; cmd_tests "$@";;
  wheel)        shift; cmd_wheel "$@";;
  package)      shift; cmd_package "$@";;
  all)          shift; cmd_all "$@";;
  -h|--help|help) usage;;
  *) echo "unknown subcommand: $1" >&2; usage; exit 1;;
esac
