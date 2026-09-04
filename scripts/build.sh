#!/usr/bin/env bash
# =============================================================================
# build.sh — CANN-LM 顶层 end-to-end 构建入口
#
# subcommand:
#   check         校验基底 (CANN / P1 / P2 / stubs / third_party), 不构建
#   deps          L1: torchair + torch-delegate + cann_lm/engine wheel → 装进 P1 (① 就绪, 幂等)
#   cpp [flags]   delegate build_cpp.sh (libtorch_delegate_backend.so + cann_lm_compile + tests)  [②]
#   wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
#   om [args]     delegate run_compile.sh (epair → OM)  [②, run_compile 自带 source]
#   package       delegate package_run.sh (打 .run 自解压包)  [②]
#   all           deps → cpp → wheel → om
#
# 各 subcommand 自动 source 对应 env:
#   ① export (P1+stubs)  → check (校验基底含 stubs)
#   ② cann   (真 CANN)   → deps / cpp / om / package  (构建/编译要真 CANN, 非 stubs)
#   wheel 不需 env (P1 venv pip 纯打包)
# 跨仓路径由 build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME) 提供, 可被环境变量覆盖.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${SCRIPT_DIR}/build.conf"

_log()  { printf '\n▶ %s\n' "$*"; }
_step() { printf '  • %s\n' "$*"; }
_ok()   { printf '  ✓ %s\n' "$*"; }
_ng()   { printf '  ✗ %s\n' "$*" >&2; }
_skip() { printf '  ○ %s\n' "$*"; }

src_export() { source "${SCRIPT_DIR}/set_env_export.sh"; }
src_cann()   { source "${SCRIPT_DIR}/set_env_cann.sh" "${1:-}"; }
_have()      { "${CONVERTER_PYTHON}" -c "import $1" 2>/dev/null; }

# ── check ─────────────────────────────────────────────────────────────────────
cmd_check() {
  _log "基底校验 (不构建)"
  # 用 ② cann (真 CANN, 无 stubs): stubs-first LD 会用 libtorch_npu_local_stub 污染 P2 import torch_npu.
  # P1 的 CPU torch 导入不依赖 CANN; stubs 目录走路径检查 (不靠 LD).
  src_cann >/dev/null 2>&1
  local fail=0
  set +e
  if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" ]]; then _ok "CANN ${ASCEND_HOME_PATH}"; else _ng "CANN libascendcl.so (source CANN set_env)"; fail=1; fi
  if [[ -x "${CONVERTER_PYTHON}" ]]; then _ok "P1 venv ${CONVERTER_PYTHON}"; else _ng "P1 venv python (${CONVERTER_PYTHON})"; fail=1; fi
  if _have torch; then _ok "torch ($("${CONVERTER_PYTHON}" -c 'import torch;print(torch.__version__)' 2>/dev/null))"; else _ng "torch (build.sh deps)"; fail=1; fi
  if _have torchair;       then _ok "torchair";       else _ng "torchair (bash build.sh deps)";        fail=1; fi
  if _have torch_delegate; then _ok "torch_delegate"; else _ng "torch_delegate (bash build.sh deps)";  fail=1; fi
  if _have cann_lm;        then _ok "cann_lm";        else _ng "cann_lm (bash build.sh wheel)";        fail=1; fi
  if [[ -d "${STUBS_DIR}" ]]; then _ok "stubs ${STUBS_DIR}"; else _ng "stubs (build.sh deps, torchair build 产出)"; fail=1; fi
  if [[ -x "${DOPT_PYTHON}" ]]; then _ok "P2 ${DOPT_PYTHON}"; else _ng "P2 python (${DOPT_PYTHON})"; fail=1; fi
  if "${DOPT_PYTHON}" -c "import torch_npu" 2>/dev/null; then _ok "torch_npu (P2)"; else _ng "torch_npu (P2)"; fail=1; fi
  if [[ -d "${TORCH_DELEGATE_HOME}/output/third_party/json" ]]; then _ok "third_party json"; else _ng "third_party (build.sh deps)"; fail=1; fi
  set -e
  echo ""
  if [[ $fail -eq 0 ]]; then _ok "基底齐 (① export + ②③ cann 就绪)"; else _ng "缺 ${fail} 项 (按提示补)"; return 1; fi
}

# ── deps (L1 → ①) ────────────────────────────────────────────────────────────
cmd_deps() {
  _log "L1 deps build → ① export env 就绪 (幂等: 已装则跳)"
  src_cann >/dev/null 2>&1   # 构建要真 CANN (torch-delegate cmake), 非 stubs

  if _have torchair; then _skip "torchair (已装 P1)"; else
    _step "build torchair (NO_ASCEND_SDK=1, target P1)"
    TARGET_PYTHON_PATH="${CONVERTER_PYTHON}" bash "${TORCHAIR_ROOT}/build.sh" -c
    local whl; whl="$(ls "${TORCHAIR_ROOT}"/output/torchair-*.whl 2>/dev/null | head -1 || true)"
    [[ -f "$whl" ]] || { _ng "torchair whl 未产出 (${TORCHAIR_ROOT}/output/)"; return 1; }
    "${CONVERTER_PYTHON}" -m pip install --force-reinstall --no-deps "$whl"
    _ok "torchair installed"
  fi

  if _have torch_delegate; then _skip "torch_delegate (已装 P1)"; else
    _step "build torch-delegate (third_party + cmake + whl)"
    bash "${TORCH_DELEGATE_HOME}/build.sh"
    local dwhl; dwhl="$(ls "${TORCH_DELEGATE_HOME}"/output/package/torch_delegate-*.whl 2>/dev/null | head -1 || true)"
    [[ -f "$dwhl" ]] || { _ng "torch_delegate whl 未产出 (${TORCH_DELEGATE_HOME}/output/package/)"; return 1; }
    "${CONVERTER_PYTHON}" -m pip install --force-reinstall --no-deps "$dwhl"
    _ok "torch_delegate installed"
  fi

  if _have cann_lm; then _skip "cann_lm (已装 P1)"; else
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
  _ok "① export env 就绪: source scripts/set_env_export.sh"
}

# ── cpp (②) ──────────────────────────────────────────────────────────────────
cmd_cpp() {
  _log "C++ build (② real-CANN, delegate build_cpp.sh)"
  src_cann >/dev/null 2>&1
  bash "${REPO}/scripts/cann_lm_compile/build_cpp.sh" "$@"
  _ok "C++ build done (build/)"
}

# ── wheel (不需 env) ──────────────────────────────────────────────────────────
cmd_wheel() {
  _log "wheel build (P1 venv pip, 不需 CANN)"
  bash "${REPO}/scripts/cann_lm_converter/build_wheel.sh"
  bash "${REPO}/scripts/cann_lm_engine/build_wheel.sh"
  _ok "wheels → ${REPO}/dist/"
}

# ── om (②, run_compile 自带 source) ───────────────────────────────────────────
cmd_om() {
  _log "OM compile (②, delegate run_compile.sh)"
  bash "${REPO}/scripts/cann_lm_compile/run_compile.sh" "$@"
}

# ── package (②) ───────────────────────────────────────────────────────────────
cmd_package() {
  _log "package .run (②, delegate package_run.sh)"
  src_cann >/dev/null 2>&1
  bash "${REPO}/scripts/cann_lm_compile/package_run.sh" "$@"
}

# ── all ───────────────────────────────────────────────────────────────────────
cmd_all() {
  _log "all: deps → cpp → wheel → om"
  cmd_deps
  cmd_cpp
  cmd_wheel
  cmd_om "$@"
}

usage() {
  cat <<'EOF'
build.sh — CANN-LM 顶层 end-to-end 构建入口

subcommand:
  check         校验基底 (CANN / P1 / P2 / stubs / third_party), 不构建
  deps          L1: torchair + torch-delegate + cann_lm/engine wheel → 装进 P1 (① 就绪, 幂等)
  cpp [flags]   delegate build_cpp.sh (.so + cann_lm_compile + tests)  [② real-CANN]
  wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
  om [args]     delegate run_compile.sh (epair → OM)  [②]
  package       delegate package_run.sh (打 .run 自解压包)  [②]
  all           deps → cpp → wheel → om

env 映射: check·deps·cpp·om·package→②cann / ①export 仅供日常 source 跑 converter / wheel 无 env
路径默认值在 scripts/build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME 等, 可被环境变量覆盖)

用法: bash scripts/build.sh <subcommand> [args]
EOF
}

case "${1:-help}" in
  check)   shift; cmd_check "$@";;
  deps)    shift; cmd_deps "$@";;
  cpp)     shift; cmd_cpp "$@";;
  wheel)   shift; cmd_wheel "$@";;
  om)      shift; cmd_om "$@";;
  package) shift; cmd_package "$@";;
  all)     shift; cmd_all "$@";;
  -h|--help|help) usage;;
  *) echo "unknown subcommand: $1" >&2; usage; exit 1;;
esac
