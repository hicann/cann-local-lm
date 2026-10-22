#!/usr/bin/env bash
#=============================================================================
# build.sh — CANN-LM 全量构建/打包/UT 入口 (仓库根)
#
# 用法:
#   bash build.sh --pkg   编译三部分(compile C++ / converter whl / engine OHOS)
#                         并打包成两个 .run 分发物 (dist/):
#                           - cann_lm_compile_converter-<ver>-<arch>.run   (compile + converter)
#                           - cann_lm_engine_ohos-<ver>-<arch>.run         (engine, OHOS 部署包)
#   bash build.sh -u      编译含 UT 的部分并运行:
#                           - engine host C++ UT   (GTest, 13 目标 + bbit)
#                           - converter C++ UT     (tests/c++/epair_archive, 需 CANN)
#                           - Python UT            (tests/py/converter + tests/py/llm_models + tests/python)
#
# 依赖/环境:
#   - CANN toolkit: 先 source 其 set_env.sh 设 ASCEND_HOME_PATH (compile + converter C++ UT 需要)
#   - C++ 三方库:   output/third_party (json/gtest/makeself), 缺则先 `bash scripts/build.sh third_party`
#   - engine OHOS 交叉编译: 前置硬依赖(SDK clang + OHOS SDK + 签名 jar + GTest tarball)缺任一即中断
#   - converter wheel: 系统 pip 即可 (--no-deps 纯源码打包, 不需 torch)
#
# 环境覆盖: VERSION / ARCH / THIRD_PARTY / BUILD_DIR / UT_BUILD_DIR / PIP
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR" && pwd)"
source "${SCRIPT_DIR}/scripts/build.conf"   # 自定位 PROJECT_ROOT + 版本/路径默认

VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-$(uname -m)}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
DIST="$REPO/dist"

_log()  { printf '\n▶ %s\n' "$*"; }
_step() { printf '  • %s\n' "$*"; }
_ok()   { printf '  ✓ %s\n' "$*"; }
_warn() { printf '  ⚠ %s\n' "$*" >&2; }
_ng()   { printf '  ✗ %s\n' "$*" >&2; }

# ── 前置校验: CANN 环境 (compile / converter C++ UT 需要) ────────────────
need_cann() {
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    _ng "ASCEND_HOME_PATH 未设。请先 source CANN 环境:"
    _ng "    source /path/to/Ascend/cann/set_env.sh"
    return 1
  fi
  if [[ ! -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" ]]; then
    _ng "CANN 无效 (无 ${ASCEND_HOME_PATH}/lib64/libascendcl.so)"
    return 1
  fi
  _ok "CANN ${ASCEND_HOME_PATH}"
}

# ── 前置校验: third_party (json/gtest/makeself) ───────────────────────────
need_third_party() {
  local miss=0
  [[ -d "${THIRD_PARTY}/json" ]]        || { _ng "缺 third_party/json (先 bash scripts/build.sh third_party)"; miss=1; }
  [[ -d "${THIRD_PARTY}/gtest" ]]       || { _ng "缺 third_party/gtest (先 bash scripts/build.sh third_party)"; miss=1; }
  [[ -f "${THIRD_PARTY}/makeself/makeself.sh" ]] || { _ng "缺 makeself (先 bash scripts/build.sh third_party)"; miss=1; }
  return $miss
}

# ── ① compile (C++ epair→OM 工具 + backend lib) ───────────────────────────
build_compile() {
  _log "[1/4] compile: build_cpp.sh --no-tests --target cann_lm_compile"
  need_cann || return 1
  bash "${REPO}/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile
  [[ -f "$REPO/build/cann_lm_compile" && -f "$REPO/build/libtorch_delegate_backend.so" ]] || {
    _ng "compile 产物缺失 (build/cann_lm_compile 或 build/libtorch_delegate_backend.so)"; return 1; }
  _ok "compile 产物就绪: build/cann_lm_compile + build/libtorch_delegate_backend.so"
}

# ── ② converter (HF→epair 导出 whl) ───────────────────────────────────────
build_converter() {
  _log "[2/4] converter: build cann_lm wheel (--no-deps 纯源码打包)"
  mkdir -p "$DIST"
  # PIP 覆盖: 默认系统 pip; 有 deps venv 则用它 (与 package_run_all.sh 的 PIP_CMD 同源)
  local pip_cmd="${PIP:-}"
  [[ -x "$pip_cmd" ]] || pip_cmd="pip"
  if [[ -x "${REPO}/deps/torch-delegate/.venv/bin/pip" ]]; then
    pip_cmd="${REPO}/deps/torch-delegate/.venv/bin/pip"
  fi
  "$pip_cmd" wheel "${REPO}/scripts/cann_lm_converter" --no-deps -w "$DIST" 2>&1 \
    | grep -E "Built|Created|error|Error" | tail -3
  local whl
  whl="$(ls -1 "$DIST"/cann_lm-*.whl 2>/dev/null | head -1 || true)"
  [[ -n "$whl" ]] || { _ng "converter whl 构建失败 (dist/ 无 cann_lm-*.whl)"; return 1; }
  _ok "converter whl: $whl ($(du -h "$whl" | cut -f1))"
}

# ── ③ engine (OHOS 交叉编译, 强制环境检查) ────────────────────────────────
# 前置硬依赖(缺任一即逐项 ✗ 并中断 --pkg, 不打包/不跳过现有 deploy_ohos/):
#   1) 编译器: SDK 自带 clang(llvm/bin/clang, 路线A) 或 本机 clang-18/clang++-18 (路线B) 二选一
#   2) OHOS_SDK_NATIVE 含 sysroot
#   3) java + binary-sign-tool.jar (tools/ 或 cann_lm/engine/ohos/)
#   4) GTEST_TAR (googletest 源码 tarball, 默认 /mnt/workspace/gtest-1.14.0.tar.gz, 可 export 覆盖)
#   5) cmake / make
build_engine() {
  _log "[3/4] engine: OHOS 交叉编译 (build_package_ohos.sh, 强制环境检查)"
  local miss=0

  # 1) 编译器: 路线A(SDK clang) 或 路线B(clang-18) 任一满足即可
  if [[ -x "${OHOS_SDK_NATIVE:-/nonexistent}/llvm/bin/clang" && -x "${OHOS_SDK_NATIVE:-/nonexistent}/llvm/bin/clang++" ]]; then
    _ok "编译器: SDK clang (${OHOS_SDK_NATIVE}/llvm/bin/clang)"
  elif command -v clang-18 >/dev/null 2>&1 && command -v clang++-18 >/dev/null 2>&1; then
    _ok "编译器: 本机 clang-18/clang++-18 (精简 SDK 路线B)"
  else
    _ng "缺少编译器 (需二选一):"
    _ng "  路线A: SDK 自带 clang (${OHOS_SDK_NATIVE:-<未设>}/llvm/bin/clang)"
    _ng "  路线B: apt install -y --no-install-recommends clang-18 llvm-18"
    miss=1
  fi

  # 2) OHOS SDK
  if [[ -n "${OHOS_SDK_NATIVE:-}" && -d "${OHOS_SDK_NATIVE}/sysroot" ]]; then
    _ok "OHOS_SDK_NATIVE: ${OHOS_SDK_NATIVE}"
  else
    _ng "OHOS_SDK_NATIVE 未设或无效 (无 sysroot 子目录)"
    _ng "  设置: export OHOS_SDK_NATIVE=<path>/ohos_native/native"
    miss=1
  fi

  # 3) java + 签名 jar
  if ! command -v java >/dev/null 2>&1; then
    _ng "缺少 java (签名步骤需要)"
    miss=1
  else
    _ok "java: $(command -v java)"
  fi
  if [[ -f "$REPO/tools/binary-sign-tool.jar" || -f "$REPO/cann_lm/engine/ohos/binary-sign-tool.jar" ]]; then
    _ok "binary-sign-tool.jar: 已就绪 (tools/ 或 cann_lm/engine/ohos/)"
  else
    _ng "缺少 binary-sign-tool.jar (应放 tools/ 或 cann_lm/engine/ohos/, 见 tools/README.md)"
    miss=1
  fi

  # 4) GTest 源码 tarball
  GTEST_TAR="${GTEST_TAR:-/mnt/workspace/gtest-1.14.0.tar.gz}"
  if [[ -f "$GTEST_TAR" ]]; then
    _ok "GTEST_TAR: $GTEST_TAR"
  else
    _ng "缺少 GTest 源码 tarball: $GTEST_TAR"
    _ng "  覆盖: export GTEST_TAR=<path>/googletest-1.14.0.tar.gz"
    miss=1
  fi

  # 5) cmake / make
  if ! command -v cmake >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    _ng "缺少 cmake 或 make (构建器需要)"
    miss=1
  else
    _ok "cmake/make: 已就绪"
  fi

  if [[ $miss -ne 0 ]]; then
    _ng "engine 交叉编译前置不满足 (见上); 已中断 --pkg, 不会打包现有 deploy_ohos/"
    return 1
  fi

  _ok "engine 交叉编译前置就绪 (SDK clang 或 clang-18 + OHOS SDK + 签名 + GTest)"
  GTEST_TAR="$GTEST_TAR" bash "${REPO}/scripts/cann_lm_engine/build_package_ohos.sh" \
    || { _ng "engine 交叉编译/打包失败"; return 1; }

  if [[ ! -f "$REPO/deploy_ohos/libcann_lm_engine.so" ]]; then
    _ng "交叉编译完成后 deploy_ohos/ 仍无 libcann_lm_engine.so"
    return 1
  fi
  _ok "engine 交叉编译 + 打包完成: deploy_ohos/ (libcann_lm_engine.so + UT/bbit/smoke)"
}

# ── ④ 打包 (compile_converter.run + engine_ohos.run) ──────────────────────
package_all() {
  _log "[4/4] 打包 .run (compile_converter + engine_ohos)"

  # 4a. compile_converter.run (compile C++ + converter whl)
  if [[ -f "$REPO/build/cann_lm_compile" && -f "$REPO/build/libtorch_delegate_backend.so" ]] \
     && ls "$DIST"/cann_lm-*.whl >/dev/null 2>&1; then
    _step "package_run_all.sh → cann_lm_compile_converter-${VERSION}-${ARCH}.run"
    bash "${REPO}/scripts/cann_lm_compile/package_run_all.sh"
  else
    _warn "compile/converter 产物不全, 跳过 compile_converter.run (先 --pkg 完整构建)"
  fi

  # 4b. engine_ohos.run (engine deploy_ohos/)
  if [[ -f "$REPO/deploy_ohos/libcann_lm_engine.so" ]]; then
    _step "package_run_ohos.sh → cann_lm_engine_ohos-${VERSION}-${ARCH}.run"
    bash "${REPO}/scripts/cann_lm_engine/package_run_ohos.sh" \
      || { _ng "engine_ohos.run 打包失败"; return 1; }
  else
    _ng "deploy_ohos/ 无 libcann_lm_engine.so, engine_ohos.run 打包失败 (engine 阶段应已中断)"
    return 1
  fi

  _log "打包结果 (dist/)"
  ls -lh "$DIST"/*.run 2>/dev/null | awk '{print "  "$5, $9}' || _warn "dist/ 无 .run 产物"
}

# ── --pkg: 编译三部分 + 打包 ───────────────────────────────────────────────
cmd_pkg() {
  _log "=== CANN-LM 全量构建 + 打包 (--pkg) ==="
  need_third_party || return 1
  build_compile || return 1
  build_converter || return 1
  build_engine || return 1
  package_all
  _log "=== --pkg 完成 ==="
}

# ── UT 运行环境: 编译 stub .so + LD_LIBRARY_PATH (触碰 NDK 的 UT 需要) ────
ensure_stub_ndk() {
  # tests/stub_ndk 提供 libneural_network_core.so + libhiai_foundation.so,
  # NDKProxy 裸名 dlopen; 需放在 UT exe 目录并用 LD_LIBRARY_PATH 指到。
  # 注: 项目 CMake 内建 stub_ndk 有路径 bug (../../engine 少一层 cann_lm), 此处直接 g++ 编译。
  local exe_dir="$1"
  local ext="$REPO/cann_lm/engine/external_cann"
  local compat="$REPO/cann_lm/engine/hiai_acl/src/ohos_compat.h"
  if [[ ! -f "$exe_dir/libneural_network_core.so" ]]; then
    g++ -std=c++17 -fPIC -shared -I"$ext" -include"$compat" \
        "$REPO/tests/stub_ndk/stub_neural_network_core.cpp" \
        -o "$exe_dir/libneural_network_core.so" -lpthread
  fi
  if [[ ! -f "$exe_dir/libhiai_foundation.so" ]]; then
    g++ -std=c++17 -fPIC -shared -I"$ext" -include"$compat" \
        "$REPO/tests/stub_ndk/stub_hiai_foundation.cpp" \
        -o "$exe_dir/libhiai_foundation.so"
  fi
  _ok "stub NDK .so → $exe_dir/"
}

# ── engine host C++ UT ─────────────────────────────────────────────────────
run_engine_ut() {
  _log "[UT/1] engine host C++ UT (GTest)"
  local build_dir="${UT_BUILD_DIR:-/tmp/llm_ut_host/engine}"
  _step "cmake configure (BUILD_ENGINE=ON, BUILD_CONVERTER=OFF)"
  cmake -S "$REPO" -B "$build_dir" \
      -DBUILD_ENGINE=ON -DBUILD_CONVERTER=OFF \
      -DCMAKE_BUILD_TYPE=Debug \
      -DGTest_DIR="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
      -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
      >/dev/null
  _step "cmake build (engine + UT 目标)"
  cmake --build "$build_dir" -j"$(nproc)" >/dev/null
  local exe_dir="$build_dir/cann_lm/engine"
  [[ -d "$exe_dir" ]] || exe_dir="$build_dir"
  ensure_stub_ndk "$exe_dir"
  export LD_LIBRARY_PATH="$exe_dir:${LD_LIBRARY_PATH:-}"

  _step "运行 UT 二进制 (含 stub NDK)"
  local fail=0 rc=0
  local -a bins
  mapfile -t bins < <(find "$exe_dir" -maxdepth 1 -type f -name 'ut_*' -executable 2>/dev/null | sort)
  [[ -f "$exe_dir/model_executor_ut" ]] && bins+=("$exe_dir/model_executor_ut")
  if [[ ${#bins[@]} -eq 0 ]]; then _ng "engine UT 二进制未找到 ($exe_dir)"; return 1; fi
  for b in "${bins[@]}"; do
    printf '  == %s\n' "$(basename "$b")"
    timeout 300 "$b" --gtest_brief=1 >/dev/null 2>&1; rc=$?
    if [[ $rc -eq 0 ]]; then
      _ok "  $(basename "$b"): PASS"
    else
      _ng "  $(basename "$b"): FAIL (exit=$rc)"
      fail=1
    fi
  done

  # bbit 黑盒: 只跑 --check (默认完整 flow 需设备模型文件, 本机无模型必然非0)
  _step "bbit --check (黑盒自检)"
  timeout 300 "$exe_dir/bbit" --check >/dev/null 2>&1; rc=$?
  if [[ $rc -eq 0 ]]; then
    _ok "bbit --check: PASS"
  else
    _ng "bbit --check: FAIL (exit=$rc)"; fail=1
  fi

  if [[ $fail -eq 0 ]]; then
    _ok "engine host UT 全部通过"
  else
    _warn "engine host UT 有失败 (见上; 已知 model_executor_ut 3 个 TensorManager/KvCachePool 尺寸断言与 NDK 无关)"
    return 1
  fi
}

# ── converter C++ UT (tests/c++/epair_archive) ─────────────────────────────
run_converter_cpp_ut() {
  _log "[UT/2] converter C++ UT (tests/c++/epair_archive, 需 CANN)"
  need_cann || return 1
  local build_dir="${UT_BUILD_DIR:-/tmp/llm_ut_host}/converter"
  _step "cmake configure (BUILD_ENGINE=OFF, BUILD_CONVERTER=ON)"
  cmake -S "$REPO" -B "$build_dir" \
      -DBUILD_ENGINE=OFF -DBUILD_CONVERTER=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DGTest_DIR="${THIRD_PARTY}/gtest/lib/cmake/GTest" \
      -Dnlohmann_json_DIR="${THIRD_PARTY}/json/share/cmake/nlohmann_json" \
      >/dev/null
  _step "cmake build (test_* + fixtures)"
  cmake --build "$build_dir" -j"$(nproc)" >/dev/null
  _step "ctest"
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
  ctest --test-dir "$build_dir" --output-on-failure
}

# ── Python UT ──────────────────────────────────────────────────────────────
run_python_ut() {
  _log "[UT/3] Python UT (pytest)"
  local fail=0
  for suite in tests/py/converter tests/py/llm_models tests/python; do
    if [[ -d "$REPO/$suite" ]]; then
      _step "pytest $suite"
      if ( cd "$REPO" && timeout 600 python3 -m pytest "$suite" -q ); then
        _ok "$suite: PASS"
      else
        _ng "$suite: FAIL"; fail=1
      fi
    fi
  done
  return $fail
}

# ── -u: 编译含 UT 的部分并运行 ─────────────────────────────────────────────
cmd_ut() {
  _log "=== CANN-LM UT 编译 + 运行 (-u) ==="
  need_third_party || return 1
  local fail=0
  run_engine_ut      || fail=1
  run_converter_cpp_ut || fail=1
  run_python_ut      || fail=1
  if [[ $fail -eq 0 ]]; then
    _log "=== -u 全部通过 ==="
  else
    _log "=== -u 有失败 (见上) ==="
    return 1
  fi
}

# ── usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
build.sh — CANN-LM 全量构建/打包/UT 入口 (仓库根)

用法:
  bash build.sh --pkg   编译三部分并打包成两个 .run 分发物 (dist/):
                          - cann_lm_compile_converter-<ver>-<arch>.run  (compile C++ + converter whl)
                          - cann_lm_engine_ohos-<ver>-<arch>.run        (engine OHOS 部署包)
  bash build.sh -u      编译含 UT 的部分并运行:
                          - engine host C++ UT (GTest, stub NDK, 13 目标 + bbit)
                          - converter C++ UT   (tests/c++/epair_archive, 需 CANN)
                          - Python UT          (tests/py/converter + llm_models + tests/python)
  bash build.sh -h      帮助

前置:
  - source CANN set_env.sh (设 ASCEND_HOME_PATH)   [--pkg 的 compile / -u 的 converter C++ UT 需要]
  - bash scripts/build.sh third_party              [C++ 三方库 json/gtest/makeself]
  - engine 交叉编译前置硬依赖(缺任一中断): SDK clang + OHOS SDK(sysroot) + java + binary-sign-tool.jar + GTest tarball(GTEST_TAR)

环境覆盖: VERSION / ARCH / THIRD_PARTY / BUILD_DIR / UT_BUILD_DIR / PIP
EOF
}

case "${1:-help}" in
  --pkg)      shift; cmd_pkg;;
  -u)         shift; cmd_ut;;
  -h|--help|help) usage;;
  *)          echo "unknown arg: ${1:-}" >&2; usage; exit 1;;
esac
