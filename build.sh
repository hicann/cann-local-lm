#!/usr/bin/env bash
# =============================================================================
# build.sh — CANN-LM 顶层 end-to-end 构建入口
#
# subcommand:
#   check         校验基底 (CANN / P1 venv / torchair / torch_delegate / cann_lm / third_party), 不构建
#   deps          装 cann_lm/engine wheels 进 P1 venv (torchair+torch-delegate 为前置条件, 见 docs/build_guide.md)
#   third_party   自带 C++ 三方库 (nlohmann_json + GTest + makeself) → output/third_party [不需 env]
#   cpp [flags]   delegate build_cpp.sh (.so + cann_lm_compile, 不含测试)
#   wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
#   package       delegate package_run.sh (打 .run 自解压包)
#   all           deps → third_party → cpp → wheel
#
# 前置条件 (不本仓构建, 见 docs/build_guide.md):
#   - CANN toolkit (source 其 set_env.sh 设 ASCEND_HOME_PATH)
#   - torchair (TORCHAIR_ROOT) + torch-delegate (TORCH_DELEGATE_HOME, 含 .venv [torch+torchair+torch_delegate])
# converter (①) + compile (②) 复用一套 env: source scripts/cann_lm_compile/set_env.sh (真 CANN, 无 stubs/无 0715)
#   check / deps / cpp / tests / package 均 source set_env.sh (需先 source CANN set_env.sh 设 ASCEND_HOME_PATH)
#   ③ NPU 运行: source scripts/cann_lm_compile/set_env_cann.sh --npu
#   wheel / third_party 不需 env
# OM 编译 (epair→OM) 属测试/示例, 不在构建流程: 直接跑 examples/test_converter/export_llm_model/run_compile.sh
# 跨仓路径由 cann_lm_compile/build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME) 提供, 可被环境变量覆盖.
# 根 build.sh 自包含完整逻辑 (CI 流水线直接用本文件, 内容不可移走); scripts/build.sh 为 thin entry 反向透传到此。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"   # 根 build.sh 位于仓库根, SCRIPT_DIR 即 repo root
source "${SCRIPT_DIR}/scripts/cann_lm_compile/build.conf"   # build.conf 自定位 PROJECT_ROOT (C8 后入 cann_lm_compile/)

# ── pkg/ut 默认值 ──
VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-$(uname -m)}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
DIST="${DIST:-$REPO/build_out}"   # 分发产物根 (whl + .run); 可环境变量覆盖

# ── 镜像内场景: 三方库由 lib_cache 扁平布局预置, 不联网下载 ──
# 由 --cann_3rd_lib_path=<PATH> 触发 (compile.sh 入口已传 /home/jenkins/opensource)
# 直连引用 lib_cache/{json,gtest} + opensource/makeself (modules Find 模块),
# 不走 cmake-master 桥.
MIRROR_THIRD_PARTY="OFF"
CANN_3RD_LIB_PATH=""

_log()  { printf '\n▶ %s\n' "$*"; }
_step() { printf '  • %s\n' "$*"; }
_ok()   { printf '  ✓ %s\n' "$*"; }
_ng()   { printf '  ✗ %s\n' "$*" >&2; }
_warn() { printf '  ⚠ %s\n' "$*" >&2; }
_skip() { printf '  ○ %s\n' "$*"; }

src_env()   { source "${SCRIPT_DIR}/scripts/cann_lm_compile/set_env.sh"; }          # converter+compile 统一 env (①②)
src_npu()   { source "${SCRIPT_DIR}/scripts/cann_lm_compile/set_env_cann.sh" --npu; }  # ③ NPU runtime
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
  _ok "converter+compile env 就绪: source scripts/cann_lm_compile/set_env.sh"
}

# ── third_party (C++ 三方库, 不需 env) ──────────────────────────────────────────
cmd_third_party() {
  if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
    # 镜像内: 三方库已预置在 lib_cache (扁平布局), 不联网下载构建;
    # 只跑 need_third_party 校验 lib_cache/{json,gtest} + makeself 布局完整.
    _log "third_party (镜像: lib_cache 直连, 仅校验不下载)"
    need_third_party || return 1
    _ok "镜像三方库就绪 (lib_cache 直连, 跳过 build_third_party.sh)"
  else
    _log "third_party build (自带 nlohmann_json + GTest + makeself)"
    bash "${REPO}/scripts/cann_lm_compile/build_third_party.sh" "$@"
    _ok "third_party → ${REPO}/output/third_party"
  fi
}

# ── cpp (②) ──────────────────────────────────────────────────────────────────
cmd_cpp() {
  _log "C++ build (.so + cann_lm_compile, 不含测试; delegate build_cpp.sh)"
  src_env >/dev/null 2>&1
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


# ──────────────────────────────────────────────────────────────────────────
# pkg/ut (C9: 由根 build.sh --pkg/-u 迁入; engine host UT + converter + engine OHOS 打包)
# ──────────────────────────────────────────────────────────────────────────
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

# ── 镜像 gtest 惰性构建: lib_cache/gtest 缺静态库时, 从 googletest tar.gz 现场构建 ──
# 背景: 镜像 gcc15 变体预置的 gtest 只有空 include 占位 (cmake-master 桥原本会
# ExternalProject 现场编译, 直连简化后无构建逻辑). 镜像 lib_cache/gtest 下有
# googletest-1.14.0.tar.gz, 故在此补齐惰性构建. 幂等: 静态库齐全则跳过.
ensure_mirror_gtest() {
  local gt="${CANN_3RD_LIB_PATH}/lib_cache/gtest"
  # 已就位 → 直接通过
  if [[ -f "${gt}/include/gtest/gtest.h" && -f "${gt}/lib/libgtest.a" && -f "${gt}/lib/libgtest_main.a" ]]; then
    # ABI 自检: 静态库缺 [abi:cxx11] 符号 = 旧 ABI (=0) 构建; 当前编译器默认新 ABI (=1)
    # 时两者不匹配 → UT 链接期全部 undefined reference (testing::*[abi:cxx11]).
    # 检测: nm 静态库是否含 B5cxx11 符号 (新 ABI 独有, 无需 demangle).
    local compiler_abi
    compiler_abi="$(printf '#include <string>' | ${CXX:-c++} -E -dM -x c++ - 2>/dev/null | grep -c '_GLIBCXX_USE_CXX11_ABI 1')"
    if [[ "$compiler_abi" -ge 1 ]] && ! nm "${gt}/lib/libgtest.a" 2>/dev/null | grep -q 'B5cxx11'; then
      if [[ -f "${gt}/googletest-1.14.0.tar.gz" ]]; then
        _warn "lib_cache/gtest 为旧 ABI(=0) 构建, 编译器默认新 ABI(=1), 用 tar.gz 现场重建 (否则 UT 链接期 undefined reference)"
      else
        _ng "lib_cache/gtest 为旧 ABI(=0) 构建, 且无 googletest-1.14.0.tar.gz 可重建 (镜像三方库不完整)"
        return 1
      fi
    else
      _ok "lib_cache/gtest 完整 (头 + libgtest.a + libgtest_main.a)"
      return 0
    fi
  fi
  # 缺库但无 tar.gz → 报错
  if [[ ! -f "${gt}/googletest-1.14.0.tar.gz" ]]; then
    _ng "lib_cache/gtest 缺静态库且无 googletest-1.14.0.tar.gz (镜像三方库不完整)"
    return 1
  fi
  _step "lib_cache/gtest 缺静态库, 从 googletest-1.14.0.tar.gz 现场构建..."
  local build_dir
  build_dir="$(mktemp -d)"
  local src_dir="${build_dir}/src"
  mkdir -p "${src_dir}"
  # 解压 tar.gz (顶层为 googletest-1.14.0/)
  tar xzf "${gt}/googletest-1.14.0.tar.gz" -C "${src_dir}" || { _ng "gtest tar.gz 解压失败"; rm -rf "$build_dir"; return 1; }
  local srctree="${src_dir}/googletest-1.14.0"
  if [[ ! -d "${srctree}" ]]; then
    # tar 顶层目录名可能不同, 取唯一子目录
    srctree="$(find "${src_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  fi
  cmake -S "${srctree}" -B "${build_dir}/build" \
    -DCMAKE_INSTALL_PREFIX="${gt}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1 -O2 -fPIC" >/dev/null \
    && cmake --build "${build_dir}/build" -j"$(nproc 2>/dev/null || echo 4)" >/dev/null \
    && cmake --install "${build_dir}/build" >/dev/null \
    || { _ng "gtest 现场构建失败"; rm -rf "$build_dir"; return 1; }
  rm -rf "$build_dir"
  # 校验产物
  if [[ -f "${gt}/include/gtest/gtest.h" && -f "${gt}/lib/libgtest.a" && -f "${gt}/lib/libgtest_main.a" ]]; then
    _ok "gtest 现场构建完成: ${gt} (gtest.h + libgtest.a + libgtest_main.a)"
    return 0
  else
    _ng "gtest 构建后产物不完整"
    return 1
  fi
}

# ── 前置校验: third_party (json/gtest/makeself) ───────────────────────────
# 镜像内 (--cann_3rd_lib_path): 检查 lib_cache 布局 (json/gtest) + opensource/makeself;
# 非镜像: 检查 output/third_party 布局.
need_third_party() {
  local miss=0
  if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
    _step "镜像三方库校验 (lib_cache 直连, CANN_3RD_LIB_PATH=$CANN_3RD_LIB_PATH)"
    [[ -d "${CANN_3RD_LIB_PATH}/lib_cache/json/include/nlohmann" ]] || { _ng "缺 lib_cache/json (镜像三方库)"; miss=1; }
    # gtest: 缺失静态库时尝试从 lib_cache/gtest/googletest-1.14.0.tar.gz 现场构建 (镜像惰性构建兜底)
    ensure_mirror_gtest || miss=1
    [[ -f "${CANN_3RD_LIB_PATH}/makeself/makeself.sh" ]] || { _ng "缺 makeself/makeself.sh (镜像 opensource 下)"; miss=1; }
  else
    [[ -d "${THIRD_PARTY}/json" ]]        || { _ng "缺 third_party/json (先 bash scripts/build.sh third_party)"; miss=1; }
    [[ -d "${THIRD_PARTY}/gtest" ]]       || { _ng "缺 third_party/gtest (先 bash scripts/build.sh third_party)"; miss=1; }
    [[ -f "${THIRD_PARTY}/makeself/makeself.sh" ]] || { _ng "缺 makeself (先 bash scripts/build.sh third_party)"; miss=1; }
  fi
  return $miss
}

# ── ① compile (C++ epair→OM 工具 + backend lib) ───────────────────────────
build_compile() {
  _log "[1/4] compile: build_cpp.sh --no-tests --target cann_lm_compile"
  need_cann || return 1
  if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
    _step "镜像内: 透传 MIRROR_THIRD_PARTY=ON + CANN_3RD_LIB_PATH (lib_cache 直连)"
    MIRROR_THIRD_PARTY=ON CANN_3RD_LIB_PATH="$CANN_3RD_LIB_PATH" \
      bash "${REPO}/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile
  else
    bash "${REPO}/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile
  fi
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
  [[ -n "$whl" ]] || { _ng "converter whl 构建失败 (build_out/ 无 cann_lm-*.whl)"; return 1; }
  _ok "converter whl: $whl ($(du -h "$whl" | cut -f1))"
}

# ── 镜像 engine 依赖自动发现: SDK / 签名 jar / GTest tarball / libcxx 软链 ──
# 仅 MIRROR_THIRD_PARTY=ON 时启用(镜像 command-line-tools SDK + lib_cache 三方库),
# 非镜像仍走显式环境变量, 不改变既有行为. 幂等: 已满足则跳过.
# 镜像依赖落点(2026-10-26 核实):
#   - OHOS SDK native: /opt/ndktools/command-line-tools/sdk/default/openharmony/native
#                      (llvm/bin/clang -> clang-15 为 ARM-native, 路线A)
#   - 签名 jar:        sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar
#   - GTest tarball:   ${CANN_3RD_LIB_PATH}/lib_cache/gtest/googletest-1.14.0.tar.gz
#   - libcxx-ohos:     SDK 为扁平布局(头直接平铺), 工具链 cmake 期望嵌套路径, 需软链补出
engine_probe_mirror_deps() {
  [[ "$MIRROR_THIRD_PARTY" == "ON" ]] || return 0
  _step "镜像 engine 依赖自动发现 (OHOS SDK / 签名 jar / GTest / libcxx)"

  # 1) OHOS SDK native (路线A: ARM-native clang + sysroot)
  if [[ -z "${OHOS_SDK_NATIVE:-}" ]]; then
    for _cand in \
        /opt/ndktools/command-line-tools/sdk/default/openharmony/native \
        /opt/ndktools/command-line-tools/sdk/openharmony/native; do
      if [[ -x "$_cand/llvm/bin/clang" && -x "$_cand/llvm/bin/clang++" && -d "$_cand/sysroot" ]]; then
        OHOS_SDK_NATIVE="$_cand"
        break
      fi
    done
    if [[ -n "${OHOS_SDK_NATIVE:-}" ]]; then
      export OHOS_SDK_NATIVE
      _ok "自动发现 SDK native: ${OHOS_SDK_NATIVE}"
    fi
  fi

  # 2) 签名 jar: 仓库无 jar 时从 SDK toolchains/lib 拷入 tools/
  if [[ ! -f "$REPO/tools/binary-sign-tool.jar" && ! -f "$REPO/cann_lm/engine/ohos/binary-sign-tool.jar" ]]; then
    local _jar_src=""
    for _cand in \
        "${OHOS_SDK_NATIVE:-}/../toolchains/lib/binary-sign-tool.jar" \
        /opt/ndktools/command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar \
        /opt/ndktools/command-line-tools/sdk/openharmony/toolchains/lib/binary-sign-tool.jar; do
      if [[ -f "$_cand" ]]; then _jar_src="$_cand"; break; fi
    done
    if [[ -n "$_jar_src" ]]; then
      mkdir -p "$REPO/tools"
      cp "$_jar_src" "$REPO/tools/binary-sign-tool.jar"
      _ok "签名 jar 自 SDK: $_jar_src -> $REPO/tools/binary-sign-tool.jar"
    fi
  fi

  # 3) GTest tarball: 镜像 lib_cache/gtest/googletest-1.14.0.tar.gz
  if [[ -z "${GTEST_TAR:-}" && -n "${CANN_3RD_LIB_PATH:-}" ]]; then
    local _gt="${CANN_3RD_LIB_PATH}/lib_cache/gtest/googletest-1.14.0.tar.gz"
    if [[ -f "$_gt" ]]; then
      export GTEST_TAR="$_gt"
      _ok "自动发现 GTEST_TAR: ${GTEST_TAR}"
    fi
  fi

  # 4) libcxx-ohos 扁平头软链: 工具链 ohos.toolchain.cmake 期望嵌套路径
  #    -isystem ${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos/include/c++/v1,
  #    镜像 SDK 头直接平铺在 libcxx-ohos/ 下, 补软链 include/c++/v1 -> ../../ (幂等)
  if [[ -n "${OHOS_SDK_NATIVE:-}" && -d "${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos" \
        && ! -e "${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos/include/c++/v1" ]]; then
    mkdir -p "${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos/include/c++"
    ln -s ../../ "${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos/include/c++/v1"
    _ok "libcxx-ohos 软链补出嵌套路径: include/c++/v1 -> ../../"
  fi

  return 0
}

# ── ③ engine (OHOS 交叉编译, 强制环境检查) ────────────────────────────────
# 前置硬依赖(缺任一即逐项 ✗ 并中断 --pkg, 不打包/不跳过现有 deploy_ohos/):
#   1) 编译器: SDK 自带 clang(llvm/bin/clang, 路线A) 或 本机 clang-18/clang++-18 (路线B) 二选一
#   2) OHOS_SDK_NATIVE 含 sysroot
#   3) java + binary-sign-tool.jar (tools/ 或 cann_lm/engine/ohos/)
#   4) GTEST_TAR (googletest 源码 tarball, 默认 /mnt/workspace/gtest-1.14.0.tar.gz, 可 export 覆盖)
#   5) cmake / make
build_engine() {
  local sign_flag=""
  if [[ "${NO_SIGN:-0}" == "1" ]]; then
    sign_flag="--no-signed"
    _log "[3/4] engine: OHOS 交叉编译 (build_package_ohos.sh --no-signed, 不签名)"
  else
    _log "[3/4] engine: OHOS 交叉编译 (build_package_ohos.sh)"
  fi
  # 交叉编译编译器: SDK 自带 clang (OHOS_SDK_NATIVE/llvm/bin/clang); OHOS SDK + java/jar;
  local can_xcompile=1
  engine_probe_mirror_deps
  local miss=0

  # 1) 编译器: 路线A(SDK clang) 或 路线B(clang-18) 任一满足即可
  if [[ -x "${OHOS_SDK_NATIVE:-/nonexistent}/llvm/bin/clang" && -x "${OHOS_SDK_NATIVE:-/nonexistent}/llvm/bin/clang++" ]]; then
    _ok "编译器: SDK clang (${OHOS_SDK_NATIVE}/llvm/bin/clang)"
  else
    _warn "缺少 OHOS SDK clang (${OHOS_SDK_NATIVE:-<未设>}/llvm/bin/clang)"
    _warn "  1. 从 https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos 下载 Command Line Tools for Linux(ARM)"
    _warn "  2. unzip commandline-tools-linux-arm64-*.zip -d /your/ohos-sdk-path"
    _warn "  3. export OHOS_SDK_NATIVE=/your/ohos-sdk-path/command-line-tools/sdk/default/openharmony/native"
    can_xcompile=0
  fi
  if [[ ! -d "${OHOS_SDK_NATIVE:-/nonexistent}/sysroot" ]]; then
    _warn "OHOS_SDK_NATIVE 无效 (无 sysroot 子目录)"
    _warn "  export OHOS_SDK_NATIVE=<解压路径>/command-line-tools/sdk/default/openharmony/native"
    can_xcompile=0
  fi

  if [[ $can_xcompile -eq 1 ]]; then
    # 镜像下透传 MIRROR_THIRD_PARTY/CANN_3RD_LIB_PATH 给 build_package_ohos.sh → build_ohos.sh → cmake,
    # 否则 engine OHOS 交叉编译找不到 lib_cache 下的 nlohmann_json.
    if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
      MIRROR_THIRD_PARTY=ON CANN_3RD_LIB_PATH="$CANN_3RD_LIB_PATH" \
        bash "${REPO}/scripts/cann_lm_engine/build_package_ohos.sh" $sign_flag
    else
      bash "${REPO}/scripts/cann_lm_engine/build_package_ohos.sh" $sign_flag
    fi
    if [[ $? -eq 0 ]]; then
      _ok "engine 交叉编译 + deploy_ohos/ 更新完成"
    else
      _ng "engine 交叉编译失败, 终止 --pkg"
      return 1
    fi
  else
    return 1
  fi

  _ok "engine 部署产物就绪: deploy_ohos/ (libcann_lm_engine.so + UT/bbit/smoke)"
}

# ── ④ 打包 (compile_converter.run + engine whl) ──────────────────────────
package_all() {
  _log "[4/4] 打包 (compile_converter.run + engine whl)"

  # 4a. compile_converter.run (compile C++ + converter whl)
  if [[ -f "$REPO/build/cann_lm_compile" && -f "$REPO/build/libtorch_delegate_backend.so" ]] \
     && ls "$DIST"/cann_lm-*.whl >/dev/null 2>&1; then
    _step "package_run_all.sh → cann_lm_compile_converter-${VERSION}-${ARCH}.run"
    if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
      _step "镜像内: THIRD_PARTY → $CANN_3RD_LIB_PATH (makeself 在 opensource 下)"
      MIRROR_THIRD_PARTY=ON CANN_3RD_LIB_PATH="$CANN_3RD_LIB_PATH" \
        THIRD_PARTY="$CANN_3RD_LIB_PATH" bash "${REPO}/scripts/cann_lm_compile/package_run_all.sh"
    else
      bash "${REPO}/scripts/cann_lm_compile/package_run_all.sh"
    fi
  else
    _warn "compile/converter 产物不全, 跳过 compile_converter.run (先 --pkg 完整构建)"
  fi

  # 4b. engine whl（已由 build_engine 阶段产出到 dist/，无需额外打包）
  local engine_whl
  engine_whl="$(ls "$DIST"/cann_lm_engine-*.whl 2>/dev/null | head -1 || true)"
  if [[ -n "$engine_whl" ]]; then
    _ok "engine whl: $(basename "$engine_whl") ($(du -h "$engine_whl" | cut -f1))"
  else
    _warn "engine whl 未产出 (dist/ 无 cann_lm_engine-*.whl)"
  fi

  _log "打包结果 (dist/)"
  ls -lh "$DIST"/*.whl "$DIST"/*.run 2>/dev/null | awk '{print "  "$5, $9}' || _warn "dist/ 无产物"
}

# ── --pkg: 编译三部分 + 打包 ───────────────────────────────────────────────
cmd_pkg() {
  local args="$*"
  for a in "$@"; do
    case "$a" in
      --no-signed) export NO_SIGN=1; shift;;
    esac
  done
  _log "=== CANN-LM 全量构建 + 打包 (--pkg${NO_SIGN:+ --no-signed}) ==="
  cmd_third_party || return 1
  build_compile || return 1
  build_converter || return 1
  build_engine || return 1
  package_all
  _log "=== --pkg 完成 ==="
}

# ── --smoke: 从 examples/test_engine/ 独立构建 smoke + bbit（OHOS clang）──
cmd_smoke() {
  _log "=== cann-device-lm smoke 构建 (--smoke) ==="
  need_cann || return 1

  # 前置: engine 已编译（build_ohos/ 有 libcann_lm_engine.so）
  local engine_lib="$REPO/build_ohos/libcann_lm_engine.so"
  if [[ ! -f "$engine_lib" ]]; then
    _ng "build_ohos/libcann_lm_engine.so 不存在 (先 bash build.sh --pkg)"
    return 1
  fi
  _ok "engine .so: $engine_lib"

  # 前置: OHOS SDK
  if [[ ! -x "${OHOS_SDK_NATIVE:-/nonexistent}/llvm/bin/clang" ]]; then
    _ng "缺少 OHOS SDK clang (${OHOS_SDK_NATIVE:-<未设>}/llvm/bin/clang)"
    _ng "  export OHOS_SDK_NATIVE=<path>/command-line-tools/sdk/default/openharmony/native"
    return 1
  fi
  _ok "编译器: SDK clang (${OHOS_SDK_NATIVE}/llvm/bin/clang)"

  local toolchain="$REPO/cann_lm/engine/ohos/ohos.toolchain.cmake"
  local engine_inc="$REPO/cann_lm/engine"
  local smoke_build="$REPO/build_smoke"

  _step "cmake configure (examples/test_engine/)"
  cmake -S "$REPO/examples/test_engine" -B "$smoke_build" \
      -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
      -DOHOS_SDK_NATIVE="$OHOS_SDK_NATIVE" \
      -DCANN_LM_ENGINE_LIB="$engine_lib" \
      -DCANN_LM_ENGINE_INCLUDE="$engine_inc" \
      -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -vE "warning|WARNING" | tail -5

  _step "cmake build (api_smoke c_api_smoke golden_verify bbit)"
  cmake --build "$smoke_build" -j"$(nproc)" 2>&1 | grep -E "Built|error|Error" | tail -10

  # 签名（如果有 jar）
  local jar="$REPO/tools/binary-sign-tool.jar"
  if [[ -f "$jar" ]] && command -v java >/dev/null 2>&1; then
    _step "sign smoke binaries"
    for f in api_smoke c_api_smoke golden_verify bbit; do
      if [[ -f "$smoke_build/$f" ]]; then
        java -jar "$jar" -selfSign 1 -i "$smoke_build/$f" 2>/dev/null && _ok "$f signed" || _warn "$f 签名失败（不影响构建）"
      fi
    done
  fi

  _log "=== --smoke 完成 ==="
  echo "  产物: $smoke_build/"
  ls -lh "$smoke_build"/api_smoke "$smoke_build"/c_api_smoke "$smoke_build"/bbit 2>/dev/null | awk '{print "  "$5, $9}'
}

# ── -u: 编译含 UT 的部分并运行 ─────────────────────────────────────────────
cmd_ut() {
  _log "=== cann-device-lm UT 编译 + 运行 (-u) ==="
  cmd_third_party || return 1
  local ut_rc=0
  if [[ "$MIRROR_THIRD_PARTY" == "ON" ]]; then
    _step "镜像内: 透传 MIRROR_THIRD_PARTY=ON + CANN_3RD_LIB_PATH 给 build_test.sh"
    MIRROR_THIRD_PARTY=ON CANN_3RD_LIB_PATH="$CANN_3RD_LIB_PATH" \
      bash "${REPO}/scripts/build_test.sh" || ut_rc=$?
  else
    bash "${REPO}/scripts/build_test.sh" || ut_rc=$?
  fi
  if [[ $ut_rc -eq 0 ]]; then
    _log "=== -u 全部通过 ==="
  else
    _log "=== -u 有失败 (见上) ==="
    return 1
  fi
}

# ── usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
build.sh — cann-device-lm 顶层 end-to-end 构建入口 (仓库根, 自包含完整逻辑)

subcommand:
  check         校验基底 (CANN / P1 venv / torchair / torch_delegate / cann_lm / third_party), 不构建
  deps          装 cann_lm/engine wheels 进 P1 venv (torchair+torch-delegate 为前置条件, 见 docs/build_guide.md)
  third_party   自带 C++ 三方库 (nlohmann_json + GTest + makeself) → output/third_party
  cpp [flags]   delegate build_cpp.sh (.so + cann_lm_compile, 不含测试)
  wheel         delegate converter + engine build_wheel.sh (P1 venv, 不需 CANN)
  package       delegate package_run.sh (打 .run 自解压包)
  all           deps → third_party → cpp → wheel
  pkg|--pkg     全量构建 + 打包: compile C++ / converter whl / engine OHOS → whl + .run (dist/)
  ut|-u         编译含 UT 的部分并运行: C++ UT (engine+converter ASAN) / Python UT
  smoke|--smoke  从 examples/test_engine/ 独立构建 smoke + bbit（OHOS clang，需先 --pkg）
  --no-signed   与 --pkg 连用: engine .so 不签名 (跳过 java/jar 签名步骤)

env 映射: check·deps·cpp·tests·package→②cann / ①export 仅供日常 source 跑 converter / wheel·third_party 无 env
  pkg/ut: third_party(json/gtest/makeself) 必需; --pkg 的 compile / -u 的 tests 需先 source CANN set_env.sh;
          engine OHOS 交叉编译需 OHOS SDK clang, 缺则终止
路径默认值在 scripts/cann_lm_compile/build.conf (TORCHAIR_ROOT / TORCH_DELEGATE_HOME 等, 可被环境变量覆盖)
OM 编译 (epair→OM) 属测试/示例, 不在构建流程: 直接跑 examples/test_converter/export_llm_model/run_compile.sh

用法: bash build.sh <subcommand> [args]    # 仓库根入口 (CI 直接用); bash scripts/build.sh 等价 (thin 透传到此)
EOF
}

# ── 参数解析 (支持 --pkg/-u 后跟 --cann_3rd_lib_path=<path>) ──────────────
CMD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pkg)  CMD="pkg"; shift;;
    -u)     CMD="ut"; shift;;
    --cann_3rd_lib_path=*)
            MIRROR_THIRD_PARTY="ON"
            CANN_3RD_LIB_PATH="${1#*=}"
            shift;;
    --cann_3rd_lib_path)
            MIRROR_THIRD_PARTY="ON"
            CANN_3RD_LIB_PATH="${2:-}"
            shift 2;;
    -h|--help|help) usage; exit 0;;
    *)      echo "unknown arg: $1" >&2; usage; exit 1;;
  esac
done

case "$CMD" in
  # 注意: 上方 while 循环已解析并 shift 完所有参数 (含 --pkg/-u/--cann_3rd_lib_path),
  # 此处 $@ 已空, 不可再 shift (set -e 下 shift 无参数会非零退出).
  # 若需向子命令传额外 args, 应在 while 循环里收集而非依赖位置参数.
  check)        cmd_check "$@";;
  deps)         cmd_deps "$@";;
  third_party)  cmd_third_party "$@";;
  cpp)          cmd_cpp "$@";;
  wheel)        cmd_wheel "$@";;
  package)      cmd_package "$@";;
  all)          cmd_all "$@";;
  pkg|--pkg)    cmd_pkg "$@";;
  ut|-u)        cmd_ut "$@";;
  smoke|--smoke) cmd_smoke "$@";;
  -h|--help|help) usage;;
  *) echo "unknown subcommand: $1" >&2; usage; exit 1;;
esac
