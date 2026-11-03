#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# package_run_all.sh — 打 converter + compile 整合 .run 自解压包 (makeself)
#
# 整合两模块为一个 .run 分发物 (构建期无依赖, 安装后仍是两个独立入口):
#   - compile:   cann_lm_compile 二进制 + lib/libtorch_delegate_backend.so
#                + fusion_autoub_off.cfg (C++ epair→OM 编译工具, 运行时需 CANN)
#   - converter: wheels/cann_local_lm-*.whl (HF→epair 导出, 运行时需 torch 生态)
# 衔接: converter 导出 epair → cann_lm_compile 编译 OM (shell 手工, Python 不内嵌调用)
#
# 流程: build_cpp.sh(--no-tests --target cann_lm_compile) + build_wheel.sh
#       → stage (binary+lib+cfg+whl+op_cpu+op_pass+set_env) → patchelf rpath → makeself
# 产物: build_out/cann_local_lm_converter-<ver>-<arch>.run
# 安装: bash <pkg>.run --target /opt/cann-local-lm
# 运行: source <CANN set_env.sh> && source /opt/cann-local-lm/set_env.sh
#       && pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl   # converter (仅需 torch 时)
#       && /opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> [out]  # compile
# 量化 mobile: SOC_VERSION 为 Kirin 系时自动启用 (op_cpu/op_pass + GE 开关)
#       (GE_EXTERNAL_WEIGHT / GE_MOBILE_WEIGHT_SHARE_OPS / ASCEND_CUSTOM_OPP_PATH / ASCEND_GE_PY_PASS_PATH)
#
# 环境覆盖: VERSION / ARCH / THIRD_PARTY / BUILD_DIR / PIP(可指定 python -m pip)
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-$(uname -m)}"
PKG_NAME="cann_local_lm_converter-${VERSION}-${ARCH}"
STAGE="$REPO/build/package_stage/$PKG_NAME"
DIST="${DIST:-$REPO/build_out}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
MAKESELF="${THIRD_PARTY}/makeself/makeself.sh"
# whl 打包 pip: 默认用 build_wheel.sh 的 venv (deps/torch-delegate/.venv), 缺失回退系统 pip
PIP_CMD="${PIP:-${REPO}/deps/torch-delegate/.venv/bin/pip}"
[[ -x "$PIP_CMD" ]] || PIP_CMD="pip"

# ── 1. build compile (cann_lm_compile + .so, 跳 tests) ──
echo "### [1/5] build compile (build_cpp.sh --no-tests --target cann_lm_compile)"
bash "$REPO/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile

BIN="$REPO/build/cann_lm_compile"
SO="$REPO/build/libtorch_delegate_backend.so"
PLUGIN_SO="$REPO/build/libcustom_pass_plugin.so"
CFG="$REPO/cann_local_lm/converter/export/csrc/tools/fusion_autoub_off.cfg"
for f in "$BIN" "$SO" "$PLUGIN_SO" "$CFG"; do
    [[ -f "$f" ]] || { echo "ERROR: 缺产物 $f" >&2; exit 1; }
done

# ── 2. build converter whl (复用 build_wheel.sh 的产物; 跳过其内部 venv pip 调用) ──
echo "### [2/5] build converter whl"
mkdir -p "$DIST"
"$PIP_CMD" wheel "$REPO/scripts/cann_lm_converter" --no-deps -w "$DIST" 2>&1 \
    | grep -E "Built|Created|error|Error" | tail -3
WHL="$(ls -1 "$DIST"/cann_local_lm-*.whl 2>/dev/null | head -1 || true)"
[[ -n "$WHL" ]] || { echo "ERROR: whl 构建失败 (build_out/ 无 cann_local_lm-*.whl)" >&2; exit 1; }
echo "  whl: $WHL ($(du -h "$WHL" | cut -f1))"

# ── 3. stage (binary + lib/ + cfg + wheels/ + op_cpu/ + op_pass/ + set_env.sh) ──
echo "### [3/5] stage → $STAGE"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib" "$STAGE/wheels"
cp "$BIN" "$STAGE/cann_lm_compile"
cp "$SO"  "$STAGE/lib/libtorch_delegate_backend.so"
cp "$PLUGIN_SO" "$STAGE/lib/libcustom_pass_plugin.so"
cp "$CFG" "$STAGE/fusion_autoub_off.cfg"
cp "$WHL" "$STAGE/wheels/"

# 量化 mobile 链所需的 GE python 插件 (Bitcast/TransData HostCpu + QBMV4 x2 TransData pass)。
# 源码目录随 converter wheel 一起分发, 但 pip 装进 venv 后路径不稳定, 故此处平铺到安装根,
# 由 set_env.sh --mobile 通过 ASCEND_CUSTOM_OPP_PATH / ASCEND_GE_PY_PASS_PATH 指向。
for d in op_cpu op_pass; do
    src="$REPO/cann_local_lm/converter/export/$d"
    [[ -d "$src" ]] || { echo "ERROR: 缺插件目录 $src" >&2; exit 1; }
    cp -r "$src" "$STAGE/$d"
    find "$STAGE/$d" -name __pycache__ -type d -prune -exec rm -rf {} +
done

# 生成 set_env.sh (安装后 source; 设 LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE;
#   --npu 额外设 DEVICE + 探测 DOPT_PYTHON; --mobile 额外设量化 mobile 链的 GE 开关与插件路径)
cat > "$STAGE/set_env.sh" <<'SCRIPT'
#!/bin/bash
# set_env.sh — cann-local-lm 整合包 env (安装后 source)
# 前置: 安装机需先 source CANN set_env.sh (提供 CANN lib64/driver/opp)
#
# 用法:
#   source /opt/cann-local-lm/set_env.sh           # 编译/导出模式
#   source /opt/cann-local-lm/set_env.sh --npu     # NPU 量化模式 (额外设 DEVICE + 探测 DOPT_PYTHON)
#
# SOC 为 Kirin 系 (KirinX90/Kirin9030/...) 时自动进入 mobile 分支
# (外置权重 + 权重共享算子 + GE python 插件 op_cpu/op_pass), 无需额外参数。
# 切换目标: export SOC_VERSION=Ascend910_9362 (或任意非 Kirin 值) 即关闭 mobile 分支。
#
# ── 环境变量说明 ────────────────────────────────────────────────────────────
# SOC_VERSION                目标 SoC (ge.socVersion)。Kirin 系 → mobile 编译链
#                            (.omc 产物 + MobileModelHelper 路径); 其余 → 通用 .om 路径。
# GE_FUSION_SWITCH_FILE      GE 图融合开关配置。默认包内 fusion_autoub_off.cfg
#                            (关掉 auto-UB 融合, 避免 mobile 路径出无 kernel 的融合子图)。
# TORCH_DEVICE_BACKEND_AUTOLOAD
#                            置 0 阻止 torch 自动加载 npu backend (converter venv 无 torch_npu)。
# ── 以下仅 Kirin 系 (mobile 分支) 生效 ──
# GE_EXTERNAL_WEIGHT         2 = 权值外置: 产物为 <out>.omc + weight/*.bin, 运行时加载;
#                            0 = 权值内嵌进 omc (体积大)。mobile 路径默认 2。
# GE_MOBILE_WEIGHT_SHARE_OPS mobile 路径的共享权重算子白名单 (GE [Mobile] 路径读取):
#                            列出的算子若输入来自同一 FileConstant, 只存一份权重。
#                            默认 MatMulV3,QuantBatchMatmulV4,RmsNorm。
# ASCEND_CUSTOM_OPP_PATH     GE python 自定义算子插件目录。这里指向包内 op_cpu/,
#                            注册 Bitcast / TransData 的 HostCpu kernel (int4 布局常量折叠)。
# ASCEND_GE_PY_PASS_PATH     GE python pass 插件目录。这里指向包内 op_pass/,
#                            注册 QbmX2TransDataPass (QBMV4 的 x2 int4 ND→FRACTAL_Z)。
# ───────────────────────────────────────────────────────────────────────────

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
export GE_FUSION_SWITCH_FILE="${INSTALL_DIR}/fusion_autoub_off.cfg"
export SOC_VERSION="${SOC_VERSION:-KirinX90}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

_MODE_NPU=0
for _arg in "$@"; do
    case "$_arg" in
        --npu) _MODE_NPU=1 ;;
        *) echo "[cann-local-lm] 未知参数: $_arg (支持 --npu)" >&2 ;;
    esac
done
unset _arg

# Kirin 系 SOC → mobile 编译链 (与 compile_main.cc / ge_converter.py 的判别口径一致)
case "$SOC_VERSION" in
    Kirin*|kirin*)
        export GE_EXTERNAL_WEIGHT="${GE_EXTERNAL_WEIGHT:-2}"
        export GE_MOBILE_WEIGHT_SHARE_OPS="${GE_MOBILE_WEIGHT_SHARE_OPS:-MatMulV3,QuantBatchMatmulV4,RmsNorm}"
        # 幂等: 重复 source 不重复追加
        [[ ":${ASCEND_CUSTOM_OPP_PATH:-}:" == *":${INSTALL_DIR}/op_cpu:"* ]] \
            || export ASCEND_CUSTOM_OPP_PATH="${INSTALL_DIR}/op_cpu${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
        [[ ":${ASCEND_GE_PY_PASS_PATH:-}:" == *":${INSTALL_DIR}/op_pass:"* ]] \
            || export ASCEND_GE_PY_PASS_PATH="${INSTALL_DIR}/op_pass${ASCEND_GE_PY_PASS_PATH:+:${ASCEND_GE_PY_PASS_PATH}}"
        [[ ":${LD_LIBRARY_PATH:-}:" == *":${INSTALL_DIR}/op_cpu:"* ]] \
            || export LD_LIBRARY_PATH="${INSTALL_DIR}/op_cpu:${LD_LIBRARY_PATH:-}"
        _MODE_MOBILE=1
        ;;
    *)
        _MODE_MOBILE=0
        ;;
esac

if [[ "$_MODE_NPU" -eq 1 ]]; then
    export DEVICE=npu
    # 探测装了 torch_npu 的 python
    _detect_dopt_python() {
        # 收集候选 python: PATH 里的 + 常见路径
        local candidates=()
        local p
        # PATH 里所有 python3
        while IFS= read -r p; do candidates+=("$p"); done < <(command -v python3 2>/dev/null; which -a python3 2>/dev/null)
        # 常见路径
        for p in /usr/local/bin/python3 /usr/bin/python3 "$HOME/.conda/envs/*/bin/python3" "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3"; do
            for match in $p; do [[ -x "$match" ]] && candidates+=("$match"); done
        done
        # 去重
        local -A seen=()
        local plib
        for p in "${candidates[@]}"; do
            [[ -n "${seen[$p]:-}" ]] && continue
            seen[$p]=1
            [[ -x "$p" ]] || continue
            plib="$(dirname "$(dirname "$p")")/lib"; [[ -d "$plib" ]] || plib=""
            if LD_LIBRARY_PATH="${plib:+${plib}:}${ASCEND_HOME_PATH:-}/lib64:${LD_LIBRARY_PATH:-}" \
               "$p" -c "import torch_npu" 2>/dev/null; then
                export DOPT_PYTHON="$p"
                [[ -n "$plib" ]] && export LD_LIBRARY_PATH="${plib}:${LD_LIBRARY_PATH:-}"
                return 0
            fi
        done
        return 1
    }
    if _detect_dopt_python; then
        echo "[cann-local-lm] NPU mode: DEVICE=npu, DOPT_PYTHON=$DOPT_PYTHON"
    else
        echo "[cann-local-lm] NPU mode: DEVICE=npu, DOPT_PYTHON 未探测到 (需手动 export DOPT_PYTHON=/path/to/python3)"
    fi
fi

echo "[cann-local-lm] env ready (INSTALL_DIR=$INSTALL_DIR, SOC=${SOC_VERSION})"
echo "  OM 编译: $INSTALL_DIR/cann_lm_compile <model_json> <weights_index> [output_name]"
echo "  converter 安装: pip install $INSTALL_DIR/wheels/cann_local_lm-*.whl"
echo "  converter 运行: cllm-converter --help (需 torch 环境)"
if [[ "$_MODE_MOBILE" -eq 1 ]]; then
    echo "  mobile: GE_EXTERNAL_WEIGHT=${GE_EXTERNAL_WEIGHT} SHARE_OPS=${GE_MOBILE_WEIGHT_SHARE_OPS}"
    echo "  mobile: ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
    echo "  mobile: ASCEND_GE_PY_PASS_PATH=${ASCEND_GE_PY_PASS_PATH}"
fi
SCRIPT
chmod +x "$STAGE/set_env.sh"

# ── 4. patchelf .so rpath → $ORIGIN (去 build 机绝对路径, 可重定位; 运行靠 set_env 的 LD_LIBRARY_PATH) ──
echo "### [4/5] patchelf --set-rpath \$ORIGIN (libtorch_delegate_backend.so)"
if command -v patchelf >/dev/null; then
    patchelf --set-rpath '$ORIGIN' "$STAGE/lib/libtorch_delegate_backend.so"
else
    echo "  (patchelf 未装, 跳过; 运行靠 set_env.sh 的 LD_LIBRARY_PATH)"
fi

# ── 5. makeself → .run ──
echo "### [5/5] makeself → $DIST/$PKG_NAME.run"
# 优先用自带 makeself.sh (build_third_party 产出); 缺失则回退 PATH 上的 makeself
MAKESELF_BIN=""
if [[ -f "$MAKESELF" ]]; then
    MAKESELF_BIN="$MAKESELF"
    MAKESELF_HDR="${THIRD_PARTY}/makeself/makeself-header.sh"
elif command -v makeself >/dev/null; then
    MAKESELF_BIN="makeself"
else
    echo "ERROR: makeself 未找到 (自带: $MAKESELF 或 PATH)" >&2
    echo "  先构建: bash scripts/build.sh third_party" >&2
    exit 1
fi
if [[ "$MAKESELF_BIN" == "$MAKESELF" ]]; then
    bash "$MAKESELF_BIN" --header "$MAKESELF_HDR" --notemp "$STAGE" "$DIST/$PKG_NAME.run" \
        "cann_lm_compile+converter $VERSION (compile: CANN-only; converter: torch)" ./set_env.sh
else
    "$MAKESELF_BIN" --notemp "$STAGE" "$DIST/$PKG_NAME.run" \
        "cann_lm_compile+converter $VERSION (compile: CANN-only; converter: torch)" ./set_env.sh
fi

echo ""
echo "=== package ==="
ls -lh "$DIST/$PKG_NAME.run" | awk '{print "  "$5, $9}'
echo "  安装: bash $DIST/$PKG_NAME.run --extract=/opt/cann-local-lm"
echo "  运行: source /opt/cann-local-lm/set_env.sh && /opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> [out]"
echo "  converter: pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl && cllm-converter --help"
