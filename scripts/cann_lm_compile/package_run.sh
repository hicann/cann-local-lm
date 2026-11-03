#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# package_run.sh — 打 cann_lm_compile .run 自解压安装包 (makeself)
#
# 流程: build (--no-tests) → stage (binary+.so+cfg+set_env) → patchelf rpath → makeself
# 产物: build_out/cann_lm_compile-<ver>-<arch>.run
# 包内容: cann_lm_compile / lib/libtorch_delegate_backend.so / fusion_autoub_off.cfg / set_env.sh
# 安装: bash cann_lm_compile-<ver>-<arch>.run --extract=/opt/cann-local-lm
# 运行: source /opt/cann-local-lm/set_env.sh (需先 source CANN set_env.sh) → /opt/cann-local-lm/cann_lm_compile ...
# 注: 工具不依赖 torch/venv, 安装机只需 CANN
#
# 环境覆盖: VERSION / ARCH / THIRD_PARTY
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-$(uname -m)}"
PKG_NAME="cann_lm_compile-${VERSION}-${ARCH}"
STAGE="$REPO/build/package_stage/$PKG_NAME"
DIST="${DIST:-$REPO/build_out}"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
MAKESELF="${THIRD_PARTY}/makeself/makeself.sh"

# ── 1. build (cann_lm_compile + .so, 跳 tests) ──
echo "### [1/4] build (build_cpp.sh --no-tests --target cann_lm_compile)"
bash "$REPO/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile

BIN="$REPO/build/cann_lm_compile"
SO="$REPO/build/libtorch_delegate_backend.so"
CFG="$REPO/cann_local_lm/converter/export/csrc/tools/fusion_autoub_off.cfg"
for f in "$BIN" "$SO" "$CFG"; do
    [[ -f "$f" ]] || { echo "ERROR: 缺产物 $f" >&2; exit 1; }
done

# ── 2. stage (binary + lib/ + cfg + op_cpu/ + op_pass/ + set_env.sh) ──
echo "### [2/4] stage → $STAGE"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib"
cp "$BIN" "$STAGE/cann_lm_compile"
cp "$SO"  "$STAGE/lib/libtorch_delegate_backend.so"
cp "$CFG" "$STAGE/fusion_autoub_off.cfg"

# 量化 mobile 链所需的 GE python 插件 (Bitcast/TransData HostCpu + QBMV4 x2 TransData pass)
for d in op_cpu op_pass; do
    src="$REPO/cann_local_lm/converter/export/$d"
    [[ -d "$src" ]] || { echo "ERROR: 缺插件目录 $src" >&2; exit 1; }
    cp -r "$src" "$STAGE/$d"
    find "$STAGE/$d" -name __pycache__ -type d -prune -exec rm -rf {} +
done

# 生成 set_env.sh (安装后 source; 设 LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE;
#   --mobile 额外设量化 mobile 链的 GE 开关与插件路径)
cat > "$STAGE/set_env.sh" <<'EOF'
#!/bin/bash
# set_env.sh — cann_lm_compile 运行 env (安装后 source)
# 前置: 安装机需先 source CANN set_env.sh (提供 CANN lib64 / simulator / driver / opp)
#
# 用法: source /opt/cann-local-lm/set_env.sh
# SOC 为 Kirin 系 (KirinX90/Kirin9030/...) 时自动进入 mobile 分支
# (外置权重 + 权重共享算子 + GE python 插件 op_cpu/op_pass), 无需额外参数。
# 切换目标: export SOC_VERSION=Ascend910_9362 (或任意非 Kirin 值) 即关闭 mobile 分支。
#
# ── 环境变量说明 ────────────────────────────────────────────────────────────
# SOC_VERSION                目标 SoC (ge.socVersion)。Kirin 系 → mobile 编译链
#                            (.omc 产物 + MobileModelHelper 路径); 其余 → 通用 .om 路径。
# GE_FUSION_SWITCH_FILE      GE 图融合开关配置。默认包内 fusion_autoub_off.cfg
#                            (关掉 auto-UB 融合, 避免 mobile 路径出无 kernel 的融合子图)。
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
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"

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
        _MOBILE=1
        ;;
    *)
        _MOBILE=0
        ;;
esac

echo "[cann_lm_compile] env ready (INSTALL_DIR=$INSTALL_DIR, SOC=${SOC_VERSION})"
echo "  运行: $INSTALL_DIR/cann_lm_compile <model_json> <weights_index> [output_name]"
if [[ "$_MOBILE" -eq 1 ]]; then
    echo "  mobile: GE_EXTERNAL_WEIGHT=${GE_EXTERNAL_WEIGHT} SHARE_OPS=${GE_MOBILE_WEIGHT_SHARE_OPS}"
    echo "  mobile: ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
    echo "  mobile: ASCEND_GE_PY_PASS_PATH=${ASCEND_GE_PY_PASS_PATH}"
fi
unset _MOBILE
EOF
chmod +x "$STAGE/set_env.sh"

# ── 3. patchelf .so rpath → $ORIGIN (去 build 机绝对路径, 可重定位; 运行靠 set_env 的 LD_LIBRARY_PATH) ──
echo "### [3/4] patchelf --set-rpath \$ORIGIN (libtorch_delegate_backend.so)"
if command -v patchelf >/dev/null; then
    patchelf --set-rpath '$ORIGIN' "$STAGE/lib/libtorch_delegate_backend.so"
else
    echo "  (patchelf 未装, 跳过; 运行靠 set_env.sh 的 LD_LIBRARY_PATH)"
fi

# ── 4. makeself → .run ──
echo "### [4/4] makeself → $DIST/$PKG_NAME.run"
mkdir -p "$DIST"
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
        "cann_lm_compile $VERSION (CANN-only, no torch)" ./set_env.sh
else
    "$MAKESELF_BIN" --notemp "$STAGE" "$DIST/$PKG_NAME.run" \
        "cann_lm_compile $VERSION (CANN-only, no torch)" ./set_env.sh
fi

echo ""
echo "=== package ==="
ls -lh "$DIST/$PKG_NAME.run" | awk '{print "  "$5, $9}'
echo "  安装: bash $DIST/$PKG_NAME.run --extract=/opt/cann-local-lm"
echo "  运行: source /opt/cann-local-lm/set_env.sh && /opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> [out]"
