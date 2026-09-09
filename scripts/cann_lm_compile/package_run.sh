#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# package_run.sh — 打 cann_lm_compile .run 自解压安装包 (makeself)
#
# 流程: build (--no-tests) → stage (binary+.so+cfg+set_env) → patchelf rpath → makeself
# 产物: dist/cann_lm_compile-<ver>-<arch>.run
# 包内容: cann_lm_compile / lib/libtorch_delegate_backend.so / fusion_autoub_off.cfg / set_env.sh
# 安装: bash cann_lm_compile-<ver>-<arch>.run --target /opt/cann-lm
# 运行: source /opt/cann-lm/set_env.sh (需先 source CANN set_env.sh) → /opt/cann-lm/cann_lm_compile ...
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
DIST="$REPO/dist"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
MAKESELF="${THIRD_PARTY}/makeself/makeself.sh"

# ── 1. build (cann_lm_compile + .so, 跳 tests) ──
echo "### [1/4] build (build_cpp.sh --no-tests --target cann_lm_compile)"
bash "$REPO/scripts/cann_lm_compile/build_cpp.sh" --no-tests --target cann_lm_compile

BIN="$REPO/build/cann_lm_compile"
SO="$REPO/build/libtorch_delegate_backend.so"
CFG="$REPO/cann_lm/converter/export/csrc/tools/fusion_autoub_off.cfg"
for f in "$BIN" "$SO" "$CFG"; do
    [[ -f "$f" ]] || { echo "ERROR: 缺产物 $f" >&2; exit 1; }
done

# ── 2. stage (binary + lib/ + cfg + set_env.sh) ──
echo "### [2/4] stage → $STAGE"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib"
cp "$BIN" "$STAGE/cann_lm_compile"
cp "$SO"  "$STAGE/lib/libtorch_delegate_backend.so"
cp "$CFG" "$STAGE/fusion_autoub_off.cfg"

# 生成 set_env.sh (安装后 source; 设 LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE)
cat > "$STAGE/set_env.sh" <<'EOF'
#!/bin/bash
# set_env.sh — cann_lm_compile 运行 env (安装后 source)
# 前置: 安装机需先 source CANN set_env.sh (提供 CANN lib64 / simulator / driver / opp)
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
export GE_FUSION_SWITCH_FILE="${INSTALL_DIR}/fusion_autoub_off.cfg"
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
echo "[cann_lm_compile] env ready (INSTALL_DIR=$INSTALL_DIR, SOC=${SOC_VERSION})"
echo "  运行: $INSTALL_DIR/cann_lm_compile <model_json> <weights_index> [output_name]"
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
echo "  安装: bash $DIST/$PKG_NAME.run --target /opt/cann-lm"
echo "  运行: source /opt/cann-lm/set_env.sh && /opt/cann-lm/cann_lm_compile <ep_json> <weight_info> [out]"
