#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# package_run_ohos.sh — 打 engine OHOS 部署包 .run 自解压安装包 (makeself)
#
# engine 是纯 C++(7 个 __init__.py 仅 namespace 占位), 可运行产物为 OHOS
# 交叉编译的 aarch64-musl 二进制 (libcann_lm_engine.so + libc++_shared.so
# + 9 可执行, 已自签名 .codesign)。产物不进 whl (平台标记硬伤: OHOS 不在
# wheel 平台 tag), 走 .run 单文件分发, 与 cann_lm_compile 同机制。
#
# 输入: deploy_ohos/ (package_ohos.sh 从 build_ohos/ 签名产物组装, 已含
#       run.sh/README.md/testcase.json, 见 package_ohos.sh)
# 流程: 校验 deploy_ohos/ → makeself → dist/cann_lm_engine_ohos-<ver>-<arch>.run
# 安装(鸿蒙 PC 上): bash <pkg>.run --target /data/local/tmp/cann_lm_engine
#   (或解压后整个目录推到设备) → ./run.sh 跑全部 UT/bbit/smoke
#
# 环境覆盖: VERSION / ARCH / THIRD_PARTY / OUT_DIR(默认 deploy_ohos)
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-$(uname -m)}"
PKG_NAME="cann_lm_engine_ohos-${VERSION}-${ARCH}"
OUT_DIR="${OUT_DIR:-$REPO/deploy_ohos}"
STAGE="$REPO/build/package_stage/$PKG_NAME"
DIST="$REPO/dist"
THIRD_PARTY="${THIRD_PARTY:-$REPO/output/third_party}"
MAKESELF="${THIRD_PARTY}/makeself/makeself.sh"

# ── 1. 校验 deploy_ohos/ ──
echo "### [1/3] 校验 deploy_ohos/ → $OUT_DIR"
for f in libcann_lm_engine.so libc++_shared.so run.sh; do
    [[ -f "$OUT_DIR/$f" ]] || { echo "ERROR: 缺 $OUT_DIR/$f (先 build_ohos.sh + package_ohos.sh)" >&2; exit 1; }
done
exe=(bbit api_smoke c_api_smoke model_executor_ut ut_c_api ut_engine_executor ut_scheduler ut_text_embedding ut_tokenizer)
missing=0
for t in "${exe[@]}"; do
    [[ -f "$OUT_DIR/$t" ]] || { echo "  (缺可执行, 跳过): $t" >&2; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "ERROR: deploy_ohos/ 产物不全, 先 ./build_ohos.sh" >&2; exit 1; }

# ── 2. stage (直接搬整个 deploy_ohos/, 保持 run.sh 相对路径可用) ──
echo "### [2/3] stage → $STAGE"
rm -rf "$STAGE"
cp -a "$OUT_DIR" "$STAGE"
chmod +x "$STAGE"/run.sh "$STAGE"/bbit "$STAGE"/*ut "$STAGE"/api_smoke "$STAGE"/c_api_smoke 2>/dev/null || true

# ── 3. makeself → .run ──
echo "### [3/3] makeself → $DIST/$PKG_NAME.run"
mkdir -p "$DIST"
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
        "cann_lm_engine $VERSION (OHOS aarch64-musl, self-signed)" ./run.sh
else
    "$MAKESELF_BIN" --notemp "$STAGE" "$DIST/$PKG_NAME.run" \
        "cann_lm_engine $VERSION (OHOS aarch64-musl, self-signed)" ./run.sh
fi

echo ""
echo "=== package ==="
ls -lh "$DIST/$PKG_NAME.run" | awk '{print "  "$5, $9}'
echo "  安装(鸿蒙 PC): bash $DIST/$PKG_NAME.run --target /data/local/tmp/cann_lm_engine"
echo "  运行: cd /data/local/tmp/cann_lm_engine && ./run.sh   (或 hdc file send 推到设备)"