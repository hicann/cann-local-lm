#!/usr/bin/env bash
# engine OHOS 一键: 编译 + 签名 + 打 whl（产物在 build_ohos/ + dist/）
#
# 流程:
#   1. build_ohos.sh → cmake 编译 + 签名（签名版就地覆盖裸名 .so）→ build_ohos/*.so
#   2. 打 engine python whl（内嵌 .so + libc++_shared.so）→ dist/
#   3. 拷一份到 deploy_ohos/（.so + libc++ + whl，方便设备部署）
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_ohos}"
DIST="${DIST:-${REPO_ROOT}/build_out}"
DEPLOY="${REPO_ROOT}/deploy_ohos"
PY_SRC="${REPO_ROOT}/cann_lm/engine/python"
SCRIPTS_DIR="${REPO_ROOT}/scripts/cann_lm_engine"

# ── 参数 ──
NO_SIGNED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-signed)  NO_SIGNED=1; shift;;
        --clean|-c)  rm -rf "$BUILD_DIR"; shift;;
        *) echo "用法: $0 [--no-signed] [--clean]" >&2; exit 1;;
    esac
done

# ── 1) 签名前置预检 ──
# jar 探测链对齐 sign_ohos.sh: BINARY_SIGN_TOOL env > OHOS_SDK_NATIVE 派生(toolchains/ 与 native/ 平级)
#   > 仓内 tools/ > engine/ohos/
if [[ $NO_SIGNED -eq 0 ]]; then
    command -v java >/dev/null 2>&1 || { echo "ERROR: java 不在 PATH（或用 --no-signed）" >&2; exit 1; }
    JAR="${BINARY_SIGN_TOOL:-}"
    if [ -z "$JAR" ] && [ -n "${OHOS_SDK_NATIVE:-}" ]; then
        for _j in "${OHOS_SDK_NATIVE}/../toolchains/lib/binary-sign-tool.jar" \
                  "${OHOS_SDK_NATIVE}/toolchains/lib/binary-sign-tool.jar"; do
            [ -f "$_j" ] && JAR="$_j" && break
        done
    fi
    if [ -z "$JAR" ]; then
        for _j in "${REPO_ROOT}/tools/binary-sign-tool.jar" "${REPO_ROOT}/cann_lm/engine/ohos/binary-sign-tool.jar"; do
            [ -f "$_j" ] && JAR="$_j" && break
        done
    fi
    [ -z "$JAR" ] && { echo "ERROR: 找不到 binary-sign-tool.jar（设 BINARY_SIGN_TOOL，或 export OHOS_SDK_NATIVE=<sdk>/native，或放 tools/）" >&2; exit 1; }
    echo "[self] java: $(java -version 2>&1 | head -1)"
    echo "[self] jar:  $JAR"
    SIGN_FLAG=""
else
    echo "[self] --no-signed: 跳过签名"
    SIGN_FLAG="--no-signed"
fi

# ── 2) 编译 + 签名 → build_ohos/ ──
echo; echo "=== [1/2] OHOS 交叉编译 + 签名 ==="
bash "${THIS_DIR}/build_ohos.sh" $SIGN_FLAG

# ── 3) 打 engine python whl（内嵌 .so）→ dist/ ──
echo; echo "=== [2/2] 打 engine whl + 拷到 deploy_ohos/ ==="
mkdir -p "$DIST" "$DEPLOY"

# .so 来源: build_ohos/ 裸名=未签名, signed/=签名版 (sign_ohos.sh 产出)
ENGINE_SO_UNSIGNED="$BUILD_DIR/libcann_lm_engine.so"
CPP_SO_UNSIGNED="$BUILD_DIR/libc++_shared.so"
ENGINE_SO_SIGNED="$BUILD_DIR/signed/libcann_lm_engine.so"
CPP_SO_SIGNED="$BUILD_DIR/signed/libc++_shared.so"

[ -f "$ENGINE_SO_UNSIGNED" ] || { echo "ERROR: $ENGINE_SO_UNSIGNED 不存在" >&2; exit 1; }
[ -f "$CPP_SO_UNSIGNED" ] || { echo "ERROR: $CPP_SO_UNSIGNED 不存在" >&2; exit 1; }

PIP_CMD="pip"
[ -x "${REPO_ROOT}/deps/torch-delegate/.venv/bin/pip" ] && PIP_CMD="${REPO_ROOT}/deps/torch-delegate/.venv/bin/pip"

mkdir -p "$PY_SRC/cann_lm/lib"

# ── 2a. 未签名 wheel (含未签名 .so, 版本 0.1.0) ──
echo "  [unsigned] embedding unsigned .so → cann_local_lm_engine-0.1.0-*.whl"
cp "$ENGINE_SO_UNSIGNED" "$PY_SRC/cann_lm/lib/libcann_lm_engine.so"
cp "$CPP_SO_UNSIGNED" "$PY_SRC/cann_lm/lib/libc++_shared.so"
(cd "$SCRIPTS_DIR" && CANN_LM_ENGINE_VERSION="0.1.0" "$PIP_CMD" wheel . --no-deps -w "$DIST") 2>&1 | grep -E "Created|Built|error" | tail -3

# ── 2b. 签名 wheel (含签名 .so, 版本 0.1.0+signed) ──
if [[ $NO_SIGNED -eq 0 ]] && [ -f "$ENGINE_SO_SIGNED" ]; then
    echo "  [signed]   embedding signed .so → cann_local_lm_engine-0.1.0+signed-*.whl"
    cp "$ENGINE_SO_SIGNED" "$PY_SRC/cann_lm/lib/libcann_lm_engine.so"
    cp "$CPP_SO_SIGNED" "$PY_SRC/cann_lm/lib/libc++_shared.so"
    (cd "$SCRIPTS_DIR" && CANN_LM_ENGINE_VERSION="0.1.0+signed" "$PIP_CMD" wheel . --no-deps -w "$DIST") 2>&1 | grep -E "Created|Built|error" | tail -3
fi

# 清理临时文件
rm -rf "$PY_SRC/cann_lm/lib" "$SCRIPTS_DIR/cann_local_lm_engine.egg-info"

echo; echo "=== 完成 ==="
echo "  build_ohos/: .so （编译+签名产物）"
echo "  build_out/: $(ls "$DIST"/cann_local_lm_engine-*.whl 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"
