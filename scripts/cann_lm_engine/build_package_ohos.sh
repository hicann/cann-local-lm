#!/usr/bin/env bash
# CANN-LM engine OHOS 一键编译 + 签名 + 打包(产物 → deploy_ohos/)
#
# 适配当前精简 SDK 环境:SDK 路径 / 本机 clang-18 / GTest / 签名均由 build_ohos.sh
# 内部处理——SDK 走环境变量 OHOS_SDK_NATIVE、编译器用本机 clang-18
# (ohos.toolchain.cmake find_program)、GTest tarball 由 engine/ohos/CMakeLists
# FetchContent、签名 jar 在 tools/。本脚本仅做: --clean 可选 + java/jar 预检
# + 编排 build_ohos.sh(编译+签名) → package_ohos.sh(打包)。
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"

# ── 0) 参数 ──
if [ "${1:-}" = "--clean" ] || [ "${1:-}" = "-c" ]; then
    BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_ohos}"
    rm -rf "${BUILD_DIR}"
    echo "[self] --clean: 已删除 ${BUILD_DIR},全量重建"
elif [ -n "${1:-}" ]; then
    echo "用法: $0 [--clean]" >&2
    exit 1
fi

# ── 1) 签名前置预检(build_ohos.sh 末尾会调 sign_ohos.sh,签名需 java + jar) ──
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: java 不在 PATH(签名步骤会失败)" >&2
    exit 1
fi
JAR=""
for _j in "${REPO_ROOT}/tools/binary-sign-tool.jar" \
          "${REPO_ROOT}/cann_lm/engine/ohos/binary-sign-tool.jar"; do
    if [ -f "$_j" ]; then JAR="$_j"; break; fi
done
if [ -z "$JAR" ]; then
    echo "ERROR: 找不到 binary-sign-tool.jar (应放 ${REPO_ROOT}/tools/,见 tools/README.md)" >&2
    exit 1
fi
echo "[self] java : $(java -version 2>&1 | head -1)"
echo "[self] jar  : $JAR"

# ── 2) 编译 + 签名(SDK/clang-18/GTest 均由 build_ohos.sh 内部处理) ──
echo
echo "=== [1/2] OHOS 交叉编译 + 签名(build_ohos.sh) ==="
bash "${THIS_DIR}/build_ohos.sh"

# ── 3) 打包 ──
echo
echo "=== [2/2] 打包部署包(package_ohos.sh → deploy_ohos/) ==="
bash "${THIS_DIR}/package_ohos.sh"

# ── 4) 收尾 ──
echo
echo "=== 一键完成: deploy_ohos/ ==="
ls -lh "${REPO_ROOT}/deploy_ohos"
echo
echo "推到鸿蒙设备: hdc file send deploy_ohos /data/local/tmp/ && cd /data/local/tmp/deploy_ohos && ./run.sh"
