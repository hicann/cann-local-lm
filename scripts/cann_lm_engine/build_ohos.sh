#!/usr/bin/env bash
# OHOS 交叉编译 + 签名
#
# 双 SDK 路线(ohos.toolchain.cmake 内编译器自动二选一):
#   A) 完整 SDK(华为 command-line-tools 的 native 子树):llvm/bin/clang 是 ARM-native,
#      本机可直接运行,自带 builtins/crt resource dir(无需补 /usr/lib/clang/18)——推荐
#   B) 精简 SDK(OpenHarmony 镜像提取,无 llvm/bin):回退本机 clang-18 + 手动补 builtins
#   - SDK 路径从环境变量 OHOS_SDK_NATIVE 取;否则自动探测(cli_native > ohos_native/native)
#   - 构建器用 Unix Makefiles(本机 make;完整 SDK 的 build-tools 仅含 cmake,无 ninja)
#   - 直接配 cann_lm/engine/ohos(SHARED .so + 部署产物),不走仓库根 CMakeLists
#     (根 CMakeLists 仅 add_subdirectory(cann_lm/engine) host 版,无 OHOS 分支)
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"

# ── 1) SDK 路径(环境变量优先;否则自动探测:完整 SDK > 精简 SDK) ──
if [ -z "${OHOS_SDK_NATIVE:-}" ]; then
    for _cand in /mnt/workspace/ohos_sdk/cli_native \
                 /mnt/workspace/ohos_sdk/ohos_native/native; do
        if [ -d "${_cand}/sysroot" ]; then
            OHOS_SDK_NATIVE="${_cand}"; break
        fi
    done
fi
if [ -z "${OHOS_SDK_NATIVE}" ] || [ ! -d "${OHOS_SDK_NATIVE}/sysroot" ]; then
    echo "ERROR: OHOS_SDK_NATIVE 无效(无 sysroot 子目录): ${OHOS_SDK_NATIVE:-<空>}" >&2
    echo "       export OHOS_SDK_NATIVE=<...>/native,或把 SDK 放到:" >&2
    echo "       /mnt/workspace/ohos_sdk/cli_native (完整 command-line-tools native)" >&2
    echo "       /mnt/workspace/ohos_sdk/ohos_native/native (精简 OpenHarmony 镜像)" >&2
    exit 1
fi
export OHOS_SDK_NATIVE

# ── 2) 编译器:SDK 自带 clang 优先(ARM-native);无则查本机 clang-18(精简路线) ──
# ohos.toolchain.cmake 内同样双路线;此处仅预检给清晰报错
if [ ! -x "${OHOS_SDK_NATIVE}/llvm/bin/clang" ] || [ ! -x "${OHOS_SDK_NATIVE}/llvm/bin/clang++" ]; then
    if ! command -v clang-18 >/dev/null 2>&1 || ! command -v clang++-18 >/dev/null 2>&1; then
        echo "ERROR: SDK 无 ${OHOS_SDK_NATIVE}/llvm/bin/clang,且本机无 clang-18/clang++-18" >&2
        echo "       路线A(推荐):用完整 command-line-tools SDK(自带 ARM-native clang)" >&2
        echo "       路线B(精简):apt install -y --no-install-recommends clang-18 llvm-18" >&2
        exit 1
    fi
    echo "    compiler = 本机 clang-18(精简 SDK 路线)"
else
    echo "    compiler = SDK clang(ARM-native,${OHOS_SDK_NATIVE}/llvm/bin/clang)"
fi

# ── 3) 工具链文件 + engine/ohos 源目录 ──
ENGINE_OHOS_DIR="${REPO_ROOT}/cann_lm/engine/ohos"
TOOLCHAIN="${ENGINE_OHOS_DIR}/ohos.toolchain.cmake"
if [ ! -f "${TOOLCHAIN}" ]; then
    echo "ERROR: 工具链文件不存在: ${TOOLCHAIN}" >&2
    exit 1
fi

# ── 4) 构建目录 ──
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_ohos}"
mkdir -p "${BUILD_DIR}"

# ── 5) CMake configure(交叉编译) ──
echo "=== cmake configure (OHOS cross) ==="
echo "    SDK       = ${OHOS_SDK_NATIVE}"
echo "    toolchain = ${TOOLCHAIN}"
echo "    source    = ${ENGINE_OHOS_DIR}"
echo "    build     = ${BUILD_DIR}"
cmake -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DOHOS_SDK_NATIVE="${OHOS_SDK_NATIVE}" \
    -DOHOS_PLATFORM=OHOS \
    -DGTEST_TAR="${GTEST_TAR:-/mnt/workspace/gtest-1.14.0.tar.gz}" \
    -DCMAKE_BUILD_TYPE=Release \
    -B "${BUILD_DIR}" \
    -S "${ENGINE_OHOS_DIR}"

# ── 6) 编译(make,全量产物) ──
echo "=== build (make -j$(nproc)) ==="
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# ── 7) 归集/校验产物(ohos/CMakeLists 产物平铺在 build 根;此处校验 + libc++ 兜底) ──
echo "=== 产物校验 ==="
for _f in libcann_lm_engine.so libc++_shared.so \
          bbit api_smoke c_api_smoke \
          ut_engine_executor ut_tokenizer ut_c_api ut_scheduler \
          model_executor_ut ut_text_embedding; do
    if [ -f "${BUILD_DIR}/${_f}" ]; then
        echo "  [ok]   ${_f}"
    else
        echo "  [缺]   ${_f}"
    fi
done

# libc++_shared.so:ohos/CMakeLists 已 POST_BUILD 拷贝;若仍缺失则从 SDK 兜底拷
if [ ! -f "${BUILD_DIR}/libc++_shared.so" ]; then
    _SO="$(find "${OHOS_SDK_NATIVE}" -path '*aarch64-linux-ohos*libc++_shared.so' 2>/dev/null | head -n1)"
    if [ -n "${_SO}" ]; then
        cp -f "${_SO}" "${BUILD_DIR}/libc++_shared.so"
        echo "  [兜底拷贝] libc++_shared.so <- ${_SO}"
    else
        echo "  ERROR: SDK 内未找到 aarch64 libc++_shared.so" >&2
        exit 1
    fi
fi

# ── 8) 签名 ──
echo "=== sign binaries ==="
bash "${THIS_DIR}/sign_ohos.sh" "${BUILD_DIR}"

echo "=== build_ohos.sh 完成 ==="
