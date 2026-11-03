#!/usr/bin/env bash
# 对 OHOS 交叉编译产物逐一自签名(-selfSign 1,照 harmony_acl)。
# 用法: ./sign_ohos.sh [BUILD_DIR] [JAR]
#   BUILD_DIR 产物目录,默认 ./build_ohos
#   JAR       binary-sign-tool.jar 路径,缺省按下述探测链自动取
set -euo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build_ohos}"
# jar 探测链(优先级从高到低,命中即用):
#   ① $2 命令行参数 / $BINARY_SIGN_TOOL 环境变量 (显式指定,最高优先)
#   ② $OHOS_SDK_NATIVE 派生: <sdk>/../toolchains/lib/binary-sign-tool.jar
#      —— jar 是 command-line-tools SDK 自带物, 与 native/ 平级 (toolchains/
#      是 openharmony/ 的兄弟目录); 交叉编译机必装 SDK, 零拷贝。
#   ③ 仓内 tools/binary-sign-tool.jar (旧约定, 见 tools/README.md)
#   ④ 仓内 cann_local_lm/engine/ohos/binary-sign-tool.jar (最老回退)
# 闭源二进制不入库, 探测链保证: 只要装了 SDK 就能签, 无需任何手工拷贝。
JAR="${2:-${BINARY_SIGN_TOOL:-}}"
if [ -z "$JAR" ] && [ -n "${OHOS_SDK_NATIVE:-}" ]; then
  for _jar in "${OHOS_SDK_NATIVE}/../toolchains/lib/binary-sign-tool.jar" \
              "${OHOS_SDK_NATIVE}/toolchains/lib/binary-sign-tool.jar"; do
    [ -f "$_jar" ] && JAR="$_jar" && break
  done
fi
if [ -z "$JAR" ]; then
  if [ -f "${REPO_ROOT}/tools/binary-sign-tool.jar" ]; then
    JAR="${REPO_ROOT}/tools/binary-sign-tool.jar"
  else
    JAR="${REPO_ROOT}/cann_local_lm/engine/ohos/binary-sign-tool.jar"
  fi
fi

if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: java 不在 PATH" >&2; exit 1
fi
[ -f "$JAR" ] || { echo "ERROR: 找不到 jar: $JAR" >&2; exit 1; }
[ -d "$BUILD_DIR" ] || { echo "ERROR: 产物目录不存在: $BUILD_DIR" >&2; exit 1; }

# 待签名产物:两个 .so（UT 不交叉编译,不签名）
SO_FILES=(libcann_lm_engine.so libc++_shared.so)

# 签名产物放独立子目录 signed/ —— 裸名原件永不被改写。
# 契约(build_ohos.sh / build_package_ohos.sh 消费):
#   build_ohos/<name>        = 编译原产物 (未签名) — 永远是原文, 模式切换无黏滞
#   build_ohos/signed/<name> = 签名版 (本脚本产出; 存在即"当前是签名模式")
SIGNED_SUBDIR="signed"
mkdir -p "${BUILD_DIR}/${SIGNED_SUBDIR}"


sign_one() {
  local infile="$1" outfile="$2" log rc=0
  if [ ! -f "$BUILD_DIR/$infile" ]; then
    echo "  跳过(不存在): $infile"; return 0
  fi
  log="$(mktemp)"
  # 成功则静默;失败则把 jar 侧诊断(indent)打到 stderr,便于排障
  java -jar "$JAR" sign -inFile "$BUILD_DIR/$infile" -outFile "$BUILD_DIR/$outfile" -selfSign 1 \
    >"$log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  ✓ $infile -> $outfile"
  else
    echo "  ✗ 签名失败: $infile" >&2
    sed 's/^/    /' "$log" >&2
  fi
  rm -f "$log"
  return "$rc"
}

echo "=== 签名 OHOS 产物 ==="
echo "  jar      : $JAR"
echo "  产物目录: $BUILD_DIR"
echo "  java     : $(java -version 2>&1 | head -1)"

echo "-- .so --"
sign_one libcann_lm_engine.so "${SIGNED_SUBDIR}/libcann_lm_engine.so"
sign_one libc++_shared.so     "${SIGNED_SUBDIR}/libc++_shared.so"

echo "=== 完成。签名产物在 $BUILD_DIR/$SIGNED_SUBDIR/ (裸名原件未动) ==="
