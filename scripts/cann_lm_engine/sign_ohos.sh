#!/usr/bin/env bash
# 对 OHOS 交叉编译产物逐一自签名(-selfSign 1,照 harmony_acl)。
# 用法: ./sign_ohos.sh [BUILD_DIR] [JAR]
#   BUILD_DIR 产物目录,默认 ./build_ohos
#   JAR       binary-sign-tool.jar 路径,默认 ./cann_lm/engine/ohos/binary-sign-tool.jar
set -euo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build_ohos}"
# jar 优先用 tools/(见 tools/README.md),回退到 cann_lm/engine/ohos/
JAR="${2:-}"
if [ -z "$JAR" ]; then
  if [ -f "${REPO_ROOT}/tools/binary-sign-tool.jar" ]; then
    JAR="${REPO_ROOT}/tools/binary-sign-tool.jar"
  else
    JAR="${REPO_ROOT}/cann_lm/engine/ohos/binary-sign-tool.jar"
  fi
fi

if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: java 不在 PATH" >&2; exit 1
fi
[ -f "$JAR" ] || { echo "ERROR: 找不到 jar: $JAR" >&2; exit 1; }
[ -d "$BUILD_DIR" ] || { echo "ERROR: 产物目录不存在: $BUILD_DIR" >&2; exit 1; }

# 待签名产物:两个 .so + 9 个可执行
SO_FILES=(libcann_lm_engine.so libc++_shared.so)
EXE_FILES=(model_executor_ut ut_engine_executor ut_tokenizer ut_c_api ut_scheduler ut_text_embedding bbit api_smoke c_api_smoke)

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
sign_one libcann_lm_engine.so libcann_lm_engine_signed.so
sign_one libc++_shared.so     libc++_shared_signed.so
echo "-- 可执行 --"
for t in "${EXE_FILES[@]}"; do sign_one "$t" "${t}_signed"; done

echo "=== 完成。签名产物在 $BUILD_DIR (*_signed / *_signed.so) ==="
