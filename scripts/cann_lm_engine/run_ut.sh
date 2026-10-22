#!/usr/bin/env bash
# 运行 host UT，gtest XML 报告统一写入 tests/result/（不再散落仓库根目录）。
# 用法：
#   bash scripts/cann_lm_engine/run_ut.sh                        # 跑探测到的构建目录中全部 ut_* 二进制
#   bash scripts/cann_lm_engine/run_ut.sh ut_c_api ut_scheduler  # 只跑指定目标
# 环境变量：
#   UT_BUILD_DIR   UT 二进制所在目录；未设时依次探测 /tmp/llm_ut_host/engine、<repo>/build/engine
#                  （host UT 编法见 PROJECT.md「host UT 编译」节）
# 退出码：0=全部通过；1=有失败或二进制缺失；2=未找到 UT 构建目录。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_DIR="$REPO_ROOT/tests/result"

# ── 探测 UT 构建目录 ──────────────────────────────────────────────────────────
if [ -z "${UT_BUILD_DIR:-}" ]; then
  for _d in /tmp/llm_ut_host/engine "$REPO_ROOT/build/engine"; do
    if find "$_d" -maxdepth 1 -type f -name 'ut_*' -executable -print -quit 2>/dev/null | grep -q .; then
      UT_BUILD_DIR="$_d"
      break
    fi
  done
fi
if [ -z "${UT_BUILD_DIR:-}" ] || [ ! -d "${UT_BUILD_DIR:-}" ]; then
  echo "ERROR: 未找到 UT 构建目录。请先按 PROJECT.md「host UT 编译」节构建，或 export UT_BUILD_DIR=<ut二进制目录>" >&2
  exit 2
fi
echo "UT 构建目录: $UT_BUILD_DIR"
echo "XML 报告目录: $RESULT_DIR"
mkdir -p "$RESULT_DIR"

# ── 组装待跑目标 ──────────────────────────────────────────────────────────────
if [ $# -gt 0 ]; then
  BINS=()
  for name in "$@"; do BINS+=("$UT_BUILD_DIR/$name"); done
else
  # 仅可执行二进制（排除同名 .xml 报告）
  mapfile -t BINS < <(find "$UT_BUILD_DIR" -maxdepth 1 -type f -name 'ut_*' -executable | sort)
fi
if [ ${#BINS[@]} -eq 0 ]; then
  echo "ERROR: $UT_BUILD_DIR 下无 ut_* 可执行二进制" >&2
  exit 2
fi

# ── 运行 ─────────────────────────────────────────────────────────────────────
FAILED=()
for bin in "${BINS[@]}"; do
  name="$(basename "$bin")"
  if [ ! -x "$bin" ]; then
    echo "==> $name  ERROR: 二进制不存在或不可执行: $bin" >&2
    FAILED+=("$name(缺二进制)")
    continue
  fi
  echo "==> $name"
  # gtest 对已存在的输出文件会自动加 _1/_2 后缀，先删旧报告保证文件名稳定
  rm -f "$RESULT_DIR/$name.xml"
  if "$bin" --gtest_output="xml:$RESULT_DIR/$name.xml"; then
    echo "    PASS  报告: tests/result/$name.xml"
  else
    echo "    FAIL  报告: tests/result/$name.xml"
    FAILED+=("$name")
  fi
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "全部通过（${#BINS[@]} 项）。XML 报告: tests/result/"
  exit 0
else
  echo "失败 ${#FAILED[@]}/${#BINS[@]} 项: ${FAILED[*]}（详见 tests/result/ 对应 XML）"
  exit 1
fi
