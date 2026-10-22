#!/usr/bin/env bash
# 从 build_ohos/ 的签名产物生成 deploy_ohos/ 部署包。
# 做的事:
#   - 签名版 .so 重命名为加载器期望名(libcann_lm_engine.so / libc++_shared.so)
#   - 签名版可执行去掉 _signed 后缀(直接可跑)
#   - 恢复可执行位(签名产物为 644;git 跟踪模式,设备 pull 后直接可执行)
#   - 生成 run.sh(自动 chmod + LD_LIBRARY_PATH 跑全部 UT/bbit/smoke)
# 用法: ./package_ohos.sh [BUILD_DIR] [OUT_DIR]
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build_ohos}"
OUT_DIR="${2:-${REPO_ROOT}/deploy_ohos}"

[ -d "$BUILD_DIR" ] || { echo "ERROR: 构建目录不存在: $BUILD_DIR (先 ${THIS_DIR}/build_ohos.sh)" >&2; exit 1; }

# 待打包产物:(构建目录内签名文件名 -> 部署目录内目标名)
# 全量 11 产物(主库 + libc++ + 6 UT + bbit + 2 smoke),与 build_ohos.sh 产物清单一致。
declare -a MAP=(
  "libcann_lm_engine_signed.so:libcann_lm_engine.so"
  "libc++_shared_signed.so:libc++_shared.so"
  "model_executor_ut_signed:model_executor_ut"
  "ut_engine_executor_signed:ut_engine_executor"
  "ut_tokenizer_signed:ut_tokenizer"
  "ut_c_api_signed:ut_c_api"
  "ut_scheduler_signed:ut_scheduler"
  "ut_text_embedding_signed:ut_text_embedding"
  "bbit_signed:bbit"
  "api_smoke_signed:api_smoke"
  "c_api_smoke_signed:c_api_smoke"
)

echo "=== 打包部署包 deploy_ohos/ ==="
echo "  源: $BUILD_DIR"
echo "  目标: $OUT_DIR"
# testcase.json 为手工维护的部署配置(bbit flow 模式消费),不在签名产物清单内——
# rm 前暂存、打包后回放,避免每次重打包丢失(默认搜索 ./testcase.json,设备 run.sh 依赖)。
KEEP_TESTCASE=""
if [ -f "$OUT_DIR/testcase.json" ]; then
  KEEP_TESTCASE="$(mktemp)"
  cp "$OUT_DIR/testcase.json" "$KEEP_TESTCASE"
fi
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

missing=0
for pair in "${MAP[@]}"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [ -f "$BUILD_DIR/$src" ]; then
    cp "$BUILD_DIR/$src" "$OUT_DIR/$dst"
    # 恢复可执行位:签名产物 cp 过来是 644,而 git 跟踪文件模式(索引中二进制为 755),
    # 设备 git pull 后也应直接可执行(run.sh 里的 chmod 仅为兜底)。
    # libc++_shared.so 除外:索引中即 644,.so 由 dlopen 加载无需执行位。
    [ "$dst" = "libc++_shared.so" ] || chmod +x "$OUT_DIR/$dst"
    echo "  ✓ $src -> $dst"
  else
    echo "  ✗ 缺失: $BUILD_DIR/$src" >&2; missing=1
  fi
done
[ $missing -eq 0 ] || { echo "ERROR: 有产物缺失,先 ./build_ohos.sh" >&2; exit 1; }

# 可执行 + run.sh
cat > "$OUT_DIR/run.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$(pwd):${LD_LIBRARY_PATH:-}"

UTS=(model_executor_ut ut_engine_executor ut_tokenizer ut_c_api ut_scheduler ut_text_embedding)
SMOKE=(api_smoke c_api_smoke)
chmod +x bbit "${UTS[@]}" "${SMOKE[@]}" 2>/dev/null || true

run_args() { echo "===== $* ====="; local rc=0; "$@" || rc=$?; echo "(exit=$rc)"; }

if [ "$#" -gt 0 ]; then
  for t in "$@"; do run_args ./"$t"; done
  exit 0
fi

echo "── 单元测试 UT ──"
for t in "${UTS[@]}"; do run_args ./"$t"; done
echo "── 黑盒集成 bbit ──"
run_args ./bbit --check
run_args ./bbit
echo "── 冒烟 ──"
for t in "${SMOKE[@]}"; do run_args ./"$t"; done
echo "===== 全部完成 ====="
RUNEOF
chmod +x "$OUT_DIR/run.sh"

# 回放手工维护的 testcase.json(见上方暂存说明)
if [ -n "$KEEP_TESTCASE" ]; then
  cp "$KEEP_TESTCASE" "$OUT_DIR/testcase.json"
  rm -f "$KEEP_TESTCASE"
  chmod 644 "$OUT_DIR/testcase.json"   # mktemp 为 600,恢复 git 跟踪的 644
  echo "  ✓ testcase.json 保留(手工维护,重打包不重生)"
fi

# 生成 README(随包分发,设备上可查;每次打包重生)
cat > "$OUT_DIR/README.md" <<'READMEEOF'
# deploy_ohos/ — 鸿蒙部署包

鸿蒙(aarch64-linux-ohos / musl)PC 上可直接运行的 CANN-LM engine 产物,已自签名(`.codesign` 段)。由 `scripts/cann_lm_engine/package_ohos.sh` 从 `build_ohos/` 生成。

## 文件
| 文件 | 作用 |
|------|------|
| `libcann_lm_engine.so` | engine SHARED 库(签名版,已按 DT_NEEDED 命名) |
| `bbit` | 黑盒集成(`--check` + 完整流程) |
| `testcase.json` | bbit flow 模式配置(手工维护,重打包保留)。指向设备 `/data/local/tmp/llm_engine_bbit/` 下的 model.omc / model.json / tokenizer.json / SubGraph_0.weight |
| `api_smoke`/`c_api_smoke` | C++/C API 冒烟 |
| `run.sh` | 一键跑全部(自动 chmod + LD_LIBRARY_PATH) |

`libc.so`(musl)由鸿蒙系统提供,不在包内。

## 在鸿蒙 PC 上验证
```bash
# 1) 传到设备(任选):
#    hdc file send deploy_ohos /data/local/tmp/   # 从连了USB的机器
#    或本机 git pull(本目录已入库)
cd /data/local/tmp/deploy_ohos        # 或你放置的路径
chmod +x * run.sh
./run.sh                               # 跑全部 UT + bbit + smoke
./run.sh ut_c_api                      # 或只跑单个
```
预期:UT 全 PASSED(与 Linux 基线一致);bbit --check 过、完整流程跑通。Generate 返回 FAILURE 或 INVALID_STATE 属正常(后者 = tokenizer 显式供给但设备路径不存在,快速失败)。

## 注意
- selfSign 是开发签名,设备需**开发者模式**才允许运行自签名二进制。
- 目录需可写可执行(`/data/local/tmp` 通常可以;sdcard 可能 noexec)。
- `.so` 名字不可改(加载器按 DT_NEEDED 精确查找)。

## 重新生成 / 移除
重新生成(改源码后):`./scripts/cann_lm_engine/build_ohos.sh && ./scripts/cann_lm_engine/package_ohos.sh`
整体移除:`git rm -r deploy_ohos/`(源码在 engine/,Linux 构建不受影响)
READMEEOF

echo
echo "=== deploy_ohos/ 内容 ==="
ls -lh "$OUT_DIR"
echo
echo "下一步:把整个 deploy_ohos/ 推到鸿蒙设备,进目录 ./run.sh"
