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
