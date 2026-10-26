# CANN-LM

CANN-LM：CANN 上的 LLM 推理引擎（C++）+ 模型转换工具链（converter）+ Python 绑定。

## 测试分层

**UT（单元测试）**:
单组件纯逻辑测试，不依赖外部运行时/设备/真后端。置于 `tests/`。
_Avoid_: unit test、单元 case

**ST / 端到端**:
跨组件集成、需真后端(CANN)/设备、或端到端跑通的用例。置于 `examples/`。
_Avoid_: integration test、e2e、系统测试

**bbit**:
黑盒端到端集成测试，经 C API 驱动完整推理流。属 ST，置 `examples/`。

**stub_ndk**:
host 回归用的 NDK 桩（probe_* 探针 + 桩 .so），属测试基础设施而非端到端，留 `tests/`。

## 构建环境（docs/adr/0002）

**converter 环境（①）**:
HF→epair + 量化，Linux CPU venv。已就绪。

**compile 环境（②）**:
epair→OM，C++ 编译 + OM 编译，需真 CANN。已就绪。

**generate 环境（③）**:
鸿蒙交叉编译产物构建。原 `generate/` 预留占位已删，由 `scripts/cann_lm_engine/build_ohos.sh` 承载。

**npu 环境（④）**:
verify-runtime，CANN 轻量化工具 GPTQ / NPU 出字 / OM 验证。已就绪。

## 依赖管理

**vendored 源码**:
直接入库的第三方源码（如 nlohmann/json.hpp），置于 `cmake/third_party/<lib>/`，经 include 路径直接消费。
_Avoid_: 第三方源码、vendored 依赖

**find module（Find*.cmake）**:
CMake `find_package` 模块，定位预构建的第三方库。置于 `cmake/third_party/modules/`（engine/converter 共享）或 `cmake/modules/`（converter 专用）。

## 目录 context

**engine context**:
`cann_lm/engine/`（C++ 推理引擎源码 + `python/` 绑定）+ `docs/cann_lm_engine/`（对应文档）。fork 主线，不依赖 torch。

**converter context**:
`cann_lm/converter/`（epair→OM 工具 + backend lib）+ `docs/cann_lm_converter/`。依赖 torch+CANN。
