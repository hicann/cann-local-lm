# cann-device-lm

简体中文 | [English](README_en.md)

## 🔥 Latest News

- [2026/10] converter当前能力：HuggingFace模型 → epair导出（torch.export + FX→GE转换 + 权重分离），支持W4A8量化（GPTQ离线 + 运行时动态量化）。引擎当前能力：三层推理引擎架构，支持Continuous Batching、KV Cache管理、异步流式推理、多轮续接、请求取消，适配Kirin平台，提供C++ API以及python API入口。

## 🚀 概述

cann-device-lm旨在支持开发者在华为Kirin芯片上跑通大模型推理，提供**模型量化、模型编译、模型推理**端到端能力。将HuggingFace模型导出为epair格式（IR + 权重分离），经C++ 后端加载编译为omc，最终在Kirin芯片上执行推理。

## ⬇️ 源码下载

环境准备好后，下载与CANN版本配套的分支源码，通用命令如下，`${tag_version}` 替换为分支标签名：

```bash
# 通用命令：git clone -b ${tag_version} https://gitcode.com/cann/cann-device-lm.git
git clone -b 2026_09_22 https://gitcode.com/cann/cann-device-lm.git
```

> 说明：若环境中已存在配套分支源码，可跳过本步骤。

## 📦 支持模型与芯片

| 模型　　| 架构特点　　　　　　　　　　　　　　　　　　　　　　 | 支持芯片　|
| ---------| ------------------------------------------------------|-----------|
| gemma4　| KV共享 + sliding window + PLE (Per-Layer Embedding) | KirinX90　|

## 📂 目录结构

```
cann-device-lm/
├── cann_lm/                  # 核心源码
│   ├── converter/            # 导出工具链 (HF 模型 → epair → omc 编译)
│   ├── engine/               # 推理引擎 (omc 加载 + 推理执行)
│   └── models/               # 模型定义 (gemma4 + fused_ops 算子)
├── torch_delegate/           # torch-delegate (已合入): epair 序列化/加载 + C++ backend + .venv 构建
│   ├── csrc/                 # C++ 后端 (epair archive 加载 → GE Graph → OM)
│   ├── torch_delegate/       # Python 包 (api.py + converter/ + epair/ + core/)
│   ├── cmake/                # torch-delegate 专用 CMake 模块
│   ├── docs/                 # torch-delegate 设计文档 + 构建指南
│   ├── tests/                # torch-delegate 单元测试 (C++ GTest + Python pytest)
│   ├── build.sh              # 一站式构建 (deps → cpp → wheel → .run)
│   └── examples/             # 示例 (导出加载/精度对比/异步执行)
├── docs/                     # 设计文档 (converter + engine + 构建 + 版本演进)
├── examples/                 # 端到端用例 (omc 编译 + NPU 出字验证)
├── scripts/                  # 构建脚本 (build.sh + build.conf + set_env*.sh + cann_lm_compile/)
├── tests/                    # 单元测试 (cpp/ + python/)
├── ci/                       # CI/CD
└── cmake/                    # CMake 公共模块 (modules/ + third_party/ + TorchDelegatePackage)
```

### 各目录介绍

| 目录　　　　　　　　　　　　 | 职责　　　　　　　　　　　 | 关键文件　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------------------------| ----------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **cann_lm/converter/**　　　 | HF模型 → epair → omc导出工具链 | `cann_lm_converter.py`(CLI入口) / `loader.py`(加载) / `saver.py`(保存) / `quant/`(量化:swap/materialize/auto_quant/dopt_wrapper) / `export/`(ge_converter + C++ csrc) |
| **cann_lm/engine/**　　　　　| omc推理引擎　　　　　　　| `api/`(Python接口) / `core/`(核心逻辑) / `backend/`(C++ 后端) / `pipeline/`(推理流水线)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cann_lm/models/**　　　　　|模型定义 + 自定义算子　| `gemma4/`(Gemma4ForCausalLM + export_wrapper)  / `fused_ops/`(cann_lm:: + npu:: 算子schema+fake-meta)　　　　　　　　　　　　　　　　　　　　　|
| **torch_delegate/**　　　　　| epair序列化/加载 + C++ 后端 | `csrc/backend/`(EpairModelLoader → GE Graph → OM) / `torch_delegate/`(api.py + converter + epair + core) / `build.sh`(一键构建) / `examples/`(示例) |
| **docs/cann_lm_converter/**　| converter设计文档　　　　 | `cli/cli_design.md` / `export/export_design.md` / `quant/quant_design.md` / `README.md`(总览) / `version_evolution.md`(版本演进)　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **examples/test_converter/** | 端到端测试用例　　　　　　 | `export_llm_model/`(C++ omc编译 + om_npu_verify出字验证) / `cann_lm_cli_export/run_pd6_cases.sh`(6用例)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **scripts/**　　　　　　　　 | 构建脚本　　　　　　　　　 | `build.sh`(顶层check/deps/third_party/cpp/tests/wheel/package/all) / `build.conf`(版本+git源+自定位派生路径) / `set_env.sh`(converter+compile统一env, 真CANN无stubs) / `set_env_cann.sh`(③ NPU) / `_env_common.sh`(公共) / `cann_lm_compile/build_cpp.sh`(C++ .so+cann_lm_compile) / `build_test.sh`(ASAN测试) / `build_third_party.sh`(自建json/gtest/makeself) / `cann_lm_{converter,engine}/build_wheel.sh`(whl打包)　 |
| **tests/**　　　　　　　　　 | 单元测试　　　　　　　　　 | `cpp/`(C++ GTest; 跑: `bash scripts/build.sh tests` → ASAN+ctest) / `python/`(pytest)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **ci/**　　　　　　　　　　　| CI/CD　　　　　　　　　　　| 流水线定义、构建脚本、镜像配置　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cmake/**　　　　　　　　　 | CMake共享　　　　　　　　 | `modules/`(DelegateCommonFlags + Find*.cmake + TorchDelegateCannCommon) / `third_party/`(ExternalProject下载json/gtest/makeself)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

## ⚡️ 快速开始

详见 [快速入门](docs/zh/QUICKSTART.md)，涵盖安装、导出epair、量化导出、omc编译、部署到鸿蒙设备全流程。

## 📚 文档

| 文档 | 内容 |
|------|------|
| [快速入门](docs/zh/QUICKSTART.md) | 安装 / 导出epair / 量化导出 / omc编译 / 部署到鸿蒙 |
| [构建指南](docs/zh/build_guide.md) | 构建前置 / 一键构建+安装+使用 / 速查 / FAQ |
| [环境依赖](docs/zh/env_dependencies.md) | 构建/运行依赖分类 / P1 venv / P2 torch_npu / 多env速查表 |
| [版本演进](docs/zh/version_evolution.md) | 发布节奏 / 配套表（HarmonyOS / CANN / 芯片 / PyTorch / TorchNPU） |
| [CLI设计](docs/zh/cann_lm_converter/cli/cli_design.md) | CLI入口 / convert() / 命令行参数 / 调用链 |
| [导出设计](docs/zh/cann_lm_converter/export/export_design.md) | loader / saver / ExportWrapper / C++ 加载 |
| [量化设计](docs/zh/cann_lm_converter/quant/quant_design.md) | W4A8方案 / 轻量化工具GPTQ / swap / 算子规格 |
| [converter总览](docs/zh/cann_lm_converter/README.md) | 跨模块调用链 / 模型差异 / 精度汇总 |

## 🌐 生态集成

以下项目与cann-device-lm存在上下游依赖或协作关系：

- **[GE (Graph Engine)](https://gitcode.com/cann/ge)**：CANN图引擎，cann-device-lm的FX→GE转换和omc编译均基于GE能力
- **[ops-math](https://gitcode.com/cann/ops-math)**：CANN数值计算算子库，为模型导出提供基础数学算子支持
- **[ops-nn](https://gitcode.com/cann/ops-nn)**：CANN神经网络算子库，为模型导出提供NN层算子支持
- **ops-transformers**（CANN大模型算子库,内部仓）：CANN大模型算子库，为模型导出提供Transformer结构算子支持
- **[TorchAir](https://gitcode.com/Ascend/torchair)**：将GE接入PyTorch图模式，cann-device-lm converter通过torchair完成FX→GE转换
- **[CANN Toolkit](https://hiascend.com/software/cann)**：CANN软件包（atc / acl），提供omc编译和NPU运行时

## 📝 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [构建指南](docs/zh/build_guide.md)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/device-ai)

## 💬 反馈与交流

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/cann-device-lm/issues)提交问题
- **社区互动**：通过GitCode [Discussions](https://gitcode.com/cann/cann-device-lm/discussions)参与交流

---

PS：本项目功能和文档正在持续更新和完善中，欢迎关注最新版本。
