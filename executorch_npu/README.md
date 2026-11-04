# ExecuTorch-NPU

简体中文 | [English](README_en.md)

ExecuTorch-NPU是ExecuTorch的Ascend AOT Backend。它在导出阶段把PyTorch
`ExportedProgram` 转换为GE图并通过CANN生成OM和外置权重，在运行阶段由
ExecuTorch C++ Runtime加载PTE/PTD，再通过AscendCL在昇腾NPU上执行。

## 工作流

```text
ExecuTorch-NPU 源码
    │  构建
    ▼
executorch_npu wheel + C++ Backend SDK
    │  Python 导出
    ▼
model.pte + weights.ptd
    │  C++ 应用加载
    ▼
ExecuTorch Runtime → AscendBackend → AscendCL → Ascend NPU
```

构建、导出和运行是三个独立阶段，可以位于不同机器和不同Python环境中。

模型导出必须使用PyTorch官方 `+cpu` wheel。ExecuTorch-NPU不自动安装Torch，
具体组合见 [支持版本](docs/supported_versions.md)。

导出与编译阶段不需要NPU设备在位：目标芯片仅通过 `ACL_SOC_NAME` 传入，
只有运行阶段需要匹配的NPU、驱动和AscendCL运行时。该模式称Lite导出，
机制与约束见 [模型导出](docs/export.md)。

## 按任务选择文档

| 目标 | 文档 |
|---|---|
| 第一次完成端到端验证 | [快速开始](docs/quickstart.md) |
| 查看构建依赖并生成wheel | [源码构建](docs/build.md) |
| 查看导出依赖并生成PTE/PTD | [模型导出](docs/export.md) |
| 查看运行依赖并用C++ 加载PTE/PTD | [C++ 运行](docs/runtime.md) |
| 排查构建、导出或运行错误 | [故障排查](docs/troubleshooting.md) |
| 集成wheel内的头文件、静态库和CMake target | [C++ Backend SDK](docs/cpp_backend_sdk.md) |

完整文档分类见 [文档索引](docs/README.md)。

## 交付物边界

| 交付物 | 产生阶段 | 用途 |
|---|---|---|
| `executorch_npu-*.whl` | 源码构建 | Python AOT Backend和C++ Backend SDK |
| `model.pte` | 模型导出 | ExecuTorch Program、Delegate元数据和OM |
| `ascend.weights.package.ge_aot.ptd` | 模型导出 | 外置权重named data |
| 用户Runner/应用 | C++ 应用构建 | 加载PTE/PTD并调用ExecuTorch Runtime |

wheel不包含CANN Toolkit、驱动、固件、NNRT、TorchAir、torch-delegate、模型文件
或通用Runner可执行文件。

## 核心能力

- 使用ExecuTorch delegation、PTE、PTD、Program和Method。
- OM和权重通过named-data key、SHA-256和逻辑长度建立确定性关联。
- 支持多Method权重复用及显式state contract。
- 导出与编译全程与NPU设备解耦，产物可跨机器交付到运行环境。
- C++ Backend通过静态注册接入ExecuTorch Runtime，用户不直接调用Backend内部接口。

## 许可证

CANN Open Software License Agreement Version 2.0
