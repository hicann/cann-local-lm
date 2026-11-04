# ExecuTorch-NPU文档索引

简体中文 | [English](README_en.md)

## 六个首先要回答的问题

| 问题 | 唯一主文档 |
|---|---|
| 构建依赖是什么？ | [源码构建：构建依赖](build.md#构建依赖) |
| 如何构建wheel和Backend库？ | [源码构建：执行构建](build.md#执行构建) |
| 导出依赖是什么？ | [模型导出：依赖边界](export.md#依赖边界) |
| 如何导出PTE/PTD？ | [模型导出：阶段B生成PTE/PTD](export.md#阶段b从pt2生成pteptd) |
| C++ 编译和设备运行依赖是什么？ | [C++ 运行：依赖边界](runtime.md#依赖边界) |
| 如何加载PTE/PTD？ | [C++ 运行：加载模型](runtime.md#加载pteptd) |

同一命令或依赖只在其主文档完整描述，其他文档通过链接引用，避免多处内容漂移。

## 入门教程

| 文档 | 内容 |
|---|---|
| [快速开始](quickstart.md) | MobileNetV2构建、导出、Runner编译和NPU验证 |

## 操作指南

| 文档 | 内容 |
|---|---|
| [源码构建](build.md) | 构建依赖、`build.sh`、wheel产物和安装验证 |
| [模型导出](export.md) | CPU Torch、TorchAir、torch-delegate、PTE/PTD生成 |
| [C++ 运行](runtime.md) | C++ 编译依赖、设备运行依赖、Module加载和执行 |
| [故障排查](troubleshooting.md) | 按构建、导出、C++ 编译和NPU运行分类的问题 |

## 参考文档

| 文档 | 内容 |
|---|---|
| [支持版本](supported_versions.md) | ExecuTorch、CPU Torch、Torchvision、CANN和设备矩阵 |
| [C++ Backend SDK](cpp_backend_sdk.md) | wheel内SDK布局、CMake targets、状态API和链接边界 |
| [产物协议](protocol.md) | PTE/PTD、Manifest、named data和state的协议规则 |

字段级Manifest参考模式位于
[`examples/manifest_schema/manifest-v1.schema.json`](../examples/manifest_schema/manifest-v1.schema.json)。

## 设计说明

| 文档 | 内容 |
|---|---|
| [总体设计](design/executorch_npu_design.md) | 编译、产物、运行时、状态和兼容性设计 |
| [架构栈图](design/executorch_npu_stack.md) | 端到端组件分层和代码映射 |
| [Qwen3.5 static-state](design/qwen3_5_export_design.md) | Qwen Prefill/Decode state契约 |
| [Gemma4 static-state](design/gemma4_e2b_it_export_design.md) | Gemma4 state adapter设计 |

`docs/passes/` 保存开发和调试材料，不属于用户安装、导出或运行主路径。
