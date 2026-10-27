# TorchDelegate

简体中文 | [English](README_en.md)

## 概述

**TorchDelegate** 是一个基于 `torch.export` 的模型部署框架，通过 **delegate（委派）机制** 将PyTorch模型切分后分别委派给不同的执行后端：NPU子图委派给加速器后端编译执行，CPU部分委派给libtorch运行时fallback执行，最终生成统一的部署产物（`.epair` 格式），支持Python导出与C++端加载运行。

---

## 核心特性

- **基于torch.export**：复用PyTorch 2.0+的导出能力，确保FX图完整性和算子语义
- **自动算子委派**：自动识别模型中可委派给NPU执行的算子子图，其余算子回退到libtorch
- **三阶段API**：`export_to_epair → epair_save → epair_load`，导出/序列化/加载路径清晰，支持回退eager模式执行
- **C++运行时**：独立的 `libtorch_delegate_backend.so`，加载 `.epair` 后在C++侧完成切图、编译、执行

---

## 架构概览

```text
ExportedProgram ──▶ [Python Converter] ──▶ .epair 产物 ──▶ [C++ Runtime]
                                                        │
                                    NPU 子图 ──────────▶ GE Engine
                                    CPU 算子 ──────────▶ libtorch fallback
```

| 模块 | 语言 | 职责 |
|------|------|------|
| **Converter** | Python | 解析 `ExportedProgram`，将NPU兼容节点转换为后端IR，生成 `.epair` |
| **Backend** | C++ | 加载 `.epair`，解析EP JSON，构建GE Graph，加载权重 |
| **Fallback Codegen** | C++ | 为不支持的节点生成CPU fallback代码 |
| **Common** | C++ | 公共基础设施（日志、归档读取、Base64、ABI兼容） |

---

## 编译安装

```bash
# 一站式构建（三方库 → C++ → 单一 whl）
./build.sh

# 编译并运行测试
./build_test.sh
```

产物为唯一发布物 `output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl`，同时包含Python API与C++制品（头文件、动态库、CMake配置包）；与Python版本无关，可安装进3.10 ~ 3.13。

详细构建说明请参考 [docs/zh/guide/build.md](docs/zh/guide/build.md)。

---

## 本地验证

以下步骤帮助你在5分钟内完成第一个 `.epair` 模型的导出与C++加载运行。

### 安装CANN

```bash
# 安装 CANN Toolkit（详见 docs/zh/guide/build.md）
# root 用户默认路径
source /usr/local/Ascend/cann/set_env.sh
# 普通用户
source ${HOME}/Ascend/cann/set_env.sh
```

### 构建并安装

```bash
./build.sh
pip install output/package/torch_delegate-*-py3-none-linux_*.whl
```

C++前缀与运行期库路径（whl不再提供 `set_env.sh`）：

```bash
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

### 运行示例

```bash
cd examples
# 创建并激活测试虚拟环境（会自动安装上一步产出的 whl）
bash prepare_env.sh
source ./venv_test/bin/activate

# 执行示例
cd export_and_load_example
bash run.sh
```

`run.sh` 自动完成：
- 环境检查（CANN、TorchDelegate、Python、cmake）
- 生成示例epair文件
- 编译C++加载程序
- 执行端到端验证

成功输出示意：

```text
 [1/4] 检查环境
  [✓] ASCEND_HOME_PATH=/usr/local/Ascend/cann
  [✓] TORCH_DELEGATE_ROOT=/path/to/venv/lib/python3.12/site-packages/torch_delegate
  ...
 [4/4] 运行示例
========================================
 验证通过
========================================
```

更多示例请参考 [examples/](examples/README.md) 目录。

## 文档索引

| 文档 | 说明 |
|------|------|
| [docs/README.md](docs/README.md) | 文档总览和阅读路径 |
| [docs/zh/guide/build.md](docs/zh/guide/build.md) | 编译安装与环境配置 |
| [docs/zh/guide/build_epair.md](docs/zh/guide/build_epair.md) | epair打包指南 |
| [docs/zh/guide/fallback_codegen_guide.md](docs/zh/guide/fallback_codegen_guide.md) | Fallback Codegen使用指南 |
| [docs/zh/api/python_api.md](docs/zh/api/python_api.md) | Python API参考 |
| [docs/zh/api/epair_model_loader.md](docs/zh/api/epair_model_loader.md) | C++ API参考 |
| [docs/zh/features/architecture.md](docs/zh/features/architecture.md) | 整体架构说明 |
| [examples/README.md](examples/README.md) | 示例目录 |

## 贡献指南

参与项目贡献前请阅读[《贡献指南》](../CONTRIBUTING.md)。

## 许可证

本项目遵循[CANN Open Software License Agreement Version 2.0](../LICENSE)。
