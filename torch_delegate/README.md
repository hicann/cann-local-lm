# TorchDelegate

**TorchDelegate** 是一个基于 `torch.export` 的模型部署框架，通过 **delegate（委派）机制** 将 PyTorch 模型切分后分别委派给不同的执行后端：NPU 子图委派给加速器后端编译执行，CPU 部分委派给 libtorch 运行时 fallback 执行，最终生成统一的部署产物（`.epair` 格式），支持 Python 导出与 C++ 端加载运行。

---

## 核心特性

- **基于 torch.export**：复用 PyTorch 2.0+ 的导出能力，确保 FX 图完整性和算子语义
- **自动算子委派**：自动识别模型中可委派给 NPU 执行的算子子图，其余算子回退到 libtorch
- **三阶段 API**：`export_to_epair → epair_save → epair_load`，导出/序列化/加载路径清晰，支持回退 eager 模式执行
- **C++ 运行时**：独立的 `libtorch_delegate_backend.so`，加载 `.epair` 后在 C++ 侧完成切图、编译、执行

---

## 架构概览

```
ExportedProgram ──▶ [Python Converter] ──▶ .epair 产物 ──▶ [C++ Runtime]
                                                        │
                                    NPU 子图 ──────────▶ GE Engine
                                    CPU 算子 ──────────▶ libtorch fallback
```

| 模块 | 语言 | 职责 |
|------|------|------|
| **Converter** | Python | 解析 `ExportedProgram`，将 NPU 兼容节点转换为后端 IR，生成 `.epair` |
| **Backend** | C++ | 加载 `.epair`，解析 EP JSON，构建 GE Graph，加载权重 |
| **Fallback Codegen** | C++ | 为不支持的节点生成 CPU fallback 代码 |
| **Common** | C++ | 公共基础设施（日志、归档读取、Base64、ABI 兼容） |

---

## 构建

```bash
# 一站式构建（三方库 → C++ → 单一 whl）
./build.sh

# 编译并运行测试
./build_test.sh
```

产物为唯一发布物 `output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl`，同时包含 Python API 与 C++ 制品（头文件、动态库、CMake 配置包）；与 Python 版本无关，可安装进 3.10 ~ 3.13。

详细构建说明请参考 [docs/build.md](docs/build.md)。

---

## 快速开始

以下步骤帮助你在 5 分钟内完成第一个 `.epair` 模型的导出与 C++ 加载运行。

### 1. 安装 CANN

```bash
# 安装 CANN Toolkit（详见 docs/build.md）
# root 用户默认路径
source /usr/local/Ascend/cann/set_env.sh
# 普通用户
source ${HOME}/Ascend/cann/set_env.sh
```

### 2. 构建并安装

```bash
./build.sh
pip install output/package/torch_delegate-*-py3-none-linux_*.whl
```

C++ 前缀与运行期库路径（whl 不再提供 `set_env.sh`）：

```bash
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

### 3. 运行示例

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
- 生成示例 epair 文件
- 编译 C++ 加载程序
- 执行端到端验证

成功输出示意：

```
 [1/4] 检查环境
  [✓] ASCEND_HOME_PATH=/usr/local/Ascend/cann
  [✓] TORCH_DELEGATE_ROOT=/path/to/venv/lib/python3.12/site-packages/torch_delegate
  ...
 [4/4] 运行示例
========================================
 验证通过
========================================
```

更多示例请参考 [examples/](examples/) 目录。

---
