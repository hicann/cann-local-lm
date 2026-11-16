# Epair打包指南

本文档描述如何使用torch_delegate Python API将PyTorch ExportedProgram打包为epair文件，供C++ delegate库加载执行。

---

## 前置条件

- **Python >= 3.10**（whl支持3.10 ~ 3.13）
- **torch >= 2.9.0**（whl不声明运行时依赖，不会自动安装，需自行安装配套版本）
- **torch_npu**（完整导出模式必需，提供 `torchair`；whl不会自动安装，需自行安装配套版本）

---

## 安装torch_delegate

### 从whl包安装（推荐）

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

whl是唯一发布产物，同时包含Python API与C++制品（头文件、动态库、CMake配置包）。

### 从源码构建

```bash
cd torch-delegate
./build.sh  # 一站式构建，产出单一 whl 到 output/package/
```

详见 **[build.md](build.md)**。

### 验证安装

```bash
python -c "from torch_delegate import export_to_epair, epair_save, epair_load; print('torch_delegate installed successfully')"
```

---

## Python API

torch_delegate提供三个核心API：

### export_to_epair

将ExportedProgram转换为EpairContents，包含Ascend IR标注：

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents:
    """
    转换 ExportedProgram 到 EpairContents

    Args:
        ep: torch.export 导出的 ExportedProgram
        config: EpairConfig（可选），控制优化开关

    Returns:
        EpairContents: 包含 manifest、ep_json 和可执行的 ExportedProgram
    """
```

### epair_save

将EpairContents保存为 .epair文件：

```python
def epair_save(
    contents: EpairContents,
    f: str | os.PathLike
) -> None:
    """
    保存 EpairContents 到 epair 文件

    Args:
        contents: EpairContents 对象
        f: 输出文件路径（如 "model.epair"）
    """
```

### epair_load

从 .epair文件加载EpairContents：

```python
def epair_load(f: str | os.PathLike) -> EpairContents:
    """
    加载 epair 文件

    Args:
        f: epair 文件路径

    Returns:
        EpairContents: 包含 manifest、ep_json 和可执行的 ExportedProgram
    """
```

---

## EpairConfig配置

控制图优化开关和算子分解行为：

```python
from torch_delegate.epair import EpairConfig

config = EpairConfig(
    remove_noop_ops=True,        # 移除无操作算子（默认 True）
    pattern_fusion_pass=True,    # 模式融合优化（默认 True）
    require_aten_on_npu=False,   # aten算子落到CPU时是否报错（默认 False，仅告警）
    whitelist_decompositions=None  # 算子分解白名单（默认分解 linear/matmul/layernorm 等）
)
```

### 配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `remove_noop_ops` | `bool` | `True` | 移除无操作算子（如identity、reshape等无实际计算的节点） |
| `pattern_fusion_pass` | `bool` | `True` | 启用模式融合优化，将多个算子融合为单个Ascend算子以提升NPU执行效率 |
| `require_aten_on_npu` | `bool` | `False` | 严格校验开关：aten算子被判到CPU执行时导出阶段直接报错（用于CI/性能基线看护），自定义namespace算子不受影响 |
| `whitelist_decompositions` | `set \| None` | `None` | 允许分解的算子白名单。`None` 表示使用默认白名单，`set()` 表示禁用所有分解 |

### whitelist_decompositions详解

控制哪些PyTorch算子会被分解为更基础的算子，以便Ascend后端处理。

**默认白名单包含以下算子：**

| 算子 | 说明 |
|------|------|
| `torch.ops.aten.linear.default` | 线性层，分解为matmul + add |
| `torch.ops.aten.matmul.default` | 矩阵乘法 |
| `torch.ops.aten.layer_norm.default` | LayerNorm |
| `torch.ops.aten.softmax.int` | Softmax |
| `torch.ops.aten.log_softmax.int` | LogSoftmax |
| `torch.ops.aten.__and__.Tensor` | 按位与 |
| `torch.ops.aten.__or__.Tensor` | 按位或 |
| `torch.ops.aten.isfinite.default` | 判断元素是否有限值，分解为abs + ne + mul |
| `torch.ops.aten.negative.default` | 取负，分解为neg（alias归一化） |
| `torch.ops.aten.absolute.default` | 取绝对值，分解为abs（alias归一化） |
| `torch.ops.aten.relu6.default` | ReLU6，分解为hardtanh |
| `torch.ops.aten.arccos.default` | 反余弦，分解为acos（alias归一化） |
| `torch.ops.aten.arcsin.default` | 反正弦，分解为asin（alias归一化） |
| `torch.ops.aten.arctan.default` | 反正切，分解为atan（alias归一化） |
| `torch.ops.aten.arccosh.default` | 反双曲余弦，分解为acosh（alias归一化） |
| `torch.ops.aten.arcsinh.default` | 反双曲正弦，分解为asinh（alias归一化） |
| `torch.ops.aten.conv2d.default` | 二维卷积，分解为convolution |
| `torch.ops.aten.square.default` | 平方，分解为pow.Tensor_Scalar |

**配置示例：**

```python
import torch
from torch_delegate.epair import EpairConfig

# 禁用所有算子分解
config = EpairConfig(whitelist_decompositions=set())

# 仅分解 linear
config = EpairConfig(
    whitelist_decompositions={torch.ops.aten.linear.default}
)

# 自定义分解列表
config = EpairConfig(
    whitelist_decompositions={
        torch.ops.aten.linear.default,
        torch.ops.aten.layer_norm.default,
    }
)
```

---

## 完整示例

### 基础示例 导出并保存

```python
import torch
import torch.nn as nn
from torch.export import export
from torch_delegate import export_to_epair, epair_save

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = SimpleModel()

# 导出为 ExportedProgram
example_inputs = (torch.randn(1, 10),)
ep = export(model, example_inputs)

# 转换为 EpairContents
contents = export_to_epair(ep)

# 查看转换统计
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")
print(f"Backend: {contents.manifest.backend_name}")

# 保存为 epair 文件
epair_save(contents, "model.epair")
print("Saved to model.epair")
```

### 加载并执行

```python
from torch_delegate import epair_load

# 加载 epair 文件
contents = epair_load("model.epair")

# 查看元数据
print(f"Backend: {contents.manifest.backend_name}")
print(f"Converted nodes: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")

# 执行模型
inputs = torch.randn(1, 10)
outputs = contents.ep.module()(inputs)
print(f"Output shape: {outputs.shape}")
```

### 使用配置优化

```python
from torch_delegate import export_to_epair, epair_save
from torch_delegate.epair import EpairConfig

# 自定义配置
config = EpairConfig(
    remove_noop_ops=True,
    pattern_fusion_pass=False  # 关闭融合优化
)

# 导出并转换
ep = export(model, example_inputs)
contents = export_to_epair(ep, config=config)

# 保存
epair_save(contents, "optimized_model.epair")
```

---

## epair文件结构

`.epair` 文件是ZIP归档格式，兼容PT2格式：

```text
model.epair (ZIP 结构)
├── extra/
│   ├── epair_manifest.json           ← epair 元数据（新增）
│   └── ep_json_with_ascend_ir.json   ← 带 ascend_ir 的 EP JSON（新增）
├── models/
│   └── model.json                    ← 标准 EP JSON
└── data/
    ├── weights/
    │   ├── model_weights_config.json
    │   └── weight_0, weight_1, ...
    ├── constants/
    │   ├── model_constants_config.json
    │   └── tensor_0, tensor_1, ...
    └── model_sample_inputs.pt
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `extra/epair_manifest.json` | 元数据：版本、backend、统计信息、CANN版本 |
| `extra/ep_json_with_ascend_ir.json` | 带 `ascend_ir` 字段标注的完整EP JSON |
| `models/model.json` | 标准EP JSON（不含ascend_ir） |
| `data/weights/` | 模型权重张量 |
| `data/constants/` | 常量张量 |
| `data/model_sample_inputs.pt` | 示例输入 |

### manifest.json格式

```json
{
  "version": 1,
  "backend_name": "ascend_ge",
  "cann_version": "9.2.0",
  "total_nodes": 20,
  "converted_nodes": 15,
  "cpu_nodes": 3
}
```

---

## ascend_ir字段格式

EP JSON中每个node对象可选包含 `ascend_ir` 字段：

### 正常转换节点

```json
{
  "target": "aten.add.Tensor",
  "inputs": [...],
  "outputs": [...],
  "ascend_ir": {
    "source": "converter",
    "ops": [
      {
        "name": "Add",
        "attr": [{"key": "alpha", "value": {"f": 1.0}}],
        "input": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
        "output": [{"name": "z", "type": "tensor"}]
      }
    ],
    "mapping": {
      "outputs": ["add_1:0"]
    }
  }
}
```

### 转换失败节点（Mock）

```json
{
  "target": "aten.custom_op",
  "ascend_ir": {
    "ops": [{"attr": [...]}]
  }
}
```

**说明**：
- `source = "converter"`：正常GE转换，C++侧划入NPU子图
- `source = ""`：Mock节点，C++侧划入CPU段

---

## 日志调试

torch_delegate提供完整的日志系统，支持环境变量控制：

### 日志级别

| 级别 | 使用场景 |
|------|----------|
| ERROR | 转换失败终止、关键依赖缺失 |
| WARNING | 节点转换失败创建mock、CANN版本获取失败 |
| INFO | API入口/出口、转换统计 |
| DEBUG | 配置详情、中间步骤 |

### 启用DEBUG日志

```bash
# 默认：WARN 级别
python your_script.py

# 启用 DEBUG 级别
TORCH_DELEGATE_DEBUG=1 python your_script.py
```

### 日志格式

```text
[LEVEL] TORCH_DELEGATE(pid,python) timestamp [file:line] thread_id message
```

示例：
```text
[INFO] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:15 [api.py:29] Exporting ExportedProgram to epair...
[WARNING] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:16 [epair_graph.py:142] Failed to convert aten.custom_op: creating mock node
[INFO] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:17 [api.py:46] Export complete: 15/20 nodes converted
```

---

## 与C++ delegate的关系

### 工作流程

```text
Python 侧 → C++ 侧

1. Python: export_to_epair(ep)
   ↓ 生成 EpairContents（带 ascend_ir 标注）

2. Python: epair_save(contents, "model.epair")
   ↓ 保存为 .epair 文件

3. C++: td::EpairModelLoader loader("model.epair")
   ↓ 加载 .epair，解析 ep_json_with_ascend_ir.json 并构建 GE 图

4. C++: loader.CompileAndLoad(session_options, stream)
   ↓ 编译（含 fallback 自定义算子）并加载

5. C++: loader.RunModelWithStreamAsync(stream, inputs, outputs)
   ↓ 执行推理
```

### C++加载示例

完整可运行示例见 `examples/export_and_load_example/main.cpp`。

```cpp
#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <map>
#include <vector>

// 1. 初始化 GE
std::map<ge::AscendString, ge::AscendString> ge_options = {
    {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
ge::GEInitializeV2(ge_options);

// 2. 加载 epair 文件（Python 侧生成），完成构图与编译加载
td::EpairModelLoader loader("model.epair");
std::map<ge::AscendString, ge::AscendString> session_options;
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
td::Status status = loader.CompileAndLoad(session_options, stream);
if (status != td::SUCCESS) {
    // 编译加载失败，报错返回
    return FAILED;
}

// 3. 准备输入并执行推理（动态 shape 场景 outputs 可不预分配）
std::vector<gert::Tensor> inputs;   // 按 GE 图输入 desc 填充
std::vector<gert::Tensor> outputs;
status = loader.RunModelWithStreamAsync(stream, inputs, outputs);
if (status != td::SUCCESS) {
    // 推理失败，报错返回
    return FAILED;
}
// ...
```

---

## 常见问题

### Q1 torch_npu未安装导致CANN版本为unknown

**现象**：manifest中 `cann_version` 为 "unknown"

**解决方案**：
1. 安装torch_npu: `pip install torch_npu`
2. 或忽略此字段，不影响epair文件的生成和加载

### Q2转换失败节点过多

**现象**：`converted_nodes` 远小于 `total_nodes`

**解决方案**：
1. 检查模型使用的op是否在TorchAir支持列表中
2. 查看WARNING日志，定位具体失败的op
3. Mock节点会被划入CPU段执行，不影响整体功能

### Q3 epair文件过大

**现象**：生成的 .epair文件体积过大

**解决方案**：
1. 检查权重和常量是否正确外部化
2. 使用 `torch.export.save` 的权重压缩选项
3. 避免在模型中硬编码大常量张量

### Q4加载epair后执行失败

**现象**：Python侧 `contents.ep.module()` 执行报错

**解决方案**：
1. 确保加载环境与导出环境的torch版本一致
2. 检查 `ep_json_with_ascend_ir.json` 是否完整
3. 使用DEBUG日志查看详细错误信息

---

## 相关文档

- **[build.md](build.md)** - C++ delegate库构建指南
- **[epair_format_design.md](../features/epair_format_design.md)** - epair实现方案详细设计

---

## 参考资料

- [PyTorch Export文档](https://pytorch.org/docs/stable/export.html)
- [PT2 Archive格式](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/pt2_archive.html)
- [TorchAir GE Converter](https://gitcode.com/Ascend/pytorch)
