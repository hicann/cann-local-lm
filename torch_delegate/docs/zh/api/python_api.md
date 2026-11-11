# Python API参考

## 概览

`torch_delegate` 提供三个核心函数，实现PyTorch `ExportedProgram` → `.epair` 的完整流程：

```text
ExportedProgram → export_to_epair() → EpairContents → epair_save() → .epair 文件
                                                                  ↓
                                                          epair_load() → EpairContents
```

**导入方式：**

```python
from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate import EpairContents, EpairManifest, EpairConfig
```

**包信息与C++前缀：**

```python
import torch_delegate

torch_delegate.__version__      # 构建期注入的版本号（debug 构建带 +debug 后缀）
torch_delegate.cmake_prefix()   # C++ 制品前缀，可直接加入 CMAKE_PREFIX_PATH
```

等价CLI：`torch-delegate-config --cmake-prefix` / `torch-delegate-config --version`。

> 完整导出模式依赖 `torch_npu` 提供的 `torchair`。whl只声明 `torch>=2.9`，不会自动安装 `torch_npu`；缺失时 `export_to_epair()` 抛出 `RuntimeError` 并提示安装配套版本。无CANN运行时的场景请使用 [Lite导出模式](#实验特性lite导出模式无cann运行时)。

**实验特性（Lite导出模式）导入：**

```python
from torch_delegate import LiteConfig, lite_export_context
```

> Lite导出模式及其相关接口为实验特性，未来接口可能发生变更，不保证向后兼容。详见 [实验特性：Lite导出模式](#实验特性lite导出模式无cann运行时)。

---

## 核心函数

### `export_to_epair(ep, config)`

将PyTorch `ExportedProgram` 转换为含Ascend IR注解的 `EpairContents`。

**签名：**

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ep` | `ExportedProgram` | 必填 | 通过 `torch.export.export()` 获得的导出程序 |
| `config` | `EpairConfig` | `EpairConfig()` | 转换配置，控制算子分解、模式融合等行为 |

**返回值：** `EpairContents` 对象

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `RuntimeError` | torchair转换失败（如不支持的算子） |
| `ValueError` | ExportedProgram格式不合法 |

**示例：**

```python
import torch
from torch_delegate import export_to_epair, epair_save

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y) + 1.0

model = MyModel()
ep = torch.export.export(model, (torch.randn(3, 3), torch.randn(3, 3)))
contents = export_to_epair(ep)
print(f"转换: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")
epair_save(contents, "model.epair")
```

**注意事项：**

- 该函数会临时修改PyTorch内部函数（`do_auto_functionalized`），调用结束后自动恢复
- 转换结果中的 `ascend_ir.source` 字段标识转换来源：`"converter"` 表示成功转换，空字符串表示fallback节点

---

### `epair_save(contents, f)`

将 `EpairContents` 序列化为 `.epair` 文件。

**签名：**

```python
def epair_save(contents: EpairContents, f: str | os.PathLike) -> None
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `contents` | `EpairContents` | `export_to_epair()` 返回的对象 |
| `f` | `str \| os.PathLike` | 输出文件路径，建议以 `.epair` 为后缀 |

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `OSError` | 文件写入失败 |
| `TypeError` | contents不是EpairContents类型 |

**文件格式：**

`.epair` 文件是PT2兼容的ZIP归档，包含：

| 路径 | 内容 |
|------|------|
| `models/model.json` | 标准EP JSON |
| `data/weights/` | 模型权重 |
| `extra/epair_manifest.json` | 转换元数据 |
| `extra/ep_json_with_ascend_ir.json` | 含ascend_ir注解的EP JSON |

**示例：**

```python
from torch_delegate import export_to_epair, epair_save

contents = export_to_epair(ep)
epair_save(contents, "model.epair")
```

---

### `epair_load(archive_path)`

加载 `.epair` 文件并返回 `EpairContents`。

**签名：**

```python
def epair_load(archive_path: str | os.PathLike) -> EpairContents
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `archive_path` | `str \| os.PathLike` | `.epair` 文件路径 |

**返回值：** `EpairContents` 对象

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `FileNotFoundError` | 文件不存在 |
| `zipfile.BadZipFile` | 文件不是有效的ZIP归档 |
| `KeyError` | 归档缺少必要文件（manifest.json等） |

**示例：**

```python
from torch_delegate import epair_load

contents = epair_load("model.epair")
print(f"后端: {contents.manifest.backend_name}")
print(f"转换: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")

# 在 Python eager 模式下执行
result = contents.ep.module()(input_tensor)
```

---

## 数据类

### `EpairManifest`

Epair转换元数据，记录转换统计信息。

```python
@dataclasses.dataclass
class EpairManifest:
    version: int
    backend_name: str
    cann_version: str
    total_nodes: int
    converted_nodes: int
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | `int` | 格式版本号，当前为1 |
| `backend_name` | `str` | 后端名称，固定为 `"ascend_ge"` |
| `cann_version` | `str` | CANN编译环境版本号，如 `"9.2.0"`。无法获取时为 `"unknown"` |
| `total_nodes` | `int` | EP JSON中的总算子节点数 |
| `converted_nodes` | `int` | 成功转换为Ascend IR的节点数 |

---

### `EpairContents`

Epair转换结果，包含完整的转换产物。

```python
@dataclasses.dataclass
class EpairContents:
    manifest: EpairManifest
    ep_json: dict[str, Any]
    ep: ExportedProgram
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `manifest` | `EpairManifest` | 转换元数据 |
| `ep_json` | `dict[str, Any]` | 含 `ascend_ir` 注解的EP JSON dict |
| `ep` | `ExportedProgram` | 可执行的ExportedProgram |

**检查转换结果：**

```python
contents = export_to_epair(ep)

# 检查转换覆盖率
rate = contents.manifest.converted_nodes / contents.manifest.total_nodes
print(f"转换率: {rate:.1%}")

# 分析 fallback 节点
for node in contents.ep_json["graph_module"]["graph"]["nodes"]:
    ascend_ir = node.get("ascend_ir", {})
    if ascend_ir.get("source") == "":
        print(f"Fallback: {node['target']}")
    elif "source" not in ascend_ir:
        print(f"未标注: {node['target']}")
```

---

### `EpairConfig`

Epair转换配置，控制转换行为。

```python
@dataclasses.dataclass
class EpairConfig:
    remove_noop_ops: bool = True
    pattern_fusion_pass: bool = True
    whitelist_decompositions: set[Any] = field(default_factory=...)
```

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `remove_noop_ops` | `bool` | `True` | 是否移除无操作节点（如identity、reshape） |
| `pattern_fusion_pass` | `bool` | `True` | 是否启用模式融合优化 |
| `whitelist_decompositions` | `set[Any]` | 见下表 | 允许分解的算子集合 |

**默认分解算子：**

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
from torch_delegate import export_to_epair, EpairConfig

# 禁用模式融合（用于调试）
config = EpairConfig(pattern_fusion_pass=False)
contents = export_to_epair(ep, config)

# 禁用所有算子分解
config = EpairConfig(whitelist_decompositions=set())
contents = export_to_epair(ep, config)

# 仅分解 linear
config = EpairConfig(whitelist_decompositions={torch.ops.aten.linear.default})
contents = export_to_epair(ep, config)

# 在默认分解算子基础上追加
config = EpairConfig()
config.whitelist_decompositions.update({
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten.add.Tensor,
})
contents = export_to_epair(ep, config)
```

---

## 实验特性Lite导出模式（无CANN运行时）

> **实验特性**：本节介绍的所有接口（`LiteConfig`、`lite_export_context` 及各 `Lite*` 异常）均为实验特性，未来接口可能发生变更，不保证向后兼容。

Lite导出模式允许在不安装CANN、`torch_npu` 且无NPU设备的环境中构建EPAIR所需的GE IR。它通过 `lite_export_context` 在进程内打桩TorchAir与 `torch_npu` 依赖，仅在构图时替代原生能力，不加载、编译或执行GE图。

**约束：**

- 进入 `lite_export_context` 前，进程不得已加载真实 `torchair` 或 `torch_npu`。若环境安装了 `torch_npu`，须在导入 `torch` 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- Lite模式与完整模式（默认 `export_to_epair`）互斥，同一进程内不可混用
- Lite模式不允许任何未转换（fallback / 未标注）节点
- Lite模式不支持多线程；上下文由进程内状态驱动，退出后状态不可完全恢复

### `lite_export_context(config)`

进入进程级、仅构建（build-only）的EPAIR导出模式。

**签名：**

```python
@contextlib.contextmanager
def lite_export_context(config: LiteConfig) -> Iterator[None]
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `LiteConfig` | 目标平台配置，含设备型号、HF32、目标CANN版本 |

**返回值：** 上下文管理器，`with` 块内可通过 `export_to_epair()` 进行Lite转换。

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `LiteConfigError` | `config` 类型或字段取值不合法 |
| `LiteImportError` | 进程已加载 `torchair` / `torch_npu` 等冲突依赖，或打桩后导入TorchAir失败 |
| `LiteStateError` | 与进程模式冲突（如已处于完整模式）、跨线程使用、嵌套时配置不一致 |

**示例：**

```python
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from torch_delegate import (
    LiteConfig,
    epair_save,
    export_to_epair,
    lite_export_context,
)

ep = torch.export.export(model, example_inputs)

config = LiteConfig(target_soc="Ascend910_9362", target_cann_version="9.2.0")

with lite_export_context(config):
    contents = export_to_epair(ep)

# epair_save 不调用 TorchAir，可放在 context 外执行。
epair_save(contents, "model.epair")
```

**注意事项：**

- 进入context后，转换流程复用 `TorchAirGEConverter`；manifest不再记录本机CANN版本（`cann_version` 为 `"unknown"`）。
- 嵌套进入context时仅当配置完全相同才允许，否则抛出 `LiteStateError`。

---

### `LiteConfig`

Lite导出模式的目标平台配置。

```python
@dataclasses.dataclass(frozen=True)
class LiteConfig:
    target_soc: str
    allow_hf32: bool = False
    target_cann_version: str | None = None
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_soc` | `str` | 必填 | 目标部署设备型号，如 `"Ascend910_9362"`；取值语义须与TorchAir平台判断一致 |
| `allow_hf32` | `bool` | `False` | 目标图是否启用HF32 |
| `target_cann_version` | `str \| None` | `None` | 目标部署CANN版本，预留字段；命中依赖版本查询时会报错 |

---

### Lite异常类型

Lite模式的异常均派生自 `LiteExportError`（基类 `RuntimeError`），可统一捕获：


| 异常类型 | 基类 | 触发条件 |
|---------|------|---------|
| `LiteExportError` | `RuntimeError` | 所有Lite导出失败的基类 |
| `LiteImportError` | `LiteExportError` | 进程已加载冲突的原生依赖（`torchair` / `torch_npu`），或打桩后TorchAir导入失败 |
| `LiteStateError` | `LiteExportError` | 操作与进程 / 上下文状态冲突（模式互斥、跨线程、配置不一致） |
| `LiteDependencyError` | `LiteExportError` | converter依赖的能力在Lite模式不可用 |
| `LiteConversionError` | `LiteExportError` | Lite模式下TorchAir转换失败或产生fallback / 未标注节点 |

---

## 异常处理

### 转换失败处理

```python
from torch_delegate import export_to_epair, epair_save

try:
    contents = export_to_epair(ep)
    epair_save(contents, "model.epair")
except RuntimeError as e:
    print(f"转换失败: {e}")
    # 检查哪些算子未被转换
    for node in ep.graph.nodes:
        if node.op == "call_function":
            print(f"  算子: {node.target}")
```

### 加载失败处理

```python
from torch_delegate import epair_load

try:
    contents = epair_load("model.epair")
except FileNotFoundError:
    print("文件不存在，请先执行 export_to_epair")
except KeyError as e:
    print(f"文件损坏，缺少: {e}")
```

---

## 日志控制

通过环境变量控制日志输出：

```bash
# 启用调试日志
export TORCH_DELEGATE_DEBUG=1
python my_script.py
```

---

## 相关文档

- [epair_format_design.md](../features/epair_format_design.md) — `.epair` 格式详细设计文档
- [build_epair.md](../guide/build_epair.md) — 用户指南（含完整工作流示例）
- [architecture.md](../features/architecture.md) — 整体架构概览
