# TorchDelegate 0630 v1 总体设计

## 1. 简介

### 1.1 目的

将 PyTorch 模型部署到昇腾 NPU（Ascend）上执行，行业主流方案是通过 `torch.compile` 调用 torchair 提供的 NPU 后端入图，或通过 `torchair.dynamo_export` 导出 `.air` 模型后经过 ATC 或构造 Graph 的方式入图。这些方案存在的问题是：

- **Python 环境强绑定**：IR 转换和编译都依赖 Python 环境和 torchair，部署侧仍需保留完整 Python 软件栈，无法做到纯 C++ 部署。
- **格式封闭**：`.air` 为二进制私有格式，无法与 PyTorch 生态工具链（`torch.export.save/load`、`ExportedProgram`）互通。
- **算子扩展门槛高**：接入自定义算子需要手写 GE 自定义算子代码（`REG_OP` + 形状推导 + Execute 方法），开发成本高，迭代周期长。

**TorchDelegate** 提供一条中间路径，核心思路是将"IR 转换"与"构图"分离：

- Python 侧通过 `torch.export.ExportedProgram` + torchair converter 完成原子级的 IR 标注，输出兼容 PT2 格式的 `.epair` 归档
- C++ 侧通过 `EpairModelLoader` 加载 `.epair`，完成 GE Graph 构建、编译和执行，实现纯 C++ 部署
- 对自定义算子通过 **Fallback Codegen** 自动生成 GE 自定义算子代码，用户只需通过 `torch.library` 注册 PyTorch 算子，无需手写 GE 代码

**亮点**：

| 亮点 | 说明 |
|------|------|
| PT2 格式兼容 | `.epair` 是合法 PT2 Archive，可直接 `torch.export.load` 在 CPU 上执行，无缝对接 PyTorch 工具链 |
| 部署环境精简 | C++ 侧运行只需 ACL Runtime + GE Runtime，不依赖 Python 环境 |
| 算子扩展零成本 | 用户注册 `torch.library` 算子后，Fallback Codegen 自动生成 `REG_OP` + `EagerExecuteOp` 代码并编译加载 |
| ABI 安全 | 对外接口使用 `ge::AscendString` + PIMPL，支持 `_GLIBCXX_USE_CXX11_ABI=0/1` 混合使用 |
| 委派机制灵活 | Python 只标注不决策，C++ 侧根据 IR 连续性和目标设备能力动态决定执行边界，`.epair` 可跨硬件复用 |

本文档面向 GE 和 torch-delegate 开发者，描述 TorchDelegate 0630 v1 首发版本的软件需求和总体设计。

### 1.2 范围

**包含**：

- Python 侧 epair 导出（`export_to_epair` / `epair_save` / `epair_load`）
- C++ Backend（epair 归档解析、GE Graph 构建、权重加载、编译执行）
- Fallback Codegen（集成在 Backend 构图流程中，自动生成 GE 自定义算子代码）
- Python 与 C++ 对外接口设计

**不包含**：

- 图切分（延迟至后续版本实现）
- CPU Fallback（节点全量委派 NPU，不支持 CPU 段生成）
- 异构调度代码生成（HeteroCodegen + runner .so 打包）
- 离线可部署的 deployable.epair（当前仅支持导出仅标注的 annotated.epair）

---

## 2. 总体概述

### 2.1 软件概述

#### 2.1.1 项目介绍

TorchDelegate 是 GE 后端与 PyTorch 模型导出之间的中间层，将 `torch.export.ExportedProgram` 转换为携带 Ascend IR 标注的 `.epair` 模型归档文件，在 C++ 侧完成 GE Graph 构建和编译执行。其核心思路是**委派机制**：Python 侧完成原子级的 Ascend IR 标注，但将"哪些节点委派给 GE 执行、如何划分执行边界"的决策权延迟到 C++ 侧——Python 只标注不决策，C++ 侧根据 IR 标注的连续性和目标设备动态决定子图边界并驱动编译执行。

```mermaid
flowchart TD
  subgraph User["用户层"]
    CODE["用户代码 (Python/C++)"]
  end

  subgraph Python["Python 层"]
    PT["PyTorch<br/>torch.export.ExportedProgram"]
    TA["TorchAir<br/>FX → Ascend GE IR Converter"]
    TDP["TorchDelegate Python<br/>export_to_epair / epair_save"]
  end

  subgraph Archive["归档"]
    EPAIR[".epair 文件<br/>权重 + IR 标注 + manifest"]
  end

  subgraph CPP["C++ 层"]
    TDC["TorchDelegate C++ Backend<br/>EpairModelLoader / GeGraphBuilder / GeWeightLoader"]
    GE["GE<br/>GeSession::CompileGraph + RunGraphWithStreamAsync"]
  end

  CODE -->|"torch.export"| PT
  PT -->|"export()"| TDP
  TA -->|"converter 注册"| TDP
  TDP -->|"epair_save"| EPAIR
  EPAIR -->|"EpairModelLoader(path)"| TDC
  TDC -->|"GeGraphBuilder.Build"| GE
  TDC -->|"GetGEGraph"| GE
```

---

### 2.2 Delegate 使用场景

#### 场景 1：模型导出（Python 侧）

用户在 Python 环境中导出模型并生成 `.epair` 归档：

```python
from torch_delegate import export_to_epair, epair_save
from torch.export import export

ep = export(model, (inputs,))
contents = export_to_epair(ep)
epair_save(contents, "model.epair")
```

产物 `.epair` 是兼容 PT2 格式的 ZIP 归档，可通过 `torch.export.load` 直接加载并执行。

#### 场景 2：NPU 部署（C++ 侧）

部署方在 NPU 设备上加载 `.epair`，完成 GE Graph 构建、JIT 编译和执行：

```cpp
#include "torch_delegate/epair_model_loader.h"

td::EpairModelLoader loader("model.epair");

std::map<ge::AscendString, ge::AscendString> options = {
    {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
};
void* stream = nullptr;
aclrtCreateStream(&stream);

loader.CompileAndLoad(options, stream);

std::vector<gert::Tensor> inputs, outputs;
loader.RunModelWithStreamAsync(stream, inputs, outputs);
```

v1 版本仅支持 JIT 在线路径，运行侧使用 GE Session。

#### 场景 3：自定义算子接入

用户通过 `torch.library` 注册自定义算子，在模型中直接调用。导出时这些算子不携带 ascend_ir 标注，C++ Backend 的 Fallback Codegen 模块自动识别并生成 GE 自定义算子代码，通过 PyTorch Dispatcher 调用用户 NPU 实现。

> **v1 约束**：630 版本仅支持纯 Tensor 输入的自定义算子。如果算子 schema 中包含 Scalar/int/float/bool 等非 Tensor 参数，Fallback Codegen 会报错提示。Scalar 参数支持计划 715 版本交付。

**Step 1：Python 侧注册自定义算子**

```python
import torch

# 定义算子 schema
lib = torch.library.Library("my_ops", "DEF")
lib.define("add_scalar(Tensor x, float alpha) -> Tensor")

# Meta 实现（用于图模式形状推导）
@torch.library.register_fake("my_ops::add_scalar")
def _add_scalar_meta(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.empty_like(x)

# NPU 实现（注册到 PrivateUse1 dispatch key）
@torch.library.impl("my_ops::add_scalar", "PrivateUse1")
def _add_scalar_npu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # 实际 NPU 实现（aclnn / AscendC 等），此处为占位
    return x + alpha
```

**Step 2：在模型中使用并导出 epair**

```python
class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.my_ops.add_scalar(x, 0.5)

ep = torch.export.export(MyModel(), (torch.randn(4, 4),))
contents = export_to_epair(ep)  # add_scalar 节点无 ascend_ir，source=""
epair_save(contents, "model.epair")
```

**Step 3：C++ Fallback Codegen 处理**

GeGraphBuilder 构造阶段，Fallback Codegen 扫描到 `source=""` 的节点，批量生成自定义算子代码并编译加载。PlanNodes 阶段通过 `QueryRegisteredIrDef` 查找已注册的 op 定义，用 `NamedIoNodeBuilder` 构图。Execute 运行时算子参数映射：
1. Tensor 参数 → `GetInputTensor` + `CreateNpuTensor`（`from_blob` 零拷贝）
2. Scalar 参数 → value-dependency INPUT，运行时 `aclrtMemcpy` D2H 读出
3. Dispatcher `callBoxed` 调用用户 NPU 实现

---

### 2.3 软件功能

| 功能域 | 模块 | 语言 | 描述 |
|--------|------|------|------|
| Epair 导出 | `torch_delegate/api.py` + `converter/` + `epair/` | Python | 将 `ExportedProgram` 转换为带 Ascend IR 标注的 `.epair` 归档 |
| Backend 加载编译 | `csrc/backend/` | C++ | 解析 epair → 构建 GE Graph → 编译 → 执行 |
| Fallback Codegen | `csrc/fallback_codegen/` | C++ | 自动生成 GE 自定义算子代码，集成在构图流程中 |
| 对外接口 | Python API + C++ API | Python/C++ | 面向用户的顶层入口 |

---

### 2.4 设计约束

#### 格式约束

- 对 PT2 Archive 格式做**最小化扩展**——仅在 `torch.export.save` 的 `extra_files` 机制下新增两个文件：
  - `epair_manifest.json`：元数据（版本、后端名称、CANN 版本、节点统计）
  - `ep_json_with_ascend_ir.json`：带 Ascend IR 标注的完整 EP JSON
- `models/model.json` 保持标准 EP JSON 不变，不侵入
- `.epair` 文件完全兼容 PT2 工具链，支持 `torch.export.load` 接口加载并直接执行

#### ABI 约束

- C++ 对外头文件禁止暴露 `std::string`、`std::unordered_map`、`std::filesystem::path`
- 统一使用：`ge::AscendString` 替代 `std::string`、`std::map<ge::AscendString, ge::AscendString>`、`const char*` 替代路径
- 对外类通过 PIMPL 隔离实现细节（`std::unique_ptr<Impl>`）
- 新代码必须在 `-Wall -Wextra -Werror` 下零警告编译

---

### 2.5 假设和依赖关系

TorchDelegate 与周边组件的依赖关系如下：

- **PyTorch 2.9**：提供 `ExportedProgram`、`torch.export.save/load` 序列化框架、`torch.library` 自定义算子注册机制、`torch::Dispatcher` C++ 运行时调度。Python 侧和 C++ 侧均深度依赖。
- **torch_npu**：与 PyTorch 2.9 配套版本，提供 NPU 设备后端（`at::kPrivateUse1`）、`torch_npu.npu.utils.get_cann_version()` 版本查询。Custom op 的 NPU 实现通过 `PrivateUse1` dispatch key 注册；C++ Fallback Codegen 执行时通过 `libtorch_npu.so` 完成 GE tensor 到 `at::Tensor` 的零拷贝转换（`at_npu::native::from_blob`）。
- **torchair**：与 PyTorch 2.9 配套版本，提供 FX → Ascend GE IR 的 converter 注册机制。Python 侧 `TorchAirGEConverter` 通过 torchair 的 `GeConcreteGraph` 和 `parse_node` 完成节点级 IR 转换。
- **CANN >= 9.1.0**：提供 `GeSession`（`libge_runner_v2.so`，含 `CompileGraph` / `RunGraphWithStreamAsync`）、ACL runtime（`libacl_rt.so`，含 stream 管理、设备管理）、`ge::NamedIoNodeBuilder` 构图接口（`<graph/named_io_node_builder.h>`）。安装路径通过 `ASCEND_HOME_PATH` 环境变量指定。

---

## 3. 特性 1：Python 侧 Epair 导出

### 3.1 整体介绍

Python 侧通过三阶段 API 完成模型导出：`export_to_epair`（转换）→ `epair_save`（持久化）→ `epair_load`（加载）。核心思路是将 PyTorch 的 `ExportedProgram` 经 torchair converter 转换为 Ascend GE IR 标注的 EP JSON，连同权重和 manifest 打包为 `.epair` 归档。

```mermaid
sequenceDiagram
  autonumber
  participant User as 用户
  participant API as api.py
  participant Patcher as patch_pytorch
  participant Converter as TorchAirGEConverter
  participant Serializer as serialize
  participant Writer as EpairWriter

  User->>API: export_to_epair(ep, config)
  API->>Patcher: patch_pytorch()→do_auto_functionalize v2
  API->>Converter: TorchAirGEConverter(config)
  Converter->>Converter: convert_graph(ep)
  Note over Converter: run_decompositions→_optimize_fx→parse_node逐个转换
  Converter-->>API: converted_ep (含ascend_ir_map)
  API->>Serializer: serialize(converted_ep)
  Serializer-->>API: artifact
  API->>API: json.loads→ep_json
  API->>Writer: write(ep_json)
  Writer->>Writer: 遍历nodes匹配ascend_ir_map→注入
  Writer-->>API: (ep_json, total, converted)
  API->>API: EpairContents(manifest, ep_json, ep)
  API-->>User: EpairContents
```

### 3.2 功能需求

#### 3.2.1 export_to_epair

**介绍**：将 `ExportedProgram` 转换为 `EpairContents`。内部调用 torchair converter 完成 FX → Ascend GE IR 的节点级转换，转换结果记录在 `ascend_ir_map` 中。

**输入**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ep` | `ExportedProgram` | PyTorch 导出的计算图，含权重和元数据 |
| `config` | `EpairConfig`（可选） | 控制优化开关和保留分解列表 |

`EpairConfig` 字段说明：

```python
@dataclasses.dataclass
class EpairConfig:
    remove_noop_ops: bool = True            # 是否移除无操作算子
    pattern_fusion_pass: bool = True        # 是否启用 pattern fusion
    whitelist_decompositions: set = {       # 保留不分解的算子集合
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.softmax.int,
        # ...
    }
```

**处理**：

1. patch PyTorch：将 `do_auto_functionalize` 替换为 v2 版本（处理 mutable args 和 view 信息）
2. 创建 `TorchAirGEConverter`，传入 `config`
3. `converter.convert_graph(ep)`：
   - `ep.run_decompositions()` 展开 composite ops
   - `_optimize_fx` 应用图优化（remove_noop、pattern fusion）
   - `EpairGraphConverter` 遍历 FX Graph 每个节点，调用 `parse_node` 完成 GE 转换
   - 转换成功：记录 opdef + mapping 到 `ascend_ir_map`
   - 转换失败：创建 mock node，`source=""` 标识
4. `serialize(converted_ep)` → `artifact.exported_program`（JSON bytes）
5. `EpairWriter._inject_ascend_ir(ep_json)`：遍历 EP JSON nodes，匹配 `ascend_ir_map`，注入 `ascend_ir` 字段
6. 构建 `EpairManifest`（version、backend_name、cann_version、节点统计）
7. 封装 `EpairContents` 并返回
8. finally 块中恢复 patched PyTorch 函数

**输出**：`EpairContents(manifest, ep_json, ep)`

**使用示例**：

```python
from torch_delegate import export_to_epair, epair_save
from torch_delegate import EpairConfig

config = EpairConfig(
    remove_noop_ops=True,
    pattern_fusion_pass=True,
)
contents = export_to_epair(ep, config=config)

# 查看转换统计
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")

# 保存
epair_save(contents, "model.epair")
```

#### 3.2.2 epair_save

**介绍**：将 `EpairContents` 写入 `.epair` 文件。

**处理**：
- `manifest` → JSON → `epair_manifest.json`
- `ep_json` → JSON → `ep_json_with_ascend_ir.json`
- 通过 `torch.export.save(contents.ep, f, extra_files={...})` 写入 PT2 Archive

**EP 格式扩展 —— 文件结构**：

```
model.epair（ZIP 结构，兼容 PT2 格式）
├── epair_manifest.json               ← TorchDelegate 新增：元数据
├── ep_json_with_ascend_ir.json       ← TorchDelegate 新增：带 ascend_ir 的 EP JSON
├── models/
│   └── model.json                    ← 标准 EP JSON（无 ascend_ir，由 torch.export.save 生成）
└── data/
    ├── weights/                      ← 权重张量（由 torch.export.save 管理）
    ├── constants/                    ← 常量张量（由 torch.export.save 管理）
    └── model_sample_inputs.pt        ← 示例输入
```

`epair_manifest.json` 格式：

```json
{
  "version": 1,
  "backend_name": "ascend_ge",
  "cann_version": "9.1.0",
  "total_nodes": 20,
  "converted_nodes": 15
}
```

`ep_json_with_ascend_ir.json` 中节点的 `ascend_ir` 字段示例：

```json
{
  "target": "torch.ops.aten.add.Tensor",
  "inputs": [...],
  "outputs": [...],
  "ascend_ir": {
    "source": "converter",
    "ops": [{
      "name": "Add",
      "attr": [{"key": "alpha", "value": {"f": 1.0}}],
      "input": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
      "output": [{"name": "z", "type": "tensor"}]
    }],
    "mapping": {
      "inputs": {"input_0": "x", "input_1": "y"},
      "outputs": {"add_1": "z:0"}
    }
  }
}
```

**兼容性说明**：`.epair` 文件是合法的 PT2 Archive，支持使用 `torch.export.load` 加载并直接执行推理（忽略额外文件即可）。

#### 3.2.3 epair_load

**介绍**：从 `.epair` 文件还原 `EpairContents`。

**处理**：
- `torch.export.load(f, extra_files={'epair_manifest.json': '', 'ep_json_with_ascend_ir.json': ''})`
- 解析 manifest 和 annotated EP JSON
- 返回 `EpairContents(manifest, ep_json, ep)`

**使用示例**：

```python
from torch_delegate import epair_load

contents = epair_load("model.epair")
print(f"Backend: {contents.manifest.backend_name}")

# 直接执行（兼容 PT2）
outputs = contents.ep.module()(inputs)
```

#### 3.2.4 Mock 机制

对于 torchair converter 无法转换的节点，创建 mock node：

```python
try:
    ge_outputs = super().parse_node(target, args, kwargs, meta_outputs)
    self._record_ascend_ir(fx_node_name, ge_outputs, new_opdefs, is_mock=False)
except RuntimeError as e:
    logger.warning(f"Failed to convert {fx_node_name}: {e}")
    ge_outputs = self._create_mock_node_for_op(fx_node_name, target, args, kwargs)
    self._record_ascend_ir(fx_node_name, ge_outputs, new_opdefs, is_mock=True)
```

| source 值 | 含义 | C++ 侧处理 |
|-----------|------|-----------|
| `"converter"` | 正常 GE 转换 | 使用 ascend_ir 中的 opdef 构图 |
| `""`（空字符串） | Mock 节点 | 通过 Fallback Codegen 生成 GE 自定义算子代码 |

### 3.3 数据结构

```python
@dataclasses.dataclass
class NodeIR:
    source: str                      # "converter" 或 ""（mock）
    ops: list[dict]                  # GE opdef protobuf 转 dict
    mapping: dict[str, str] | None   # 输出映射（aten→GE）

@dataclasses.dataclass
class EpairManifest:
    version: int                     # 格式版本号，当前为 1
    backend_name: str                # 后端名称，"ascend_ge"
    cann_version: str                # CANN 版本号
    total_nodes: int                 # EP JSON 中计算节点总数
    converted_nodes: int             # 成功转换的节点数

@dataclasses.dataclass
class EpairContents:
    manifest: EpairManifest          # 元数据
    ep_json: dict[str, Any]          # 带 ascend_ir 的 EP JSON
    ep: ExportedProgram              # 可执行的 ExportedProgram
```

---

## 4. 特性 2：C++ Backend

### 4.1 整体介绍

C++ Backend 负责加载 `.epair` 归档，解析 annotated EP JSON，构建 `ge::Graph`，编译并执行。由以下核心组件协作完成：

```mermaid
classDiagram
  class EpairArchive {
    +TryCreate(const char*) unique_ptr
    +Path() const char*
    +HasRecord(name) bool
    +ReadRecord(name) RecordData
  }
  class EpairModelParser {
    +Parse(json, result) ParsedModel
    +GraphInputs() list~ParsedInputSpec~
    +Nodes() list~ParsedNode~
    +GraphOutputs() list~ParsedOutputSpec~
    +PlaceholderKindOf(name) PlaceholderKind
    +FindInputSpec(name) ParsedInputSpec
    +FindTensorValue(name) ParsedTensorMeta
  }
  class GeWeightLoader {
    +Create(archive, model, result)
    +ReadTensorData(name) TensorHolder
    +RestoreInlineConstTensor(ir_op, result) TensorHolder
  }
  class GeGraphBuilder {
    +GeGraphBuilder(archive, result)
    +Build(result) unique_ptr~ge::Graph~
    +IsFullyOffloaded() bool
  }
  class EpairModelLoader {
    +EpairModelLoader(const char* archive_path)
    +CompileAndLoad(options, stream) Status
    +RunModelWithStreamAsync(stream, inputs, outputs) Status
    +GetGEGraph(graph) Status
  }
  class ValueRegistry {
    +RegisterPlaceholderSource(name, value)
    +RegisterProducerOutput(ref, value)
    +RegisterAlias(alias, ref)
    +Resolve(ref) RegisteredValue
  }
  class FallbackCodegen {
    +GenerateFromModelJson(json_str) CodegenResult
  }

  EpairModelLoader --> EpairArchive
  EpairModelLoader --> GeGraphBuilder
  GeGraphBuilder --> EpairModelParser
  GeGraphBuilder --> GeWeightLoader
  GeGraphBuilder --> ValueRegistry
  GeGraphBuilder --> FallbackCodegen
```

### 4.2 功能需求

#### 4.2.1 EpairArchive

PT2 ZIP 归档的 C++ 封装，提供读取能力。

**接口**：

| 方法 | 说明 |
|------|------|
| `EpairArchive(const char* path)` / `TryCreate(path)` | 构造或工厂方法打开归档 |
| `Path()` → `const char*` | 归档路径 |
| `HasRecord(name)` → `bool` | 是否包含指定 record |
| `ReadRecord(name)` → `RecordData` | 读取指定 record 的字节数据 |

**实现**：基于 `caffe2::serialize::PyTorchStreamReader`。

#### 4.2.2 EpairModelParser

将 annotated EP JSON 解析为强类型 C++ 结构体。

**输入**：`ep_json_with_ascend_ir.json` 的 JSON 字符串

**处理**：
- 解析 `graph_module.graph.nodes` → `ParsedNode` 列表
- 解析 `graph_module.graph.inputs` → placeholder 列表
- 解析 `graph_module.graph.outputs` → output 列表
- 提取每个节点的 `ascend_ir` 字段（source、ops、mapping）
- 提取非 Tensor 参数（scalar、int、bool 等），为 Fallback Codegen 提供输入

**输出**：

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `GraphInputs()` | `std::vector<ParsedInputSpec>` | 所有 graph input |
| `Nodes()` | `std::vector<ParsedNode>` | 所有计算节点 |
| `GraphOutputs()` | `std::vector<ParsedOutputSpec>` | 所有 graph output |
| `PlaceholderKindOf(name)` | `PlaceholderKind` | placeholder 类型（UserInput/Parameter/Buffer/TensorConstant） |
| `FindInputSpec(name)` | `ParsedInputSpec` | 查找指定 input 的 spec 信息 |
| `FindTensorValue(name)` | `ParsedTensorMeta` | 查找指定 tensor 的 shape/dtype 信息 |

**关键数据结构**：

```cpp
struct ParsedNode {
    std::string target;              // 如 "torch.ops.aten.add.Tensor"
    std::vector<std::string> input_names;  // 仅 tensor 输入
    std::vector<std::string> output_names;
    AscendIrInfo ascend_ir;          // 可能为空（fallback 节点）
};

struct AscendIrInfo {
    std::string source;              // "converter" 或 ""
    std::vector<AscendIrOpDef> ops;  // GE opdef 列表
    AscendIrMapping mapping;         // 输入输出名称映射
};
```

#### 4.2.3 GeWeightLoader

从归档读取权重和常量 tensor 数据。

**处理流程**：

```
GeWeightLoader::Create(archive, model, result)
  1. 解析 placeholder → logical_name 映射
  2. 读取 data/weights/model_weights_config.json
  3. 读取 data/constants/model_constants_config.json
  4. 构建 logical_name → TensorRecordInfo 映射
  5. 校验所有 placeholder 在 records 中有对应项

GeWeightLoader::ReadTensorData(name)
  1. 查找 logical_name → TensorRecordInfo
  2. 校验 archive record 存在
  3. 校验 byteorder（仅支持 little-endian）
  4. 读取 bytes → 解码为 vector<T>
  5. 构造 TensorHolder（values + dims + dtype + format）
```

**校验规范**：
- `use_pickle=True` → 报 Error（不支持 pickle 序列化的权重）
- bytes 大小不匹配 → 报 Error
- 不支持的 dtype → 报 Error

#### 4.2.4 GeGraphBuilder（核心）

多阶段构图流程，是本版本最核心的设计。GeGraphBuilder 构造时已完成 Fallback Codegen 预执行、epair 解析和权重加载，`Build()` 驱动以下阶段构图：

```mermaid
sequenceDiagram
  autonumber
  participant Builder as GeGraphBuilder
  participant Context as GraphBuildContext
  participant Parser as EpairModelParser
  participant Loader as GeWeightLoader
  participant Registry as ValueRegistry
  participant Codegen as FallbackCodegen

  Note over Builder,Codegen: 构造阶段（Build()调用前）
  Codegen->>Builder: GenerateFromModelJson(json)
  Note over Codegen: 解析所有source=""节点→生成custom_op.cpp→g++编译→dlopen
  Builder->>Parser: Parse(graph_json, result)
  Builder->>Loader: Create(archive, model, result)

  rect rgb(240,248,255)
    Note over Builder: Build() 阶段1: PrepareContext
    Builder->>Context: new GraphBuildContext("epair_graph")
    Context->>Context: graph = unique_ptr~ge::Graph~("epair_graph")
  end

  rect rgb(255,250,240)
    Note over Builder: 阶段2: PreparePlaceholders
    Builder->>Parser: GraphInputs()
    loop 每个input
      alt UserInput
        Builder->>Builder: NamedIoNodeBuilder(graph).Type("Data").Name(name).Build()
        Builder->>Registry: RegisterPlaceholderSource(name, value)
      else Parameter/Buffer/TensorConstant
        Builder->>Loader: ReadTensorData(name)
        Builder->>Builder: NamedIoNodeBuilder(graph).Type("Const").Name(name).Build()
        Builder->>Registry: RegisterPlaceholderSource(name, value)
      end
    end
  end

  rect rgb(230,255,230)
    Note over Builder: 阶段3: PlanNodes
    Builder->>Parser: Nodes()
    loop 每个node
      alt IsConverterNode (有ascend_ir且source!="")
        Builder->>Context: PlanConverterNode(node, parser, mapper)
        Note over Context: 遍历ascend_ir.ops→Const op走RestoreInlineConstTensor<br/>非Const op使用NamedIoNodeBuilder创建GNode<br/>记录DeferredEdge + DeferredOutputBinding
      else NoOp节点（无GE op的identity passthrough）
        Builder->>Context: PlanConverterNoOpNode(node)
        Note over Context: 直接alias输出到第一个tensor输入
      else Fallback节点（source=""或无ascend_ir）
        Builder->>Builder: PickFallbackOpType(node)
        Builder->>Builder: QueryRegisteredIrDef(op_type)
        Note over Builder: 查找Codegen已注册的自定义op定义
        Builder->>Builder: NamedIoNodeBuilder创建GNode
        Builder->>Context: 记录DeferredEdge + DeferredOutputBinding
      end
    end
  end

  rect rgb(255,240,245)
    Note over Builder: 阶段4: BindDeferredOutputs
    loop 每个deferred_output
      Builder->>Registry: Resolve(source_ref)
      Builder->>Registry: RegisterAlias(alias, source_ref)
    end
  end

  rect rgb(240,255,240)
    Note over Builder: 阶段5: BindDeferredEdges
    loop 每个deferred_edge
      Builder->>Registry: Resolve(src_ref)
      Builder->>Builder: graph->AddDataEdge(src_node, src_idx, dst_node, dst_idx)
    end
  end

  rect rgb(230,230,255)
    Note over Builder: 阶段6: FinalizeGraph
    Builder->>Parser: GraphOutputs()
    loop 每个output
      Builder->>Registry: Resolve(output_name)
      Builder->>Builder: outputs.push_back({node, output_index})
    end
    Builder->>Builder: graph->SetOutputs(outputs)
    Builder->>Builder: return std::move(graph)
  end
```

**关键设计点**：

1. **两阶段构图**：PlanNodes 阶段只创建节点骨架和注册输出引用，不实际连边；BindDeferredEdges 阶段统一连接数据边。这是因为在 PlanNodes 阶段，被引用的 producer 节点可能尚未创建（图中顺序不一定是拓扑序）。

2. **ValueRegistry**：`name → RegisteredValue`（含 `ge::GNode` + `output_index`）的延迟解析注册表，支持：
   - `RegisterPlaceholderSource(name, value)` — 注册输入/常量节点输出
   - `RegisterProducerOutput(ref, value)` — 注册算子节点输出（`"op_name:0"`）
   - `RegisterAlias(alias, ref)` — 注册输出别名（FX tensor name → GE producer ref）
   - `Resolve(ref)` → `RegisteredValue` — 递归解析引用

3. **FallbackCodegen 的预执行**：codegen 在 `Build()` 调用前完成，生成完整 `.so` 并 `dlopen`，注册到 GE 算子表。PlanFallbackNode 只需通过 `QueryRegisteredIrDef()` 查找已注册的定义。

4. **Converter 节点 vs Fallback 节点**：
   - 有 `ascend_ir` 且 `source="converter"`：使用 ascend_ir 中的 opdef 信息，通过 `NamedIoNodeBuilder` 构建 GE 节点
   - 无 `ascend_ir` 或 `source=""`：使用 FallbackCodegen 预注册的 op_type，同样通过 `NamedIoNodeBuilder` 构建

#### 4.2.5 EpairModelLoader（顶层入口）

v1 版本对外提供 `EpairModelLoader` 作为统一的 C++ 入口。构造时即完成 epair 解析和 GE Graph 构建，随后可单步完成编译执行：

```cpp
class EpairModelLoader {
public:
    explicit EpairModelLoader(const char *archive_path);

    Status CompileAndLoad(
        const std::map<ge::AscendString, ge::AscendString> &options,
        void *stream);

    Status RunModelWithStreamAsync(
        void *stream,
        const std::vector<gert::Tensor> &inputs,
        std::vector<gert::Tensor> &outputs);

    Status GetGEGraph(ge::Graph *graph) const;
};
```

**处理流程**：
1. 构造：打开归档 → 校验 annotated payload → 初始化 GeGraphBuilder（内部创建 Parser + WeightLoader）→ 调用 Build 构建 GE Graph
2. `CompileAndLoad`：通过 `GeSession::AddGraph` + `CompileGraph` 完成在线编译并加载到指定 stream
3. `RunModelWithStreamAsync`：通过 `GeSession::RunGraphWithStreamAsync` 执行推理
4. `GetGEGraph`：返回内部 `ge::Graph` **副本**，供用户自管编译执行

---

## 5. 特性 3：Fallback Codegen

### 5.1 整体介绍

#### 三种算子场景

TorchDelegate v1 根据算子是否携带 `ascend_ir` 标注、是否有 GE 入图交付件，将算子分为三种场景：

| 场景 | 描述 | ascend_ir | GE 交付件 | 构图方式 | v1 支持 |
|------|------|-----------|-----------|----------|---------|
| **场景 1** | 有完整入图交付件的算子。GE 侧已注册算子定义（`REG_OP`），torchair 提供完整 converter 转换 | `source="converter"`，ops 含完整 opdef | 有 | `NamedIoNodeBuilder` 根据 ascend_ir.opdef 直接构图 | ✅ 支持 |
| **场景 2** | Fallback 回调算子，无 GE 入图交付件。用户通过 `torch.library` 注册 PyTorch 自定义算子，不需要手写 GE 自定义算子代码 | `source=""` 或无 ascend_ir | 无，由 Fallback Codegen 自动生成 | Codegen 生成 GE `EagerExecuteOp` 代码 → 编译 `.so` → `dlopen` 注册 → 运行时回调 PyTorch Dispatcher 调用用户实现 | ✅ 支持 |
| **场景 3** | CPU 算子，需要在 CPU 上通过 libtorch 执行 | 无 | 无 | 需 HeteroCodegen 生成 CPU 执行代码 | ❌ 暂不支持 |

三种场景在 GeGraphBuilder 中的分流逻辑：

```
GeGraphBuilder::Build()
  ├── 构造阶段：FallbackCodegen::GenerateFromModelJson()
  │     扫描所有 source="" 的节点 → 批量生成场景2 的自定义算子代码
  │
  └── PlanNodes 阶段
        ├── IsConverterNode(node) == true    → 场景1：PlanConverterNode
        ├── IsConverterNoOpNode(node) == true → 场景1 NoOp：PlanConverterNoOpNode（alias 直通）
        └── else                              → 场景2：PlanFallbackNode（查已注册的自定义 op）

v1 不支持场景3。若图中有场景3节点，GeGraphBuilder::IsFullyOffloaded() 返回 false。
```

#### Fallback Codegen 流程

```mermaid
sequenceDiagram
  autonumber
  participant Builder as GeGraphBuilder
  participant Codegen as FallbackCodegen
  participant Schema as PyTorch Dispatcher
  participant Compiler as g++
  participant GEGraph as GE Graph

  Note over Builder,Codegen: 构造阶段：GeGraphBuilder(archive, result)
  Builder->>Builder: 从archive读取ep_json_with_ascend_ir.json
  Builder->>Codegen: GenerateFromModelJson(graph_json)
  Codegen->>Codegen: ParseTargetsFromJson → 过滤source=""节点
  loop 每个fallback target
    Codegen->>Schema: findSchemaOrThrow(op_name, overload)
    Schema-->>Codegen: op_handle, schema
    Codegen->>Codegen: 分析arguments: Tensor/非Tensor类型
  end
  Codegen->>Codegen: EmitCustomOpCpp(schemas)
  Note over Codegen: 生成custom_op.cpp<br/>含MapGeDtypeToTorch/ReadScalarInput/CreateNpuTensor<br/>每个op一个EagerExecuteOp子类
  Codegen->>Compiler: g++ -shared -fPIC -o libcust_opapi.so
  Compiler-->>Codegen: .so文件
  Codegen->>Codegen: dlopen → REG_OP + REG_AUTO_MAPPING_OP注册
  Codegen-->>Builder: CodegenResult(so_handle, op_names)

  Note over Builder,GEGraph: Build阶段：PlanFallbackNode
  Builder->>Builder: PickFallbackOpType(node.target)
  Builder->>Builder: QueryRegisteredIrDef(op_type)
  Note over Builder: 查找Codegen已注册到GE的自定义op定义
  Builder->>Builder: NamedIoNodeBuilder.Type(op_type).Name(name)
  Note over Builder: 根据RegisteredIrDef构建inputs/outputs/attrs
  Builder->>Builder: graph->AddDataEdge(src, dst)
```

### 5.2 功能需求

#### 5.2.1 算子描述提取

Codegen 从 EP JSON 的每个 `source=""` 节点提取：

| 信息 | 来源 | 用途 |
|------|------|------|
| 算子 target | `node.target` | 通过 Dispatcher 查找 PyTorch Schema |
| Tensor 输入 | `node.inputs[].arg.as_tensor` | 生成 GE INPUT 定义 |
| 非 Tensor 参数 | `node.inputs[].arg.as_int/as_float/as_bool/...` | 生成 value-dependency INPUT（IntTensorType 等），运行时 D2H 读出 |
| 输出名 | `node.outputs[]` | 生成 GE OUTPUT 定义 |

#### 5.2.2 Schema 解析

通过 `torch::Dispatcher::singleton().findSchemaOrThrow(target, "")` 获取算子 schema，每个参数映射为 `SchemaArgInfo`（含 `type_category`、`input_kind`、`ge_tensor_type`、`has_default` 等）。区分 21 种 `InputKind`：

- `TENSOR` / `OPTIONAL_TENSOR` / `TENSOR_LIST` → GE `TensorType::ALL()` input
- `INT` / `FLOAT` / `BOOL` / `STRING` → GE `IntTensorType()` / `FloatTensorType()` / `BoolTensorType()` / `StringTensorType()` 等 value-dependency input
- `SCALAR` → GE `TensorType::ALL()`，运行时根据 dtype 分支为 int64 或 float→double

所有非 tensor 参数通过 `InputsDataDependency` 注册，GE 在执行前等待这些 1 元素 tensor 从 device 回读到 host。

#### 5.2.3 Execute 代码生成（实际模式）

```cpp
graphStatus Execute(gert::EagerOpExecutionContext* ctx) override {
    // 1. Tensor 输入：GetInputTensor + CreateNpuTensor（零拷贝 from_blob）
    at::Tensor torch_input_0 = CreateNpuTensor(ctx->GetInputTensor(0));

    // 2. 标量输入：D2H 读出 1 元素 buffer → reinterpret_cast
    int64_t scalar_val = *reinterpret_cast<const int64_t*>(
        ReadScalarInput(ctx->GetInputTensor(1)));  // aclrtMemcpy D2H

    // 3. 组装 c10::IValue 参数列表
    std::vector<c10::IValue> args;
    args.emplace_back(torch_input_0);
    args.emplace_back(scalar_val);

    // 4. Dispatcher callBoxed 调用
    static const auto op_handle = torch::Dispatcher::singleton()
        .findSchemaOrThrow("my_ops::add_scalar", "");
    op_handle.callBoxed(args);

    // 5. 输出：aclrtMemcpyAsync D2D 或 MakeOutputRefInput（inplace op）
    return ge::GRAPH_SUCCESS;
}
```

**Tensor 零拷贝**：使用 `at_npu::native::from_blob` 直接引用 GE tensor 的 device 内存，无数据拷贝。

**Scalar 参数处理**：不作为 GE ATTR，而是作为 value-dependency INPUT（`IntTensorType()` 等 1 元素 tensor）。Execute 时 D2H 读出标量值，转为对应 C++ 类型后组装 `c10::IValue`。

#### 5.2.4 参数类型映射

| PyTorch Schema 类型 | InputKind | GE input type | Execute 读取方式 |
|---------------------|-----------|---------------|-----------------|
| `Tensor` | TENSOR | `TensorType::ALL()` | `CreateNpuTensor` |
| `Tensor?` | OPTIONAL_TENSOR | `.OPTIONAL_INPUT` | 判空 |
| `Tensor[]` | TENSOR_LIST | `.DYNAMIC_INPUT` | 循环索引 |
| `int` | INT | `IntTensorType()` | `reinterpret_cast<int64_t>` |
| `float` / `double` | FLOAT | `FloatTensorType()` | `reinterpret_cast<float>` → `double` |
| `bool` | BOOL | `BoolTensorType()` | `reinterpret_cast<uint8_t>` → `bool` |
| `str` | STRING | `StringTensorType()` | 读 buffer → `string` |
| `int[]` | INT_LIST | `IntTensorType()` | 按 `sizeof(int64_t)` 切分 |
| `Scalar` | SCALAR | `TensorType::ALL()` | 按 dtype 分支 int64 或 float→double |

---

## 6. 对外接口

### 6.1 Python 接口

#### export_to_epair

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents
```

#### epair_save

```python
def epair_save(contents: EpairContents, f: str | os.PathLike) -> None
```

#### epair_load

```python
def epair_load(archive_path: str) -> EpairContents
```

#### 完整调用示例

```python
from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate import EpairConfig
from torch.export import export

# 1. 导出
ep = export(model, (inputs,))
config = EpairConfig(remove_noop_ops=True, pattern_fusion_pass=True)
contents = export_to_epair(ep, config=config)
epair_save(contents, "model.epair")
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")

# 2. 加载并执行（兼容 PT2）
loaded = epair_load("model.epair")
outputs = loaded.ep.module()(inputs)
```

### 6.2 C++ 对外接口（ABI-safe）

对外头文件目录：`csrc/inc/external/torch_delegate/`，包含两个文件：`td_types.h` 和 `epair_model_loader.h`。

#### td::Status（td_types.h）

```cpp
namespace td {
using Status = int64_t;

inline constexpr Status SUCCESS = 0;
inline constexpr Status FAILED = 1;
inline constexpr Status INVALID_PARAM = 2;
inline constexpr Status INTERNAL_ERROR = 3;
inline constexpr Status UNIMPLEMENTED = 4;
}
```

#### td::EpairModelLoader（epair_model_loader.h）

```cpp
class EpairModelLoader {
public:
    explicit EpairModelLoader(const char *archive_path);
    ~EpairModelLoader();
    EpairModelLoader(const EpairModelLoader &) = delete;
    EpairModelLoader &operator=(const EpairModelLoader &) = delete;

    Status CompileAndLoad(
        const std::map<ge::AscendString, ge::AscendString> &options,
        void *stream);

    Status RunModelWithStreamAsync(
        void *stream,
        const std::vector<gert::Tensor> &inputs,
        std::vector<gert::Tensor> &outputs);

    Status GetGEGraph(ge::Graph *graph) const;
};
```

#### 完整调用示例（JIT 在线编译并执行）

```cpp
#include "torch_delegate/epair_model_loader.h"
#include "torch_delegate/td_types.h"

int main() {
    // 1. 构造即完成 epair 解析和 GE Graph 构建
    td::EpairModelLoader loader("model.epair");

    // 2. JIT 编译并加载
    std::map<ge::AscendString, ge::AscendString> options = {
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
        {ge::AscendString(ge::ir_option::SOC_VERSION), ge::AscendString("Ascend910B2")},
    };
    void* stream = nullptr;
    aclrtCreateStream(&stream);

    td::Status status = loader.CompileAndLoad(options, stream);
    if (status != td::SUCCESS) return 1;

    // 3. 执行推理
    std::vector<gert::Tensor> inputs = /* ... */;
    std::vector<gert::Tensor> outputs;
    status = loader.RunModelWithStreamAsync(stream, inputs, outputs);
    if (status != td::SUCCESS) return 1;

    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);

    // 或只取 Graph，由用户自管理编译执行
    // ge::Graph graph;
    // loader.GetGEGraph(&graph);
    return 0;
}
```

#### 接口检查项

| 检查项 | 子检查项 | 是否涉及 |
|--------|----------|---------|
| **接口说明** | 接口是否需要评审 | 是（组件间接口） |
| **接口说明** | 是否需要补充资料说明 | 是 |
| **接口说明** | 是否明确接口原型、功能、返回值 | 是 |
| **接口兼容** | 行为兼容性，修改前后是否变化 | 是（新接口，无兼容问题） |
| **接口兼容** | 新接口在老版本上是否能工作 | 是（v1 新接口，不涉及） |
| **接口兼容** | 是否涉及使用场景、调用时序约束 | 是（需先 CompileAndLoad 再 Run） |
| **接口约束** | 不满足约束条件时是否能清晰报错 | 是（Status 错误码） |
| **接口约束** | 是否需要设计单独的测试用例 | 是 |

---

## 7. 交叉特性分析

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态 Shape | 适用 | GeGraphBuilder 构图、GeSession::CompileGraph 在线编译均支持；构图时使用固定 shape 创建 NamedIoNodeBuilder 节点 |
| 动态 Shape | 适用 | 在 `torch.export` 阶段通过 `dynamic_shapes` 参数指定输入 shape 范围为动态，导出到 epair 后 C++ 侧根据算子的符号信息自动设置 GE 节点为动态 shape，由 GE 内部处理运行时 shape 推导 |
| 动态 Shape 静态子图（图拆分） | 适用 | GeSession 同时管理静态子图（v1 路径）和动态子图（v2 路径）；TorchDelegate 构图阶段不区分，由 GE 内部在 CompileGraph 时根据 shape 信息分流 |
| 离线场景（atc / aclgrphBuildModel） | 不支持 | v1 仅提供 JIT 在线路径，不调用 aclgrphBuildModel 等离线编译接口 |
| 在线场景（EpairModelLoader / GeSession） | 适用 | 全流程通过 GeSession::AddGraph + CompileGraph + RunGraphWithStreamAsync 完成 |

---

## 8. 软件设计

### 8.1 关键技术

#### 8.1.1 两阶段构图

PlanNodes 与 BindDeferredEdges 分离。PlanNodes 阶段按图顺序逐节点创建 GE 节点骨架，但数据边先不连接（此时被依赖节点可能尚未创建）。所有节点的输出先注册到 `ValueRegistry`，待所有节点骨架创建完成后，BindDeferredEdges 阶段统一解析引用并连边。

#### 8.1.2 ValueRegistry 注册表

核心数据结构，支持 name → RegisteredValue（含 `ge::GNode` + `output_index`）的延迟解析：

```
ValueRegistry:
  placeholder_sources_:  name → RegisteredValue    // 输入/常量节点
  producer_outputs_:     "op:idx" → RegisteredValue // 算子输出
  aliases_:              fx_name → "op:idx"        // FX tensor名 → GE producer引用
```

解析链：`Resolve(fx_output_name)` → 查 aliases_ 得 `"op:0"` → 查 producer_outputs_ 得 `RegisteredValue`。

#### 8.1.3 严格报错

- 校验宏：`TD_ASSERT`、`TD_ASSERT_NOTNULL`、`TD_ASSERT_TRUE`、`TD_CHK_STATUS`
- `ErrorResult` 类型推导：失败时自动返回默认构造的对应类型（`false`、`nullptr`、空 vector、空 string 等）
- 严格解析：JSON 中遇到未知字段类型 → Error，不静默跳过

#### 8.1.4 ABI 安全

- 对外头文件使用 `ge::AscendString` 替代 `std::string`
- 路径参数使用 `const char*`
- PIMPL 隔离实现细节
- 编译选项使用 `std::map<ge::AscendString, ge::AscendString>`

### 8.2 错误处理

#### 8.2.1 系统错误

| 错误类型 | 处理方式 |
|----------|----------|
| 归档文件不存在/不可读 | 构造时返回 FAILED |
| JSON 解析失败/字段缺失 | parser 返回 Error，透传至调用方 |
| 权重数据损坏/size 不匹配 | weight loader 返回 Error |
| Graph 构建失败 | builder 返回 Error |
| 编译失败（GeSession::CompileGraph） | 返回对应 Status 错误码 |

#### 8.2.2 接口错误

| Status | 值 | 说明 |
|--------|-----|------|
| `SUCCESS` | 0 | 成功 |
| `FAILED` | 1 | 通用失败 |
| `INVALID_PARAM` | 2 | 参数错误 |
| `INTERNAL_ERROR` | 3 | 内部错误 |
| `UNIMPLEMENTED` | 4 | 未实现 |

### 8.3 安全检查

| 检查项 | 是否涉及 | 说明 |
|--------|----------|------|
| 资源生命周期管理 | 是 | GeSession 在 EpairModelLoader 内部管理，遵循 RAII |
| 是否创建新线程 | 否 | GE 编译和执行均为同步调用，不创建新线程 |
| 编码军规 | 是 | 遵循 Google C++ Style Guide，零警告原则 |

---

## 9. 兼容性检查

| 检查项 | 说明 | 状态 |
|--------|------|------|
| .epair 向后兼容 | v1 是首个版本，无兼容问题 | N/A |
| PT2 兼容性 | .epair 是合法 PT2 Archive，可由 torch.export.load 加载 | 兼容 |

---

## 10. DT 设计

### 10.1 测试边界

- **测试入口**：Python 侧 `export_to_epair` + `epair_save` / `epair_load`；C++ 侧 `EpairModelLoader`
- **测试出口**：验证 EP JSON 结构、ascend_ir 字段完整性、GE Graph 拓扑正确性、权重加载正确性
- **打桩模块**：CANN stub（libascendcl.so）、GE stub（libge_runner_v2.so）

### 10.2 测试设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|-----------|----------|----------|
| **功能** | export_to_epair 基本导出 | 单算子模型 → 验证 EP JSON 含 ascend_ir | UT |
| **功能** | Mock 节点生成 | 不识别的 op → 验证 source="" | UT |
| **功能** | epair_save / epair_load 往返 | 保存后加载 → 验证 manifest、ep_json、ep 一致 | UT |
| **功能** | GeGraphBuilder 构图 | 全连接/多分支模型 → 验证 GE Graph 结构和边 | UT |
| **功能** | GeWeightLoader 权重读取 | 含 weight/constant 模型 → 验证 tensor 数据正确 | UT |
| **功能** | EpairModelLoader 端到端 | 完整模型 → CompileAndLoad → RunModelWithStreamAsync → 验证输出 | ST |
| **功能** | Fallback Codegen | 不含 ascend_ir 的节点 → 触发 codegen 路径 | UT |
| **功能** | EpairConfig 控制 | 不同 config 配置 → 验证转换行为差异 | UT |
| **性能** | 编译时长 | 多模型统计编译耗时 | ST |
| **精度** | 输出精度对比 | 导出后执行 vs 原始模型 → 验证输出一致性 | ST |
| **兼容性** | PT2 加载 | .epair 通过 torch.export.load 执行 | ST |
| **兼容性** | ABI 兼容 | 在 _GLIBCXX_USE_CXX11_ABI=0/1 下编译运行 | UT |
| **特性交叉** | 静态 Shape | 固定 shape 模型的全流程通过 | ST |

---

## 11. 验收标准

1. **CI 工程验收**：按照 README.md 和构建指南，可在测试环境上完成 torch-delegate 的构建、安装、升级、demo 执行。可使用脚本一条命令完成环境准备与构建（不含网络问题导致的失败）。

2. **功能验收**：
   - 基于 JD 客户网络验收，成功导出 epair 模型，使用 C++ 接口能够加载并执行成功。
   - 注册 torch 自定义算子，导出包含自定义算子的模型，使用 C++ 接口能正常加载，正确生成 Fallback 回调函数，模型执行成功。（约束：630 之前仅支持 Tensor 输入算子，如果算子中包含 scalar，报错提示，scalar 预 715 支持）
   - 导出 epair 后，可以使用 `torch.export.load` 接口正常加载并在 CPU 侧执行推理。

3. **精度验收**：加载执行 epair 模型，输出精度应与单算子 / torchair 图模式相近。

4. **性能验收**：加载执行 epair 模型，端到端执行性能需要与 torchair 图模式一致。

5. **交叉特性**：GE 在线场景支持的特性都应当支持，建议重点看护语言无关自定义算子、自动融合、动态 shape、DataDump、Profiling 相关的特性。

6. **接口验收**：对外 API 支持在 ABI=0 和 ABI=1 的应用中使用。对外接口资料清晰、易懂，并提供调用示例。

7. **维测功能**：支持打印 APP plog，调用接口失败有明确的 ERROR 日志。说明：暂不支持错误码特性。
