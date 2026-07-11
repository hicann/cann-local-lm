# Export 模块设计文档

## 概述

Export 模块负责将 PyTorch 模型导出为 epair 格式，包含模型结构描述和权重数据。导出产物为 **独立文件集合**（非归档），便于 C++ 后端按需加载。

### 端到端流程

```
PyTorch Model
    │
    ▼  torch.export.export
ExportedProgram
    │
    ▼  torch_delegate.export_to_epair
EpairContents (ep_json + state_dict)
    │
    ▼  consolidated_weights_save
output_dir/
├── ep_json_with_ascend_ir.json
├── weight_info.json
├── weights_0.bin
├── weights_1.bin
└── ...
    │
    ▼  C++ EpairModelLoader(model_json_path, weights_index_path)
ge::Graph
```

## 核心数据结构

### EpairContents

`EpairContents` 来自 `torch_delegate`，是 epair 加载后的内存表示，包含：

- `ep`: `ExportedProgram` 对象，可调用 `state_dict()` 获取权重
- `ep_json`: 带 ascend IR 标注的模型 JSON
- `graph_signature`: 图签名，描述参数/缓冲区/常量的占位符映射

### ConsolidatedWeights

```python
@dataclasses.dataclass
class ConsolidatedWeights:
    bin_files: list[bytes]                    # 多个 bin 文件的原始字节列表
    weights_index: dict[str, list[dict]]      # {bin文件名: [{"name", "offset", "size"}, ...]}
```

`ConsolidatedWeights` 只包含权重数据，不含 EP JSON。由 `pack_weights()` 生成，`consolidated_weights_save()` 写入磁盘。

## 文件布局

```
output_dir/
├── ep_json_with_ascend_ir.json    # 带 ascend_ir 的完整模型描述
├── weight_info.json               # 张量索引（含 bin 文件名引用）
├── weights_0.bin                  # 拼接的原始张量数据（文件 0）
├── weights_1.bin                  # 拼接的原始张量数据（文件 1，切分时产生）
└── ...                            # 更多 bin 文件（按需切分）
```

所有文件直接写入 `output_dir/`，不再创建 `data/` 子目录。

### ep_json_with_ascend_ir.json

标准的 ExportedProgram JSON 序列化，每个节点上增加 `ascend_ir` 字段（包含 `source`、`ops`、`mapping`），描述该节点对应的 Ascend 算子映射。此文件始终写入。

### weight_info.json

```json
{
  "weights_0.bin": [
    {"name": "weight", "offset": 0,  "size": 48}
  ],
  "weights_1.bin": [
    {"name": "bias",   "offset": 0,  "size": 36}
  ]
}
```

JSON 对象，key 为 bin 文件名（相对于 `weight_info.json` 所在目录解析），value 为该文件中的张量条目数组。每个条目字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 张量的逻辑名称（与 state_dict key 一致） |
| `offset` | int | 在对应 bin 文件中的字节偏移量 |
| `size` | int | 张量数据的字节大小 |

### weights_N.bin

所有参数（parameters）、缓冲区（buffers）和 lifted tensor constant 的原始张量数据按顺序拼接成的二进制文件。当单个 bin 文件超过 `max_file_size` 时，后续张量自动写入下一个文件。

## 文件切分机制

`pack_weights` 支持按 `max_file_size` 参数（默认 5KB）对输出 bin 文件进行切分：

### 切分规则

1. **逐张量检查**：遍历张量，检查将当前张量加入当前 bin 文件后是否超过 `max_file_size`
2. **不跨文件切分**：单个张量永远不会被拆分到两个文件中。若当前 bin 文件非空且加入新张量会超限，则先将当前 bin 文件封存，新张量从下一个文件开始写入
3. **超大张量**：若单个张量本身超过 `max_file_size`，它将独占一个 bin 文件（不拆分）
4. **命名规则**：bin 文件命名为 `weights_0.bin`、`weights_1.bin`、...，序号从 0 开始递增

### 切分示例

假设有 3 个张量，大小分别为 48B、36B、6000B，`max_file_size=5120`（5KB）：

```
weights_0.bin: [tensor_0: 48B] [tensor_1: 36B]    → 84B (< 5120)
weights_1.bin: [tensor_2: 6000B]                   → 6000B (单独文件，超限但不拆分)
```

对应 `weight_info.json`：
```json
{
  "weights_0.bin": [
    {"name": "tensor_0", "offset": 0,  "size": 48},
    {"name": "tensor_1", "offset": 48, "size": 36}
  ],
  "weights_1.bin": [
    {"name": "tensor_2", "offset": 0,  "size": 6000}
  ]
}
```

## 接口说明

### pack_weights(contents, *, max_file_size=5120) -> ConsolidatedWeights

从 `EpairContents` 提取张量数据并打包为多个 bin 文件（内存中）。

参数：
- `contents`: `EpairContents` 对象
- `max_file_size`: 单个 bin 文件最大字节数，超过则切分到下一个文件。默认 5KB（测试用）

返回：
- `ConsolidatedWeights` 对象（`bin_files` + `weights_index`）

实现逻辑：
1. 从 `contents.ep.state_dict()` 收集所有参数和缓冲区
2. 根据 `graph_signature` 的 `inputs_to_parameters`、`inputs_to_buffers`、`inputs_to_lifted_tensor_constants` 筛选出图中引用的张量
3. 张量数据按 `numpy.tobytes()` 序列化
4. 按切分规则依次写入 bin 文件，构建索引（name/file/offset/size）

### consolidated_weights_save(contents, output_dir, *, save_weights=True, max_file_size=5120)

将 `EpairContents` 写入磁盘。

参数：
- `contents`: `EpairContents` 对象（由 `torch_delegate.export_to_epair()` 返回）
- `output_dir`: 输出目录（文件直接写入此目录，不创建子目录）
- `save_weights`: 为 `True` 时调用 `pack_weights` 生成 `weights_*.bin` 和 `weight_info.json`；为 `False` 时只写 EP JSON
- `max_file_size`: 单个 bin 文件最大字节数，传递给 `pack_weights`

内部流程：
1. 若 `save_weights=True`，调用 `pack_weights(contents, max_file_size=...)` 得到 `ConsolidatedWeights`
2. 依次写入 `weights_0.bin`、`weights_1.bin`、...（受 `save_weights` 控制）
3. 写入 `weight_info.json`（受 `save_weights` 控制）
4. 从 `contents.ep_json` 写入 `ep_json_with_ascend_ir.json`（始终写入）

## C++ 加载接口

### EpairModelLoader

```cpp
// 构造：传入 model JSON 路径和 weight_info.json 路径
td::EpairModelLoader loader(
    "/path/to/ep_json_with_ascend_ir.json",
    "/path/to/weight_info.json");

// 获取 GE Graph 副本
ge::Graph graph;
loader.GetGEGraph(&graph);
```

- 构造函数接收 2 个文件路径：`model_json_path` 和 `weights_index_path`
- 权重 bin 文件路径从 `weight_info.json` 的 `file` 字段自动推导（相对于索引文件目录）
- 使用 PIMPL 模式，对外 ABI 稳定
- `GetGEGraph` 返回内部 Graph 的拷贝副本，可单独编译和执行

### C++ 内部调用链

```
EpairModelLoader(model_json_path, weights_index_path)
    │
    ▼  Impl::Initialize
    ├── 读取 model JSON 文件内容
    └── GeGraphBuilder(model_json_string, weights_index_path, result)
            │
            ├── (可选) FallbackCodegen::GenerateFromModelJson
            ├── EpairModelParser::Parse(model_json) → ParsedModel
            ├── GeWeightLoader::Create(weights_index_path, model, result)
            │       │
            │       ├── 解析 weight_info.json
            │       ├── 推导 bin 文件绝对路径（index 目录 + file 字段）
            │       └── 构建 logical_name → TensorRecordInfo 映射
            │
            └── Build(result) → ge::Graph
                    ├── PrepareContext
                    ├── PreparePlaceholders (创建 Data/Const 节点)
                    ├── PlanNodes (创建 Ascend 算子节点)
                    ├── BindDeferredOutputs
                    ├── BindDeferredEdges
                    └── FinalizeGraph
```

### GeWeightLoader 懒加载机制

`GeWeightLoader` 不在构造时一次性读取所有 bin 文件，而是采用 **懒加载** 策略：

1. `LoadConsolidated` 只解析 `weight_info.json`，为每个张量记录其所属 bin 文件的绝对路径（存入 `TensorRecordInfo::record_path`）
2. `ReadTensorData` 首次访问某个 bin 文件时，才将其读入 `bin_files_cache_`（`unordered_map<path, vector<uint8_t>>`）
3. 后续访问同一 bin 文件的张量直接从缓存读取

### Fallback Codegen

`GeGraphBuilder` 构造时可选执行 fallback codegen（由环境变量 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` 控制）：

- 未设置或 `1`/`true`：执行 fallback codegen，为无法直接映射到 Ascend IR 的算子生成 C++ 代码
- `0`/`false`：跳过 fallback codegen

## 使用示例

### Python 导出

```python
import torch
import torch_delegate
from cann_lm.converter.export.epair_utils import consolidated_weights_save

model = MyModel().eval()
sample_inputs = (torch.randn(3, 4),)
ep = torch.export.export(model, sample_inputs, strict=False)
epair = torch_delegate.export_to_epair(ep)

# 保存完整目录（含权重，默认 5KB 切分）
consolidated_weights_save(epair, "/path/to/output", save_weights=True)

# 仅保存 EP JSON，跳过权重
consolidated_weights_save(epair, "/path/to/output", save_weights=False)

# 自定义切分大小（如 1MB）
consolidated_weights_save(epair, "/path/to/output", max_file_size=1024*1024)
```

### C++ 加载

```cpp
#include "torch_delegate/epair_model_loader.h"

td::EpairModelLoader loader(
    "/path/to/ep_json_with_ascend_ir.json",
    "/path/to/weight_info.json");

ge::Graph graph("delegate_graph");
td::Status status = loader.GetGEGraph(&graph);
if (status != td::SUCCESS) { /* error handling */ }

// 后续可独立编译和执行 graph
```

## 与 torch_delegate 的关系

- `EpairContents` 由 `torch_delegate.export_to_epair()` 返回
- `consolidated_weights_save` 生成的目录可由 C++ 后端 `EpairModelLoader` 直接加载
- `pack_weights` 不依赖 `torch_delegate` 之外的第三方库
- `torch_delegate.epair_save()` 可将 `EpairContents` 另存为 `.pt2` 归档文件（用于 Python 侧 `epair_load` 回环验证）

## 源文件索引

| 文件 | 职责 |
|------|------|
| `epair_utils.py` | 权重打包（`pack_weights`）、磁盘写入（`consolidated_weights_save`） |
| `csrc/inc/external/torch_delegate/epair_model_loader.h` | C++ 公共 API：`EpairModelLoader` |
| `csrc/backend/epair_model_loader.cc` | `EpairModelLoader::Impl` 实现（初始化、Graph 构建） |
| `csrc/backend/ge_graph_builder.h/.cc` | `GeGraphBuilder`：解析 model JSON + 构建 GE Graph |
| `csrc/backend/ge_weight_loader.h/.cc` | `GeWeightLoader`：解析 `weight_info.json`、懒加载 bin 文件 |
| `csrc/backend/epair_model_parser.h/.cc` | `EpairModelParser`：解析 `ep_json_with_ascend_ir.json` |
| `csrc/fallback_codegen/fallback_codegen.h/.cc` | `FallbackCodegen`：为未映射算子生成 C++ 代码 |

## 待办事项

1. **pt2 格式中 `data/constants/model_constants_config.json` 尚未处理**
   - pt2 归档内 `data/constants/model_constants_config.json` 描述了 lifted tensor constants 的元信息（dtype、shape 等），当前导出流程未解析和保留此文件
   - 需在 `export_to_epair` 或 `consolidated_weights_save` 中补充解析逻辑，确保 lifted tensor constants 的元信息完整传递到 C++ 侧

2. **HuggingFace 模型路径的 tokenizer 和 LLM config 处理**
   - 当前导出流程仅处理 PyTorch ExportedProgram（权重 + 图结构），不涉及 tokenizer 和模型配置
   - 若输入为 HuggingFace 模型路径（如 `/path/to/Qwen3-5`），还需处理：
     - `tokenizer.json` / `tokenizer_config.json`：分词器配置
     - `config.json` / `generation_config.json`：模型架构配置（层数、隐藏维度、头数等）
     - 需将这些配置文件一并导出或转换为 C++ 后端可读取的格式
