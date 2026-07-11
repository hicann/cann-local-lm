# Export 模块设计文档

## 概述

Export 模块负责将 PyTorch 模型导出为 epair 格式，包含模型结构描述和权重数据。支持两种导出模式：

- **epair 目录模式**：将权重和模型描述拆分为独立文件，便于 C++ 后端按需加载
- **epair 归档模式**：使用 PyTorch PT2Archive 打包为单个文件（默认）

## 核心数据结构

### EpairContents

`EpairContents` 来自 `torch_delegate`，是 epair 加载后的内存表示。`consolidated_weights_save` 直接接受此类型作为输入，从中提取 `ep_json` 写入 `ep_json_with_ascend_ir.json`。

### ConsolidatedWeights

```python
@dataclasses.dataclass
class ConsolidatedWeights:
    weights_bin: bytes          # 拼接的原始张量字节
    weights_index: list[dict]   # 索引: [{"name", "offset", "size"}, ...]
```

`ConsolidatedWeights` 只包含权重数据，不含 EP JSON。由 `pack_weights()` 生成，也可作为内部中转结构使用。

## 文件布局 (目录模式)

```
output_dir/
└── data/
    ├── weights.bin                    # 拼接的原始张量数据
    ├── weights_index.json             # 张量索引
    └── ep_json_with_ascend_ir.json    # 带 ascend_ir 的完整模型描述
```

### weights.bin

所有参数、缓冲区和 lifted tensor constant 的原始张量数据按顺序拼接成的二进制文件。

### weights_index.json

```json
[
  {"name": "weight", "offset": 0, "size": 48},
  {"name": "bias",   "offset": 48, "size": 12}
]
```

每个条目记录张量的逻辑名称、在 `weights.bin` 中的偏移量和字节大小。

### ep_json_with_ascend_ir.json

标准的 ExportedProgram JSON 序列化，每个节点上增加 `ascend_ir` 字段（包含 `source`、`ops`、`mapping`）。

## 接口说明

### pack_weights(contents) -> ConsolidatedWeights

从 `EpairContents` 提取张量数据并打包。

参数：
- `contents`: `EpairContents` 对象

返回：
- `ConsolidatedWeights` 对象（内存中的打包结果）

实现逻辑：
1. 从 `contents.ep.state_dict()` 收集所有参数和缓冲区
2. 根据 `graph_signature` 筛选出图中引用的张量
3. 张量数据按 `numpy.tobytes()` 序列化后拼接
4. 构建索引（name/offset/size）

### consolidated_weights_save(contents, output_dir, *, save_weights=True)

将 `EpairContents` 写入磁盘。

参数：
- `contents`: `EpairContents` 对象（由 `torch_delegate.epair_load()` 返回）
- `output_dir`: 输出目录（会在其下创建 `data/` 子目录）
- `save_weights`: 为 `True` 时调用 `pack_weights` 生成 `weights.bin` 和 `weights_index.json`；为 `False` 时只写 EP JSON

内部流程：
1. 若 `save_weights=True`，调用 `pack_weights(contents)` 得到 `ConsolidatedWeights`
2. 写入 `weights.bin` 和 `weights_index.json`（受 `save_weights` 控制）
3. 从 `contents.ep_json` 写入 `ep_json_with_ascend_ir.json`（始终写入）

## 使用示例

```python
from epair_utils import consolidated_weights_save

# 保存完整目录（含权重）
consolidated_weights_save(contents, "/path/to/output")

# 仅保存 EP JSON，跳过权重
consolidated_weights_save(contents, "/path/to/output", save_weights=False)
```

## 与 torch_delegate 的关系

- `EpairContents` 由 `torch_delegate.epair_load()` 返回
- `consolidated_weights_save` 生成的目录可由 C++ 后端（`EpairModelLoader` 的目录模式）直接加载
- `pack_weights` 不依赖 `torch_delegate` 之外的第三方库
