# Epair支持SymPack优化与AssertTensorMetadata类型

## 简介

### 目的

本文档描述在 C++ Backend 的 GE Graph 构建阶段引入的两项优化：

1. **SymPack 合并优化**：消除 epair 路径中冗余的 SymPack 节点，在 Pack 消除效果上与 torchair `torch.compile` 路径完全一致。
2. **AssertTensorMetadata 节点过滤**：在 `PlanNodes` 阶段过滤 `aten._assert_tensor_metadata` 节点，解决 PyTorch 2.9 `torch.export` 自动插入的元数据校验节点阻断 GE 图转换的问题。

两项优化均在 C++ Backend 内部完成，不修改 Python 侧导出流程，不修改 epair 格式。

### 范围

**包含**：

- `OptimizeGraph` 阶段：`SymPackMergePass`（Pack 替换为合并 CPU Data）+ `RemoveDeadDataAndReorderPass`（死 Data 清理 + index 重排）
- `InputBindingMap`：合并 Data 与用户原始输入的映射，供运行时构造合并 tensor
- `PlanNodes` 阶段的 `_assert_tensor_metadata` 节点过滤
- torchair 与 delegate 路径的流程差异对比

**不包含**：

- torchair 其他 pass 的迁移（与 Pack 消除无关）
- epair 序列化格式变更（绑定信息仅在 C++ 内存中，不持久化）
- Python 侧 converter 修改

---

## 总体概述

### 软件概述

torch-delegate 将 PyTorch 模型通过 `torch.export` + torchair converter 转换为 `.epair` 归档，在 C++ 侧加载并构建 GE Graph 执行。

#### 问题一：SymPack 冗余

模型使用动态 SymInt（如 `x.view([m， n])`）时，torchair `torch.compile` 路径通过 `optimize_sym_pack` 消除所有 Pack 节点，但 epair 路径缺失此优化，导致：冗余 Pack 节点保留、CondGraph 条件子图膨胀、每次推理额外的 Pack kernel 调用。

本设计在 `GeGraphBuilder::Build` 中新增 `OptimizeGraph` 阶段，实现等价的图变换。

#### 问题二：AssertTensorMetadata 噪音节点

PyTorch 2.9 的 `torch.export` 在每个 `aten.to.*` 前插入 `aten._assert_tensor_metadata` 校验节点。该节点 schema 为 `-> ()`（无输出），无计算语义，却因 torchair 无可匹配 converter 而阻断转换。

本设计在 C++ `PlanNodes` 阶段按 `node.target` 过滤该节点，EP JSON 保持原样。

#### 产品环境介绍

```mermaid
flowchart TD
  subgraph Python["Python 导出侧（不修改）"]
    EP["torch.export.ExportedProgram<br/>（含 _assert_tensor_metadata）"]
    TA["torchair _optimize_and_convert"]
    EPAIR[".epair 归档<br/>含 ascend_ir（Pack + mock assert）"]
  end

  subgraph CPP["C++ Backend（本设计修改范围）"]
    BUILDER["GeGraphBuilder::Build"]
    PLAN["PlanNodes<br/>（过滤 _assert_tensor_metadata）"]
    FINALIZE["FinalizeGraph"]
    OPT["OptimizeGraph（新增）<br/>SymPack 合并 + 死节点清理"]
    LOADER["EpairModelLoader<br/>（InputBindingMap 运行时适配）"]
    SESSION["GeSession"]
  end

  EP --> TA --> EPAIR
  EPAIR --> BUILDER
  BUILDER --> PLAN --> FINALIZE --> OPT --> LOADER --> SESSION
```

### 软件功能

| 功能 | 模块 | 描述 |
|------|------|------|
| SymPack 合并 | `GeGraphOptimizer` | 将 `_inputs_all_sym=True` 的 Pack 替换为合并 CPU Data 节点 |
| 死节点清理与 index 重排 | `GeGraphOptimizer` | 删除无引用的原始标量 Data，index 重排为连续序列 |
| 运行时输入适配 | `EpairModelLoader` | 根据 `InputBindingMap` 将用户标量打包为合并 tensor |
| AssertTensorMetadata 过滤 | `PlanNodes` | 按 `node.target` 跳过 `_assert_tensor_metadata`，不创建 GE GNode |

### 设计约束

1. **不改 epair 格式**：绑定信息仅在 C++ 内存中，不持久化；EP JSON 保持原样
2. **不改 Python 侧**：`torchair_converter.py`、`epair/writer.py` 不变
3. **GE API 约束**：使用 `ge::Graph` + `ge::GNode` 公开 API，不依赖内部实现
4. **零开销 no-op**：对不含目标节点的图，两条优化路径均为 no-op
5. **修改最小**：assert_tensor_metadata 仅在 `PlanNodes` 循环增加一个 target 匹配条件

---

## 特性 1：SymPack 合并优化

### 整体介绍

在 `GeGraphBuilder::Build` 的 `FinalizeGraph` 之后新增 `OptimizeGraph` 阶段：

```mermaid
flowchart LR
  A["FinalizeGraph 输出<br/>ge::Graph（含 Pack）"] --> B["SymPackMergePass<br/>Pack → 合并 Data"]
  B --> C["RemoveDeadDataAndReorderPass<br/>清理死 Data + 重排 index"]
  C --> D["优化后的 ge::Graph<br/>+ InputBindingMap"]
```

### 功能需求 1：SymPackMergePass

将图中所有 `_inputs_all_sym=True` 的 Pack 节点替换为合并的 CPU Data 节点。多个 Pack 输入完全相同时共享同一个合并 Data。

**输入**：
- `ge::Graph`（`FinalizeGraph` 输出）
- CPU Data 节点（SymInt 标量，`shape=[]`，`placement=Host`）
- Pack 节点（携带 `_inputs_all_sym=True`，`axis=0`）

**处理**：

```
// 步骤 1：扫描 CPU Data 和 SymPack
for node in graph.GetAllNodes():
    if node is CPU Data:  record index
    elif node is _inputs_all_sym Pack:  collect it

// 步骤 2：按输入组合去重，创建合并 Data
for each Pack:
    if same-input Data already exists:  reuse it
    else:
        create CPU Data(shape=[N], dtype=int64, placement=Host)
        record InputBindingMap: {new_data_name → [source user input indices]}
    delete Pack node

// 步骤 3：重连下游引用
for each node consuming Pack output:
    RemoveEdge(Pack, 0, consumer, port)
    AddDataEdge(merged_data, 0, consumer, port)
```

**输出**：修改后的 `ge::Graph` + `InputBindingMap`

| 变换前 | 变换后 |
|--------|--------|
| Data `m` + Data `n`（CPU 标量） | 删除（pass 2 清理） |
| Pack(m,n) × 4 + Pack(n,m) × 2 | Data_A(shape=[2]) + Data_B(shape=[2]) |
| Reshape(x, Pack) × 6 | Reshape(x, Data_A) × 4 + Reshape(x, Data_B) × 2 |

### 功能需求 2：RemoveDeadDataAndReorderPass

清理被替换后无引用的原始标量 Data（如 `m`、`n`），将剩余 Data 的 index 重排为连续序列。

**reindex 必要性**：torchair `remove_dead_data_and_reorder_data_index` 同样执行 reindex（`graph_pass.py:387-389`），GE 要求 Data index 连续从 0 起。删除死 Data 后若不重排，index 会出现空洞，导致 `AddGraph`/`CompileGraph` 阶段报错或运行时输入错位。

**处理**：

```
// 步骤 1：统计 Data 引用计数
for all nodes:  count references to each Data

// 步骤 2：按图遍历顺序删除死 Data，重排存活 Data index
new_index = 0
for each Data (in graph traversal order):
    if refcount == 0:  RemoveNode
    else:
        old_index = data.GetAttr("index")
        data.SetAttr("index", new_index)
        record: old_index → new_index
        new_index++

// 步骤 3：构建完整 InputBindingMap（按 new_index 顺序）
for each surviving Data (ordered by new_index):
    if Data is merged (created by SymPackMergePass):
        binding = {source_user_indices = pack_input_idx_list}  // size > 1
    else:
        binding = {source_user_indices = [old_index]}          // size == 1, 直接传入
```

**对本模型的效果**（delegate 路径，EP JSON 输入顺序为 `[x， m， n]`）：

| 变换前 | 变换后 |
|--------|--------|
| Data `x` (index=0， NPU) | Data `x` (index=0， NPU) — 直接传入 user inputs[0] |
| Data `m` (index=1， CPU) — 死节点 | 删除 |
| Data `n` (index=2， CPU) — 死节点 | 删除 |
| Data_A (index=3， CPU， 新合并) | Data_A (index=1， CPU) — 打包 user inputs[1，2] |
| Data_B (index=4， CPU， 新合并) | Data_B (index=2， CPU) — 打包 user inputs[1，2]（顺序相反） |

> 注：torchair 路径中 SymInt Data 可能排在 tensor 之前（demo 中 m=0， n=1， x=2），优化后 x 从 idx=2 → 0。delegate 路径按 EP JSON 顺序创建 Data（x=0， m=1， n=2），优化后 x 保持 idx=0。两条路径最终 GE Graph 结构一致，但 reindex 映射不同。delegate 路径中 Data index 初始就等于 user 参数 index（`PreparePlaceholders` 按 `model_.GraphInputs()` 顺序递增 `user_input_index`），简化了 binding map 的构建。

### 功能需求 3：运行时输入适配

SymPack 优化后 GE Graph 的输入结构改变了，但用户传入的输入不变，需运行时适配。

```
优化前：用户输入 [x, m标量, n标量]  →  GE Graph: Data(x, idx=0) Data(m, idx=1) Data(n, idx=2) Pack(m,n)
优化后：用户输入 [x, m标量, n标量]  →  GE Graph: Data(x, idx=0) Data_A(idx=1, shape=[2])
                                                        ↑ 需要的是 tensor([m,n])
```

**`InputBindingMap`** 覆盖所有存活 Data，按 ge_index 顺序存储，等价于 torchair 的 `saved_inputs_info`：

```
InputBindingMap = [
    { source_user_indices = [0] },        // ge_idx=0: x, 直接传入 user inputs[0]
    { source_user_indices = [1, 2] },     // ge_idx=1: Data_A, 打包 user inputs[1,2]
    { source_user_indices = [1, 2] },     // ge_idx=2: Data_B, 打包 user inputs[1,2]
]
```

- `source_user_indices.size() == 1`：直接传入（等价 torchair `_TensorInput`）
- `source_user_indices.size() > 1`：打包为 int64 Host tensor（等价 torchair `_SymPackInput`）
- **空表 = identity**：未触发 SymPack 优化时，loader 直接将 user_inputs 透传给 GE

### `gert::Tensor` move-only 约束

`gert::TensorData` 是 move-only（拷贝构造被 delete），因此 `gert::Tensor` 也是 move-only。不能 `ge_inputs.push_back(user_inputs[idx])` 浅拷贝，需要分场景处理：

| 场景 | 处理方式 | placement |
|------|----------|-----------|
| `InputBindingMap` 为空 | 直接将 `user_inputs` 透传给 `RunGraphWithStreamAsync` | 保持原 placement |
| `size==1`（tensor 输入） | `CloneTensorView`：复制元数据 + `ShareFrom` 共享数据指针 | 保持原 placement |
| `size>1`（标量打包） | `PackScalars`：读 int64 值 → 新建 `vector<int64_t>` → `MakeInt64HostTensor` | `kOnHost` |

### host 输入维度限制

参考 torchair `DynamicNpuGraphExecutor::UpdateHostInput`（`dynamic_npu_graph_executor.cpp:108`）：**GE 只支持 1维或 Scalar 的 Host 输入**。多维 host 输入需要 H2D 拷贝到 device（torchair 通过 `H2DMemcpy` + `host_input_holders_` 实现）。

SymPack 合并后的输入是 1维 `[N]`（`_SymPackInput` 产生 `torch.tensor([idx_list])`，永远是 1维），符合 GE 约束，直接传 `kOnHost` 即可。

> **TODO**：若未来出现多维 host 输入场景，需在 `PackScalars` 后增加 H2D 逻辑（`aclrtMalloc` + `aclrtMemcpy` → `kOnDeviceHbm` tensor），参考 torchair `UpdateHostInput`。当前对多维 host 输入报错处理。

### `EpairModelLoader` 运行时逻辑

```
if InputBindingMap is empty:
    // 无 SymPack 优化：直接透传，零开销
    return session->RunGraphWithStreamAsync(graph_id, stream, user_inputs, outputs)

// 有 SymPack 优化：按 InputBindingMap 构造 ge_inputs
std::vector<gert::Tensor> ge_inputs
std::vector<std::vector<int64_t>> packed_storage  // 局部持有打包数据

for ge_idx in 0..InputBindingMap.size():
    indices = InputBindingMap[ge_idx].source_user_indices
    if indices.size() == 1:
        // 直接传入：ShareFrom 浅拷贝（共享数据指针，不深拷贝）
        ge_inputs.push_back(CloneTensorView(user_inputs[indices[0]]))
    else:
        // 打包标量：读 int64 值 → 新 vector → 新 Host tensor
        packed_storage.emplace_back()
        for idx in indices:
            packed_storage.back().push_back(ReadInt64Scalar(user_inputs[idx]))
        ge_inputs.push_back(MakeInt64HostTensor(packed_storage.back()))

session->RunGraphWithStreamAsync(graph_id, stream, ge_inputs, outputs)
// packed_storage 在函数返回时析构，GE 在调用返回前已完成数据读取
```

等价于 torchair 的 `_GeInputInfo.func.__call__(*user_args)` + `AssembleInputs`。

---

## 特性 2：`_assert_tensor_metadata` 节点过滤

### 整体介绍

在 `PlanNodes` 遍历 EP JSON 节点时，按 `node.target` 跳过 `_assert_tensor_metadata` 节点。该节点 schema 为 `-> ()`（无输出），跳过不影响任何数据依赖。

### 功能需求 4：PlanNodes 阶段过滤

```cpp
static bool IsAssertTensorMetadataNode(const ParsedNode &node) {
    return node.target == "torch.ops.aten._assert_tensor_metadata.default";
}

Status PlanNodes(...) {
    for (const auto &node : parser.Nodes()) {
        if (IsAssertTensorMetadataNode(node)) {
            TDLOGI("PlanNodes skipping _assert_tensor_metadata node: %s", node.name.c_str());
            continue;
        }
        // ... existing node planning logic
    }
}
```

**安全性**：节点 `outputs` 为空（schema `-> ()`），无下游引用，不需要断边/重连。其输入节点（如 `aten.mm`）始终被其他节点（如 `aten.to.dtype`）正常消费。

### 产生机制

`torch.export.export()` 内部执行 `_insert_aten_to_metadata_assert_pass`（`torch/_export/utils.py:659`），在 `aten.to.device` / `aten.to.dtype` / `aten.to.dtype_layout` 节点前插入 `aten._assert_tensor_metadata.default`。任何含 `tensor.to(dtype)` / `tensor.float()` / `tensor.bfloat16()` 的模型均触发。

**FX Graph 示例**（含 2 个 `aten.to.dtype` 的简单模型）：

```
%mm → %_assert_tensor_metadata_default(num_users=0) → %to.dtype → %relu
     → %_assert_tensor_metadata_default_1(num_users=0) → %to.dtype → return
```

每个校验节点 `num_users=0`（无消费者）。

**EP JSON 中**该节点保持原样，torchair try/except 为其创建 mock `ascend_ir`（`source=""`）。C++ `PlanNodes` 按 `target` 过滤，不创建 GE GNode。

---

## 与 torchair 流程对比

### SymPack 对比

| 维度 | torchair `torch.compile` | epair（本设计） |
|------|--------------------------|----------------|
| 图变换位置 | Python 侧 `optimize_graph_without_runtime` | C++ 侧 `GeGraphBuilder::OptimizeGraph` |
| 操作对象 | `GraphDef` protobuf（字段直接读写） | `ge::Graph` + `ge::GNode`（API 调用） |
| 输入绑定 | `_GeInputInfo(func=_SymPackInput(...))` | `InputBindingMap`（C++ `unordered_map`） |
| 运行时构建 | `_SymPackInput.__call__` | `PackScalarInputs` 函数 |
| Pass 范围 | 8 个 pass | 2 个 pass（仅 SymPack 合并 + 死清理） |

这两种路径实现方式不同，但语义等价，最终 GE Graph 结构一致。

### AssertTensorMetadata 对比

| 路径 | 含该节点 | 行为 | 结果 |
|------|----------|------|------|
| torchair `torch.compile`（直接） | 否（dynamo trace 不经过 `torch.export`） | 天然免疫 | 成功 |
| torchair `torch.compile`（基于 EP） | 是 | `get_or_auto_gen_converter` 抛出 RuntimeError | 失败 |
| delegate epair（不过滤） | 是 | mock 节点进入 `PlanFallbackNode` | 失败 |
| delegate epair（过滤后） | 是（EP JSON 不变） | PlanNodes 按 target 跳过 | 成功 |

**核心**：torchair 直接编译路径不经过 `torch.export` 天然免疫；delegate 必须经过，在 C++ 构图阶段过滤即可。

---

## 非功能需求

### 可维护性

- `GeGraphOptimizer` 独立类，各 pass 为静态方法，不侵入现有构图逻辑
- assert_tensor_metadata 过滤为 3 行函数，放在 `PlanNodes` 循环顶部

### 可测试性

- 两个 SymPack pass 可独立单元测试（构造含/不含 Pack 的图）
- assert_tensor_metadata 过滤可构造最小模型（仅 `aten.to.dtype`）验证 GE Graph dump
- 端到端验证：epair dump vs torchair dump 图结构一致

### 可靠性

- Pack 的 `axis` 必须为 0、输入必须为 CPU Data，否则报错
- 边重连失败返回错误码，不静默跳过
- `_assert_tensor_metadata` 无输出，过滤无悬空边风险
- 精确字符串匹配 `node.target`，不误伤其他节点
- **host 输入维度限制**：GE 只支持 1维或 Scalar 的 Host 输入（参考 torchair `UpdateHostInput`）；SymPack 合并后输入为 1维 `[N]`，符合约束；多维 host 输入暂报错处理（TODO: H2D）

### 性能

- `OptimizeGraph`：两轮图遍历，O(N+E)，典型模型微秒级
- assert_tensor_metadata 过滤：每次 O(1) 字符串比较
- OM 体积降低（Pack 删除 > Data 新增；无用校验节点消除）
- 运行时消除 Pack kernel 调用（6→0）和 CondGraph 调度（12→0）
- `InputBindingMap` 为空时零开销（直接透传 user_inputs）
- `size==1` 时 `ShareFrom` 仅复制元数据，无数据拷贝
- `size>1` 时打包标量，数据量极小（int64 标量，通常 N≤10）

---

## 接口设计

本设计不新增对外公开接口。内部变更：

- `ge_graph_optimizer.h/cc`（新增）：定义 `GeGraphOptimizer` 类和 `InputBindingMap`
- `ge_graph_builder.h/cc`：`Build()` 新增 `InputBindingMap &` 出参；`PlanNodes()` 新增过滤
- `epair_model_loader.cc`：`Impl` 新增 `input_binding_map_` 成员，`RunModelWithStreamAsync` 新增输入适配

---

## 软件设计

### 关键数据结构

```cpp
struct GeInputBinding {
    std::vector<int64_t> source_user_indices;  // user 参数 index 列表
    // size==1 → 直接传入（等价 _TensorInput）
    // size>1  → 打包为 int64 Host tensor（等价 _SymPackInput）
};
using InputBindingMap = std::vector<GeInputBinding>;  // 按 ge_index 顺序
```

`InputBindingMap` 覆盖所有存活 Data，按 `ge_index` 顺序存储，等价于 torchair 的 `saved_inputs_info`。空表表示未触发 SymPack 优化（identity）。生命周期随 `EpairModelLoader::Impl`。

> delegate 路径中，Data index 初始等于 user 参数 index（`PreparePlaceholders` 按 `model_.GraphInputs()` 顺序递增），因此非 packed Data 的 `source_user_indices` 为 `[old_index]`（单元素），packed Data 为 `[pack_input_idx_list]`（多元素）。

assert_tensor_metadata 过滤不新增数据结构，直接使用 `ParsedNode::target`。

### 关键技术

#### 1. Pack 输入去重与边重连

通过比较 Pack 的输入 Data 列表判断是否相同组合，命中则复用已创建的合并 Data。边重连使用 `RemoveEdge` + `AddDataEdge` 两步：

```cpp
graph.RemoveEdge(pack_node, 0, consumer, port);
graph.AddDataEdge(merged_data, 0, consumer, port);
```

#### 2. 合并 Data 与运行时打包

```cpp
// 创建合并 Data（shape=[N]，int64，Host）
ge::Operator data_op("Data");
data_op.SetAttr("index", next_index);  // 初始 = graph 中 Data 节点数量, 后续由 reindex pass 重排
ge::TensorDesc desc(ge::Shape({N}), ge::FORMAT_ND, ge::DT_INT64);
desc.SetPlacement(ge::kPlacementHost);
ge::GNode data_node = graph.AddNodeByOp(data_op);
// 记录 pack_input_idx_list 供 RemoveDeadDataAndReorderPass 构建 InputBindingMap

// 运行时：从 user 标量读取 int64 值
static int64_t ReadInt64Scalar(const gert::Tensor &tensor) {
    const void *addr = tensor.GetAddr();
    TD_ASSERT_NOTNULL(addr, "ReadInt64Scalar failed: tensor addr is null");
    return *static_cast<const int64_t *>(addr);
}

// 运行时：打包标量为 1维 Host tensor（shape=[N], kOnHost）
// 注意：GE 只支持 1维或 Scalar 的 Host 输入，多维 host 输入需 H2D（TODO）
static gert::Tensor MakeInt64HostTensor(const std::vector<int64_t> &values) {
    gert::StorageShape shape;
    shape.MutableOriginShape().AppendDim(static_cast<int64_t>(values.size()));
    shape.MutableStorageShape().AppendDim(static_cast<int64_t>(values.size()));

    gert::Tensor tensor;
    tensor.GetShape() = shape;
    tensor.SetDataType(ge::DT_INT64);
    tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});

    // 视图模式：manager=nullptr，数据地址指向 values.data()
    // values 由调用方的 packed_storage 持有，存活到 RunGraphWithStreamAsync 返回
    const size_t byte_size = values.size() * sizeof(int64_t);
    gert::TensorData tensor_data(const_cast<int64_t *>(values.data()), nullptr, byte_size, gert::kOnHost);
    tensor.SetData(std::move(tensor_data));
    return tensor;
}

// 运行时：浅拷贝 tensor（ShareFrom 共享数据指针，不深拷贝）
// 用于 size==1 场景，因 gert::Tensor 是 move-only，不能直接拷贝
static gert::Tensor CloneTensorView(const gert::Tensor &src) {
    gert::Tensor dst;
    dst.GetShape() = src.GetShape();
    dst.SetDataType(src.GetDataType());
    dst.MutableFormat() = src.GetFormat();
    dst.MutableTensorData().ShareFrom(src.GetTensorData());
    return dst;
}
```

#### 3. assert_tensor_metadata 过滤

在 `PlanNodes` 循环顶部按 `node.target` 精确匹配，`continue` 跳过。节点无输出（schema `-> ()`），无需任何边操作。其输入始终有其他消费者（如 `aten.to.dtype`）。

### 流程设计

```mermaid
sequenceDiagram
  autonumber
  participant Builder as GeGraphBuilder
  participant Optimizer as GeGraphOptimizer
  participant Loader as EpairModelLoader::Impl

  Note over Builder: Build() 流水线
  Builder->>Builder: PreparePlaceholders
  Builder->>Builder: PlanNodes（过滤 _assert_tensor_metadata）
  Builder->>Builder: FinalizeGraph

  rect rgb(255,250,240)
    Note over Builder,Optimizer: OptimizeGraph（新增）
    Builder->>Optimizer: Optimize(graph, out_bindings)
    Note over Optimizer: SymPackMergePass: Pack → 合并 Data
    Note over Optimizer: RemoveDeadDataAndReorderPass: 清理 + 重排
    Optimizer-->>Builder: SUCCESS + InputBindingMap
  end

  Builder-->>Loader: graph + input_binding_map
```

### 对子模块的修改

| 模块 | 修改 |
|------|------|
| `csrc/backend/ge_graph_optimizer.h/cc` | 新增：`GeGraphOptimizer` 类 + `InputBindingMap` |
| `csrc/backend/ge_graph_builder.cc` | `PlanNodes()` 新增过滤；`Build()` 新增 `OptimizeGraph` 调用 |
| `csrc/backend/ge_graph_builder.h` | `Build()` 签名新增出参 |
| `csrc/backend/epair_model_loader.cc` | 新增 `input_binding_map_` + 输入适配逻辑 |
| `csrc/CMakeLists.txt` | 新增 `ge_graph_optimizer.cc` 编译目标 |

### 错误处理

| 错误 | Status |
|------|--------|
| `AddNodeByOp` / `RemoveNode` / 边操作失败 | `FAILED` + ERROR 日志 |
| Pack 输入非 CPU Data / axis≠0 | `INVALID_PARAM` + ERROR 日志 |
| 图中无目标节点 | `SUCCESS`（no-op，零开销） |
| `_assert_tensor_metadata` outputs 非空 | 不会发生（schema 保证） |
| host 输入维度 > 1 | `UNIMPLEMENTED` + ERROR 日志（TODO: 需 H2D，参考 torchair `UpdateHostInput`） |

---

## 安全检查

| 检查项 | 说明 |
|--------|------|
| 资源生命周期 | `InputBindingMap` 随 `EpairModelLoader::Impl` 管理；`packed_storage` 局部持有，函数返回时析构；过滤无资源分配 |
| 数据生命周期 | `ShareFrom` 共享 user_inputs 数据指针，user_inputs 须存活到 `RunGraphWithStreamAsync` 返回；`MakeInt64HostTensor` 数据由 `packed_storage` 持有 |
| 线程 | 均为同步执行 |

---

## 兼容性检查

| 场景 | 结果 |
|------|------|
| 老 epair 在新代码上运行 | no-op 或正常过滤，行为不变 |
| 新 epair 在老代码上运行 | 无优化但可通过（EP JSON 不变） |
| epair 格式 / 公开 API | 不修改 |

---

## DT 设计

### 测试设计

| 类别 | 特性 | 测试项 | 方法 |
|------|------|--------|------|
| 功能 | SymPack | 无 Pack → no-op | 构造简单图，验证 Optimize 无变化 |
| 功能 | SymPack | Pack 被替换 | 构造含 `_inputs_all_sym` Pack 的图，验证 Pack=0 + Data 创建 |
| 功能 | SymPack | 冗余 Pack 去重 | 4 个相同输入 Pack → 仅 1 个合并 Data |
| 功能 | SymPack | 死 Data 清理 + index 重排 | 验证死 Data 删除，剩余 index 连续 |
| 功能 | SymPack | InputBindingMap 正确性 | 验证 source_indices 与 Pack 输入对应 |
| 功能 | SymPack | 边重连正确性 | Reshape 输入从 Pack → 合并 Data |
| 功能 | Assert | EP JSON 含校验节点 | `torch.export` 后检查 JSON |
| 功能 | Assert | C++ GE Graph 不含校验节点 | dump GE Graph 验证 |
| 功能 | Assert | 无 `aten.to` → no-op | 过滤不匹配任何节点 |
| 功能 | Assert | 过滤后输出一致 | C++ vs Python 参考 `max‖diff‖=0` |
| 端到端 | 两者 | 精度 | C++ vs Python/torchair `max_err < 1e-5` |
| 兼容 | 两者 | 纯 tensor 模型 | 行为不变 |

### 测试框架

UT 用 GoogleTest，ST 用现有 `pack_pass` 示例项目。

---

## 验收标准

1. **SymPack**：Pack 节点数 = 0，CondGraph = 0，输出 max_err < 1e-5
2. **AssertTensorMetadata**：EP JSON 含节点，GE Graph 不含，输出 max|diff| = 0
3. **编译性能**：编译时长增加 < 5%
4. **兼容**：老 epair 正常加载；纯 tensor 模型行为不变；EP JSON 不变

---

## 附录

### 附录 A：`_assert_tensor_metadata` schema

```
aten::_assert_tensor_metadata(
    Tensor a, SymInt[]? size=None, SymInt[]? stride=None,
    ScalarType? dtype=None, *, Device? device=None, Layout? layout=None
) -> ()
```

- **输出**：无（`-> ()`）
- **语义**：运行时校验 tensor 元数据是否与导出时一致
- **触发**：图中存在 `aten.to.device` / `aten.to.dtype` / `aten.to.dtype_layout`

### 附录 B：相关 PyTorch 内部代码位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `torch/_export/utils.py` | 659 | `_insert_aten_to_metadata_assert_pass` 定义 |
| `torch/_export/utils.py` | 735 | pass 调用点 |
| `torch/_subclasses/functional_tensor.py` | 263 | `FunctionalTensor.to()` 另一插入路径 |

### 附录 C：最小验证示例

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))
    def forward(self, x):
        y = torch.mm(x, self.weight)
        y = y.to(torch.float32)     # → aten.to.dtype → _assert_tensor_metadata
        y = torch.relu(y)
        return y.to(torch.bfloat16)  # → aten.to.dtype → _assert_tensor_metadata
```

仅靠标准 `aten.to.dtype` 触发，无需自定义算子，无需 fallback codegen。
