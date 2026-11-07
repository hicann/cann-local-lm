# TorchDelegate整体架构

## 项目背景与定位

### 问题陈述

将基于 `torch.export` 的模型部署到NPU（如Ascend）时，存在两种典型方案：

1. **AOTInductor（AOTI）**：在Python环境中完成所有编译，输出 `.so`。优点是运行时无编译开销；缺点是编译绑定Inductor，NPU backend难以替换，且需要在有Python/Torch环境的机器上完成编译。

2. **torch.export +延迟编译**：仅导出计算图，在目标设备上再编译。优点是部署灵活；但标准PT2格式不携带backend IR信息，目标机器需要重新做IR转换，效率低。

### TorchDelegate的定位

TorchDelegate是一个**中间路径**：

- 在Python侧完成IR转换（aten op → Ascend IR），但**不做切图，不做编译**
- 将IR信息内联到EP JSON，连同权重打包为 `.epair`
- C++侧加载 `.epair`，在目标设备上完成切图 → Codegen → 编译 → 部署

与AOTI的对比：

| 维度 | AOTI | TorchDelegate IR Archive |
|------|------|--------------------------|
| Python侧产物 | `.so`（完整编译产物） | `.epair`（IR +权重） |
| 切图时机 | Python侧 | C++侧（延迟） |
| 编译时机 | Python侧 | C++侧（延迟） |
| Backend绑定 | Inductor绑定 | 可插拔（通过 `NodeIRConverter`） |
| 部署环境要求 | 需与编译环境一致 | 灵活，C++侧决定 |

---

## 四大模块概览

```text
┌────────────────────────────────────────────────────────┐
│  模块 1（Python）：ep → epair                           │
│  ExportedProgram → (节点级 IR 转换) → .epair 归档       │
└─────────────────────────┬──────────────────────────────┘
                          │ .epair 文件
                          ▼
┌────────────────────────────────────────────────────────┐
│  模块 2（C++）：图构建（GeGraphBuilder）                 │
│  解析 EP JSON，根据 ascend_ir.source 字段区分             │
│  converter 节点（直接构建 GE 算子）和 fallback 节点       │
├────────────────────────────────────────────────────────┤
│  模块 3（C++）：编译后端（Backend）                     │
│  IUnitEmitter：emit 时随路完成编译，                    │
│  将编译产物写入输出 epair archive                       │
├────────────────────────────────────────────────────────┤
│  模块 4（C++）：异构代码生成（HeteroCodegen）           │
│  驱动 Emitter 生成 Runner 源码 → runner.so              │
│  最终产出：可部署 epair（权重 + NPU 模型 + Runner .so） │
└────────────────────────────────────────────────────────┘
```

| 模块 | 名称 | 语言 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | EP → EPair序列化 | Python | `ExportedProgram` | `.epair` 文件（IR +权重） |
| 2 | 切图（EpairPartitioner） | C++ | `.epair` | `PartitionedGraph`（NPU/CPU分区，带拓扑顺序） |
| 3 | 编译后端（Backend） | C++ | `ExecutionUnit`（`NpuSubgraph` / `CpuSegment`） | Emitter实现+编译产物写入archive |
| 4 | 异构代码生成（HeteroCodegen） | C++ | `PartitionedGraph` + `IUnitEmitter` 实例 | Runner `.so` +可部署epair |

---

## 模块间数据流

```text
ExportedProgram
      │
      │ (1) torch._export.serde.serialize(ep)
      ▼
  EP JSON bytes
      │
      │ (2) 遍历 nodes，调用 NodeIRConverter.convert_node()
      ▼
  EP JSON + ascend_ir 扩展字段
      │
      │ (3) PT2ArchiveWriter 打包权重 + manifest
      ▼
  .epair 文件（输入 epair）
      │
      │ (4) EpairPartitioner::partition()
      │     EpairArchiveReader 读取 model.json + weights
      ▼
  PartitionedGraph（ExecutionUnit 列表，带拓扑顺序）
      │
      │ (5) HeteroCodegen::emit(graph, out_archive)
      │     ├─ CpuSegment  → ICpuUnitEmitter::emit()
      │     │    → 生成内联 at::_ops 调用代码
      │     └─ NpuSubgraph → IDeviceUnitEmitter::emit()
      │          → 触发 GE 编译（GeGraphBuilder + aclgrphBuildModel 或 GeSession）
      │          → ge_aot：将 .om 写入 out_archive
      │          → 生成对应执行代码（aot: 从 archive 加载执行；jit: 通过 Session 执行）
      │     SyncEmitter：每个 Device Emitter 之后插入 aclrtSynchronizeStream(g_stream)
      │
      │ (6) HeteroCodegen::compile_and_package(out_archive)
      │     系统编译器编译 runner.cpp → runner.so，写入 out_archive
      ▼
  输出 .epair（可部署：原始权重 + 编译后 NPU 模型 + Runner .so）
      │
      │ (7) load_runner(epair_path)
      ▼
  RunnerFn → run(inputs) → outputs
```

---

## epair格式总览

epair复用 **PT2 Archive格式**，文件结构与PT2完全相同，在 `extra/` 目录下新增两个文件（完整字段说明详见 [epair.md](epair_format_design.md)）：

```text
model.epair（ZIP 结构）
├── extra/                                 ← epair 专有，新增
│   ├── epair_manifest.json                ← 转换元数据（版本、backend、统计信息）
│   └── ep_json_with_ascend_ir.json        ← 带 ascend_ir 字段的 EP JSON
├── models/
│   └── model.json                         ← 标准 EP JSON
└── data/
    ├── weights/
    │   ├── weight_0, weight_1 ...
    │   └── model_weights_config.json
    └── constants/
        ├── tensor_0, tensor_1 ...
        └── model_constants_config.json
```

可部署epair在输入epair基础上额外包含：

```text
model_compiled.epair（ZIP 结构）
├── ...（继承输入 epair 全部内容）
├── runner.so                              ← HeteroCodegen 编译产物
└── npu/
    ├── subgraph_0.om                      ← ge_aot 模式下的 NPU 编译产物
    └── subgraph_1.om
```

### 唯一的格式扩展

`models/model.json` 中每个node对象可新增可选字段 `ascend_ir`：

```json
{
  "target": "aten::mm.default",
  "inputs": [...],
  "outputs": [...],
  "metadata": {},
  "ascend_ir": {
    "source": "converter",
    "ops": [
      {
        "name": "MatMul",
        "type": "MatMul",
        "input": ["x", "weight"],
        "attr": {"transpose_x1": {"b": false}, "transpose_x2": {"b": false}},
        "inputDesc": [{"name": "x1", "dtype": "DT_FLOAT", "shape": {"dim": ["3", "3"]}, "layout": "ND"}],
        "outputDesc": [{"name": "y", "dtype": "DT_FLOAT", "shape": {"dim": ["3", "3"]}, "layout": "ND"}]
      }
    ],
    "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}}
  }
}
```

- `source` 字段标识转换来源：`"converter"` 表示由Python converter转换，空字符串表示fallback节点
- 无 `ascend_ir` 字段或 `source` 为空的node：C++侧作为fallback节点，由 `FallbackCodegen` 生成GE自定义算子
- 有 `ascend_ir` 且 `source` 为 `"converter"` 的node：C++侧直接构建GE算子节点

`epair_manifest.json` 包含 `version`、`backend_name`、`total_nodes`、`converted_nodes` 等字段。

---

## 关键设计原则

### 严格报错原则

对所有外部输入（`.epair` 文件内容、EP JSON结构、字段类型）采用严格解析策略：遇到不认识的字段类型、缺失必要字段、类型不符，一律报Error级别的 `ValidationIssue`，不静默跳过、不猜测意图。

### 组件规模原则

单个组件实现代码控制在50行以内时，优先聚合而非拆分；实现复杂度上升后再重构拆分。

### 延迟切图

切图在C++侧完成，Python侧仅做IR标注。这使得：

- `.epair` 文件可在不同硬件配置下复用（切图策略可在C++侧调整）
- Python侧实现保持简单，不需要感知图的连通性和子图边界

### Codegen而非解释执行

C++侧采用codegen模型（对齐AOTI）：

- `HeteroCodegen` 驱动各 `IUnitEmitter` 生成Runner C++源码
- 源码编译为 `runner.so`，部署侧通过 `load_runner` 加载执行
- 不在运行时解释图结构，执行路径固定、可预测

### 格式克制扩展

对PT2格式仅做最小化扩展：

- 不新增顶层目录，不改变weights/constants存储方式
- 唯一新增文件：`extra/epair_manifest.json`
- 唯一格式修改：node JSON对象上的可选 `ascend_ir` 字段
- 这确保 `.epair` 可被标准PT2工具部分读取（忽略 `ascend_ir` 字段即可）

### 后端可替换

`EmitterFactory` 根据 `CompileOptions.backend` 字符串决定 `IDeviceUnitEmitter` 实现（`ge_aot` 或 `ge_jit`），当前默认为 `ge_aot`。接入新backend只需实现新的 `IDeviceUnitEmitter`，不影响上下游接口。

### 节点级IR，而非子图级

IR转换粒度为单个节点（node），而非子图。这带来：

- 更简单的转换接口（`convert_node` 只看单个node）
- 子图边界由C++侧根据 `ascend_ir` 字段的连续性决定
- Python侧不需要理解图的拓扑结构

---

## 依赖复用原则

本项目最大化复用PyTorch和CANN的已有能力，不自己造轮子。典型复用点：

- **序列化**：ZIP读取复用 `caffe2::serialize::PyTorchStreamReader`，JSON解析复用nlohmann/json
- **CPU算子调用**：生成 `at::_ops::op_name::call(...)` 静态分发，对齐AOTI `CppWrapperCpu` 模式
- **设备同步**：复用CANN runtime的 `aclrtSynchronizeStream` 通用接口

---

## 阶段依赖概览

本项目按阶段划分依赖边界，体现「前端重、后端轻」的架构特征：

| 阶段 | 对外依赖 | 说明 |
|------|---------|------|
| IR转换（Python侧） | PyTorch、torchair等Python软件栈 | 依赖最重，需完整Python环境 |
| 编译（C++侧） | 图编译器（GE）、系统C++编译器 | 需编译工具链，无需Python |
| 执行（部署侧） | ACL runtime | 依赖最轻，仅需runtime库 |

从IR转换到执行，依赖链逐阶段收窄。最终部署产物（`runner.so` + `.epair`）只需ACL runtime即可运行，无需Python环境或图编译器。各模块的具体依赖约束详见各模块文档。

---

## 用户接口概述

本项目对用户暴露三个入口：

- **Python侧**：调用 `export_to_epair(ep)` + `epair_save(contents, path)` 完成IR转换，得到携带Ascend IR标注的 `.epair` 文件
- **C++编译侧**：调用 `EpairPartitioner::partition` 切图，再通过 `HeteroCodegen` 完成codegen和GE编译，得到可部署的 `.epair`
- **C++部署侧**：调用 `load_runner(epair_path)` 加载Runner，返回可直接调用的推理函数
