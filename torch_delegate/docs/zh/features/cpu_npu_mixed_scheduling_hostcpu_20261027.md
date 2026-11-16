# TorchDelegate CPU/NPU 混合调度设计方案（HostCpu 后端）

> [!NOTE]文档状态
> 本文为2026-10-27的设计稿，描述 CPU/NPU 混合调度（HostCpu 后端）设计方案。当前使用说明请参考[《Fallback Codegen用户指南》](../guide/fallback_codegen_guide.md)。

## 简介

## 目的

本文档描述 TorchDelegate CPU/NPU 混合调度能力的需求与设计，解决一个问题：

> 当模型中存在没有 NPU 执行路径、只能在 CPU 上执行的算子时，如何让它与 NPU 算子在同一张 GE 图中协同工作，且不引入逐算子的 host↔device 数据往返

## 范围

**包含**：

- 现状分析与关键前提核实
- 端到端设计：IR 扩展、Device 归属判定、代码生成、构图、GE 调度与执行、可诊断性
- 改动文件清单、分阶段实施步骤、DT 设计、验收标准、风险清单

**不包含**：

- GE 内部图优化的实现细节
- torchair converter 的算子覆盖率提升
- TorchDelegate 的离线 OM 部署路径
- 训练场景

## 术语

| 术语 | 含义 |
|------|------|
| converter 节点 | torchair 转换成功、`ascend_ir.source == "converter"` 的节点，映射为 GE 内置算子 |
| fallback 节点 | `ascend_ir.source == ""` 的节点，由 Fallback Codegen 生成 GE 自定义算子，运行时回调 PyTorch Dispatcher |
| Device 后端自定义算子 | 注册 `OpBackend::kDevice`（`REG_AUTO_MAPPING_OP`），`Execute` 回调 PyTorch `PrivateUse1`（NPU）实现 |
| CPU 节点 / HostCpu 自定义算子 | 本特性新增：`ascend_ir.device == "cpu"` 的节点，注册 `OpBackend::kHostCPU`（`REG_OP_BACKEND`），`Execute` 回调 PyTorch `CPU` 实现 |
| `HostCpuExecuteOp` | GE 自定义算子能力接口，`Execute(gert::HostCpuOpExecutionContext*)` 在 host 侧执行，数据可直接通过 `GetData<T>()` 读写 |
| 归属判定 | 导出期为每个 fallback 节点决定用 NPU 还是 CPU 实现执行，并固化到 epair 的过程 |
| 穿越次数 | host↔device 数据搬运（PCIe）次数。本方案的目标是让它取决于 CPU 区段数，而不是 CPU 算子数 |

---

## 总体概述

本节描述影响本方案的一般性因素：项目背景、产品环境、功能概览、设计约束与依赖关系。

## 软件概述

### 项目介绍

**现状**：TorchDelegate 当前只支持全图下沉 NPU。图中的计算节点分为三类：

| 场景 | 描述 | 当前支持 |
|------|------|---------|
| 场景 1 | converter 转换为 GE 内置算子（`source="converter"`） | ✅ |
| 场景 2 | Fallback 回调算子（`source=""`），回调 PyTorch `PrivateUse1`（NPU）实现 | ✅ |
| 场景 3 | CPU 算子（无 NPU 执行路径） | ❌ 本特性交付 |

**问题一：CPU 算子在 IR 上没有表达方式。** Python 侧 converter 转换失败时会创建 mock 节点并写入 `source=""` 的 ascend_ir，因此当前所有计算节点都带 ascend_ir。"这个算子该在 NPU 还是 CPU 执行"这件事，IR 里没有字段记录，必须新增。

**问题二：构图有全图下沉门禁。** GeGraphBuilder::IsFullyOffloaded() 要求所有节点带 ascend_ir，否则放弃构图。CPU 节点需要被纳入同一套构图流程。

**业务驱动**：推荐网络推理链路是典型的 CPU + NPU + CPU 三段式：

```
Stage 1（CPU 前处理）      Stage 2（NPU 核心计算）        Stage 3（CPU 后处理）
特征提取、Embedding 查表 → Embedding 投影、Attention →  结果聚合、排序打分、
序列组装、长度裁剪        Task Tower、CTR/CVR 预估       业务逻辑、响应封装
```

这类模型算子数量多（300~800）、单算子粒度小（1~5us）、Host Bound 明显，对混合调度提出四条要求：

1. CPU 算子在 host 内存上原地执行，不能每个算子一次 PCIe 往返；
2. 连续的 CPU 算子之间数据驻留 host，穿越次数取决于区段数而非算子数；
3. NPU 子图尽量保持完整，GE 的融合、内存复用、多流调度继续生效；
4. host 侧调度开销低，不为小算子引入额外的框架层循环。

**实现路径**：CPU 算子复用现有 Fallback Codegen 机制，自动生成注册到 GE HostCpu 后端的自定义算子，与 NPU 算子同处一张 `ge::Graph`，由 GE 统一调度。框架侧不实现切图与搬运逻辑。

### 产品环境介绍

本特性涉及四个组件的协作关系：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PyTorch / torch_npu（同一进程内）                  │
│  ExportedProgram ──torchair 转换──▶ 逐节点 ascend_ir 标注             │
│  PyTorch Dispatcher：PrivateUse1（NPU 实现） / CPU（libtorch 实现）   │
└───────────────▲──────────────────────────────────┬───────────────────┘
                │ callBoxed（fallback 回调）         │
┌───────────────┴──────────────────────────────────▼───────────────────┐
│                        TorchDelegate（csrc/）                          │
│  Python：归属判定 → epair 序列化（.epair 归档）                        │
│  C++：EpairModelLoader → FallbackCodegen（生成 .so → dlopen 注册）     │
│       GeGraphBuilder（构图：Data/Const/内置算子/自定义算子）           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ GeSession::AddGraph / CompileGraph / LoadGraph
                                │ RunGraphWithStreamAsync
┌───────────────────────────────▼──────────────────────────────────────┐
│                             GE（libge）                               │
│  编译期：引擎选择 → HostcpuEngineUpdatePass → 动静拆分 → 分区 → 融合   │
│          （含常量折叠；HostCpu 节点标记 unknown，走动态执行器）        │
│  运行期：v2 运行时 Lowering → ExecuteGraph（host kernel / device 流）  │
│          host/device 内存放置、搬运点插入、执行顺序由 GE 统一管理      │
└──────────────────────────────────────────────────────────────────────┘
```

对外接口保持不变：

- `EpairModelLoader`对外签名与 ABI 不变；
- `.epair` 归档格式仅新增可选字段；
- GE 侧仅使用 `REG_OP` / `REG_OP_BACKEND` 注册面与 `GeSession` 已有 API，不新增 GE 接口

## 软件功能

本特性需实现六项功能，详细设计见「特性需求分析&设计」章节：

1. **IR 扩展与 Device 归属判定**（导出期，Python）：新增 `device` 字段；自动判定（有 PrivateUse1 实现→NPU，仅 CPU 实现→CPU，两者皆无→报错）；判定结果固化进 epair。
2. **HostCpu 自定义算子代码生成**（构图前，C++）：Fallback Codegen 新增 HostCpu 变体，生成继承 `HostCpuExecuteOp + ShapeInferOp`、注册 `REG_OP_BACKEND(..., kHostCPU)` 的算子并编译加载。
3. **构图与 GE 统一调度**（编译期）：CPU 节点以 host placement 入图；引擎归属、host/device 内存放置、搬运点、执行顺序、多流、常量折叠全部由 GE 负责。
4. **host 侧执行与数据写回**（运行期）：`Execute` 内零拷贝包装 host 张量、调用 PyTorch CPU kernel、直接写回 host 输出内存。
5. **归属可诊断性与严格校验**：逐节点判定记录 + 导出期归属报告 + C++ 加载期复述 + 严格校验开关；不做运行期自动回退。
6. **兼容性**：老 epair 新版本正常工作；不引入本地 CANN 版本/能力校验门禁，环境能力由 GE 编译期自然暴露。

## 设计约束

| # | 约束 | 说明 |
|---|------|------|
| C1 | 平台化（onetrack） | 不引入芯片相关分支。数据搬运、allocator、stream 等底层调用全部在 GE 内部，TorchDelegate 不直接调用新 RTS 接口 |
| C2 | 归属判定无配置项 | 归属完全由自动判定决定，不提供"强制某算子走 CPU/NPU"的配置。唯一新增开关是校验严格度开关（`require_aten_on_npu`），它不改变归属结论 |
| C3 | data-dependent shape 不支持 | `ShapeInferOp` 需在编译期推导输出 shape；输出形状取决于运行时数据值的算子（如 `nonzero`、`unique`）直接报错，不静默降级 |
| C4 | 不做运行期自动回退 | 归属在导出时一次性固化；NPU 算子运行期失败直接报错并带算子名，不允许悄悄改走 CPU |
| C5 | 对外 API/ABI 不变 | `EpairModelLoader` 签名与 ABI 不变；含 CPU 节点的模型走同一条加载/执行路径 |
| C6 | 严格报错 | 对外部输入（epair、EP JSON、未知 device 取值）严格解析，不静默跳过 |
| C7 | ABI 一致性 | 生成的 .so 沿用现有 g++ 命令。GE 样例使用 ABI=0，HostCpu 通路对 ABI 是否有额外约束，在 P0.5 阶段确认 |
| C8 | CPU 算子是 NPU 融合屏障 | HostCpu 自定义算子对 GE 不透明，两侧 NPU 算子无法跨越它融合。通过归属判定尽量收敛 CPU 算子数量来缓解 |
| C9 | 编译期内存复用阶段禁止改图 | 本方案所有构图动作发生在 TorchDelegate 构图期与 GE 编译 pass 内，不触碰内存复用并发阶段 |

## 假设和依赖关系

| # | 依赖 | 说明 |
|---|------|------|
| D1 | GE HostCpu 能力（关键依赖） | 接口、lowering、执行 kernel 均已就绪；配套 CANN 包最低版本号在 P0.5 敲定 |
| D2 | PyTorch Dispatcher CPU kernel | 部署环境需有 libtorch CPU 实现。`torch_delegate_backend` 已链接 `libtorch_cpu`，生成 .so 的链接线已含 `-ltorch_cpu` |
| D3 | torchair `_optimize_and_convert` | 提供逐节点 ascend_ir 标注与 mock 节点路径 |
| D4 | epair manifest v2 | 新增 `cpu_nodes` 统计；`cann_version` 仅作诊断信息记录 |
| D5 | P0.5 接口确认结论 | 见「对子模块的修改」末尾的待确认项清单 |

---

## 特性需求分析&设计

## 整体介绍

**一句话方案**：CPU 算子在导出期被判为 `device="cpu"` 并固化进 epair；构图前为每个 CPU 算子生成一个注册到 GE HostCpu 后端的自定义算子（`HostCpuExecuteOp` + `ShapeInferOp`），其 `Execute` 回调 PyTorch Dispatcher 的 CPU 实现；CPU 算子与 NPU 算子（GE 内置算子 + Device 后端自定义算子）同处一个 `ge::Graph`，一次编译、一次执行，由 GE 统一调度。框架侧不实现任何调度与搬运逻辑。

**改动总览**：

| # | 改动 | 位置 | 量级 |
|---|------|------|------|
| 1 | 导出期 Device 归属判定，结果写入 epair | Python 侧（converter / epair / api） | 小 |
| 2 | Fallback Codegen 增加 HostCpu 变体（含 `ShapeInferOp` 生成） | `csrc/fallback_codegen/` | 中 |
| 3 | 构图支持 host placement 与三分派 | csrc/backend/ | 小 |
| 4 | 归属可诊断性与版本校验 | Python + C++ | 中 |
| 5 | GE 侧 | 无代码修改，仅使用既有能力 | — |

以下「功能需求」逐项说明输入、处理与输出；「对子模块的修改」给出逐文件改动清单

## 功能需求

### 功能需求 1：IR 扩展与 Device 归属判定

#### 1. 介绍

为节点引入显式的执行 Device 归属标记，并在导出期自动完成判定。判定结果固化进 epair，运行期只读不改

#### 2. 输入

- 导出对象：`ExportedProgram`
- 探测对象：每个 fallback 节点对应的算子 target
- 探测依据：PyTorch Dispatcher 注册表
- 配置输入：`EpairConfig`

#### 3. 处理

第一步，`NodeIR` 增加 `device` 字段（`torch_delegate/converter/base.py:8-12`）：

```python
@dataclasses.dataclass
class NodeIR:
    source: str
    ops: list[dict[str, Any]]
    mapping: dict[str, Any] | None = None
    device: str = "npu"                          # ★ 新增："npu" | "cpu"
```

第二步，`EpairConfig` 新增校验严格度开关（不新增归属覆盖类字段，约束 C2）：

```python
require_aten_on_npu: bool = False    # ★ 新增：aten 算子落到 CPU 时报错而非告警
```

第三步，实现自动判定 `_resolve_device`（在 `torchair_converter.py` 中调用）：

```python
def _resolve_device(self, target: torch._ops.OpOverload) -> str:
    op_name = f"{target.namespace()}::{target._schema.name}"
    if torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"):
        return "npu"                              # 有 NPU 实现 → 走 Device 后端 fallback
    if torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "CPU"):
        return "cpu"                              # 仅有 CPU 实现 → 走 HostCpu 后端 fallback
    raise RuntimeError(f"op {op_name} has neither PrivateUse1 nor CPU kernel")
```

挂载方式：`TorchAirGEConverter.convert_graph` 拿到 `ascend_ir_map` 之后，遍历转换后图上的计算节点；对 `ascend_ir_map[node.name].source == ""` 的 mock 节点，用 `node.target` 执行判定，结果写回对应的 `NodeIR.device`。

第四步，判定结果写入 EP JSON。`writer.py` 注入 `ascend_ir` 时增加 `device` 字段（`torch_delegate/epair/writer.py:234-241` 处）：

```json
"ascend_ir": {
  "source": "",
  "device": "cpu",
  "ops": [],
  "mapping": {"inputs": {}, "outputs": {}}
}
```

第五步，manifest 升级（`torch_delegate/epair/schema.py:7-26`）：

```python
version: int = 2          # ★ 1 → 2
cpu_nodes: int = 0        # ★ 新增：CPU 节点计数
```

兼容性：`device` 为可选字段，缺失时按 `"npu"` 处理，等价于现状行为；C++ 侧对 manifest version 1 保持全 NPU 语义。

#### 4. 输出

- 带 `ascend_ir.device` 标注的 `ep_json_with_ascend_ir.json`；
- manifest v2（含 `cpu_nodes`）；
- 归属判定记录与归属报告

### 功能需求 2：HostCpu 自定义算子代码生成

#### 1. 介绍

Fallback Codegen 增加 HostCpu 变体。核心思路是最大化复用：算子原型、schema 解析、参数编解码、编译加载链路全部沿用现有实现，只替换与 Device 相关的三处，并新增一处 shape 推导生成

#### 2. 输入

- epair 归档中的 EP JSON（含 `ascend_ir.device="cpu"` 的节点）
- torch 包路径、输出目录（沿用现有入参）
- 每个 CPU 节点的 PyTorch schema（`findSchemaOrThrow`）
- 每个 CPU 节点全部输出的 `tensor_meta`（`ParsedTensorMeta`，作为 `ShapeInferOp` 的数据源）

#### 3. 处理

先看复用矩阵，了解哪些不动、哪些要改：

| 现有组件 | 位置 | HostCpu 变体处置 |
|---------|------|----------------|
| `InputKind`（21 种）/ `SchemaInfo` / `GetSchemaInfo()` | `fallback_codegen.h:14-67` | 原样复用 |
| `GenerateGeOpReg()` 的原型部分（`REG_OP(...).INPUT/.OUTPUT/.ATTR ... .OP_END_FACTORY_REG` 宏链） | `fallback_codegen.cc:742-863` | 原样复用（GE 样例 `add_custom_ir.h` 同用 `graph/operator_reg.h` 宏链） |
| 算子绑定宏 | `fallback_codegen.cc:1316` | `REG_AUTO_MAPPING_OP(x)` 改为 `REG_OP_BACKEND(x, "x", ge::OpBackend::kHostCPU)` |
| 参数组装逻辑（`InputKind` 分派组 args） | `GenerateExecuteMethod` | 复用（IR-indexed 取参 API 与 HostCpu 上下文同名同语义） |
| `CreateNpuTensor()`（`at::kPrivateUse1` 零拷贝包装） | `fallback_codegen.cc:700-708` | 保留；新增对应的 `CreateCpuTensor()`（`at::kCPU`） |
| `ReadScalarInput()`（device 张量 D2H 读取） | `fallback_codegen.cc:723-737` | HostCpu 版不再需要（host 内存直接 `GetData<T>()` 读取） |
| `NPUStreamGuard` + `getStreamFromExternal` | `fallback_codegen.cc:1166-1167` | HostCpu 版不再需要（host 执行不涉及 device stream） |
| 输出写回（`MallocOutputTensor` + `aclrtMemcpyAsync` D2D） | `GenerateExecuteMethod` 写回段 | HostCpu 版改为直接写 host 输出内存（非连续结果先 `.contiguous()`） |
| inplace 处理（`MakeOutputRefInput`） | `fallback_codegen.cc:1262` | 复用（HostCpu 上下文提供同族接口） |
| `GenerateFromTargets()`（g++ 编译 + dlopen） | `fallback_codegen.cc:1566` | 原样复用（同一 .so 可同时包含两类算子；产物组织方式见待确认项） |
| — | — | 新增 `GenerateShapeInferMethod()` |

生成的 HostCpu 算子类形态如下（示意）：

```cpp
#include "graph/custom_op.h"    // 不再需要 torch_npu 的 from_blob / NPUStream / NPUGuard

namespace ge {
class aten__relu_ final : public HostCpuExecuteOp, public ShapeInferOp {
 public:
  // 替换 1：基类从 EagerExecuteOp 改为 HostCpuExecuteOp + ShapeInferOp
  // 替换 2：上下文类型从 gert::EagerOpExecutionContext 改为 gert::HostCpuOpExecutionContext
  graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    const gert::Tensor *ge_input_0 = ctx->GetRequiredInputTensor(0);   // 取参方式与 NPU 版相同
    std::vector<c10::IValue> args;
    args.emplace_back(CreateCpuTensor(ge_input_0));   // 替换 3：host 指针零拷贝包装
    // 标量参数直接 GetData<T>() 读取；无 stream guard；参数组装逻辑逐字复用
    static const auto op_handle = torch::Dispatcher::singleton().findSchemaOrThrow("aten::relu", "");
    auto results = op_handle.callBoxed(args);         // 调用 PyTorch CPU kernel
    // 写回：MallocOutputTensor 后直接 memcpy 到 host 输出内存
    return GRAPH_SUCCESS;
  }

  // 新增：shape / dtype 推导（HostCpu 后端必须提供）
  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    // 数据来源：EP JSON tensor_meta（生成时内联为常量，或从输入形状透传动态维）
    return GRAPH_SUCCESS;
  }
  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    // 同上：内联 dtype 或按规则透传
    return GRAPH_SUCCESS;
  }
};

REG_OP_BACKEND(aten__relu_, "aten__relu_", ge::OpBackend::kHostCPU);
}  // namespace ge
```

净变化：消除 3 项（标量 D2H、输出 D2D、stream guard）、新增 1 项（`ShapeInferOp` 生成）、改名 3 项（基类、上下文类型、绑定宏）。

#### 4. 输出

- 含 Device 与 HostCpu 两类算子的 `libcust_opapi.so`，`dlopen` 后注册进 GE 自定义算子注册表（`REG_AUTO_MAPPING_OP` → kDevice；`REG_OP_BACKEND` → kHostCPU）。

### 功能需求 3：构图与节点分派

#### 1. 介绍

CPU 节点复用 `FallbackNodePlanner` 入图。该 planner 共 360 行，与 Device 相关的逻辑只有 2 行（张量输入的 placement），因此不新建类，只加一个带默认值的构造参数。

#### 2. 输入

- `ParsedModel`（新增 `ParsedAscendIr.device` / `is_cpu`）
- 构图上下文 `GraphBuildContext` / `GraphPlanningDeps`

#### 3. 处理

第一步，C++ 解析新增字段。`model_types.h` 的 `ParsedAscendIr` 增加 `device`（默认 `"npu"`）与派生字段 `is_cpu`；`parse_ascend_ir.cc` 负责解析，遇到未知取值报 `ValidationIssue`（Error 级）。

第二步，新增节点谓词。在 `graph_build_utils.cc:169-193` 现有谓词旁增加：

```cpp
bool IsCpuNode(const ParsedNode &node);   // ascend_ir.has_value() && ascend_ir->is_cpu
```

第三步，`FallbackNodePlanner` 增加 placement 构造参数（`plan_fallback_node.h:19-21`、`plan_fallback_node.cc:21`）：

```cpp
FallbackNodePlanner(GraphBuildContext &ctx, const GraphPlanningDeps &deps,
                    ge::Placement tensor_placement = ge::kPlacementDevice);  // ★ 新增，带默认值
// 新增成员：ge::Placement tensor_placement_ = ge::kPlacementDevice;
```

第四步，`BuildInputDefs` 中两处 `desc.SetPlacement(ge::kPlacementDevice)`（`plan_fallback_node.cc:196`、`:211`）改用 `tensor_placement_`。标量 Const 输入本来就是 host placement（`:151`），无需改动。输出 placement 是否也要显式设 host，属待确认项（见「对子模块的修改」）。

第五步，`PlanNodes()`（`ge_graph_builder.cc:437-469`）：

```cpp
if (IsConverterNode(node)) {
  // 场景 1：GE 内置算子
} else if (IsCpuNode(node)) {
  // ★ 场景 3（新增）：HostCpu 自定义算子 → fallback 回调 CPU
  TD_CHK_STATUS(FallbackNodePlanner(ctx_, deps, ge::kPlacementHost).Plan(node));
} else {
  // 场景 2：Device 自定义算子 → fallback 回调 PrivateUse1（默认参数，现有行为零变化）
  TD_CHK_STATUS(FallbackNodePlanner(ctx_, deps).Plan(node));
}
```

第六步，`IsFullyOffloaded()`（`ge_graph_builder.cc:401-408`）无需修改：CPU 节点同样携带 ascend_ir（`source=""` + `device="cpu"`），判据天然通过。不新增 Build 前置的能力校验或版本校验门禁（见「功能需求 7：兼容性与能力校验」）。

#### 4. 输出

- 单个 `ge::Graph`，其中 Data/Const、GE 内置算子、Device 后端自定义算子、HostCpu 后端自定义算子同图共存。

### 功能需求 4：GE 统一调度链路

#### 1. 介绍

本需求说明构图产物交给 GE 之后，GE 如何完成从"注册信息"到"host 执行"的全过程。这些全部是 GE 既有机制，本方案不改 GE 代码，是"框架侧不需要任何调度逻辑"的底气所在。

#### 2. 输入

- 已注册 kHostCPU 后端、带 host 执行 lowering 标记的自定义算子
- 带 host placement 张量描述的构图结果

#### 3. 处理

GE 内部链路如下：

| 步骤 | GE 做什么 | 代码位置 |
|------|----------|---------|
| G1 引擎选择 | 注册 kHostCPU 的自定义算子被显式设置到 `DNN_VM_CUSTOM` 引擎并挂 host 执行 lowering func（`host_cpu_custom_op_lower_func`）。不经过任何 Device 引擎，因此"必须走 CPU"由注册方式确定性保证 | `dnnengine_manager.cc:76-89`（`SetHostCpuCustomOp`）、`:357-368`、`:427-434` |
| G2 图 unknown 标记 | 含 host 执行节点的图被标记为 unknown，走 v2 执行器路径 | `mark_graph_unknown_status_pass.cc:22-31,79` |
| G3 动静拆分 | host 执行节点被划入动态侧，影响动静拆分边界（见「特性交叉分析」） | `dynamic_shape_partition.cc:58-66`；`graph_split.md` 规则 6 |
| G4 运行期 Lowering | 命中 host 特化 lowering：用 kOnHost allocator 分配输入/输出，生成 `ExecuteHostCustomOp`（或带 shape 推导的变体）kernel 节点 | `custom_node_converter.cc:62-73,203-248` |
| G5 kernel 执行 | 从注册表取回 kHostCPU 实例，强转 `HostCpuExecuteOp` 并回调 `Execute(HostCpuOpExecutionContext*)` | `custom_op_kernel.cc:98-110,240-252` |
| G6 常量折叠 | 常量输入的 CPU 节点在编译期直接求值并从图中消除，运行时零开销 | `constant_folding` 样例 |

关键结论有三条：

1. 归属性是确定的。kHostCPU 注册的算子走 `DNN_VM_CUSTOM` + host lowering，不参与 Device 引擎选择，也不会被小 shape 传播 pass 改判（该 pass 只作用于 Device 引擎上的算子）。"必须走 CPU"由注册方式保证。
2. 内存由 GE 统一管理。host lowering 使用 kOnHost allocator，相邻 host 节点之间数据驻留 host 内存；host↔device 边界由 GE 插入搬运。穿越次数与 CPU 区段数同量级，而不是与 CPU 算子数同量级。
3. 执行仍是一次调用。对外仍是一次 `RunGraphWithStreamAsync`，TorchDelegate 不实现 runner、不实现 unit 循环。

#### 4. 输出

- 一次编译完成 host/device 内存规划、融合、多流、常量折叠；
- 一次执行调用完成 CPU/NPU 混合图推理。

### 功能需求 5：执行期数据流与内存放置

#### 1. 介绍

说明执行期数据的流转路径与内存归属，用于核对开篇提出的四条目标是否达成。

#### 2. 输入

- v2 Lowering 产生的 ExecuteGraph（host kernel 节点 + device 算子节点）
- 带 placement 的张量描述

#### 3. 处理

| 数据流 | 实现 |
|--------|------|
| host → host（相邻 CPU 算子之间） | host 内存原地传递，无 PCIe |
| host → device（CPU 区 → NPU 区） | GE 在执行图中插入 H2D 搬运 |
| device → host（NPU 区 → CPU 区） | GE 在执行图中插入 D2H 搬运 |
| device → device | 不变（现有 NPU 路径） |
| CPU 算子输出内存 | `MallocOutputTensor` 分配 host 内存，生命周期由 GE 上下文管理，生成代码不显式释放 |
| 标量参数 | 生成为 host Const 节点（`plan_fallback_node.cc:151` 已是 host placement），host 直接读取 |

#### 4. 输出

- host 区段内数据零搬运原地流转；
- host↔device 搬运次数与区段数同量级，由 GE 在边界统一插入。

### 功能需求 6：归属可诊断性与严格校验

#### 1. 介绍

自动判定只看一件事：`PrivateUse1` kernel 是否存在。于是下面两种情况探测结果完全相同，框架无法从探测本身区分：

| 情况 | 含义 | 性质 |
|------|------|------|
| A. 预期走 CPU | 用户本就只为该算子写了 CPU 实现（如纯 host 侧特征处理算子） | 正常 |
| B. 意外走 CPU | 该算子本应在 NPU 执行，但 PrivateUse1 实现缺失、漏注册或注册到错误 overload | 故障，且表现为"能跑通、结果对、只是慢"的静默劣化 |

本需求通过两条互补机制解决：机制一把判定结论与依据完整记录下来，便于事后归因；机制二把可疑归属在导出阶段直接拦成错误，避免事后归因。两者都不改变归属结论本身。

#### 2. 输入

- 逐节点判定结果（节点名、target、探测项、结论）
- `EpairConfig.require_aten_on_npu` 开关
- 导出日志 / 部署侧加载日志

#### 3. 处理

**机制一：维测日志（归属清单 + 判定依据）**

1. 逐节点记录 `DeviceDecision`（导出期内存记录，用于归属报告与严格校验，
   不随 epair 落盘；C++ 加载期按 `ascend_ir.device` 与 target namespace 推断复述）：

```python
@dataclasses.dataclass
class DeviceDecision:
    node_name: str        # FX 节点名
    target: str           # 完整 target，如 "torch.ops.aten.nonzero.default"
    namespace: str        # "aten" | 自定义 namespace
    op_name: str          # 探测用的 dispatcher 名，如 "aten::nonzero"
    overload: str         # 精确 overload（用于识别"注册到错误 overload"）
    device: str           # "npu" | "cpu"
    rule: str             # "converter" | "has_privateuse1" | "cpu_only"
    has_privateuse1: bool
    has_cpu: bool
```

2. 导出结束时打印完整归属清单（NPU 与 CPU 两侧都列全），按 namespace 分级：

| 判定结果 | namespace | 级别 | 理由 |
|---------|-----------|------|------|
| CPU（`cpu_only`） | `aten::` | WARNING | torch_npu 契约应覆盖 aten，缺失属异常，很可能是版本不匹配或覆盖缺口 |
| CPU（`cpu_only`） | 自定义 namespace | INFO | 用户可能本就只写了 CPU 实现，属预期 |
| NPU（`converter` / `has_privateuse1`） | 任意 | INFO | 正常 |

```
[torch_delegate] Device ownership report: 512 compute nodes

── NPU (500) ──────────────────────────────────────────────
  converter -> GE built-in（480）:
    aten::mm.default            aten::add.Tensor        aten::layer_norm.default  ...
  fallback -> PrivateUse1（20）:
    my_ops::rope.default        aten::topk.default      ...

── CPU (12) ───────────────────────────────────────────────
  suspicious: aten op has no PrivateUse1 kernel（3）:
    aten::nonzero.default       has_privateuse1=False  has_cpu=True
    aten::_unique2.default      has_privateuse1=False  has_cpu=True
    aten::bincount.default      has_privateuse1=False  has_cpu=True
  expected: custom namespace with CPU-only impl（9）:
    my_ops::feature_merge.default   has_privateuse1=False  has_cpu=True
    ...
```

3. C++ 加载期复述同一分类（部署侧通常没有 Python 导出日志），只告警不报错。归属已固化在 epair 中，此处报错无助于修复；硬约束前移到导出期。

**机制二：严格校验开关**

- 用法：`EpairConfig(require_aten_on_npu=True)`。
- 语义：aten 命名空间的算子若被判为 CPU，导出阶段直接报错，并一次性列出全部命中算子（含 `op_name`、`overload`）。
- 默认 `False`（仅告警），避免 torch_npu 尚未覆盖的 aten 算子阻塞正常导出。
- 典型用法：CI / 性能基线看护，确保不会因 torch_npu 版本变动或算子漏注册而静默退化到 CPU。
- 自定义 namespace 算子落到 CPU 不受该开关影响（属预期情形）。

**补充约束：不做运行期自动回退**

归属在导出时一次性决定并固化进 epair，运行期只按 `ascend_ir.device` 执行。若允许运行期回退，情况 B 会被彻底掩盖（NPU kernel 抛异常后悄悄改走 CPU，结果正确但变慢且无痕迹），上述两条机制也会失去意义。当前设计下，NPU 算子运行期失败会直接报错并带算子名。

#### 4. 输出

- 归属报告（完整清单 + 分级告警）；
- `require_aten_on_npu=True` 时导出阶段报错与完整命中清单；
- C++ 加载期归属分布日志。

### 功能需求 7：兼容性与能力校验

#### 1. 介绍

保证新老 epair 在版本演进下行为明确：老格式保持原语义；环境能力不在
torch_delegate 侧做防呆校验，由 GE 编译期自然暴露。

#### 2. 输入

- epair 归档（`ascend_ir.device`、manifest `version` / `cann_version` / `cpu_nodes`）

#### 3. 处理

- 老 epair 新版本：`device` 缺失按 `"npu"` 处理；manifest v1 保持全 NPU 语义，零回归。
- 含 CPU 节点的 epair 在能力不足的环境上：torch_delegate **不做**加载期能力校验或
  本地 CANN 版本校验门禁；算子未注册、环境不支持 HostCpu 等能力问题由 GE 编译期
  报错自然暴露（带节点名与算子类型的错误信息），不在此处做防呆处理。
- 最低 CANN 版本 9.2.0 是 torch-delegate 首个支持的商发版本，仅作为文档化要求
  （见 `docs/zh/guide/build.md`）；`manifest.cann_version` 仅作诊断信息记录，
  不参与任何校验。

#### 4. 输出

- 兼容结论：老 epair 全兼容；新 epair 在不支持 HostCpu 的环境由 GE 编译期报错。

## 非功能需求

### 可维护性

- 代码生成最大化复用（见「功能需求 2」的复用矩阵）：HostCpu 变体与 NPU 变体共享参数编解码、重载解析、编译加载链路，避免双份维护；
- 生成的每个 HostCpu 算子 `Execute` 带 DEBUG 级节点名日志；异常路径有明确 ERROR 日志；
- 归属报告带 `rule` / `namespace` 字段，机器可读，支持下游工具过滤。

### 可测试性

- Python 侧：`device` 字段序列化往返、`_resolve_device` 三分支、归属报告分级，均可纯 Python 单测；
- C++ 侧：生成代码文本可断言（含 `REG_OP_BACKEND`、`HostCpuExecuteOp`，不含 `aclrtMemcpyAsync` / `NPUStreamGuard`）；`PlanNodes` 三分派可基于 GE stub 单测；
- 端到端：新增 examples 覆盖纯 CPU、混合、非相邻 CPU 段等形态。

### 可移植性

- 零芯片相关分支（约束 C1）；host 执行路径不依赖 NPU 型号；
- 生成 .so 的 ABI 与链接以 P0.5 结论为准（约束 C7）。

### 可靠性

- 严格报错（约束 C6）：未知 device 取值、无 CPU kernel、data-dependent shape，均在最早可检测阶段报错；
- host 输出内存生命周期遵循 GE 已文档化契约；生成代码不持有跨调用状态；
- 不做静默降级/回退（约束 C4）。

### 平台化要求

GE 是 onetrack 部件，本特性不区分芯片。数据搬运、allocator、stream 等底层调用全部在 GE 内部由既有 HostCpu/RT2 通路承载；TorchDelegate 生成代码不新增 RTS 接口调用点。

### 特性交叉分析

按 `ge/docs/zh/design/cross_feature_check.md` 逐场景分析：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态 Shape | 适用 | 含 CPU 节点的图在 GE 内部被标记 unknown 并走 v2 执行器，不进入 v1 `DavinciModel` 下沉路径；全 NPU 静态图路径零变化。静态 shape 下 CPU 节点的 `ShapeInferOp` 全部内联常量，无动态推导 |
| 动态 Shape | 适用 | CPU 节点 `ShapeInferOp` 按"静态维内联、动态维透传"生成；data-dependent shape 编译期报错（约束 C3）。host 输出复用 GE 已有 host allocator，符合 `rt2_runtime.md` 的既有约束 |
| 动态 Shape 静态子图（图拆分） | 适用（关键交叉） | 遵循 `graph_split.md` 规则 6：HostCpu 节点被划入动态图执行器，影响动静拆分边界。本方案不新增任何切分规则、不写节点属性，相关行为均为 GE 既有实现。v2→v1 边界不新增传递数据。需要在 DT 中验证混合图的拆分边界无回归 |
| 离线场景（atc 编译） | 部分适用 | TorchDelegate 走在线 GeSession 编译加载，本期不交付离线 OM 路径。GE 侧 HostCpu 算子已具备随 ATC 进 OM 的能力（`offline` 样例，需要算子实现 `PortableOp::Serialize/Deserialize`），本方案暂不生成 `PortableOp`，不涉及 OM schema 变更 |
| 在线场景（框架适配） | 适用 | 本特性主战场。epair 导出（S1-S3）→ C++ 加载构图（S4-S7）→ GeSession 编译执行（S8-S14）全链路已覆盖；`EpairModelLoader` API/ABI 不变；不涉及 Adapter 交互变更 |

已加载的约束文档与一致性结论：

- `rt2_runtime.md`：不修改 v2 kernel 实现（host 执行 kernel 为 GE 既有）；生成代码不创建线程、不动态申请内存（输出走 GE allocator 上下文）；不修改外部 API/OM schema；
- `known_shape_runtime.md`：含 CPU 节点的图不走 v1 静态执行器，不触碰 `DavinciModel` 地址刷新/下沉约束；全 NPU 图行为不变；
- `memory-constraints.md`：HostCpu 输出为 host 内存，不占用 HBM 复用块，峰值 HBM 不上升；构图期设置 placement 发生在内存复用并发阶段之前；
- `graph_split.md` / `engine.md`：遵守既有 HostCpu 引擎语义与切分规则，不新增规则、不扩散属性写入。

## 性能

### 模型编译时长

| 项 | 评估 |
|----|------|
| GE 编译流水线 | 单次编译（一张 `ge::Graph`）。HostCpu 相关处理均为一次编译内的既有 pass，不增加流水线遍数 |
| Fallback Codegen | 与现有 fallback 共用一次 g++ 编译。新增 HostCpu 变体代码与 `ShapeInfer` 方法，增量与 CPU 节点数线性相关，预计 <1s 量级，P3 阶段实测 |
| 常量折叠收益 | 常量输入的 CPU 节点编译期消除，运行时零开销 |

### OM 大小和加载占用内存

- CPU 节点的输出与中间张量使用 host 内存，不增加 HBM 峰值（目标：不超过纯 NPU 基线）；
- 生成的 .so 体积增量与 CPU 算子数线性相关，量级小；
- 加载内存由 GE 已有 host allocator 承载，无新增内存池。

### 执行性能

| 指标 | 目标 / 评估 |
|------|------------|
| host↔device 穿越次数 | 与 CPU 区段数同量级（三段式理想情况为 2 次）。P3 需 profiling 确认"CPU 区段内不产生逐算子 PCIe 往返" |
| 单 CPU 算子 host 开销 | Dispatcher boxed 路径约 1us 量级，可忽略（CPU 段本身 host bound）；无 stream guard、无 D2H、无 D2D |
| 端到端时延 | 推荐网络（CPU 段位于头尾）：不高于纯 NPU 基线的 1.05 倍 |
| host/device 重叠 | host 执行能否与 device stream 重叠，属 P4 实测项（见「风险与应对」R1） |
| 编译期常量折叠 | CPU 区段中常量输入的算子零运行时开销 |

## 接口设计

### 新增/修改接口描述

| 接口/字段 | 类型 | 说明 | 兼容性 |
|-----------|------|------|--------|
| `ascend_ir.device` | epair 内部格式（可选字段） | `"npu"`（缺省）\| `"cpu"` | 老 epair 缺失即 NPU 语义，向后兼容 |
| `EpairManifest.version` | Python dataclass | 1 → 2 | C++ 对 v1 保持全 NPU 语义 |
| `EpairManifest.cpu_nodes` | Python dataclass | CPU 节点计数 | 新增字段 |
| `EpairConfig.require_aten_on_npu` | Python dataclass | 严格校验开关，默认 False | 新增字段，不改变归属结论 |
| `NodeIR.device` | Python dataclass | `"npu"` / `"cpu"`，默认 `"npu"` | 新增字段 |
| `FallbackNodePlanner` 构造参数 | C++ 内部类 | 新增 `ge::Placement` 带默认值参数 | 默认值保证现有调用零变化 |
| `EpairModelLoader` 对外接口 | C++ ABI 边界 | 无变化 | — |

### 接口检查项

| **检查项** | **子检查项** | **是否涉及** |
| ---------- | ------------ | :----------- |
| **接口说明** | 接口是否需要评审，评审需关注接口兼容和接口约束（模块间可选，组件间必选） | 是（epair 格式 v2 与 `EpairConfig` 新增字段） |
| **接口说明** | 是否需要补充资料说明 | 是（更新 `docs/zh/guide/fallback_codegen_guide.md`、`docs/zh/features/epair_format_design.md`） |
| **接口说明** | 接口文档中是否明确接口原型、功能、返回值等的说明 | 是（dataclass docstring + epair 格式文档） |
| **接口兼容** | 接口行为兼容性，修改前后行为是否发生变化，是否要通知相关组件 | 是（默认行为不变；`device="cpu"` 为新行为，需通知 torchair/交付团队） |
| **接口兼容** | 接口有无兼容性，新接口在老版本上是否能正常工作，涉及本组件和外组件新增接口 | 是（老 epair 新版本 OK；新 epair 含 CPU 节点在旧 CANN 上加载期明确报错） |
| **接口兼容** | 接口是否涉及使用场景、调用时序等约束 | 是（CPU 节点要求图走 v2 执行器，属 GE 既有约束） |
| **接口约束** | 接口调用不满足约束条件时是否能清晰报错 | 是（见「错误处理」） |
| **接口约束** | 是否需要设计单独的测试用例 | 是（见「DT 设计」） |

## 软件设计

### 关键数据结构

```python
# Python 侧（torch_delegate/converter/base.py）
@dataclasses.dataclass
class NodeIR:
    source: str
    ops: list[dict[str, Any]]
    mapping: dict[str, Any] | None = None
    device: str = "npu"                       # ★ 新增
```

```cpp
// C++ 侧（csrc/backend/model_types.h）
struct ParsedAscendIr {
  std::string source;
  std::string device = "npu";      // ★ 新增
  bool is_cpu = false;             // ★ 派生：device == "cpu"
  // ...（其余不变）
};
```

`DeviceDecision` 结构见「功能需求 6：归属可诊断性与严格校验」。

### 关键技术/算法

**A. 归属自动判定**（见「功能需求 1」）：三条分支，无配置项、无优先级规则，导出期一次性固化。

**B. FallbackCodegen HostCpu 变体**（见「功能需求 2」）：净变化为"消除 3 项、新增 1 项、改名 3 项"，其余全部复用。

**C. `ShapeInferOp` 生成**。数据来源是 EP JSON 中已有的 `tensor_meta`，无需用户额外编写 Meta 实现：

```
对 CPU 节点的每个输出 tensor_name：
  meta ← ParsedModel::FindTensorMeta(tensor_name)
  若 meta.dynamic_dim_symbols 为空（全静态）：
      InferShape    → 内联常量 shape（meta.sizes）
      InferDataType → 内联常量 dtype（meta.dtype）
  否则（含动态维）：
      动态维从 ctx->GetInputShape(i) 透传，静态维内联常量
      若动态维无法从输入推导（data-dependent）→ 报 Error（约束 C3）
```

这些 meta 由 Python 侧导出时经 `register_fake`（Meta key）推导并序列化，C++ 侧 `ParsedModel` 已完整解析。

**D. placement 传递**（见「功能需求 3」）：构造参数 + 默认值；与 `ConstNodeSpec` 既有模式一致（`graph_build_utils.h:31`）。

**E. GE 调度链路**（见「功能需求 4」）：全部为 GE 既有机制。TorchDelegate 只负责两件事：把算子注册到正确的后端（kHostCPU）、把张量描述成正确的 placement（host）。

### 流程设计

**阶段 A · 导出期（Python）**

1. `export_to_epair(ep, config)` 触发 converter 转换图并生成 `ascend_ir_map`；
2. 对所有 mock 节点执行 `_resolve_device(target)`，判定结果写入 `NodeIR.device`；
3. writer 将 `device` 注入每个节点的 `ascend_ir`；manifest 汇总 `cpu_nodes`；
4. 输出归属报告（完整清单 + 分级告警）；若 `require_aten_on_npu=True` 且存在 aten→CPU，导出报错终止。

**阶段 B · 加载与构图期（C++）**

5. `EpairModelLoader` 构造：读归档 → `FallbackCodegen::GenerateFromModelJson`（生成 HostCpu 变体、一次 g++ 编译、dlopen 注册）→ `ParsedModel::Parse` → 权重加载；
6. `GeGraphBuilder::Build()`：归属复述（含 CPU 节点时输出，仅日志）→ `PreparePlaceholders` → `PlanNodes`（三分派）→ `BindDeferredOutputs` → `BindDeferredEdges` → `FinalizeGraph` → `NormalizeGraph` → `GeGraphOptimizer::Optimize`；
7. `CompileAndLoad`：`GeSession::AddGraph` → `CompileGraph` → `LoadGraph`。

**阶段 C · 编译期（GE）**

8. 引擎选择：kHostCPU 自定义算子 → `DNN_VM_CUSTOM` + host lowering；
9. unknown 标记 → 动静拆分（CPU 节点归动态侧）→ 分区 → 融合 / 内存规划；
10. 常量折叠：常量输入的 CPU 节点编译期求值并消除。

**阶段 D · 执行期（GE + PyTorch）**

11. `RunModelWithStreamAsync(stream, inputs, outputs)`（一次调用）；
12. v2 Lowering：host 节点分配 host 内存，插入 host/device 边界搬运，构建 ExecuteGraph；
13. `ExecuteHostCustomOp`（必要时带 shape 推导）回调 `HostCpuExecuteOp::Execute`；
14. `Execute` 内：host 张量零拷贝包装 → `callBoxed` 调用 PyTorch CPU kernel → 直接写回 host 输出。

失败路径：无 CPU kernel、data-dependent shape、后端不可用 → 在最早期报错并携带节点名与原因；运行期 NPU 算子失败直接报错，不回落。

### 对子模块的修改

**1）Python 侧（`torch_delegate/`）**

| 文件 | 改动 |
|------|------|
| `converter/base.py` | `NodeIR` 增加 `device` 字段（默认 `"npu"`） |
| `converter/torchair_converter.py` | 新增 `_resolve_device()`；在 `convert_graph` 中对 `ascend_ir_map` 里 `source==""` 的节点取 `node.target` 执行判定并写回 |
| `epair/writer.py` | 注入 `ascend_ir` 时增加 `device` 字段；汇总 `cpu_nodes` 计数 |
| `epair/schema.py` | `EpairManifest`：`version` 提升为 2、新增 `cpu_nodes`；`DeviceDecision` 结构定义 |
| `epair/config.py` | `EpairConfig` 新增 `require_aten_on_npu: bool = False` |
| `api.py` | manifest 填充（v2 + `cpu_nodes`）；导出末尾打印归属报告；触发严格校验报错 |
| （可选新增）`converter/device_decision.py` | 归属清单渲染与 namespace 分级告警逻辑（若不并入 converter/api） |

**2）C++ 侧（`csrc/`）**

| 文件 | 改动 |
|------|------|
| `backend/model_types.h` | `ParsedAscendIr` 增加 `device` / `is_cpu` |
| `backend/parse_ascend_ir.cc` | 解析 `device` 字段；未知取值报 `ValidationIssue` |
| `backend/graph_build_utils.h` / `.cc` | 新增 `IsCpuNode` 谓词 |
| `backend/plan_fallback_node.h` / `.cc` | 构造参数增加 `ge::Placement`（默认 device）；`:196`、`:211` 两处 `SetPlacement` 改用参数 |
| `backend/ge_graph_builder.cc` | `PlanNodes` 三分派；`Build()` 含 CPU 节点时输出归属复述日志（仅日志，无校验门禁） |
| `fallback_codegen/fallback_codegen.h` / `.cc` | HostCpu 变体：`GenerateExecuteMethod` 按 device 分支、新增 `CreateCpuTensor`、新增 `GenerateShapeInferMethod`、绑定宏换 `REG_OP_BACKEND` |
| `backend/epair_model_loader.cc` | 读取 `extra/epair_manifest.json`（`cann_version` 仅作诊断记录，常量已在 `common/epair_archive_constants.h`） |

**3）测试（`tests/`）**

| 文件 | 改动 |
|------|------|
| `python/test_schema.py` | manifest v2 / `cpu_nodes` 用例（现有 `version == 1` 断言需同步修改） |
| `python/test_writer.py` | `device` 字段注入用例 |
| `python/test_torchair_converter.py` | `_resolve_device` 三分支与判定记录用例 |
| `python/test_api.py` | 端到端导出、归属报告、`require_aten_on_npu` 用例 |
| `cpp/test_ge_graph_builder.cc`（+ helpers） | 三分派、全 NPU 零回归用例 |
| `cpp/test_fallback_codegen.cc` | HostCpu 生成代码内容断言 |
| `cpp/test_epair_model_loader.cc` | manifest 内容一致性用例 |
| `cpp/test_epair_model_parser.cc` | `device` 字段解析用例（按现有用例组织方式扩展） |

**4）示例与文档**

| 文件 | 改动 |
|------|------|
| `examples/host_cpu_mixed_case/`（新增） | 纯 CPU / 混合 / 非相邻 CPU 段端到端示例 |
| `examples/run_all.sh` | 挂接新示例 |
| `docs/zh/guide/fallback_codegen_guide.md` | 三种场景表更新（场景 3 从"暂不支持"改为"支持"） |
| `docs/zh/features/epair_format_design.md` | `device` 字段与 manifest v2 格式说明 |
| `docs/zh/features/overall_design_20260630.md` | 场景 3 支持状态更新 |

**5）GE 侧**

无代码修改。全部依赖既有能力（引擎选择、unknown 标记、动静拆分、v2 host lowering/执行 kernel、常量折叠）。

**P0.5 待确认项（已确认，结论如下）**

1. `.so` 打包与加载方式：**单 `.so` + `dlopen` 可行，不需要 OPP 目录布局**（无需 `ASCEND_CUSTOM_OPP_PATH`）；
2. 构图 API 兼容性：**`REG_OP_BACKEND` 注册的算子可用 `NamedIoNodeBuilder` 构图**；
3. ABI：**HostCpu 通路无额外约束**（沿用 TorchDelegate 现有 g++ 命令与 `_GLIBCXX_USE_CXX11_ABI=1`）；
4. HostCpu 算子输入/输出 placement：**不必须由构图侧声明**（`BuildOutputDefs` 无需设 host；输入 placement 按功能需求 3 显式声明 host，属增强而非必需）；
5. 最低 CANN 版本：**9.2.0**（torch-delegate 首个支持的商发版本，仅作为文档化要求；
   不做本地 CANN 版本/能力校验门禁）。

### 错误处理

#### 系统错误

| 场景 | 处理 |
|------|------|
| epair 解析失败（含未知 `device` 取值） | `ValidationIssue`（Error 级）+ 明确报错，不静默跳过 |
| Fallback Codegen 编译失败（g++ / dlopen） | 现有链路已覆盖：报错并中止构图 |
| GE 编译失败（含 host 节点 lowering / 内存规划） | 透传 GE 错误码 + 上下文日志（节点名、graph_id） |
| 运行期 host kernel 异常 | `Execute` 返回 `GRAPH_FAILED`，GE 执行栈报错；异常路径打 ERROR 日志（含算子名） |

#### 接口错误

| 错误 | 提示 |
|------|------|
| 算子无任何可执行 kernel（PrivateUse1 / CPU / Composite 均无） | 导出期 `RuntimeError`：`op X has neither PrivateUse1, CPU nor CompositeImplicitAutograd kernel` |
| `require_aten_on_npu=True` 且 aten 算子落到 CPU | 导出期 `RuntimeError` + 完整命中清单（`op_name` / `overload`） |
| data-dependent shape 的 CPU 算子 | 编译期 Error + 指引（把该算子移出导出图，放到模型外前后处理） |
| 当前环境不支持 HostCpu（算子未注册） | GE 编译期报错（torch_delegate 不做加载期校验门禁） |
| 部署环境缺 libtorch CPU kernel | codegen 阶段对每个 CPU 节点做 dispatch key 预检（编译期失败）；运行期 `callBoxed` 异常带算子名 |

## 安全检查

### 编码军规

参考《编码军规.md》。本项目附加要求：`-Wall -Wextra -Werror` 零警告、ASAN 构建下测试通过。

### 编码检查项

| **检查项** | **检查项说明** | **是否涉及** |
| ---------- | ------------- | ------------ |
| **是否涉及资源生命周期管理（如线程池、内存池等）** | 生成 .so 生命周期为进程级（沿用现有 fallback 机制，dlopen 后常驻）；HostCpu 输出内存生命周期由 GE 上下文管理（已文档化契约），生成代码不持有资源。不新建线程池/内存池 | 是（沿用既有机制，不新增长生命周期资源） |
| **是否创建新线程** | HostCpu `Execute` 在 GE 执行器线程内同步执行；生成代码不创建线程 | 否 |

## 兼容性检查

| 项 | 结论 |
|----|------|
| 老 epair 新版本 | 兼容：`device` 可选、manifest v1 按全 NPU 语义 |
| 新 epair（含 CPU 节点）旧环境 | 不承诺；GE 编译期报错 |
| 全 NPU 图回归 | 零回归：`FallbackNodePlanner` 默认参数保证现有路径逐字节等价；`IsFullyOffloaded` 判据不变；Device 后端自定义算子行为不变 |
| 对外 ABI | `EpairModelLoader` 不变；CANN ABI 取值以 P0.5 结论为准 |
| 维测口径 | 归属报告为新增输出，不改变既有日志语义 |

## DT 设计

### 测试边界

- 测试入口：Python `export_to_epair`（含 `EpairConfig`）；C++ `EpairModelLoader`（CompileAndLoad / RunModelWithStreamAsync）；生成代码文本（codegen 单测直接断言）
- 测试出口：执行结果逐元素比对（eager CPU / torchair 图模式基线）；manifest/JSON 结构断言；生成代码内容断言；归属报告与日志断言
- 打桩模块：GE stub（`tests/cpp/ge_stub/`）、CANN stub（`tests/cpp/cann_stub/`）、PyTorch Dispatcher（真实环境）

### 测试设计

| **测试类别** | **关键测试项** | **测试方法** | **用例类型** |
| ------------ | -------------- | ------------ | ------------ |
| **功能** | `device` 字段读写往返；`_resolve_device` 三分支；三分派构图；HostCpu 生成代码含 `REG_OP_BACKEND` / `HostCpuExecuteOp` / `InferShape` 且无 `aclrtMemcpyAsync` / `NPUStreamGuard`；纯 CPU 模型端到端；混合模型端到端；非相邻 CPU 段；共享输入 | pytest（`test_schema.py` / `test_writer.py` / `test_torchair_converter.py`）；GTest（`test_fallback_codegen.cc` / `test_ge_graph_builder.cc`）；新增 `examples/host_cpu_mixed_case` | UT、ST、BBIT |
| **性能** | CPU 区段内无逐算子 PCIe 往返；穿越次数=区段数；常量折叠命中数；峰值 HBM；端到端时延 ≤ 纯 NPU 基线×1.05；编译时长 ≤ 基线×1.1 | profiling（msprof）+ 基线对比 | ST |
| **精度** | 混合调度输出与 eager（CPU）/ torchair 图模式（NPU）基线逐元素一致 | `examples/accuracy_compare` | ST |
| **兼容性** | manifest v1 老 epair 加载零回归；全 NPU examples 行为/性能零回归；`_GLIBCXX_USE_CXX11_ABI=0/1` 构建 | 回归套件 + 构造旧 manifest fixture | UT、ST |
| **特性交叉** | 动态 shape 图含 CPU 节点（静态维内联、动态维透传）；动静拆分边界验证（CPU 节点归动态侧、纯 NPU 部分与基线一致）；data-dependent shape 报错；常量折叠路径 | 构造专用 epair + GE stub / 真机 | UT、ST |
| **维测** | 归属报告完整性（双清单、`overload` 字段）；aten→CPU WARNING / 自定义→INFO 分级；注册错误 overload 用例可定位；`require_aten_on_npu=True` 导出期报错；C++ 加载期复述；无运行期自动回退 | pytest + 真机运行观察日志 | UT、ST |

### 测试框架设计

现有框架满足要求：Python pytest + C++ GTest（GE/CANN stub 可替换）+ examples 端到端脚本（`examples/run_all.sh`）。新增内容仅为用例与 fixture（旧 manifest、混合图 epair），无需新增测试框架类。

## 实施步骤

| 阶段 | 内容 | 交付物 | 验证方式 | 依赖 |
|------|------|--------|----------|------|
| **P0** | IR 扩展与归属判定：`NodeIR.device`、manifest v2、`_resolve_device`、C++ 解析、`IsCpuNode` 谓词、归属可诊断性两条机制 | Python + C++ 解析改动 | 单测往返/分支用例；归属报告用例 | 无 |
| **P0.5** | 确认「对子模块的修改」中的五项待确认项；跑通 `host_cpu_add_custom` 三个样例；敲定最低 CANN 版本号 | 接口确认结论 | 三样例 `bash run.sh` 全过 | 无（与 P0 并行） |
| **P1** | FallbackCodegen HostCpu 变体：Execute 分支 + `CreateCpuTensor` + `GenerateShapeInferMethod` + 绑定宏 | `fallback_codegen.{h,cc}` | 生成代码内容断言 + `KEEP_ARTIFACTS=1` 人工审阅 | P0.5 |
| **P2** | `FallbackNodePlanner` placement 参数 + `PlanNodes` 三分派 | `plan_fallback_node.{h,cc}`、`ge_graph_builder.cc` | 全 NPU 图零回归（默认参数）；纯 CPU 模型构图成功 | P0、P1 |
| **P3** | 端到端验证：纯 CPU → 混合 → 推荐网络真实模型 | 新增 `examples/host_cpu_mixed_case` | 精度对齐；`examples/run_all.sh` 全绿 | P2 |
| **P4** | profiling 与调优：穿越次数、常量折叠命中、HBM 峰值、host/device 重叠 | profiling 报告 | 推荐网络端到端 vs 纯 NPU 基线 | P3 |

**里程碑判据**：P0 + P0.5 完成即具备全部信息且 IR 就绪；P2 完成即可构图；P3 完成即具备端到端混合调度能力。

## 实施记录

- P0.5 五项确认结论已回填上方清单；最低 CANN 版本 9.2.0 作为文档化要求，
  不做本地 CANN 版本/能力校验门禁（环境能力由 GE 编译期自然暴露）。
- GE Registered IR 查询统一使用 `GetRegisteredIrDefV2`（CANN 9.2.0 商分版本已
  提供，声明于 `graph/operator_reg.h`），不再经兼容包装调用旧接口。
- 实现落点与本文档「对子模块的修改」清单一致；端到端示例见
  `examples/host_cpu_mixed_case/`（纯 CPU / 混合 / 非相邻 CPU 段 / 类推荐网络四形态）。

### 归属可诊断性与兼容性闭环补充

- **DeviceDecision 导出期记录**：导出侧逐节点记录判定结果
  （node_name/target/namespace/op_name/overload/device/rule/has_privateuse1/has_cpu），
  用于归属报告与严格校验；不随 epair 落盘（避免扩展 `extra/` 目录引入格式
  演进负担），加载期所需信息由 `ascend_ir.device` 与节点 target 承载。
- **C++ 加载期归属复述**：`ReportCpuOwnership` 在 Build 前按
  `ascend_ir.device` 与 target namespace 打印归属分布，aten→CPU 以 WARNING
  复述（可疑），自定义 namespace→CPU 以 INFO 复述（预期），只告警不报错
  （硬约束前移导出期）。
- **生成代码日志**：HostCpu `Execute` 入口打 DEBUG 级节点名日志
  （`aclAppLog(ACL_DEBUG, "HostCpuExecute", ...)`，与 TDLOG 同通道）；异常路径
  try/catch 打 ERROR 日志（含算子名）并返回 GRAPH_FAILED。
- **CPU dispatch key 预检**：codegen 阶段对每个 CPU target 预检
  `Dispatcher::hasComputedKernelForDispatchKey(CPU)`，部署环境缺 libtorch CPU
  kernel 时编译期失败并给出算子名与修复指引。

### 真机性能数据（Ascend910，CANN 9.2.0，torch 2.9.0）

`host_cpu_runner` 内置计时（加载+codegen / GE CompileLoad / 首次推理）与
HBM 占用打印，验收 4 性能项的实测数据（2026-10-27 采集）：

| 模型 | 节点数（CPU） | load+codegen | GE CompileLoad | 首次推理 | HBM 峰值 |
|------|--------------|--------------|----------------|----------|----------|
| model_mixed（NPU-CPU-NPU） | 3（1） | 23.1 s | 5.2 s | 7.3 ms | 416 MB |
| model_pure_cpu | 2（2） | 23.6 s | 0.10 s | 0.7 ms | 415 MB |
| model_non_adjacent | 4（2） | 22.7 s | 0.12 s | 1.3 ms | 415 MB |
| model_recommender（三段式） | 7（2） | 27.0 s | 0.16 s | 1.5 ms | 421 MB |

结论：

- load+codegen 为首次冷编译（g++ 编译生成算子，与 CPU 节点数线性相关，<30 s），
  进程内多模型共享一次 libtorch 头编译时逐模型降至秒级；
- GE CompileLoad：混合图走 v2 执行器（5.2 s 首图，含 host lowering 初始化），
  后续混合图秒级；峰值 HBM 415~421 MB（含 GE/RT 常驻），CPU 输出与中间张量
  走 host 内存，未观察到 HBM 峰值随 CPU 段上升（与「峰值 HBM 不上升」目标一致）；
- host/device 重叠与穿越次数为 P4 项，可用 msprof 对 `host_cpu_runner` 采集
  timeline 验证（脚本 `examples/host_cpu_mixed_case/run.sh` 的执行阶段可直接
  作为 msprof --application 入口）。

## 验收标准

1. **前置确认验收（P0.5）**：`host_cpu_add_custom` 三场景在目标 CANN 版本全部通过；五项待确认项有明确结论并回填本文档；最低 CANN 版本 9.2.0 确定为文档化要求（不引入本地版本/能力校验门禁）。
2. **功能验收**：全 NPU 模型行为与性能零回归；纯 CPU 模型端到端跑通且与 eager CPU 逐元素一致；混合模型（含非相邻 CPU 段、共享输入形态）端到端跑通；推荐网络真实模型全链路成功；data-dependent shape 给出明确 Error 而非崩溃或静默错误。
3. **精度验收**：与 eager（CPU）/ torchair 图模式（NPU）基线相对误差在既有阈值内。
4. **性能验收**：端到端时延 ≤ 纯 NPU 基线×1.05（头尾 CPU 段场景）；CPU 区段内不产生逐算子 PCIe 往返；常量折叠有量化数据；峰值 HBM 不高于基线；编译时长 ≤ 基线×1.1。
5. **接口验收**：`EpairModelLoader` 对外签名与 ABI 不变；`_GLIBCXX_USE_CXX11_ABI=0/1` 下均可用（以 P0.5 结论为准）。
6. **工程验收**：`-Wall -Wextra -Werror` 零警告；ASAN 全量测试通过；`./build.sh`、`./build_test.sh`、`examples/run_all.sh` 全绿。
7. **维测验收**：归属报告完整（双清单 + `overload`）；分级告警正确；注册到错误 overload 的用例可定位；`require_aten_on_npu=True` 导出期报错；C++ 加载期复述；验证无运行期自动回退；每个 HostCpu 算子有 DEBUG 级节点名日志。

## 风险与应对

| # | 风险 | 影响 | 应对 |
|---|------|------|------|
| R1 | host 执行与 device stream 的重叠能力不确定 | CPU 区段成为 device 流水的串行点，抵消多流收益 | P4 profiling 实测；若不重叠，评估 CPU 区段并行化手段 |
| R2 | 产物组织与构图 API 兼容性未验证（待确认项 1、2） | 可能需改造 `GenerateFromTargets()` 或调整打包方式 | P0.5 以最小 demo 实测两条路径；任一不成立则按 GE 样例改用 OPP 目录 / ES 构图，工作量增加但路径明确 |
| R3 | 最低 CANN 版本抬高 | 旧环境无法加载含 CPU 节点的 epair | GE 编译期报错；纯 NPU 模型不受影响 |
| R4 | dtype/format 映射失败（CPU kernel 返回非连续张量等） | 运行期异常深埋在 GE 执行栈 | 写回前统一 `.contiguous()`；失败时打印算子名与 dtype；P3 全量算子覆盖测试 |
| R5 | 可观测性较弱（没有框架层的分段边界） | 问题定位需跨 codegen → 生成 .so → GE 三层 | 生成代码节点名日志 + `DUMP_GE_GRAPH` + profiling |
| R6 | CPU 算子仍是 NPU 融合屏障（约束 C8） | 两侧 NPU 算子无法跨越融合 | 通过归属判定收敛 CPU 算子数量；该限制为混合调度的固有代价 |

经复核不作为风险的三项（已核实排除）：

- "`HostcpuEngineUpdatePass` 可能改判归属"：该 pass 只作用于 Device 引擎上的算子；kHostCPU 注册的自定义算子在引擎选择阶段即确定性落到 host（见「功能需求 4」G1），shape 启发式无从改判；
- "`ShapeInferOp` 无法覆盖 data-dependent shape"：属范围边界，已定义明确报错处置（约束 C3）；
- "HostCpu 输出内存生命周期未知"：属已文档化契约，GE 样例与现有路径均依赖同一契约。

---

## 附录

## 附录 A：设计文档检查结果

- [x] **跨特性交叉影响**：已按 `cross_feature_check.md` 逐场景分析（见「特性交叉分析」）。方案涉及 TorchDelegate（导出/构图/代码生成）与 GE（`compiler/graph/partition/`、`compiler/engines/`、`runtime/v2/engine/custom/`、`graph_metadef/`），已加载 `rt2_runtime.md`、`known_shape_runtime.md`、`memory-constraints.md`、`graph_split.md`、`engine.md`，方案与现有约束一致：
  - 含 CPU 节点的图走 v2 执行器（GE 既有行为），不触碰 v1 `DavinciModel` / 地址刷新约束；
  - HostCpu 输出为 host 内存，峰值 HBM 不上升，不参与 HBM 复用块；
  - 不新增切分规则、不写节点属性，遵守 `graph_split.md` 规则 6 与最小侵入原则；
  - 不修改对外 API / OM schema / v2 kernel 实现。
- [x] **关键特性设计原则与软件约束**：本方案使用 GE 既有 HostCpu 能力，GE 侧零代码修改；TorchDelegate 侧改动局限于 converter / epair / backend / fallback_codegen，无突破已有设计原则项。

## 附录 B：关键代码位置索引

### TorchDelegate 侧

| 主题 | 位置 |
|------|------|
| 全图下沉门禁 | `csrc/backend/ge_graph_builder.cc:401-408`（`IsFullyOffloaded`）、`:581-584`（Build 门禁） |
| Fallback Codegen 触发 | `csrc/backend/ge_graph_builder.cc:356-377` |
| 节点分类谓词（需新增 `IsCpuNode`） | `csrc/backend/graph_build_utils.cc:169-193` |
| 节点分派（需改三分派） | `csrc/backend/ge_graph_builder.cc:437-469`（`PlanNodes`） |
| Fallback 节点构图 | `csrc/backend/plan_fallback_node.cc:21`（构造）、`:140-162`（标量 Const）、`:164-254`（输入 placement `:196`/`:211`）、`:256-283`（输出）、`:350-358`（`Plan` 六步） |
| schema 解析 / InputKind | `csrc/fallback_codegen/fallback_codegen.h:14-67` |
| `CreateNpuTensor`（需对应 `CreateCpuTensor`） | `csrc/fallback_codegen/fallback_codegen.cc:700-708` |
| `ReadScalarInput`（HostCpu 版不需要） | `csrc/fallback_codegen/fallback_codegen.cc:723-737` |
| `GenerateGeOpReg`（原型复用 / 绑定换宏） | `csrc/fallback_codegen/fallback_codegen.cc:742-863`、绑定宏 `:1316` |
| `GenerateExecuteMethod`（Device 分支点） | `csrc/fallback_codegen/fallback_codegen.cc:917-1303`（stream guard `:1166-1167`、写回 helper `:869-885`） |
| g++ 命令 / ABI / 链接线 | `csrc/fallback_codegen/fallback_codegen.cc:1482-1524`（ABI `:1486`，链接 `:1524`） |
| 张量 meta（`ShapeInferOp` 数据源） | `csrc/backend/model_types.h:38-47`（`ParsedTensorMeta`）、`:115-123`（`ParsedAscendIr`） |
| session 编译执行 | `csrc/backend/epair_model_loader.cc:108-173`（Compile/Load）、`:175-206`（Run） |
| 链接依赖 | `csrc/CMakeLists.txt:47`（`libtorch_cpu`）、`:53`（ABI=1） |
| manifest 常量 | `csrc/common/epair_archive_constants.h:25`（`kAnnotatedManifestJson`） |
| Python：NodeIR | `torch_delegate/converter/base.py:8-12` |
| Python：ascend_ir 注入 | `torch_delegate/epair/writer.py:234-241` |
| Python：manifest | `torch_delegate/epair/schema.py:7-26` |
| Python：配置 | `torch_delegate/epair/config.py`（`EpairConfig`） |
| Python：converter | `torch_delegate/converter/torchair_converter.py:105-118` |
| 文档：三种场景表 | `docs/zh/guide/fallback_codegen_guide.md` 三种算子场景表、`docs/zh/features/overall_design_20260630.md` 场景表 |

### GE 侧（本方案依赖，不修改）

| 主题 | 位置 |
|------|------|
| HostCpu 能力接口族 | `inc/graph_metadef/external/graph/custom_op.h:27-30,119-128,168-183,208-218` |
| HostCpu 执行上下文 | `inc/graph_metadef/external/exe_graph/runtime/host_cpu_op_execution_context.h:30-101` |
| 引擎选择（HostCpu 自定义算子 → `DNN_VM_CUSTOM` + host lowering） | `compiler/engines/manager/engine_manager/dnnengine_manager.cc:76-89,357-368,427-434` |
| 小 shape 传播 pass | `compiler/graph/partition/optimizer/hostcpu_engine_update_pass.cc:33,205-215,316-342,472-560` |
| pass 编排 | `compiler/graph/manager/graph_manager.cc:4139-4141`、`compiler/graph/partition/engine_place.cc:235-247` |
| HostCpu 连通分量融合 | `compiler/graph/partition/optimizer/host_cpu_fusion_pass.h:25-26` |
| 图 unknown 标记 | `compiler/graph/passes/shape_optimize/mark_graph_unknown_status_pass.cc:22-31,79` |
| 动静拆分（host 执行节点 → 动态侧） | `compiler/graph/partition/dynamic_shape_partition.cc:58-66` |
| v2 host lowering | `runtime/v2/engine/custom/converter/custom_node_converter.cc:62-73,203-248` |
| v2 host 执行 kernel | `runtime/v2/engine/custom/kernel/custom_op_kernel.cc:98-110,240-263,373-402` |
| 样例（三场景） | `examples/custom_op/host_cpu_add_custom/{constant_folding,host_scheduling,offline}` |
| 架构文档（HostCpu 语义） | `docs/zh/design/features/engine.md:77,338,343`、`docs/zh/design/features/graph_splitter.md:178,208`、`docs/zh/design/constraints/graph_split.md:47` |

## 附录 C：参考文档

**TorchDelegate**

- `docs/zh/features/overall_design_20260630.md` — v1 总体设计（§5 三种算子场景，场景 3 = CPU 算子，本方案交付该场景）
- `docs/zh/features/` 下的 Fallback 特性设计 — Fallback Codegen 特性设计
- `docs/zh/features/aten_ir_callback_codegen_20260715.md` — ATen IR 回调方案（其"不包括 CPU 自定义算子"的范围声明由本方案补齐）
- `docs/zh/guide/fallback_codegen_guide.md` — Fallback Codegen 用户指南
- 《基于 GE 的推荐网络方案》 — 业务场景与三段式架构

**GE / CANN**

- `examples/custom_op/host_cpu_add_custom/README.md` 及三个子场景 README — HostCpu 能力验证样例
- `docs/zh/design/design_document_template.md` — 本文档编写模板
- `docs/zh/design/cross_feature_check.md` — 特性交叉分析指引
- `docs/zh/design/constraints/{rt2_runtime,known_shape_runtime,memory-constraints,graph_split}.md`、`docs/zh/design/features/{engine,graph_splitter}.md` — 设计约束与既有语义
