# EpairModelLoader同步执行接口（RunModel）

> [!NOTE]文档状态
> 本文为2026-09-24的设计稿，描述EpairModelLoader同步执行接口。当前接口说明请参考[《EpairModelLoader API》](../api/epair_model_loader.md)。

## 简介

### 目的

为 `td::EpairModelLoader` 新增同步执行接口 `RunModel`，封装GE的 `ge::GeSession::RunGraph`。接口阻塞直到图执行完成，返回时输出数据已就绪，用户无需再手动调用 `aclrtSynchronizeStream` 同步流，降低C++部署侧的接入门槛。

### 范围

| 文件 | 变更内容 |
|------|----------|
| `csrc/inc/external/torch_delegate/epair_model_loader.h` | 新增 `RunModel` 接口声明（ABI边界，仅增量） |
| `csrc/backend/epair_model_loader.cc` | 新增 `Impl::RunModel` 实现与外层转发 |
| `tests/cpp/ge_stub/ge_stub.cc` | 补充 `ge::GeSession::RunGraph` 桩实现 |
| `tests/cpp/test_epair_model_loader.cc` | 新增同步接口UT用例 |
| `docs/zh/api/epair_model_loader.md` | 新增RunModel接口说明章节 |
| `examples/run_model_multi_instance` | 多线程多实例端到端示例（生产者-消费者4实例4线程调用 `RunModel`） |

---

## 总体概述

### 软件概述

当前 `EpairModelLoader` 对外仅提供异步执行接口 `RunModelWithStreamAsync`（对应 `ge::GeSession::RunGraphWithStreamAsync`），调用返回后输出尚未就绪，用户必须自行对stream调用 `aclrtSynchronizeStream` 完成同步后才能读取输出。GE层已提供同步接口 `RunGraph`（阻塞直到图执行完成），但td层此前未封装。本特性补齐同步执行语义，与GE API形成一一对应：

| GE接口 | 语义 | td封装接口 |
|---------|------|-------------|
| `RunGraph` | 同步（阻塞直到图执行完成） | `RunModel`（本特性新增） |
| `RunGraphWithStreamAsync` | 异步（指定stream） | `RunModelWithStreamAsync`（已有） |

### 软件功能

```cpp
Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);
```

- 同步运行模型，阻塞直到图执行完成后返回，返回后可直接读取 `outputs`
- 输入张量数据可位于Host侧或Device侧（Host输入进入数据队列时会产生一次内存申请与拷贝开销）
- 输出张量调用方无需分配内存，执行完成后由GE自动分配Host内存赋值返回（对齐GE `RunGraph` 能力）

### 设计约束

- **ABI边界**：`epair_model_loader.h` 为对外ABI边界，`RunModel` 仅使用已暴露类型（`std::vector<gert::Tensor>`、`Status`），且为新增非虚成员函数（PIMPL类），不破坏既有ABI
- **前置状态检查**：与异步接口一致——`init_status_ != SUCCESS` 时透传初始化错误码；模型未加载（`CompileAndLoad` 未调用或失败）时返回 `FAILED` 并打印ERROR日志
- **SymPack输入适配复用**：`input_binding_map_` 非空时，复用 `BuildGeInputs` 完成用户输入到GE输入的绑定/打包，行为与异步接口完全一致
- **无stream参数**：对齐GE `RunGraph` 语义，不引入额外stream参数与同步组合逻辑，避免过度设计
- **互斥约束**：对齐GE `RunGraph`，`RunModel` 与 `RunModelWithStreamAsync` 互斥，同一模型不可混用同步与异步接口执行。GE `GeSession` 内部有执行模式检查（`CheckRunMode`），但其报错不易定位；为此 `EpairModelLoader` 层增加防呆校验——`Impl` 内记录首次成功执行的接口模式（同步/异步），此后另一接口的调用在进入GE前即被拦截，返回 `FAILED` 并打印明确日志提示两接口互斥，并为其补充UT用例
- **并发约束**：本接口封装GE同步 `RunGraph`，多实例并发行为受GE该接口约束；同一实例不支持多线程并发调用

---

## 特性1 RunModel同步执行接口

### 整体介绍

`RunModel` 在 `EpairModelLoader::Impl` 中实现，执行路径与 `RunModelWithStreamAsync` 保持一致，仅将底层GE调用替换为同步的 `RunGraph`：

```text
RunModel(inputs, outputs)
    │ 检查 init_status_ / loaded_flag_ / run_mode_（与异步接口互斥，混用直接报错拒绝）
    ▼
input_binding_map_ 为空？
    ├─ 是 → session_->RunGraph(graph_id_, inputs, outputs)
    └─ 否 → BuildGeInputs() 构造 ge_inputs
            → session_->RunGraph(graph_id_, ge_inputs, outputs)
    ▼
ge::Status → td::Status 映射（失败打印 TDLOGE）
```

### 功能需求1同步执行

- `CompileAndLoad` 成功后调用 `RunModel`，接口阻塞直到图执行完成，返回 `SUCCESS` 且 `outputs` 已填充
- `outputs` 调用方无需预分配内存，执行完成后由GE自动分配Host内存赋值返回
- 输入数据可位于Host侧或Device侧；位于Host侧而图在Device上执行时，每个输入进入数据队列会产生一次内存申请与拷贝，输入较大或较多时可能存在性能瓶颈；Device侧输入地址须32字节对齐

### 功能需求2错误处理

| 场景 | 返回值 | 行为 |
|------|--------|------|
| 构造失败（归档解析 / Graph构建失败） | 透传 `init_status_`（`INVALID_PARAM` / `FAILED`） | 与既有接口一致 |
| 模型未加载 | `FAILED` | `TDLOGE` 打印 "model is not loaded" |
| `BuildGeInputs` 输入索引越界 | `INVALID_PARAM` | `TDLOGE` 打印越界索引 |
| `ge::GeSession::RunGraph` 失败 | `ToTdStatus(ge_status)` | `TDLOGE` 打印ge_status与graph_id |

---

## 非功能需求

- **可维护性**：`RunModel` 与 `RunModelWithStreamAsync` 共享状态检查、输入适配（`BuildGeInputs`）和错误映射（`ToTdStatus`）逻辑，无新增抽象层
- **可测试性**：GE stub补充 `RunGraph` 桩实现，复用既有 `SetRunGraphStatus` 注错入口
- **兼容性**：仅新增接口，不修改 `.epair` 格式、既有接口签名与行为；老用户代码无需任何改动

---

## DT设计

### 测试边界

- **测试入口**：C++侧 `td::EpairModelLoader::RunModel`
- **测试出口**：返回状态码、`outputs` 填充结果、ERROR日志
- **打桩模块**：GE stub（`libge_runner_v2` 桩，`tests/cpp/ge_stub/ge_stub.cc`），通过 `GeApiStub::SetRunGraphStatus` 注入执行失败

### 测试设计

| 类别 | 测试项 | 方法 | 用例类型 |
|------|--------|------|----------|
| 功能 | 未加载拒绝执行 | 构造loader后直接 `RunModel` → 返回 `FAILED` | UT |
| 功能 | 同步执行成功 | `CompileAndLoad` → `RunModel` → 返回 `SUCCESS` 且 `outputs` 非空 | UT |
| 功能 | 执行失败传播 | `SetRunGraphStatus(ge::FAILED)` 后执行 → 返回 `FAILED` | UT |
| 功能 | 与异步接口互斥（防呆） | 同一实例先 `RunModelWithStreamAsync` 成功后再调 `RunModel` → 返回 `FAILED` 并打印互斥日志 | UT |
| 功能 | 异步接口反向互斥（防呆） | 同一实例先 `RunModel` 成功后再调 `RunModelWithStreamAsync` → 返回 `FAILED` 并打印互斥日志 | UT |
| 功能 | SymPack输入适配 | 含 `InputBindingMap` 的epair走 `BuildGeInputs` 路径，输出与异步接口一致 | ST（`examples/sym_pack_optimization_case`） |
| 功能 | Host/Device侧输入 | 分别以Host侧、Device侧数据作为输入执行 → 均成功且结果一致 | ST（真机） |
| 功能 | 输出内存自动分配 | `outputs` 传空vector → GE分配Host内存赋值返回，可直接读取 | ST（真机） |
| 端到端 | 同步/异步输出一致性 | 真机同一模型分别加载两个实例，一个执行 `RunModel`、另一个执行 `RunModelWithStreamAsync` + `aclrtSynchronizeStream`（两接口互斥，不可在同一实例混用），输出max‖diff‖=0 | ST |
| 端到端 | 多线程多实例任务分发 | 生产者-消费者模型下4个 `EpairModelLoader` 并行接任务调用 `RunModel`，全部任务成功且输出checksum一致 | ST/示例（`examples/run_model_multi_instance`） |
| 兼容 | 既有接口行为不变 | 全量既有UT回归通过 | UT |

### 测试框架

UT使用GoogleTest（`tests/cpp/test_epair_model_loader.cc` + GE stub），在ASAN构建下运行（`./build_test.sh` 默认启用）；同步/异步输出一致性可复用 `examples/export_and_load_example` 流程在真机验证，多线程多实例端到端参见示例工程 `examples/run_model_multi_instance`。

---

## 验收标准

1. **功能验收**：`CompileAndLoad` 成功后调用 `RunModel`，同步返回 `SUCCESS`，输出可直接读取，无需额外流同步
2. **UT验收**：新增5个UT用例（未加载拒绝 / 同步执行成功 / 失败传播 / 与异步接口互斥拒绝 / 异步接口反向互斥拒绝）在ASAN构建下全部通过，既有用例无回归
3. **接口资料验收**：`docs/zh/api/epair_model_loader.md` 包含RunModel章节（功能说明 / 函数原型 / 参数说明 / 返回值说明 / 约束说明），并与异步接口互相引用
4. **ABI验收**：仅新增非虚成员函数，既有二进制接口不受影响
5. **示例验收**：`examples/run_model_multi_instance` 可完成模型导出、C++构建与4实例4线程生产者-消费者运行，全部任务成功且输出checksum一致
