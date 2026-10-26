---
upstream: 各模块 SPEC.md
upstream_version: 多版本聚合
spec_version: 0.1.0
status: draft
module: ut
depends_on:
  - pipeline+engine_executor/SPEC.md (v0.4.0)
  - c_api/SPEC.md (v0.1.0)
  - tokenizer/SPEC.md (v0.4.0)
  - model_executor/SPEC.md
---

# 单元测试（UT）规格（SPEC）

> **模块定位**：UT 规格定义 CANN-LM Engine 各 UT 二进制中每个用例的输入、前置条件、预期输出与验证点。How（UT 二进制组织、命名规范、Stub/Real Era 分类、CMake 接线）见 DESIGN。本文档与各模块 SPEC 的 TC 编号对齐（同一行为同一 TC，UT 是其可执行版本）。

---

## 第 1 章 概述

本规格覆盖 `cann_lm/engine/` 下对外接口表面与 Stub Era 可独立验证的内部行为。读者应在 30 秒内理解：UT 用例以 `Given/When/Then` 描述，每条用例对齐一个被测行为契约，命名规范（含 Stub Era 后缀）防止"测试名宣称 X 但实际测 Y"的漂移。

需求编号：**SR20260825001**。软件层次：**test**。

## 第 2 章 术语表

| 术语 | 定义 |
|------|------|
| Stub Era | 引擎当前阶段——`stub_model_manager`/`stub_built_model` 返回 nullptr，`BpeTokenizer::encode`/`Sampler::Sample` 返回 FAILURE。但 Executor 构造与 Context 创建仍成功。 |
| Real Era | 真实后端接入后的阶段（本期 UT 不覆盖） |
| 漂移 | 用例名 `<Behavior>` 部分与实际断言类型不一致（详见 DESIGN §3.2） |
| 空句柄 | `RequestHandle handle;` 默认构造——`impl_` 为 nullptr |
| Empty Tensor | `Tensor()` 或 `Tensor(nullptr, 0)`——`operator bool()` 为 false |
| TC | Test Case 编号，与各模块 SPEC §8 对齐 |

## 第 3 章 范围

### 3.1 纳入

- `tests/cpp/engine/` 下 5 个 UT 二进制（含本期新增 `test_sampler.cpp` 与 `test_request_handle.cpp`）的所有用例
- `tests/cpp/engine/model_executor/` 下 5 个文件的用例（仅引用，不在本规格重写——见各模块 SPEC）
- `examples/test_engine/bbit/bbit.cpp` 的 `--check` 与 flow 模式（仅引用——本期不修）

### 3.2 不纳入

- 各模块 SPEC 自身（tokenizer/c_api/pipeline+engine_executor/model_executor 各自有 §8 TC 表）——本规格引用 TC 编号但不重复
- Real Era 推理路径（Generate 成功、Tokenizer 与 Python oracle 一致性）——本期 deferred（Sampler 算法级真实采样已于 2026-09-07 落地，见 §4.5）
- 已有 model_executor UT 的细节修改——见 docs/model_executor/SPEC.md §8
- 各模块 SPEC 的 TC 是否在 UT 中已实现的映射表——本规格仅列"本期新增/修复"的 22 + 2 条

## 第 4 章 功能需求

> 编号体系：`UT-TC-<MODULE>-<NNN>`，其中 MODULE 取 `EE`/`RH`/`CAPI`/`CAPI-FB`/`SAMPLER`/`TK`/`SCHED`。每条用例对应 DESIGN §9 表的一行，对齐各模块 SPEC §8 的 TC 编号（如有）。

### 4.1 ut_engine_executor（UT-TC-EE-*）

#### UT-TC-EE-001 CreateContextSucceeds（关联 docs/pipeline+engine_executor/SPEC §4.1.1）

- **Given** `EngineExecutor ee(DummyPath(), DummyPath())` 已构造
- **When** `ee.CreateContext(&ctx)`
- **Then** 返回 `Status::SUCCESS`；`ctx` 非空；`ctx.id()` 非零
- **断言**：`EXPECT_TRUE(s.ok())` + `EXPECT_TRUE(static_cast<bool>(ctx))` + `EXPECT_TRUE(static_cast<bool>(ctx.id()))`

#### UT-TC-EE-002 DestroyContextReturnsSuccessIdempotent

- **Given** `EngineExecutor ee` 构造
- **When** `ee.DestroyContext(ContextId{42})`（不存在的 id）
- **Then** 返回 `Status::SUCCESS`（幂等约定）
- **断言**：`EXPECT_TRUE(s.ok())`

#### UT-TC-EE-003 ConfigureContextReturnsNotFound（关联 EE SPEC §4.1 异常）

- **Given** `EngineExecutor ee` + 不存在的 ContextId（如 99）
- **When** `ee.ConfigureContext(99, opts)`
- **Then** 返回 `Status::NOT_FOUND`
- **断言**：`EXPECT_EQ(s.code, Status::NOT_FOUND)`

#### UT-TC-EE-004 CancelContextReturnsNotFound

- 同 UT-TC-EE-003，调 `ee.CancelContext(99)`，预期 `NOT_FOUND`

#### UT-TC-EE-005 ConfigureContextSucceedsAfterCreate

- **Given** `ee.CreateContext(&ctx)` 成功
- **When** `ee.ConfigureContext(ctx.id(), opts)`
- **Then** 返回 `Status::SUCCESS`（空 opts 占位，无配置项）
- **断言**：`EXPECT_TRUE(s.ok())`

#### UT-TC-EE-006 CancelContextSucceedsAfterCreate

- 同上但调 `ee.CancelContext(ctx.id())`，预期 `SUCCESS`

#### UT-TC-EE-007 GenerateReturnsFailureOnEmptyContext

- **Given** `EngineExecutor ee` + 空 Context `ctx`（id 为 0）
- **When** `ee.Generate(prompt, ctx, sp, &result)`
- **Then** 返回非 SUCCESS（pipeline 找不到 ctx → NOT_FOUND 或 FAILURE）
- **断言**：`EXPECT_FALSE(s.ok())`

#### UT-TC-EE-008 GenerateReturnsFailureOnValidContextStubEra

- **Given** `ee.CreateContext(&ctx)` 成功（ctx.id() 非零）
- **When** `ee.Generate(prompt, ctx, sp, &result)`
- **Then** 返回非 SUCCESS——PreparePrefill 失败（tokenizer stub encode 返回 FAILURE）
- **断言**：`EXPECT_FALSE(s.ok())`

#### UT-TC-EE-009 GenerateAsyncReturnsFailureOnEmptyContext

- 同 UT-TC-EE-007，调 `ee.GenerateAsync(prompt, ctx, sp, &handle)`，预期 `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-010 GenerateAsyncReturnsFailureOnValidContextStubEra

- 同 UT-TC-EE-008，调 `ee.GenerateAsync(...)`，预期 `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-011 ConcurrentGenerateSameContextReturnsFailureStubEra（**DRIFT-1 修复**）

- **Given** `ee.CreateContext(&ctx)` 成功
- **When** 连续两次 `ee.GenerateAsync(prompt, ctx, sp, &h1)` → `&h2`（同一 ctx）
- **Then** 两次均返回非 SUCCESS——首次：PreparePrefill 同步失败；清除 active 标记后第二次仍 FAILURE（不是 INVALID_STATE，因 active 标记已被 RAII 清理）
- **断言**：`EXPECT_FALSE(s1.ok())` + `EXPECT_FALSE(s2.ok())`
- **修复说明**：旧名 `ConcurrentGenerateSameContextReturnsInvalidState` 宣称 `InvalidState`，但实际断言未验证该码且语义与 stub 时代行为不符。新名对齐真实行为。

### 4.2 ut_tokenizer（UT-TC-TK-*）

#### UT-TC-TK-001 BatchEncodeContinueOnFailureNotAbort（关联 tokenizer SPEC TC_2xx）

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({"text1","text2","text3"}, opts, &out)`
- **Then** 返回非 SUCCESS（首条 encode FAILURE）；`out.size() == 3`；每条空 vector（失败条目填空）
- **断言**：`EXPECT_FALSE(s.ok())` + `EXPECT_EQ(out.size(), 3u)` + 循环 `EXPECT_TRUE(v.empty())`

#### UT-TC-TK-002 BatchEncodeEmptyInputReturnsSuccess

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({}, opts, &out)`
- **Then** 返回 SUCCESS；`out.empty()`
- **断言**：`EXPECT_TRUE(s.ok())` + `EXPECT_TRUE(out.empty())`

#### UT-TC-TK-003 BatchEncodeNullOutReturnsInvalidArgument

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({"test"}, opts, nullptr)`
- **Then** 返回 `Status::INVALID_ARGUMENT`（基类默认实现首行检查）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-TK-004 ListRegisteredReturnsBPEAndSP（关联 tokenizer SPEC TC_004）

- **Given** 静态初始化期 BPE + SP 已注册
- **When** `TokenizerFactory::list_registered()`
- **Then** 返回 2 个算法
- **断言**：`EXPECT_EQ(algos.size(), 2u)`

#### UT-TC-TK-005 CreateBPEReturnsInvalidArgumentStubEra（**DRIFT-2 修复**）

- **Given** `LoadParams params;` （空）
- **When** `TokenizerFactory::create(Algorithm::BPE, params, &tok)`
- **Then** 返回 `Status::INVALID_ARGUMENT`（stub 时代 `TokenizerFactory::create` 始终返回该码，未真实查注册表）
- **断言**：`EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`
- **修复说明**：旧名 `CreateUnknownReturnsFailure` 既错（入参 BPE 是已注册算法）又错（码不是 FAILURE）。新名反映真实入参与真实码。

### 4.3 ut_c_api（UT-TC-CAPI-* + UT-TC-CAPI-FB-*）

#### UT-TC-CAPI-001 ContextCancelReturnsNotFoundForNonexistentContext

- **Given** `CANN_LMEngine_ExecutorCreate(&p,&p,&opts,&exec)` 成功
- **When** `CANN_LMEngine_ContextCancel(exec, 999)`
- **Then** 返回 `CANN_LM_ENGINE_NOT_FOUND`
- **Then** `CANN_LMEngine_ContextDestroy(exec, 999)` 返回 SUCCESS（幂等）

#### UT-TC-CAPI-002 NullExecutorReturnsInvalidArgument

- **Given** NULL executor
- **When** `CANN_LMEngine_ContextCreate(nullptr, &id)` / `CANN_LMEngine_ContextCancel(nullptr, 0)` / `CANN_LMEngine_ContextDestroy(nullptr, 0)`
- **Then** 均返回 `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-003 RequestIsDoneReturnsFalseForNull

- `CANN_LMEngine_RequestIsDone(nullptr)` == 0

#### UT-TC-CAPI-004 RequestCancelReturnsSuccessForNull

- `CANN_LMEngine_RequestCancel(nullptr)` == `CANN_LM_ENGINE_SUCCESS`（空句柄幂等）

#### UT-TC-CAPI-005 RequestGetStatusReturnsQueuedForNull

- `CANN_LMEngine_RequestGetStatus(nullptr)` == `CANN_LM_ENGINE_REQUEST_QUEUED`

#### UT-TC-CAPI-006 GenerateOnNonexistentContextReturnsNotFound

- **Given** Executor 已创建
- **When** `CANN_LMEngine_ExecutorGenerate(exec, "hello", 999, nullptr, &result)`
- **Then** 返回 `CANN_LM_ENGINE_NOT_FOUND`

#### UT-TC-CAPI-007 ContextCreateSucceeds

- **Given** Executor 已创建
- **When** `CANN_LMEngine_ContextCreate(exec, &id)`
- **Then** 返回 `CANN_LM_ENGINE_SUCCESS`；`id != 0`

#### UT-TC-CAPI-008 ContextCancelSucceedsOnCreatedContext

- 同 UT-TC-CAPI-007 + `CANN_LMEngine_ContextCancel(exec, id)` 返回 SUCCESS

#### UT-TC-CAPI-009 GenerateReturnsFailureOnCreatedContextStubEra

- **Given** Executor + Context 已创建
- **When** `CANN_LMEngine_ExecutorGenerate(exec, "hello", id, &params, &result)`
- **Then** 返回非 SUCCESS（stub FAILURE）
- **Then** 调用方需 `CANN_LMEngine_ResultDestroy(&result)` 释放

#### UT-TC-CAPI-FB-001 CreateFromBufferSucceedsStubEra（**新增 NEW-12**）

- **Given** 构造 `CANN_LMEngine_Buffer buf{data, size}`（`char buf_data[8]` 栈缓冲）；`CANN_LMEngine_ModelArtifact ma{buf, buf, buf}`；`CANN_LMEngine_EngineOptions opts{}`
- **When** `CANN_LMEngine_ExecutorCreateFromBuffer(&ma, &ma, &opts, &exec)`
- **Then** 返回 `CANN_LM_ENGINE_SUCCESS`；`exec` 非 NULL
- **Then** 后续调 `CANN_LMEngine_ExecutorDestroy(exec)` 释放（无 leak）

#### UT-TC-CAPI-FB-002 CreateFromBufferNullPrefillReturnsInvalidArgument（**新增 NEW-13**）

- **Given** `CANN_LMEngine_ModelArtifact ma{...}`；`opts`；`exec`
- **When** `CANN_LMEngine_ExecutorCreateFromBuffer(nullptr, &ma, &opts, &exec)`
- **Then** 返回 `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-003 CreateFromBufferNullDecodeReturnsInvalidArgument（**新增 NEW-14**）

- 同 UT-TC-CAPI-FB-002，但 `prefill=&ma, decode=nullptr`
- **Then** 返回 `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-004 CreateFromBufferNullOutReturnsInvalidArgument（**新增 NEW-15**）

- 同上，但 `out=nullptr`
- **Then** 返回 `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-005 CreateFromBufferSupportsContextCreate（**新增 NEW-16**）

- **Given** `ExecutorCreateFromBuffer` 返回 SUCCESS（UT-TC-CAPI-FB-001 同 setup）
- **When** `CANN_LMEngine_ContextCreate(exec, &id)`
- **Then** 返回 `CANN_LM_ENGINE_SUCCESS`；`id != 0`——验证 buffer 入口构造的 Executor 内部 `pipeline_` 真就绪（与 `executor_buffer_ctor_returns_failure_when_factory_fails` 行为互补）
- **Then** `CANN_LMEngine_ContextDestroy(exec, id)` + `ExecutorDestroy(exec)`

### 4.4 ut_scheduler（UT-TC-SCHED-*）

#### UT-TC-SCHED-001 StepWithNoWorkReturnsInvalidState

- **Given** `SchedulerFixture f`（prefill_exec/decode_exec 均 `nullptr`；`max_batch_size=1`）
- **When** `sched.Step(&out)`
- **Then** 返回 `Status::INVALID_STATE`（无活任务）
- **断言**：`EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-SCHED-002 DecodeCapacityMatchesMaxBatchSize

- **Given** `SchedulerFixture f`，`max_batch_size=4`
- **Then** `sched.DecodeCapacity() == 4`；`ActiveDecodeCount() == 0`；`PendingPrefillCount() == 0`

#### UT-TC-SCHED-003 ConstructWithBatchSize1

- 同 UT-TC-SCHED-002 但 `max_batch_size=1`，验证 `DecodeCapacity() == 1`

#### UT-TC-SCHED-101 SubmitWithNullReqReturnsInvalidArgument（**新增 NEW-17**）

- **Given** `SchedulerFixture f`
- **When** `sched.Submit({}, nullptr, &out)`（req=nullptr）
- **Then** 返回 `Status::INVALID_ARGUMENT`（impl `scheduler.cpp:59`）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SCHED-102 SubmitWithNullOutReturnsInvalidArgument（**新增 NEW-18**）

- 同上但 `req` 有效、`out=nullptr`
- **Then** 返回 `Status::INVALID_ARGUMENT`
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SCHED-004 SubmitSucceedsIncrementsPendingCount（**新增 NEW-19**）

- **Given** `SchedulerFixture f`，`max_batch_size=2`
- **When** `sched.Submit({}, std::make_shared<RequestImpl>(), &id)`（empty chunks 合法）
- **Then** 返回 `Status::SUCCESS`；`id.value != 0`
- **Then** `sched.PendingPrefillCount() == 1`（impl `scheduler.cpp:70` push_back 后）
- **断言**：`EXPECT_TRUE(s.ok())` + `EXPECT_NE(id.value, 0u)` + `EXPECT_EQ(sched.PendingPrefillCount(), 1)`

#### UT-TC-SCHED-005 SubmitAssignsMonotonicRequestIds（**新增 NEW-20**）

- **Given** 同上
- **When** 连续两次 `Submit`，记 `id1` 与 `id2`
- **Then** `id1.value != 0`；`id2.value != 0`；`id2.value > id1.value`（`next_id_` 单调递增）
- **断言**：`EXPECT_GT(id2.value, id1.value)`

#### UT-TC-SCHED-103 CancelNonexistentReturnsNotFound（**新增 NEW-21**）

- **Given** `SchedulerFixture f`，无任务
- **When** `sched.Cancel(RequestId{42})`
- **Then** 返回 `Status::NOT_FOUND`（impl `scheduler.cpp:251` 路径未匹配）
- **断言**：`EXPECT_EQ(s.code, Status::NOT_FOUND)`

#### UT-TC-SCHED-104 CancelContextWithNoMatchingContextsReturnsSuccess（**新增 NEW-22**）

- **Given** `SchedulerFixture f`，无任务
- **When** `sched.CancelContext(ContextId{42})`
- **Then** 返回 `Status::SUCCESS`（impl `scheduler.cpp:294` 始终返回 SUCCESS）
- **断言**：`EXPECT_TRUE(s.ok())`

### 4.5 ut_sampler（UT-TC-SAMPLER-*）——**新建文件**

#### UT-TC-SAMPLER-001 SampleWithNullOutReturnsInvalidArgument_TC201（**新增 NEW-1**）

- **Given** `Sampler sampler`；空 Tensor `Tensor t_empty`；`SamplingParams params`
- **When** `sampler.Sample(t_empty, 0, params, nullptr)`（out=nullptr）
- **Then** 返回 `Status::INVALID_ARGUMENT`（impl `sampler.cpp` 参数守卫）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SAMPLER-002 SampleWithNegativeSlotReturnsInvalidArgument_TC202（**新增 NEW-2**）

- **Given** `Sampler sampler`；`TokenId out`
- **When** `sampler.Sample(t_empty, -1, params, &out)`（slot<0）
- **Then** 返回 `Status::INVALID_ARGUMENT`（impl `sampler.cpp` 参数守卫）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SAMPLER-003 SampleWithNullDataReturnsFailure_TC203（**新增 NEW-3**；2026-09-07 改名，原 SampleWithEmptyLogitsReturnsFailureStubEra）

- **Given** `Sampler sampler`；GetData()==nullptr 的 Tensor；`TokenId out`
- **When** `sampler.Sample(t_empty, 0, params, &out)`
- **Then** 返回 `Status::FAILURE`（logits 缺失防御分支）
- **断言**：`EXPECT_EQ(s.code, Status::FAILURE)`

#### UT-TC-SAMPLER-004 SampleWithDefaultsReturnsGreedyArgmax_TC204（**新增 NEW-4**；2026-09-07 改写，原 SampleWithValidLogitsReturnsFailureStubEra——算法 Real Era 后不再是 FAILURE）

- **Given** `Sampler sampler`；FLOAT32 logits `{1.0, 3.0, 2.0, 0.5}`（dims `[4]`）；默认 `SamplingParams`
- **When** `sampler.Sample(t_valid, 0, params, &out)`
- **Then** 返回 `Status::SUCCESS`，`out == 1`（贪心 argmax；默认参数 temp=1/top_k=0/top_p=1 → 贪心）
- **断言**：`EXPECT_EQ(s.code, Status::SUCCESS)` + `EXPECT_EQ(out, 1u)`

#### UT-TC-SAMPLER-005 SampleBatchedDimsPicksPerSlotRow_TC205（**2026-09-07 新增**）

- **Given** FLOAT32 logits dims `[2,3]`（两行 argmax 分别为 1、0）
- **When** `sampler.Sample(t, 0/1/2, params, &out)`
- **Then** slot=0 → `out==1`；slot=1 → `out==0`；slot>=2（越界）→ `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-006 SampleWith1DDimsRejectsSlotBeyondRow_TC206（**2026-09-07 新增**）

- **Given** dims `[4]`（单行容量）
- **When** `sampler.Sample(t, 1, params, &out)`
- **Then** `INVALID_ARGUMENT`（slot 超出单行容量）

#### UT-TC-SAMPLER-007 SampleWithNonFloat32ReturnsInvalidArgument_TC207（**2026-09-07 新增**）

- **Given** dataType=INT32 的 logits
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`（仅支持 FLOAT32，见 sampler.h 已知限制）

#### UT-TC-SAMPLER-008 SampleWithMisalignedBytesReturnsInvalidArgument_TC208（**2026-09-07 新增**）

- **Given** size=6 字节（6 % 4 != 0）的 FLOAT32 缓冲
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-009 SampleWithZeroSizeReturnsInvalidArgument_TC209（**2026-09-07 新增**）

- **Given** GetData() 非空但 size==0
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-010 SampleWithoutDimsAndSlotPositiveReturnsFailure_TC210（**2026-09-07 新增**）

- **Given** dims 缺失（host 回退缓冲常见形态）、slot=1
- **When** `sampler.Sample(t, 1, params, &out)`
- **Then** `FAILURE`（无 dims 无法安全推导行宽）

#### UT-TC-SAMPLER-011 SampleWithTopKAndZeroTemperatureStaysGreedy_TC211（**2026-09-07 新增**）

- **Given** logits `{1.0, 5.0, 4.9, 0.0}`；`top_k=2`、`temperature=0.0`
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `SUCCESS` 且 `out==1`（temp=0 经 1e-8 钳制退化为贪心，确定性）

#### UT-TC-SAMPLER-012 SampleWithTopPWiresNucleusSampling_TC212（**2026-09-07 新增**）

- **Given** logits `{1.0, 5.0, 4.9, 0.0}`；`top_p=0.9`
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `SUCCESS` 且 `out ∈ {1, 2}`（核采样截尾后 top-2；概率性只断言集合）

#### UT-TC-SAMPLER-013 SampleWithDescBufferMismatchReturnsInvalidArgument_TC213（**2026-09-07 新增**）

- **Given** dims `[2,4]` 但缓冲仅 2 元素（desc 与实际分配脱节）
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`（真实后端 desc 失准最先暴露的路径）

#### UT-TC-SAMPLER-014 SampleWithNonPositiveDimsReturnsInvalidArgument_TC214（**2026-09-07 新增**）

- **Given** dims 含非正维（`{0}` 或 `{2,-3,4}`）
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`（末维/前缀维非正均拒绝）

#### 算法层直测（`SamplerAlgoTest`，2026-09-07 新增）

静态纯函数直测（不经 ITensorBuffer）：`ApplyTemperature`（除法/非正钳制）、`ApplyTopK`
（top-k 保留其余 -inf / k 越界 no-op）、`ApplyPrecomputedTopK`（集合外置 -inf）、`ApplyTopP`
（最高概率项必留）、`ApplyRepetitionPenalty`（score<0 ×p、score≥0 ÷p、未生成不动）、
`Normalize`（非负、和≈1、概率序随 logit 序）、`Pick`（贪心 argmax / seeded 可复现）、
全流水线（默认贪心 / seeded 采样可复现）。期望行为另由
`cann_lm/engine/core/sample/test/sampler_ut.cc`（100 断言，scripts/build_sample.sh）
对参考实现侧锚定，两份实现勿单方漂移。

### 4.6 ut_request_handle（UT-TC-RH-*）——**新建文件**

#### UT-TC-RH-001 EmptyHandleGetStatusReturnsQueued（**新增 NEW-5**）

- **Given** `RequestHandle handle;`（默认构造，`impl_` 为空）
- **When** `handle.GetStatus()`
- **Then** 返回 `RequestStatus::Queued`（impl `request_handle.cpp:17`）
- **断言**：`EXPECT_EQ(handle.GetStatus(), RequestStatus::Queued)`

#### UT-TC-RH-002 EmptyHandleIsDoneReturnsFalse（**新增 NEW-6**）

- **Given** 空句柄
- **When** `handle.IsDone()`
- **Then** 返回 false（impl `request_handle.cpp:22`）
- **断言**：`EXPECT_FALSE(handle.IsDone())`

#### UT-TC-RH-003 EmptyHandleBoolConversionReturnsFalse（**新增 NEW-7**）

- **Given** 空句柄
- **When** `static_cast<bool>(handle)` 或 `if (handle)`
- **Then** 返回 false（impl `request_handle.cpp:103`）
- **断言**：`EXPECT_FALSE(static_cast<bool>(handle))`

#### UT-TC-RH-004 EmptyHandleCancelReturnsSuccess（**新增 NEW-8**）

- **Given** 空句柄
- **When** `handle.Cancel()`
- **Then** 返回 `Status::SUCCESS`（impl `request_handle.cpp:34`，空句柄幂等）
- **断言**：`EXPECT_TRUE(handle.Cancel().ok())`

#### UT-TC-RH-005 EmptyHandleWaitReturnsInvalidState（**新增 NEW-9**）

- **Given** 空句柄；`Result out`
- **When** `handle.Wait(&out)`
- **Then** 返回 `Status::INVALID_STATE`（impl `request_handle.cpp:41`，空句柄不阻塞立即返回）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-RH-006 EmptyHandleTryGetResultReturnsInvalidState（**新增 NEW-10**）

- **Given** 空句柄；`Result out`
- **When** `handle.TryGetResult(&out)`
- **Then** 返回 `Status::INVALID_STATE`（impl `request_handle.cpp:82`）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-RH-007 EmptyHandleWaitWithNullOutReturnsInvalidArgument（**新增 NEW-11**）

- **Given** 空句柄
- **When** `handle.Wait(nullptr)`（out=nullptr）
- **Then** 返回 `Status::INVALID_ARGUMENT`（impl `request_handle.cpp:40`，参数校验先于空句柄检查）
- **断言**：`EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`
- **说明**：参数校验顺序：先 `!out` 再 `!impl_`。所以传 nullptr 即使 impl_ 也空也走 INVALID_ARGUMENT 分支。
- **TryGetResult 同样**：`handle.TryGetResult(nullptr)` 返回 `INVALID_ARGUMENT`

## 第 5 章 非功能需求

| 维度 | 指标 | 测试条件 | 证据来源 |
|------|------|----------|----------|
| 单 UT 文件编译时延 | < 5s（增量） | aarch64 + clang/gcc，O0 Debug | 现有构建基线 |
| 全 UT 跑完时延 | < 10s | CTest 全部 discover 出的用例 | 现有 bbit + 5 UT 二进制 |
| UT 数量增长 | 本期 +22 条 | 新增后总用例数 ≈ 88（原 66 + 新 22） | DESIGN §9 |
| 内存 | 无 leak | ASAN/LSAN 干净 | 实现 §8 测试钩子 |
| 线程安全 | bbit 与 worker 线程交互无竞争 | TSan 干净 | deferred（本期不跑 TSan） |

## 第 6 章 约束与依赖

- **被测代码**：`cann_lm/engine/*`（44 个 cpp/h 文件）
- **测试框架**：GoogleTest 1.14（`/usr/src/googletest/`，CMake `find_package(GTest REQUIRED)`）
- **编译器**：g++ 13 / clang-18 / C++17（与生产代码一致）
- **构建系统**：CMake ≥ 3.10（与 `cann_lm/engine/CMakeLists.txt` 一致）
- **依赖的模块 SPEC**：tokenizer / c_api / pipeline+engine_executor / model_executor（各模块 §8 TC 编号对齐）

## 第 7 章 验证标准

| 类别 | 覆盖 TC（按 §4 编号） |
|------|----------------------|
| 正常路径（0xx） | EE-001/002/005/006/011；TK-002/004/005；CAPI-007/008；CAPI-FB-001/002/005；SCHED-002/003/004/005/104；RH-001/002/003/004 |
| 异常路径（1xx） | EE-003/004/007/008/009/010；TK-001/003；CAPI-001/002/006/009；CAPI-FB-002/003/004；SCHED-101/102/103；SAMPLER-001/002/003/004；RH-005/006/007 |
| 边界条件（2xx） | TK-001（continue-on-failure）；SAMPLER-003/004（空/非空 Tensor）；RH-001/002/003（空句柄） |
| 并发场景（3xx） | EE-011（同 ctx 连续 GenerateAsync） |

**自检**：4 类最小覆盖维度每类 ≥1 条；本期新增 22 条 + 修复 2 条，全部 Stub Era 行为可验证。Real Era 推理路径 deferred。

## 第 8 章 漂移修复与新增用例规格

### 8.1 漂移修复（DRIFT-1, DRIFT-2）

详见 DESIGN §8。修复仅改用例名 + 调整断言，不改被测实现行为。

| 编号 | 旧名 → 新名 | 旧断言 → 新断言 | 关联 UT-TC |
|------|-------------|-----------------|------------|
| DRIFT-1 | `ConcurrentGenerateSameContextReturnsInvalidState` → `ConcurrentGenerateSameContextReturnsFailureStubEra` | 两次 `EXPECT_FALSE(s.ok())` → 两次 `EXPECT_FALSE(s.ok())`（断言不变，仅改名） | UT-TC-EE-011 |
| DRIFT-2 | `CreateUnknownReturnsFailure` → `CreateBPEReturnsInvalidArgumentStubEra` | `EXPECT_FALSE(s.ok())` → `EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`（补码验证） | UT-TC-TK-005 |

### 8.2 新增用例（NEW-1..NEW-22）

详见 DESIGN §9 与 §4 各 UT-TC 条目。每条新增用例均含：Given / When / Then / 断言 / 关联 SPEC TC。

### 8.3 覆盖维度核验

- **stub 时代可观测行为**：所有 22 条新增 + 2 条修复覆盖 EE/RH/CAPI/SAMPLER/TK/SCHED 6 个模块的对外接口表面与已暴露的内部行为缺口 ✓
- **不验证**：Real Era 推理路径（Generate 成功、Tokenizer 真实 encode/decode）—— deferred 至真实后端接入（Sampler 算法级真实采样已于 2026-09-07 落地，见 §4.5）
- **未覆盖模块**：`text_embedding` / `pipeline_executor` / `preprocessor` / `history_prompt_manager` / `infra/logger` / `core/context` / `core/tensor_manager` / `request_handle` 真实生命周期（非空 impl_）—— 见 DESIGN §10 D-05

## 第 9 章 未决项登记

| ID | 未决项 | 类别 | 阻塞 spec 放行 | 处置阶段 |
|----|--------|------|----------------|----------|
| UT-U-1 | `Algorithm` 枚举仅 BPE/SP 两值，stub 时代无法构造真正"未知"枚举 | 系统边界 | 否（本期改名 `CreateBPEReturnsInvalidArgumentStubEra`，不验证 unknown 路径） | Real Era：扩 Algorithm 枚举或经测试 hook 注入未知值后补 `CreateUnknownAlgorithmReturnsInvalidArgument_RealEra` |
| UT-U-2 | `ExecutorCreateFromBuffer` 真实成功路径依赖 `PipelineExecutorFactory::create(buffer)` 不返回错误 | 接口契约 | 否（UT-TC-CAPI-FB-005 通过 `ContextCreate` 间接验证 pipeline_ 就绪） | Real Era：追加 `GenerateFromBufferExecutorSucceeds_RealEra` |
| UT-U-3 | bbit 在 stub 时代跑全 FAILURE 路径，是否需要为 bbit 追加 stub-era 期望 | 集成测试 | 否（bbit 自身已对齐） | 后续：bbit 审计 |
| UT-U-4 | 真实 RequestHandle 生命周期（impl_ 非空 → Wait 阻塞 → 终态语义）测试 | 接口契约 | 否（本期仅覆盖空句柄） | Real Era：构造非空 RequestImpl，经 Scheduler::Submit 后测试 Wait/Cancel/TryGetResult |

**阻塞级未决项清零**：4 项均不阻塞本期 UT 落地。

## 第 10 章 一致性自检（生成期）

- 命名规范一致：全部用例遵循 `<Module>Test, <Behavior>_<TCXXX>` + Stub Era 后缀 ✓
- 编号一致：UT-TC-EE-* / UT-TC-TK-* / UT-TC-CAPI-* / UT-TC-CAPI-FB-* / UT-TC-SCHED-* / UT-TC-SAMPLER-* / UT-TC-RH-* 各自独立编号空间，与 DESIGN §9 表对齐 ✓
- 漂移修复一致：DRIFT-1 / DRIFT-2 在 §4.1 / §4.2 与 §8.1 表均出现且行为描述一致 ✓
- 新增用例一致：22 条新增用例在 §4.x 与 DESIGN §9 表对齐 ✓
- Stub/Real 分类一致：所有 Stub Era 用例明确标 `_StubEra` 后缀或前置"Stub Era"语义；Real Era 标 deferred ✓
- 覆盖维度自检：4 类（正常/异常/边界/并发）每类 ≥1 条 ✓
