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

# Unit Test (UT) Specification (SPEC)

English | [简体中文](../../../zh/cann_lm_engine/ut/SPEC.md)

> **Module positioning**: UT spec defines each case's input, preconditions, expected output, and verification points in each UT binary of CANN-LM Engine. How (UT binary organization, naming conventions, Stub/Real Era classification, CMake wiring) see DESIGN. This document aligns with each module SPEC's TC numbering (same behavior same TC, UT is its executable version).

---

## Chapter 1 Overview

This spec covers the external interface surface of `cann_lm/engine/` and Stub Era independently verifiable internal behaviors. The reader should understand within 30 seconds: UT cases described with `Given/When/Then`, each case aligned with a tested behavior contract, naming conventions (including Stub Era suffix) prevent "test name claims X but actually tests Y" drift.

Requirement ID: **SR20260825001**. Software layer: **test**.

## Chapter 2 Glossary

| Term | Definition |
|------|------|
| Stub Era | Engine's current phase — `stub_model_manager`/`stub_built_model` return nullptr, `BpeTokenizer::encode`/`Sampler::Sample` return FAILURE. But Executor construction and Context creation still succeed. |
| Real Era | Phase after real backend integration (this phase UT doesn't cover) |
| Drift | Case name `<Behavior>` part inconsistent with actual assertion type (see DESIGN §3.2) |
| Empty handle | `RequestHandle handle;` default construction — `impl_` is nullptr |
| Empty Tensor | `Tensor()` or `Tensor(nullptr, 0)` — `operator bool()` is false |
| TC | Test Case number, aligned with each module SPEC §8 |

## Chapter 3 Scope

### 3.1 In Scope

- All cases in 5 UT binaries under `tests/cpp/engine/` (including this phase's new `test_sampler.cpp` and `test_request_handle.cpp`)
- Cases in 5 files under `tests/cpp/engine/model_executor/` (reference only, not rewritten in this spec — see each module SPEC)
- `examples/test_engine/bbit/bbit.cpp`'s `--check` and flow modes (reference only — not fixed this phase)

### 3.2 Out of Scope

- Each module SPEC itself (tokenizer/c_api/pipeline+engine_executor/model_executor each has §8 TC table) — this spec references TC numbers but doesn't duplicate
- Real Era inference path (Generate success, Tokenizer Python oracle consistency) — this phase deferred (Sampler algorithm-level real sampling landed 2026-09-07, see §4.5)
- Detail modifications of existing model_executor UT — see docs/model_executor/SPEC.md §8
- Mapping table of each module SPEC's TC whether implemented in UT — this spec only lists "this phase new/fixed" 22 + 2 cases

## Chapter 4 Functional Requirements

> Numbering system: `UT-TC-<MODULE>-<NNN>`, where MODULE takes `EE`/`RH`/`CAPI`/`CAPI-FB`/`SAMPLER`/`TK`/`SCHED`. Each case corresponds to one row in DESIGN §9 table, aligned with each module SPEC §8 TC numbering (if any).

### 4.1 ut_engine_executor (UT-TC-EE-*)

#### UT-TC-EE-001 CreateContextSucceeds (related docs/pipeline+engine_executor/SPEC §4.1.1)

- **Given** `EngineExecutor ee(DummyPath(), DummyPath())` already constructed
- **When** `ee.CreateContext(&ctx)`
- **Then** returns `Status::SUCCESS`; `ctx` non-null; `ctx.id()` non-zero
- **Assertion**: `EXPECT_TRUE(s.ok())` + `EXPECT_TRUE(static_cast<bool>(ctx))` + `EXPECT_TRUE(static_cast<bool>(ctx.id()))`

#### UT-TC-EE-002 DestroyContextReturnsSuccessIdempotent

- **Given** `EngineExecutor ee` constructed
- **When** `ee.DestroyContext(ContextId{42})` (non-existent id)
- **Then** returns `Status::SUCCESS` (idempotent convention)
- **Assertion**: `EXPECT_TRUE(s.ok())`

#### UT-TC-EE-003 ConfigureContextReturnsNotFound (related EE SPEC §4.1 exception)

- **Given** `EngineExecutor ee` + non-existent ContextId (e.g. 99)
- **When** `ee.ConfigureContext(99, opts)`
- **Then** returns `Status::NOT_FOUND`
- **Assertion**: `EXPECT_EQ(s.code, Status::NOT_FOUND)`

#### UT-TC-EE-004 CancelContextReturnsNotFound

- Same as UT-TC-EE-003, call `ee.CancelContext(99)`, expect `NOT_FOUND`

#### UT-TC-EE-005 ConfigureContextSucceedsAfterCreate

- **Given** `ee.CreateContext(&ctx)` success
- **When** `ee.ConfigureContext(ctx.id(), opts)`
- **Then** returns `Status::SUCCESS` (empty opts placeholder, no config items)
- **Assertion**: `EXPECT_TRUE(s.ok())`

#### UT-TC-EE-006 CancelContextSucceedsAfterCreate

- Same as above but call `ee.CancelContext(ctx.id())`, expect `SUCCESS`

#### UT-TC-EE-007 GenerateReturnsFailureOnEmptyContext

- **Given** `EngineExecutor ee` + empty Context `ctx` (id is 0)
- **When** `ee.Generate(prompt, ctx, sp, &result)`
- **Then** returns non-SUCCESS (pipeline can't find ctx → NOT_FOUND or FAILURE)
- **Assertion**: `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-008 GenerateReturnsFailureOnValidContextStubEra

- **Given** `ee.CreateContext(&ctx)` success (ctx.id() non-zero)
- **When** `ee.Generate(prompt, ctx, sp, &result)`
- **Then** returns non-SUCCESS — PreparePrefill fails (tokenizer stub encode returns FAILURE)
- **Assertion**: `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-009 GenerateAsyncReturnsFailureOnEmptyContext

- Same as UT-TC-EE-007, call `ee.GenerateAsync(prompt, ctx, sp, &handle)`, expect `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-010 GenerateAsyncReturnsFailureOnValidContextStubEra

- Same as UT-TC-EE-008, call `ee.GenerateAsync(...)`, expect `EXPECT_FALSE(s.ok())`

#### UT-TC-EE-011 ConcurrentGenerateSameContextReturnsFailureStubEra (**DRIFT-1 fix**)

- **Given** `ee.CreateContext(&ctx)` success
- **When** consecutive two `ee.GenerateAsync(prompt, ctx, sp, &h1)` → `&h2` (same ctx)
- **Then** both return non-SUCCESS — first: PreparePrefill synchronous failure; after clearing active flag, second still FAILURE (not INVALID_STATE, because active flag already cleaned by RAII)
- **Assertion**: `EXPECT_FALSE(s1.ok())` + `EXPECT_FALSE(s2.ok())`
- **Fix note**: old name `ConcurrentGenerateSameContextReturnsInvalidState` claimed `InvalidState`, but actual assertion didn't verify that code and semantics inconsistent with stub era behavior. New name aligns with real behavior.

### 4.2 ut_tokenizer (UT-TC-TK-*)

#### UT-TC-TK-001 BatchEncodeContinueOnFailureNotAbort (related tokenizer SPEC TC_2xx)

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({"text1","text2","text3"}, opts, &out)`
- **Then** returns non-SUCCESS (first encode FAILURE); `out.size() == 3`; each empty vector (failed entries filled empty)
- **Assertion**: `EXPECT_FALSE(s.ok())` + `EXPECT_EQ(out.size(), 3u)` + loop `EXPECT_TRUE(v.empty())`

#### UT-TC-TK-002 BatchEncodeEmptyInputReturnsSuccess

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({}, opts, &out)`
- **Then** returns SUCCESS; `out.empty()`
- **Assertion**: `EXPECT_TRUE(s.ok())` + `EXPECT_TRUE(out.empty())`

#### UT-TC-TK-003 BatchEncodeNullOutReturnsInvalidArgument

- **Given** `BpeTokenizer tok`
- **When** `tok.batch_encode({"test"}, opts, nullptr)`
- **Then** returns `Status::INVALID_ARGUMENT` (base class default implementation first-line check)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-TK-004 ListRegisteredReturnsBPEAndSP (related tokenizer SPEC TC_004)

- **Given** BPE + SP already registered during static initialization
- **When** `TokenizerFactory::list_registered()`
- **Then** returns 2 algorithms
- **Assertion**: `EXPECT_EQ(algos.size(), 2u)`

#### UT-TC-TK-005 CreateBPEReturnsInvalidArgumentStubEra (**DRIFT-2 fix**)

- **Given** `LoadParams params;` (empty)
- **When** `TokenizerFactory::create(Algorithm::BPE, params, &tok)`
- **Then** returns `Status::INVALID_ARGUMENT` (stub era `TokenizerFactory::create` always returns this code, doesn't actually look up registry)
- **Assertion**: `EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`
- **Fix note**: old name `CreateUnknownReturnsFailure` is wrong both ways (input BPE is registered algorithm) and (code is not FAILURE). New name reflects real input and real code.

### 4.3 ut_c_api (UT-TC-CAPI-* + UT-TC-CAPI-FB-*)

#### UT-TC-CAPI-001 ContextCancelReturnsNotFoundForNonexistentContext

- **Given** `CANN_LMEngine_ExecutorCreate(&p,&p,&opts,&exec)` success
- **When** `CANN_LMEngine_ContextCancel(exec, 999)`
- **Then** returns `CANN_LM_ENGINE_NOT_FOUND`
- **Then** `CANN_LMEngine_ContextDestroy(exec, 999)` returns SUCCESS (idempotent)

#### UT-TC-CAPI-002 NullExecutorReturnsInvalidArgument

- **Given** NULL executor
- **When** `CANN_LMEngine_ContextCreate(nullptr, &id)` / `CANN_LMEngine_ContextCancel(nullptr, 0)` / `CANN_LMEngine_ContextDestroy(nullptr, 0)`
- **Then** all return `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-003 RequestIsDoneReturnsFalseForNull

- `CANN_LMEngine_RequestIsDone(nullptr)` == 0

#### UT-TC-CAPI-004 RequestCancelReturnsSuccessForNull

- `CANN_LMEngine_RequestCancel(nullptr)` == `CANN_LM_ENGINE_SUCCESS` (empty handle idempotent)

#### UT-TC-CAPI-005 RequestGetStatusReturnsQueuedForNull

- `CANN_LMEngine_RequestGetStatus(nullptr)` == `CANN_LM_ENGINE_REQUEST_QUEUED`

#### UT-TC-CAPI-006 GenerateOnNonexistentContextReturnsNotFound

- **Given** Executor already created
- **When** `CANN_LMEngine_ExecutorGenerate(exec, "hello", 999, nullptr, &result)`
- **Then** returns `CANN_LM_ENGINE_NOT_FOUND`

#### UT-TC-CAPI-007 ContextCreateSucceeds

- **Given** Executor already created
- **When** `CANN_LMEngine_ContextCreate(exec, &id)`
- **Then** returns `CANN_LM_ENGINE_SUCCESS`; `id != 0`

#### UT-TC-CAPI-008 ContextCancelSucceedsOnCreatedContext

- Same as UT-TC-CAPI-007 + `CANN_LMEngine_ContextCancel(exec, id)` returns SUCCESS

#### UT-TC-CAPI-009 GenerateReturnsFailureOnCreatedContextStubEra

- **Given** Executor + Context already created
- **When** `CANN_LMEngine_ExecutorGenerate(exec, "hello", id, &params, &result)`
- **Then** returns non-SUCCESS (stub FAILURE)
- **Then** caller needs `CANN_LMEngine_ResultDestroy(&result)` to release

#### UT-TC-CAPI-FB-001 CreateFromBufferSucceedsStubEra (**new NEW-12**)

- **Given** construct `CANN_LMEngine_Buffer buf{data, size}` (`char buf_data[8]` stack buffer); `CANN_LMEngine_ModelArtifact ma{buf, buf, buf}`; `CANN_LMEngine_EngineOptions opts{}`
- **When** `CANN_LMEngine_ExecutorCreateFromBuffer(&ma, &ma, &opts, &exec)`
- **Then** returns `CANN_LM_ENGINE_SUCCESS`; `exec` non-NULL
- **Then** subsequently call `CANN_LMEngine_ExecutorDestroy(exec)` to release (no leak)

#### UT-TC-CAPI-FB-002 CreateFromBufferNullPrefillReturnsInvalidArgument (**new NEW-13**)

- **Given** `CANN_LMEngine_ModelArtifact ma{...}`; `opts`; `exec`
- **When** `CANN_LMEngine_ExecutorCreateFromBuffer(nullptr, &ma, &opts, &exec)`
- **Then** returns `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-003 CreateFromBufferNullDecodeReturnsInvalidArgument (**new NEW-14**)

- Same as UT-TC-CAPI-FB-002, but `prefill=&ma, decode=nullptr`
- **Then** returns `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-004 CreateFromBufferNullOutReturnsInvalidArgument (**new NEW-15**)

- Same as above, but `out=nullptr`
- **Then** returns `CANN_LM_ENGINE_INVALID_ARGUMENT`

#### UT-TC-CAPI-FB-005 CreateFromBufferSupportsContextCreate (**new NEW-16**)

- **Given** `ExecutorCreateFromBuffer` returns SUCCESS (UT-TC-CAPI-FB-001 same setup)
- **When** `CANN_LMEngine_ContextCreate(exec, &id)`
- **Then** returns `CANN_LM_ENGINE_SUCCESS`; `id != 0` — verifies Executor constructed via buffer entry has internal `pipeline_` truly ready (complementary to `executor_buffer_ctor_returns_failure_when_factory_fails` behavior)
- **Then** `CANN_LMEngine_ContextDestroy(exec, id)` + `ExecutorDestroy(exec)`

### 4.4 ut_scheduler (UT-TC-SCHED-*)

#### UT-TC-SCHED-001 StepWithNoWorkReturnsInvalidState

- **Given** `SchedulerFixture f` (prefill_exec/decode_exec both `nullptr`; `max_batch_size=1`)
- **When** `sched.Step(&out)`
- **Then** returns `Status::INVALID_STATE` (no active tasks)
- **Assertion**: `EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-SCHED-002 DecodeCapacityMatchesMaxBatchSize

- **Given** `SchedulerFixture f`, `max_batch_size=4`
- **Then** `sched.DecodeCapacity() == 4`; `ActiveDecodeCount() == 0`; `PendingPrefillCount() == 0`

#### UT-TC-SCHED-003 ConstructWithBatchSize1

- Same as UT-TC-SCHED-002 but `max_batch_size=1`, verify `DecodeCapacity() == 1`

#### UT-TC-SCHED-101 SubmitWithNullReqReturnsInvalidArgument (**new NEW-17**)

- **Given** `SchedulerFixture f`
- **When** `sched.Submit({}, nullptr, &out)` (req=nullptr)
- **Then** returns `Status::INVALID_ARGUMENT` (impl `scheduler.cpp:59`)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SCHED-102 SubmitWithNullOutReturnsInvalidArgument (**new NEW-18**)

- Same as above but `req` valid, `out=nullptr`
- **Then** returns `Status::INVALID_ARGUMENT`
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SCHED-004 SubmitSucceedsIncrementsPendingCount (**new NEW-19**)

- **Given** `SchedulerFixture f`, `max_batch_size=2`
- **When** `sched.Submit({}, std::make_shared<RequestImpl>(), &id)` (empty chunks legal)
- **Then** returns `Status::SUCCESS`; `id.value != 0`
- **Then** `sched.PendingPrefillCount() == 1` (impl `scheduler.cpp:70` after push_back)
- **Assertion**: `EXPECT_TRUE(s.ok())` + `EXPECT_NE(id.value, 0u)` + `EXPECT_EQ(sched.PendingPrefillCount(), 1)`

#### UT-TC-SCHED-005 SubmitAssignsMonotonicRequestIds (**new NEW-20**)

- **Given** same as above
- **When** consecutive two `Submit`, record `id1` and `id2`
- **Then** `id1.value != 0`; `id2.value != 0`; `id2.value > id1.value` (`next_id_` monotonically increasing)
- **Assertion**: `EXPECT_GT(id2.value, id1.value)`

#### UT-TC-SCHED-103 CancelNonexistentReturnsNotFound (**new NEW-21**)

- **Given** `SchedulerFixture f`, no tasks
- **When** `sched.Cancel(RequestId{42})`
- **Then** returns `Status::NOT_FOUND` (impl `scheduler.cpp:251` path not matched)
- **Assertion**: `EXPECT_EQ(s.code, Status::NOT_FOUND)`

#### UT-TC-SCHED-104 CancelContextWithNoMatchingContextsReturnsSuccess (**new NEW-22**)

- **Given** `SchedulerFixture f`, no tasks
- **When** `sched.CancelContext(ContextId{42})`
- **Then** returns `Status::SUCCESS` (impl `scheduler.cpp:294` always returns SUCCESS)
- **Assertion**: `EXPECT_TRUE(s.ok())`

### 4.5 ut_sampler (UT-TC-SAMPLER-*) ——**new file**

#### UT-TC-SAMPLER-001 SampleWithNullOutReturnsInvalidArgument_TC201 (**new NEW-1**)

- **Given** `Sampler sampler`; empty Tensor `Tensor t_empty`; `SamplingParams params`
- **When** `sampler.Sample(t_empty, 0, params, nullptr)` (out=nullptr)
- **Then** returns `Status::INVALID_ARGUMENT` (impl `sampler.cpp` param guard)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SAMPLER-002 SampleWithNegativeSlotReturnsInvalidArgument_TC202 (**new NEW-2**)

- **Given** `Sampler sampler`; `TokenId out`
- **When** `sampler.Sample(t_empty, -1, params, &out)` (slot<0)
- **Then** returns `Status::INVALID_ARGUMENT` (impl `sampler.cpp` param guard)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`

#### UT-TC-SAMPLER-003 SampleWithNullDataReturnsFailure_TC203 (**new NEW-3**; 2026-09-07 renamed, originally SampleWithEmptyLogitsReturnsFailureStubEra)

- **Given** `Sampler sampler`; Tensor with GetData()==nullptr; `TokenId out`
- **When** `sampler.Sample(t_empty, 0, params, &out)`
- **Then** returns `Status::FAILURE` (logits missing defense branch)
- **Assertion**: `EXPECT_EQ(s.code, Status::FAILURE)`

#### UT-TC-SAMPLER-004 SampleWithDefaultsReturnsGreedyArgmax_TC204 (**new NEW-4**; 2026-09-07 rewritten, originally SampleWithValidLogitsReturnsFailureStubEra — algorithm Real Era no longer FAILURE)

- **Given** `Sampler sampler`; FLOAT32 logits `{1.0, 3.0, 2.0, 0.5}` (dims `[4]`); default `SamplingParams`
- **When** `sampler.Sample(t_valid, 0, params, &out)`
- **Then** returns `Status::SUCCESS`, `out == 1` (greedy argmax; default params temp=1/top_k=0/top_p=1 → greedy)
- **Assertion**: `EXPECT_EQ(s.code, Status::SUCCESS)` + `EXPECT_EQ(out, 1u)`

#### UT-TC-SAMPLER-005 SampleBatchedDimsPicksPerSlotRow_TC205 (**2026-09-07 new**)

- **Given** FLOAT32 logits dims `[2,3]` (two rows argmax are 1, 0 respectively)
- **When** `sampler.Sample(t, 0/1/2, params, &out)`
- **Then** slot=0 → `out==1`; slot=1 → `out==0`; slot>=2 (out of bounds) → `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-006 SampleWith1DDimsRejectsSlotBeyondRow_TC206 (**2026-09-07 new**)

- **Given** dims `[4]` (single-row capacity)
- **When** `sampler.Sample(t, 1, params, &out)`
- **Then** `INVALID_ARGUMENT` (slot exceeds single-row capacity)

#### UT-TC-SAMPLER-007 SampleWithNonFloat32ReturnsInvalidArgument_TC207 (**2026-09-07 new**)

- **Given** dataType=INT32 logits
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT` (only FLOAT32 supported, see sampler.h known limitation)

#### UT-TC-SAMPLER-008 SampleWithMisalignedBytesReturnsInvalidArgument_TC208 (**2026-09-07 new**)

- **Given** FLOAT32 buffer with size=6 bytes (6 % 4 != 0)
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-009 SampleWithZeroSizeReturnsInvalidArgument_TC209 (**2026-09-07 new**)

- **Given** GetData() non-null but size==0
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT`

#### UT-TC-SAMPLER-010 SampleWithoutDimsAndSlotPositiveReturnsFailure_TC210 (**2026-09-07 new**)

- **Given** dims missing (common form of host fallback buffer), slot=1
- **When** `sampler.Sample(t, 1, params, &out)`
- **Then** `FAILURE` (no dims can't safely derive row width)

#### UT-TC-SAMPLER-011 SampleWithTopKAndZeroTemperatureStaysGreedy_TC211 (**2026-09-07 new**)

- **Given** logits `{1.0, 5.0, 4.9, 0.0}`; `top_k=2`, `temperature=0.0`
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `SUCCESS` and `out==1` (temp=0 via 1e-8 clamp degenerates to greedy, deterministic)

#### UT-TC-SAMPLER-012 SampleWithTopPWiresNucleusSampling_TC212 (**2026-09-07 new**)

- **Given** logits `{1.0, 5.0, 4.9, 0.0}`; `top_p=0.9`
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `SUCCESS` and `out ∈ {1, 2}` (nucleus sampling truncated to top-2; probabilistic only asserts set)

#### UT-TC-SAMPLER-013 SampleWithDescBufferMismatchReturnsInvalidArgument_TC213 (**2026-09-07 new**)

- **Given** dims `[2,4]` but buffer only 2 elements (desc and actual allocation disconnected)
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT` (path where real backend desc inaccuracy first exposed)

#### UT-TC-SAMPLER-014 SampleWithNonPositiveDimsReturnsInvalidArgument_TC214 (**2026-09-07 new**)

- **Given** dims contain non-positive dimension (`{0}` or `{2,-3,4}`)
- **When** `sampler.Sample(t, 0, params, &out)`
- **Then** `INVALID_ARGUMENT` (last dim/prefix dim non-positive both rejected)

#### Algorithm layer direct tests (`SamplerAlgoTest`, 2026-09-07 new)

Static pure function direct tests (not through ITensorBuffer): `ApplyTemperature` (division/non-positive clamp), `ApplyTopK`
(top-k keep rest -inf / k out of bounds no-op), `ApplyPrecomputedTopK` (outside set -inf), `ApplyTopP`
(highest probability item must keep), `ApplyRepetitionPenalty` (score<0 ×p, score≥0 ÷p, not generated untouched),
`Normalize` (non-negative, sum≈1, probability order follows logit order), `Pick` (greedy argmax / seeded reproducible),
full pipeline (default greedy / seeded sampling reproducible). Expected behavior also anchored by
`cann_lm/engine/core/sample/test/sampler_ut.cc` (100 assertions, scripts/build_sample.sh)
on reference implementation side, two implementations must not drift unilaterally.

### 4.6 ut_request_handle (UT-TC-RH-*) ——**new file**

#### UT-TC-RH-001 EmptyHandleGetStatusReturnsQueued (**new NEW-5**)

- **Given** `RequestHandle handle;` (default construction, `impl_` empty)
- **When** `handle.GetStatus()`
- **Then** returns `RequestStatus::Queued` (impl `request_handle.cpp:17`)
- **Assertion**: `EXPECT_EQ(handle.GetStatus(), RequestStatus::Queued)`

#### UT-TC-RH-002 EmptyHandleIsDoneReturnsFalse (**new NEW-6**)

- **Given** empty handle
- **When** `handle.IsDone()`
- **Then** returns false (impl `request_handle.cpp:22`)
- **Assertion**: `EXPECT_FALSE(handle.IsDone())`

#### UT-TC-RH-003 EmptyHandleBoolConversionReturnsFalse (**new NEW-7**)

- **Given** empty handle
- **When** `static_cast<bool>(handle)` or `if (handle)`
- **Then** returns false (impl `request_handle.cpp:103`)
- **Assertion**: `EXPECT_FALSE(static_cast<bool>(handle))`

#### UT-TC-RH-004 EmptyHandleCancelReturnsSuccess (**new NEW-8**)

- **Given** empty handle
- **When** `handle.Cancel()`
- **Then** returns `Status::SUCCESS` (impl `request_handle.cpp:34`, empty handle idempotent)
- **Assertion**: `EXPECT_TRUE(handle.Cancel().ok())`

#### UT-TC-RH-005 EmptyHandleWaitReturnsInvalidState (**new NEW-9**)

- **Given** empty handle; `Result out`
- **When** `handle.Wait(&out)`
- **Then** returns `Status::INVALID_STATE` (impl `request_handle.cpp:41`, empty handle doesn't block returns immediately)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-RH-006 EmptyHandleTryGetResultReturnsInvalidState (**new NEW-10**)

- **Given** empty handle; `Result out`
- **When** `handle.TryGetResult(&out)`
- **Then** returns `Status::INVALID_STATE` (impl `request_handle.cpp:82`)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_STATE)`

#### UT-TC-RH-007 EmptyHandleWaitWithNullOutReturnsInvalidArgument (**new NEW-11**)

- **Given** empty handle
- **When** `handle.Wait(nullptr)` (out=nullptr)
- **Then** returns `Status::INVALID_ARGUMENT` (impl `request_handle.cpp:40`, param validation before empty handle check)
- **Assertion**: `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)`
- **Note**: param validation order: first `!out` then `!impl_`. So passing nullptr even if impl_ also empty goes to INVALID_ARGUMENT branch.
- **TryGetResult same**: `handle.TryGetResult(nullptr)` returns `INVALID_ARGUMENT`

## Chapter 5 Non-Functional Requirements

| Dimension | Metric | Test Condition | Evidence Source |
|------|------|----------|----------|
| Single UT file compile time | < 5s (incremental) | aarch64 + clang/gcc, O0 Debug | existing build baseline |
| Full UT run time | < 10s | CTest all discovered cases | existing bbit + 5 UT binaries |
| UT count growth | this phase +22 cases | total cases after new ≈ 88 (original 66 + new 22) | DESIGN §9 |
| Memory | no leak | ASAN/LSAN clean | implementation §8 test hooks |
| Thread safety | bbit and worker thread interaction no races | TSan clean | deferred (this phase doesn't run TSan) |

## Chapter 6 Constraints and Dependencies

- **Code under test**: `cann_lm/engine/*` (44 cpp/h files)
- **Test framework**: GoogleTest 1.14 (`/usr/src/googletest/`, CMake `find_package(GTest REQUIRED)`)
- **Compiler**: g++ 13 / clang-18 / C++17 (consistent with production code)
- **Build system**: CMake ≥ 3.10 (consistent with `cann_lm/engine/CMakeLists.txt`)
- **Dependent module SPEC**: tokenizer / c_api / pipeline+engine_executor / model_executor (each module §8 TC numbering aligned)

## Chapter 7 Verification Standards

| Category | Covered TC (by §4 numbering) |
|------|----------------------|
| Normal path (0xx) | EE-001/002/005/006/011; TK-002/004/005; CAPI-007/008; CAPI-FB-001/002/005; SCHED-002/003/004/005/104; RH-001/002/003/004 |
| Exception path (1xx) | EE-003/004/007/008/009/010; TK-001/003; CAPI-001/002/006/009; CAPI-FB-002/003/004; SCHED-101/102/103; SAMPLER-001/002/003/004; RH-005/006/007 |
| Boundary condition (2xx) | TK-001 (continue-on-failure); SAMPLER-003/004 (empty/non-empty Tensor); RH-001/002/003 (empty handle) |
| Concurrency scenario (3xx) | EE-011 (same ctx consecutive GenerateAsync) |

**Self-check**: 4 categories minimum coverage dimension each ≥1 case; this phase new 22 cases + fix 2 cases, all Stub Era behavior verifiable. Real Era inference path deferred.

## Chapter 8 Drift Fix and New Case Specification

### 8.1 Drift Fix (DRIFT-1, DRIFT-2)

See DESIGN §8 for details. Fix only changes case name + adjusts assertion, doesn't change tested implementation behavior.

| ID | Old Name → New Name | Old Assertion → New Assertion | Related UT-TC |
|------|-------------|-----------------|------------|
| DRIFT-1 | `ConcurrentGenerateSameContextReturnsInvalidState` → `ConcurrentGenerateSameContextReturnsFailureStubEra` | two `EXPECT_FALSE(s.ok())` → two `EXPECT_FALSE(s.ok())` (assertion unchanged, only rename) | UT-TC-EE-011 |
| DRIFT-2 | `CreateUnknownReturnsFailure` → `CreateBPEReturnsInvalidArgumentStubEra` | `EXPECT_FALSE(s.ok())` → `EXPECT_FALSE(s.ok())` + `EXPECT_EQ(s.code, Status::INVALID_ARGUMENT)` (add code verification) | UT-TC-TK-005 |

### 8.2 New Cases (NEW-1..NEW-22)

See DESIGN §9 and §4 each UT-TC entry for details. Each new case includes: Given / When / Then / Assertion / Related SPEC TC.

### 8.3 Coverage Dimension Verification

- **Stub era observable behavior**: all 22 new + 2 fixed cover EE/RH/CAPI/SAMPLER/TK/SCHED 6 modules' external interface surface and exposed internal behavior gaps ✓
- **Not verified**: Real Era inference path (Generate success, Tokenizer real encode/decode) — deferred until real backend integration (Sampler algorithm-level real sampling landed 2026-09-07, see §4.5)
- **Uncovered modules**: `text_embedding` / `pipeline_executor` / `preprocessor` / `history_prompt_manager` / `infra/logger` / `core/context` / `core/tensor_manager` / `request_handle` real lifecycle (non-empty impl_) — see DESIGN §10 D-05

## Chapter 9 Open Items Register

| ID | Open Item | Category | Blocks spec release | Resolution Phase |
|----|--------|------|----------------|----------|
| UT-U-1 | `Algorithm` enum only has BPE/SP two values, stub era can't construct truly "unknown" enum | system boundary | no (this phase renamed `CreateBPEReturnsInvalidArgumentStubEra`, doesn't verify unknown path) | Real Era: extend Algorithm enum or inject unknown value via test hook then add `CreateUnknownAlgorithmReturnsInvalidArgument_RealEra` |
| UT-U-2 | `ExecutorCreateFromBuffer` real success path depends on `PipelineExecutorFactory::create(buffer)` not returning error | interface contract | no (UT-TC-CAPI-FB-005 indirectly verifies pipeline_ ready via `ContextCreate`) | Real Era: add `GenerateFromBufferExecutorSucceeds_RealEra` |
| UT-U-3 | bbit runs full FAILURE path in stub era, whether need to add stub-era expectations for bbit | integration test | no (bbit itself aligned) | later: bbit audit |
| UT-U-4 | real RequestHandle lifecycle (impl_ non-empty → Wait blocks → terminal semantics) test | interface contract | no (this phase only covers empty handle) | Real Era: construct non-empty RequestImpl, test Wait/Cancel/TryGetResult after Scheduler::Submit |

**Blocking-level open items cleared**: all 4 items don't block this phase UT landing.

## Chapter 10 Consistency Self-Check (Generation Phase)

- Naming convention consistency: all cases follow `<Module>Test, <Behavior>_<TCXXX>` + Stub Era suffix ✓
- Numbering consistency: UT-TC-EE-* / UT-TC-TK-* / UT-TC-CAPI-* / UT-TC-CAPI-FB-* / UT-TC-SCHED-* / UT-TC-SAMPLER-* / UT-TC-RH-* each independent numbering space, aligned with DESIGN §9 table ✓
- Drift fix consistency: DRIFT-1 / DRIFT-2 appear in both §4.1 / §4.2 and §8.1 table with consistent behavior description ✓
- New case consistency: 22 new cases aligned in §4.x and DESIGN §9 table ✓
- Stub/Real classification consistency: all Stub Era cases explicitly marked `_StubEra` suffix or prefixed with "Stub Era" semantics; Real Era marked deferred ✓
- Coverage dimension self-check: 4 categories (normal/exception/boundary/concurrency) each ≥1 case ✓
