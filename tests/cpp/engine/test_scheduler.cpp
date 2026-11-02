#include "pipeline/scheduler.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/model_executor/model_executor.h"
#include "pipeline/preprocessor.h"
#include "core/component/text_embedding/text_embedding.h"
#include "core/component/tokenizer/tokenizer.h"

#include <gtest/gtest.h>
#include <memory>

namespace cann_lm::engine {
namespace {

struct SchedulerFixture {
    ModelExecutor prefill_exec{nullptr, nullptr, nullptr};
    ModelExecutor decode_exec{nullptr, nullptr, nullptr};
    BPETokenizer tok{TokenizerConfig{}, nullptr}; // stub 时代：空产物 not-ready（SR20260903001 D-14）
    TextEmbedding te;
    Preprocessor pp{&tok, &te, 1024};
};

TEST(SchedulerTest, StepWithNoWorkReturnsInvalidState)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 1);

    StepOutput out;
    auto s = sched.Step(&out);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

TEST(SchedulerTest, DecodeCapacityMatchesMaxBatchSize)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 4);

    EXPECT_EQ(sched.DecodeCapacity(), 4);
    EXPECT_EQ(sched.ActiveDecodeCount(), 0);
    EXPECT_EQ(sched.PendingPrefillCount(), 0);
}

TEST(SchedulerTest, ConstructWithBatchSize1)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 1);

    EXPECT_EQ(sched.DecodeCapacity(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// UT-TC-SCHED-* : Submit / Cancel / CancelContext 覆盖（NEW-17..NEW-22）
// 关联 docs/ut/SPEC.md §4.4 / docs/ut/DESIGN.md §9
// ═══════════════════════════════════════════════════════════════════════════

// UT-TC-SCHED-101 (NEW-17): req=nullptr → INVALID_ARGUMENT（scheduler.cpp:59）
TEST(SchedulerTest, SubmitWithNullReqReturnsInvalidArgument_TC101)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    RequestId out;
    auto s = sched.Submit({}, /*req=*/nullptr, &out);
    EXPECT_EQ(s.code, Status::INVALID_ARGUMENT);
}

// UT-TC-SCHED-102 (NEW-18): out=nullptr → INVALID_ARGUMENT（scheduler.cpp:59）
TEST(SchedulerTest, SubmitWithNullOutReturnsInvalidArgument_TC102)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto req = std::make_shared<RequestImpl>();
    auto s = sched.Submit({}, req, /*out=*/nullptr);
    EXPECT_EQ(s.code, Status::INVALID_ARGUMENT);
}

// UT-TC-SCHED-004 (NEW-19): 合法 Submit → SUCCESS + PendingPrefillCount=1
TEST(SchedulerTest, SubmitSucceedsIncrementsPendingCount_TC001)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto req = std::make_shared<RequestImpl>();
    RequestId id;
    auto s = sched.Submit({}, req, &id);
    EXPECT_TRUE(s.ok());
    EXPECT_NE(id.value, 0u);
    EXPECT_EQ(sched.PendingPrefillCount(), 1);
}

// UT-TC-SCHED-005 (NEW-20): 连续 Submit → RequestId 单调递增
TEST(SchedulerTest, SubmitAssignsMonotonicRequestIds_TC002)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto req = std::make_shared<RequestImpl>();
    RequestId id1, id2;
    ASSERT_TRUE(sched.Submit({}, req, &id1).ok());
    ASSERT_TRUE(sched.Submit({}, req, &id2).ok());
    EXPECT_NE(id1.value, 0u);
    EXPECT_NE(id2.value, 0u);
    EXPECT_GT(id2.value, id1.value);
}

// UT-TC-SCHED-103 (NEW-21): Cancel 未找到 → NOT_FOUND（scheduler.cpp:251）
TEST(SchedulerTest, CancelNonexistentReturnsNotFound_TC103)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto s = sched.Cancel(RequestId{42});
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

// UT-TC-SCHED-104 (NEW-22): CancelContext 无匹配 → SUCCESS（scheduler.cpp:294 始终返回 SUCCESS）
TEST(SchedulerTest, CancelContextWithNoMatchingContextsReturnsSuccess_TC104)
{
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto s = sched.CancelContext(ContextId{42});
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code, Status::SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════════════
// UT-TC-SCHED-201/TC_903 (PromptCache 命中工位直用修复，2026-10-21)：
// req->kv_slot_prefilled 预绑定随 task 入队（scheduler.cpp Submit 路径）；
// host 无图形态 executor 恒 not-ready（model_executor.cpp:1073-1075
// operator bool = mm_ && pool_ && asm_，host 未 Load → mm_ 空），Step 走
// "prefill executor not ready" 分支（scheduler.cpp:388-396）：SetTerminal
// (Failed) + RollbackCache_ + 出队 + return FAILURE。就绪分支
// （kv_slot<0 才 AllocKvSlot、:167 回写 kv_slot_actual）host 不可达，
// 本组断言选定 not-ready 路径上的确定性可观测面：
//   1) 预绑定值不触池记账（若误走 AllocKvSlot 复用/驱逐路径会把
//      total_written 清零——本用例的池预写账即回归捕获器）；
//   2) not-ready 不虚假回写 kv_slot_actual（回写点 :167 在就绪分支内）；
//   3) 请求终态 Failed、出队（PendingPrefillCount 归 0）。
// ═══════════════════════════════════════════════════════════════════════════

// UT-TC-SCHED-201 (TC_903): 预绑定工位请求（kv_slot_prefilled=0）+ 真实池
// 预写账 6（模拟 ResumeSlot 恢复后的记账——GenerateAsync 命中链产物）→
// Submit → Step(not-ready FAILURE)。断言：请求 Failed 出队、池槽 0 记账
// 6 不被清、kv_slot_actual 保持 -1（not-ready 不虚假回写）。
TEST(SchedulerTest, PrefilledSlotTaskNotReadyKeepsPoolLedger_TC903)
{
    // 真实池 + 预写账：槽 0 active、total_written=6（模拟上一轮命中链
    // ResumeSlot(0, 6) 的池侧状态；alloc_seq 顺序无关紧要——池满驱逐/
    // 复用清零路径正是本修复要绕开的对象）
    auto pool = std::make_shared<KvCachePool>(/*max_batch=*/2, /*kv_max=*/16,
                                              /*layers=*/2, /*kv_heads=*/4,
                                              /*head_dims=*/8);
    SlotId warmup = -1;
    ASSERT_EQ(pool->AllocKvSlot(&warmup).code, Status::SUCCESS);
    ASSERT_EQ(pool->Advance(warmup, 6).code, Status::SUCCESS);
    ASSERT_EQ(pool->TotalWritten(warmup), 6u) << "前置：槽预写账 6";

    // prefill executor 带池构造（pool_ 非空、asm_ 非空、mm_ 空 → not-ready）
    ModelExecutor prefill_exec(pool, std::make_shared<InputAssembler>(pool, 1),
                               /*dfx=*/nullptr);
    ASSERT_FALSE(static_cast<bool>(prefill_exec)) << "host 未 Load 恒 not-ready";

    BPETokenizer tok{TokenizerConfig{}, nullptr};
    TextEmbedding te;
    Preprocessor pp{&tok, &te, 1024};
    Scheduler sched(&prefill_exec, /*decode_exec=*/&prefill_exec, &pp,
                    /*text_embedding=*/nullptr, 2);

    // 预绑定工位 0 的命中请求（GenerateAsync 命中链：ResumeSlot 已恢复
    // 记账、kv_slot_prefilled=台账登记工位，pipeline_executor.cpp:1382）
    auto req = std::make_shared<RequestImpl>();
    req->kv_slot_prefilled = 0;
    req->prompt_ids = {1, 2, 3};
    RequestId id;
    ASSERT_TRUE(sched.Submit({{7, 8}, {9}}, req, &id).ok()) << "chunks 任意非空";
    EXPECT_EQ(sched.PendingPrefillCount(), 1);

    StepOutput out;
    auto s = sched.Step(&out);
    EXPECT_EQ(s.code, Status::FAILURE) << "not-ready 分支固定 FAILURE";

    // 请求终态 Failed + 出队 + not-ready 不虚假回写工位
    EXPECT_EQ(req->status.load(), RequestStatus::Failed);
    EXPECT_EQ(req->finish_reason, FinishReason::Error);
    EXPECT_EQ(sched.PendingPrefillCount(), 0) << "失败任务已出队";
    EXPECT_EQ(req->kv_slot_actual, -1) << "not-ready 分支不回写实际工位（回写点在就绪分支内）";

    // 核心回归锚：预绑定工位的池记账未被清零/清零重分配（AllocKvSlot
    // 复用/驱逐路径清 total_written 的回归在此捕获）。
    EXPECT_EQ(pool->TotalWritten(0), 6u) << "槽 0 记账保持（预绑定未触发 Alloc 清零）";
    EXPECT_EQ(pool->NextPosition(0), 6u);
}

// UT-TC-SCHED-202 (TC_904): 对照组——非预绑定请求（kv_slot_prefilled=-1，
// miss/普通请求路径）同条件 not-ready：同样 Failed 出队 FAILURE、池记账
// 不动（miss 链上 ResetLedgerOnSlot 只发生在 GenerateAsync miss 分支，
// Step 不碰池）。锁定"两条路径在 not-ready 形态下行为一致"基线，供
// TC_903 差分比对（预绑定值本身不改变 not-ready 路径行为）。
TEST(SchedulerTest, NonPrefilledTaskNotReadySameFailurePath_TC904)
{
    auto pool = std::make_shared<KvCachePool>(2, 16, 2, 4, 8);
    SlotId warmup = -1;
    ASSERT_EQ(pool->AllocKvSlot(&warmup).code, Status::SUCCESS);
    ASSERT_EQ(pool->Advance(warmup, 6).code, Status::SUCCESS);

    ModelExecutor prefill_exec(pool, std::make_shared<InputAssembler>(pool, 1), nullptr);
    ASSERT_FALSE(static_cast<bool>(prefill_exec));

    BPETokenizer tok{TokenizerConfig{}, nullptr};
    TextEmbedding te;
    Preprocessor pp{&tok, &te, 1024};
    Scheduler sched(&prefill_exec, &prefill_exec, &pp, nullptr, 2);

    auto req = std::make_shared<RequestImpl>(); // kv_slot_prefilled 缺省 -1
    ASSERT_EQ(req->kv_slot_prefilled, -1) << "缺省非绑定";
    RequestId id;
    ASSERT_TRUE(sched.Submit({{7}}, req, &id).ok());

    StepOutput out;
    EXPECT_EQ(sched.Step(&out).code, Status::FAILURE);
    EXPECT_EQ(req->status.load(), RequestStatus::Failed);
    EXPECT_EQ(sched.PendingPrefillCount(), 0);
    EXPECT_EQ(req->kv_slot_actual, -1);
    EXPECT_EQ(pool->TotalWritten(0), 6u) << "对照组池记账同样不受 Step 干扰";
}

// ═══════════════════════════════════════════════════════════════════════════
// UT-TC-SCHED-203 (TC_905): 空块链 Step 防御收口（DTS202610220945 Fix B）。
// Submit 契约保留空链入队（TC001/TC002 基线锁空块 SUCCESS）；host 形态
// executor 恒 not-ready（mm_ 空）→ Step 命中 not-ready 分支（scheduler.cpp
// "prefill executor not ready"）而非就绪分支空块守卫（守卫在就绪分支头部，
// host 不可达——Fix A 后命中链恒非空块，守卫真机/实机验证面，见
// DTS202610220945 §7 V-D）。本用例锁定 not-ready 路径对空块任务的可达
// 收口：req Failed 出队、池记账不动、不虚假回写工位——与 TC903/TC904
// 差分对照（空块本身不改变 not-ready 行为，防御纵深在两侧都闭环）。
// ═══════════════════════════════════════════════════════════════════════════
TEST(SchedulerTest, EmptyChunksTaskNotReadyFailsAndDrains_TC905)
{
    auto pool = std::make_shared<KvCachePool>(2, 16, 2, 4, 8);
    SlotId warmup = -1;
    ASSERT_EQ(pool->AllocKvSlot(&warmup).code, Status::SUCCESS);
    ASSERT_EQ(pool->Advance(warmup, 6).code, Status::SUCCESS);

    ModelExecutor prefill_exec(pool, std::make_shared<InputAssembler>(pool, 1), nullptr);
    ASSERT_FALSE(static_cast<bool>(prefill_exec)) << "host 未 Load 恒 not-ready";

    BPETokenizer tok{TokenizerConfig{}, nullptr};
    TextEmbedding te;
    Preprocessor pp{&tok, &te, 1024};
    Scheduler sched(&prefill_exec, &prefill_exec, &pp, nullptr, 2);

    // 空块链（zero-token prompt / 上游回归形态）：Submit 保留 SUCCESS 入队
    auto req = std::make_shared<RequestImpl>();
    RequestId id;
    ASSERT_TRUE(sched.Submit({}, req, &id).ok()) << "空链入队 = 既有契约面";
    EXPECT_EQ(sched.PendingPrefillCount(), 1);

    StepOutput out;
    auto s = sched.Step(&out);
    EXPECT_EQ(s.code, Status::FAILURE) << "not-ready 分支固定 FAILURE";
    EXPECT_EQ(req->status.load(), RequestStatus::Failed);
    EXPECT_EQ(req->finish_reason, FinishReason::Error);
    EXPECT_EQ(sched.PendingPrefillCount(), 0) << "失败任务已出队";
    EXPECT_EQ(req->kv_slot_actual, -1) << "not-ready 不虚假回写工位";
    EXPECT_EQ(pool->TotalWritten(0), 6u) << "池记账不动";
    EXPECT_TRUE(out.slots.empty()) << "FAILURE 路径不填 slots（RunLoop 不消费）";
}

} // namespace
} // namespace cann_lm::engine
