#include "pipeline/scheduler.h"
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
    BPETokenizer tok{TokenizerConfig{}, nullptr};  // stub 时代：空产物 not-ready（SR20260903001 D-14）
    TextEmbedding te;
    Preprocessor pp{&tok, &te, 1024};
};

TEST(SchedulerTest, StepWithNoWorkReturnsInvalidState) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 1);

    StepOutput out;
    auto s = sched.Step(&out);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

TEST(SchedulerTest, DecodeCapacityMatchesMaxBatchSize) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 4);

    EXPECT_EQ(sched.DecodeCapacity(), 4);
    EXPECT_EQ(sched.ActiveDecodeCount(), 0);
    EXPECT_EQ(sched.PendingPrefillCount(), 0);
}

TEST(SchedulerTest, ConstructWithBatchSize1) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 1);

    EXPECT_EQ(sched.DecodeCapacity(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// UT-TC-SCHED-* : Submit / Cancel / CancelContext 覆盖（NEW-17..NEW-22）
// 关联 docs/ut/SPEC.md §4.4 / docs/ut/DESIGN.md §9
// ═══════════════════════════════════════════════════════════════════════════

// UT-TC-SCHED-101 (NEW-17): req=nullptr → INVALID_ARGUMENT（scheduler.cpp:59）
TEST(SchedulerTest, SubmitWithNullReqReturnsInvalidArgument_TC101) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    RequestId out;
    auto s = sched.Submit({}, /*req=*/nullptr, &out);
    EXPECT_EQ(s.code, Status::INVALID_ARGUMENT);
}

// UT-TC-SCHED-102 (NEW-18): out=nullptr → INVALID_ARGUMENT（scheduler.cpp:59）
TEST(SchedulerTest, SubmitWithNullOutReturnsInvalidArgument_TC102) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto req = std::make_shared<RequestImpl>();
    auto s = sched.Submit({}, req, /*out=*/nullptr);
    EXPECT_EQ(s.code, Status::INVALID_ARGUMENT);
}

// UT-TC-SCHED-004 (NEW-19): 合法 Submit → SUCCESS + PendingPrefillCount=1
TEST(SchedulerTest, SubmitSucceedsIncrementsPendingCount_TC001) {
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
TEST(SchedulerTest, SubmitAssignsMonotonicRequestIds_TC002) {
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
TEST(SchedulerTest, CancelNonexistentReturnsNotFound_TC103) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto s = sched.Cancel(RequestId{42});
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

// UT-TC-SCHED-104 (NEW-22): CancelContext 无匹配 → SUCCESS（scheduler.cpp:294 始终返回 SUCCESS）
TEST(SchedulerTest, CancelContextWithNoMatchingContextsReturnsSuccess_TC104) {
    SchedulerFixture f;
    Scheduler sched(&f.prefill_exec, &f.decode_exec, &f.pp, /*text_embedding=*/nullptr, 2);
    auto s = sched.CancelContext(ContextId{42});
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code, Status::SUCCESS);
}

}  // namespace
}  // namespace cann_lm::engine
