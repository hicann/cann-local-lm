// RequestHandle 空状态单元测试。关联 docs/ut/SPEC.md §4.6 / docs/ut/DESIGN.md §9（NEW-5..NEW-11）。
// 全部覆盖默认构造的空句柄（impl_ == nullptr）下的状态查询与控制方法。
// 关联实现 api/cpp/request_handle.cpp:16-104。
#include "api/cpp/request_handle.h"
#include "api/cpp/types.h"
#include "infra/status.h"

#include <gtest/gtest.h>

namespace cann_local_lm::engine {
namespace {

// UT-TC-RH-001 (NEW-5): 空句柄 GetStatus 返回 Queued（impl_ 为空分支）
TEST(RequestHandleTest, EmptyHandleGetStatusReturnsQueued_TC001)
{
    RequestHandle handle;
    EXPECT_EQ(handle.GetStatus(), RequestStatus::Queued);
}

// UT-TC-RH-002 (NEW-6): 空句柄 IsDone 返回 false
TEST(RequestHandleTest, EmptyHandleIsDoneReturnsFalse_TC002)
{
    RequestHandle handle;
    EXPECT_FALSE(handle.IsDone());
}

// UT-TC-RH-003 (NEW-7): 空句柄 operator bool 返回 false
TEST(RequestHandleTest, EmptyHandleBoolConversionReturnsFalse_TC003)
{
    RequestHandle handle;
    EXPECT_FALSE(static_cast<bool>(handle));
    // 也覆盖 if (handle) 用法
    if (handle) {
        FAIL() << "空句柄不应进入 if (handle) 分支";
    }
}

// UT-TC-RH-004 (NEW-8): 空句柄 Cancel 返回 SUCCESS（幂等）
TEST(RequestHandleTest, EmptyHandleCancelReturnsSuccess_TC004)
{
    RequestHandle handle;
    auto s = handle.Cancel();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code, Status::SUCCESS);
}

// UT-TC-RH-005 (NEW-9): 空句柄 Wait(out) 返回 INVALID_STATE（不阻塞）
TEST(RequestHandleTest, EmptyHandleWaitReturnsInvalidState_TC101)
{
    RequestHandle handle;
    Result out;
    auto s = handle.Wait(&out);
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

// UT-TC-RH-006 (NEW-10): 空句柄 TryGetResult(out) 返回 INVALID_STATE
TEST(RequestHandleTest, EmptyHandleTryGetResultReturnsInvalidState_TC102)
{
    RequestHandle handle;
    Result out;
    auto s = handle.TryGetResult(&out);
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

// UT-TC-RH-007 (NEW-11): 参数校验先于空句柄检查——Wait(nullptr) / TryGetResult(nullptr)
// 即使 impl_ 为空，也返回 INVALID_ARGUMENT（impl request_handle.cpp:40 / 81）
TEST(RequestHandleTest, EmptyHandleWaitWithNullOutReturnsInvalidArgument_TC103)
{
    RequestHandle handle;
    auto s_wait = handle.Wait(nullptr);
    EXPECT_EQ(s_wait.code, Status::INVALID_ARGUMENT);
    auto s_try = handle.TryGetResult(nullptr);
    EXPECT_EQ(s_try.code, Status::INVALID_ARGUMENT);
}

} // namespace
} // namespace cann_local_lm::engine
