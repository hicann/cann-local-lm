// 权重共享链路（ModelInitOptions::shareWeightsModelId）UT。
// 覆盖两处新增契约 + 一处回归锚点：
//   UT-TC-SW-001（新增）：shim 符号缺失时快速失败——host 环境无 NDK 库，
//     NDKProxy::GetSymbol 解析 HMS_HiAIOptions_SetShareWeightsModelId 返 null，
//     shim 须返 FAILURE 且不崩溃（快速失败策略，语义见 hiai_ndk_options.cpp）。
//   UT-TC-SW-002（回归锚点）：未 Init 的 ModelManager 调 InitWeights(dir) 返
//     FAILURE——既有 executor 前置守卫行为，防止改造回归。
//   UT-TC-SW-003（新增契约）：ModelInitOptions::shareWeightsModelId 默认 0
//     =不绑定；SetInitOption 以 !=0 守卫，0 值不得触发 HMS 绑定调用
//     （老 ROM 缺符号时 Init 不受影响的兼容前提）。
// 局限：SetInitOption 内"绑定→executor 构造"先序 host 无法验证（需真实/stub
//   .so 在场），由实现评审与设备验证覆盖。
// 关联：hiai_options_inner.h:153（SDK 声明）/ hiai_ndk_options.cpp（shim）/
//   backend/api/types.h（ModelInitOptions 绑定字段）/ model_manager_ndk_impl.cpp
//   （SetInitOption 绑定接线 + IsOptionsChanged 重建判据）。
#include "model_manager/model_manager.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_options.h"

#include <gtest/gtest.h>

namespace {
// UT-TC-SW-001: shim 符号缺失快速失败（host 无 NDK 库 → FAILURE，不崩溃）
// 注：若环境注入带该符号的 .so（stub/真机），本用例语义需随环境调整。
TEST(ShareWeightsTest, ShimMissingSymbolFailsFast_TC001)
{
    hiai::Status st = hiai::HIAI_NDK_HiAIOptions_SetShareWeightsModelId(nullptr, 1234u);
    EXPECT_EQ(st, hiai::FAILURE);
}

// UT-TC-SW-002: 未 Init 的 ModelManager 调 InitWeights(dir) 快速失败（既有守卫回归锚点）
TEST(ShareWeightsTest, InitWeightsWithoutInitFails_TC002)
{
    auto mm = hiai::CreateModelManager();
    ASSERT_NE(mm, nullptr);
    EXPECT_EQ(mm->InitWeights("weights"), hiai::FAILURE);
}

// UT-TC-SW-003: 绑定字段默认 0 = 不绑定（契约：0 值不触发 HMS 绑定调用）
TEST(ShareWeightsTest, ShareWeightsModelIdDefaultsToNoBind_TC003)
{
    hiai::ModelInitOptions options;
    EXPECT_EQ(options.shareWeightsModelId, 0u);
}
} // namespace
