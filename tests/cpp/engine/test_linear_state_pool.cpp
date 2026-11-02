// LinearStatePool 单元测试（support_qwen3.5.md §4.4 / §7 状态池 UT 行）。
// 两层结构：
//  - 纯逻辑用例（层表派生 / 错误路径 / 空池行为）：不依赖 tensor 申请，host
//    （无 NDK so）无条件可跑；
//  - 创建类用例（Ensure 创建/复用/失配、ResetLane lane 切片清零）：tensor
//    申请经 TensorManager → 真后端；host 无 NDK so 时探测一次失败即
//    GTEST_SKIP（真机 / stub .so 旁 exe 时全量执行，见 PROJECT.md host UT 节）。
// 设计 §5 D4/D1（§8 Q3 已确认）：batch 维 = dims[0]（lane 步长 = GetSize()/dims[0]）；
// 双图一致性校验（dims/dtype 失配 INVALID_ARGUMENT）在 Ensure 处把关。
#include "core/component/linear_state_pool/linear_state_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cann_lm::engine {
namespace {

// ── 公共夹具 ─────────────────────────────────────────────────────────────────

TensorDesc MakeDesc(std::vector<int32_t> dims, DataType dt = DataType::FLOAT16, const std::string& name = "")
{
    TensorDesc d;
    d.dims = std::move(dims);
    d.dataType = dt;
    d.format = Format::ND;
    d.tensorName = name;
    return d;
}

/// host 可用性探测：CreateTensor 走真后端，无 NDK so → 失败（据此 SKIP）。
bool TensorCreateAvailable()
{
    TensorDesc d = MakeDesc({1, 1});
    d.tensorName = "lsp_ut_probe";
    std::shared_ptr<ITensorBuffer> t;
    return TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t).ok();
}

bool AllZero(const void* p, std::size_t n)
{
    const auto* b = static_cast<const uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        if (b[i] != 0)
            return false;
    }
    return true;
}

bool AllEqual(const void* p, std::size_t n, uint8_t v)
{
    const auto* b = static_cast<const uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        if (b[i] != v)
            return false;
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// 层逻辑（纯逻辑层，host 无条件跑）
// ═════════════════════════════════════════════════════════════════════════════

// 空 layer_types（非 Qwen3.5 / 未声明）→ 池空转：零 linear 层、零 tensor、
// getter 全空、ResetLane no-op、Ensure 一律 INVALID_ARGUMENT（零行为变化口径）
TEST(LinearStatePoolTest, EmptyLayerTypesYieldsInertPool)
{
    LinearStatePool pool({});
    EXPECT_EQ(pool.NumLinearLayers(), 0);
    EXPECT_TRUE(pool.LinearLayerIds().empty());
    EXPECT_EQ(pool.TensorCount(), 0u);
    EXPECT_FALSE(pool.ConvTensor(0));
    EXPECT_FALSE(pool.RTensor(0));
    EXPECT_TRUE(pool.ResetLane(0).ok()); // 池空 = no-op SUCCESS
    // 图带状态工位而配置未声明 → 显式报错不静默兼容（§4.4）
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({2, 4})).code,
              Status::INVALID_ARGUMENT);
}

// 全 full_attention → 同空表语义（无 linear 层）
TEST(LinearStatePoolTest, AllFullLayersYieldNoLinearLayers)
{
    LinearStatePool pool({"full_attention", "full_attention", "full_attention"});
    EXPECT_EQ(pool.NumLinearLayers(), 0);
    EXPECT_FALSE(pool.ConvTensor(1));
    EXPECT_FALSE(pool.RTensor(2));
    EXPECT_EQ(pool.EnsureStateTensor(2, LinearStatePool::StateKind::R, MakeDesc({2, 4})).code,
              Status::INVALID_ARGUMENT);
}

// linear 层号表构造期派生：升序、全局层号（§5 D2）
TEST(LinearStatePoolTest, LinearLayerIdsDerivedAscending)
{
    LinearStatePool pool(
        {"full_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention"});
    EXPECT_EQ(pool.NumLinearLayers(), 3);
    EXPECT_EQ(pool.LinearLayerIds(), (std::vector<int>{1, 3, 4}));
    // 未 Ensure 前 tensor 计数为 0（构造期只派生层清单）
    EXPECT_EQ(pool.TensorCount(), 0u);
}

// Ensure 到非 linear 层 / 越界层号 → INVALID_ARGUMENT（在 tensor 申请前拒绝，
// host 无后端亦可验证）
TEST(LinearStatePoolTest, EnsureOnNonLinearOrOutOfRangeLayerRejected)
{
    LinearStatePool pool({"full_attention", "linear_attention"});
    const auto desc = MakeDesc({2, 4, 8});
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, desc).code,
              Status::INVALID_ARGUMENT); // full 层
    EXPECT_EQ(pool.EnsureStateTensor(9, LinearStatePool::StateKind::R, desc).code,
              Status::INVALID_ARGUMENT); // 越界
    EXPECT_EQ(pool.EnsureStateTensor(-1, LinearStatePool::StateKind::Conv, desc).code, Status::INVALID_ARGUMENT);
}

// desc 非法（dims 空 / 含非正维 / dtype RESERVED）→ 创建前拒绝 INVALID_ARGUMENT
// （host 无后端不触及 CreateTensor）
TEST(LinearStatePoolTest, EnsureWithInvalidDescRejectedBeforeCreate)
{
    LinearStatePool pool({"linear_attention"});
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({})).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({2, 0, 8})).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({2, 4}, DataType::RESERVED)).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.TensorCount(), 0u); // 未产生任何 tensor
}

// ═════════════════════════════════════════════════════════════════════════════
// 创建 / 复用 / 清零（需真后端；host 无 NDK so → SKIP）
// 注：SKIP 判定必须写在 TEST 体内——本仓 gtest 为无异常模式，GTEST_SKIP 展开
//     为 `return`，从 helper 调用只会跳出 helper、测试继续跑（实证踩坑）。
// ═════════════════════════════════════════════════════════════════════════════

// Ensure 创建：工位名规约 conv_state_layer_<x> / rstate_layer_<x>（§4.5 全局层号）；
// 创建即整体清零；conv/r 相互独立（各自 tensor、独立计数）
TEST(LinearStatePoolTest, EnsureCreatesZeroedPerKindTensors)
{
    if (!TensorCreateAvailable()) {
        GTEST_SKIP() << "host 无 NDK so：TensorManager::CreateTensor 不可用"
                        "（需真后端或 stub .so 旁 exe，见 PROJECT.md「host UT 编译」）";
    }
    LinearStatePool pool({"linear_attention"});
    const auto conv_desc = MakeDesc({2, 4, 8});
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, conv_desc).ok());
    auto conv = pool.ConvTensor(0);
    ASSERT_TRUE(conv);
    EXPECT_EQ(conv->GetTensorDesc().dims, (std::vector<int32_t>{2, 4, 8}));
    EXPECT_EQ(conv->GetTensorDesc().dataType, DataType::FLOAT16);
    EXPECT_EQ(conv->GetTensorDesc().tensorName, "conv_state_layer_0");
    EXPECT_EQ(conv->GetSize(), 2u * 4u * 8u * 2u); // fp16 字节
    // 创建即整体清零（图首次读须是干净状态）
    EXPECT_TRUE(AllZero(conv->GetData(), conv->GetSize()));
    EXPECT_EQ(pool.TensorCount(), 1u);
    // r 通道独立未创建
    EXPECT_FALSE(pool.RTensor(0));

    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::R, conv_desc).ok());
    auto r = pool.RTensor(0);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->GetTensorDesc().tensorName, "rstate_layer_0");
    EXPECT_NE(conv.get(), r.get());
    EXPECT_EQ(pool.TensorCount(), 2u);
}

// 双图一致性把关：二次 Ensure 同 desc → SUCCESS 且复用同一 tensor（prefill/decode
// 跨图共享）；dims/dtype 失配 → INVALID_ARGUMENT，原 tensor 不动（§5 D1）
TEST(LinearStatePoolTest, SecondEnsureReusesOnMatchRejectsOnMismatch)
{
    if (!TensorCreateAvailable()) {
        GTEST_SKIP() << "host 无 NDK so：TensorManager::CreateTensor 不可用"
                        "（需真后端或 stub .so 旁 exe，见 PROJECT.md「host UT 编译」）";
    }
    LinearStatePool pool({"linear_attention"});
    const auto ref = MakeDesc({2, 4, 8});
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, ref).ok());
    auto first = pool.ConvTensor(0);
    ASSERT_TRUE(first);

    // 完全一致 → 复用（同一句柄，计数不增）
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, ref).ok());
    EXPECT_EQ(pool.ConvTensor(0).get(), first.get());
    EXPECT_EQ(pool.TensorCount(), 1u);

    // dims 失配 → INVALID_ARGUMENT（decode 图状态形状与 prefill 不一致的报错点）
    TensorDesc d1 = MakeDesc({2, 4, 9});
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, d1).code, Status::INVALID_ARGUMENT);
    // dtype 失配 → 同口径
    TensorDesc d2 = MakeDesc({2, 4, 8}, DataType::FLOAT32);
    EXPECT_EQ(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, d2).code, Status::INVALID_ARGUMENT);
    // 原 tensor 未被替换/改动
    auto after = pool.ConvTensor(0);
    ASSERT_TRUE(after);
    EXPECT_EQ(after.get(), first.get());
    EXPECT_EQ(after->GetTensorDesc().dims, (std::vector<int32_t>{2, 4, 8}));
    EXPECT_EQ(after->GetTensorDesc().dataType, DataType::FLOAT16);
}

// ResetLane 只清目标 lane 切片：batch 维 = dims[0]（§8 Q3），lane 步长 =
// GetSize()/dims[0]；其余 lane 字节保持原样（多请求共池不串扰）
TEST(LinearStatePoolTest, ResetLaneZeroesOnlyTargetSlice)
{
    if (!TensorCreateAvailable()) {
        GTEST_SKIP() << "host 无 NDK so：TensorManager::CreateTensor 不可用"
                        "（需真后端或 stub .so 旁 exe，见 PROJECT.md「host UT 编译」）";
    }
    LinearStatePool pool({"linear_attention"});
    // batch=3（dims[0]），每 lane 4 元素 × fp16 = 8 字节
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({3, 4})).ok());
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::R, MakeDesc({3, 4})).ok());
    auto conv = pool.ConvTensor(0);
    auto r = pool.RTensor(0);
    ASSERT_TRUE(conv && r);

    std::memset(conv->GetData(), 0xAA, conv->GetSize());
    std::memset(r->GetData(), 0xBB, r->GetSize());
    ASSERT_TRUE(pool.ResetLane(1).ok());

    // lane 步长按实现同式计算（GetSize()/dims[0]，后端若对齐 size 亦不破断言）
    for (const auto& t : {conv, r}) {
        const std::size_t lane_bytes = t->GetSize() / 3u;
        const uint8_t fill = (t.get() == conv.get()) ? 0xAA : 0xBB;
        const auto* base = static_cast<const uint8_t*>(t->GetData());
        EXPECT_TRUE(AllEqual(base + 0u * lane_bytes, lane_bytes, fill));
        EXPECT_TRUE(AllZero(base + 1u * lane_bytes, lane_bytes)); // 仅 lane1 清零
        EXPECT_TRUE(AllEqual(base + 2u * lane_bytes, lane_bytes, fill));
    }
}

// 多层池：ResetLane 清全层状态切片（层间独立 tensor，逐层清目标 lane）
TEST(LinearStatePoolTest, ResetLaneSweepsAllLinearLayers)
{
    if (!TensorCreateAvailable()) {
        GTEST_SKIP() << "host 无 NDK so：TensorManager::CreateTensor 不可用"
                        "（需真后端或 stub .so 旁 exe，见 PROJECT.md「host UT 编译」）";
    }
    LinearStatePool pool({"linear_attention", "full_attention", "linear_attention"});
    for (const int layer : {0, 2}) {
        ASSERT_TRUE(pool.EnsureStateTensor(layer, LinearStatePool::StateKind::Conv, MakeDesc({2, 4})).ok());
    }
    auto c0 = pool.ConvTensor(0);
    auto c2 = pool.ConvTensor(2);
    ASSERT_TRUE(c0 && c2);
    std::memset(c0->GetData(), 0xFF, c0->GetSize());
    std::memset(c2->GetData(), 0xFF, c2->GetSize());
    ASSERT_TRUE(pool.ResetLane(0).ok());
    // 两层各自的 lane0（前半，batch=2）均被清零 —— 全 0xFF 底上可见
    EXPECT_TRUE(AllZero(static_cast<uint8_t*>(c0->GetData()), c0->GetSize() / 2u));
    EXPECT_TRUE(AllZero(static_cast<uint8_t*>(c2->GetData()), c2->GetSize() / 2u));
}

// slot 越界（≥ dims[0]）/ 负值 → WARN 跳过不动数据，仍 SUCCESS（防御不炸）
TEST(LinearStatePoolTest, ResetLaneOutOfRangeSlotDefensivelySkipped)
{
    if (!TensorCreateAvailable()) {
        GTEST_SKIP() << "host 无 NDK so：TensorManager::CreateTensor 不可用"
                        "（需真后端或 stub .so 旁 exe，见 PROJECT.md「host UT 编译」）";
    }
    LinearStatePool pool({"linear_attention"});
    ASSERT_TRUE(pool.EnsureStateTensor(0, LinearStatePool::StateKind::Conv, MakeDesc({3, 4})).ok());
    auto conv = pool.ConvTensor(0);
    ASSERT_TRUE(conv);
    std::memset(conv->GetData(), 0xCC, conv->GetSize());

    EXPECT_TRUE(pool.ResetLane(3).ok()); // == dims[0] 边界
    EXPECT_TRUE(pool.ResetLane(99).ok());
    EXPECT_TRUE(pool.ResetLane(-1).ok());
    EXPECT_TRUE(AllEqual(conv->GetData(), conv->GetSize(), 0xCC)); // 数据原样
}

} // namespace
} // namespace cann_lm::engine
