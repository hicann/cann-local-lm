// TEST-WRITE(GREEN)：TensorManager 单元测试（v0.13 迁移：hiai NDK backend 代理）。
// 验证 CreateTensor 委托 hiai::CreateNDTensorBuffer（容量 = dims×dtype，desc 无
// size 字段）+ 单例 + 异常；CreateKvCacheTensor 层参改 layer_ids 向量
// （非连续层号直通 tensorName）、轴序/命名/容量契约与 ut_engine_executor
// KvCacheTensorApiTest（TC608/609）同口径；非法 dtype 用 RESERVED（UNDEFINED 已删）。
// host 经 stub NDK so 真建 tensor（CANN_LM_NDK_LIB_DIR 注入，注册同
// ut_engine_executor；本目标原属 model_executor_ut 注册面）。
#include "core/tensor_manager.h"

#include <gtest/gtest.h>
#include <memory>

namespace cann_lm::engine {
namespace {

// 按 dims×dtype 容量申请 tensor（size 字段已不存在——NDTensorDesc 无此成员）
TEST(TensorManagerTest, CreateTensorReturnsBufferWithCorrectSize)
{
    TensorDesc d;
    d.dims = {2, 3};
    d.dataType = DataType::FLOAT32;
    d.tensorName = "plain";
    std::shared_ptr<ITensorBuffer> t;
    ASSERT_TRUE(TensorManager::Instance().CreateTensor(d, /*modelId=*/1, &t).ok());
    ASSERT_TRUE(t);
    EXPECT_EQ(t->GetSize(), 24u); // 2×3×4（元素数 × fp32 字节）
    EXPECT_TRUE(t->GetData() != nullptr);
    EXPECT_EQ(t->GetTensorDesc().dims, (std::vector<int32_t>{2, 3}));
    EXPECT_EQ(t->GetTensorDesc().tensorName, "plain");
}

// dims 空 → backend 拒绝（Convert2NNTensorDesc 0 维不成立）→ 优雅降级 FAILURE
// （旧 ZeroSizeUsesBackendDefault 的"size=0 走 backend 默认 1024"语义随 size
// 字段一并消亡：容量唯一源 = dims×dtype）
TEST(TensorManagerTest, EmptyDimsRejectedGracefully)
{
    TensorDesc d;
    d.dataType = DataType::FLOAT32;
    std::shared_ptr<ITensorBuffer> t;
    EXPECT_EQ(TensorManager::Instance().CreateTensor(d, 1, &t).code, Status::FAILURE);
    EXPECT_FALSE(t);
}

// 单例：Instance() 返回同一实例
TEST(TensorManagerTest, InstanceIsSingleton)
{
    auto& a = TensorManager::Instance();
    auto& b = TensorManager::Instance();
    EXPECT_EQ(&a, &b);
}

// 无效入参 → INVALID_ARGUMENT
TEST(TensorManagerTest, NullOutRejected)
{
    TensorDesc d;
    EXPECT_EQ(TensorManager::Instance().CreateTensor(d, 1, nullptr).code, Status::INVALID_ARGUMENT);
}

// ── DataTypeSize：dtype→字节数辅助（RESERVED → 0）──

TEST(TensorManagerTest, DataTypeSizeCommonTypes)
{
    EXPECT_EQ(DataTypeSize(DataType::FLOAT16), 2u);
    EXPECT_EQ(DataTypeSize(DataType::FLOAT32), 4u);
    EXPECT_EQ(DataTypeSize(DataType::INT8), 1u);
    EXPECT_EQ(DataTypeSize(DataType::INT32), 4u);
    EXPECT_EQ(DataTypeSize(DataType::INT64), 8u);
    EXPECT_EQ(DataTypeSize(DataType::RESERVED), 0u); // 原 UNDEFINED 位（已删）
}

// ── CreateKvCacheTensor：KV 池专用批量通道（单次创建全部 2N tensor）──

// 一次调用创建 2×层数个 tensor：数量/输出序/name/dims/dtype/size
// （层参为 layer_ids 向量——层号直通 tensorName，非连续表 {0,3} 验证直通语义）
TEST(TensorManagerTest, CreateKvCacheTensorCreatesAllTensorsOneCall)
{
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ASSERT_TRUE(TensorManager::Instance()
                    .CreateKvCacheTensor(/*layer_ids=*/{0, 3},
                                         /*kv_cache_max_len=*/16,
                                         /*num_attention_kv_heads=*/4,
                                         /*batch=*/2,
                                         /*num_attention_head_dims=*/8, DataType::FLOAT16, &ts)
                    .ok());
    ASSERT_EQ(ts.size(), 4u); // 2 层 × (K+V)
    // 默认轴序 [len, heads, batch, dim]（引擎旧口径，TC609 同锁）
    const std::vector<int32_t> expect_dims = {16, 4, 2, 8};
    const size_t expect_size = 16u * 4u * 2u * 8u * 2u; // 元素数 × fp16 字节
    for (const auto& t : ts) {
        ASSERT_TRUE(t);
        EXPECT_EQ(t->GetTensorDesc().dims, expect_dims);
        EXPECT_EQ(t->GetTensorDesc().dataType, DataType::FLOAT16);
        EXPECT_EQ(t->GetTensorDesc().format, Format::ND);
        EXPECT_EQ(t->GetSize(), expect_size);
        EXPECT_TRUE(t->GetData() != nullptr);
    }
    // 输出序 [k(l0), v(l0), k(l1), v(l1)]；tensorName 用列表中的全局层号
    EXPECT_EQ(ts[0]->GetTensorDesc().tensorName, "kv_cache_k_0");
    EXPECT_EQ(ts[1]->GetTensorDesc().tensorName, "kv_cache_v_0");
    EXPECT_EQ(ts[2]->GetTensorDesc().tensorName, "kv_cache_k_3");
    EXPECT_EQ(ts[3]->GetTensorDesc().tensorName, "kv_cache_v_3");
}

// 维度/dtype 逐项断言：FLOAT32 路径（字节数 4）
TEST(TensorManagerTest, CreateKvCacheTensorFloat32Size)
{
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ASSERT_TRUE(TensorManager::Instance().CreateKvCacheTensor({1}, 8, 2, 1, 16, DataType::FLOAT32, &ts).ok());
    ASSERT_EQ(ts.size(), 2u);
    const size_t expect_size = 8u * 2u * 1u * 16u * 4u;
    EXPECT_EQ(ts[0]->GetSize(), expect_size);
    EXPECT_EQ(ts[1]->GetSize(), expect_size);
    EXPECT_EQ(ts[0]->GetTensorDesc().tensorName, "kv_cache_k_1");
}

// 参数非法（layer_ids 空/含负 / 其余任一 ≤ 0 / dtype RESERVED / out 空）
// → INVALID_ARGUMENT，无产出
// 已知限制：中途 FAILURE（底层分配失败 → all-or-nothing 整体不产出）无注入缝
// （TensorManager 具体单例 + hiai::CreateNDTensorBuffer 自由函数），stub so
// 正常形态下不可触发——留待后续加失败注入缝再补用例（test-audit LOW-3）。
TEST(TensorManagerTest, CreateKvCacheTensorRejectsInvalidParams)
{
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    EXPECT_EQ(tm.CreateKvCacheTensor({}, 16, 4, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // layer_ids 空（原 num_hidden_layers=0 位）
    EXPECT_EQ(tm.CreateKvCacheTensor({-1}, 16, 4, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // 层号负
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 0, 4, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // kv_cache_max_len
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 0, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // kv_heads
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 0, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // batch
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 0, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT); // head_dims
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 8, DataType::RESERVED, &ts).code,
              Status::INVALID_ARGUMENT); // dtype 保留值（原 UNDEFINED 位）
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 8, DataType::FLOAT16, nullptr).code,
              Status::INVALID_ARGUMENT); // out
    EXPECT_TRUE(ts.empty());             // 失败路径不产出
}

// 维度乘积溢出 uint64 → INVALID_ARGUMENT（分配前拦截，不产生大额申请；
// 该路径可自 meta_json 大整数字段到达——无防护会回绕成过小 buffer）
TEST(TensorManagerTest, CreateKvCacheTensorRejectsDimProductOverflow)
{
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    EXPECT_EQ(
        TensorManager::Instance().CreateKvCacheTensor({1}, 1 << 30, 1 << 30, 1 << 30, 1, DataType::FLOAT16, &ts).code,
        Status::INVALID_ARGUMENT); // (2^30)^3 = 2^90 > 2^64
    EXPECT_TRUE(ts.empty());
}

} // namespace
} // namespace cann_lm::engine
