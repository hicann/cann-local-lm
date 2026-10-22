// TEST-WRITE(GREEN)：TensorManager 单元测试（v0.12：backend 代理）。
// 验证 CreateTensor 委托 hiai::CreateNDTensorBuffer，按 dims×dtype 分配 + 单例 + 异常。
#include "core/tensor_manager.h"

#include <gtest/gtest.h>
#include <memory>

namespace cann_lm::engine {
namespace {

// 按 dims×dtype 字节数申请 tensor
TEST(TensorManagerTest, CreateTensorReturnsBufferWithCorrectSize) {
    TensorDesc d;
    d.dims = {2, 3};
    d.dataType = DataType::FLOAT32;   // 2×3×4 = 24 字节
    std::shared_ptr<ITensorBuffer> t;
    ASSERT_TRUE(TensorManager::Instance().CreateTensor(d, /*modelId=*/1, &t).ok());
    ASSERT_TRUE(t);
    EXPECT_EQ(t->GetSize(), 24u);
    EXPECT_TRUE(t->GetData() != nullptr);
}

// dims 含 0 → 0 字节 buffer（旧 TensorBufferImpl 默认 1024 已删，size 纯由 dims×dtype 推导）
TEST(TensorManagerTest, ZeroDimYieldsZeroSize) {
    TensorDesc d;
    d.dims = {0};
    d.dataType = DataType::FLOAT32;
    std::shared_ptr<ITensorBuffer> t;
    ASSERT_TRUE(TensorManager::Instance().CreateTensor(d, 1, &t).ok());
    ASSERT_TRUE(t);
    EXPECT_EQ(t->GetSize(), 0u);
}

// 单例：Instance() 返回同一实例
TEST(TensorManagerTest, InstanceIsSingleton) {
    auto& a = TensorManager::Instance();
    auto& b = TensorManager::Instance();
    EXPECT_EQ(&a, &b);
}

// 无效入参 → INVALID_ARGUMENT
TEST(TensorManagerTest, NullOutRejected) {
    TensorDesc d;
    EXPECT_EQ(TensorManager::Instance().CreateTensor(d, 1, nullptr).code,
              Status::INVALID_ARGUMENT);
}

// ── DataTypeSize：dtype→字节数辅助 ──

TEST(TensorManagerTest, DataTypeSizeCommonTypes) {
    EXPECT_EQ(DataTypeSize(DataType::FLOAT16), 2u);
    EXPECT_EQ(DataTypeSize(DataType::FLOAT32), 4u);
    EXPECT_EQ(DataTypeSize(DataType::INT8), 1u);
    EXPECT_EQ(DataTypeSize(DataType::INT32), 4u);
    EXPECT_EQ(DataTypeSize(DataType::RESERVED), 0u);
}

// ── CreateKvCacheTensor：KV 池专用批量通道（单次创建全部 2N tensor）──

// 一次调用创建 2×num_hidden_layers 个 tensor：数量/输出序/name/dims/dtype/size
TEST(TensorManagerTest, CreateKvCacheTensorCreatesAllTensorsOneCall) {
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ASSERT_TRUE(TensorManager::Instance()
                    .CreateKvCacheTensor(/*layer_ids=*/{0, 1},
                                         /*kv_cache_max_len=*/16,
                                         /*num_attention_kv_heads=*/4,
                                         /*batch=*/2,
                                         /*num_attention_head_dims=*/8,
                                         DataType::FLOAT16, &ts)
                    .ok());
    ASSERT_EQ(ts.size(), 4u);   // 2 层 × (K+V)
    const std::vector<int32_t> expect_dims = {16, 4, 2, 8};
    const size_t expect_size = 16u * 4u * 2u * 8u * 2u;   // 元素数 × fp16 字节
    for (const auto& t : ts) {
        ASSERT_TRUE(t);
        EXPECT_EQ(t->GetTensorDesc().dims, expect_dims);
        EXPECT_EQ(t->GetTensorDesc().dataType, DataType::FLOAT16);
        EXPECT_EQ(t->GetSize(), expect_size);
        EXPECT_TRUE(t->GetData() != nullptr);
    }
    // 输出序 [k0, v0, k1, v1]（池按 2*layer+{0=K,1=V} 直接消费）
    EXPECT_EQ(ts[0]->GetTensorDesc().tensorName, "kv_cache_k_0");
    EXPECT_EQ(ts[1]->GetTensorDesc().tensorName, "kv_cache_v_0");
    EXPECT_EQ(ts[2]->GetTensorDesc().tensorName, "kv_cache_k_1");
    EXPECT_EQ(ts[3]->GetTensorDesc().tensorName, "kv_cache_v_1");
}

// 维度/dtype 逐项断言：FLOAT32 路径（字节数 4）
TEST(TensorManagerTest, CreateKvCacheTensorFloat32Size) {
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ASSERT_TRUE(TensorManager::Instance()
                    .CreateKvCacheTensor({0}, 8, 2, 1, 16, DataType::FLOAT32, &ts)
                    .ok());
    ASSERT_EQ(ts.size(), 2u);
    const size_t expect_size = 8u * 2u * 1u * 16u * 4u;
    EXPECT_EQ(ts[0]->GetSize(), expect_size);
    EXPECT_EQ(ts[1]->GetSize(), expect_size);
}

// 参数非法（layer_ids 空 / 任一维度 ≤ 0 / dtype RESERVED / out 空）→ INVALID_ARGUMENT，无产出
// 已知限制：中途 FAILURE（第 N 个 tensor 底层分配失败 → all-or-nothing 整体不产出）
// 无注入缝（TensorManager 具体单例 + CreateNDTensorBuffer 自由函数），host malloc
// 回退下不可触发——留待后续加失败注入缝再补用例（test-audit LOW-3）。
TEST(TensorManagerTest, CreateKvCacheTensorRejectsInvalidParams) {
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    EXPECT_EQ(tm.CreateKvCacheTensor({}, 16, 4, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT);   // layer_ids 空
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 0, 4, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT);   // kv_cache_max_len
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 0, 2, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT);   // kv_heads
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 0, 8, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT);   // batch
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 0, DataType::FLOAT16, &ts).code,
              Status::INVALID_ARGUMENT);   // head_dims
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 8, DataType::RESERVED, &ts).code,
              Status::INVALID_ARGUMENT);   // dtype
    EXPECT_EQ(tm.CreateKvCacheTensor({0, 1}, 16, 4, 2, 8, DataType::FLOAT16, nullptr).code,
              Status::INVALID_ARGUMENT);   // out
    EXPECT_TRUE(ts.empty());   // 失败路径不产出
}

// 维度乘积溢出 uint64 → INVALID_ARGUMENT（分配前拦截，不产生大额申请；
// 该路径可自 meta_json 大整数字段到达——无防护会回绕成过小 buffer）
TEST(TensorManagerTest, CreateKvCacheTensorRejectsDimProductOverflow) {
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    EXPECT_EQ(TensorManager::Instance()
                  .CreateKvCacheTensor({0}, 1 << 30, 1 << 30, 1 << 30, 1,
                                       DataType::FLOAT16, &ts)
                  .code,
              Status::INVALID_ARGUMENT);   // (2^30)^3 = 2^90 > 2^64
    EXPECT_TRUE(ts.empty());
}

}  // namespace
}  // namespace cann_lm::engine
