// 采样器单元测试。关联 docs/ut/SPEC.md §4.5 / docs/ut/DESIGN.md §9。
// 算法 Real Era：sampler.cpp 算法层 6 步流水线（参考原型已删，本文为唯一行为锚），双层覆盖：
//   - 引擎接入层（ITensorBuffer + slot + SamplingParams）：贪心路径确定性断言；
//   - 算法层（静态纯函数直测）：seeded 采样确定性断言。
// 2026-09-07 F-SAMPLE 松绑：seed/repetition_penalty 经 SamplingParams 对外，
// 引擎接口层新增 TC215/216 覆盖（seed 复现 / rep 降权，含 rep=0 零初始化防御）。
//
// 2026-10-19 mock 修复：原引用 backend/tensor_buffer_impl.h（TensorBufferImpl host
// malloc 回退类，hiai 迁移中已删除；该头在全部 git 历史均不存在，本目标自始
// 编不过——engine/CMakeLists ut_linear_state_pool 处注释的"预存损坏"清单成员）。
// 按当前 backend API（tensor_manager.h 别名：ITensorBuffer=hiai::INDTensorBuffer、
// TensorDesc=hiai::NDTensorDesc{dims:int32_t[],dataType,format,tensorName}——
// 无独立 size 字段，Get 以缓冲为准）重写为 UT 内 HostTensorBuffer 自实现 mock
// （同 ut_share_weights_executor HostTensorBuffer 模式），不再依赖任何 backend
// 运行时/so：接入层只消费 GetData/GetSize/GetTensorDesc，自实现即足。
#include "core/component/sampler/sampler.h"
#include "utils/data_type_utils.h" // FloatToHalfBits（R1_4 fp16 位型构造）

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace cann_lm::engine {
namespace {

// ── UT 内 host mock（当前 backend API：INDTensorBuffer 五虚函数）─────────────
// desc 逻辑行宽 = dims 总积 × dtype 字节数由调用方在 desc_/data_ 各自设定——
// 刻意解耦，以构造"desc 与缓冲不符"等病态输入（TC213/214 需要）。

class HostTensorBuffer : public ITensorBuffer {
public:
    HostTensorBuffer() = default;
    // desc 与缓冲各自独立设定（供病态组合用例）
    HostTensorBuffer(const TensorDesc& desc, std::vector<uint8_t> data) : desc_(desc), data_(std::move(data)) {}
    // 一致构造：按 desc 元素积 × 4 字节（fp32）分配零缓冲
    explicit HostTensorBuffer(const TensorDesc& desc) : desc_(desc)
    {
        size_t elems = 1;
        for (int32_t d : desc_.dims)
            elems *= static_cast<size_t>(d > 0 ? d : 0);
        data_.assign(elems * sizeof(float), 0);
    }
    void* GetData() override { return data_.empty() ? nullptr : data_.data(); }
    size_t GetSize() const override { return data_.size(); }
    const TensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

    std::vector<uint8_t>& mutable_data() { return data_; }

private:
    TensorDesc desc_;
    std::vector<uint8_t> data_;
    uint8_t cache_status_ = 0;
};

// GetData()==nullptr → 接入层 logits 缺失 FAILURE 路径。
class NullTensorBuffer : public ITensorBuffer {
public:
    void* GetData() override { return nullptr; }
    size_t GetSize() const override { return 0; }
    const TensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

private:
    TensorDesc desc_; // dataType 默认（走不到 dtype 校验，!data 先短路）
    uint8_t cache_status_ = 0;
};

// 已知 logits 的 TensorBuffer（host 内存，GetData 返回确定数据）。
// 用于验证真实采样路径：贪心 argmax 选出确定 token。
class FixedTensorBuffer : public ITensorBuffer {
public:
    explicit FixedTensorBuffer(std::vector<float> logits) : data_(std::move(logits))
    {
        desc_.dataType = DataType::FLOAT32; // M1：适配层要求 fp32
    }
    void* GetData() override { return data_.data(); }
    size_t GetSize() const override { return data_.size() * sizeof(float); }
    const TensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

private:
    std::vector<float> data_;
    TensorDesc desc_;
    uint8_t cache_status_ = 0;
};

// GetData() 非空但 size==0 → 接入层空缓冲 INVALID_ARGUMENT 路径。
class EmptyDataBuffer : public ITensorBuffer {
public:
    void* GetData() override { return this; } // 非空哨指针
    size_t GetSize() const override { return 0; }
    const TensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

private:
    TensorDesc desc_;
    uint8_t cache_status_ = 0;
};

/// 构造 dims/数据自洽的 FLOAT32 logits 缓冲（host mock 自实现，无后端依赖）。
/// values.size() 须等于 dims 总元素数。
std::shared_ptr<ITensorBuffer> MakeLogits(const std::vector<int32_t>& dims, const std::vector<float>& values,
                                          DataType dtype = DataType::FLOAT32)
{
    TensorDesc d;
    d.dataType = dtype;
    d.dims = dims;
    auto buf = std::make_shared<HostTensorBuffer>(d);
    buf->mutable_data().assign(reinterpret_cast<const uint8_t*>(values.data()),
                               reinterpret_cast<const uint8_t*>(values.data()) + values.size() * sizeof(float));
    return buf;
}

// ────────────── 引擎接入层：参数守卫（保持 stub 期契约） ──────────────

// UT-TC-SAMPLER-001 (NEW-1): out=nullptr → INVALID_ARGUMENT
TEST(SamplerTest, SampleWithNullOutReturnsInvalidArgument_TC201)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 3.0f, 2.0f, 0.5f});
    SamplingParams params;
    EXPECT_EQ(sampler.Sample(*logits, /*slot=*/0, params, /*generated_ids=*/{}, /*out=*/nullptr).code,
              Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-002 (NEW-2): slot<0 → INVALID_ARGUMENT
TEST(SamplerTest, SampleWithNegativeSlotReturnsInvalidArgument_TC202)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 3.0f, 2.0f, 0.5f});
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(*logits, /*slot=*/-1, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-003 (NEW-3): GetData()==nullptr → FAILURE（logits 缺失）
TEST(SamplerTest, SampleWithNullDataReturnsFailure_TC203)
{
    Sampler sampler;
    NullTensorBuffer empty; // GetData() = nullptr
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(empty, /*slot=*/0, params, /*generated_ids=*/{}, &out).code, Status::FAILURE);
}

// ────────────── 引擎接入层：真实采样路径 ──────────────

// UT-TC-SAMPLER-004 (NEW-4 改写，算法 Real Era): 默认参数 → SUCCESS + 贪心 argmax
TEST(SamplerTest, SampleWithDefaultsReturnsGreedyArgmax_TC204)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 3.0f, 2.0f, 0.5f});
    SamplingParams params; // 默认：temp=1、top_k=0、top_p=1 → 贪心
    TokenId out = 0;
    Status st = sampler.Sample(*logits, /*slot=*/0, params, /*generated_ids=*/{}, &out);
    EXPECT_EQ(st.code, Status::SUCCESS);
    EXPECT_EQ(out, 1u); // argmax = index 1（3.0 最大）
}

// UT-TC-SAMPLER-005: batched dims [N,V] → 按 slot 取行，各行独立 argmax；slot 越界拒绝
TEST(SamplerTest, SampleBatchedDimsPicksPerSlotRow_TC205)
{
    Sampler sampler;
    auto logits = MakeLogits({2, 3}, {0.0f, 5.0f, 1.0f,   // slot0 行：argmax=1
                                      7.0f, 2.0f, 3.0f}); // slot1 行：argmax=0
    SamplingParams params;
    TokenId out = 0;

    ASSERT_EQ(sampler.Sample(*logits, 0, params, /*generated_ids=*/{}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 1u);
    ASSERT_EQ(sampler.Sample(*logits, 1, params, /*generated_ids=*/{}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 0u);
    // slot >= N（行容量）→ INVALID_ARGUMENT
    EXPECT_EQ(sampler.Sample(*logits, 2, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-006: 1-D dims [V]（单行容量）且 slot>0 → INVALID_ARGUMENT
TEST(SamplerTest, SampleWith1DDimsRejectsSlotBeyondRow_TC206)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 3.0f, 2.0f, 0.5f});
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(*logits, 1, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-007: dtype 非 FLOAT32/FLOAT16 → INVALID_ARGUMENT（接入层类型校验）
TEST(SamplerTest, SampleWithNonFloat32ReturnsInvalidArgument_TC207)
{
    Sampler sampler;
    auto logits = MakeLogits({2}, {1.0f, 2.0f}, DataType::INT32);
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(*logits, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-008: 字节数非元素对齐（6 字节，fp16 下 6%2==0 但 6%4!=0 的 fp32 路径）
// → INVALID_ARGUMENT
TEST(SamplerTest, SampleWithMisalignedBytesReturnsInvalidArgument_TC208)
{
    Sampler sampler;
    TensorDesc d;
    d.dataType = DataType::FLOAT32;
    HostTensorBuffer buf(d, std::vector<uint8_t>(6)); // 6 % 4 != 0
    ASSERT_NE(buf.GetData(), nullptr);
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(buf, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-009: 数据指针非空但 size==0 → INVALID_ARGUMENT
TEST(SamplerTest, SampleWithZeroSizeReturnsInvalidArgument_TC209)
{
    Sampler sampler;
    EmptyDataBuffer empty;
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(empty, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-010: dims 缺失且 slot>0 → FAILURE（无法安全推导行宽）
TEST(SamplerTest, SampleWithoutDimsAndSlotPositiveReturnsFailure_TC210)
{
    Sampler sampler;
    TensorDesc d;
    d.dataType = DataType::FLOAT32;
    HostTensorBuffer buf(d, std::vector<uint8_t>(16)); // 4 floats，dims 空（host 回退缓冲常见形态）
    ASSERT_NE(buf.GetData(), nullptr);
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(buf, 1, params, /*generated_ids=*/{}, &out).code, Status::FAILURE);
}

// UT-TC-SAMPLER-011: 旋钮接线——top_k=2 + temperature=0 → 钳制退化贪心，确定性 argmax
TEST(SamplerTest, SampleWithTopKAndZeroTemperatureStaysGreedy_TC211)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 5.0f, 4.9f, 0.0f});
    SamplingParams params;
    params.top_k = 2;
    params.temperature = 0.0f; // → 1e-8 钳制，softmax≈one-hot，采样必取 argmax
    TokenId out = 0;
    Status st = sampler.Sample(*logits, 0, params, /*generated_ids=*/{}, &out);
    EXPECT_EQ(st.code, Status::SUCCESS);
    EXPECT_EQ(out, 1u);
}

// UT-TC-SAMPLER-012: top_p=0.9 → 核采样截尾后仍在 top-2 内（概率性，只断言集合）
TEST(SamplerTest, SampleWithTopPWiresNucleusSampling_TC212)
{
    Sampler sampler;
    auto logits = MakeLogits({4}, {1.0f, 5.0f, 4.9f, 0.0f});
    SamplingParams params;
    params.top_p = 0.9f; // 截尾后仅剩 {index1, index2}
    TokenId out = 0;
    Status st = sampler.Sample(*logits, 0, params, /*generated_ids=*/{}, &out);
    EXPECT_EQ(st.code, Status::SUCCESS);
    EXPECT_TRUE(out == 1u || out == 2u);
}

// UT-TC-SAMPLER-013: desc 与缓冲不符（dims [2,4] → 需 32 字节，缓冲仅 8 字节）
// → INVALID_ARGUMENT
TEST(SamplerTest, SampleWithDescBufferMismatchReturnsInvalidArgument_TC213)
{
    Sampler sampler;
    TensorDesc d;
    d.dataType = DataType::FLOAT32;
    d.dims = {2, 4};                                  // 行宽 4 × 行数 2 → 需 32 字节
    HostTensorBuffer buf(d, std::vector<uint8_t>(8)); // 实际仅 8 字节
    ASSERT_NE(buf.GetData(), nullptr);
    SamplingParams params;
    TokenId out = 0;
    EXPECT_EQ(sampler.Sample(buf, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// UT-TC-SAMPLER-014: dims 含非正维（back()<=0 或前缀维<=0）→ INVALID_ARGUMENT
TEST(SamplerTest, SampleWithNonPositiveDimsReturnsInvalidArgument_TC214)
{
    Sampler sampler;
    SamplingParams params;
    TokenId out = 0;

    TensorDesc zero_last;
    zero_last.dataType = DataType::FLOAT32;
    zero_last.dims = {0};
    HostTensorBuffer buf1(zero_last, std::vector<uint8_t>(4));
    ASSERT_NE(buf1.GetData(), nullptr);
    EXPECT_EQ(sampler.Sample(buf1, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);

    TensorDesc neg_prefix;
    neg_prefix.dataType = DataType::FLOAT32;
    neg_prefix.dims = {2, -3, 4};
    HostTensorBuffer buf2(neg_prefix, std::vector<uint8_t>(16));
    ASSERT_NE(buf2.GetData(), nullptr);
    EXPECT_EQ(sampler.Sample(buf2, 0, params, /*generated_ids=*/{}, &out).code, Status::INVALID_ARGUMENT);
}

// ────────────── 算法层：静态纯函数（确定性断言） ──────────────

TEST(SamplerAlgoTest, ApplyTemperatureDividesLogits)
{
    auto out = Sampler::ApplyTemperature({2.0f, 4.0f}, 2.0f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST(SamplerAlgoTest, ApplyTemperatureClampsNonPositive)
{
    auto out = Sampler::ApplyTemperature({1.0f, -2.0f}, 0.0f); // 钳 1e-8，不除零
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
}

TEST(SamplerAlgoTest, ApplyTopKKeepsTopKOnly)
{
    auto out = Sampler::ApplyTopK({1.0f, 5.0f, 3.0f, 2.0f}, 2);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_FLOAT_EQ(out[1], 5.0f); // top-2 保留原值
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_TRUE(std::isinf(out[0])); // 其余 -inf
    EXPECT_TRUE(std::isinf(out[3]));
}

TEST(SamplerAlgoTest, ApplyTopKNoOpWhenKOutOfRange)
{
    auto out = Sampler::ApplyTopK({1.0f, 2.0f}, 5); // k >= n → 不启用
    ASSERT_EQ(out.size(), 2u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST(SamplerAlgoTest, ApplyPrecomputedTopKMasksOthers)
{
    PrecomputedTopK pre;
    pre.token_ids = {0, 2};
    auto out = Sampler::ApplyPrecomputedTopK({1.0f, 2.0f, 3.0f, 4.0f}, pre);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_TRUE(std::isinf(out[1]));
    EXPECT_TRUE(std::isinf(out[3]));
}

TEST(SamplerAlgoTest, ApplyTopPKeepsHighestProbability)
{
    auto out = Sampler::ApplyTopP({5.0f, 1.0f, 1.0f, 1.0f}, 0.5f);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_FLOAT_EQ(out[0], 5.0f); // 最高概率项必留
    for (size_t i = 1; i < out.size(); ++i) {
        EXPECT_TRUE(std::isinf(out[i])); // 累计概率已达 p，其余截尾
    }
}

TEST(SamplerAlgoTest, ApplyRepetitionPenaltyMirrorsHF)
{
    auto out = Sampler::ApplyRepetitionPenalty({-1.0f, 1.0f, 3.0f}, {0, 1}, 2.0f);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], -2.0f); // score<0 → *=p
    EXPECT_FLOAT_EQ(out[1], 0.5f);  // score>=0 → /=p
    EXPECT_FLOAT_EQ(out[2], 3.0f);  // 未生成的不动
}

TEST(SamplerAlgoTest, NormalizeProducesProbabilityDistribution)
{
    auto probs = Sampler::Normalize({1.0f, 2.0f, 3.0f});
    ASSERT_EQ(probs.size(), 3u);
    double sum = 0.0;
    for (float p : probs) {
        EXPECT_GE(p, 0.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-5);
    EXPECT_GT(probs[2], probs[1]); // 概率序随 logit 序
    EXPECT_GT(probs[1], probs[0]);
}

TEST(SamplerAlgoTest, PickGreedyReturnsArgmax)
{
    EXPECT_EQ(Sampler::Pick({0.1f, 0.7f, 0.2f}, /*do_sample=*/false, std::nullopt), 1);
}

TEST(SamplerAlgoTest, PickSeededIsReproducible)
{
    auto a = Sampler::Pick({0.2f, 0.5f, 0.3f}, /*do_sample=*/true, 42);
    auto b = Sampler::Pick({0.2f, 0.5f, 0.3f}, /*do_sample=*/true, 42);
    EXPECT_EQ(a, b);
    EXPECT_GE(a, 0);
    EXPECT_LE(a, 2);
}

TEST(SamplerAlgoTest, FullPipelineGreedyPicksArgmax)
{
    SamplerInputs in;
    in.logits = {1.0f, 3.0f, 2.0f};
    SamplerConfig cfg; // 默认贪心
    SamplerOutput out = Sampler::Sample(std::move(in), cfg);
    EXPECT_EQ(out.next_token_id, 1);
    // R1（perf_optimize.md §4.1/§5 R1③）：贪心快路径 probs 惰性产出——
    // 接入层只消费 next_token_id，贪心下不再物化 262K 概率分布。
    EXPECT_TRUE(out.probs.empty());
}

TEST(SamplerAlgoTest, FullPipelineSeededSamplingIsReproducible)
{
    SamplerConfig cfg;
    cfg.do_sample = true;
    cfg.seed = 7;
    SamplerInputs in;
    in.logits = {1.0f, 5.0f, 4.9f, 0.0f};
    SamplerOutput a = Sampler::Sample(in, cfg);
    SamplerOutput b = Sampler::Sample(in, cfg);
    EXPECT_EQ(a.next_token_id, b.next_token_id); // 同 seed 可复现
}

// ── seed / repetition_penalty 经 SamplingParams 对外（F-SAMPLE 松绑，2026-09-07）──

// UT-TC-SAMPLER-215: 概率采样 + 同 seed → 跨调用可复现。
// 断言强度：单次"两调用同值"在 seed 断裂时仍有 ~0.36 假阳性概率（分布 Σp²），
// 故循环 20 次全等（假阳性 ≈ 0.36^20 ≈ 4e-9）。
TEST(SamplerTest, SampleWithSeedReproducesAcrossCalls_TC215)
{
    Sampler sampler;
    SamplingParams params;
    params.temperature = 0.8f; // 任一采样旋钮非默认 → do_sample
    params.seed = 99;
    FixedTensorBuffer logits_buf({1.0f, 2.0f, 0.5f, 1.5f});
    TokenId first = 0;
    ASSERT_EQ(sampler.Sample(logits_buf, 0, params, /*generated_ids=*/{}, &first).code, Status::SUCCESS);
    for (int i = 0; i < 20; ++i) {
        TokenId again = 0;
        ASSERT_EQ(sampler.Sample(logits_buf, 0, params, /*generated_ids=*/{}, &again).code, Status::SUCCESS);
        EXPECT_EQ(again, first) << "seed 复现断裂于第 " << i << " 次调用";
    }
}

// UT-TC-SAMPLER-217: C ABI 零初始化防御——rep=0.0 须视为关闭（H-1 回归锚点：
// 旧行为 0.0 != 1.0 放行 → penalty=0 除零 → 已生成 token 概率被 softmax 兜底归零）
TEST(SamplerTest, SampleWithZeroRepetitionPenaltyTreatedAsOff_TC217)
{
    Sampler sampler;
    FixedTensorBuffer logits_buf({3.0f, 2.9f});
    TokenId out = 0;
    SamplingParams zero_init;            // 模拟 C 侧 = {} 零初始化
    zero_init.repetition_penalty = 0.0f; // != 1.0 但非法 → 须关闭
    // 已生成 token 0，若 0.0 被放行：3.0/0=+inf → softmax 兜底触发 → 0 必被选（永不重复）
    ASSERT_EQ(sampler.Sample(logits_buf, 0, zero_init, /*generated_ids=*/{0}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 0u); // rep 关闭 → 贪心照常选最高分 0（而非被除零污染）
}

// UT-TC-SAMPLER-216: 贪心 + repetition_penalty：已生成的最高分 token 被降权 → 次高胜出
TEST(SamplerTest, SampleWithRepetitionPenaltyDownweightsGenerated_TC216)
{
    Sampler sampler;
    FixedTensorBuffer logits_buf({3.0f, 2.9f});
    TokenId out = 0;
    // 无 penalty：贪心选 0（3.0 > 2.9）
    SamplingParams plain;
    ASSERT_EQ(sampler.Sample(logits_buf, 0, plain, /*generated_ids=*/{0}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 0u);
    // penalty=2.0：已生成的 0 → 3.0/2=1.5 < 2.9 → 选 1（HF 语义：score>=0 除以 p）
    SamplingParams rep;
    rep.repetition_penalty = 2.0f;
    ASSERT_EQ(sampler.Sample(logits_buf, 0, rep, /*generated_ids=*/{0}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 1u);
}

// ── R1 贪心免归一化快路径（perf_optimize.md §4.1/§5 R1，2026-10-27）──────────
// ① 快路径与改前输出逐 token 一致：含并列最大值首个胜出的 tie-break 对抗——
//   softmax 保序（贪心下 top-k/top-p 均 no-op，logits 原样进归一化），
//   argmax(x) ≡ argmax(Normalize(x))，须有专门用例锁 tie 取首个（改前
//   Normalize+argmax(probs) 同样取首个，语义逐字保留）。
// ② do_sample=true 回归：softmax/multinomial 路径不被贪心短路触碰。
TEST(SamplerAlgoTest, GreedyShortpathTieBreakFirstMaxWins_R1_1)
{
    // 并列最大 3.0 出现在下标 1 与 3：argmax 语义 = 首个（下标 1）
    SamplerInputs in;
    in.logits = {1.0f, 3.0f, 0.5f, 3.0f, 2.0f};
    SamplerConfig cfg; // 默认贪心
    SamplerOutput out = Sampler::Sample(std::move(in), cfg);
    EXPECT_EQ(out.next_token_id, 1);
    EXPECT_TRUE(out.probs.empty()); // R1③：probs 惰性产出
}

TEST(SamplerAlgoTest, GreedyShortpathMatchesLegacyNormalizeArgmax_R1_2)
{
    // 对抗性等价抽样：同一 logits 下贪心快路径 id ≡ 手工 Normalize+argmax(id)。
    // 段内逐 token 独立、覆盖含相等/负值/0 间隔的多形态。等价前提 = 输入差量
    // 可被 softmax 的 (float)ei 量化分辨（> 半 ulp ~3e-8 的近 1.0 域）。
    const std::vector<float> cases[] = {
        {0.1f, 0.7f, 0.2f},
        {-5.0f, -1.0f, -3.0f},
        {2.0f, 2.0f, 2.0f},             // 全并列 → 下标 0
        {1e-4f, 2e-4f, 0.0f},           // 近零域（差量 >> float ulp）
        {3.3f, 3.3f, -7.0f, 0.0f, 3.3f} // 间隔重复最大
    };
    for (const auto& logits : cases) {
        SamplerInputs in;
        in.logits = logits;
        SamplerConfig cfg; // 贪心
        const SamplerOutput out = Sampler::Sample(std::move(in), cfg);
        // 改前行为复刻：Normalize + argmax（Pick 贪心分支）
        const auto probs = Sampler::Normalize(logits);
        const int32_t legacy = Sampler::Pick(probs, /*do_sample=*/false, std::nullopt);
        EXPECT_EQ(out.next_token_id, legacy);
    }
}

TEST(SamplerAlgoTest, GreedyShortpathSubUlpCollapseDeviationDocumented_R1_2b)
{
    // 登记偏差（perf_optimize.md §4.1 亚 ulp 尾带）：{1e-8, 2e-8, 0} 的差量 <
    // softmax_stable (float)ei 量化的半 ulp ⇒ exp(-1e-8) 与 exp(-2e-8) 同舍入
    // 1.0f ⇒ 改前 Normalize 后三值并列 → argmax 取下标 0（首个）。快路径按
    // 原始 logits 严格比较取下标 1（更大者胜，信息不丢——更贴近输入）。
    // 部署输入为 fp16 量化 logits：两 token 差 <6e-8 且同时存活 fp16 仅在
    // denormal 尾带，实际不可达；本用例显式锁界防漂移归因，勿改回 0。
    SamplerInputs in;
    in.logits = {1e-8f, 2e-8f, 0.0f};
    SamplerConfig cfg; // 贪心
    const SamplerOutput out = Sampler::Sample(std::move(in), cfg);
    EXPECT_EQ(out.next_token_id, 1);
}

TEST(SamplerAlgoTest, SeededSamplingUnchangedByGreedyShortpath_R1_3)
{
    // R1②：do_sample=true 全量路径回归（seeded 可复现——贪心短路不触碰
    // softmax/multinomial 输出）
    SamplerConfig cfg;
    cfg.do_sample = true;
    cfg.seed = 42;
    SamplerInputs a_in;
    a_in.logits = {0.2f, 5.0f, 4.9f, 0.0f};
    const SamplerOutput a = Sampler::Sample(a_in, cfg);
    const SamplerOutput b = Sampler::Sample(a_in, cfg);
    EXPECT_EQ(a.next_token_id, b.next_token_id);
    ASSERT_EQ(a.probs.size(), 4u); // 概率采样仍物化 probs
    double sum = 0.0;
    for (float p : a.probs)
        sum += p;
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

// ── R1 接入层：行免中转贪心 + fp16 等价（§4.1 配套 + §5 R1④）────────────────
// ④ fp16 行免中转：HalfBitsToFloat 逐元素展开对照——快路径扫描在 fp16 位型
//   原地完成，输出须与"先整行转 fp32 再 argmax"逐 token 一致；贪心 +
//   repetition_penalty 启用时仍走算法层全量路径（rep 降权语义不变）。
TEST(SamplerTest, GreedyShortpathFp16RowEquivToExpand_R1_4)
{
    Sampler sampler;
    SamplingParams greedy; // 默认参数 → 贪心（小词表 top_k 原 no-op）

    // 非并列 + 并列两种行形态；fp16 位型数据（等价上浮到 fp32 也作对照行）
    const std::vector<float> fp32_row = {1.5f, 4.0f, 4.0f, -2.0f, 0.25f};
    std::vector<uint16_t> fp16_row(fp32_row.size());
    for (size_t i = 0; i < fp32_row.size(); ++i) {
        fp16_row[i] = FloatToHalfBits(fp32_row[i]);
    }
    // fp16 缓冲：dims={1,5} FLOAT16，数据位型 uint16
    TensorDesc d16;
    d16.dataType = DataType::FLOAT16;
    d16.dims = {1, 5};
    auto buf16 = std::make_shared<HostTensorBuffer>(d16);
    buf16->mutable_data().assign(
        reinterpret_cast<const uint8_t*>(fp16_row.data()),
        reinterpret_cast<const uint8_t*>(fp16_row.data()) + fp16_row.size() * sizeof(uint16_t));

    TokenId out = 0;
    ASSERT_EQ(sampler.Sample(*buf16, 0, greedy, /*generated_ids=*/{}, &out).code, Status::SUCCESS);
    EXPECT_EQ(out, 1u); // 并列 4.0@1 与@4 前者仍首个最大（HalfBits 保序）

    // 对照：同一行先整行转 fp32 再走 fp32 贪心快路径，id 须一致
    auto buf32 = MakeLogits({1, 5}, fp32_row);
    TokenId out32 = 0;
    ASSERT_EQ(sampler.Sample(*buf32, 0, greedy, /*generated_ids=*/{}, &out32).code, Status::SUCCESS);
    EXPECT_EQ(out, out32);

    // rep 启用 + 贪心：不短路（降权须作用），行为与改前全量路径一致。
    // 行内两个并列最大 4.0@{1,2}，rep=3.0 且已生成 {1,2}：两处 4.0/3≈1.33 <
    // 1.5 ⇒ 选下标 0——若误短路，会直接选 1（原始 argmax）。
    SamplingParams rep;
    rep.repetition_penalty = 3.0f;
    TokenId rep_out = 0;
    ASSERT_EQ(sampler.Sample(*buf16, 0, rep, /*generated_ids=*/{1, 2}, &rep_out).code, Status::SUCCESS);
    EXPECT_EQ(rep_out, 0u);
}

} // namespace
} // namespace cann_lm::engine
