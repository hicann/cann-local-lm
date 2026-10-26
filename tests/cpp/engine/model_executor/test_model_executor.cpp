// TEST-WRITE(GREEN)：ModelExecutor 单元测试（v0.13：全链路改 hiai backend 类型）。
// Mock 实现 hiai::IBuiltModel/IModelManager 全部纯虚方法；
// 覆盖 hiai 模型生命周期工厂（CreateBuiltModel + CreateModelManager）。
//   - cann_lm_engine 为 STATIC 库链入 .so，ut 对象定义的 hiai:: 工厂符号优先于
//     库内 factory TU（链接期符号替换）。张量创建（hiai::CreateNDTensorBuffer）
//     不再在此 ODR 顶替：退役 HostTensorBuffer mock 后经 cann_backend 真 impl
//     → NDKProxy → dlopen stub .so（host UT 经 CANN_LM_NDK_LIB_DIR 注入 build 目录
//     stub）。本目标其余 3 个 .cpp（test_kv_cache_pool / test_dfx_collector /
//     test_tensor_manager）的 tensor 申请同走 stub 真路径——切勿在那些文件重复
//     定义 hiai::CreateNDTensorBuffer 同名符号（已退役 mock，无兜底必要）。
#include "core/component/model_executor/model_executor.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/dfx_collector/dfx_collector.h"
#include "core/tensor_manager.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace cann_lm::engine {

// backend 类型别名经 core/tensor_manager.h 引入（ITensorBuffer/IBuiltModel/...）；
// NDTensorDesc 在 engine 域别名为 TensorDesc，此处补 using 使裸名 NDTensorDesc 可见。
using hiai::NDTensorDesc;

// ════════════════════════════════════════════════════════════════════════
// Mock IBuiltModel — 实现 hiai::backend 全部纯虚方法
// ════════════════════════════════════════════════════════════════════════
class MockBuiltModel : public hiai::IBuiltModel {
public:
    // KV 型 IO 注入（非空则 GetInputTensorDescs/GetOutputTensorDescs 返回之）。
    // mock 在 Load 内创建、无法预置实例态，故用类级静态 + ResetMocks 清理。
    static std::vector<NDTensorDesc> inputs_override;
    static std::vector<NDTensorDesc> outputs_override;

    std::atomic<bool> fail_restore{false};
    mutable std::atomic<int> restore_count{0};

    // ===== Serialization =====
    hiai::Status RestoreFromBuffer(const std::shared_ptr<hiai::IBuffer>& buffer) override
    {
        restore_count.fetch_add(1);
        if (fail_restore.load())
            return hiai::FAILURE;
        if (!buffer || !buffer->GetData())
            return hiai::FAILURE;
        return hiai::SUCCESS;
    }
    hiai::Status RestoreFromFile(const char* file) override
    {
        restore_count.fetch_add(1);
        if (fail_restore.load())
            return hiai::FAILURE;
        if (file == nullptr || file[0] == '\0')
            return hiai::FAILURE;
        return hiai::SUCCESS;
    }

    // ===== IO Query =====
    std::vector<NDTensorDesc> GetInputTensorDescs() const override
    {
        if (!inputs_override.empty())
            return inputs_override; // KV 型 IO 测试注入
        // 6 个输入工位（见 docs/model_executor/INPUT_INFO.md）
        // 名称决定 InputAssembler 按 name→index 查找工位
        auto mk = [](const char* name, DataType dt, std::vector<int32_t> dims) {
            NDTensorDesc d;
            d.tensorName = name;
            d.dataType = dt;
            d.dims = std::move(dims);
            return d;
        };
        return {
            mk("input_embed", DataType::FLOAT16, {64}),      mk("embed_scales", DataType::FLOAT32, {64}),
            mk("attention_mask", DataType::FLOAT32, {1024}), mk("position_ids", DataType::INT32, {64}),
            mk("new_kv_cache_pos", DataType::INT32, {64}),   mk("kvCache", DataType::FLOAT32, {1024}),
        };
    }
    std::vector<NDTensorDesc> GetOutputTensorDescs() const override
    {
        if (!outputs_override.empty())
            return outputs_override; // KV 型 IO 测试注入
        NDTensorDesc d;
        d.dims = {1};
        d.dataType = DataType::FLOAT32;
        return {d}; // logits
    }

    // ===== Metadata =====
    void SetName(const std::string& /*name*/) override {}
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

private:
    hiai::CustomModelData custom_data_;
};

// ════════════════════════════════════════════════════════════════════════
// Mock IModelManager — 实现 hiai::backend 全部纯虚方法
// ════════════════════════════════════════════════════════════════════════
class MockModelManager : public hiai::IModelManager {
public:
    std::atomic<bool> fail_init{false};
    std::atomic<bool> fail_forward{false};
    std::atomic<bool> fail_initweights{false};
    mutable std::atomic<int> init_count{0};
    mutable std::atomic<int> getmodelid_count{0};
    mutable std::atomic<int> forward_count{0};
    mutable std::atomic<int> deinit_count{0};
    mutable std::atomic<int> initweights_count{0};
    mutable std::atomic<int> deinitweights_count{0};
    mutable std::atomic<int> in_flight{0};
    mutable std::atomic<int> max_in_flight{0};
    std::shared_ptr<hiai::IBuiltModel> built_;
    uint32_t model_id_ = 1;

    // ===== Lifecycle =====
    hiai::Status Init(const hiai::ModelInitOptions& /*opts*/, const std::shared_ptr<hiai::IBuiltModel>& built,
                      const std::shared_ptr<hiai::IModelManagerListener>& /*listener*/) override
    {
        init_count.fetch_add(1);
        if (fail_init.load())
            return hiai::FAILURE;
        built_ = built;
        return hiai::SUCCESS;
    }
    void DeInit() override
    {
        deinit_count.fetch_add(1);
        built_.reset();
    }

    // ===== Execution =====
    // Run 末次调用参数捕获（KV 绑定端到端断言用；裸指针仅在对应 Load 生命周期内有效）
    static std::vector<const hiai::INDTensorBuffer*> last_run_inputs;
    static std::vector<const hiai::INDTensorBuffer*> last_run_outputs;

    hiai::Status Run(const std::vector<std::shared_ptr<hiai::INDTensorBuffer>>& inputs,
                     std::vector<std::shared_ptr<hiai::INDTensorBuffer>>& outputs) override
    {
        last_run_inputs.clear();
        for (const auto& b : inputs)
            last_run_inputs.push_back(b.get());
        last_run_outputs.clear();
        for (const auto& b : outputs)
            last_run_outputs.push_back(b.get());
        int cur = in_flight.fetch_add(1) + 1;
        int prev = max_in_flight.load();
        while (cur > prev) {
            if (max_in_flight.compare_exchange_weak(prev, cur))
                break;
        }
        forward_count.fetch_add(1);
        auto st = fail_forward.load() ? hiai::FAILURE : hiai::SUCCESS;
        in_flight.fetch_sub(1);
        return st;
    }

    // ===== Weight Management =====
    hiai::Status InitWeights(const std::string& w) override
    {
        initweights_count.fetch_add(1);
        if (fail_initweights.load() || w.empty())
            return hiai::FAILURE;
        return hiai::SUCCESS;
    }
    hiai::Status DeinitWeights() override
    {
        deinitweights_count.fetch_add(1);
        return hiai::SUCCESS;
    }
    hiai::Status GetWeightBuffer(const std::string&, std::shared_ptr<hiai::IBuffer>&) override
    {
        return hiai::UNSUPPORTED;
    }
    hiai::Status FlushWeight(const std::string&, size_t, size_t) override { return hiai::UNSUPPORTED; }
    hiai::Status UpdateWeight(const std::string&, size_t, size_t, const void*) override { return hiai::UNSUPPORTED; }

    // ===== Info / Memory（预留通路，引擎未调用）=====
    uint32_t GetModelID() override
    {
        getmodelid_count.fetch_add(1);
        return model_id_;
    }
    hiai::Status ReleaseMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    hiai::Status RecoverMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }

    bool has_built() const { return static_cast<bool>(built_); }
};

// ════════════════════════════════════════════════════════════════════════
// 工厂函数覆盖（hiai 命名空间，符号优先于库内 factory_impl.cpp）
// ════════════════════════════════════════════════════════════════════════
namespace {
std::vector<MockBuiltModel*> g_built_mocks;
std::vector<MockModelManager*> g_mocks;
MockBuiltModel* g_last_built = nullptr;
MockModelManager* g_last_mock = nullptr;
} // namespace

// mock 静态成员定义（KV 型 IO 注入通道）
std::vector<NDTensorDesc> MockBuiltModel::inputs_override;
std::vector<NDTensorDesc> MockBuiltModel::outputs_override;
std::vector<const hiai::INDTensorBuffer*> MockModelManager::last_run_inputs;
std::vector<const hiai::INDTensorBuffer*> MockModelManager::last_run_outputs;

void ResetMocks()
{
    g_built_mocks.clear();
    g_mocks.clear();
    g_last_built = nullptr;
    g_last_mock = nullptr;
    MockBuiltModel::inputs_override.clear();
    MockBuiltModel::outputs_override.clear();
    MockModelManager::last_run_inputs.clear();
    MockModelManager::last_run_outputs.clear();
}
} // namespace cann_lm::engine

// ── hiai 命名空间：覆盖模型生命周期工厂（CreateBuiltModel/CreateModelManager）──
// 张量创建（hiai::CreateNDTensorBuffer）不再在此 ODR 顶替：退役 HostTensorBuffer
// mock 后，链期解析到 cann_backend 真 impl（CreateNDTensorBufferFromNDK → NDKProxy
// → dlopen stub .so libneural_network_core.so），host UT 经 CANN_LM_NDK_LIB_DIR
// 命中 build 目录 stub（见 cann_lm/engine/CMakeLists.txt + ndk_proxy.cpp）。模型
// 生命周期仍由 MockBuiltModel/MockModelManager 注入（Load 的 Restore/Init/GetDescs
// 不触达 NDK），故 stub 的 canned executor descs 不影响本目标。
namespace hiai {

std::shared_ptr<IBuiltModel> CreateBuiltModel()
{
    auto* raw = new cann_lm::engine::MockBuiltModel();
    cann_lm::engine::g_built_mocks.push_back(raw);
    cann_lm::engine::g_last_built = raw;
    return std::shared_ptr<IBuiltModel>(raw);
}

std::shared_ptr<IModelManager> CreateModelManager()
{
    auto* raw = new cann_lm::engine::MockModelManager();
    cann_lm::engine::g_mocks.push_back(raw);
    cann_lm::engine::g_last_mock = raw;
    return std::shared_ptr<IModelManager>(raw);
}

} // namespace hiai

// ════════════════════════════════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════════════════════════════════
namespace cann_lm::engine {
namespace {

// SR20260903001：单图位双态源（D-15）+ 合法配置 helper（kv_cache_max_len/num_hidden_layers）
ArtifactSource MakeOmc(bool om_valid = true)
{
    ArtifactSource s;
    static char omc_buf[8];
    if (om_valid) {
        s.data = Buffer{omc_buf, sizeof(omc_buf)};
    }
    return s;
}

ModelConfig MakeConfig(int kv_cache_max_len = 16, int num_hidden_layers = 2, size_t vocab_align_size = 0)
{
    ModelConfig c;
    c.kv_cache_max_len = kv_cache_max_len;
    c.num_hidden_layers = num_hidden_layers;
    c.vocab_align_size = vocab_align_size;
    return c;
}

constexpr const char* kWeight = "/valid/weight"; // weight 为路径（图与权重分离）

// seq_len=0：跳过 CorrectTensorDescs（保留 om 原始 dims）——与旧 Load（无 seq_len
// 修正通路）行为逐值一致；KV 型测试的显式 dims {16,4,4,8} / logits {4,16,128}
// 不被占位修正改写，池绑定与 logits shape 定位逻辑复刻原意。
constexpr int kNoCorrectSeqLen = 0;

struct ExecFixture {
    std::shared_ptr<KvCachePool> pool;
    std::shared_ptr<InputAssembler> asm_;
    std::shared_ptr<DfxCollector> dfx;
    std::unique_ptr<ModelExecutor> exec;
    ExecFixture(int max_batch_size = 4, int max_seq_len = 16)
        : pool(std::make_shared<KvCachePool>(max_batch_size, max_seq_len, 2, 4, 8)),
          asm_(std::make_shared<InputAssembler>(pool, max_batch_size)),
          dfx(std::make_shared<DfxCollector>(DfxConfig{}, pool))
    {
        ResetMocks();
        exec = std::make_unique<ModelExecutor>(pool, asm_, dfx);
    }
};

struct LoadedFixture : ExecFixture {
    ModelConfig config;
    LoadedFixture(int max_batch_size = 4, int max_seq_len = 16)
        : ExecFixture(max_batch_size, max_seq_len), config(MakeConfig(max_seq_len))
    {
        (void)exec->Load(MakeOmc(), kWeight, config, kNoCorrectSeqLen);
    }
};

// ── 正常路径 ──

TEST(ModelExecutorTest, LoadBufferAndInitAndGetConfig_TC005)
{
    ExecFixture f;
    ModelConfig cfg = MakeConfig();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, cfg, kNoCorrectSeqLen).ok());
    ModelConfig out{};
    ASSERT_TRUE(f.exec->GetConfig(&out).ok());
    EXPECT_EQ(out.kv_cache_max_len, 16);
    EXPECT_EQ(out.num_hidden_layers, 2);
    EXPECT_TRUE(static_cast<bool>(*f.exec));
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_GT(g_last_mock->init_count.load(), 0);
    EXPECT_GT(g_last_mock->getmodelid_count.load(), 0);
    EXPECT_TRUE(g_last_mock->has_built());
    EXPECT_EQ(f.exec->InputTensorCount(), 6u);
    EXPECT_EQ(f.exec->OutputTensorCount(), 1u);
}

TEST(ModelExecutorTest, LoadValidPathAndInit_TC005path)
{
    ExecFixture f;
    ArtifactSource p;
    p.path = "/valid/omc";
    ASSERT_TRUE(f.exec->Load(p, kWeight, MakeConfig(), kNoCorrectSeqLen).ok());
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_GT(g_last_mock->init_count.load(), 0);
}

TEST(ModelExecutorTest, PrefillForwardAppendsKvAndReturnsLogits_TC001)
{
    LoadedFixture f;
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 5;
    opts.slots = {slot};
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.pool->NextPosition(slot), 5u);
}

TEST(ModelExecutorTest, ChunkedPrefillAccumulatesCursor_TC007)
{
    LoadedFixture f;
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 3;
    opts.slots = {slot};
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_EQ(f.pool->NextPosition(slot), 3u);
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_EQ(f.pool->NextPosition(slot), 6u);
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_EQ(f.pool->NextPosition(slot), 9u);
}

TEST(ModelExecutorTest, DecodeForwardPaddedReturnsLogits_TC002_008)
{
    LoadedFixture f;
    std::vector<SlotId> slots;
    for (int i = 0; i < 2; ++i) {
        SlotId s = -1;
        ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
        slots.push_back(s);
    }
    ForwardOptions opts;
    opts.batch = 2;
    opts.slots = slots;
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_TRUE(f.exec->Logits());
    for (SlotId s : slots)
        EXPECT_EQ(f.pool->NextPosition(s), 1u);
}

TEST(ModelExecutorTest, SingleTokenDecodeAdvancesCursorByOne_TC204)
{
    LoadedFixture f;
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {slot};
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    EXPECT_EQ(f.pool->NextPosition(slot), 1u);
}

TEST(ModelExecutorTest, DISABLED_TokenIdAndEmbeddingPathsConsistent_TC003)
{
    GTEST_SKIP() << "TC_003 需真实 om+权重 oracle。";
}

// ── 异常路径 ──

TEST(ModelExecutorTest, MissingOmLoadFails_TC101)
{
    ExecFixture f;
    ArtifactSource a = MakeOmc(/*om_valid=*/false);
    EXPECT_EQ(f.exec->Load(a, kWeight, MakeConfig(), kNoCorrectSeqLen).code, Status::FAILURE);
    EXPECT_FALSE(static_cast<bool>(*f.exec));
}

TEST(ModelExecutorTest, MissingOmPathLoadFails_TC101path)
{
    ExecFixture f;
    ArtifactSource p; // 双态均空 → 无图
    EXPECT_EQ(f.exec->Load(p, kWeight, MakeConfig(), kNoCorrectSeqLen).code, Status::FAILURE);
}

TEST(ModelExecutorTest, MissingMaxBatchSizeFails_TC106)
{
    ExecFixture f;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(/*kv_cache_max_len=*/0), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// 未加载 Forward：生产 A2 决策返 FAILURE（非 INVALID_STATE）——
// model_executor.cpp Forward 注释「bbit 契约只认 SUCCESS/FAILURE」。
TEST(ModelExecutorTest, ForwardBeforeLoadRejected_TC102)
{
    ExecFixture f;
    ForwardOptions opts;
    opts.batch = 1;
    opts.slots = {0};
    EXPECT_EQ(f.exec->Forward(opts).code, Status::FAILURE);
}

TEST(ModelExecutorTest, BackendForwardFailureMapsToFailure_TC103)
{
    LoadedFixture f;
    ASSERT_NE(g_last_mock, nullptr);
    g_last_mock->fail_forward.store(true);
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {slot};
    EXPECT_EQ(f.exec->Forward(opts).code, Status::FAILURE);
}

// ── 并发 ──

TEST(ModelExecutorTest, ConcurrentForwardSerialized_TC301)
{
    LoadedFixture f(/*max_batch_size=*/4, /*max_seq_len=*/256);
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {slot};
    std::atomic<int> ok_count{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t) {
        ts.emplace_back([&]() {
            for (int i = 0; i < 25; ++i) {
                if (f.exec->Forward(opts).ok() && f.exec->Logits())
                    ok_count.fetch_add(1);
            }
        });
    }
    for (auto& th : ts)
        th.join();
    EXPECT_EQ(ok_count.load(), 100);
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_EQ(g_last_mock->max_in_flight.load(), 1);
}

TEST(ModelExecutorTest, MultiInstancePoolsAndMocksIsolated_TC303)
{
    ExecFixture a(/*max_batch_size=*/4);
    (void)a.exec->Load(MakeOmc(), kWeight, MakeConfig(), kNoCorrectSeqLen);
    MockModelManager* mock_a = g_last_mock;

    ExecFixture b(/*max_batch_size=*/4);
    (void)b.exec->Load(MakeOmc(), kWeight, MakeConfig(), kNoCorrectSeqLen);
    MockModelManager* mock_b = g_last_mock;

    EXPECT_NE(a.pool.get(), b.pool.get());
    ASSERT_NE(mock_a, nullptr);
    ASSERT_NE(mock_b, nullptr);
    EXPECT_NE(mock_a, mock_b);

    SlotId sa = 0;
    a.pool->AllocKvSlot(&sa);
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {sa};
    a.exec->Forward(opts);
    EXPECT_GT(mock_a->forward_count.load(), mock_b->forward_count.load());
}

// ════════════════════════════════════════════════════════════════════════
// KV 工位绑定：输入 past_key/value_in<层> 按名识别、输出按 shape（dims+dtype）
// 识别，绑同一池 tensor（om 原地更新）；meta_json ↔ om 校验失配 → 创建失败。
// 池基准：ExecFixture 默认池 (batch=4, kv_len=16, 层 2, kv_heads=4, dim=8, FP16)
// → KV tensor dims HF 序 {batch,heads,kv_len,dim}={4,4,16,8}（R3：图 desc 须 HF 序逐位
// 比对 meta，2026-10-15 裁定；lazy-create 下 Load 期 EnsureKvTensor 按图 desc 直建）；
// logits {batch, seqLen, vocab_align=128}。
// ════════════════════════════════════════════════════════════════════════

NDTensorDesc MkDesc(const char* name, DataType dt, std::vector<int32_t> dims)
{
    NDTensorDesc d;
    d.tensorName = name;
    d.dataType = dt;
    d.dims = std::move(dims);
    return d;
}

/// KV 型输入集：5 常规工位 + 4 KV 工位（层 0/1，K 前 V 后）
std::vector<NDTensorDesc> KvModeInputs()
{
    const auto kv = DataType::FLOAT16;
    const std::vector<int32_t> kv_dims = {4, 4, 16, 8};
    return {
        MkDesc("input_embed", DataType::FLOAT16, {64}),
        MkDesc("embed_scales", DataType::FLOAT32, {64}),
        MkDesc("attention_mask", DataType::FLOAT32, {1024}),
        MkDesc("position_ids", DataType::INT32, {64}),
        MkDesc("new_kv_cache_pos", DataType::INT32, {64}),
        MkDesc("past_key_in0", kv, kv_dims),
        MkDesc("past_value_in0", kv, kv_dims),
        MkDesc("past_key_in1", kv, kv_dims),
        MkDesc("past_value_in1", kv, kv_dims),
    };
}

/// KV 型输出集：4 个任意名 KV 形输出（层序 K 前 V 后）+ logits 末位
/// （末位放置验证 Logits() 按 shape 定位、非固定 [0]）
std::vector<NDTensorDesc> KvModeOutputs()
{
    const auto kv = DataType::FLOAT16;
    const std::vector<int32_t> kv_dims = {4, 4, 16, 8};
    return {
        MkDesc("out_kv_a", kv, kv_dims),
        MkDesc("out_kv_b", kv, kv_dims),
        MkDesc("out_kv_c", kv, kv_dims),
        MkDesc("out_kv_d", kv, kv_dims),
        MkDesc("logits", DataType::FLOAT32, {4, 16, 128}),
    };
}

// 零拷贝载体：两执行器共享同一池对象；prefill Load 期 EnsureKvTensor 惰性建池
// tensor，decode（共享同池）即时见同组 tensor（跨 executor 的 KV 工位绑定由
// LoadBindsKvInputsByNameNotPosition 等用例钉死）。R1/T1 lazy-create：ctor 后
// TensorCount()==0，Load 后 ==2×NumKvLayers()。
TEST(ModelExecutorTest, SharedKvPoolZeroCopy_TC004structural)
{
    auto pool = std::make_shared<KvCachePool>(4, 16, 2, 4, 8);
    auto asm_ = std::make_shared<InputAssembler>(pool, 4);
    auto dfx = std::make_shared<DfxCollector>(DfxConfig{}, pool);
    ResetMocks();
    ModelExecutor prefill(pool, asm_, dfx);
    ModelExecutor decode(pool, asm_, dfx);
    EXPECT_EQ(prefill.Pool().get(), decode.Pool().get());
    EXPECT_EQ(pool->TensorCount(), 0u);
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(prefill.Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    EXPECT_EQ(pool->TensorCount(), 4u);
    EXPECT_EQ(prefill.Pool()->KTensor(0).get(), decode.Pool()->KTensor(0).get());
    EXPECT_EQ(prefill.Pool()->VTensor(1).get(), decode.Pool()->VTensor(1).get());
}

// 绑定同一性：输入按名、输出按 shape 层序，全部绑同一池 tensor（不新申请）
TEST(ModelExecutorTest, LoadBindsPoolTensorsToKvInputsAndOutputs)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    // 工位数不变（KV 槽位存在，只是内容为池 tensor）
    EXPECT_EQ(f.exec->InputTensorCount(), 9u);
    EXPECT_EQ(f.exec->OutputTensorCount(), 5u);
    // 输入侧：past_key/value_in<层> → 池 K/V tensor（同底层句柄）
    EXPECT_EQ(f.exec->InputTensors()[5].get(), f.pool->KTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[6].get(), f.pool->VTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[7].get(), f.pool->KTensor(1).get());
    EXPECT_EQ(f.exec->InputTensors()[8].get(), f.pool->VTensor(1).get());
    // 输出侧：按 shape 识别 + 层序 [k0,v0,k1,v1] 绑同一池 tensor（原地更新）
    EXPECT_EQ(f.exec->OutputTensors()[0].get(), f.pool->KTensor(0).get());
    EXPECT_EQ(f.exec->OutputTensors()[1].get(), f.pool->VTensor(0).get());
    EXPECT_EQ(f.exec->OutputTensors()[2].get(), f.pool->KTensor(1).get());
    EXPECT_EQ(f.exec->OutputTensors()[3].get(), f.pool->VTensor(1).get());
    // Logits() 按 shape 定位（末位 [batch,seqLen,vocab_align]，非 output[0]）
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.exec->Logits()->GetTensorDesc().dims, (std::vector<int32_t>{4, 16, 128}));
    EXPECT_NE(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());
}

// meta_json ↔ om 校验：KV 输入 desc dims 与池 tensor 不符 → 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvDimsMismatch)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in0", DataType::FLOAT16, {4, 4, 8, 8}); // kv_len 8 (dims[2]) ≠ 池 16
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// meta_json ↔ om 校验：dtype 不符（om fp32 vs 池 fp16）→ 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvDtypeMismatch)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in0", DataType::FLOAT32, {4, 4, 16, 8}); // dtype 异、dims 同
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// meta_json ↔ om 校验：om KV 工位数 ≠ 2×层数（只有层 0）→ 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvSlotCountMismatch)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in.resize(7); // 5 常规 + 仅 past_key_in0/past_value_in0
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// 输出侧 KV 形工位数与输入侧不等 → 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvOutputCountMismatch)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    auto out = KvModeOutputs();
    out.resize(3); // 仅 2 个 KV 形 + logits
    MockBuiltModel::outputs_override = out;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// 池无可绑 KV 层 → Load 资源类 FAILURE。
// 2b99be2 R1/T1 lazy-create 后 ctor 即 TensorCount()==0 对所有池成立，旧"kv_heads=0
// → 预缺 tensor"前提作废；改以 layer_types 全 linear_attention（kv_layer_ids_ 空）
// 模拟"池无建池层"：om 声明 KV 工位时 EnsureKvTensor 命中 LayerPos<0 → FAILURE。
TEST(ModelExecutorTest, LoadFailsWhenPoolMissingKvTensors)
{
    const std::vector<std::string> all_linear{"linear_attention", "linear_attention"};
    auto pool = std::make_shared<KvCachePool>(4, 16, 2, 4, 8, all_linear);
    auto asm_ = std::make_shared<InputAssembler>(pool, 4);
    auto dfx = std::make_shared<DfxCollector>(DfxConfig{}, pool);
    ResetMocks();
    ModelExecutor exec(pool, asm_, dfx);
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(exec.Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code, Status::FAILURE);
}

// 池未挂载（pool=nullptr）但 om 声明 KV 工位 → 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvSlotsWithoutPool)
{
    ExecFixture f;
    ModelExecutor exec(/*pool=*/nullptr, f.asm_, f.dfx);
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(exec.Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code, Status::FAILURE);
}

// 层号越界（past_key_in5，池仅 2 层）→ 创建失败
TEST(ModelExecutorTest, LoadFailsOnOutOfRangeLayerInKvName)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[7] = MkDesc("past_key_in5", DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code, Status::FAILURE);
}

// 前缀命中但层数后缀非法（past_key_in_x）→ 创建失败（配置类）
TEST(ModelExecutorTest, LoadFailsOnMalformedKvName)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in_x", DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// 输入侧按名绑定钉死（H-1）：KV 工位乱序排列，绑定须按名字解析层号、非按位置
TEST(ModelExecutorTest, LoadBindsKvInputsByNameNotPosition)
{
    ExecFixture f;
    auto in = KvModeInputs();
    // 打乱 KV 段：[v1, k0, v0, k1]——与层序完全错开
    in[5] = MkDesc("past_value_in1", DataType::FLOAT16, {4, 4, 16, 8});
    in[6] = MkDesc("past_key_in0", DataType::FLOAT16, {4, 4, 16, 8});
    in[7] = MkDesc("past_value_in0", DataType::FLOAT16, {4, 4, 16, 8});
    in[8] = MkDesc("past_key_in1", DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    EXPECT_EQ(f.exec->InputTensors()[5].get(), f.pool->VTensor(1).get()); // 名字说了算
    EXPECT_EQ(f.exec->InputTensors()[6].get(), f.pool->KTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[7].get(), f.pool->VTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[8].get(), f.pool->KTensor(1).get());
}

// 输出识别 dtype 参与判定（M-1a）：dims 相同但 dtype 不同 → 不识别为 KV → 数量失配
TEST(ModelExecutorTest, LoadFailsWhenKvOutputDtypeDiffers)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    auto out = KvModeOutputs();
    out[1] = MkDesc("out_kv_b", DataType::FLOAT32, {4, 4, 16, 8}); // 同 dims 异 dtype
    MockBuiltModel::outputs_override = out;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// kv_om 闸门（M-1b）：非 KV 型 om（无 past_* 输入）下 KV 形输出不做池绑定
TEST(ModelExecutorTest, KvShapedOutputNotBoundForNonKvOmc)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = {
        // 常规 5 工位（无 past_*）→ 非 KV 型 om
        MkDesc("input_embed", DataType::FLOAT16, {64}),      MkDesc("embed_scales", DataType::FLOAT32, {64}),
        MkDesc("attention_mask", DataType::FLOAT32, {1024}), MkDesc("position_ids", DataType::INT32, {64}),
        MkDesc("new_kv_cache_pos", DataType::INT32, {64}),
    };
    MockBuiltModel::outputs_override = {
        MkDesc("logits", DataType::FLOAT32, {4, 16, 128}),
        MkDesc("mystery", DataType::FLOAT16, {4, 4, 16, 8}), // KV 形但 om 非 KV 型
    };
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    // 不绑池：独立申请的 tensor ≠ 任何池 tensor
    EXPECT_NE(f.exec->OutputTensors()[1].get(), f.pool->KTensor(0).get());
    EXPECT_NE(f.exec->OutputTensors()[1].get(), f.pool->VTensor(1).get());
    // Logits() 仍按 shape 正常定位
    EXPECT_EQ(f.exec->Logits()->GetTensorDesc().dims, (std::vector<int32_t>{4, 16, 128}));
}

// logits_index_ 移动语义（M-2a）：move 后新对象保留定位、源对象失效
TEST(ModelExecutorTest, LogitsIndexSurvivesMove)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    ModelExecutor moved(std::move(*f.exec));
    ASSERT_TRUE(moved.Logits());
    EXPECT_EQ(moved.Logits()->GetTensorDesc().dims, (std::vector<int32_t>{4, 16, 128}));
    EXPECT_FALSE(f.exec->Logits()); // 源对象输出已搬空
}

// logits_index_ 生命周期（M-2b）：Unload → 重 Load 后 index 重置回 0
TEST(ModelExecutorTest, LogitsIndexResetOnReload)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_NE(f.exec->Logits().get(), f.exec->OutputTensors()[0].get()); // index=4
    ASSERT_TRUE(f.exec->Unload().ok());
    MockBuiltModel::inputs_override.clear(); // 回默认（非 KV 型）descs
    MockBuiltModel::outputs_override.clear();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get()); // index 重置
}

// KV 型 om 端到端（M-3）：Forward 送入 Run 的 KV 槽位就是池 tensor（原地更新契约）
TEST(ModelExecutorTest, ForwardPassesPoolTensorsToRun)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kNoCorrectSeqLen).ok());
    SlotId slot = 0;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {slot};
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    ASSERT_EQ(MockModelManager::last_run_inputs.size(), 9u);
    EXPECT_EQ(MockModelManager::last_run_inputs[5], f.pool->KTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[6], f.pool->VTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[7], f.pool->KTensor(1).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[8], f.pool->VTensor(1).get());
    ASSERT_EQ(MockModelManager::last_run_outputs.size(), 5u);
    EXPECT_EQ(MockModelManager::last_run_outputs[0], f.pool->KTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_outputs[3], f.pool->VTensor(1).get());
    EXPECT_EQ(MockModelManager::last_run_outputs[4], f.exec->Logits().get());
}

// Logits() 回退：vocab_align_size 未供给（=0，桩/mock 场景）→ 回退 output[0]
TEST(ModelExecutorTest, LogitsFallsBackToFirstOutputWhenNoVocabAlign)
{
    ExecFixture f;
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(),
                             kNoCorrectSeqLen).ok()); // 默认 descs
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());
}

// Logits() 回退：vocab_align 供给但无 3D 末维匹配 → WARN 降级回退 output[0]
TEST(ModelExecutorTest, LogitsFallsBackWhenNoVocabShapedOutput)
{
    ExecFixture f;
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128),
                             kNoCorrectSeqLen).ok()); // 默认输出 {1}
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());
}

} // namespace
} // namespace cann_lm::engine
