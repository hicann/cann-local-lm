// TEST-WRITE(GREEN)：ModelExecutor 单元测试（v0.13 迁移：hiai NDK backend 代际）。
// Mock hiai::IBuiltModel/IModelManager 全部纯虚方法；链接期符号覆盖
// hiai::CreateModelManager/CreateBuiltModel 两工厂（同 ut_scheduler_multichunk/
// ut_eos_stop 手法，符号优先于 cann_lm_engine 静态库内 NDKProxy 通路）；
// tensor 申请不覆盖——经真实 stub NDK so（CANN_LM_NDK_LIB_DIR，注册同
// ut_engine_executor），TensorManager/池/assemble 链路测真路径。
// 迁移要点（原 cann_llm_engine 接口代际 → hiai 代际）：
//   - Load 增第 4 参 seq_len（>1=prefill 修形 / ==1=decode）；A2：未加载
//     Forward 返 FAILURE（非 INVALID_STATE）
//   - KV 绑定语义 ADR-0002 重构：Load 期 EnsureKvDesc 只记池 desc 账本 +
//     工位空占位（不建 tensor）；实建在 EnsureKvTensor（Context 池），
//     RebindKvPool 把空占位填实、DetachKvPool 回空——原"Load 即绑池 tensor"
//     组用例按新契约改写
//   - KV 图 desc 轴序 HF [batch,heads,kv_len,head_dim]（R3 比对序；旧用例
//     [len,heads,batch,dim] 会被 kv_len 维失配致命拒绝）
//   - logits_index_ Q13 裁定恒钉 output[0]（原"按 shape 扫描定位/回退"组
//     收敛为单一定位契约）
//   - SlotInfo::cursor → total_written（R4 分离记账）；KvCachePool 构造无
//     dtype 参（tensor 实形唯一权威 = om 图工位 desc，R1）
#include "core/component/model_executor/model_executor.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/dfx_collector/dfx_collector.h"
#include "core/tensor_manager.h"
#include "model/built_model.h"           // hiai::IBuiltModel（覆盖 CreateBuiltModel 用）
#include "model_manager/model_manager.h" // hiai::IModelManager（覆盖 CreateModelManager 用）
#include "infra/status.h"
#include "api/cpp/types.h"

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace cann_local_lm::engine {

// ════════════════════════════════════════════════════════════════════════
// Mock hiai::IBuiltModel — 实现接口全部纯虚方法
// ════════════════════════════════════════════════════════════════════════
class MockBuiltModel : public hiai::IBuiltModel {
public:
    // KV 型 IO 注入（非空则 GetInputTensorDescs/GetOutputTensorDescs 返回之）。
    // mock 在 Load 内创建、无法预置实例态，故用类级静态 + ResetMocks 清理。
    static std::vector<hiai::NDTensorDesc> inputs_override;
    static std::vector<hiai::NDTensorDesc> outputs_override;

    // restore 失败注入旗（一次性）：工厂在 Load 内新建 mock，实例旗无处预置
    // ——经"下只必失败"类级静态，构造时原子取走消费。
    static std::atomic<bool> fail_restore_next;

    std::atomic<bool> fail_restore{false};
    mutable std::atomic<int> restore_count{0};

    MockBuiltModel() : fail_restore(fail_restore_next.exchange(false)) {}

    // ===== Serialization =====
    hiai::Status RestoreFromBuffer(const std::shared_ptr<hiai::IBuffer>& buffer) override
    {
        restore_count.fetch_add(1);
        if (fail_restore.load() || !buffer || !buffer->GetData())
            return hiai::FAILURE;
        return hiai::SUCCESS;
    }
    hiai::Status RestoreFromFile(const char* file) override
    {
        restore_count.fetch_add(1);
        if (fail_restore.load() || file == nullptr || *file == '\0')
            return hiai::FAILURE;
        return hiai::SUCCESS;
    }

    // ===== Metadata =====
    void SetName(const std::string&) override {}
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

    // ===== IO Query =====
    std::vector<hiai::NDTensorDesc> GetInputTensorDescs() const override
    {
        if (!inputs_override.empty())
            return inputs_override; // KV 型 IO 测试注入
        // 6 个输入工位：名称决定 InputAssembler 按 name→index 查找工位
        // （CorrectTensorDescs 按 seq_len 修形 position_ids/new_kv_cache_pos/
        // attention_mask；input_embed 无 hidden_size 不修；embed_scales/kvCache
        // 非知名工位原样过）
        auto mk = [](const char* name, hiai::DataType dt, std::vector<int32_t> dims) {
            hiai::NDTensorDesc d;
            d.tensorName = name;
            d.dataType = dt;
            d.dims = std::move(dims);
            return d;
        };
        return {
            mk("input_embed", hiai::DataType::FLOAT16, {64}),
            mk("embed_scales", hiai::DataType::FLOAT32, {64}),
            mk("attention_mask", hiai::DataType::FLOAT32, {1, 1, 1, 16}),
            mk("position_ids", hiai::DataType::INT32, {1, 1}),
            mk("new_kv_cache_pos", hiai::DataType::INT32, {1}),
            mk("kvCache", hiai::DataType::FLOAT32, {1, 1, 1, 16}),
        };
    }
    std::vector<hiai::NDTensorDesc> GetOutputTensorDescs() const override
    {
        if (!outputs_override.empty())
            return outputs_override; // KV 型 IO 测试注入
        hiai::NDTensorDesc d;
        d.tensorName = "logits";
        d.dataType = hiai::DataType::FLOAT32;
        d.dims = {1};
        return {d}; // logits
    }

private:
    hiai::CustomModelData custom_data_;
};

// ════════════════════════════════════════════════════════════════════════
// Mock hiai::IModelManager — 实现接口全部纯虚方法
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
    uint32_t last_share_weights_id = 0; // Init 实参捕获（权重共享接线断言）
    uint32_t model_id_ = 1;

    // ===== Lifecycle =====
    hiai::Status Init(const hiai::ModelInitOptions& opts, const std::shared_ptr<hiai::IBuiltModel>& built,
                      const std::shared_ptr<hiai::IModelManagerListener>&) override
    {
        init_count.fetch_add(1);
        if (fail_init.load())
            return hiai::FAILURE;
        built_ = built;
        last_share_weights_id = opts.shareWeightsModelId;
        return hiai::SUCCESS;
    }
    void DeInit() override
    {
        deinit_count.fetch_add(1);
        built_.reset();
    }

    // ===== Execution =====
    // Run 末次调用参数捕获（KV 绑定端到端断言用；裸指针仅在对应 Load 生命周期内有效）
    static std::vector<const ITensorBuffer*> last_run_inputs;
    static std::vector<const ITensorBuffer*> last_run_outputs;

    hiai::Status Run(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                     std::vector<std::shared_ptr<ITensorBuffer>>& outputs) override
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
        return hiai::UNSUPPORTED; // 引擎链路未调用（预留通路）
    }
    hiai::Status FlushWeight(const std::string&, size_t, size_t) override
    {
        return hiai::UNSUPPORTED; // 引擎链路未调用（预留通路）
    }
    hiai::Status UpdateWeight(const std::string&, size_t, size_t, const void*) override
    {
        return hiai::UNSUPPORTED; // 引擎链路未调用（预留通路）
    }

    // ===== Priority / Info =====
    hiai::Status ReleaseMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    hiai::Status RecoverMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    uint32_t GetModelID() override
    {
        getmodelid_count.fetch_add(1);
        return model_id_;
    }

    bool has_built() const { return static_cast<bool>(built_); }
};

// ════════════════════════════════════════════════════════════════════════
// 工厂函数覆盖（hiai 命名空间，符号优先于库内 NDKProxy 通路）
// ════════════════════════════════════════════════════════════════════════
namespace {
// mock 保活池：ReleaseLocked/重 Load 会释放 executor 持有的唯一引用，
// 工厂侧留 shared_ptr 副本使裸指针（g_last_* / 用例局部捕获）在测试全程
// 恒有效（否则悬垂——单独跑碰巧过、全量跑堆布局漂移后读脏值）。
std::vector<std::shared_ptr<MockBuiltModel>> g_built_refs;
std::vector<std::shared_ptr<MockModelManager>> g_mock_refs;
std::vector<MockBuiltModel*> g_built_mocks;
std::vector<MockModelManager*> g_mocks;
MockBuiltModel* g_last_built = nullptr;
MockModelManager* g_last_mock = nullptr;
} // namespace

// mock 静态成员定义（KV 型 IO 注入 + 失败注入通道）
std::vector<hiai::NDTensorDesc> MockBuiltModel::inputs_override;
std::vector<hiai::NDTensorDesc> MockBuiltModel::outputs_override;
std::atomic<bool> MockBuiltModel::fail_restore_next{false};
std::vector<const ITensorBuffer*> MockModelManager::last_run_inputs;
std::vector<const ITensorBuffer*> MockModelManager::last_run_outputs;

void ResetMocks()
{
    g_built_refs.clear();
    g_mock_refs.clear();
    g_built_mocks.clear();
    g_mocks.clear();
    g_last_built = nullptr;
    g_last_mock = nullptr;
    MockBuiltModel::inputs_override.clear();
    MockBuiltModel::outputs_override.clear();
    MockBuiltModel::fail_restore_next.store(false);
    MockModelManager::last_run_inputs.clear();
    MockModelManager::last_run_outputs.clear();
}
} // namespace cann_local_lm::engine

// ── hiai 命名空间：覆盖工厂函数（CreateNDTensorBuffer/CreateLocalBuffer 不
//    覆盖——tensor 申请走真实 stub NDK so，TensorManager/池链路测真路径）──
namespace hiai {

std::shared_ptr<IBuiltModel> CreateBuiltModel()
{
    auto sp = std::make_shared<cann_local_lm::engine::MockBuiltModel>();
    cann_local_lm::engine::g_built_refs.push_back(sp); // 保活（见 g_built_refs 注释）
    cann_local_lm::engine::g_built_mocks.push_back(sp.get());
    cann_local_lm::engine::g_last_built = sp.get();
    return sp;
}

std::shared_ptr<IModelManager> CreateModelManager()
{
    auto sp = std::make_shared<cann_local_lm::engine::MockModelManager>();
    cann_local_lm::engine::g_mock_refs.push_back(sp); // 保活（Release 后裸指针仍有效）
    cann_local_lm::engine::g_mocks.push_back(sp.get());
    cann_local_lm::engine::g_last_mock = sp.get();
    return sp;
}

} // namespace hiai

// ════════════════════════════════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════════════════════════════════
namespace cann_local_lm::engine {
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
constexpr int kLoadSeqLen = 16;                  // Load 修形 seq_len（>1 = prefill 形态）

struct ExecFixture {
    std::shared_ptr<KvCachePool> pool;
    std::shared_ptr<InputAssembler> asm_;
    std::shared_ptr<DfxCollector> dfx;
    std::unique_ptr<ModelExecutor> exec;
    // 池构造无 dtype 参（R1：tensor 实形权威 = om 图工位 desc；构造期不建 tensor，T1）
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
        (void)exec->Load(MakeOmc(), kWeight, config, kLoadSeqLen);
    }
};

// ── 正常路径 ──

TEST(ModelExecutorTest, LoadBufferAndInitAndGetConfig_TC005)
{
    ExecFixture f;
    ModelConfig cfg = MakeConfig();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, cfg, kLoadSeqLen).ok());
    ModelConfig out{};
    ASSERT_TRUE(f.exec->GetConfig(&out).ok());
    EXPECT_EQ(out.kv_cache_max_len, 16);
    EXPECT_EQ(out.num_hidden_layers, 2);
    EXPECT_TRUE(static_cast<bool>(*f.exec));
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_GT(g_last_mock->init_count.load(), 0);
    EXPECT_GT(g_last_mock->getmodelid_count.load(), 0);
    EXPECT_GT(g_last_mock->initweights_count.load(), 0); // weight 非空 → InitWeights
    EXPECT_TRUE(g_last_mock->has_built());
    EXPECT_EQ(f.exec->InputTensorCount(), 6u);
    EXPECT_EQ(f.exec->OutputTensorCount(), 1u);
    EXPECT_EQ(f.exec->InputNameIndex().count("position_ids"), 1u); // name→index 建立
}

TEST(ModelExecutorTest, LoadValidPathAndInit_TC005path)
{
    ExecFixture f;
    ArtifactSource p;
    p.path = "/valid/omc";
    ASSERT_TRUE(f.exec->Load(p, kWeight, MakeConfig(), kLoadSeqLen).ok());
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_GT(g_last_mock->init_count.load(), 0);
}

// buffer 态 Restore 失败（D-15 通路）→ Load FAILURE
TEST(ModelExecutorTest, RestoreFailureFailsLoad)
{
    ExecFixture f;
    MockBuiltModel::fail_restore_next.store(true); // Load 内新建 mock 必失败
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen).code, Status::FAILURE);
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

// 权重共享接线：Load 第 5 参 share_weights_model_id 经 ModelInitOptions 透传
// 到 backend Init；ModelId() 暴露 GetModelID 取值供 decode 执行器绑定
TEST(ModelExecutorTest, ShareWeightsIdFlowsToInit)
{
    ExecFixture f;
    ASSERT_TRUE(f.exec
                    ->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen,
                           /*share_weights_model_id=*/7)
                    .ok());
    ASSERT_NE(g_last_mock, nullptr);
    EXPECT_EQ(g_last_mock->last_share_weights_id, 7u);
    EXPECT_EQ(f.exec->ModelId(), g_last_mock->model_id_);
}

// ── 异常路径 ──

TEST(ModelExecutorTest, MissingOmLoadFails_TC101)
{
    ExecFixture f;
    ArtifactSource a = MakeOmc(/*om_valid=*/false);
    EXPECT_EQ(f.exec->Load(a, kWeight, MakeConfig(), kLoadSeqLen).code, Status::FAILURE);
    EXPECT_FALSE(static_cast<bool>(*f.exec));
}

TEST(ModelExecutorTest, MissingOmPathLoadFails_TC101path)
{
    ExecFixture f;
    ArtifactSource p; // 双态均空 → 无图
    EXPECT_EQ(f.exec->Load(p, kWeight, MakeConfig(), kLoadSeqLen).code, Status::FAILURE);
}

TEST(ModelExecutorTest, MissingKvCacheMaxLenFails_TC106)
{
    ExecFixture f;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(/*kv_cache_max_len=*/0), kLoadSeqLen).code,
              Status::INVALID_ARGUMENT);
}

// A2 契约：未加载 Forward 返 FAILURE（bbit 只认 SUCCESS/FAILURE；非 INVALID_STATE）
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

// 重 Load 释放旧实例（ReleaseLocked → DeinitWeights + DeInit）
TEST(ModelExecutorTest, ReloadReleasesPreviousManager)
{
    ExecFixture f;
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen).ok());
    MockModelManager* first = g_last_mock;
    ASSERT_NE(first, nullptr);
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen).ok());
    EXPECT_GT(first->deinit_count.load(), 0);
    EXPECT_GT(first->deinitweights_count.load(), 0);
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
    (void)a.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen);
    MockModelManager* mock_a = g_last_mock;

    ExecFixture b(/*max_batch_size=*/4);
    (void)b.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen);
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

// 池共享（零拷贝载体）：同池经两执行器访问；T1：构造期/Load 期 0 tensor，
// desc 账本 Load 后完整（ADR-0002 建点双轨）
TEST(ModelExecutorTest, PoolSharedAcrossExecutors_T1Ledger_TC004structural)
{
    auto pool = std::make_shared<KvCachePool>(4, 16, 2, 4, 8);
    auto asm_ = std::make_shared<InputAssembler>(pool, 4);
    auto dfx = std::make_shared<DfxCollector>(DfxConfig{}, pool);
    ResetMocks();
    ModelExecutor prefill(pool, asm_, dfx);
    ModelExecutor decode(pool, asm_, dfx);
    EXPECT_EQ(prefill.Pool().get(), decode.Pool().get());
    EXPECT_EQ(pool->TensorCount(), 0u);       // T1：构造期不建 tensor
    EXPECT_FALSE(pool->DescLedgerComplete()); // 未 Load：账本未记
    (void)prefill.Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen);
    EXPECT_EQ(prefill.Pool()->TensorCount(), 0u); // 非 KV 型图：无 KV 工位可记
}

// ════════════════════════════════════════════════════════════════════════
// KV 工位（ADR-0002）：Load 期 EnsureKvDesc 记池 desc 账本 + 工位空占位
// （不建 tensor 不开显存）；Context 池按账本 EnsureKvTensor 实建；SetContext
// 原语 RebindKvPool 把空占位填实（输入按名解析层号、输出按账本序），om 原地
// 更新 KV 语义不变；DestroyContext 原语 DetachKvPool 回空占位。
// 池基准：ExecFixture 默认池 (batch=4, kv_len=16, 层 2, kv_heads=4, dim=8)
// → HF 期望 dims [batch,heads,kv_len,dim] = {4,4,16,8}；logits 首位 {4,16,128}
// （Q13：logits 恒 output[0]，真实图布局 logits 在 KV 形输出之前）。
// ════════════════════════════════════════════════════════════════════════

hiai::NDTensorDesc MkDesc(const char* name, hiai::DataType dt, std::vector<int32_t> dims)
{
    hiai::NDTensorDesc d;
    d.tensorName = name;
    d.dataType = dt;
    d.dims = std::move(dims);
    return d;
}

/// KV 型输入集：5 常规工位 + 4 KV 工位（层 0/1，K 前 V 后；HF 轴序 {4,4,16,8}）
std::vector<hiai::NDTensorDesc> KvModeInputs()
{
    const auto kv = hiai::DataType::FLOAT16;
    const std::vector<int32_t> kv_dims = {4, 4, 16, 8};
    return {
        MkDesc("input_embed", hiai::DataType::FLOAT16, {64}),
        MkDesc("embed_scales", hiai::DataType::FLOAT32, {64}),
        MkDesc("attention_mask", hiai::DataType::FLOAT32, {1, 1, 1, 16}),
        MkDesc("position_ids", hiai::DataType::INT32, {1, 1}),
        MkDesc("new_kv_cache_pos", hiai::DataType::INT32, {1}),
        MkDesc("past_key_in0", kv, kv_dims),
        MkDesc("past_value_in0", kv, kv_dims),
        MkDesc("past_key_in1", kv, kv_dims),
        MkDesc("past_value_in1", kv, kv_dims),
    };
}

/// KV 型输出集：logits 首位（Q13 真实布局）+ 4 个任意名 KV 形输出（层序 K 前 V 后）
std::vector<hiai::NDTensorDesc> KvModeOutputs()
{
    const auto kv = hiai::DataType::FLOAT16;
    const std::vector<int32_t> kv_dims = {4, 4, 16, 8};
    return {
        MkDesc("logits", hiai::DataType::FLOAT32, {4, 16, 128}),
        MkDesc("out_kv_a", kv, kv_dims),
        MkDesc("out_kv_b", kv, kv_dims),
        MkDesc("out_kv_c", kv, kv_dims),
        MkDesc("out_kv_d", kv, kv_dims),
    };
}

/// Context 池就绪：同参构造 + 按模板池 desc 账本逐层 EnsureKvTensor（真实建点）
std::shared_ptr<KvCachePool> MakeReadyContextPool(const ExecFixture& f)
{
    auto ctx = std::make_shared<KvCachePool>(4, 16, 2, 4, 8);
    for (int layer = 0; layer < 2; ++layer) {
        for (bool is_value : {false, true}) {
            auto desc = f.pool->KvDesc(layer, is_value);
            if (!desc)
                continue;
            (void)ctx->EnsureKvTensor(layer, is_value, *desc);
        }
    }
    return ctx;
}

// Load 记账不建显存（ADR-0002 建点双轨）：KV 工位空占位 + 池 desc 账本完整；
// 常规工位照建；logits 工位照建（Q13 首位）
TEST(ModelExecutorTest, LoadRecordsKvLedgerAndLeavesPlaceholders)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    EXPECT_EQ(f.exec->InputTensorCount(), 9u);
    EXPECT_EQ(f.exec->OutputTensorCount(), 5u);
    // KV 工位空占位（SetContext 前不可被 Run 消费——S2 门禁的前提形态）
    for (std::size_t i = 5; i <= 8; ++i)
        EXPECT_FALSE(f.exec->InputTensors()[i]) << i;
    for (std::size_t i = 1; i <= 4; ++i)
        EXPECT_FALSE(f.exec->OutputTensors()[i]) << i;
    // 常规/logits 工位已建
    EXPECT_TRUE(f.exec->InputTensors()[0]);
    EXPECT_TRUE(f.exec->OutputTensors()[0]);
    // 模板池：desc 账本完整（每层 K/V 全记）、0 tensor（T1）
    EXPECT_TRUE(f.pool->DescLedgerComplete());
    EXPECT_EQ(f.pool->TensorCount(), 0u);
    auto k0 = f.pool->KvDesc(0, false);
    ASSERT_TRUE(k0);
    EXPECT_EQ(k0->dims, (std::vector<int32_t>{4, 4, 16, 8})); // om 图为形状权威（R1）
    EXPECT_EQ(k0->dataType, hiai::DataType::FLOAT16);
    EXPECT_FALSE(f.pool->KTensor(0)); // 实建在 Context 池（EnsureKvTensor）
}

// RebindKvPool 填实/DetachKvPool 回空（SetContext/DestroyContext 工位刷新原语）
TEST(ModelExecutorTest, RebindFillsPlaceholdersAndDetachClears)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto ctx = MakeReadyContextPool(f);
    EXPECT_EQ(ctx->TensorCount(), 4u); // Context 池实建 2 层 × K/V
    ASSERT_TRUE(f.exec->RebindKvPool(ctx).ok());
    // 输入侧：past_key/value_in<层> → ctx 池同层 K/V tensor（同底层句柄）
    EXPECT_EQ(f.exec->InputTensors()[5].get(), ctx->KTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[6].get(), ctx->VTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[7].get(), ctx->KTensor(1).get());
    EXPECT_EQ(f.exec->InputTensors()[8].get(), ctx->VTensor(1).get());
    // 输出侧：按账本序 [k0,v0,k1,v1] 绑同一池 tensor（原地更新）
    EXPECT_EQ(f.exec->OutputTensors()[1].get(), ctx->KTensor(0).get());
    EXPECT_EQ(f.exec->OutputTensors()[2].get(), ctx->VTensor(0).get());
    EXPECT_EQ(f.exec->OutputTensors()[3].get(), ctx->KTensor(1).get());
    EXPECT_EQ(f.exec->OutputTensors()[4].get(), ctx->VTensor(1).get());
    // Logits 独立工位（Q13 首位）：≠ 任何池 tensor
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());
    EXPECT_NE(f.exec->Logits().get(), ctx->KTensor(0).get());
    // Detach：KV 工位回空占位、池引用置空（回"未指向"态）。真实 DestroyContext
    // （活动）为双动作：本原语 + 模板池侧 ClearTensors 由调用方同步
    ASSERT_TRUE(f.exec->DetachKvPool().ok());
    f.pool->ClearTensors(); // 模板池回"未指向"态（薄壳解绑）
    for (std::size_t i = 5; i <= 8; ++i)
        EXPECT_FALSE(f.exec->InputTensors()[i]) << i;
    for (std::size_t i = 1; i <= 4; ++i)
        EXPECT_FALSE(f.exec->OutputTensors()[i]) << i;
    EXPECT_EQ(f.pool->TensorCount(), 0u); // 转发面解除
    EXPECT_FALSE(f.exec->Pool());
    EXPECT_FALSE(static_cast<bool>(*f.exec)); // S6 后无活动态：Forward 门禁关门
}

// Rebind 前置防御：Context 池半建（未 Ensure）→ FAILURE（半建池不允许被激活）
TEST(ModelExecutorTest, RebindRejectsHalfBuiltPool)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto ctx = std::make_shared<KvCachePool>(4, 16, 2, 4, 8); // 0 tensor：半建
    EXPECT_EQ(f.exec->RebindKvPool(ctx).code, Status::FAILURE);
}

// Rebind 一致性防御：新池层数 ≠ Load 记账 → INVALID_ARGUMENT（宁拒不混绑）
TEST(ModelExecutorTest, RebindRejectsLayerCountMismatch)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto ctx = std::make_shared<KvCachePool>(4, 16, 3, 4, 8); // 3 层 ≠ 记账 2 层
    for (int layer = 0; layer < 2; ++layer) {
        for (bool is_value : {false, true}) {
            auto desc = f.pool->KvDesc(layer, is_value);
            if (desc)
                (void)ctx->EnsureKvTensor(layer, is_value, *desc);
        }
    }
    EXPECT_EQ(f.exec->RebindKvPool(ctx).code, Status::INVALID_ARGUMENT);
}

// KV 型 om 端到端（M-3 同构，ADR-0002 版）：Forward 送入 Run 的 KV 槽位就是
// Context 池 tensor（原地更新契约）。
// 流程按真实管线 SetContext 双动作：①模板池 BindFrom(ctx)——assembler/槽位
// 记账恒在模板池（Q10 乙薄壳：tensors_ 转发 ctx 实体）；②executor
// RebindKvPool(ctx)——工位填实。slot 分配/Advance 都落模板池（assembler
// 的 pool_ 构造期绑定模板池，不随 rebind 换）。
TEST(ModelExecutorTest, ForwardPassesPoolTensorsToRun)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto ctx = MakeReadyContextPool(f);
    ASSERT_TRUE(f.pool->BindFrom(*ctx).ok());    // ① 模板池 tensor 面转发
    ASSERT_TRUE(f.exec->RebindKvPool(ctx).ok()); // ② 工位填实
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok()); // 记账池 = 模板池
    ForwardOptions opts;
    opts.batch = 1;
    opts.chunk_len = 1;
    opts.slots = {slot};
    ASSERT_TRUE(f.exec->Forward(opts).ok());
    ASSERT_EQ(MockModelManager::last_run_inputs.size(), 9u);
    EXPECT_EQ(MockModelManager::last_run_inputs[5], ctx->KTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[6], ctx->VTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[7], ctx->KTensor(1).get());
    EXPECT_EQ(MockModelManager::last_run_inputs[8], ctx->VTensor(1).get());
    ASSERT_EQ(MockModelManager::last_run_outputs.size(), 5u);
    EXPECT_EQ(MockModelManager::last_run_outputs[0], f.exec->Logits().get());
    EXPECT_EQ(MockModelManager::last_run_outputs[1], ctx->KTensor(0).get());
    EXPECT_EQ(MockModelManager::last_run_outputs[4], ctx->VTensor(1).get());
}

// 输入侧按名绑定钉死（H-1）：KV 工位乱序排列，记账须按名字解析层号、非按位置
TEST(ModelExecutorTest, LoadBindsKvInputsByNameNotPosition)
{
    ExecFixture f;
    auto in = KvModeInputs();
    // 打乱 KV 段：[v1, k0, v0, k1]——与层序完全错开
    in[5] = MkDesc("past_value_in1", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    in[6] = MkDesc("past_key_in0", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    in[7] = MkDesc("past_value_in0", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    in[8] = MkDesc("past_key_in1", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto ctx = MakeReadyContextPool(f);
    ASSERT_TRUE(f.exec->RebindKvPool(ctx).ok());
    EXPECT_EQ(f.exec->InputTensors()[5].get(), ctx->VTensor(1).get()); // 名字说了算
    EXPECT_EQ(f.exec->InputTensors()[6].get(), ctx->KTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[7].get(), ctx->VTensor(0).get());
    EXPECT_EQ(f.exec->InputTensors()[8].get(), ctx->KTensor(1).get());
}

// meta_json ↔ om 校验：KV 输入 desc kv_len 维（HF 序 dims[2]）与 meta 失配
// → 致命终止（R3 分档：光标容量/mask/Advance 越界全线以 meta 为准）
TEST(ModelExecutorTest, LoadFailsOnKvDimsMismatch)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in0", hiai::DataType::FLOAT16, {4, 4, 8, 8}); // kv_len 8 ≠ 池 16
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// dtype 权威在图（R1）：全 KV 工位 fp32 → Load 成功、账本记 fp32、Context 池
// 按 fp32 desc 实建并可换绑（旧"om fp32 vs 池 fp16 失配拒绝"语义随池 dtype
// 参数一并消亡——池无 dtype 期望，图 desc 是唯一权威）
TEST(ModelExecutorTest, KvDtypeLedgerAuthority)
{
    ExecFixture f;
    auto in = KvModeInputs();
    auto out = KvModeOutputs();
    for (auto& d : in) {
        if (d.tensorName.rfind("past_", 0) == 0)
            d.dataType = hiai::DataType::FLOAT32;
    }
    for (auto& d : out) {
        if (d.tensorName.rfind("out_kv", 0) == 0)
            d.dataType = hiai::DataType::FLOAT32;
    }
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = out;
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    auto k0 = f.pool->KvDesc(0, false);
    ASSERT_TRUE(k0);
    EXPECT_EQ(k0->dataType, hiai::DataType::FLOAT32); // 账本随图记 fp32
    auto ctx = MakeReadyContextPool(f);
    ASSERT_TRUE(f.exec->RebindKvPool(ctx).ok()); // fp32 tensor 换绑成功
    EXPECT_EQ(f.exec->InputTensors()[5].get(), ctx->KTensor(0).get());
}

// om KV 工位数 ≠ 2×建池层数（只有层 0）→ 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvSlotCountMismatch)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in.resize(7); // 5 常规 + 仅 past_key_in0/past_value_in0
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// 稀疏池层表失配（Qwen3.5 §4.5-4）：config.layer_types 声明层 1 linear
// （无 KV 工位）而 om 声明 2 层 KV → 拒绝（原"池缺 tensor 拒绝"用例的现行
// 等价物：稀疏口径 = 层型表失配，非 tensor 在场性）
TEST(ModelExecutorTest, LoadFailsOnSparsePoolLayerTableMismatch)
{
    const std::vector<std::string> layer_types{"full_attention", "linear_attention"};
    auto pool = std::make_shared<KvCachePool>(4, 16, 2, 4, 8, layer_types);
    auto asm_ = std::make_shared<InputAssembler>(pool, 4);
    auto dfx = std::make_shared<DfxCollector>(DfxConfig{}, pool);
    ResetMocks();
    ModelExecutor exec(pool, asm_, dfx);
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ModelConfig cfg = MakeConfig(16, 2, 128);
    cfg.layer_types = {"full_attention", "linear_attention"}; // 层型校验配置源
    EXPECT_EQ(exec.Load(MakeOmc(), kWeight, cfg, kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// 输出侧 KV 形工位数与输入侧不等 → 创建失败
TEST(ModelExecutorTest, LoadFailsOnKvOutputCountMismatch)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    auto out = KvModeOutputs();
    out.resize(3); // 仅 logits + 2 个 KV 形
    MockBuiltModel::outputs_override = out;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// 池未挂载（pool=nullptr）但 om 声明 KV 工位 → 创建失败（配置类）
TEST(ModelExecutorTest, LoadFailsOnKvSlotsWithoutPool)
{
    ExecFixture f;
    ModelExecutor exec(/*pool=*/nullptr, f.asm_, f.dfx);
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(exec.Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// 层号越界（past_key_in5，池仅 2 层）→ 创建失败（资源类）
TEST(ModelExecutorTest, LoadFailsOnOutOfRangeLayerInKvName)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in5", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::FAILURE);
}

// 前缀命中但层数后缀非法（past_key_in_x）→ 创建失败（配置类）
TEST(ModelExecutorTest, LoadFailsOnMalformedKvName)
{
    ExecFixture f;
    auto in = KvModeInputs();
    in[5] = MkDesc("past_key_in_x", hiai::DataType::FLOAT16, {4, 4, 16, 8});
    MockBuiltModel::inputs_override = in;
    MockBuiltModel::outputs_override = KvModeOutputs();
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// 输出识别 dtype 参与判定（M-1a）：dims 相同但 dtype 不同 → 不识别为 KV → 数量失配
TEST(ModelExecutorTest, LoadFailsWhenKvOutputDtypeDiffers)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    auto out = KvModeOutputs();
    out[2] = MkDesc("out_kv_b", hiai::DataType::FLOAT32, {4, 4, 16, 8}); // 同 dims 异 dtype
    MockBuiltModel::outputs_override = out;
    EXPECT_EQ(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).code, Status::INVALID_ARGUMENT);
}

// kv_om 闸门（M-1b）：非 KV 型 om（无 past_* 输入）下 KV 形输出不做池绑定
TEST(ModelExecutorTest, KvShapedOutputNotBoundForNonKvOmc)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = {
        // 常规 5 工位（无 past_*）→ 非 KV 型 om
        MkDesc("input_embed", hiai::DataType::FLOAT16, {64}),
        MkDesc("embed_scales", hiai::DataType::FLOAT32, {64}),
        MkDesc("attention_mask", hiai::DataType::FLOAT32, {1, 1, 1, 16}),
        MkDesc("position_ids", hiai::DataType::INT32, {1, 1}),
        MkDesc("new_kv_cache_pos", hiai::DataType::INT32, {1}),
    };
    MockBuiltModel::outputs_override = {
        MkDesc("logits", hiai::DataType::FLOAT32, {4, 16, 128}),
        MkDesc("mystery", hiai::DataType::FLOAT16, {4, 4, 16, 8}), // KV 形但 om 非 KV 型
    };
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    // 不绑池：独立申请的 tensor（非空；模板池本就 0 tensor）
    EXPECT_EQ(f.exec->OutputTensorCount(), 2u);
    EXPECT_TRUE(f.exec->OutputTensors()[1]);
    // Logits() 按 Q13 恒首位定位
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());
}

// logits 定位契约（Q13 裁定收敛：原"shape 扫描定位/两路回退"组并入）：
// 恒钉 output[0]——vocab_align 供给与否、3D 末维匹配与否不再影响定位
TEST(ModelExecutorTest, LogitsAlwaysPinsFirstOutput_Q13)
{
    ExecFixture f;
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(), kLoadSeqLen).ok()); // 无 vocab_align
    ASSERT_TRUE(f.exec->Logits());
    EXPECT_EQ(f.exec->Logits().get(), f.exec->OutputTensors()[0].get());

    ExecFixture g;
    ASSERT_TRUE(g.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128),
                             kLoadSeqLen)
                    .ok()); // vocab_align=128，默认输出 {1} 无 3D 匹配
    ASSERT_TRUE(g.exec->Logits());
    EXPECT_EQ(g.exec->Logits().get(), g.exec->OutputTensors()[0].get());
}

// logits 工位生命周期（M-2 并组）：move 后新对象保留首位定位、源对象失效；
// Unload → 重 Load 后照常首位
TEST(ModelExecutorTest, LogitsSurvivesMoveAndReload)
{
    ExecFixture f;
    MockBuiltModel::inputs_override = KvModeInputs();
    MockBuiltModel::outputs_override = KvModeOutputs();
    ASSERT_TRUE(f.exec->Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    ModelExecutor moved(std::move(*f.exec));
    ASSERT_TRUE(moved.Logits());
    EXPECT_EQ(moved.Logits().get(), moved.OutputTensors()[0].get());
    EXPECT_FALSE(f.exec->Logits()); // 源对象输出已搬空

    ASSERT_TRUE(moved.Unload().ok());
    MockBuiltModel::inputs_override.clear(); // 回默认（非 KV 型）descs
    MockBuiltModel::outputs_override.clear();
    ASSERT_TRUE(moved.Load(MakeOmc(), kWeight, MakeConfig(16, 2, 128), kLoadSeqLen).ok());
    ASSERT_TRUE(moved.Logits());
    EXPECT_EQ(moved.Logits().get(), moved.OutputTensors()[0].get());
}

} // namespace
} // namespace cann_local_lm::engine
