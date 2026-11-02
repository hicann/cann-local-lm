// Scheduler 多块 prefill 出口语义 UT（is_prefill_last 旗，
// tmp_docs/fix_prefill_sample_cache_update_202610222243.md §3 S6）。
// 覆盖契约（UT-TC-SCHED-301..304）：
//   UT-TC-SCHED-301（多块主链）：两块 prefill + decode 全弧——中间块出口
//     is_prefill=true / is_prefill_last=false（旗语义核心回归锚点：中间块
//     logits 是对本块末位 prompt 真值的预测，不采样不记账），池逐块 Advance
//     （TotalWritten 6→12），kv_slot_actual 回写 = 0，请求保持 Queued（中间
//     出口无终态/阶段跳变）；末块出口双旗同真、请求转 Decoding、出队；
//     decode 步旗归 false、光标 +1；cancel 驱逐后无活任务 INVALID_STATE。
//   UT-TC-SCHED-302（单块）：单块请求一步双旗同真、出队（is_prefill_last
//     语义 = 末块或单块完成点）。
//   UT-TC-SCHED-303（空块守卫就绪路径，DTS202610220945 Fix B）：既有
//     ut_scheduler 12 例因 host 无图恒走 not-ready 分支，本例首次在就绪
//     分支触达守卫（scheduler.cpp:144）——FAILURE + 请求 Failed 出队 +
//     out.slots 不携带 + 双旗恒 false + kv_slot_actual 未回写（-1）。
//   UT-TC-SCHED-304（中间块 Forward 失败）：失败出口双旗恒 false（本修复
//     防回归：任何实现不得在 Forward 前置旗）+ req 经 out->slots 携带上抛
//     （R10 回退载体）+ 请求 Failed 出队。
// 独立目标缘由（同 ut_share_weights_executor）：本文件经链接期符号覆盖
// hiai::CreateModelManager / CreateBuiltModel / CreateNDTensorBuffer
// （cann_lm_engine 为 STATIC 库，ut 对象定义优先）；不并入其他 binary——工厂
// 覆盖为二进制级符号替换，劫持面不可控。不并入既有 ut_scheduler：其 12 例
// 的契约恰是"host 真工厂返 nullptr → executor 恒 not-ready"，mock 覆盖会
// 反转其被测前提。
// mock 图工位族 = assembler 消费面最简形态（input_embed + position_ids +
// new_kv_cache_pos + attention_mask float 单 mask）；**无 KV 族工位**
// （past_key_in*/past_value_in*/past_key_values_*/conv_state_in*/rstate_in*）
// → Load 期 kv_om=false，不触发 KV 池绑定校验（真机形态 KV tensor 由
// CreateContext 按池建轨，host UT 不触）。dims 经 CorrectTensorDescs 按名
// 修正（prefill seq=8：{1,8}/{8}/{1,1,8,16}；decode seq=1：{1,1}/{1}/{1,1,1,16}），
// HostTensorBuffer 容量 = 4×dims 积自洽。
// 关联：engine/pipeline/scheduler.cpp（末块旗置点 :296-299 / 守卫 :144-153 /
// kv_slot_actual 回写 :186）、pipeline_executor.cpp RunLoop（旗消费点）、
// docs/prompt_cache_spec.md REQ-2（末 chunk 出口一次记账的调度器侧前提）。
#include "pipeline/scheduler.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/model_executor/model_executor.h"
#include "core/component/input_assembler/input_assembler.h"
#include "pipeline/preprocessor.h"
#include "core/component/text_embedding/text_embedding.h"
#include "core/component/tokenizer/tokenizer.h"
#include "model_manager/model_manager.h"
#include "model/built_model.h"
#include "tensor/nd_tensor_buffer.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ── mock 基建（具名 namespace：外部链接类型用于工厂覆盖签名）──────────────────
namespace mc_sched_mock {

// host 可用 INDTensorBuffer 最小实现（host 无 NDK so，真工厂返 nullptr）
class HostTensorBuffer final : public hiai::INDTensorBuffer {
public:
    explicit HostTensorBuffer(const hiai::NDTensorDesc& desc) : desc_(desc)
    {
        // 字节数 ≥ 4（GetData 非空约定；非正维按 1 估算——本 UT descs 无 0 维）
        std::size_t bytes = 4;
        for (int32_t d : desc_.dims) {
            bytes *= static_cast<std::size_t>(d > 0 ? d : 1);
        }
        data_.resize(bytes);
    }
    void* GetData() override { return data_.data(); }
    std::size_t GetSize() const override { return data_.size(); }
    const hiai::NDTensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

private:
    hiai::NDTensorDesc desc_;
    std::vector<uint8_t> data_;
    uint8_t cache_status_ = 0;
};

// Mock IBuiltModel：buffer 态 Restore 恒过。输入工位族 = assembler 消费面
// （float 单 mask 最简形：attention_mask 在场即 kFloatSingle 路径）；
// position/new_kv_cache_pos 交 CorrectTensorDescs 按名修正容量。
class MockBuiltModel final : public hiai::IBuiltModel {
public:
    hiai::Status RestoreFromBuffer(const std::shared_ptr<hiai::IBuffer>& buffer) override
    {
        return (buffer && buffer->GetData()) ? hiai::SUCCESS : hiai::FAILURE;
    }
    hiai::Status RestoreFromFile(const char* file) override
    {
        return (file != nullptr && *file != '\0') ? hiai::SUCCESS : hiai::FAILURE;
    }
    void SetName(const std::string&) override {}
    std::vector<hiai::NDTensorDesc> GetInputTensorDescs() const override
    {
        hiai::NDTensorDesc embed;
        embed.tensorName = "input_embed";
        embed.dataType = hiai::DataType::FLOAT16;
        embed.dims = {1, 1, 8};
        hiai::NDTensorDesc pos;
        pos.tensorName = "position_ids";
        pos.dataType = hiai::DataType::INT32;
        pos.dims = {1, 1};
        hiai::NDTensorDesc npos;
        npos.tensorName = "new_kv_cache_pos";
        npos.dataType = hiai::DataType::INT32;
        npos.dims = {1};
        hiai::NDTensorDesc mask;
        mask.tensorName = "attention_mask";
        mask.dataType = hiai::DataType::FLOAT32;
        // CorrectTensorDescs 修形为 {batch,1,seq_len,kv_max} 全宽
        // （float mask 静态全宽工位语义）；此处给合法占位，Load 期
        // 按测试配置批次修正。
        mask.dims = {1, 1, 1, 16};
        return {embed, pos, npos, mask};
    }
    std::vector<hiai::NDTensorDesc> GetOutputTensorDescs() const override
    {
        hiai::NDTensorDesc d;
        d.dataType = hiai::DataType::FLOAT32;
        d.dims = {1, 1};
        return {d};
    }
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

private:
    hiai::CustomModelData custom_data_;
};

// Mock IModelManager：Run 返回值经类级静态 run_status 可配（UT-TC-SCHED-304
// 中间块 Forward 失败注入通道）；0（hiai::SUCCESS）外任意值按后端失败映射。
class MockModelManager final : public hiai::IModelManager {
public:
    static int run_status; // Run 返回值（默认 hiai::SUCCESS=0）

    hiai::Status Init(const hiai::ModelInitOptions&, const std::shared_ptr<hiai::IBuiltModel>& builtModel,
                      const std::shared_ptr<hiai::IModelManagerListener>&) override
    {
        built_ = builtModel;
        return hiai::SUCCESS;
    }
    hiai::Status InitWeights(const std::string& weightDir) override
    {
        return weightDir.empty() ? hiai::FAILURE : hiai::SUCCESS;
    }
    hiai::Status DeinitWeights() override { return hiai::SUCCESS; }
    hiai::Status GetWeightBuffer(const std::string&, std::shared_ptr<hiai::IBuffer>&) override
    {
        return hiai::UNSUPPORTED;
    }
    hiai::Status FlushWeight(const std::string&, size_t, size_t) override { return hiai::UNSUPPORTED; }
    hiai::Status UpdateWeight(const std::string&, size_t, size_t, const void*) override { return hiai::UNSUPPORTED; }
    uint32_t GetModelID() override { return 1; }
    hiai::Status Run(const std::vector<std::shared_ptr<hiai::INDTensorBuffer>>&,
                     std::vector<std::shared_ptr<hiai::INDTensorBuffer>>&) override
    {
        return static_cast<hiai::Status>(run_status);
    }
    hiai::Status ReleaseMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    hiai::Status RecoverMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    void DeInit() override { built_.reset(); }

private:
    std::shared_ptr<hiai::IBuiltModel> built_;
};

int MockModelManager::run_status = 0;

void ResetMocks() { MockModelManager::run_status = 0; }

// 权重路径（图与权重分离；空路径 = Load 跳过 InitWeights）
constexpr const char* kWeight = "/mock/weights";

// 合法 omc buffer 源（buffer 态优先，D-15）
cann_lm::engine::ArtifactSource MakeOmc()
{
    static char omc_buf[8];
    cann_lm::engine::ArtifactSource s;
    s.data = cann_lm::engine::Buffer{omc_buf, sizeof(omc_buf)};
    return s;
}

// ── U1 专用：gemma 双 bool mask 图变体（sliding_kv_seq_dim_rebind §5 U1）──
// 既有 MockBuiltModel 工位族 = float 单 mask 最简形态（attention_mask 在场），
// CorrectTensorDescs 的 mask_sliding 分支在其上不可达。本变体改为 gemma 工位族
// （mask_full + mask_sliding 双 bool 工位，无 attention_mask），om 导出占位 dims
// 列宽=kv_max 旧形——Load 真链经 CorrectTensorDescs 修形后 mask_sliding dims[3]
// 按契约切换（U1 分主链/副链两态断言）：新契 → C = W+P（SlidingLayerWidth）；
// 旧契（S6 主链，sliding_kv_mask_width_opt 时代口径已废 D8）→ kv_max，
// mask_full 恒 kv_max。
// 观测点：Load 后 ModelExecutor::InputTensors()[i]->GetTensorDesc().dims
// （HostTensorBuffer 按修形后 desc 建形）。
class GemmaMaskBuiltModel final : public hiai::IBuiltModel {
public:
    hiai::Status RestoreFromBuffer(const std::shared_ptr<hiai::IBuffer>& buffer) override
    {
        return (buffer && buffer->GetData()) ? hiai::SUCCESS : hiai::FAILURE;
    }
    hiai::Status RestoreFromFile(const char* file) override
    {
        return (file != nullptr && *file != '\0') ? hiai::SUCCESS : hiai::FAILURE;
    }
    void SetName(const std::string&) override {}
    std::vector<hiai::NDTensorDesc> GetInputTensorDescs() const override
    {
        hiai::NDTensorDesc full;
        full.tensorName = "mask_full";
        full.dataType = hiai::DataType::BOOL;
        full.dims = {1, 1, 8, 16}; // om 占位旧形（kv_max=16 全宽）
        hiai::NDTensorDesc slid;
        slid.tensorName = "mask_sliding";
        slid.dataType = hiai::DataType::BOOL;
        slid.dims = {1, 1, 8, 16}; // om 占位旧形——修形后列宽按契约定
        hiai::NDTensorDesc pos;
        pos.tensorName = "position_ids";
        pos.dataType = hiai::DataType::INT32;
        pos.dims = {1, 1};
        hiai::NDTensorDesc npos;
        npos.tensorName = "new_kv_cache_pos";
        npos.dataType = hiai::DataType::INT32;
        npos.dims = {1};
        // 1027 新图滑窗层专用 scatter 写位（与 new_kv_cache_pos 成对；1027
        // 实图 om 占位 [64,1,1,1]，此处同构缩样 [1,1,1,1]——修形后 [seq_len]）
        hiai::NDTensorDesc spos;
        spos.tensorName = "sliding_kv_pos";
        spos.dataType = hiai::DataType::INT32;
        spos.dims = {1, 1, 1, 1};
        return {full, slid, pos, npos, spos};
    }
    std::vector<hiai::NDTensorDesc> GetOutputTensorDescs() const override
    {
        hiai::NDTensorDesc d;
        d.dataType = hiai::DataType::FLOAT32;
        d.dims = {1, 1};
        return {d};
    }
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

private:
    hiai::CustomModelData custom_data_;
};

// U1 mock 择路开关：CreateBuiltModel 经类级静态 flag 返回 float 形 / gemma 形
// （default false = 既有 5 用例不受扰；U1 用例置 true，RAII guard 保证用例退出
// ——含 ASSERT 提前 return——恒复位，无需 fixture）。
bool g_use_gemma_mask_model = false;

// 容量域合法配置（TC_106 前置）。batch_num=1：decode 执行器 Load（seq_len=1）
// 时 batch_size 取 cc.batch_num，须为正（0 会把 position_ids 修正成 0 维）。
// hidden_size/vocab_align_size 留 0：input_embed/输出侧特征修正跳过。
cann_lm::engine::ModelConfig MakeConfig()
{
    cann_lm::engine::ModelConfig c;
    c.kv_cache_max_len = 16;
    c.num_hidden_layers = 2;
    c.batch_num = 1;
    return c;
}

// chunk token 载荷（值不参与语义：mock Run 不读输入，token_ids 由 pipeline
// 侧采样产生，Scheduler 只消费块长）
std::vector<cann_lm::engine::TokenId> MakeIds(std::size_t n) { return std::vector<cann_lm::engine::TokenId>(n, 7); }

} // namespace mc_sched_mock

// ── 工厂覆盖（hiai 命名空间；链接期符号优先于库内实现）──────────────────────
namespace hiai {
std::shared_ptr<IModelManager> CreateModelManager() { return std::make_shared<mc_sched_mock::MockModelManager>(); }
std::shared_ptr<IBuiltModel> CreateBuiltModel()
{
    // U1 gemma 双 mask 图变体择路（flag 仅 U1 fixture 期间为 true）
    if (mc_sched_mock::g_use_gemma_mask_model) {
        return std::make_shared<mc_sched_mock::GemmaMaskBuiltModel>();
    }
    return std::make_shared<mc_sched_mock::MockBuiltModel>();
}
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc)
{
    return std::make_shared<mc_sched_mock::HostTensorBuffer>(tensorDesc);
}
} // namespace hiai

namespace cann_lm::engine {
namespace {

// 就绪管线 fixture：真实 KvCachePool(1,16,2,4,8) + InputAssembler + 双
// ModelExecutor Load（mock buffer 态图）→ host 首次触达 Scheduler::Step
// 就绪分支。mock 图无 KV 族工位（kv_om=false），但工位族覆盖 assembler 的
// position/mask 消费面 → Forward 全链（AssemblePrefill/AssembleDecode/
// mm Run mock）host 走通。text_embedding 传 nullptr：FillInputs 跳过
// （input_embed 短块补齐逻辑不参与本 UT 面被测语义）。
class SchedulerMultiChunkTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mc_sched_mock::ResetMocks();
        pool_ = std::make_shared<KvCachePool>(/*max_batch=*/1, /*kv_max=*/16,
                                              /*layers=*/2, /*heads=*/4, /*dims=*/8);
        asm_ = std::make_shared<InputAssembler>(pool_, /*max_batch=*/1);
        prefill_exec_ = std::make_unique<ModelExecutor>(pool_, asm_, nullptr);
        decode_exec_ = std::make_unique<ModelExecutor>(pool_, asm_, nullptr);
        tok_ = std::make_unique<BPETokenizer>(TokenizerConfig{}, nullptr);
        te_ = std::make_unique<TextEmbedding>();
        pp_ = std::make_unique<Preprocessor>(tok_.get(), te_.get(), 1024);
        ASSERT_TRUE(
            prefill_exec_
                ->Load(mc_sched_mock::MakeOmc(), mc_sched_mock::kWeight, mc_sched_mock::MakeConfig(), /*seq_len=*/8)
                .ok());
        ASSERT_TRUE(
            decode_exec_
                ->Load(mc_sched_mock::MakeOmc(), mc_sched_mock::kWeight, mc_sched_mock::MakeConfig(), /*seq_len=*/1)
                .ok());
        sched_ = std::make_unique<Scheduler>(prefill_exec_.get(), decode_exec_.get(), pp_.get(),
                                             /*text_embedding=*/nullptr,
                                             /*max_batch_size=*/1);
    }

    std::shared_ptr<KvCachePool> pool_;
    std::shared_ptr<InputAssembler> asm_;
    std::unique_ptr<ModelExecutor> prefill_exec_;
    std::unique_ptr<ModelExecutor> decode_exec_;
    std::unique_ptr<BPETokenizer> tok_;
    std::unique_ptr<TextEmbedding> te_;
    std::unique_ptr<Preprocessor> pp_;
    std::unique_ptr<Scheduler> sched_;
};

// UT-TC-SCHED-301: 多块 prefill 主链——中间块双旗分离 + 池逐块推进 + 末块
// 双旗同真 + decode 旗归 false + cancel 收尾。
TEST_F(SchedulerMultiChunkTest, PrefillFlagsAndBookkeeping_TC301)
{
    auto req = std::make_shared<RequestImpl>();
    req->prompt_ids = mc_sched_mock::MakeIds(12);
    std::vector<std::vector<TokenId>> chunks = {mc_sched_mock::MakeIds(6), mc_sched_mock::MakeIds(6)};
    RequestId id;
    ASSERT_TRUE(sched_->Submit(std::move(chunks), req, &id).ok());
    EXPECT_EQ(sched_->PendingPrefillCount(), 1);

    // ── Step 1：中间块出口 ──
    StepOutput out1;
    ASSERT_TRUE(sched_->Step(&out1).ok());
    EXPECT_TRUE(out1.is_prefill);
    EXPECT_FALSE(out1.is_prefill_last);                   // 中间块：末块旗恒 false（本修复语义核心）
    EXPECT_EQ(pool_->TotalWritten(0), 6u);                // 池按块 Advance（assembler 尾部）
    EXPECT_EQ(req->kv_slot_actual, 0);                    // 实际工位回写（Alloc 分配路）
    EXPECT_EQ(req->status.load(), RequestStatus::Queued); // 中间出口无阶段跳变
    EXPECT_EQ(sched_->PendingPrefillCount(), 1);          // 请求仍在队列
    ASSERT_EQ(out1.slots.size(), 1u);
    EXPECT_EQ(out1.slots.front(), req);

    // ── Step 2：末块出口（采样/记账唯一点）──
    StepOutput out2;
    ASSERT_TRUE(sched_->Step(&out2).ok());
    EXPECT_TRUE(out2.is_prefill);
    EXPECT_TRUE(out2.is_prefill_last); // 末块：双旗同真
    EXPECT_EQ(pool_->TotalWritten(0), 12u);
    EXPECT_EQ(req->status.load(), RequestStatus::Decoding);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0); // 出队转 pending_decode

    // ── Step 3：decode 步（ReclaimAndFill_ 补位后同 Step 内跑批）──
    StepOutput out3;
    ASSERT_TRUE(sched_->Step(&out3).ok());
    EXPECT_FALSE(out3.is_prefill);
    EXPECT_FALSE(out3.is_prefill_last); // decode 出口：两旗归 false
    EXPECT_EQ(pool_->TotalWritten(0), 13u);
    EXPECT_EQ(sched_->ActiveDecodeCount(), 1);

    // ── Step 4：cancel 驱逐 + 无活任务 ──
    req->canceled.store(true, std::memory_order_release);
    StepOutput out4;
    auto s4 = sched_->Step(&out4);
    EXPECT_FALSE(s4.ok());
    EXPECT_EQ(s4.code, Status::INVALID_STATE); // cancel 清理后无活任务
    EXPECT_EQ(req->status.load(), RequestStatus::Canceled);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0);
    EXPECT_EQ(sched_->ActiveDecodeCount(), 0);
}

// UT-TC-SCHED-302: 单块——一步双旗同真、出队（is_prefill_last = 末块或单块）。
TEST_F(SchedulerMultiChunkTest, SingleChunkSetsBothFlags_TC302)
{
    auto req = std::make_shared<RequestImpl>();
    req->prompt_ids = mc_sched_mock::MakeIds(6);
    std::vector<std::vector<TokenId>> chunks = {mc_sched_mock::MakeIds(6)};
    RequestId id;
    ASSERT_TRUE(sched_->Submit(std::move(chunks), req, &id).ok());

    StepOutput out;
    ASSERT_TRUE(sched_->Step(&out).ok());
    EXPECT_TRUE(out.is_prefill);
    EXPECT_TRUE(out.is_prefill_last);
    EXPECT_EQ(pool_->TotalWritten(0), 6u);
    EXPECT_EQ(req->status.load(), RequestStatus::Decoding);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0);
}

// UT-TC-SCHED-303: 空块守卫就绪路径（Fix B 防御纵深）——空块链（外层空）
// 入队 → FAILURE 早期终态化，不触池、不回写工位、不携带 slots、双旗恒 false。
TEST_F(SchedulerMultiChunkTest, EmptyChunkGuardReadyPath_TC303)
{
    auto req = std::make_shared<RequestImpl>();
    std::vector<std::vector<TokenId>> chunks; // 空块链（外层空：守卫判 chunks.empty()）
    RequestId id;
    ASSERT_TRUE(sched_->Submit(std::move(chunks), req, &id).ok());

    StepOutput out;
    auto s = sched_->Step(&out);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, Status::FAILURE);
    EXPECT_EQ(req->status.load(), RequestStatus::Failed);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0);
    EXPECT_TRUE(out.slots.empty()); // 守卫不填 slots（无 Forward 产物）
    EXPECT_FALSE(out.is_prefill);   // 失败出口双旗恒 false
    EXPECT_FALSE(out.is_prefill_last);
    EXPECT_EQ(req->kv_slot_actual, -1);    // 守卫先于工位分配/回写（:144 < :186）
    EXPECT_EQ(pool_->TotalWritten(0), 0u); // 池未推进
}

// UT-TC-SCHED-304: 中间块 Forward 失败（Run 注入）——失败出口双旗恒 false +
// req 经 out->slots 携带（R10 回退载体）+ 请求 Failed 出队。
TEST_F(SchedulerMultiChunkTest, MidChunkForwardFailureFlags_TC304)
{
    auto req = std::make_shared<RequestImpl>();
    req->prompt_ids = mc_sched_mock::MakeIds(12);
    std::vector<std::vector<TokenId>> chunks = {mc_sched_mock::MakeIds(6), mc_sched_mock::MakeIds(6)};
    RequestId id;
    ASSERT_TRUE(sched_->Submit(std::move(chunks), req, &id).ok());

    StepOutput out1;
    ASSERT_TRUE(sched_->Step(&out1).ok()); // 首块成功（TotalWritten=6）

    mc_sched_mock::MockModelManager::run_status = static_cast<int>(hiai::FAILURE);
    StepOutput out2;
    auto s2 = sched_->Step(&out2);
    EXPECT_FALSE(s2.ok());
    EXPECT_FALSE(out2.is_prefill); // 失败出口双旗恒 false（防回归）
    EXPECT_FALSE(out2.is_prefill_last);
    ASSERT_EQ(out2.slots.size(), 1u); // 失败请求照填 slots（pipeline 统一回退）
    EXPECT_EQ(out2.slots.front(), req);
    EXPECT_EQ(req->status.load(), RequestStatus::Failed);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0);
}

// UT-TC-SCHED-305（U5，方案 §5 + REQ-9）：单轮绕回记账全链——kv_max=16、
// ids=36、chunk=8（36 = 2×16+4，绕回点 16/32 恰落第 3/第 5 块块首）＋ 尾块 4：
// 五块链 [8,8,8,8,4]（累计边界 8/16/24/32/36）。池逐块 Advance 全 SUCCESS
// （16/32 绕回点不减一行）、
// 中间块出口双旗分离、末块双旗同真 + TotalWritten=36（多圈）+
// NextPosition=MapSlot(36)=4（anchor=0 纯模绕回）、请求转 Decoding；
// decode 一步绕回后推进 37/物理位 5。台账-池对账面（REQ-2 终态一致不变量）
// 本 fixture 无台账接线不可锁——现证物：真机 V-F2 日志（绕回态台账 R6 保守
// miss 行为证）；host 真链记账 UT 属验收债务（code_review P2）。
// 生产码面：assembler 红线塌零后 abs_start+chunk_len > kv_max 块走环形
// 装配（WriteFloatMask 全宽 + npos=MapSlot 写位恒 < kv_max）。
TEST_F(SchedulerMultiChunkTest, SingleTurnWraparoundBookkeeping_TC305)
{
    auto req = std::make_shared<RequestImpl>();
    req->prompt_ids = mc_sched_mock::MakeIds(36);
    std::vector<std::vector<TokenId>> chunks;
    for (int i = 0; i < 4; ++i)
        chunks.push_back(mc_sched_mock::MakeIds(8));
    chunks.push_back(mc_sched_mock::MakeIds(4));
    RequestId id;
    ASSERT_TRUE(sched_->Submit(std::move(chunks), req, &id).ok());
    EXPECT_EQ(sched_->PendingPrefillCount(), 1);

    // 前四块：中间块出口双旗分离 + 池逐块 Advance（8/16/24/32；16、32
    // 越 kv_max 绕回点 Advance 照样 SUCCESS）
    for (int step = 0; step < 4; ++step) {
        StepOutput out;
        ASSERT_TRUE(sched_->Step(&out).ok()) << "step=" << step;
        EXPECT_TRUE(out.is_prefill) << "step=" << step;
        EXPECT_FALSE(out.is_prefill_last) << "step=" << step;
        EXPECT_EQ(pool_->TotalWritten(0), static_cast<std::size_t>(8 * (step + 1)))
            << "绕回 Advance SUCCESS（(step+1)*8）";
    }

    // 末块（起 32，块内全程绕回）：双旗同真、TotalWritten=36、物理位 MapSlot(36)=4
    StepOutput out5;
    ASSERT_TRUE(sched_->Step(&out5).ok());
    EXPECT_TRUE(out5.is_prefill);
    EXPECT_TRUE(out5.is_prefill_last);
    EXPECT_EQ(pool_->TotalWritten(0), 36u) << "多圈绕回记账 2×kv_max+4";
    EXPECT_EQ(pool_->NextPosition(0), 4u) << "MapSlot(36)=36%16=4（anchor=0）";
    EXPECT_EQ(req->status.load(), RequestStatus::Decoding);
    EXPECT_EQ(sched_->PendingPrefillCount(), 0);

    // decode 一步（绕回后）：旗归 false、37/物理位 MapSlot(37)=5
    StepOutput out6;
    ASSERT_TRUE(sched_->Step(&out6).ok());
    EXPECT_FALSE(out6.is_prefill);
    EXPECT_FALSE(out6.is_prefill_last);
    EXPECT_EQ(pool_->TotalWritten(0), 37u);
    EXPECT_EQ(pool_->NextPosition(0), 5u);
    EXPECT_EQ(sched_->ActiveDecodeCount(), 1);

    // cancel 收尾（无活任务终态）
    req->canceled.store(true, std::memory_order_release);
    StepOutput out7;
    auto s7 = sched_->Step(&out7);
    EXPECT_FALSE(s7.ok());
    EXPECT_EQ(sched_->ActiveDecodeCount(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// U1/U1-b（sliding_kv_seq_dim_rebind §5）：CorrectTensorDescs mask 列宽修形
// Load 真链集成——.cpp 匿名 namespace 不可直测（ut_engine_executor 头注释缝隙
// 说明），经本目标 mock 真链触达：mock gemma 图（mask_full+mask_sliding om 占位
// 旧形 kv_max=16 全宽）→ Load → 断言 InputTensors() 工位 dims：
//   - mask_full 恒 {batch,1,seq,kv_max}（全因果层 kv_max 锚定环，零变化）；
//   - mask_sliding dims[3] 按契约定（S5/D5/D9 分派只看 flag）：
//     旧契（sw_hd=0）→ 恒 kv_max（sw 不影响列宽——与 full 同形；
//     sw=0 window=0 走全因果，行为在 assembler 写入层分派）；
//     新契（sw_hd>0）→ C = SlidingLayerWidth(sw, P) = W+P（SlidingLayerWidth
//     单源；本测 P=8 → sw=4 列宽 12、sw=16 列宽 24）；
//   - 正交锁：position_ids {batch,seq}、new_kv_cache_pos {seq} 修形不串扰。
// ═══════════════════════════════════════════════════════════════════════════
// 独立 TEST（非 fixture）：U1 只消费 mock Load 真链，不用 SchedulerMultiChunkTest
// 的 SetUp 双执行器态——且 gtest 禁同一 suite 混用 TEST/TEST_F。
TEST(GemmaMaskCorrectDescsTest, MaskSlidingWidthDims_U1)
{
    mc_sched_mock::g_use_gemma_mask_model = true;
    auto reset = [](void*) { mc_sched_mock::g_use_gemma_mask_model = false; };
    std::unique_ptr<void, decltype(reset)> flag_guard(nullptr, reset);

    // 旧契主链（sw_hd=0）：滑窗层与 full 层同形 kv_max，sw 无关
    for (const int sw : {0, 4, 16, 20}) {
        // 逐 sw 独立 Load：prefill 执行器（seq_len=8 修形路径）
        auto pool = std::make_shared<KvCachePool>(/*max_batch=*/1, /*kv_max=*/16,
                                                  /*layers=*/2, /*heads=*/4, /*dims=*/8);
        auto asm_ = std::make_shared<InputAssembler>(pool, /*max_batch=*/1, sw);
        ModelExecutor exec(pool, asm_, nullptr);
        cann_lm::engine::ModelConfig cfg = mc_sched_mock::MakeConfig();
        cfg.sliding_window = sw;
        ASSERT_TRUE(exec.Load(mc_sched_mock::MakeOmc(), mc_sched_mock::kWeight, cfg, /*seq_len=*/8).ok())
            << "sw=" << sw;

        // InputTensors() 序 = GetInputTensorDescs() 序（full, sliding, pos, npos, spos）
        const auto& ins = exec.InputTensors();
        ASSERT_EQ(ins.size(), 5u);
        const auto& full = ins[0]->GetTensorDesc();
        const auto& slid = ins[1]->GetTensorDesc();
        const auto& pos = ins[2]->GetTensorDesc();
        const auto& npos = ins[3]->GetTensorDesc();
        const auto& spos = ins[4]->GetTensorDesc();

        // mask_full 恒 kv_max 全宽（sw 取值不影响本工位——零变化）
        EXPECT_EQ(full.dims, (std::vector<int32_t>{1, 1, 8, 16})) << "sw=" << sw;
        // 旧契（D5/D9）：列宽恒 kv_max——sliding 层与 full 层同槽域；
        // sw 不再影响修形列宽（b688221 w_eff 口径已废 D8）
        EXPECT_EQ(slid.dims, (std::vector<int32_t>{1, 1, 8, 16})) << "sw=" << sw;
        // 正交锁：pos 族修形不受 sw 串扰（batch=1 → npos 1D [seq]）
        EXPECT_EQ(pos.dims, (std::vector<int32_t>{1, 8}));
        EXPECT_EQ(npos.dims, (std::vector<int32_t>{8}));
        // sliding_kv_pos 修形与 new_kv_cache_pos 同构（batch=1 → 1D [seq_len]，
        // 与 sw/契约 flag 无关——SeqLen 工位容量只随图 seq 维走）
        EXPECT_EQ(spos.dims, (std::vector<int32_t>{8})) << "sw=" << sw;
    }

    // 新契副链（sw_hd>0）：列宽 = C = W+P（P=8 档）
    for (const int sw : {4, 16}) {
        const int P = 8;
        auto pool = std::make_shared<KvCachePool>(/*max_batch=*/1, /*kv_max=*/16,
                                                  /*layers=*/2, /*heads=*/4, /*dims=*/8);
        auto asm_ = std::make_shared<InputAssembler>(pool, /*max_batch=*/1, sw, P, /*sw_hd=*/64);
        ModelExecutor exec(pool, asm_, nullptr);
        cann_lm::engine::ModelConfig cfg = mc_sched_mock::MakeConfig();
        cfg.sliding_window = sw;
        cfg.prefill_len = {P};
        cfg.sliding_window_head_dim = 64;
        ASSERT_TRUE(exec.Load(mc_sched_mock::MakeOmc(), mc_sched_mock::kWeight, cfg, /*seq_len=*/8).ok())
            << "sw=" << sw;

        const auto& ins = exec.InputTensors();
        ASSERT_EQ(ins.size(), 5u);
        const auto& full = ins[0]->GetTensorDesc();
        const auto& slid = ins[1]->GetTensorDesc();
        const auto& spos = ins[4]->GetTensorDesc();
        // full 层不受新契影响。slid 列宽断言用测试侧独立推演的字面值
        // （sw=4 → C=12、sw=16 → C=24——非复述 SlidingLayerWidth 自身公式，
        // 顺带互证助手与字面口径一致：断言值错或助手错都逃不出本行）。
        EXPECT_EQ(full.dims, (std::vector<int32_t>{1, 1, 8, 16}));
        ASSERT_EQ(slid.dims, (std::vector<int32_t>{1, 1, 8, sw + P})) << "sw=" << sw << " 期望字面 C=sw+P";
        EXPECT_EQ(slid.dims[3], SlidingLayerWidth(sw, P)) << "助手口径互证";
        // sliding_kv_pos 新契同 {seq_len} 修形（1027 图 [64,1,1,1] → [64] 同构）
        EXPECT_EQ(spos.dims, (std::vector<int32_t>{8})) << "sw=" << sw;
    }
}

} // namespace
} // namespace cann_lm::engine
