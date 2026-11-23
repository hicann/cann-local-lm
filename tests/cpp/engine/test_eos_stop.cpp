// eos 停止（meta tokenizer_config.eos_token_id）管线级 UT（2026-10-29 裁决：
// 推翻 2026-09-07 "仅记日志"，方案见 tmp_docs/special_tokens_consume_and_eos_stop_plan.md）。
// 覆盖契约（UT-TC-EOS-001..003）：
//   UT-TC-EOS-001（eos 命中即停）：prefill 末块采样首 token 即 eos →
//     FinishReason::StopToken（D-1：复用 StopToken，归因次序在请求级
//     stop_token_ids 之后）、token_ids 含 eos（D-2）、text 空（D-2：不含
//     eos 字面量）、流式回调不触发（stop 分支无回调）、前向链恰 1 次
//     prefill 无 decode 推进（终态请求不入 decode 池——ReclaimAndFill_
//     首段竞速防御分支）。
//   UT-TC-EOS-002（D-4 台账排除 eos）：两轮同 Context——轮 1 首生成 token
//     即 eos 终止（台账 = 纯 prompt [349,349]）；轮 2 prompt 尾带 eos →
//     [349,349,1]，LCP=2=台账长 ⇒ increments_empty ⇒ reusable=LCP−1（末
//     token 强制重算）⇒ prefill 写位 {1,2}（观测通道 = mock Run 记录
//     new_kv_cache_pos；若 eos 入账则台账 [349,349,1]、LCP=3、reusable=2、
//     写位退化为 {2}——本断言即 D-4 判别锚点）。轮 2 关闭 eos 注入，走
//     max_tokens 终态（token/text/回调数全量口径）。
//   UT-TC-EOS-003（未配置零变化回归）：eos_token_id=-1 + mock 照发 eos 位
//     logits → 不停止、token_ids={1,1,1}、text="<eos>"×3（text 门控不
//     活跃）、回调 3 次（包括末 token）。
// 独立目标缘由（同 ut_scheduler_multichunk）：本文件经链接期符号覆盖
// hiai::CreateModelManager / CreateBuiltModel / CreateNDTensorBuffer
// （cann_lm_engine 为 STATIC 库，ut 对象定义优先）；不并入其他 binary——
// 工厂覆盖为二进制级符号替换，劫持面不可控。
// 与 multichunk mock 的差异：input_embed 工位 INT32（管线恒挂 text_embedding_
// 组件 → FillInputs TokenId 路径要求 INT32/INT64，FLOAT16 → INVALID_ARGUMENT
// 请求 Failed）；输出 {1,512} FLOAT32 2D（CorrectTensorDescs 直过）作 logits
// 行，Run 写 logits + 记录 npos。meta：kv_cache_max_len=64 / layers=2 /
// kv_heads=4 / head_dims=8 / batch_num=1（decode Load batch 须正）/ vocab_size
// =512 / pad_token_id=0；tokenizer_file = bpe fixture 绝对路径
// （EOS_TOKENIZER_FIXTURE，CMake 注入；词表 ids 0–349：1=<eos>、0=<pad>、
// 4='a'、349=<start_of_image>）；eos_token_id 可配（-1 = 未配置）。
// 采样确定性：默认 SamplingParams{}（temperature=1.0/top_k=0/top_p=1.0）→
// do_sample=false → 贪心 argmax 首下标——mock 行全等时选 0（<pad>）、
// eos 位独大时选 1（<eos>），零随机。
// npos 观测：float mask 路径（attention_mask 在场）AssemblePrefill 只写
// [0, chunk_len) 个值、工位容量 seq_len=8 尾部为残值——HostTensorBuffer 零
// 初始化 + 本 UT prefill chunk 恒 ≤ 2 长、写位恒在工位前部（无绕回）⇒ 残值
// 恒零，mock 记录工位全量值 = 真实写位 + 零填充（decode 工位容量 1 恰满，
// 即纯真实写位）。worker 单线程调 Run，测试线程仅在 handle.Wait 同步点
// （SetTerminal 锁序）后读，无竞态。
#include "api/cpp/engine_executor.h"
#include "model_manager/model_manager.h"
#include "model/built_model.h"
#include "tensor/nd_tensor_buffer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ── mock 基建（具名 namespace：外部链接类型用于工厂覆盖签名）──────────────────
namespace eos_stop_mock {

constexpr std::size_t kVocabWidth = 512; // logits 行宽（词表 ids 0–349，行宽余量）

// host 可用 INDTensorBuffer 最小实现（同 ut_scheduler_multichunk；host 无
// NDK so，真工厂返 nullptr）
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

// Mock IBuiltModel：buffer 态 Restore 恒过。input_embed INT32（TokenId 路径
// dtype 防线）+ position_ids/new_kv_cache_pos（INT32，CorrectTensorDescs 按
// 名修形：prefill seq=8 / decode seq=1）+ attention_mask FLOAT32 单 mask 最简
// 工位族；输出单工位 {1,512} FLOAT32 2D（>2D 截断/4D 修形均不触发，直过）。
class EosBuiltModel final : public hiai::IBuiltModel {
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
        embed.dataType = hiai::DataType::INT32; // TokenId 路径（图内查表形态）
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
        // CorrectTensorDescs 修形为 {batch,1,seq_len,kv_max} 全宽；此处给
        // 合法占位，Load 期按测试配置修正
        mask.dims = {1, 1, 1, 16};
        return {embed, pos, npos, mask};
    }
    std::vector<hiai::NDTensorDesc> GetOutputTensorDescs() const override
    {
        hiai::NDTensorDesc d;
        d.tensorName = "logits";
        d.dataType = hiai::DataType::FLOAT32;
        d.dims = {1, static_cast<int32_t>(kVocabWidth)};
        return {d};
    }
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

private:
    hiai::CustomModelData custom_data_;
};

// 采样控制（TC_EOS3 关闭）：true → logits[1]（eos 位）独大，贪心 argmax 命中
// eos；false → 全等行 argmax 取首下标 0（<pad>）
bool g_emit_eos = false;
// 每次 Run 的 new_kv_cache_pos 工位全量值（D-4 台账观测通道）。float mask
// 路径 AssemblePrefill 只写 [0, chunk_len)、工位容量 seq_len 尾部为残值——
// HostTensorBuffer 零初始化 + 本 UT 写位恒在工位前部（chunk ≤ 2 长、无绕回）
// ⇒ 残值恒零、全量值即"真实写位 + 零填充"（decode 工位容量 1 恰满无残值）
std::vector<std::vector<int32_t>> g_npos_log;

// Mock IModelManager：Run 写 logits 行 + 记录 npos；返回值经类级静态
// run_status 可配（失败注入通道，本 UT 未用，保留与 multichunk 同构）
class EosModelManager final : public hiai::IModelManager {
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
    hiai::Status Run(const std::vector<std::shared_ptr<hiai::INDTensorBuffer>>& inputs,
                     std::vector<std::shared_ptr<hiai::INDTensorBuffer>>& outputs) override
    {
        // npos 观测：按工位名取全量值（残值零语义见 g_npos_log 声明处注释；
        // decode 容量 1 恰满即纯真实写位）
        for (const auto& in : inputs) {
            if (!in || in->GetTensorDesc().tensorName != "new_kv_cache_pos")
                continue;
            const auto* p = static_cast<const int32_t*>(in->GetData());
            if (p == nullptr)
                continue;
            std::size_t n = 1;
            for (int32_t d : in->GetTensorDesc().dims) {
                n *= static_cast<std::size_t>(d > 0 ? d : 1);
            }
            g_npos_log.emplace_back(p, p + n);
        }
        // logits 行写入（outputs[0]：FLOAT32 {1,512}，行宽 = GetSize/4）
        if (!outputs.empty() && outputs[0] && outputs[0]->GetData()) {
            auto* logits = static_cast<float*>(outputs[0]->GetData());
            const std::size_t row = outputs[0]->GetSize() / sizeof(float);
            for (std::size_t i = 0; i < row; ++i)
                logits[i] = 0.1f;
            if (g_emit_eos && row > 1)
                logits[1] = 5.0f;
        }
        return static_cast<hiai::Status>(run_status);
    }
    hiai::Status ReleaseMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    hiai::Status RecoverMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    void DeInit() override { built_.reset(); }

private:
    std::shared_ptr<hiai::IBuiltModel> built_;
};

int EosModelManager::run_status = 0;

void ResetMocks()
{
    EosModelManager::run_status = 0;
    g_emit_eos = false;
    g_npos_log.clear();
}

} // namespace eos_stop_mock

// ── 工厂覆盖（hiai 命名空间；链接期符号优先于库内实现）──────────────────────
namespace hiai {
std::shared_ptr<IModelManager> CreateModelManager() { return std::make_shared<eos_stop_mock::EosModelManager>(); }
std::shared_ptr<IBuiltModel> CreateBuiltModel() { return std::make_shared<eos_stop_mock::EosBuiltModel>(); }
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc)
{
    return std::make_shared<eos_stop_mock::HostTensorBuffer>(tensorDesc);
}
} // namespace hiai

namespace cann_local_lm::engine {
namespace {

#ifndef EOS_TOKENIZER_FIXTURE
#define EOS_TOKENIZER_FIXTURE "tests/cpp/engine/fixtures/tokenizer"
#endif

// meta：容量域合法（kv_max/layers/heads/dims > 0、batch_num=1）+ tokenizer
// fixture 绝对路径 + eos_token_id 可配（-1 = 未配置 → 检查禁用）。
// max_io_tokens 缺省 4096（解析层恒正，预算闸不触）；prefill_len 缺省 →
// 块长链回退 EngineOptions.default_chunk_size；init_token_len 缺省 0（无锚定段）。
std::string MetaJson(int eos_token_id)
{
    std::string j;
    j += R"({"model_config":{"kv_cache_max_len":64,"num_hidden_layers":2,)";
    j += R"("num_attention_kv_heads":4,"num_attention_head_dims":8,"batch_num":1,)";
    j += R"("vocab_size":512,"pad_token_id":0},)";
    j += R"("tokenizer_config":{"tokenizer_file":")";
    j += EOS_TOKENIZER_FIXTURE;
    j += "/bpe_fixture_tokenizer.json\",\"eos_token_id\":";
    j += std::to_string(eos_token_id);
    j += "}}";
    return j;
}

// omc buffer 源（合法 8 字节）+ mock 权重路径（InitWeights 非空即 SUCCESS）。
// meta_json Buffer 为非所有权视图——指向调用方 meta 字符串（EosTestEnv 成员
// 声明序保证其存活至 EngineExecutor 构造返回）
ModelArtifact MakeArtifact(const std::string& meta)
{
    static char omc_buf[8];
    ModelArtifact a;
    a.prefill_omc.data = Buffer{omc_buf, sizeof(omc_buf)};
    a.decode_omc.data = Buffer{omc_buf, sizeof(omc_buf)};
    a.prefill_weight = "/mock/weights";
    a.decode_weight = "/mock/weights";
    a.meta_json.data = Buffer{meta.data(), meta.size()};
    return a;
}

// 测试载体：meta 先于 engine 构造（buffer 视图存活）、后于其析构销毁
struct EosTestEnv {
    std::string meta;
    std::unique_ptr<EngineExecutor> engine;

    explicit EosTestEnv(int eos_token_id) : meta(MetaJson(eos_token_id))
    {
        EngineOptions opts;
        opts.default_chunk_size = 8; // prefill_len 缺省 → 块长链回退 options
        engine = std::make_unique<EngineExecutor>(MakeArtifact(meta), opts);
    }
};

// ── UT-TC-EOS-001：eos 命中即停 ─────────────────────────────────────────────

TEST(EosStopTest, EosHitStopsAtFirstSampledToken_TC_EOS1)
{
    eos_stop_mock::ResetMocks();
    EosTestEnv env(/*eos_token_id=*/1);
    ASSERT_TRUE(env.engine->InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
    ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok()); // ADR-0002：显式激活

    eos_stop_mock::g_emit_eos = true; // logits eos 位独大 → 首采样 token = 1
    Prompt prompt;
    prompt.text = "<start_of_image><start_of_image>"; // encode → [349, 349]
    std::atomic<int> callbacks{0};
    RequestHandle handle;
    ASSERT_TRUE(env.engine
                    ->GenerateAsync(prompt, SamplingParams{}, RequestOptions{}, &handle,
                                    [&callbacks](TokenId, std::string) { ++callbacks; })
                    .ok());
    Result result;
    ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);

    EXPECT_EQ(result.finish_reason, FinishReason::StopToken); // D-1：复用 StopToken
    EXPECT_EQ(result.token_ids, std::vector<TokenId>({1}));   // D-2：token_ids 含 eos
    EXPECT_EQ(result.text, std::string(""));                  // D-2：text 不含 eos 字面量
    EXPECT_EQ(callbacks.load(), 0);                           // stop 分支无回调
    // 前向链：恰 1 次 prefill（首 token 即 eos 于 prefill 末块采样点终止，
    // 终态请求不入 decode 池）；写位 {0,1} = 全量 prompt（工位容量 8，尾零残值）
    ASSERT_EQ(eos_stop_mock::g_npos_log.size(), 1u);
    EXPECT_EQ(eos_stop_mock::g_npos_log[0], (std::vector<int32_t>{0, 1, 0, 0, 0, 0, 0, 0}));
}

// ── UT-TC-EOS-002：D-4 台账排除 eos（写位判别锚点）──────────────────────────

TEST(EosStopTest, PromptCacheLedgerExcludesEos_TC_EOS2)
{
    eos_stop_mock::ResetMocks();
    EosTestEnv env(/*eos_token_id=*/1);
    ASSERT_TRUE(env.engine->InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
    ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok());

    // 轮 1：首生成 token 即 eos 终止——台账 = 纯 prompt [349,349]（D-4：eos
    // 不入账，与池 total_written=2 对齐）
    eos_stop_mock::g_emit_eos = true;
    Prompt p1;
    p1.text = "<start_of_image><start_of_image>"; // [349, 349]
    RequestHandle h1;
    ASSERT_TRUE(env.engine->GenerateAsync(p1, SamplingParams{}, RequestOptions{}, &h1).ok());
    Result r1;
    ASSERT_EQ(h1.Wait(&r1).code, Status::SUCCESS);
    ASSERT_EQ(r1.finish_reason, FinishReason::StopToken);

    // 轮 2：prompt 尾带 eos → [349,349,1]；LCP=2=台账长 ⇒ reusable=1（末
    // token 强制重算）⇒ prefill 写位 {1,2}（eos 入账则 LCP=3、reusable=2、
    // 写位 {2}）。本轮关 eos 注入走 max_tokens 终态
    eos_stop_mock::g_emit_eos = false;
    eos_stop_mock::g_npos_log.clear();
    Prompt p2;
    p2.text = "<start_of_image><start_of_image><eos>"; // [349, 349, 1]
    RequestOptions ro;
    ro.max_tokens = 3;
    std::atomic<int> callbacks{0};
    RequestHandle h2;
    ASSERT_TRUE(
        env.engine->GenerateAsync(p2, SamplingParams{}, ro, &h2, [&callbacks](TokenId, std::string) { ++callbacks; })
            .ok());
    Result r2;
    ASSERT_EQ(h2.Wait(&r2).code, Status::SUCCESS);

    EXPECT_EQ(r2.finish_reason, FinishReason::MaxTokens);
    EXPECT_EQ(r2.token_ids, std::vector<TokenId>({0, 0, 0})); // 全等行 argmax → <pad>
    EXPECT_EQ(r2.text, std::string("<pad><pad><pad>"));
    EXPECT_EQ(callbacks.load(), 3); // 包括达到 max_tokens 的最后一个 token
    // 写位判别（D-4 锚点）：prefill {1,2}（尾零残值）+ decode {3},{4}
    ASSERT_EQ(eos_stop_mock::g_npos_log.size(), 3u);
    EXPECT_EQ(eos_stop_mock::g_npos_log[0], (std::vector<int32_t>{1, 2, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(eos_stop_mock::g_npos_log[1], (std::vector<int32_t>{3}));
    EXPECT_EQ(eos_stop_mock::g_npos_log[2], (std::vector<int32_t>{4}));
}

// ── UT-TC-EOS-003：eos 未配置（-1）零变化回归 ───────────────────────────────

TEST(EosStopTest, EosUnconfiguredNoBehaviorChange_TC_EOS3)
{
    eos_stop_mock::ResetMocks();
    EosTestEnv env(/*eos_token_id=*/-1); // 未配置 → eos 检查禁用
    ASSERT_TRUE(env.engine->InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
    ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok());

    // mock 照发 eos 位 logits：采样恒出 token 1，但 eos_id=-1 → 不停止、
    // token 照常入 token_ids/text/台账（改动前行为逐位不变）
    eos_stop_mock::g_emit_eos = true;
    Prompt prompt;
    prompt.text = "a"; // [4]
    RequestOptions ro;
    ro.max_tokens = 3;
    std::atomic<int> callbacks{0};
    RequestHandle handle;
    ASSERT_TRUE(
        env.engine
            ->GenerateAsync(prompt, SamplingParams{}, ro, &handle, [&callbacks](TokenId, std::string) { ++callbacks; })
            .ok());
    Result result;
    ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);

    EXPECT_EQ(result.finish_reason, FinishReason::MaxTokens);
    EXPECT_EQ(result.token_ids, std::vector<TokenId>({1, 1, 1}));
    EXPECT_EQ(result.text, std::string("<eos><eos><eos>")); // text 门控不活跃
    EXPECT_EQ(callbacks.load(), 3);                         // 末 token 也回调
    // 前向链不变：prefill {0}（单 token prompt，位 0 写值 + 尾零残值）+
    // decode {1},{2}（池逐位推进 0→1→2，逐 token 照常入账）
    ASSERT_EQ(eos_stop_mock::g_npos_log.size(), 3u);
    EXPECT_EQ(eos_stop_mock::g_npos_log[0], (std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(eos_stop_mock::g_npos_log[1], (std::vector<int32_t>{1}));
    EXPECT_EQ(eos_stop_mock::g_npos_log[2], (std::vector<int32_t>{2}));
}

TEST(EosStopTest, SingleTokenBudgetEmitsBeforeWaitReturns)
{
    eos_stop_mock::ResetMocks();
    EosTestEnv env(-1);
    ASSERT_TRUE(env.engine->InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
    ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "a";
    RequestOptions options;
    options.max_tokens = 1;
    std::vector<TokenId> streamed_ids;
    std::string streamed_text;
    RequestHandle handle;
    ASSERT_TRUE(env.engine
                    ->GenerateAsync(prompt, SamplingParams{}, options, &handle,
                                    [&](TokenId token, std::string text) {
                                        streamed_ids.push_back(token);
                                        streamed_text += text;
                                    })
                    .ok());
    Result result;
    ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);
    EXPECT_EQ(result.finish_reason, FinishReason::MaxTokens);
    ASSERT_EQ(result.token_ids.size(), 1u);
    EXPECT_EQ(streamed_ids, result.token_ids);
    EXPECT_EQ(streamed_text, result.text);
}

TEST(EosStopTest, ExplicitStopsAreNotStreamedAtTokenBudget)
{
    for (bool use_string : {false, true}) {
        eos_stop_mock::ResetMocks();
        EosTestEnv env(-1);
        ASSERT_TRUE(env.engine->InitStatus().ok());
        Context ctx;
        ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
        ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok());
        Prompt prompt;
        prompt.text = "a";
        RequestOptions options;
        options.max_tokens = 1;
        if (use_string)
            options.stop_strings = {"<pad>"};
        else
            options.stop_token_ids = {0};
        int callbacks = 0;
        RequestHandle handle;
        ASSERT_TRUE(
            env.engine
                ->GenerateAsync(prompt, SamplingParams{}, options, &handle, [&](TokenId, std::string) { ++callbacks; })
                .ok());
        Result result;
        ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);
        EXPECT_EQ(result.finish_reason, use_string ? FinishReason::StopString : FinishReason::StopToken);
        EXPECT_EQ(callbacks, 0);
    }
}

TEST(EosStopTest, FinalCallbackExceptionDoesNotPreventCompletion)
{
    eos_stop_mock::ResetMocks();
    EosTestEnv env(-1);
    ASSERT_TRUE(env.engine->InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(env.engine->CreateContext(&ctx).ok());
    ASSERT_TRUE(env.engine->SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "a";
    RequestOptions options;
    options.max_tokens = 1;
    int callbacks = 0;
    RequestHandle handle;
    ASSERT_TRUE(env.engine
                    ->GenerateAsync(prompt, SamplingParams{}, options, &handle,
                                    [&](TokenId, std::string) {
                                        ++callbacks;
                                        throw 1;
                                    })
                    .ok());
    Result result;
    ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(result.finish_reason, FinishReason::MaxTokens);
}

TEST(EosStopTest, ModelIoBudgetEmitsFinalToken)
{
    eos_stop_mock::ResetMocks();
    std::string meta = MetaJson(-1);
    const auto model_begin = meta.find("\"model_config\":{");
    ASSERT_NE(model_begin, std::string::npos);
    meta.insert(model_begin + std::string("\"model_config\":{").size(), "\"max_io_tokens\":2,");
    EngineOptions engine_options;
    engine_options.default_chunk_size = 8;
    EngineExecutor engine(MakeArtifact(meta), engine_options);
    ASSERT_TRUE(engine.InitStatus().ok());
    Context ctx;
    ASSERT_TRUE(engine.CreateContext(&ctx).ok());
    ASSERT_TRUE(engine.SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "a";
    RequestOptions options;
    options.max_tokens = 10;
    std::vector<TokenId> streamed;
    RequestHandle handle;
    ASSERT_TRUE(engine
                    .GenerateAsync(prompt, SamplingParams{}, options, &handle,
                                   [&](TokenId token, std::string) { streamed.push_back(token); })
                    .ok());
    Result result;
    ASSERT_EQ(handle.Wait(&result).code, Status::SUCCESS);
    ASSERT_EQ(result.token_ids.size(), 1u);
    EXPECT_EQ(result.finish_reason, FinishReason::MaxTokens);
    EXPECT_EQ(streamed, result.token_ids);
}

} // namespace
} // namespace cann_local_lm::engine
