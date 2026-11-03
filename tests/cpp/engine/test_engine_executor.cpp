#include "api/cpp/engine_executor.h"
#include "pipeline/pipeline_executor.h" // SR20260903001：内联解析步骤 UT 载体

#include "stderr_capture.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace cann_local_lm::engine {
namespace {

// SR20260903001：单一构件（双图位）。三路径全空 = 无模型部署形态
// （stub-era / 无设备 UT，2026-09-22 契约第 2 点——双图位全空分支）：
// 构造跳过模型加载，生命周期 API 可用，Generate 报 executor not-ready。
// stub-era 生命周期 TC 共用。
ModelArtifact DummyArtifact()
{
    ModelArtifact a;
    a.prefill_omc.path = "";
    a.decode_omc.path = "";
    a.prefill_weight = "";
    a.decode_weight = "";
    return a;
}

// 图位已配置但文件不存在（dummy.* 路径不落盘）：触发 2026-09-22 新契约——
// ModelExecutor::Load 失败 ⇒ 构造内终止初始化、显式卸载双执行器、
// init_status_ 记失败码（经 EngineExecutor InitStatus / C API 原样上抛）。
ModelArtifact MissingModelArtifact()
{
    ModelArtifact a;
    a.prefill_omc.path = "dummy.omc";
    a.decode_omc.path = "dummy.omc";
    a.prefill_weight = "dummy.weight"; // 共用权重口径（两字段同值 = 现状布局）
    a.decode_weight = "dummy.weight";
    return a;
}

TEST(EngineExecutorTest, CreateContextSucceeds_TC001)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(static_cast<bool>(ctx));
    EXPECT_TRUE(static_cast<bool>(ctx.id()));
}

TEST(EngineExecutorTest, DestroyContextReturnsSuccessIdempotent)
{
    EngineExecutor engine_executor(DummyArtifact());
    auto s = engine_executor.DestroyContext(ContextId{42});
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, ConfigureContextReturnsNotFound_TC101)
{
    EngineExecutor engine_executor(DummyArtifact());
    ContextOptions opts;
    auto s = engine_executor.ConfigureContext(ContextId{99}, opts);
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

TEST(EngineExecutorTest, CancelContextReturnsNotFound_TC102)
{
    EngineExecutor engine_executor(DummyArtifact());
    auto s = engine_executor.CancelContext(ContextId{99});
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

TEST(EngineExecutorTest, ConfigureContextSucceedsAfterCreate)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    ContextOptions opts;
    auto s = engine_executor.ConfigureContext(ctx.id(), opts);
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, CancelContextSucceedsAfterCreate)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    auto s = engine_executor.CancelContext(ctx.id());
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, GenerateReturnsFailureOnEmptyContext)
{
    // ADR-0002 S2：无活动 Context（未 SetContext）→ INVALID_STATE
    EngineExecutor engine_executor(DummyArtifact());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    Result result;
    auto s = engine_executor.Generate(prompt, sp, RequestOptions{}, &result);
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

TEST(EngineExecutorTest, GenerateReturnsFailureOnValidContextStubEra)
{
    // 活动 Context 就绪但 tokenizer 为 stub（encode 返回 FAILURE）→ Generate 失败
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    ASSERT_TRUE(engine_executor.SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    Result result;
    auto s = engine_executor.Generate(prompt, sp, RequestOptions{}, &result);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.code, Status::INVALID_STATE); // S2 已过门，失败源于 stub encode
}

TEST(EngineExecutorTest, GenerateAsyncReturnsFailureOnEmptyContext)
{
    EngineExecutor engine_executor(DummyArtifact());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    auto s = engine_executor.GenerateAsync(prompt, sp, RequestOptions{}, &handle);
    EXPECT_EQ(s.code, Status::INVALID_STATE);
}

TEST(EngineExecutorTest, GenerateAsyncReturnsFailureOnValidContextStubEra)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    ASSERT_TRUE(engine_executor.SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    auto s = engine_executor.GenerateAsync(prompt, sp, RequestOptions{}, &handle);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.code, Status::INVALID_STATE);
}

// UT-TC-EE-011 (DRIFT-1 修复): 名宣称 INVALID_STATE 但实际两次均返回 FAILURE
// (首次同步失败 + RAII 清除 active 标记后第二次仍 FAILURE)。改名反映真实行为。
// 详见 docs/ut/SPEC.md §4.1 / docs/ut/DESIGN.md §8。
TEST(EngineExecutorTest, ConcurrentGenerateSameContextReturnsFailureStubEra)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    ASSERT_TRUE(engine_executor.SetContext(ctx.id()).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle h1;
    // stub 时代下 PreparePrefill 同步失败（BPETokenizer::encode 返回 FAILURE），
    // active_generates_ 经 RAII 在返回前清除 → 第二次 GenerateAsync 不会触发 INVALID_STATE。
    auto s1 = engine_executor.GenerateAsync(prompt, sp, RequestOptions{}, &h1);
    EXPECT_FALSE(s1.ok());
    // 第二次调用同样返回 FAILURE（不是 INVALID_STATE，因 active 标记已被首次调用清除）
    RequestHandle h2;
    auto s2 = engine_executor.GenerateAsync(prompt, sp, RequestOptions{}, &h2);
    EXPECT_FALSE(s2.ok());
}

// ── 失败路径日志断言（C11 契约：Status 粗码 + 日志详因）──────────────────────

// host 无模型且图位已配置（文件不存在）：Load 失败 ⇒ 构造中止初始化并卸载
// 双执行器。日志双断言：Load 层根因键 loader.load.*（RestoreFromFile 失败）+
// pipeline 层中止键 pipeline.init（C11 契约：Status 粗码 + 日志详因，两层
// 均可观测）。失败实例不交出内部 pipeline：InitStatus 记失败码，
// CreateContext NULL 防御返回 FAILURE（工厂不交出半初始化实例）。
TEST(EngineExecutorTest, LoadFailureAbortsInitAndReports)
{
    test_util::StderrCapture cap;
    EngineExecutor engine_executor(MissingModelArtifact());
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("loader.load"), std::string::npos);
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_FALSE(engine_executor.InitStatus().ok());
    Context ctx; // 默认构造的出参 out（失败实例调用方出参）
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// ── C-1 修复回归（2026-09-23，failed_but_not_processed.md §7.3；D-1 温和口径）──
// 部署形态（omc 图位配置）+ meta_json 有供给意图（buffer 注入）+ 关键内容坏 ⇒
// 构造致命：init_status_ = INVALID_ARGUMENT、工厂拒交实例（CreateContext 走
// NULL 防御 FAILURE）、中止点先于 KV 定容/双执行器 Load——host 无 NPU 可正测
// （loader.load.restore 不出现——模型图未尝试加载 = 门在加载之前的回归锚点；
// tokenizer 的 loader.load.tokenizer 在门前的成员初始化期发出，属既有非阻塞
// 降级，不在本断言面）。buffer 用字符串字面量
// （静态存储期，构造期间存活；去尾 NUL）。

/// C-1 用例载体：部署形态（omc 配置）+ meta_json buffer 注入
template <size_t N>
ModelArtifact DeployedWithMetaJson(const char (&meta)[N])
{
    ModelArtifact a = MissingModelArtifact();
    a.meta_json.data = Buffer{meta, N - 1};
    return a;
}

// 情形①：JSON 语法非法（有意图 buffer，内容不是合法 JSON）
TEST(EngineExecutorTest, MetaJsonSyntaxFatalAbortsInitOnDeployedForm)
{
    test_util::StderrCapture cap;
    EngineExecutor engine_executor(DeployedWithMetaJson("{ not valid json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(engine_executor.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// 情形②：JSON 语法合法但 model_config 段缺失（关键段缺失 → 桩默认会在
// KV 定容/形状修正处批量爆炸，C-1 提前在构造门口暴露）
TEST(EngineExecutorTest, MetaJsonMissingModelSectionFatalAbortsInit)
{
    test_util::StderrCapture cap;
    EngineExecutor engine_executor(
        DeployedWithMetaJson(R"json({"tokenizer_config":{"tokenizer_file":"dummy.json"}})json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(engine_executor.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// 情形③：model_config 段存在但解析失败（kv_cache_max_len ≤ 0 容量非法）
TEST(EngineExecutorTest, MetaJsonBadModelSectionFatalAbortsInit)
{
    test_util::StderrCapture cap;
    EngineExecutor engine_executor(
        DeployedWithMetaJson(R"json({"model_config":{"kv_cache_max_len":-1,"num_hidden_layers":2}})json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(engine_executor.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// 情形④（U2 部署形态半，方案 §3.1 D2）：max_io_tokens 显式 ≤ 0 = 配置
// 非法 ⇒ 同 ③ 段降级 → C-1 门以 meta_json invalid 终止。REQ-9 预算闸的
// 解析层把关在部署形态的致命面（静默降默认 4096 会让配置者拿到相反行为）。
TEST(EngineExecutorTest, MaxIoTokensNonPositiveFatalAbortsInitOnDeployedForm)
{
    test_util::StderrCapture cap;
    EngineExecutor engine_executor(DeployedWithMetaJson(
        R"json({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                            "max_io_tokens":0}})json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(engine_executor.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = engine_executor.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// D-1 温和门边界：meta_json 同样坏，但 omc 双空（无模型部署形态）⇒ 门不
// 触发，维持 D-16 原口径（桩默认 + WARN 降级 + init SUCCESS，生命周期照常）。
TEST(EngineExecutorTest, BadMetaJsonToleratedWhenOmcAbsent)
{
    test_util::StderrCapture cap; // 仅收集降级 WARN，不属于本断言面
    ModelArtifact a = DummyArtifact();
    static const char kBadMeta[] = "{ not valid json";
    a.meta_json.data = Buffer{kBadMeta, sizeof(kBadMeta) - 1};
    EngineExecutor engine_executor(a);
    (void)cap.Stop();
    EXPECT_TRUE(engine_executor.InitStatus().ok());
    Context ctx;
    EXPECT_TRUE(engine_executor.CreateContext(&ctx).ok());
}

// ADR-0002 SetContext 语义矩阵（S2/S3/S5/S6）基础用例——host/stub 形态
// （零 desc 账本 → 零 tensor Context 合法，生命周期 API 恒成功，Q12 裁定 a）。
TEST(EngineExecutorTest, SetContextNotFoundOnUnregisteredId_S3)
{
    EngineExecutor engine_executor(DummyArtifact());
    EXPECT_EQ(engine_executor.SetContext(ContextId{99}).code, Status::NOT_FOUND);
}

TEST(EngineExecutorTest, SetContextSucceedsAndIdempotent_S5)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    EXPECT_TRUE(engine_executor.SetContext(ctx.id()).ok());
    EXPECT_TRUE(engine_executor.SetContext(ctx.id()).ok()); // 幂等
}

TEST(EngineExecutorTest, DestroyActiveContextReturnsToNoActiveState_S6)
{
    EngineExecutor engine_executor(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(engine_executor.CreateContext(&ctx).ok());
    ASSERT_TRUE(engine_executor.SetContext(ctx.id()).ok());
    EXPECT_TRUE(engine_executor.DestroyContext(ctx.id()).ok());
    // 无活动 Context：Generate 拒绝（S2 门禁回位）
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    EXPECT_EQ(engine_executor.GenerateAsync(prompt, sp, RequestOptions{}, &handle).code, Status::INVALID_STATE);
    // 重销毁幂等（S8）
    EXPECT_TRUE(engine_executor.DestroyContext(ctx.id()).ok());
}

// context 不存在：SetContext NOT_FOUND 粗码 + pipeline.set_context 日志详因。
TEST(EngineExecutorTest, SetContextNotFoundLogsPipelineSetContext)
{
    EngineExecutor engine_executor(DummyArtifact());
    test_util::StderrCapture cap;
    auto s = engine_executor.SetContext(ContextId{99});
    const std::string out = cap.Stop();
    EXPECT_EQ(s.code, Status::NOT_FOUND);
    EXPECT_NE(out.find("pipeline.set_context"), std::string::npos);
}

} // namespace
} // namespace cann_local_lm::engine

namespace cann_local_lm::engine {
// ═══════════════════════════════════════════════════════════════════════════
// SR20260903001：meta_json 三段解析（内联步骤）行为用例。
// 载体：AutoregressivePipelineExecutor 直构 + 只读观测点（M-08 §8）——
// host 无模型场景下模型 Load 失败属正常，解析产出与加载成败无关。
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// 极简合法 BPE 词表产物（TC_010 用；自包含，不依赖 fixture 目录）
std::string WriteTempBpeVocab()
{
    static int seq = 0;
    const std::string p = "/tmp/ut_cfg_bpe_" + std::to_string(++seq) + ".json";
    std::ofstream f(p);
    f << R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}})";
    return p;
}

/// model.type 非 "BPE" 的词表产物（TC_214 用）：BPE 强校验拒收、Qwen 不校验收下
/// ——用于从外部区分算法选择是否生效。
std::string WriteTempNonBpeVocab()
{
    static int seq = 0;
    const std::string p = "/tmp/ut_cfg_nonbpe_" + std::to_string(++seq) + ".json";
    std::ofstream f(p);
    f << R"({"model":{"type":"Unigram","vocab":{"a":0,"b":1},"merges":[]}})";
    return p;
}

/// 按值收串 + 静态保活：Buffer 是非拥有视图，须保证源串存活至构造完成
/// （初版绑临时串导致悬垂——全段降级的根因）。
Buffer Buf(std::string s)
{
    static std::vector<std::string> keep;
    keep.push_back(std::move(s));
    return Buffer{keep.back().data(), keep.back().size()};
}

} // namespace

// TC_001：三段完整解析赋值
TEST(ConfigParseTest, ThreeSectionsParsed_TC001)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "tokenizer_config": {"bos_token_id": 2, "eos_token_id": 1,
                             "bos_token": "<s>", "eos_token": "</s>",
                             "special_tokens": {"[unused9]": 45887}},
        "model_config": {"hidden_size": 2048, "max_io_tokens": 16384,
                         "model_type": "gemma4", "num_hidden_layers": 28,
                         "prefill_len": [1024], "decode_len": [16],
                         "pad_token_id": 0, "vocab_size": 153376,
                         "vocab_align_size": 153376,
                         "kv_cache_max_len": 16384, "kv_cache_format": "nd",
                         "init_token_len": 256},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_quant_mode": "per_token",
                             "embedding_load_mode": "in_memory",
                             "embedding_type": "fp16",
                             "embedding_scale_type": "fp16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    // model 段（TC_003 附带：KV 定容权威值）
    EXPECT_EQ(pipe.ParsedModelConfig().hidden_size, 2048);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 28);
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16384);
    EXPECT_EQ(pipe.ParsedModelConfig().pad_token_id, 0);
    EXPECT_EQ(pipe.ParsedModelConfig().vocab_size, 153376u);
    ASSERT_EQ(pipe.ParsedModelConfig().prefill_len.size(), 1u);
    EXPECT_EQ(pipe.ParsedModelConfig().prefill_len[0], 1024);
    // init_token_len（2026-10-22 静态化）：显式供给值解析落位
    EXPECT_EQ(pipe.ParsedModelConfig().init_token_len, 256);
    // tokenizer 段（pad 跨段 = 0）
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, 2);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().eos_token_id, 1);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().pad_token_id, 0);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token, "<s>");
    ASSERT_EQ(pipe.ParsedTokenizerConfig().special_tokens.size(), 1u);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().special_tokens.at("[unused9]"), 45887);
    // embedding 段（vocab/hidden 跨段）
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().table_path, "t.bin");
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().vocab_size, 153376u);
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().hidden_dim, 2048u);
}

// TC_002：meta_json 优先于 EngineOptions
TEST(ConfigParseTest, MetaJsonOverridesOptions_TC002)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"prefill_len":[2048],
        "kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.default_chunk_size = 512;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 2048);
}

// ── init_token_len 解析（2026-10-22 静态化）──

// 缺省 0：无锚定段（全缓冲圆环），迁移零行为变化的解析层锚点。
// TC_002 同 json 本就缺字段，这里补显式断言。
TEST(ConfigParseTest, InitTokenLenDefaultZero)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
        "num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().init_token_len, 0);
}

// 数字字符串容错（"128" 与 128 等价，TC_202 同口径）
TEST(ConfigParseTest, InitTokenLenNumericStringTolerated)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
        "num_hidden_layers":4,"init_token_len":"128"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().init_token_len, 128);
}

// 越界 → 段降级（负值 / ≥ kv_cache_max_len；口径同 sliding_window<0——
// 显式拒绝优于静默降 0）。rendered 为部署形态时构造失败由 C-1 用例族覆盖，
// 此处锁解析语义本身：同段其他字段一并不采信（fallback 桩默认）。
TEST(ConfigParseTest, InitTokenLenOutOfRangeDegradesSection)
{
    {
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4,"init_token_len":512}})"); // == kv_max
        test_util::StderrCapture cap;
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        const std::string out = cap.Stop();
        // omc 双空（无模型部署形态）⇒ C-1 门不触发（D-1 温和口径），
        // 段降级走桩默认：num_hidden_layers 回 2、init_token_len 回 0
        EXPECT_NE(out.find("字段类型非法或容量非法"), std::string::npos);
        EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
        EXPECT_EQ(pipe.ParsedModelConfig().init_token_len, 0);
    }
    {
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4,"init_token_len":-8}})");
        test_util::StderrCapture cap;
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        const std::string out = cap.Stop();
        EXPECT_NE(out.find("字段类型非法或容量非法"), std::string::npos);
        EXPECT_EQ(pipe.ParsedModelConfig().init_token_len, 0);
    }
}

// ── max_io_tokens 解析（方案 support_input_max_io_len_202610222243 §3.1
//    D2，REQ-9 预算闸的解析层锚点；口径同 init_token_len 组）──

// 缺省（键缺失/null）⇒ 字段默认 4096（输入上界归口值；与 kv_max 解耦）。
TEST(ConfigParseTest, MaxIoTokensDefault4096)
{
    {
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4}})");
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        EXPECT_EQ(pipe.ParsedModelConfig().max_io_tokens, 4096);
    }
    { // 显式 null 同缺省
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4,"max_io_tokens":null}})");
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        EXPECT_EQ(pipe.ParsedModelConfig().max_io_tokens, 4096);
    }
}

// 显式供给解析落位（含数字字符串容错，TC_202 同口径）。
TEST(ConfigParseTest, MaxIoTokensExplicitParsed)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
        "num_hidden_layers":4,"max_io_tokens":"2048"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().max_io_tokens, 2048);
}

// 越界（≤ 0 = 配置非法）→ 段降级（口径同 init_token_len 越界：显式拒绝优于
// 静默降默认——配置者显式写了 0/-1，静默给 4096 = 相反行为）。rendered 为
// 部署形态时构造失败由下方 MaxIoTokensNonPositiveFatalAbortsInitOnDeployedForm
// 承接（C-1 门），此处锁解析语义本身：同段其他字段一并不采信（桩默认）。
TEST(ConfigParseTest, MaxIoTokensNonPositiveDegradesSection)
{
    {
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4,"max_io_tokens":0}})");
        test_util::StderrCapture cap;
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        const std::string out = cap.Stop();
        EXPECT_NE(out.find("字段类型非法或容量非法"), std::string::npos);
        EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2); // 段降级回桩默认
        EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);
        EXPECT_EQ(pipe.ParsedModelConfig().max_io_tokens, 4096); // 桩默认（非采信值）
    }
    {
        ModelArtifact a;
        a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
            "num_hidden_layers":4,"max_io_tokens":-1}})");
        test_util::StderrCapture cap;
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        const std::string out = cap.Stop();
        EXPECT_NE(out.find("字段类型非法或容量非法"), std::string::npos);
        EXPECT_EQ(pipe.ParsedModelConfig().max_io_tokens, 4096);
    }
}

// ── sliding_window_head_dim 解析（sliding_kv_seq_dim_rebind D10；审查 SP-1：
//    方案 §5 U1 要求的 meta 解析拒绝元组此前缺位——"U1"编号已被 S5 修形用例
//    占用，本组补齐 meta 级 D10 面；判据单源 SlidingHeadDimComboValid 已有
//    直测，此处锁 pipeline 解析链端到端）──

// 合法新契元组：sw>0 + sw_hd>0 + prefill 档非空全正 → 解析落位（17 字段
// 之第 17 字段 sliding_window_head_dim 的直连观测点）。
TEST(ConfigParseTest, SwHeadDimValidTupleParsed)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,
        "num_hidden_layers":4,"sliding_window":256,
        "sliding_window_head_dim":128,"prefill_len":[1024, 2048]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& mc = pipe.ParsedModelConfig();
    EXPECT_EQ(mc.sliding_window, 256);
    EXPECT_EQ(mc.sliding_window_head_dim, 128);
    ASSERT_EQ(mc.prefill_len.size(), 2u);
    EXPECT_EQ(mc.prefill_len[1], 2048);
}

// 越界组合（D10）：sw_hd>0 搭配 sw=0 / prefill_len 缺省空 / prefill 档含
// 非正值 → 段降级（同段其他字段一并不采信，桩默认 kv_len=16/nhl=2）。
// sw=0 / sw<0 的段降级在 sliding_window 解析既有面（TC_204 同口径），此处
// 锁的是组合校验三元组——合法单字段组装不出 C=W+P 时整段拒绝。
TEST(ConfigParseTest, SwHeadDimD10TuplesDegradeSection)
{
    const char* cases[] = {
        // sw_hd>0 ∧ sw=0：窗口没配——新契滑窗层无窗可言
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
            "sliding_window_head_dim":128}})",
        // sw_hd>0 ∧ prefill_len 空数组：P 无档——C = W+P 算不出
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
            "sliding_window":256,"sliding_window_head_dim":128,
            "prefill_len":[]}})",
        // sw_hd>0 ∧ prefill 档含 0（非正值）
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
            "sliding_window":256,"sliding_window_head_dim":128,
            "prefill_len":[1024, 0]}})",
        // sw_hd>0 ∧ prefill 档含负值
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
            "sliding_window":256,"sliding_window_head_dim":128,
            "prefill_len":[-4]}})",
    };
    for (const char* json : cases) {
        ModelArtifact a;
        a.meta_json.data = Buf(json);
        test_util::StderrCapture cap;
        AutoregressivePipelineExecutor pipe(a, EngineOptions{});
        const std::string out = cap.Stop();
        EXPECT_NE(out.find("字段类型非法或容量非法"), std::string::npos) << "json=" << json;
        // 段降级：全段不采信（含 sliding_window 自身——组合非法牵连整段）
        EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16) << json;
        EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2) << json;
        EXPECT_EQ(pipe.ParsedModelConfig().sliding_window_head_dim, 0) << json;
    }
}

// ── U1 提交闸（方案 §3.1 §5 U1，REQ-9）────
// stub-era 达闸路径：装上真 BPE 词表（WriteTempBpeVocab，EncodeOptions 无
// special 注入——add_bos/add_eos 缺省 false）⇒ encode 'a'*N → N 个 id，构造
// 可控 ids 长度直触提交闸（闸在 Match/台账/池动作之前——"cache untouched"；
// encode 失败先于闸（stub-era 前提），故必须真词表）。断言面：
//   65 > max_io_tokens=64 → INVALID_ARGUMENT + promptcache.capacity 日志；
//   64 ≤ 64               → 过闸（后续走 stub-era not-ready/其他失败，
//   只要不是本闸的 INVALID_ARGUMENT capacity 语义即可——区分方式：日志
//   不含 promptcache.capacity）。worker 线程对两形态无副作用（闸先于 Submit）。
TEST(ConfigParseTest, SubmissionGateRejectsOverBudgetInputU1)
{
    const std::string vocab = WriteTempBpeVocab(); // {"a":0,"b":1}
    ModelArtifact a;
    a.meta_json.data = Buf(std::string(R"({"model_config":{"kv_cache_max_len":512,
        "num_hidden_layers":4,"max_io_tokens":64},
        "tokenizer_config":{"tokenizer_file":")") +
                           vocab + R"("}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    ASSERT_TRUE(pipe.TokenizerReady());
    Context ctx;
    ASSERT_TRUE(pipe.CreateContext(&ctx).ok());
    ASSERT_TRUE(pipe.SetContext(ctx.id()).ok());

    { // 65 token > 预算 64：拒绝（INVALID_ARGUMENT + capacity 日志）
        Prompt prompt;
        prompt.text = std::string(65, 'a');
        RequestHandle handle;
        test_util::StderrCapture cap;
        auto s = pipe.GenerateAsync(prompt, SamplingParams{}, RequestOptions{}, &handle,
                                    /*callback=*/{});
        const std::string out = cap.Stop();
        EXPECT_EQ(s.code, Status::INVALID_ARGUMENT);
        EXPECT_NE(out.find("promptcache.capacity"), std::string::npos);
    }
    { // 64 token == 预算边界：严格 '>' 判 ⇒ 放行对比。本例只锁提交闸语义
        // （提交成功 + capacity 日志缺席）。提交后的终态是 stub-era 形态：worker
        //  RunLoop → Scheduler Step 命中 "prefill executor not ready"
        // （scheduler.cpp not-ready 分支，本 binary 无 hiai 工厂覆盖）⇒ req 终态
        //  Failed → Wait 按契约返回 FAILURE + FinishReason::Error（request_handle
        //  cpp Wait Failed 分支）。RunLoop 侧预算数量断言（64+1>=64 即停、
        //  FinishReason=MaxTokens、promptcache.budget 日志）host 真链用例未建
        // （spec:208 验收债务，code_review P2）；现证物 = 真机 V-F4
        // （meta max_io_tokens=230、input 199 + generated 31 >= 230 恰差值终止）。
        Prompt prompt;
        prompt.text = std::string(64, 'a');
        RequestHandle handle;
        test_util::StderrCapture cap;
        auto s = pipe.GenerateAsync(prompt, SamplingParams{}, RequestOptions{}, &handle,
                                    /*callback=*/{});
        const std::string out = cap.Stop();
        EXPECT_TRUE(s.ok()) << "边界 64 <= 64 过闸（提交成功）";
        EXPECT_EQ(out.find("promptcache.capacity"), std::string::npos);
        Result result;
        auto ws = handle.Wait(&result);
        EXPECT_FALSE(ws.ok());
        EXPECT_EQ(ws.code, Status::FAILURE);                  // stub-era Failed 终态
        EXPECT_EQ(result.finish_reason, FinishReason::Error); // 非 MaxTokens：RunLoop 未达
    }
}

// TC_203：prefill_len 空数组 → options 回退
TEST(ConfigParseTest, EmptyPrefillLenFallsBackToOptions_TC203)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"prefill_len":[],
        "kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.default_chunk_size = 512;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 512);
}

// TC_202：数字字符串容错（"28" 与 28 等价）
TEST(ConfigParseTest, NumericStringTolerated_TC202)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"num_hidden_layers":"28","kv_cache_max_len":"4096"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 28);
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 4096);
}

// TC_010：tokenizer_file 词表加载就绪
TEST(ConfigParseTest, TokenizerFileLoadsVocab_TC010)
{
    ModelArtifact a;
    const std::string vocab = WriteTempBpeVocab();
    a.meta_json.data = Buf(std::string(R"({"tokenizer_config":{"tokenizer_file":")") + vocab + R"("}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_TRUE(pipe.TokenizerReady());
    EXPECT_EQ(pipe.TokenizerReady() ? 2u : 0u, 2u); // 极简词表 2 条
}

// TC_107：tokenizer_file 指向不可加载文件 → tokenizer 置空不阻断
TEST(ConfigParseTest, BadTokenizerFileUnsetsTokenizer_TC107)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"tokenizer_config":{"tokenizer_file":"/no/such/vocab.json"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_FALSE(pipe.TokenizerReady());
    // 构造未被阻断（观测点可达即证明）
}

// TC_104：空构件 → 桩默认回归（与变更前现状逐值一致）
TEST(ConfigParseTest, EmptyArtifactStubsRegress_TC104)
{
    AutoregressivePipelineExecutor pipe(ModelArtifact{}, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);  // 桩等效默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8); // 桩等效默认
    EXPECT_EQ(pipe.ParsedModelConfig().batch_num, 0);
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 1024);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 4);
    EXPECT_FALSE(pipe.TokenizerReady());
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty()); // tokenId 路径
}

// TC_101：JSON 语法非法 → 三段全部桩默认
TEST(ConfigParseTest, InvalidJsonDegradesAll_TC101)
{
    ModelArtifact a;
    a.meta_json.data = Buf("{ truncated");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, -1);
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty());
}

// TC_201：单段缺失其余生效
TEST(ConfigParseTest, SingleSectionOnly_TC201)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":8192,"num_hidden_layers":12}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 8192);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, -1);     // 桩默认
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty()); // 桩默认
}

// TC_204：容量字段非法 → model 段桩默认（而非以 0 定容）
TEST(ConfigParseTest, InvalidCapacityDegradesModelSection_TC204)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":0,"num_hidden_layers":28}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
}

// ── KV 形状参数（num_attention_kv_heads / num_attention_head_dims / batch_num /
//    kv_cache_type）解析行为 ──

// 新字段全量解析赋值（含数字字符串容错，口径同 TC_202；kv_cache_type 用 doc 词表 fp16）
TEST(ConfigParseTest, KvShapeParamsParsed)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 1024, "num_hidden_layers": 28,
                         "num_attention_kv_heads": "8",
                         "num_attention_head_dims": 128,
                         "batch_num": 1,
                         "kv_cache_type": "fp16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 8);
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 128);
    EXPECT_EQ(pipe.ParsedModelConfig().batch_num, 1);
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_type, "fp16");
}

// 新字段缺省：kv_heads/head_dims 取桩等效默认 4/8（与旧硬编码现状一致），
// batch_num=0（视同未供给）、kv_cache_type 空（消费点回退 FLOAT16）
TEST(ConfigParseTest, KvShapeParamsAbsentDefault)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8);
    EXPECT_EQ(pipe.ParsedModelConfig().batch_num, 0);
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
}

// 显式 kv_heads ≤ 0 → 段降级（与 kv_cache_max_len ≤ 0 同口径，TC_204 扩展）
TEST(ConfigParseTest, NonPositiveKvHeadsDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "num_attention_kv_heads":0}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);
}

// kv_cache_type 非 string → 段降级（get_str 口径）
TEST(ConfigParseTest, NonStringKvCacheTypeDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": 16}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
}

// batch_num 优先级链（REQ：batch_num > max_batch_size_override > 4）：
// batch_num 在段解析成功时压过 override
TEST(ConfigParseTest, BatchNumBeatsOverride)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "batch_num": 2}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 2);
}

// 链第二级：batch_num 缺省（0）→ override 生效
TEST(ConfigParseTest, OverrideAppliesWhenBatchNumAbsent)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 8);
}

// 链第三级（段降级路径）：batch_num 不可得 → override 仍生效（段降级不吞 override）
TEST(ConfigParseTest, OverrideAppliesWhenModelSectionDegraded)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":0,"batch_num": 2}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 8); // 段降级 → batch_num 不采信
}

// ── kv_cache_type → DataType 解析（消费点行为，经观测点 ParsedKvCacheDtype）──

// 小写词表（仓库惯例，同 embedding_type）正确映射
TEST(ConfigParseTest, KvCacheDtypeLowercaseVocab)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "fp16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedKvCacheDtype(), DataType::FLOAT16);

    ModelArtifact b;
    b.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    EXPECT_EQ(pipe2.ParsedKvCacheDtype(), DataType::FLOAT32);
}

// 大小写不敏感：DataType 枚举名（"FLOAT16"）与小写简写等价
TEST(ConfigParseTest, KvCacheDtypeCaseInsensitive)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "FLOAT16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedKvCacheDtype(), DataType::FLOAT16);
}

// 缺省 = FLOAT16；未知值（无对应 DataType，如 "bf16"）WARN 回退 FLOAT16 不阻断
// （WARN 而非静默：stderr 断言 config.load.kv_cache_type 详因键）
TEST(ConfigParseTest, KvCacheDtypeDefaultAndUnknownFallback)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedKvCacheDtype(), DataType::FLOAT16);

    ModelArtifact b;
    b.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "bf16"}})");
    test_util::StderrCapture cap;
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    const std::string logs = cap.Stop();
    EXPECT_EQ(pipe2.ParsedKvCacheDtype(), DataType::FLOAT16); // WARN 回退
    EXPECT_NE(logs.find("config.load.kv_cache_type"), std::string::npos);
}

// 端到端接线（组合观测，T1 改造后语义）：meta_json KV 形状参数 → 池 HF 期望
// 预计算，构造期**不建** tensor。需求 3 的组合点改锁「构造期零 tensor + 槽位
// 记账就绪」——传值/换位观测由解析观测点分别锁定（LayersParamsParsed 等），
// tensor 实形改由 Load 期 EnsureKvTensor 按图 desc 建（KvPoolEnsureTest 组）。
// host 无 NDK so 也恒成立：该用例从「host 失败基线之一」转为 host 可跑
// （PROJECT.md 基线注记须随改）。
TEST(ConfigParseTest, KvShapeParamsWiredIntoPoolTensors)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 64, "num_hidden_layers": 2,
                         "num_attention_kv_heads": 8, "num_attention_head_dims": 128,
                         "batch_num": 2, "kv_cache_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const KvCachePool* pool = pipe.ParsedKvPool();
    ASSERT_NE(pool, nullptr);
    // T1：构造期无图——池 0 tensor（kv_cache_type 亦不消费，降级纯观测）
    EXPECT_EQ(pool->TensorCount(), 0u);
    EXPECT_FALSE(pool->KTensor(0));
    EXPECT_FALSE(pool->VTensor(1));
    // 层表/容量记账经 meta 传达（Authority：构造期权威量）
    EXPECT_EQ(pool->NumKvLayers(), 2);
    EXPECT_EQ(pool->kv_cache_max_len(), 64);
    EXPECT_EQ(pool->max_batch_size(), 2);
    // 槽位记账就绪（lazy-create 不影响池对象可用性）
    std::vector<SlotInfo> snap;
    EXPECT_TRUE(const_cast<KvCachePool*>(pool)->Snapshot(&snap).ok());
    EXPECT_EQ(snap.size(), 2u);
}

// 显式 head_dims ≤ 0 → 段降级（与 kv_heads ≤ 0 同一校验，独立字段）
TEST(ConfigParseTest, NonPositiveHeadDimsDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "num_attention_head_dims":0}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8);
}

// batch_num 显式负值（段解析成功但 ≤ 0）→ 链不采信，回退 override
TEST(ConfigParseTest, NegativeBatchNumFallsThroughChain)
{
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "batch_num": -2}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 8);
}

// ── Qwen3.5 layer_types（support_qwen3.5.md §4.1/§4.2 解析；§5 D5/D6 决策；
//    §7 解析 UT 行）。接线观测走 ParsedKvPool（NumKvLayers/KvLayerIds 稀疏层表）
//    与 ParsedStatePool（构造期创建判定）——host 无 NDK so 时池 tensor 申请失败，
//    故不断言 tensor 实体（同 KvShapeParamsWiredIntoPoolTensors 的环境前提）──

// 合法混合层表：解析进 ParsedModelConfig + 双池稀疏装配（N_full=2 / N_linear=2，
// 全局层号映射）
TEST(ConfigParseTest, LayerTypesMixedParsedAndPoolsWired)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4,
                         "layer_types": ["full_attention", "linear_attention",
                                        "full_attention", "linear_attention"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& lt = pipe.ParsedModelConfig().layer_types;
    ASSERT_EQ(lt.size(), 4u);
    EXPECT_EQ(lt[0], "full_attention");
    EXPECT_EQ(lt[1], "linear_attention");
    EXPECT_EQ(lt[2], "full_attention");
    EXPECT_EQ(lt[3], "linear_attention");

    // KV 池稀疏化：只对 full 层（0,2）建工位
    const KvCachePool* kv = pipe.ParsedKvPool();
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->NumKvLayers(), 2);
    EXPECT_EQ(kv->KvLayerIds(), (std::vector<int>{0, 2}));

    // 状态池按 linear 层数创建
    const LinearStatePool* sp = pipe.ParsedStatePool();
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(sp->NumLinearLayers(), 2);
    EXPECT_EQ(sp->LinearLayerIds(), (std::vector<int>{1, 3}));
}

// 层表声明但全 full_attention → KV 行为与稠密一致；状态池不创建
// （有 linear 层才建池，避免无谓空池对象）
TEST(ConfigParseTest, LayerTypesAllFullKeepsDenseKvAndNoStatePool)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 3,
                         "layer_types": ["full_attention", "full_attention",
                                        "full_attention"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const KvCachePool* kv = pipe.ParsedKvPool();
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->NumKvLayers(), 3);
    EXPECT_EQ(kv->KvLayerIds(), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 键缺省（旧模型 / 非 Qwen3.5）→ 空表 + 稠密 KV + 无状态池（D6 零行为变化）
TEST(ConfigParseTest, LayerTypesAbsentKeepsLegacyBehavior)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_TRUE(pipe.ParsedModelConfig().layer_types.empty());
    const KvCachePool* kv = pipe.ParsedKvPool();
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->NumKvLayers(), 4);
    EXPECT_EQ(kv->KvLayerIds(), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 元素个数 ≠ num_hidden_layers → 段整体降级（D5：层表残缺不采信，
// TC_204 同口径桩默认）
TEST(ConfigParseTest, LayerTypesLengthMismatchDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4,
                         "layer_types": ["full_attention", "linear_attention"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_TRUE(pipe.ParsedModelConfig().layer_types.empty()); // 残缺表不采信
    const KvCachePool* kv = pipe.ParsedKvPool();
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->NumKvLayers(), 2); // 桩 nhl=2 稠密
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 非法取值（不在 full_attention|linear_attention 词表）→ 段降级
TEST(ConfigParseTest, LayerTypesInvalidValueDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 2,
                         "layer_types": ["full_attention", "sliding_window"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 非数组（字符串/对象）→ 段降级
TEST(ConfigParseTest, LayerTypesNonArrayDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 1,
                         "layer_types": "full_attention"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_TRUE(pipe.ParsedModelConfig().layer_types.empty());
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 元素非字符串（数字）→ 段降级
TEST(ConfigParseTest, LayerTypesNonStringElementDegradesModelSection)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 2,
                         "layer_types": ["full_attention", 3]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16); // 桩默认
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// TC_206：未知段/未知字段静默忽略，已知字段不受影响
TEST(ConfigParseTest, UnknownSectionsAndFieldsIgnored_TC206)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "unknown_section": {"x": 1},
        "model_config": {"kv_cache_max_len": 256, "num_hidden_layers": 3,
                         "future_field": 99}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 256);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 3);
}

// TC_205：meta_json 双态双给 → buffer 优先（path 忽略）
TEST(ConfigParseTest, BufferWinsOverPath_TC205)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"kv_cache_max_len":1024,"num_hidden_layers":6}})");
    a.meta_json.path = "/no/such/meta.json"; // 不存在——若被采用则全桩
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 1024);
}

// TC_211：图位 buffer+path 双给 → 图按 buffer 加载，meta_json 仍按 path 派生（解耦）
TEST(ConfigParseTest, DeriveFromGraphPathField_TC211)
{
    static int seq = 0;
    const std::string meta = "/tmp/ut_cfg_derive_" + std::to_string(++seq) + ".json";
    {
        std::ofstream f(meta);
        f << R"({"model_config":{"kv_cache_max_len":777,"num_hidden_layers":5}})";
    }
    ModelArtifact a;
    static char omc[8] = {0};
    a.prefill_omc.data = Buffer{omc, sizeof(omc)}; // buffer 生效（加载侧）
    a.prefill_omc.path = meta;                     // 派生源按 path 字段（D-17）
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 777); // 派生生效
}

// ── EngineOptions tokenizer 通路（tokenizer_type / tokenizer_path）──────────────

// TC_212：options.tokenizer_path 显式供给词表（无 meta_json 也可就绪）
TEST(ConfigParseTest, OptionsTokenizerPathLoads_TC212)
{
    ModelArtifact a; // 无 meta_json
    EngineOptions opts;
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());
}

// TC_213：options.tokenizer_path 优先于 meta_json tokenizer_config.tokenizer_file
TEST(ConfigParseTest, OptionsPathOverridesMetaTokenizerFile_TC213)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"tokenizer_config":{"tokenizer_file":"/no/such/vocab.json"}})");
    EngineOptions opts;
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady()); // meta 指向坏路径，options 覆盖后仍就绪
}

// TC_214：options.tokenizer_type 选算法——Qwen 收下 BPE 拒收的产物（model.type 非 BPE）
TEST(ConfigParseTest, OptionsTokenizerTypeSelectsQwen_TC214)
{
    ModelArtifact a;
    const std::string vocab = WriteTempNonBpeVocab();
    EngineOptions opts;
    opts.tokenizer_type = "Qwen";
    opts.tokenizer_path = vocab;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());
    // 反证：默认（BPE）时同产物被 model.type 校验拒收 → not-ready，
    // 证明上行的就绪确来自 Qwen 算法选择而非 BPE 兜底
    EngineOptions bpe_opts;
    bpe_opts.tokenizer_path = vocab;
    AutoregressivePipelineExecutor bpe_pipe(a, bpe_opts);
    EXPECT_FALSE(bpe_pipe.TokenizerReady());
}

// TC_215：未知 tokenizer_type → WARN + BPE 回退（BPE 型产物仍可加载）
TEST(ConfigParseTest, UnknownTokenizerTypeFallsBackToBpe_TC215)
{
    ModelArtifact a;
    EngineOptions opts;
    opts.tokenizer_type = "XYZ";
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());
}

// TC_216：tokenizer_config 段 add_bos / add_eos 解析（B2 通道来源接通）
TEST(ConfigParseTest, AddBosAddEosParsed_TC216)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"tokenizer_config":{"add_bos":true,"add_eos":true}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_TRUE(pipe.ParsedTokenizerConfig().add_bos);
    EXPECT_TRUE(pipe.ParsedTokenizerConfig().add_eos);

    ModelArtifact b; // 缺省 = 零注入（false）
    b.meta_json.data = Buf(R"({"tokenizer_config":{}})");
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    EXPECT_FALSE(pipe2.ParsedTokenizerConfig().add_bos);
    EXPECT_FALSE(pipe2.ParsedTokenizerConfig().add_eos);
}

// ── gemma4 per_layer_token_identity meta_json 键（docs/gemma4_token_identity.md §4/§7.2；
//    单文件四段布局：仅余两个读取类型键；embedding_dequant_scale 键已废弃）──

// TC_217：embedding_config 段 identity 两类型键全解析（显式 int8/fp32）；
// 主表键齐备 → 主表与 identity 读取类型并存（单文件四段布局的完整配置形态）
TEST(ConfigParseTest, TokenIdentityKeysParsed_TC217)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "embedding_scale_type": "fp16",
                             "token_identity_type": "int8",
                             "token_identity_scale_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Int8);
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(ec.token_identity_row_width, 0u); // 加载期自工位回填，非 meta 来源
    EXPECT_EQ(ec.table_path, "t.bin");          // 主表照常（identity 与主表同一文件）
    EXPECT_EQ(ec.vocab_size, 153376u);          // vocab/hidden 跨段
    EXPECT_EQ(ec.hidden_dim, 2048u);
}

// TC_218：identity 键未配置 → 类型缺省（fp16/fp16——Q10/Q8）且主表/跨段不受影响
TEST(ConfigParseTest, TokenIdentityUnconfiguredDefaults_TC218)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Fp16);            // 缺省（Q10）
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16); // 缺省（Q8）
    EXPECT_EQ(ec.token_identity_row_width, 0u);
    EXPECT_EQ(ec.table_path, "t.bin"); // 主表字段不受 identity 缺席影响
}

// TC_219：identity 键枚举非法（type "fp8" / scale_type "none"）→ embedding 段
// 整体降级，不波及 model 段
TEST(ConfigParseTest, TokenIdentityBadTypeDegradesSectionOnly_TC219)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "token_identity_type": "fp8"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    // 段降级 = EmbeddingConfig{} 主 stub：主表与 identity 全部回落（tokenId 形态）
    EXPECT_TRUE(ec.table_path.empty());
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Fp16);
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16);
    EXPECT_EQ(ec.vocab_size, 0u); // 跨段值一并回落
    // 段间隔离：model 段不受波及（F-02 口径）
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 512);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 4);

    // scale_type "none" 非法（Q12："none" 不再表达"不启用"——启用改由图工位
    // 判定，键仅表达读取类型）→ 同一降级口径
    ModelArtifact b;
    b.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "token_identity_scale_type": "none"}})");
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    EXPECT_TRUE(pipe2.ParsedEmbeddingConfig().table_path.empty());
}

// TC_220：主表不启用（weights 空 = TokenId 部署形态）+ identity 类型键独立解析
// （配置层独立——Q12 后 identity 数据与主表同文件，是否读取由图工位判定；
// host 加载/查表仅服务 Embedding 形态，由 pipeline Loaded() 门控——Q11）
TEST(ConfigParseTest, TokenIdentityIndependentOfMainTable_TC220)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"token_identity_type": "int8",
                             "token_identity_scale_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_TRUE(ec.table_path.empty());                     // 主表未配置（在图权重中）
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Int8); // identity 类型照常解析
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(ec.vocab_size, 153376u); // 跨段填充不受主表缺席影响
}

// ── identity scale 类型键（2026-10-19 移除 Q13 双拼兼容：仅规范拼写
//    token_identity_scale_type 生效，Q1/Q8 契约拼写遗留键不再兜底读取；
//    docs/gemma4_token_identity.md §4、pipeline_executor.cpp ParseEmbeddingSection）──

// TC_221：规范键单独在场 → 正常解析；主表/跨段照常，日志无废弃拼写残留
TEST(ConfigParseTest, IdentityScaleRegularKeyAloneParses_TC221)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "token_identity_type": "int8",
                             "token_identity_scale_type": "fp32"}})");
    test_util::StderrCapture cap; // 捕获构造期日志（唯一键直取，预期读值无歧义）
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const std::string logs = cap.Stop();
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp32); // 规范键生效
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Int8);            // 邻键不受影响
    EXPECT_EQ(ec.table_path, "t.bin");                                 // 主表照常
    EXPECT_EQ(ec.vocab_size, 153376u);                                 // 跨段照常
    EXPECT_NE(logs.find("identity_scale_type  = Fp32"), std::string::npos); // 解析观测（LogParsedConfigs 枚举名口径）
}

// TC_223：规范键类型不符（数字值）→ 直接段降级（embedding 段整体回落
// EmbeddingConfig{} 桩——table_path 空、vocab 跨段值一并回落；model 段不受波及，
// 段间隔离口径同 TC_219）
TEST(ConfigParseTest, IdentityScaleRegularKeyBadTypeDegradesSectionOnly_TC223)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "token_identity_scale_type": 16}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_TRUE(ec.table_path.empty());                                // 段降级 = EmbeddingConfig{} 主桩
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16); // 桩缺省
    EXPECT_EQ(ec.vocab_size, 0u);                                      // 跨段值一并回落
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 512);         // 段间隔离不波及 model
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 4);
}

// TC_224：旧拼写键（Q1/Q8 契约拼写）已成未知键 → 静默忽略（meta 多余键忽略
// 口径），identity scale 走缺省 fp16，embedding 段不降级、model 段不受波及——
// 2026-10-19 兼容兜底移除的回归锚点（存量文件须迁移至规范键，误用不炸但读不到值）
TEST(ConfigParseTest, IdentityScaleLegacyKeyIgnoredNotParsed_TC224)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_type": "fp16",
                             "token_identity_sacle_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.table_path, "t.bin"); // 段不降级（未知键忽略，非类型错误）
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16); // 旧键不生效 → 缺省
    EXPECT_EQ(ec.vocab_size, 153376u);                                 // 跨段照常
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 512);         // model 段隔离
}

// TC_225：embedding_dequant_scale 残留键（老双文件 meta）→ WARN 可见但不降段：
// 解析照常成功、table_path 不受影响（老产物部署须重导出，警告仅诊断）
TEST(ConfigParseTest, StaleDequantScaleKeyWarnsOnly_TC225)
{
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
                             "embedding_type": "fp16",
                             "embedding_scale_type": "fp16"}})");
    test_util::StderrCapture cap; // 捕获构造期 WARN 日志
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const std::string logs = cap.Stop();
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.table_path, "t.bin");                                  // 段不降级（残留键忽略，仅告警）
    EXPECT_EQ(ec.vocab_size, 153376u);                                  // 跨段照常
    EXPECT_NE(logs.find("embedding_dequant_scale"), std::string::npos); // WARN 可见
}

// ═══════════════════════════════════════════════════════════════════════════
// 问题 3 方案 b（2026-10-10，10101015_debug.log）：om 图 KV 输入工位 HF 命名族
// past_key_values_<层>_<0|1>（末位 0=key / 1=value）引擎侧兼容。
// 载体：model_executor.h 原地更新族命名常量 + 解析谓词（inline，经
// pipeline_executor.h 传递包含）。纯字符串逻辑，无后端/tensor 依赖——
// host 无 NDK stub so 亦可跑（区别于依赖后端在场/优雅降级形态的 backend 类用例）。
// 集成缝隙说明：CorrectTensorDescs 输入侧跳过与末输出保护守卫位于 .cpp 匿名
// 命名空间且依赖后端模型 Load，host UT 不可直测；守卫判定谓词即下列
// IsInPlaceUpdateSlotName / ParseKvInputLayer，语义经本套真值表用例锁定。
// TC_301 起接续本文件 TC 编号序列。
// ═══════════════════════════════════════════════════════════════════════════

// TC_301：HF 命名族 key 工位（末位 0）——层号解析 + key 判定；
// 含多位层号与前导零（atoi 口径：007 = 7）。is_value 先置反值再调，
// 证明 false 系函数写入而非初值侥幸
TEST(ParseKvInputLayerTest, HfKeySuffixParsesLayerAndKeyKind_TC301)
{
    bool is_value = true;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_0", &is_value), 0);
    EXPECT_FALSE(is_value);
    is_value = true;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_23_0", &is_value), 23);
    EXPECT_FALSE(is_value);
    is_value = true;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_007_0", &is_value), 7);
    EXPECT_FALSE(is_value);
}

// TC_302：HF 命名族 value 工位（末位 1）——层号解析 + value 判定
TEST(ParseKvInputLayerTest, HfValueSuffixParsesLayerAndValueKind_TC302)
{
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_1", &is_value), 0);
    EXPECT_TRUE(is_value);
    is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_23_1", &is_value), 23);
    EXPECT_TRUE(is_value);
}

// TC_303：HF 命名族畸形名 → -2（命中但后缀非法）。覆盖：裸前缀（后缀
// rfind npos）/缺末位段/末位非 0|1/层号非纯数字/末位段非数字/空层号双下
// 划线（sep==0）。-2 口径不承诺返回后 is_value 值，故不断言输出参数
TEST(ParseKvInputLayerTest, HfMalformedNameReturnsInvalidLayerCode_TC303)
{
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_2", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_ab_1", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_ab", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values__0", &is_value), -2);
}

// TC_304：旧约定族不回归（past_key_in<层> → key；past_value_in<层> → value）
TEST(ParseKvInputLayerTest, LegacyPrefixParsesKeyAndValueKinds_TC304)
{
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_in3", &is_value), 3);
    EXPECT_FALSE(is_value);
    is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_value_in11", &is_value), 11);
    EXPECT_TRUE(is_value);
}

// TC_305：旧约定族畸形 → -2（裸前缀空后缀；层号尾非法字符）
TEST(ParseKvInputLayerTest, LegacyMalformedSuffixReturnsInvalidLayerCode_TC305)
{
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_in", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_in3x", &is_value), -2);
}

// TC_306：非该族（常规输入工位/空名）→ -1（常规工位不占用解析口径）
TEST(ParseKvInputLayerTest, NonFamilyNameReturnsNotFoundLayerCode_TC306)
{
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("embeddings", &is_value), -1);
    EXPECT_EQ(ParseKvInputLayer("position_ids", &is_value), -1);
    EXPECT_EQ(ParseKvInputLayer("", &is_value), -1);
}

// TC_307：conv 状态工位前缀 → kind 0（kind 先置反值再调，证明写入）
TEST(ParseStateInputLayerTest, ConvStatePrefixParsesZeroKind_TC307)
{
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_5", &kind), 5);
    EXPECT_EQ(kind, 0);
}

// TC_308：rstate 状态工位前缀 → kind 1
TEST(ParseStateInputLayerTest, RStatePrefixParsesOneKind_TC308)
{
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("rstate_in_layer_7", &kind), 7);
    EXPECT_EQ(kind, 1);
}

// TC_309：状态族畸形 → -2（裸前缀空后缀；层号尾非法字符）
TEST(ParseStateInputLayerTest, MalformedStateSuffixReturnsInvalidLayerCode_TC309)
{
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_", &kind), -2);
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_5x", &kind), -2);
}

// TC_310：非状态族 → -1（常规工位；KV 旧族名亦非状态族——两族不串扰）
TEST(ParseStateInputLayerTest, NonStateFamilyNameReturnsNotFoundLayerCode_TC310)
{
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("embeddings", &kind), -1);
    EXPECT_EQ(ParseStateInputLayer("past_key_in0", &kind), -1);
}

// TC_311：五前缀全命中（KV 旧约定族 ×2 + HF 命名族 + conv/rstate 状态族）
TEST(IsInPlaceUpdateSlotNameTest, MatchesAllFiveFamilyPrefixes_TC311)
{
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_key_in0"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_value_in1"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_key_values_2_1"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("conv_state_in_layer_3"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("rstate_in_layer_4"));
}

// TC_312：非工位名不命中——HF 元组集体名（无尾下划线 ≠ 工位前缀）、
// 前缀中缀（find==0 位置判定）、常规工位、空串
TEST(IsInPlaceUpdateSlotNameTest, RejectsNonFamilyAndMidStringPrefix_TC312)
{
    EXPECT_FALSE(IsInPlaceUpdateSlotName("past_key_values"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName("npast_key_in0"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName("embeddings"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName(""));
}

// ═══════════════════════════════════════════════════════════════════════════
// SlidingLayerWidth / PrefillMaxLen（sliding_kv_seq_dim_rebind S2/S4 单源
// 助手，U3 边界锁）：新契（sw_head_dim>0）滑窗层序列容量 C = W + P
// （mask_sliding 列宽、池期望 seq_dim、keep_cap 同源）。
// CorrectTensorDescs（mask_sliding 修形）+ InputAssembler（填充）+ pipeline
// （PromptCache 绕回门 keep_cap）三消费点单源。旧契不经本助手（消费点直用
// kv_max）；b688221 的 SlidingMaskWidth 口径整体废弃（D8，过渡图从未产出）。
// ═══════════════════════════════════════════════════════════════════════════

// U3：C = W + P 仅在两入参全正时成立；任一非正（0/负——D10 已挡空组合，
// 此处回退仅防御）→ 0。gemma4 实形 W=256、P=2048 → C=2304。
TEST(SlidingLayerWidthTest, WidthIsWindowPlusPrefillMax_U3)
{
    EXPECT_EQ(SlidingLayerWidth(256, 2048), 2304u) << "gemma4 实形 C=W+P";
    EXPECT_EQ(SlidingLayerWidth(1, 1), 2u) << "最小 W=1/P=1 边界";
    EXPECT_EQ(SlidingLayerWidth(0, 2048), 0u) << "W=0 防御回 0";
    EXPECT_EQ(SlidingLayerWidth(256, 0), 0u) << "P=0 防御回 0";
    EXPECT_EQ(SlidingLayerWidth(-5, 2048), 0u) << "负 W 按 0";
    EXPECT_EQ(SlidingLayerWidth(256, -1), 0u) << "负 P 按 0";
    EXPECT_EQ(PrefillMaxLen(std::vector<int>{1024, 2048, 512}), 2048) << "P = 数组最大档";
    EXPECT_EQ(PrefillMaxLen(std::vector<int>{2048}), 2048) << "单档位直取";
    EXPECT_EQ(PrefillMaxLen(std::vector<int>{}), 0) << "空数组防御回 0";
}

// ═══════════════════════════════════════════════════════════════════════════
// SlidingHeadDimComboValid（D10 越界组合校验单源，ST-4 抽取后直测判据本身；
// meta 级端到端面见 ConfigParseTest.SwHeadDim* 两例，池期望面见 TC_613/614）。
// 接受面 = 旧契（sw_hd=0 无组合约束）+ 新契合法（sw>0 ∧ prefill 非空全正，
// C=W+P 可算）；拒绝面 = 字段自检负值 + D10 三元组（sw=0 / prefill 空 /
// prefill 含非正档）。
// ═══════════════════════════════════════════════════════════════════════════

TEST(SlidingHeadDimComboTest, GateAcceptsOldContractAndRejectsD10Tuples)
{
    // 接受面——旧契（sw_hd=0）：无组合约束（sw / prefill 任何形态均直落）
    EXPECT_TRUE(SlidingHeadDimComboValid(0, 0, {}));
    EXPECT_TRUE(SlidingHeadDimComboValid(256, 0, {}));
    EXPECT_TRUE(SlidingHeadDimComboValid(0, 0, {1024}));
    // 接受面——新契合法：sw>0 且 prefill 档非空全正
    EXPECT_TRUE(SlidingHeadDimComboValid(256, 128, {1024, 2048}));
    EXPECT_TRUE(SlidingHeadDimComboValid(4, 64, {8}));
    // 拒绝面——字段自检：sw / sw_hd 任一负值即拒（段降级前置判据）
    EXPECT_FALSE(SlidingHeadDimComboValid(-1, 0, {}));
    EXPECT_FALSE(SlidingHeadDimComboValid(0, -1, {1024}));
    // 拒绝面——D10 三元组（sw_hd>0 搭配）：sw=0 / prefill 空 / 含非正档
    EXPECT_FALSE(SlidingHeadDimComboValid(0, 64, {1024}));
    EXPECT_FALSE(SlidingHeadDimComboValid(256, 64, {}));
    EXPECT_FALSE(SlidingHeadDimComboValid(256, 64, {0}));
    EXPECT_FALSE(SlidingHeadDimComboValid(256, 64, {2048, -1}));
}

// ═══════════════════════════════════════════════════════════════════════════
// Cache_pool_simplify.md（2026-10-15 定稿实施）：KV tensor 按 om 图工位
// desc 惰性直建，KV 池与 LinearStatePool 统一 verify-or-create 语义。本组
// 锁定 KvCachePool::EnsureKvTensor：
//   R1 未建 → 按图 desc 直建（dims/dtype/format 直通，om 权威；命名
//      kv_cache_{k,v}_<layer> 不变；创建即整只清零——2026-10-15 裁定）
//   R2 已建 → 次图 dims+dtype 逐值全等复用；失配 INVALID_ARGUMENT（双图
//      同层 KV 布局须一致，判据比旧「元素总数同」更严）
//   R3 HF 期望序 [batch,heads,len,dim] 逐位比对（只认 HF 序，不兼容旧约定序
//      [len,heads,batch,dim]——2026-10-15 裁定）：kv_len（dims[2]）/ batch
//      （dims[0]）失配 → INVALID_ARGUMENT 致命终止（host 光标/槽位记账与图
//      强关联）；heads（dims[1]）/末维（dims[3]）失配及 dims≠4 → LOGW 告警
//      （每池同类去重一次）后 SUCCESS 按图直建
//   T1 构造期 0 tensor，TensorCount 数非空项
// 环境前提：tensor 申请经 TensorManager → host 需 NDK/stub so
// （LD_LIBRARY_PATH=/tmp/llm_ut_host）；触碰 tensor 实体的新建路径用例以
// HostBackendAvailable() 探测，无 so 时 GTEST_SKIP（纯 arg/记账用例无条件
// 跑）。池小形状：batch=2 / kv_len=16 / 2 层 / kv_heads=1 / head_dim=8 →
// HF 序期望 {2,1,16,8}（fp16 512 字节）。Load 期接线（model_executor.cpp
// EnsureKvTensor）需真 om 图，host 不可直测 e2e。TC_607~609 =
// TensorManager 通道组（本批不动）。TC 编号沿用 601~612。
// ═══════════════════════════════════════════════════════════════════════════

namespace {
/// host 可用性探测（同 test_input_assembler.cpp 口径）：无 NDK/stub so →
/// 新建路径类用例 GTEST_SKIP。
bool HostBackendAvailable()
{
    TensorDesc d;
    d.dims = {1};
    d.dataType = DataType::INT32;
    std::shared_ptr<ITensorBuffer> t;
    return TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t).ok();
}
/// 图 KV 工位 desc 构造器：dims/dtype/format 直通图（om 权威），
/// tensorName 任意（Ensure 内部覆写为池命名约定）。
TensorDesc MakeGraphKvDesc(const std::vector<int32_t>& dims, DataType dt = DataType::FLOAT16)
{
    TensorDesc d;
    d.dims = dims;
    d.dataType = dt;
    d.format = Format::ND;
    d.tensorName = "past_key_in_placeholder";
    return d;
}
} // namespace

// TC_601：T1 构造期 0 tensor + arg 防御——layer<0 / dims 空 / 任一维 ≤ 0 /
// dtype RESERVED → 全部 INVALID_ARGUMENT，拒绝不建不落位（无需后端）
TEST(KvPoolEnsureTest, InvalidDescRejectedAndNoTensorCreated_TC601)
{
    KvCachePool pool(2, 16, 2, 1, 8);
    ASSERT_EQ(pool.NumKvLayers(), 2);
    EXPECT_EQ(pool.TensorCount(), 0u); // T1：构造期 0 tensor
    EXPECT_FALSE(pool.KTensor(0));

    EXPECT_EQ(pool.EnsureKvTensor(-1, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({})).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 0, 16, 8})).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, -1, 16, 8})).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 8}, DataType::RESERVED)).code,
              Status::INVALID_ARGUMENT);

    EXPECT_EQ(pool.TensorCount(), 0u); // 拒绝不建
    EXPECT_FALSE(pool.KTensor(0));
}

// TC_602：kv_cache_max_len=0 形态（旧链路构造期申请失败的退化位）——层表/
// 槽位记账照常；arg 防御先于层表判定；环形化改造后语义（spec REQ-9 /
// design §4.3）：Advance 恒 SUCCESS（越 kv_max 绕回红线已作废），容量 0 ⇒
// MapSlot 塌零（total_written 照记、物理槽恒 0）；容量塌零防御在
// AssemblePrefill/AssembleDecode 侧拦（FAILURE），池侧记账不拦。
TEST(KvPoolEnsureTest, ZeroKvLenPoolKeepsBookkeeping_TC602)
{
    KvCachePool pool(2, 0, 2, 1, 8);
    ASSERT_EQ(pool.NumKvLayers(), 2);
    ASSERT_EQ(pool.KvLayerIds(), (std::vector<int>{0, 1}));
    EXPECT_EQ(pool.TensorCount(), 0u);
    EXPECT_EQ(pool.EnsureKvTensor(-1, true, MakeGraphKvDesc({2, 1, 16, 8})).code,
              Status::INVALID_ARGUMENT); // arg 防御先于层表
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS); // 记账可用
    // 记账守恒（环形语义）：Advance 恒 SUCCESS，total_written 单调照记；容量 0 ⇒
    // NextPosition = MapSlot 恒 0（物理槽塌零，不虚放物理空间）
    EXPECT_EQ(pool.Advance(slot, 3).code, Status::SUCCESS);
    EXPECT_EQ(pool.TotalWritten(slot), 3u);
    EXPECT_EQ(pool.NextPosition(slot), 0u);
    EXPECT_EQ(pool.FreeKvSlot(slot).code, Status::SUCCESS);
}

// TC_603：R1 主路径——未建层首调直建：dims/dtype/format 直通图、命名约定
// kv_cache_{k,v}_<layer>、容量实形、**创建即清零**（memset 全 buffer）、
// K/V 各自独立 Ensure 落位（TensorCount 逐个递增，K1 建不动 V0/V1）
TEST(KvPoolEnsureTest, FirstEnsureCreatesClearedTensorPerGraphDesc_TC603)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 2, 1, 8);
    const std::vector<int32_t> hf_dims{2, 1, 16, 8};
    ASSERT_EQ(pool.EnsureKvTensor(0, /*is_value=*/false, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 1u); // K0 就位
    ASSERT_EQ(pool.EnsureKvTensor(0, /*is_value=*/true, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 2u); // V0 就位

    auto k0 = pool.KTensor(0);
    auto v0 = pool.VTensor(0);
    ASSERT_TRUE(k0);
    ASSERT_TRUE(v0);
    const auto& d = k0->GetTensorDesc();
    EXPECT_EQ(d.dims, hf_dims); // HF 序图 dims 直通（om 权威）
    EXPECT_EQ(d.dataType, DataType::FLOAT16);
    EXPECT_EQ(d.format, Format::ND);
    EXPECT_EQ(d.tensorName, "kv_cache_k_0"); // 命名约定不变
    EXPECT_EQ(v0->GetTensorDesc().tensorName, "kv_cache_v_0");
    EXPECT_EQ(k0->GetSize(), 512u); // 2*1*16*8 × fp16 2B
    EXPECT_NE(k0.get(), v0.get());  // K/V 独立 buffer
    // 创建即清零：全 buffer memset 为 0（2026-10-15 裁定）
    const auto* bytes = static_cast<const uint8_t*>(k0->GetData());
    std::size_t non_zero = 0;
    for (std::size_t i = 0; i < k0->GetSize(); ++i) {
        if (bytes[i] != 0)
            ++non_zero;
    }
    EXPECT_EQ(non_zero, 0u);

    // K1 独立 Ensure：就位不动他层
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 3u);
    EXPECT_FALSE(pool.VTensor(1)); // V1 未 Ensure
}

// TC_604：R2 复验语义——已建层同形（dims+dtype 全等）复用同句柄；次图失配
// （逐值不等/异 dtype）→ INVALID_ARGUMENT（双图同层 KV 布局一致性把关，
// 判据 = 逐值全等，比旧「元素总数同」更严——旧机制允许轴序互换重建）
TEST(KvPoolEnsureTest, SecondGraphMustMatchExactlyOrRejected_TC604)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 2, 1, 8);
    const std::vector<int32_t> hf_dims{2, 1, 16, 8};
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    const auto first = pool.KTensor(0);
    ASSERT_TRUE(first);

    // 同形复用（次图）：SUCCESS、同句柄、不重建
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(0).get(), first.get());
    EXPECT_EQ(pool.TensorCount(), 1u);

    // 次图 dims 逐值不等（末维切实 16）→ R3 末维属告警档，但既有 K0 已按
    // {2,1,16,8} 建成——R2 判逐值全等 → 拒（双图一致性优先于告警放行）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 16})).code, Status::INVALID_ARGUMENT);
    // 次图 dtype 失配 → 拒
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc(hf_dims, DataType::FLOAT32)).code,
              Status::INVALID_ARGUMENT);
    // 拒绝无副作用：原句柄/实形原样
    EXPECT_EQ(pool.KTensor(0).get(), first.get());
    EXPECT_EQ(first->GetTensorDesc().dims, hf_dims);
    EXPECT_EQ(first->GetTensorDesc().dataType, DataType::FLOAT16);
    // 他层不受影响
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc(hf_dims)).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 2u);
}

// TC_605：R3a 致命维——kv_len（dims[2]）或 batch（dims[0]）与期望不等 →
// INVALID_ARGUMENT（Load 终止）；不建不落位。旧约定序图 [len,heads,batch,dim]
// 同被此判据拦下（batch/kv_len 位值互换后必失配）——不兼容旧约定序属预期
// （2026-10-15 裁定）
TEST(KvPoolEnsureTest, KvLenOrBatchMismatchIsFatal_TC605)
{
    KvCachePool pool(2, 16, 2, 1, 8);
    // kv_len 失配（期望 16，图 8）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 8, 8})).code, Status::INVALID_ARGUMENT);
    // kv_len 失配（图 32）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 32, 8})).code, Status::INVALID_ARGUMENT);
    // batch 失配（期望 2，图 4）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({4, 1, 16, 8})).code, Status::INVALID_ARGUMENT);
    // batch 失配（图 1）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({1, 1, 16, 8})).code, Status::INVALID_ARGUMENT);
    // 致命拒绝不建不落位
    EXPECT_EQ(pool.TensorCount(), 0u);
    EXPECT_FALSE(pool.KTensor(0));
    // 旧约定序图 [len,heads,batch,dim] = {16,1,2,8}：kv_len 位（dims[2]）值
    // =2 ≠ 期望 16 → 同判据拦下（不兼容旧约定序，2026-10-15 裁定的预期面）
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({16, 1, 2, 8})).code, Status::INVALID_ARGUMENT);
}

// TC_606：R3b 告警维——heads（dims[1]）/末维（dims[3]）失配 → SUCCESS +
// 按图直建（om 为形状权威，LOGW 每池同类去重——日志不可断言，去重的可测
// 面见 TC_611）；同一池先致命拒后，非致命维仍放行（分档互不污染）
TEST(KvPoolEnsureTest, NonFatalDimsWarnAndCreatePerGraph_TC606)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 2, 1, 8);
    // heads 失配（期望 1，图 2）：告警 + SUCCESS + 按图建
    const std::vector<int32_t> heads_off{2, 2, 16, 8};
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc(heads_off)).code, Status::SUCCESS);
    auto k0 = pool.KTensor(0);
    ASSERT_TRUE(k0);
    EXPECT_EQ(k0->GetTensorDesc().dims, heads_off); // 按图建，不改形
    // 末维失配（期望 8，图 32）：同口径（滑窗期望口径 or 图导出差异皆此档）
    const std::vector<int32_t> end_off{2, 1, 16, 32};
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc(end_off)).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, end_off);
    // heads+末维同时失配：仍 SUCCESS（一条告警文案带全维）
    const std::vector<int32_t> both_off{2, 3, 16, 64};
    ASSERT_EQ(pool.EnsureKvTensor(1, true, MakeGraphKvDesc(both_off)).code, Status::SUCCESS);
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, both_off);
    EXPECT_EQ(pool.TensorCount(), 3u);
    // 分档互不污染：致命维拒后（kv_len 失配），非致命告警档已建 tensor 原样
    EXPECT_EQ(pool.EnsureKvTensor(0, true, MakeGraphKvDesc({2, 2, 8, 8})).code,
              Status::INVALID_ARGUMENT); // V0 kv_len 失配（图 8≠16）→ 致命
    EXPECT_FALSE(pool.VTensor(0));
    EXPECT_EQ(pool.TensorCount(), 3u); // 拒绝不新增
}

// TC_607：层不在池 → FAILURE（资源类：linear 层带 KV 工位/越界层；Load 侧
// 层型校验在前，此为池内防御）
TEST(KvPoolEnsureTest, LayerNotInPoolReturnsFailure_TC607)
{
    KvCachePool pool(2, 16, 3, 1, 8, {"full_attention", "linear_attention", "full_attention"});
    ASSERT_EQ(pool.NumKvLayers(), 2);
    EXPECT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 1, 16, 8})).code,
              Status::FAILURE); // linear 层不在池
    EXPECT_EQ(pool.EnsureKvTensor(9, false, MakeGraphKvDesc({2, 1, 16, 8})).code,
              Status::FAILURE); // 越界层
    EXPECT_EQ(pool.EnsureKvTensor(-1, false, MakeGraphKvDesc({2, 1, 16, 8})).code,
              Status::INVALID_ARGUMENT); // arg 防御先于层表
    EXPECT_EQ(pool.TensorCount(), 0u);
}

// TC_610：滑窗层期望末维——sw_hd=0（旧契，尾参缺省）sliding_attention 层
// 期望末维 = head_dim（与 full 层同形；sliding_kv_seq_dim_rebind D2 改写，
// SP-5②：旧版按"期望末维 = sliding_window"建池断言——b688221 废口径 D8 下
// 该期望因末维失配仅 WARN 恒过、断言在错误前提上通过），图末维 ≠ 期望 →
// R3b 告警档放行直建（2026-10-15 裁定：仅告警、每池一次）；kv_len/batch
// 仍致命档。K 与 V 同层可异形（各按自己图 desc 建——om 是权威）。新契
// sw_hd>0 面见 TC_613/TC_614。
TEST(KvPoolEnsureTest, SlidingLayerEndDimMismatchWarnsOnly_TC610)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 2, 1, 8, {"sliding_attention", "full_attention"},
                     /*sliding_window=*/4); // sw_hd 缺省 0 = 旧契
    ASSERT_EQ(pool.NumKvLayers(), 2);       // 两建池层
    // layer 0（sliding，旧契）：期望末维 = head_dim = 8，图末维 8 → 命中直建
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    // 同层 V0 末维 ≠ 期望（图 4，b688221 过渡图口径形 ≠ 期望 8）→ 告警档
    // 直建（K/V 同层异形允许；om 权威）
    ASSERT_EQ(pool.EnsureKvTensor(0, true, MakeGraphKvDesc({2, 1, 16, 4})).code, Status::SUCCESS);
    EXPECT_EQ(pool.VTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 4}));
    // layer 1（full）：期望末维 8，图末维 8 → 正常
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 3u);
    // sliding 层 kv_len 致命档不受末维告警影响（16→8 → 拒）
    EXPECT_EQ(pool.EnsureKvTensor(1, true, MakeGraphKvDesc({2, 1, 8, 8})).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(pool.VTensor(1));
}

// TC_611：告警去重可测面——同类失配（heads）多层重复出现时全程 SUCCESS
// 照建（去重只影响日志重复次数，不阻断任何层创建；core 语义 = 每池同类
// 告警一次 + 创建不受影响）
TEST(KvPoolEnsureTest, WarnDedupDoesNotBlockSubsequentLayers_TC611)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 3, 1, 8);
    const std::vector<int32_t> heads_off{2, 2, 16, 8};
    for (int lay = 0; lay < 3; ++lay) {
        ASSERT_EQ(pool.EnsureKvTensor(lay, false, MakeGraphKvDesc(heads_off)).code, Status::SUCCESS);
    }
    EXPECT_EQ(pool.TensorCount(), 3u);
    EXPECT_EQ(pool.KTensor(2)->GetTensorDesc().dims, heads_off);
}

// TC_612：全 linear 层——空池静默成立（T1：不再有构造期显式失败日志），
// 层表空；Ensure 任意层 FAILURE（无建池层）
TEST(KvPoolEnsureTest, AllLinearPoolSilentEmpty_TC612)
{
    KvCachePool pool(2, 16, 2, 1, 8, {"linear_attention", "linear_attention"});
    ASSERT_EQ(pool.NumKvLayers(), 0);
    ASSERT_TRUE(pool.KvLayerIds().empty());
    EXPECT_EQ(pool.TensorCount(), 0u);
    EXPECT_FALSE(pool.KTensor(0));
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::FAILURE);
    EXPECT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::FAILURE);
}

// ═══════════════════════════════════════════════════════════════════════════
// 新契 typed 池期望（sliding_kv_seq_dim_rebind D1/D2；审查 SP-2 补 §5 U2
// 缺口、SP-3 补 §4/U6 兼容矩阵双向 fatal 缺口）。期望面（仅 R3 比对值，
// 不参与建 tensor——om 权威 R1）：
//   新契池（prefill_max>0 ∧ sw_hd>0）：sliding 层 = [b, h, C=W+P, sw_hd]、
//     full 层 = [b, h, kv_max, head_dim]（双层池同池异形）；
//   旧契池（尾参缺省 0）：sliding 层与 full 层同形 [b, h, kv_max, head_dim]。
// 兼容矩阵（D7 fail-loud）：kv_len 维（dims[2]）失配恒 INVALID_ARGUMENT——
//   新契 meta 配旧形图（期 C、图 kv_max）与旧契池配新形图（期 kv_max、
//   图 C）双向都 fatal；末维 dim（dims[3]）失配仍告警档放行（om 权威）。
// 池小形状：b=2 / kv_max=16 / h=1 / head_dim=8；W=4 / P=8 ⇒ C=12、sw_hd=64。
// ═══════════════════════════════════════════════════════════════════════════

// TC_613：新契双层池期望——sliding 层 expect seq=C=W+P、末维=sw_hd，full
// 层 expect 旧形 [b,h,kv_max,head_dim]。期望命中可直测（图 dims 按期望给 →
// SUCCESS 且不产生末维/heads 告警路径语义分叉）；末维失配只走告警档。
TEST(KvPoolEnsureTest, NewContractPoolPerLayerExpectDims_TC613)
{
    if (!HostBackendAvailable()) {
        GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用";
    }
    KvCachePool pool(2, 16, 2, 1, 8, {"sliding_attention", "full_attention"},
                     /*sliding_window=*/4, /*init_token_len=*/0,
                     /*prefill_max=*/8, /*sw_head_dim=*/64);
    ASSERT_EQ(pool.NumKvLayers(), 2);
    // layer 0（sliding，新契）：期望 {2,1,12,64}（C=4+8、sw_hd=64）——期望
    // 命中直建（无告警路径，dims 原样落位）
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 12, 64})).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 12, 64}));
    // layer 0 V 末维失配（图 8 < 期望 64）→ 告警档放行直建（D2 末维非
    // 致命维；om 权威口径与旧契 TC_610 同档）
    ASSERT_EQ(pool.EnsureKvTensor(0, true, MakeGraphKvDesc({2, 1, 12, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.VTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 12, 8}));
    // layer 1（full，不受新契影响）：期望 {2,1,16,8}（kv_max/head_dim）——
    // 双层池同池异形：sliding 层 C/sw_hd 与 full 层 kv_max/head_dim 并存
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    EXPECT_EQ(pool.TensorCount(), 3u);
}

// TC_614：兼容矩阵双向 fatal（D7/U6）——kv_len 维（dims[2]）失配 = 致命档：
//   新契池配旧形图 {..,kv_max,..}：期望 12（C）、图 16 → INVALID_ARGUMENT；
//   旧契池配新形图 {..,C,..}：期望 16（kv_max）、图 12 → INVALID_ARGUMENT。
// 两向都拦 = meta 契约 flag（sw_hd）与图代错配时至 Load 终止，不静默带病运行。
TEST(KvPoolEnsureTest, ContractMismatchFatalBothDirections_TC614)
{
    // 正向：新契池（期望 sliding 层 seq=C=12）配旧形图（kv_max=16）
    {
        KvCachePool pool(2, 16, 2, 1, 8, {"sliding_attention", "full_attention"},
                         /*sliding_window=*/4, /*init_token_len=*/0,
                         /*prefill_max=*/8, /*sw_head_dim=*/64);
        EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 64})).code, Status::INVALID_ARGUMENT)
            << "新契池×旧形图：kv_len 16≠12";
        EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::INVALID_ARGUMENT)
            << "新契池×旧全形图";
        // full 层不受新契期望影响（期 16 图 16 → 命中）
        EXPECT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 1, 16, 8})).code, Status::SUCCESS);
        EXPECT_FALSE(pool.KTensor(0)) << "致命拒不落位";
    }
    // 反向：旧契池（期望 sliding 层 seq=kv_max=16——尾参缺省 0 表示旧契）
    // 配新形图（C=12）
    {
        KvCachePool pool(2, 16, 2, 1, 8, {"sliding_attention", "full_attention"},
                         /*sliding_window=*/4); // init/prefill_max/sw_hd 全缺省
        EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 12, 64})).code, Status::INVALID_ARGUMENT)
            << "旧契池×新形图：kv_len 12≠16";
        // 同层末维=sw_hd 形态也救不了（致命维先拦——seq 轴失配即拒）
        EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 1, 12, 8})).code, Status::INVALID_ARGUMENT)
            << "旧契池×新形图（旧末维）";
        EXPECT_EQ(pool.TensorCount(), 0u);
    }
}

// TC_607：显式 dims 批量通道参数防御——out 空指针 / layer_ids 空、含负 /
// dims 空、含 ≤ 0 / dtype RESERVED / 维度乘积 uint64 溢出 → INVALID_ARGUMENT；
// 全部拒绝在创建循环前，失败不产出
TEST(KvCacheTensorApiTest, ExplicitDimsOverloadValidatesArguments_TC607)
{
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> out;
    const std::vector<int32_t> dims{2, 1, 16, 8};
    EXPECT_EQ(tm.CreateKvCacheTensor({}, dims, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // layer_ids 空
    EXPECT_EQ(tm.CreateKvCacheTensor({0, -1}, dims, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // 层号负值
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // dims 空
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {2, 0, 16, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // 维 0
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {2, -1, 16, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // 维负
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, dims, DataType::RESERVED, &out).code,
              Status::INVALID_ARGUMENT); // dtype 保留值
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {2147483647, 2147483647, 2, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT); // 乘积 uint64 溢出
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, dims, DataType::FLOAT16, nullptr).code,
              Status::INVALID_ARGUMENT); // out 空指针
    EXPECT_TRUE(out.empty());            // 任一校验失败均不产出（all-or-nothing 前置关）
}

// TC_608：显式 dims 批量通道正常路径——dims 逐维直通（HF 轴序原样进 desc），
// 输出按 [k(l0),v(l0),k(l1),v(l1),...] 配对序、tensorName 取 layer_ids 字面
// 层号（非连续表 {0,3}：证明是直通而非数组下标）、dtype/format/容量逐值
TEST(KvCacheTensorApiTest, ExplicitDimsOverloadPairsKvByLayerWithNames_TC608)
{
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> out;
    const std::vector<int> layers{0, 3}; // 非连续层号
    const std::vector<int32_t> hf_dims{2, 1, 16, 8};
    const auto s = tm.CreateKvCacheTensor(layers, hf_dims, DataType::FLOAT16, &out);
    ASSERT_EQ(s.code, Status::SUCCESS);
    ASSERT_EQ(out.size(), 4u);
    const std::vector<std::string> names{"kv_cache_k_0", "kv_cache_v_0", "kv_cache_k_3", "kv_cache_v_3"};
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_TRUE(out[i]);
        const auto& d = out[i]->GetTensorDesc();
        EXPECT_EQ(d.dims, hf_dims);
        EXPECT_EQ(d.dataType, DataType::FLOAT16);
        EXPECT_EQ(d.format, Format::ND);
        EXPECT_EQ(d.tensorName, names[i]);
        EXPECT_EQ(out[i]->GetSize(), 512u); // 2*1*16*8 × fp16 2B
    }
}

// TC_609：7 参重载默认轴序 [len,heads,batch,dim] 委托显式 dims 版——各维取
// 可区分值（heads=2/batch=3），锁旧引擎口径不因新增重载回归；命名不变
TEST(KvCacheTensorApiTest, SevenArgOverloadKeepsLegacyAxisOrder_TC609)
{
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> out;
    const auto s = tm.CreateKvCacheTensor({0}, 16, 2, 3, 4, DataType::FLOAT32, &out);
    ASSERT_EQ(s.code, Status::SUCCESS);
    ASSERT_EQ(out.size(), 2u);
    ASSERT_TRUE(out[0]);
    ASSERT_TRUE(out[1]);
    EXPECT_EQ(out[0]->GetTensorDesc().dims, (std::vector<int32_t>{16, 2, 3, 4}));
    EXPECT_EQ(out[1]->GetTensorDesc().dims, (std::vector<int32_t>{16, 2, 3, 4}));
    EXPECT_EQ(out[0]->GetTensorDesc().tensorName, "kv_cache_k_0");
    EXPECT_EQ(out[1]->GetTensorDesc().tensorName, "kv_cache_v_0");
    EXPECT_EQ(out[0]->GetTensorDesc().dataType, DataType::FLOAT32);
    EXPECT_EQ(out[0]->GetSize(), 16u * 2u * 3u * 4u * 4u); // fp32 4B
}

} // namespace cann_local_lm::engine
