#include "api/cpp/engine_executor.h"
#include "pipeline/pipeline_executor.h"   // SR20260903001：内联解析步骤 UT 载体

#include "stderr_capture.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace cann_lm::engine {
namespace {

// SR20260903001：单一构件（双图位）。三路径全空 = 无模型部署形态
// （stub-era / 无设备 UT，2026-09-22 契约第 2 点——双图位全空分支）：
// 构造跳过模型加载，生命周期 API 可用，Generate 报 executor not-ready。
// stub-era 生命周期 TC 共用。
ModelArtifact DummyArtifact() {
    ModelArtifact a;
    a.prefill_omc.path = "";
    a.decode_omc.path  = "";
    a.prefill_weight   = "";
    a.decode_weight    = "";
    return a;
}

// 图位已配置但文件不存在（dummy.* 路径不落盘）：触发 2026-09-22 新契约——
// ModelExecutor::Load 失败 ⇒ 构造内终止初始化、显式卸载双执行器、
// init_status_ 记失败码（经 EngineExecutor InitStatus / C API 原样上抛）。
ModelArtifact MissingModelArtifact() {
    ModelArtifact a;
    a.prefill_omc.path = "dummy.omc";
    a.decode_omc.path  = "dummy.omc";
    a.prefill_weight   = "dummy.weight";   // 共用权重口径（两字段同值 = 现状布局）
    a.decode_weight    = "dummy.weight";
    return a;
}

TEST(EngineExecutorTest, CreateContextSucceeds_TC001) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    auto s = ee.CreateContext(&ctx);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(static_cast<bool>(ctx));
    EXPECT_TRUE(static_cast<bool>(ctx.id()));
}

TEST(EngineExecutorTest, DestroyContextReturnsSuccessIdempotent) {
    EngineExecutor ee(DummyArtifact());
    auto s = ee.DestroyContext(ContextId{42});
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, ConfigureContextReturnsNotFound_TC101) {
    EngineExecutor ee(DummyArtifact());
    ContextOptions opts;
    auto s = ee.ConfigureContext(ContextId{99}, opts);
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

TEST(EngineExecutorTest, CancelContextReturnsNotFound_TC102) {
    EngineExecutor ee(DummyArtifact());
    auto s = ee.CancelContext(ContextId{99});
    EXPECT_EQ(s.code, Status::NOT_FOUND);
}

TEST(EngineExecutorTest, ConfigureContextSucceedsAfterCreate) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(ee.CreateContext(&ctx).ok());
    ContextOptions opts;
    auto s = ee.ConfigureContext(ctx.id(), opts);
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, CancelContextSucceedsAfterCreate) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(ee.CreateContext(&ctx).ok());
    auto s = ee.CancelContext(ctx.id());
    EXPECT_TRUE(s.ok());
}

TEST(EngineExecutorTest, GenerateReturnsFailureOnEmptyContext) {
    // empty context (id=0) → NOT_FOUND (no such context)
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    Result result;
    auto s = ee.Generate(prompt, ctx, sp, RequestOptions{}, &result);
    EXPECT_FALSE(s.ok());
}

TEST(EngineExecutorTest, GenerateReturnsFailureOnValidContextStubEra) {
    // Context exists but tokenizer is stub (encode returns FAILURE) → Generate fails
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(ee.CreateContext(&ctx).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    Result result;
    auto s = ee.Generate(prompt, ctx, sp, RequestOptions{}, &result);
    EXPECT_FALSE(s.ok());
}

TEST(EngineExecutorTest, GenerateAsyncReturnsFailureOnEmptyContext) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    auto s = ee.GenerateAsync(prompt, ctx, sp, RequestOptions{}, &handle);
    EXPECT_FALSE(s.ok());
}

TEST(EngineExecutorTest, GenerateAsyncReturnsFailureOnValidContextStubEra) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(ee.CreateContext(&ctx).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    auto s = ee.GenerateAsync(prompt, ctx, sp, RequestOptions{}, &handle);
    EXPECT_FALSE(s.ok());
}

// UT-TC-EE-011 (DRIFT-1 修复): 名宣称 INVALID_STATE 但实际两次均返回 FAILURE
// (首次同步失败 + RAII 清除 active 标记后第二次仍 FAILURE)。改名反映真实行为。
// 详见 docs/ut/SPEC.md §4.1 / docs/ut/DESIGN.md §8。
TEST(EngineExecutorTest, ConcurrentGenerateSameContextReturnsFailureStubEra) {
    EngineExecutor ee(DummyArtifact());
    Context ctx;
    ASSERT_TRUE(ee.CreateContext(&ctx).ok());
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle h1;
    // stub 时代下 PreparePrefill 同步失败（BPETokenizer::encode 返回 FAILURE），
    // active_generates_ 经 RAII 在返回前清除 → 第二次 GenerateAsync 不会触发 INVALID_STATE。
    auto s1 = ee.GenerateAsync(prompt, ctx, sp, RequestOptions{}, &h1);
    EXPECT_FALSE(s1.ok());
    // 第二次调用同样返回 FAILURE（不是 INVALID_STATE，因 active 标记已被首次调用清除）
    RequestHandle h2;
    auto s2 = ee.GenerateAsync(prompt, ctx, sp, RequestOptions{}, &h2);
    EXPECT_FALSE(s2.ok());
}

// ── 失败路径日志断言（C11 契约：Status 粗码 + 日志详因）──────────────────────

// host 无模型且图位已配置（文件不存在）：Load 失败 ⇒ 构造中止初始化并卸载
// 双执行器。日志双断言：Load 层根因键 loader.load.*（RestoreFromFile 失败）+
// pipeline 层中止键 pipeline.init（C11 契约：Status 粗码 + 日志详因，两层
// 均可观测）。失败实例不交出内部 pipeline：InitStatus 记失败码，
// CreateContext NULL 防御返回 FAILURE（工厂不交出半初始化实例）。
TEST(EngineExecutorTest, LoadFailureAbortsInitAndReports) {
    test_util::StderrCapture cap;
    EngineExecutor ee(MissingModelArtifact());
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("loader.load"), std::string::npos);
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_FALSE(ee.InitStatus().ok());
    Context ctx;   // 默认构造的出参 out（失败实例调用方出参）
    auto s = ee.CreateContext(&ctx);
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
ModelArtifact DeployedWithMetaJson(const char (&meta)[N]) {
    ModelArtifact a = MissingModelArtifact();
    a.meta_json.data = Buffer{meta, N - 1};
    return a;
}

// 情形①：JSON 语法非法（有意图 buffer，内容不是合法 JSON）
TEST(EngineExecutorTest, MetaJsonSyntaxFatalAbortsInitOnDeployedForm) {
    test_util::StderrCapture cap;
    EngineExecutor ee(DeployedWithMetaJson("{ not valid json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(ee.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = ee.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// 情形②：JSON 语法合法但 model_config 段缺失（关键段缺失 → 桩默认会在
// KV 定容/形状修正处批量爆炸，C-1 提前在构造门口暴露）
TEST(EngineExecutorTest, MetaJsonMissingModelSectionFatalAbortsInit) {
    test_util::StderrCapture cap;
    EngineExecutor ee(DeployedWithMetaJson(
        R"json({"tokenizer_config":{"tokenizer_file":"dummy.json"}})json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(ee.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = ee.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// 情形③：model_config 段存在但解析失败（kv_cache_max_len ≤ 0 容量非法）
TEST(EngineExecutorTest, MetaJsonBadModelSectionFatalAbortsInit) {
    test_util::StderrCapture cap;
    EngineExecutor ee(DeployedWithMetaJson(
        R"json({"model_config":{"kv_cache_max_len":-1,"num_hidden_layers":2}})json"));
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("pipeline.init"), std::string::npos);
    EXPECT_NE(out.find("meta_json invalid"), std::string::npos);
    EXPECT_EQ(out.find("loader.load.restore"), std::string::npos);
    EXPECT_EQ(ee.InitStatus().code, Status::INVALID_ARGUMENT);
    Context ctx;
    auto s = ee.CreateContext(&ctx);
    EXPECT_EQ(s.code, Status::FAILURE);
}

// D-1 温和门边界：meta_json 同样坏，但 omc 双空（无模型部署形态）⇒ 门不
// 触发，维持 D-16 原口径（桩默认 + WARN 降级 + init SUCCESS，生命周期照常）。
TEST(EngineExecutorTest, BadMetaJsonToleratedWhenOmcAbsent) {
    test_util::StderrCapture cap;   // 仅收集降级 WARN，不属于本断言面
    ModelArtifact a = DummyArtifact();
    static const char kBadMeta[] = "{ not valid json";
    a.meta_json.data = Buffer{kBadMeta, sizeof(kBadMeta) - 1};
    EngineExecutor ee(a);
    (void)cap.Stop();
    EXPECT_TRUE(ee.InitStatus().ok());
    Context ctx;
    EXPECT_TRUE(ee.CreateContext(&ctx).ok());
}

// context 不存在：NOT_FOUND 粗码 + pipeline.submit 日志详因（同步路径）。
TEST(EngineExecutorTest, GenerateAsyncNotFoundLogsPipelineSubmit) {
    EngineExecutor ee(DummyArtifact());
    test_util::StderrCapture cap;
    Context ctx;  // 未注册（id 不在 contexts_ 中）
    Prompt prompt;
    prompt.text = "hello";
    SamplingParams sp;
    RequestHandle handle;
    auto s = ee.GenerateAsync(prompt, ctx, sp, RequestOptions{}, &handle);
    const std::string out = cap.Stop();
    EXPECT_EQ(s.code, Status::NOT_FOUND);
    EXPECT_NE(out.find("pipeline.submit"), std::string::npos);
}

}  // namespace
}  // namespace cann_lm::engine


namespace cann_lm::engine {
// ═══════════════════════════════════════════════════════════════════════════
// SR20260903001：meta_json 三段解析（内联步骤）行为用例。
// 载体：AutoregressivePipelineExecutor 直构 + 只读观测点（M-08 §8）——
// host 无模型场景下模型 Load 失败属正常，解析产出与加载成败无关。
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// 极简合法 BPE 词表产物（TC_010 用；自包含，不依赖 fixture 目录）
std::string WriteTempBpeVocab() {
    static int seq = 0;
    const std::string p = "/tmp/ut_cfg_bpe_" + std::to_string(++seq) + ".json";
    std::ofstream f(p);
    f << R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}})";
    return p;
}

/// model.type 非 "BPE" 的词表产物（TC_214 用）：BPE 强校验拒收、Qwen 不校验收下
/// ——用于从外部区分算法选择是否生效。
std::string WriteTempNonBpeVocab() {
    static int seq = 0;
    const std::string p = "/tmp/ut_cfg_nonbpe_" + std::to_string(++seq) + ".json";
    std::ofstream f(p);
    f << R"({"model":{"type":"Unigram","vocab":{"a":0,"b":1},"merges":[]}})";
    return p;
}

/// 按值收串 + 静态保活：Buffer 是非拥有视图，须保证源串存活至构造完成
/// （初版绑临时串导致悬垂——全段降级的根因）。
Buffer Buf(std::string s) {
    static std::vector<std::string> keep;
    keep.push_back(std::move(s));
    return Buffer{keep.back().data(), keep.back().size()};
}

}  // namespace

// TC_001：三段完整解析赋值
TEST(ConfigParseTest, ThreeSectionsParsed_TC001) {
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
                         "kv_cache_max_len": 16384, "kv_cache_format": "nd"},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
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
    // tokenizer 段（pad 跨段 = 0）
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, 2);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().eos_token_id, 1);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().pad_token_id, 0);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token, "<s>");
    ASSERT_EQ(pipe.ParsedTokenizerConfig().special_tokens.size(), 1u);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().special_tokens.at("[unused9]"), 45887);
    // embedding 段（vocab/hidden 跨段）
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().table_path, "t.bin");
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().scale_path, "s.bin");
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().vocab_size, 153376u);
    EXPECT_EQ(pipe.ParsedEmbeddingConfig().hidden_dim, 2048u);
}

// TC_002：meta_json 优先于 EngineOptions
TEST(ConfigParseTest, MetaJsonOverridesOptions_TC002) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"prefill_len":[2048],
        "kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.default_chunk_size = 512;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 2048);
}

// TC_203：prefill_len 空数组 → options 回退
TEST(ConfigParseTest, EmptyPrefillLenFallsBackToOptions_TC203) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({"model_config":{"prefill_len":[],
        "kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.default_chunk_size = 512;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 512);
}

// TC_202：数字字符串容错（"28" 与 28 等价）
TEST(ConfigParseTest, NumericStringTolerated_TC202) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"num_hidden_layers":"28","kv_cache_max_len":"4096"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 28);
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 4096);
}

// TC_010：tokenizer_file 词表加载就绪
TEST(ConfigParseTest, TokenizerFileLoadsVocab_TC010) {
    ModelArtifact a;
    const std::string vocab = WriteTempBpeVocab();
    a.meta_json.data = Buf(std::string(R"({"tokenizer_config":{"tokenizer_file":")")
                        + vocab + R"("}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_TRUE(pipe.TokenizerReady());
    EXPECT_EQ(pipe.TokenizerReady() ? 2u : 0u, 2u);   // 极简词表 2 条
}

// TC_107：tokenizer_file 指向不可加载文件 → tokenizer 置空不阻断
TEST(ConfigParseTest, BadTokenizerFileUnsetsTokenizer_TC107) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"tokenizer_config":{"tokenizer_file":"/no/such/vocab.json"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_FALSE(pipe.TokenizerReady());
    // 构造未被阻断（观测点可达即证明）
}

// TC_104：空构件 → 桩默认回归（与变更前现状逐值一致）
TEST(ConfigParseTest, EmptyArtifactStubsRegress_TC104) {
    AutoregressivePipelineExecutor pipe(ModelArtifact{}, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);    // 桩等效默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8);   // 桩等效默认
    EXPECT_EQ(pipe.ParsedModelConfig().batch_num, 0);
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
    EXPECT_EQ(pipe.ParsedPrefillChunkSize(), 1024);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 4);
    EXPECT_FALSE(pipe.TokenizerReady());
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty());   // tokenId 路径
}

// TC_101：JSON 语法非法 → 三段全部桩默认
TEST(ConfigParseTest, InvalidJsonDegradesAll_TC101) {
    ModelArtifact a;
    a.meta_json.data = Buf("{ truncated");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, -1);
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty());
}

// TC_201：单段缺失其余生效
TEST(ConfigParseTest, SingleSectionOnly_TC201) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":8192,"num_hidden_layers":12}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 8192);
    EXPECT_EQ(pipe.ParsedTokenizerConfig().bos_token_id, -1);       // 桩默认
    EXPECT_TRUE(pipe.ParsedEmbeddingConfig().table_path.empty());   // 桩默认
}

// TC_204：容量字段非法 → model 段桩默认（而非以 0 定容）
TEST(ConfigParseTest, InvalidCapacityDegradesModelSection_TC204) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":0,"num_hidden_layers":28}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
}

// ── KV 形状参数（num_attention_kv_heads / num_attention_head_dims / batch_num /
//    kv_cache_type）解析行为 ──

// 新字段全量解析赋值（含数字字符串容错，口径同 TC_202；kv_cache_type 用 doc 词表 fp16）
TEST(ConfigParseTest, KvShapeParamsParsed) {
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
TEST(ConfigParseTest, KvShapeParamsAbsentDefault) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8);
    EXPECT_EQ(pipe.ParsedModelConfig().batch_num, 0);
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
}

// 显式 kv_heads ≤ 0 → 段降级（与 kv_cache_max_len ≤ 0 同口径，TC_204 扩展）
TEST(ConfigParseTest, NonPositiveKvHeadsDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "num_attention_kv_heads":0}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_kv_heads, 4);
}

// kv_cache_type 非 string → 段降级（get_str 口径）
TEST(ConfigParseTest, NonStringKvCacheTypeDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": 16}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_TRUE(pipe.ParsedModelConfig().kv_cache_type.empty());
}

// batch_num 优先级链（REQ：batch_num > max_batch_size_override > 4）：
// batch_num 在段解析成功时压过 override
TEST(ConfigParseTest, BatchNumBeatsOverride) {
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
TEST(ConfigParseTest, OverrideAppliesWhenBatchNumAbsent) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 8);
}

// 链第三级（段降级路径）：batch_num 不可得 → override 仍生效（段降级不吞 override）
TEST(ConfigParseTest, OverrideAppliesWhenModelSectionDegraded) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":0,"batch_num": 2}})");
    EngineOptions opts;
    opts.max_batch_size_override = 8;
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_EQ(pipe.ParsedDecodeBatchSize(), 8);   // 段降级 → batch_num 不采信
}

// ── kv_cache_type → DataType 解析（消费点行为，经观测点 ParsedKvCacheDtype）──

// 小写词表（仓库惯例，同 embedding_type）正确映射
TEST(ConfigParseTest, KvCacheDtypeLowercaseVocab) {
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
TEST(ConfigParseTest, KvCacheDtypeCaseInsensitive) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "FLOAT16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedKvCacheDtype(), DataType::FLOAT16);
}

// 缺省 = FLOAT16；未知值（无对应 DataType，如 "bf16"）WARN 回退 FLOAT16 不阻断
// （WARN 而非静默：stderr 断言 config.load.kv_cache_type 详因键）
TEST(ConfigParseTest, KvCacheDtypeDefaultAndUnknownFallback) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedKvCacheDtype(), DataType::FLOAT16);

    ModelArtifact b;
    b.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "kv_cache_type": "bf16"}})");
    test_util::StderrCapture cap;
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    const std::string logs = cap.Stop();
    EXPECT_EQ(pipe2.ParsedKvCacheDtype(), DataType::FLOAT16);   // WARN 回退
    EXPECT_NE(logs.find("config.load.kv_cache_type"), std::string::npos);
}

// 端到端接线（组合观测）：meta_json KV 形状参数 → 池内 tensor 实形。
// 需求 3 的组合点——pool 构造 6 参中 5 个同型 int，各值可经解析观测点分别锁定，
// 但"传递/换位"本身仅此处可测（batch=2 ≠ kv_heads=8，dims 各维可区分）。
// 注：池 tensor 申请走 TensorManager → host 上需 stub .so（LD_LIBRARY_PATH），
//     与 model_executor_ut 的 tensor 用例同一环境前提。
TEST(ConfigParseTest, KvShapeParamsWiredIntoPoolTensors) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 64, "num_hidden_layers": 2,
                         "num_attention_kv_heads": 8, "num_attention_head_dims": 128,
                         "batch_num": 2, "kv_cache_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const KvCachePool* pool = pipe.ParsedKvPool();
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->TensorCount(), 4u);   // 2×num_hidden_layers
    auto k0 = pool->KTensor(0);
    ASSERT_TRUE(k0);
    EXPECT_EQ(k0->GetTensorDesc().dims, (std::vector<int32_t>{64, 8, 2, 128}));
    EXPECT_EQ(k0->GetTensorDesc().dataType, DataType::FLOAT32);
    EXPECT_EQ(k0->GetSize(), 64u * 8u * 2u * 128u * 4u);
}

// 显式 head_dims ≤ 0 → 段降级（与 kv_heads ≤ 0 同一校验，独立字段）
TEST(ConfigParseTest, NonPositiveHeadDimsDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4,
                         "num_attention_head_dims":0}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_attention_head_dims, 8);
}

// batch_num 显式负值（段解析成功但 ≤ 0）→ 链不采信，回退 override
TEST(ConfigParseTest, NegativeBatchNumFallsThroughChain) {
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
TEST(ConfigParseTest, LayerTypesMixedParsedAndPoolsWired) {
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
TEST(ConfigParseTest, LayerTypesAllFullKeepsDenseKvAndNoStatePool) {
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
TEST(ConfigParseTest, LayerTypesAbsentKeepsLegacyBehavior) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":512,"num_hidden_layers":4}})");
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
TEST(ConfigParseTest, LayerTypesLengthMismatchDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4,
                         "layer_types": ["full_attention", "linear_attention"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 2);
    EXPECT_TRUE(pipe.ParsedModelConfig().layer_types.empty());  // 残缺表不采信
    const KvCachePool* kv = pipe.ParsedKvPool();
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->NumKvLayers(), 2);                             // 桩 nhl=2 稠密
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 非法取值（不在 full_attention|linear_attention 词表）→ 段降级
TEST(ConfigParseTest, LayerTypesInvalidValueDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 2,
                         "layer_types": ["full_attention", "sliding_window"]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 非数组（字符串/对象）→ 段降级
TEST(ConfigParseTest, LayerTypesNonArrayDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 1,
                         "layer_types": "full_attention"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_TRUE(pipe.ParsedModelConfig().layer_types.empty());
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// 元素非字符串（数字）→ 段降级
TEST(ConfigParseTest, LayerTypesNonStringElementDegradesModelSection) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 2,
                         "layer_types": ["full_attention", 3]}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 16);   // 桩默认
    EXPECT_EQ(pipe.ParsedStatePool(), nullptr);
}

// TC_206：未知段/未知字段静默忽略，已知字段不受影响
TEST(ConfigParseTest, UnknownSectionsAndFieldsIgnored_TC206) {
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
TEST(ConfigParseTest, BufferWinsOverPath_TC205) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"model_config":{"kv_cache_max_len":1024,"num_hidden_layers":6}})");
    a.meta_json.path = "/no/such/meta.json";   // 不存在——若被采用则全桩
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 1024);
}

// TC_211：图位 buffer+path 双给 → 图按 buffer 加载，meta_json 仍按 path 派生（解耦）
TEST(ConfigParseTest, DeriveFromGraphPathField_TC211) {
    static int seq = 0;
    const std::string meta = "/tmp/ut_cfg_derive_" + std::to_string(++seq) + ".json";
    {
        std::ofstream f(meta);
        f << R"({"model_config":{"kv_cache_max_len":777,"num_hidden_layers":5}})";
    }
    ModelArtifact a;
    static char omc[8] = {0};
    a.prefill_omc.data = Buffer{omc, sizeof(omc)};   // buffer 生效（加载侧）
    a.prefill_omc.path = meta;                        // 派生源按 path 字段（D-17）
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 777);   // 派生生效
}

// ── EngineOptions tokenizer 通路（tokenizer_type / tokenizer_path）──────────────

// TC_212：options.tokenizer_path 显式供给词表（无 meta_json 也可就绪）
TEST(ConfigParseTest, OptionsTokenizerPathLoads_TC212) {
    ModelArtifact a;   // 无 meta_json
    EngineOptions opts;
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());
}

// TC_213：options.tokenizer_path 优先于 meta_json tokenizer_config.tokenizer_file
TEST(ConfigParseTest, OptionsPathOverridesMetaTokenizerFile_TC213) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"tokenizer_config":{"tokenizer_file":"/no/such/vocab.json"}})");
    EngineOptions opts;
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());   // meta 指向坏路径，options 覆盖后仍就绪
}

// TC_214：options.tokenizer_type 选算法——Qwen 收下 BPE 拒收的产物（model.type 非 BPE）
TEST(ConfigParseTest, OptionsTokenizerTypeSelectsQwen_TC214) {
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
TEST(ConfigParseTest, UnknownTokenizerTypeFallsBackToBpe_TC215) {
    ModelArtifact a;
    EngineOptions opts;
    opts.tokenizer_type = "XYZ";
    opts.tokenizer_path = WriteTempBpeVocab();
    AutoregressivePipelineExecutor pipe(a, opts);
    EXPECT_TRUE(pipe.TokenizerReady());
}

// TC_216：tokenizer_config 段 add_bos / add_eos 解析（B2 通道来源接通）
TEST(ConfigParseTest, AddBosAddEosParsed_TC216) {
    ModelArtifact a;
    a.meta_json.data = Buf(
        R"({"tokenizer_config":{"add_bos":true,"add_eos":true}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    EXPECT_TRUE(pipe.ParsedTokenizerConfig().add_bos);
    EXPECT_TRUE(pipe.ParsedTokenizerConfig().add_eos);

    ModelArtifact b;   // 缺省 = 零注入（false）
    b.meta_json.data = Buf(R"({"tokenizer_config":{}})");
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    EXPECT_FALSE(pipe2.ParsedTokenizerConfig().add_bos);
    EXPECT_FALSE(pipe2.ParsedTokenizerConfig().add_eos);
}

// ── gemma4 per_layer_token_identity meta_json 键（docs/gemma4_token_identity.md §4/§7.2；
//    Q12 合并布局：仅余两个读取类型键，路径键已删除；meta 多余旧键静默忽略）──

// TC_217：embedding_config 段 identity 两类型键全解析（显式 int8/fp32）；
// 主表六键齐备 → 主表与 identity 读取类型并存（合并文件布局的完整配置形态）
TEST(ConfigParseTest, TokenIdentityKeysParsed_TC217) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
                             "embedding_type": "fp16",
                             "embedding_scale_type": "fp16",
                             "token_identity_type": "int8",
                             "token_identity_sacle_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Int8);
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(ec.token_identity_row_width, 0u);   // 加载期自工位回填，非 meta 来源
    EXPECT_EQ(ec.table_path, "t.bin");            // 主表照常（identity 与主表同一文件）
    EXPECT_EQ(ec.scale_path, "s.bin");
    EXPECT_EQ(ec.vocab_size, 153376u);            // vocab/hidden 跨段
    EXPECT_EQ(ec.hidden_dim, 2048u);
}

// TC_218：identity 键未配置 → 类型缺省（fp16/fp16——Q10/Q8）且主表/跨段不受影响
TEST(ConfigParseTest, TokenIdentityUnconfiguredDefaults_TC218) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
                             "embedding_type": "fp16"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Fp16);        // 缺省（Q10）
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16);   // 缺省（Q8）
    EXPECT_EQ(ec.token_identity_row_width, 0u);
    EXPECT_EQ(ec.table_path, "t.bin");   // 主表字段不受 identity 缺席影响
}

// TC_219：identity 键枚举非法（type "fp8" / sacle_type "none"）→ embedding 段
// 整体降级，不波及 model 段
TEST(ConfigParseTest, TokenIdentityBadTypeDegradesSectionOnly_TC219) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
                             "embedding_type": "fp16",
                             "token_identity_type": "fp8"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    // 段降级 = EmbeddingConfig{} 主 stub：主表与 identity 全部回落（tokenId 形态）
    EXPECT_TRUE(ec.table_path.empty());
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Fp16);
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp16);
    EXPECT_EQ(ec.vocab_size, 0u);   // 跨段值一并回落
    // 段间隔离：model 段不受波及（F-02 口径）
    EXPECT_EQ(pipe.ParsedModelConfig().kv_cache_max_len, 512);
    EXPECT_EQ(pipe.ParsedModelConfig().num_hidden_layers, 4);

    // sacle_type "none" 非法（Q12："none" 不再表达"不启用"——启用改由图工位
    // 判定，键仅表达读取类型）→ 同一降级口径
    ModelArtifact b;
    b.meta_json.data = Buf(R"({
        "model_config": {"kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"embedding_weights": "t.bin",
                             "embedding_dequant_scale": "s.bin",
                             "embedding_type": "fp16",
                             "token_identity_sacle_type": "none"}})");
    AutoregressivePipelineExecutor pipe2(b, EngineOptions{});
    EXPECT_TRUE(pipe2.ParsedEmbeddingConfig().table_path.empty());
}

// TC_220：主表不启用（weights 空 = TokenId 部署形态）+ identity 类型键独立解析
//（配置层独立——Q12 后 identity 数据与主表同文件，是否读取由图工位判定；
// host 加载/查表仅服务 Embedding 形态，由 pipeline Loaded() 门控——Q11）
TEST(ConfigParseTest, TokenIdentityIndependentOfMainTable_TC220) {
    ModelArtifact a;
    a.meta_json.data = Buf(R"({
        "model_config": {"hidden_size": 2048, "vocab_size": 153376,
                         "kv_cache_max_len": 512, "num_hidden_layers": 4},
        "embedding_config": {"token_identity_type": "int8",
                             "token_identity_sacle_type": "fp32"}})");
    AutoregressivePipelineExecutor pipe(a, EngineOptions{});
    const auto& ec = pipe.ParsedEmbeddingConfig();
    EXPECT_TRUE(ec.table_path.empty());                          // 主表未配置（在图权重中）
    EXPECT_EQ(ec.token_identity_type, EmbeddingType::Int8);       // identity 类型照常解析
    EXPECT_EQ(ec.token_identity_scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(ec.vocab_size, 153376u);   // 跨段填充不受主表缺席影响
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
TEST(ParseKvInputLayerTest, HfKeySuffixParsesLayerAndKeyKind_TC301) {
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
TEST(ParseKvInputLayerTest, HfValueSuffixParsesLayerAndValueKind_TC302) {
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
TEST(ParseKvInputLayerTest, HfMalformedNameReturnsInvalidLayerCode_TC303) {
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_values_", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_2", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_ab_1", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values_0_ab", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_values__0", &is_value), -2);
}

// TC_304：旧约定族不回归（past_key_in<层> → key；past_value_in<层> → value）
TEST(ParseKvInputLayerTest, LegacyPrefixParsesKeyAndValueKinds_TC304) {
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_in3", &is_value), 3);
    EXPECT_FALSE(is_value);
    is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_value_in11", &is_value), 11);
    EXPECT_TRUE(is_value);
}

// TC_305：旧约定族畸形 → -2（裸前缀空后缀；层号尾非法字符）
TEST(ParseKvInputLayerTest, LegacyMalformedSuffixReturnsInvalidLayerCode_TC305) {
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("past_key_in", &is_value), -2);
    EXPECT_EQ(ParseKvInputLayer("past_key_in3x", &is_value), -2);
}

// TC_306：非该族（常规输入工位/空名）→ -1（常规工位不占用解析口径）
TEST(ParseKvInputLayerTest, NonFamilyNameReturnsNotFoundLayerCode_TC306) {
    bool is_value = false;
    EXPECT_EQ(ParseKvInputLayer("embeddings", &is_value), -1);
    EXPECT_EQ(ParseKvInputLayer("position_ids", &is_value), -1);
    EXPECT_EQ(ParseKvInputLayer("", &is_value), -1);
}

// TC_307：conv 状态工位前缀 → kind 0（kind 先置反值再调，证明写入）
TEST(ParseStateInputLayerTest, ConvStatePrefixParsesZeroKind_TC307) {
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_5", &kind), 5);
    EXPECT_EQ(kind, 0);
}

// TC_308：rstate 状态工位前缀 → kind 1
TEST(ParseStateInputLayerTest, RStatePrefixParsesOneKind_TC308) {
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("rstate_in_layer_7", &kind), 7);
    EXPECT_EQ(kind, 1);
}

// TC_309：状态族畸形 → -2（裸前缀空后缀；层号尾非法字符）
TEST(ParseStateInputLayerTest, MalformedStateSuffixReturnsInvalidLayerCode_TC309) {
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_", &kind), -2);
    EXPECT_EQ(ParseStateInputLayer("conv_state_in_layer_5x", &kind), -2);
}

// TC_310：非状态族 → -1（常规工位；KV 旧族名亦非状态族——两族不串扰）
TEST(ParseStateInputLayerTest, NonStateFamilyNameReturnsNotFoundLayerCode_TC310) {
    int kind = 99;
    EXPECT_EQ(ParseStateInputLayer("embeddings", &kind), -1);
    EXPECT_EQ(ParseStateInputLayer("past_key_in0", &kind), -1);
}

// TC_311：五前缀全命中（KV 旧约定族 ×2 + HF 命名族 + conv/rstate 状态族）
TEST(IsInPlaceUpdateSlotNameTest, MatchesAllFiveFamilyPrefixes_TC311) {
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_key_in0"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_value_in1"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("past_key_values_2_1"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("conv_state_in_layer_3"));
    EXPECT_TRUE(IsInPlaceUpdateSlotName("rstate_in_layer_4"));
}

// TC_312：非工位名不命中——HF 元组集体名（无尾下划线 ≠ 工位前缀）、
// 前缀中缀（find==0 位置判定）、常规工位、空串
TEST(IsInPlaceUpdateSlotNameTest, RejectsNonFamilyAndMidStringPrefix_TC312) {
    EXPECT_FALSE(IsInPlaceUpdateSlotName("past_key_values"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName("npast_key_in0"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName("embeddings"));
    EXPECT_FALSE(IsInPlaceUpdateSlotName(""));
}

// ═══════════════════════════════════════════════════════════════════════════
// 问题 3 方案 c（2026-10-10，10101100_debug.log:6810 loader.load.kv_input ：
// om desc mismatch on 'past_key_values_0_0'）：KV 池形状按 om HF 布局对齐。
// 本组锁定两个生产接口的语义：
//   1) KvCachePool::EnsureKvLayout（2026-10-14 按层化：gemma4 42 层图含
//      full/sliding 两族 KV，各层 dims 可异——接口带全局层号 layer，锁为每层
//      一把，relayout 仅重建该层 K/V 两 tensor；层号不在池内 → FAILURE，
//      layer<0 → INVALID_ARGUMENT）—— HF 命名族图 KV 工位 dims
//      [batch,heads,len,dim] 与池默认轴序 [len,heads,batch,dim]（元素总数同）
//      的对齐 no-op / 按层重建 relayout / 首调锁层 / 拒绝把关语义；
//   2) TensorManager::CreateKvCacheTensor 显式 dims 批量通道（dims 逐维直通、
//      [k,v] 配对序、层号入名、参数校验 all-or-nothing）+ 7 参旧重载
//      默认轴序回归锁。
// 环境前提同 KvShapeParamsWiredIntoPoolTensors：tensor 申请经 TensorManager
// → host 桩 so（LD_LIBRARY_PATH=<repo>/build）；池用小形状（batch=2 / kv_len=16
// / 2 层 / kv_heads=1 / head_dim=8 / fp16 → 默认 dims {16,1,2,8}，元素 256、
// 512 字节，host 可真实申请）。Load 期接线（model_executor.cpp 输入循环
// pool_->EnsureKvLayout(layer,…)）需真 om 图，host 不可直测 e2e，不在本组
//（集成缝隙说明见上方 10101015 注释块）。TC 编号接续：本文件 301~312、
// ut_text_embedding 501~511 已占，本组 TC_601~TC_612（601~606 按层化改写，
// 610~612 = 2026-10-14 新增：滑窗异构建形 / linear 稀疏池 / 全 linear 守卫）。
// ═══════════════════════════════════════════════════════════════════════════

// TC_601：图 desc 防御——layer<0 / dims 空 / 任一维 ≤ 0（0 与负值）/
// dtype RESERVED → 全部 INVALID_ARGUMENT；且拒绝不改现状（池 desc 原样为
// 构造默认轴序）
TEST(KvPoolLayoutTest, InvalidDimsOrDtypeRejectedKeepsPoolUntouched_TC601) {
    // 小形状池：默认轴序 [len,heads,batch,dim] = {16,1,2,8}，fp16
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16);
    ASSERT_EQ(pool.TensorCount(), 4u);   // 2×NumKvLayers 构造申请成功（前置）
    const auto k0 = pool.KTensor(0);
    ASSERT_TRUE(k0);
    ASSERT_EQ(k0->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    ASSERT_EQ(k0->GetTensorDesc().dataType, DataType::FLOAT16);

    EXPECT_EQ(pool.EnsureKvLayout(0, {}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvLayout(0, {16, 0, 2, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvLayout(0, {16, -1, 2, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvLayout(0, {16, 1, 2, 8}, DataType::RESERVED).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.EnsureKvLayout(-1, {16, 1, 2, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);   // layer<0 新增防御（按层化入参）

    // 拒绝不改现状：tensor desc 仍为构造默认，计数不变
    EXPECT_EQ(pool.TensorCount(), 4u);
    EXPECT_EQ(k0->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    EXPECT_EQ(k0->GetTensorDesc().dataType, DataType::FLOAT16);
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
}

// TC_602：构造期申请失败形态（kv_len=0 → 维度非法 → 池空、槽位记账照常）
// → EnsureKvLayout SUCCESS no-op（无从对齐；绑定侧 pool tensor missing 收口）。
// 按层化：layer<0 仍 INVALID_ARGUMENT（入参防御先于空池短路）
TEST(KvPoolLayoutTest, EmptyPoolEnsureIsNoOpSuccess_TC602) {
    KvCachePool pool(2, 0, 2, 1, 8, DataType::FLOAT16);   // ctor 申请失败 → tensors_ 空
    ASSERT_EQ(pool.NumKvLayers(), 2);                    // 层表不受申请失败影响
    ASSERT_EQ(pool.KvLayerIds(), (std::vector<int>{0, 1}));
    ASSERT_EQ(pool.TensorCount(), 0u);                   // 空池形态（前置）

    EXPECT_EQ(pool.EnsureKvLayout(-1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);                 // 入参防御先于空池短路
    EXPECT_EQ(pool.EnsureKvLayout(0, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 0u);          // no-op：不为空池建 tensor
    EXPECT_FALSE(pool.KTensor(0));              // tensor 缺席如故
    EXPECT_FALSE(pool.VTensor(1));
}

// TC_603：dtype 真失配（图 KV 工位 dtype ≠ meta_json kv_cache_type 解析所得
// 池 dtype），无论 dims 是否相等 → INVALID_ARGUMENT；不置锁、不破坏现状——
// 随后语义正确的调用仍可对齐
TEST(KvPoolLayoutTest, DtypeMismatchRejectedWithoutSideEffects_TC603) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16);
    // dims 相等但 dtype 失配 → 拒（dtype 检查先于 dims 相等 no-op）
    EXPECT_EQ(pool.EnsureKvLayout(1, {16, 1, 2, 8}, DataType::FLOAT32).code,
              Status::INVALID_ARGUMENT);
    // dims、dtype 均不同（HF 轴序 + fp32）→ 同口径拒绝
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT32).code,
              Status::INVALID_ARGUMENT);

    // 不置锁：随后语义正确的 fp16 HF 轴序 relayout 仍 SUCCESS
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
}

// TC_604：同层锁语义（按层化核心）——该层 dims 全等 no-op 且锁该层，该层后续
// 换序被拒；他层不受该层锁影响（gemma4 滑窗异构 KV 各层独立对齐的根基）
TEST(KvPoolLayoutTest, EqualDimsLocksThenRejectsDivergentGraph_TC604) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16);
    EXPECT_EQ(pool.EnsureKvLayout(0, {16, 1, 2, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);   // 全等 no-op + 锁 layer 0
    EXPECT_EQ(pool.TensorCount(), 4u);   // no-op：未重建
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));

    // layer 0 首调已锁：元素总数相同的 HF 轴序（换序不换量）被拒，
    // 该层维持首图 dims
    EXPECT_EQ(pool.EnsureKvLayout(0, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    EXPECT_EQ(pool.VTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));

    // 他层不受 layer 0 锁影响：layer 1 HF 轴序 relayout 照常 SUCCESS
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
}

// TC_605：HF 轴序按层对齐主路径——未锁、元素总数同（uint64 累乘防护通过）、
// dtype 同 → 仅重建该层 2 个 tensor。锁定：新 dims/dtype/tensorName/容量实形、
// 他层 tensor 原样（140 按层化核心证据）、池结构（计数/层表）不变、槽位/光标
// 记账不受重建影响；调用幂等；relayout 定型后该层旧默认轴序被拒（首个 Load
// 图权威，按层）
TEST(KvPoolLayoutTest, HfRelayoutRebuildsTensorsKeepsBookkeeping_TC605) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16);
    ASSERT_EQ(pool.TensorCount(), 4u);
    // 记账预热：占槽 + 推进光标（relayout 不应扰动记账本）
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_GE(slot, 0);
    ASSERT_EQ(pool.Advance(slot, 5).code, Status::SUCCESS);
    ASSERT_EQ(pool.NextPosition(slot), 5u);

    // HF 图轴序 [batch,heads,len,dim]，元素总数与默认 {16,1,2,8} 同（256）
    const std::vector<int32_t> hf_dims{2, 1, 16, 8};
    EXPECT_EQ(pool.EnsureKvLayout(0, hf_dims, DataType::FLOAT16).code, Status::SUCCESS);

    // 重建后 layer 0 K/V 实形：dims/dtype/format/tensorName/容量
    const std::vector<std::string> names{"kv_cache_k_0", "kv_cache_v_0"};
    for (int kv = 0; kv < 2; ++kv) {
        const auto t = kv == 0 ? pool.KTensor(0) : pool.VTensor(0);
        ASSERT_TRUE(t) << "layer=0 kv=" << kv;
        const auto& d = t->GetTensorDesc();
        EXPECT_EQ(d.dims, hf_dims);
        EXPECT_EQ(d.dataType, DataType::FLOAT16);
        EXPECT_EQ(d.format, Format::ND);
        EXPECT_EQ(d.tensorName, names[static_cast<std::size_t>(kv)]);
        EXPECT_EQ(t->GetSize(), 512u);   // 2*1*16*8 元素 × fp16 2B
    }
    // 他层不受该层 relayout 影响（仅重建该层 K/V 两 tensor）
    for (int kv = 0; kv < 2; ++kv) {
        const auto t = kv == 0 ? pool.KTensor(1) : pool.VTensor(1);
        ASSERT_TRUE(t);
        const auto& d = t->GetTensorDesc();
        EXPECT_EQ(d.dims, (std::vector<int32_t>{16, 1, 2, 8}));   // 构造默认轴序原样
        EXPECT_EQ(d.dataType, DataType::FLOAT16);
        EXPECT_EQ(d.tensorName,
                  kv == 0 ? std::string("kv_cache_k_1") : std::string("kv_cache_v_1"));
    }
    // 池结构不变：计数/层表保持（tensor 按层重建、kv_layer_ids_ 与槽位记账不动）
    EXPECT_EQ(pool.TensorCount(), 4u);
    EXPECT_EQ(pool.NumKvLayers(), 2);
    EXPECT_EQ(pool.KvLayerIds(), (std::vector<int>{0, 1}));
    // 记账不受重建影响：光标保留、可继续推进、槽生命周期照常
    EXPECT_EQ(pool.NextPosition(slot), 5u);
    EXPECT_EQ(pool.Advance(slot, 1).code, Status::SUCCESS);
    EXPECT_EQ(pool.NextPosition(slot), 6u);
    EXPECT_EQ(pool.FreeKvSlot(slot).code, Status::SUCCESS);

    // 幂等：同层同 dims 重调 SUCCESS no-op（不重复重建）；他层随后对齐照常
    EXPECT_EQ(pool.EnsureKvLayout(0, hf_dims, DataType::FLOAT16).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 4u);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, hf_dims);
    EXPECT_EQ(pool.EnsureKvLayout(1, hf_dims, DataType::FLOAT16).code, Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, hf_dims);
    // relayout 已定型（首图权威，按层）：layer 0 旧默认轴序（元素总数同）也被拒
    EXPECT_EQ(pool.EnsureKvLayout(0, {16, 1, 2, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, hf_dims);   // 拒绝不改现状
}

// TC_606：元素总数失配（meta_json 池容量与 om 图失配）→ INVALID_ARGUMENT，
// 池 tensor 原样保留；且不置锁（拒绝无副作用）——真失配显式报错
TEST(KvPoolLayoutTest, ElementCountMismatchRejectedKeepsPool_TC606) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16);
    // 默认池 256 元素 vs 图 512 元素 → 总数失配拒绝
    EXPECT_EQ(pool.EnsureKvLayout(1, {16, 1, 2, 16}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    // 图 dims 元素乘积 uint64 溢出 → 同口径真失配拒绝（uint64 累乘防护）
    EXPECT_EQ(pool.EnsureKvLayout(
                  1, {2147483647, 2147483647, 2, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
    // 层号不在池内（全 linear 稀疏口径的资源类Failure；稠密 2 层池中越界层
    // 同口径）→ FAILURE
    EXPECT_EQ(pool.EnsureKvLayout(9, {16, 1, 2, 8}, DataType::FLOAT16).code,
              Status::FAILURE);
    EXPECT_EQ(pool.TensorCount(), 4u);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dataType, DataType::FLOAT16);
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    // 失败未置锁：随后该层元素总数相同的 HF 轴序 relayout 仍 SUCCESS
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
}

// ── 按层建形新域（2026-10-14 gemma4 42 层修复）TC_610~TC_612 ────────────────
// ① layer_types 白名单 + ② 滑窗异构末维 + ④ 按层锁 的构造/查询/对齐全链语义

// TC_610：滑窗异构建形——sliding 层末维 = sliding_window、full 层末维 =
// head_dim；两族元素数不同仍可各自按图对齐（gemma4 42 层 70×[…,256] +
// 14×[…,512] 的小形状缩影）
TEST(KvPoolLayoutTest, SlidingWindowBuildsHeterogeneousEndDims_TC610) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16,
                     {"sliding_attention", "full_attention"}, /*sliding_window=*/4);
    ASSERT_EQ(pool.NumKvLayers(), 2);      // 两种建池层都在池内
    ASSERT_EQ(pool.TensorCount(), 4u);
    // 分族建形：layer 0（sliding）末维 = SW=4；layer 1（full）末维 = head_dim=8
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 4}));
    EXPECT_EQ(pool.VTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 4}));
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    // 两族各自按图轴序对齐（元素总数各自成立，互不干扰）
    EXPECT_EQ(pool.EnsureKvLayout(0, {2, 1, 16, 4}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 4}));
    EXPECT_EQ(pool.KTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    // 异形层间不可互拷误判：layer 0 拿 layer 1 的族形 → 元素总数失配拒绝
    EXPECT_EQ(pool.EnsureKvLayout(0, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::INVALID_ARGUMENT);
}

// TC_611：linear_attention 稀疏池——linear 层不建 tensor 不占池位（NumKvLayers
// 只数建池层）；KTensor/VTensor(linear 层) 空；EnsureKvLayout(线性层) FAILURE
TEST(KvPoolLayoutTest, LinearAttentionBuildsSparsePool_TC611) {
    KvCachePool pool(2, 16, 3, 1, 8, DataType::FLOAT16,
                     {"full_attention", "linear_attention", "full_attention"});
    ASSERT_EQ(pool.NumKvLayers(), 2);                    // 线性层不进池
    ASSERT_EQ(pool.KvLayerIds(), (std::vector<int>{0, 2}));
    ASSERT_EQ(pool.TensorCount(), 4u);
    EXPECT_TRUE(pool.KTensor(0));
    EXPECT_FALSE(pool.KTensor(1));                      // linear 层无 tensor
    EXPECT_FALSE(pool.VTensor(1));
    EXPECT_TRUE(pool.KTensor(2));
    EXPECT_EQ(pool.KTensor(2)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
    // linear 层不在池内（资源类口径 FAILURE，非 INVALID_ARGUMENT）
    EXPECT_EQ(pool.EnsureKvLayout(1, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::FAILURE);
    // 池内层照常对齐（跳过 linear 层不产生空洞错位）
    EXPECT_EQ(pool.EnsureKvLayout(2, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.KTensor(2)->GetTensorDesc().dims, (std::vector<int32_t>{2, 1, 16, 8}));
    EXPECT_EQ(pool.KTensor(0)->GetTensorDesc().dims, (std::vector<int32_t>{16, 1, 2, 8}));
}

// TC_612：全 linear 层守恒——无建池层时构造显式失败（空池、层表空），
// EnsureKvLayout 走空池 SUCCESS 短路（无从对齐；口径与 TC_602 一致）
TEST(KvPoolLayoutTest, AllLinearPoolFailsClosed_TC612) {
    KvCachePool pool(2, 16, 2, 1, 8, DataType::FLOAT16,
                     {"linear_attention", "linear_attention"});
    ASSERT_EQ(pool.NumKvLayers(), 0);
    ASSERT_TRUE(pool.KvLayerIds().empty());
    ASSERT_EQ(pool.TensorCount(), 0u);
    EXPECT_FALSE(pool.KTensor(0));
    // 空池 SUCCESS no-op（绑定侧 pool tensor missing 收口）
    EXPECT_EQ(pool.EnsureKvLayout(0, {2, 1, 16, 8}, DataType::FLOAT16).code,
              Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 0u);
}

// TC_607：显式 dims 批量通道参数防御——out 空指针 / layer_ids 空、含负 /
// dims 空、含 ≤ 0 / dtype RESERVED / 维度乘积 uint64 溢出 → INVALID_ARGUMENT；
// 全部拒绝在创建循环前，失败不产出
TEST(KvCacheTensorApiTest, ExplicitDimsOverloadValidatesArguments_TC607) {
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> out;
    const std::vector<int32_t> dims{2, 1, 16, 8};
    EXPECT_EQ(tm.CreateKvCacheTensor({}, dims, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // layer_ids 空
    EXPECT_EQ(tm.CreateKvCacheTensor({0, -1}, dims, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // 层号负值
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // dims 空
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {2, 0, 16, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // 维 0
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, {2, -1, 16, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // 维负
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, dims, DataType::RESERVED, &out).code,
              Status::INVALID_ARGUMENT);                                     // dtype 保留值
    EXPECT_EQ(tm.CreateKvCacheTensor(
                  {0}, {2147483647, 2147483647, 2, 8}, DataType::FLOAT16, &out).code,
              Status::INVALID_ARGUMENT);                                     // 乘积 uint64 溢出
    EXPECT_EQ(tm.CreateKvCacheTensor({0}, dims, DataType::FLOAT16, nullptr).code,
              Status::INVALID_ARGUMENT);                                     // out 空指针
    EXPECT_TRUE(out.empty());   // 任一校验失败均不产出（all-or-nothing 前置关）
}

// TC_608：显式 dims 批量通道正常路径——dims 逐维直通（HF 轴序原样进 desc），
// 输出按 [k(l0),v(l0),k(l1),v(l1),...] 配对序、tensorName 取 layer_ids 字面
// 层号（非连续表 {0,3}：证明是直通而非数组下标）、dtype/format/容量逐值
TEST(KvCacheTensorApiTest, ExplicitDimsOverloadPairsKvByLayerWithNames_TC608) {
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> out;
    const std::vector<int> layers{0, 3};            // 非连续层号
    const std::vector<int32_t> hf_dims{2, 1, 16, 8};
    const auto s = tm.CreateKvCacheTensor(layers, hf_dims, DataType::FLOAT16, &out);
    ASSERT_EQ(s.code, Status::SUCCESS);
    ASSERT_EQ(out.size(), 4u);
    const std::vector<std::string> names{"kv_cache_k_0", "kv_cache_v_0",
                                         "kv_cache_k_3", "kv_cache_v_3"};
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_TRUE(out[i]);
        const auto& d = out[i]->GetTensorDesc();
        EXPECT_EQ(d.dims, hf_dims);
        EXPECT_EQ(d.dataType, DataType::FLOAT16);
        EXPECT_EQ(d.format, Format::ND);
        EXPECT_EQ(d.tensorName, names[i]);
        EXPECT_EQ(out[i]->GetSize(), 512u);   // 2*1*16*8 × fp16 2B
    }
}

// TC_609：7 参重载默认轴序 [len,heads,batch,dim] 委托显式 dims 版——各维取
// 可区分值（heads=2/batch=3），锁旧引擎口径不因新增重载回归；命名不变
TEST(KvCacheTensorApiTest, SevenArgOverloadKeepsLegacyAxisOrder_TC609) {
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
    EXPECT_EQ(out[0]->GetSize(), 16u * 2u * 3u * 4u * 4u);   // fp32 4B
}

}  // namespace cann_lm::engine
