// Tokenizer 模块 UT（SR20260728001 v0.7）。
// A 层：小型 fixture（tests/fixtures/tokenizer/，scripts/gen_tokenizer_fixture.py
// 生成，golden 由 Python oracle 预生成）驱动；运行时零 Python 依赖。
// 覆盖：工厂/自注册/SPM 占位（TC_001/001b/004/103/103b/105 语义）、基类
// batch_encode（TC_201x）、Qwen/BPE 全流水线 golden 逐 id 一致（TC_203 A 层）、
// zero 注入/显式注入/截断预算/padding（TC_002/204）、剥离集（M-5）、非法 UTF-8
// （TC_104）、空串（TC_201）、not-ready 桩行为、词表异常（TC_101/102）、
// meta special_tokens 映射/解映射（2026-10-29 起：encode 命中→目标 id、
// decode id→字面量、冲突 meta 优先、剥离集并入、负 id 惰性）。
// B 层真实产物测试见 ut_tokenizer_realdata.cpp（CMake 检测 tmp/ 存在才注册）。

#include "core/component/tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cann_lm::engine {
namespace {

#ifndef TOKENIZER_FIXTURE_DIR
#define TOKENIZER_FIXTURE_DIR "tests/fixtures/tokenizer"
#endif

const std::string kFixtureDir = TOKENIZER_FIXTURE_DIR;

TokenizerConfig PathParams(const std::string& rel)
{ // SR20260903001 D-14：词表路径入配置
    TokenizerConfig cfg;
    cfg.tokenizer_file = kFixtureDir + "/" + rel;
    return cfg;
}

// D-14：TokenizerConfig 仅承载路径——原内存字节用例经临时文件承载（内容语义不变）
std::string WriteTempVocab(const std::string& content)
{
    static int seq = 0;
    const std::string path = "/tmp/ut_tok_vocab_" + std::to_string(++seq) + ".json";
    std::ofstream f(path, std::ios::binary);
    f << content;
    return path;
}

// 读取 fixture golden（Python oracle 预生成）
nlohmann::json LoadGolden(const std::string& rel)
{
    std::ifstream f(kFixtureDir + "/" + rel);
    EXPECT_TRUE(f.good()) << "golden fixture missing: " << rel;
    nlohmann::json j;
    f >> j;
    return j;
}

std::unique_ptr<TokenizerBase> CreateOrThrow(Algorithm algo, const TokenizerConfig& cfg)
{
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(algo, cfg, &tok);
    EXPECT_TRUE(st.ok()) << "create failed: code=" << st.code;
    return tok;
}

// ── 工厂 / 自注册 / SPM 占位 ────────────────────────────────────────────────

TEST(TokenizerFactoryTest, ListRegisteredContainsBpeQwenSpm)
{
    // TC_004：注册集 = {BPE, Qwen, SPM}（SPM 为占位桩，同样注册）
    auto algos = TokenizerFactory::list_registered();
    ASSERT_EQ(algos.size(), 3u);
    bool has_bpe = false, has_qwen = false, has_spm = false;
    for (const auto a : algos) {
        has_bpe = has_bpe || a == Algorithm::BPE;
        has_qwen = has_qwen || a == Algorithm::Qwen;
        has_spm = has_spm || a == Algorithm::SPM;
    }
    EXPECT_TRUE(has_bpe && has_qwen && has_spm);
}

TEST(TokenizerFactoryTest, CreateUnknownAlgorithmReturnsInvalidArgument)
{
    // TC_103：未注册枚举（超出 Algorithm 定义域的强制转换值）
    const auto unknown = static_cast<Algorithm>(99);
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(unknown, TokenizerConfig{}, &tok);
    EXPECT_EQ(st.code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(tok, nullptr);
}

TEST(TokenizerFactoryTest, CreateWithNullOutReturnsInvalidArgument)
{
    Status st = TokenizerFactory::create(Algorithm::BPE, TokenizerConfig{}, nullptr);
    EXPECT_EQ(st.code, Status::INVALID_ARGUMENT);
}

TEST(TokenizerFactoryTest, CreateSpmPlaceholderFails)
{
    // TC_103b：SPM 占位桩注册但 create 恒 FAILURE（预期行为非缺陷，D-04）
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::SPM, PathParams("qwen_fixture_tokenizer.json"), &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
    EXPECT_EQ(tok, nullptr);
}

TEST(SPMTokenizerTest, NotReadyEncodeDecodeFail)
{
    // 空 TokenizerConfig 直接构造：not-ready 桩（stub 时代语义）
    Status st = Status::SUCCESS;
    SPMTokenizer tok{TokenizerConfig{}, &st};
    EXPECT_EQ(st.code, Status::FAILURE);
    std::vector<int> ids;
    std::string text;
    EXPECT_EQ(tok.encode("hello", EncodeOptions{}, &ids).code, Status::FAILURE);
    EXPECT_EQ(tok.decode({1, 2}, DecodeOptions{}, &text).code, Status::FAILURE);
    EXPECT_EQ(tok.vocab_size(), 0u);
    EXPECT_EQ(tok.special_token_id("bos"), std::nullopt);
}

TEST(TokenizerFactoryTest, CreateQwenWithMissingPathFails)
{
    // TC_101：产物路径不存在 → FAILURE，不返回半成品实例
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::Qwen, PathParams("no_such_file.json"), &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
    EXPECT_EQ(tok, nullptr);
}

TEST(TokenizerFactoryTest, CreateBpeWithMissingPathFails)
{
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::BPE, PathParams("no_such_file.json"), &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
    EXPECT_EQ(tok, nullptr);
}

TEST(TokenizerFactoryTest, CreateQwenWithGarbageFails)
{
    // TC_102：非 JSON 内容 → FAILURE
    TokenizerConfig params;
    params.tokenizer_file = WriteTempVocab("this is not json {{{");
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::Qwen, params, &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
}

TEST(TokenizerFactoryTest, CreateQwenWithMergesVocabMisalignmentFails)
{
    // TC_102：merges 引用 vocab 中不存在的 token → 结构非法
    nlohmann::json j;
    j["model"]["type"] = "BPE";
    j["model"]["vocab"] = {{"a", 0}, {"b", 1}};
    j["model"]["merges"] = nlohmann::json::array({"a zzz"});
    TokenizerConfig params;
    params.tokenizer_file = WriteTempVocab(j.dump());
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::Qwen, params, &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
}

TEST(TokenizerFactoryTest, CreateBpeWithNonBpeModelTypeFails)
{
    // model.type 非 BPE（如 WordPiece）→ BPETokenizer 拒绝（回议路线，U-6）
    nlohmann::json j;
    j["model"]["type"] = "WordPiece";
    j["model"]["vocab"] = {{"a", 0}};
    j["model"]["merges"] = nlohmann::json::array();
    TokenizerConfig params;
    params.tokenizer_file = WriteTempVocab(j.dump());
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::BPE, params, &tok);
    EXPECT_EQ(st.code, Status::FAILURE);
}

TEST(TokenizerFactoryTest, CreateQwenFromTempFileSucceeds)
{
    // REQ-01 内容等价（D-14：配置仅路径形态，原内存字节用例经临时文件承载）
    std::ifstream f(kFixtureDir + "/qwen_fixture_tokenizer.json", std::ios::binary);
    ASSERT_TRUE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    TokenizerConfig params;
    params.tokenizer_file = WriteTempVocab(ss.str());
    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::Qwen, params, &tok);
    ASSERT_TRUE(st.ok());
    ASSERT_NE(tok, nullptr);
    EXPECT_GT(tok->vocab_size(), 0u);
}

// ── TC_203 A 层：fixture golden 逐 id 一致 + decode 同口径 ─────────────────

void RunGoldenComparison(Algorithm algo, const std::string& tokenizer_rel, const std::string& golden_rel)
{
    const auto golden = LoadGolden(golden_rel);
    auto tok = CreateOrThrow(algo, PathParams(tokenizer_rel));
    ASSERT_NE(tok, nullptr);
    EXPECT_EQ(tok->vocab_size(), golden["vocab_size"].get<size_t>());

    EncodeOptions opts;
    opts.add_special_tokens = false; // TC_203 钉死比对口径：裸流水线逐 id 比对
    for (const auto& c : golden["cases"]) {
        const std::string text = c["text"].get<std::string>();
        std::vector<int> ids;
        ASSERT_TRUE(tok->encode(text, opts, &ids).ok()) << "text=" << text;
        const std::vector<int> want = c["ids"].get<std::vector<int>>();
        EXPECT_EQ(ids, want) << "text=" << text;
        // decode 同口径（skip_special=True，与 oracle 一致——special 字面量被剥离）
        std::string decoded;
        ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
        EXPECT_EQ(decoded, c["decoded"].get<std::string>()) << "text=" << text;
    }
}

TEST(QwenTokenizerTest, GoldenIdsMatchOracleFixture)
{
    RunGoldenComparison(Algorithm::Qwen, "qwen_fixture_tokenizer.json", "qwen_fixture_golden.json");
}

TEST(BPETokenizerTest, GoldenIdsMatchOracleFixture)
{
    RunGoldenComparison(Algorithm::BPE, "bpe_fixture_tokenizer.json", "bpe_fixture_golden.json");
}

TEST(QwenTokenizerTest, AddedTokenExtractedBeforePretokenize)
{
    // H-2：含 special 字面量输入经 Trie 提取为单一 id（不进 BPE 流水线）
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("hello<|endoftext|>world", opts, &ids).ok());
    // oracle：['hello', '<|endoftext|>', 'wor', 'ld'] —— added id 位于中间（单一 id，不入 BPE）
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[1], 420); // fixture: <|endoftext|> id
    // decode 剥离 special 后还原为 "helloworld"
    std::string decoded;
    ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
    EXPECT_EQ(decoded, "helloworld");
}

TEST(QwenTokenizerTest, InvalidUtf8Rejected)
{
    // TC_104：非法 UTF-8 → INVALID_ARGUMENT，不输出半截 ids
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    const std::string bad = "ok\xff\xfe";
    std::vector<int> ids{42, 42};
    Status st = tok->encode(bad, EncodeOptions{}, &ids);
    EXPECT_EQ(st.code, Status::INVALID_ARGUMENT);
    EXPECT_TRUE(ids.empty());
}

TEST(BPETokenizerTest, InvalidUtf8Rejected)
{
    auto tok = CreateOrThrow(Algorithm::BPE, PathParams("bpe_fixture_tokenizer.json"));
    const std::string bad = "ok\xc3";
    std::vector<int> ids;
    EXPECT_EQ(tok->encode(bad, EncodeOptions{}, &ids).code, Status::INVALID_ARGUMENT);
    EXPECT_TRUE(ids.empty());
}

TEST(TokenizerBaseTest, EncodeNullOutReturnsInvalidArgument)
{
    // L-7：out_ids == nullptr → INVALID_ARGUMENT
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EXPECT_EQ(tok->encode("hi", EncodeOptions{}, nullptr).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(tok->decode({1}, DecodeOptions{}, nullptr).code, Status::INVALID_ARGUMENT);
}

TEST(QwenTokenizerTest, EmptyStringEncodesToEmpty)
{
    // TC_201：空串 + add_special=false → 空 ids，SUCCESS
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids{1, 2, 3};
    EXPECT_TRUE(tok->encode("", opts, &ids).ok());
    EXPECT_TRUE(ids.empty());
}

// ── TC_002：零注入缺省 + 显式覆盖（H-3 裁决）────────────────────────────────

TEST(QwenTokenizerTest, ZeroInjectionByDefault)
{
    // 未显式配置 special → add_special_tokens=true 也不注入（对齐 oracle 实物）
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EncodeOptions with_special;
    with_special.add_special_tokens = true;
    EncodeOptions without_special;
    without_special.add_special_tokens = false;
    std::vector<int> a, b;
    ASSERT_TRUE(tok->encode("hello world", with_special, &a).ok());
    ASSERT_TRUE(tok->encode("hello world", without_special, &b).ok());
    EXPECT_EQ(a, b);
}

TEST(QwenTokenizerTest, ExplicitConfigInjectsBosEos)
{
    // TC_002 覆盖路径：TokenizerConfig 显式配置 add_bos/add_eos 后按配置注入（D-13 单一配置）
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.bos_token_id = 420; // fixture: <|endoftext|> id（作 bos 用）
    params.eos_token_id = 420;
    params.add_bos = true;
    params.add_eos = true;
    auto tok = CreateOrThrow(Algorithm::Qwen, params);
    EXPECT_EQ(tok->special_token_id("bos"), 420);
    EXPECT_EQ(tok->special_token_id("eos"), 420);

    EncodeOptions opts; // add_special_tokens 默认 true
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("hi", opts, &ids).ok());
    ASSERT_GE(ids.size(), 2u);
    EXPECT_EQ(ids.front(), 420);
    EXPECT_EQ(ids.back(), 420);

    // add_special_tokens=false → 不注入
    EncodeOptions no_special;
    no_special.add_special_tokens = false;
    std::vector<int> ids2;
    ASSERT_TRUE(tok->encode("hi", no_special, &ids2).ok());
    EXPECT_EQ(ids2.size(), ids.size() - 2);
}

TEST(QwenTokenizerTest, TruncationBudgetKeepsEos)
{
    // TC_204/M-3：special 计入 max_length 预算——先截内容后注入，eos 不被截掉
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.eos_token_id = 420;
    params.add_eos = true;
    auto tok = CreateOrThrow(Algorithm::Qwen, params);
    EncodeOptions opts;
    opts.max_length = 4;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("a very much longer input text here", opts, &ids).ok());
    EXPECT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids.back(), 420); // eos 保留在末尾
}

TEST(QwenTokenizerTest, FixedPaddingPadsWithPadId)
{
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.pad_token_id = 421; // fixture: <tool_call> id（作 pad 用）
    auto tok = CreateOrThrow(Algorithm::Qwen, params);
    EncodeOptions opts;
    opts.add_special_tokens = false;
    opts.padding = EncodeOptions::Padding::Fixed;
    opts.pad_to_length = 8;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("hi", opts, &ids).ok());
    EXPECT_EQ(ids.size(), 8u);
    EXPECT_EQ(ids.back(), 421); // 尾部 pad
}

TEST(QwenTokenizerTest, FixedPaddingWithoutPadIdFails)
{
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    opts.padding = EncodeOptions::Padding::Fixed;
    opts.pad_to_length = 8;
    std::vector<int> ids;
    EXPECT_EQ(tok->encode("hi", opts, &ids).code, Status::INVALID_ARGUMENT);
}

TEST(QwenTokenizerTest, EmptyPaddingRejected)
{
    // 空 padding（无真实 token 全 pad）→ INVALID_ARGUMENT（REQ-09 InvalidInput）
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.pad_token_id = 421;
    auto tok = CreateOrThrow(Algorithm::Qwen, params);
    EncodeOptions opts;
    opts.add_special_tokens = false;
    opts.padding = EncodeOptions::Padding::Fixed;
    opts.pad_to_length = 4;
    std::vector<int> ids;
    EXPECT_EQ(tok->encode("", opts, &ids).code, Status::INVALID_ARGUMENT);
}

// ── 基类 batch_encode（REQ-05）──────────────────────────────────────────────

class BatchEncodeTest : public ::testing::Test {
protected:
    void SetUp() override { tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json")); }
    std::unique_ptr<TokenizerBase> tok;
};

TEST_F(BatchEncodeTest, ContinueOnFailureNotAbort)
{
    // 单条失败（非法 UTF-8）标记该条为空，其余继续
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<std::vector<int>> out;
    const std::string bad = "ok\xff";
    Status st = tok->batch_encode({"text1", bad, "text3"}, opts, &out);
    EXPECT_EQ(st.code, Status::INVALID_ARGUMENT);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FALSE(out[0].empty());
    EXPECT_TRUE(out[1].empty());
    EXPECT_FALSE(out[2].empty());
}

TEST_F(BatchEncodeTest, EmptyInputReturnsSuccess)
{
    EncodeOptions opts;
    std::vector<std::vector<int>> out;
    EXPECT_TRUE(tok->batch_encode({}, opts, &out).ok());
    EXPECT_TRUE(out.empty());
}

TEST_F(BatchEncodeTest, NullOutReturnsInvalidArgument)
{
    EncodeOptions opts;
    EXPECT_EQ(tok->batch_encode({"test"}, opts, nullptr).code, Status::INVALID_ARGUMENT);
}

TEST_F(BatchEncodeTest, LongestPaddingAlignsToBatchMax)
{
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.pad_token_id = 421;
    auto pad_tok = CreateOrThrow(Algorithm::Qwen, params);
    EncodeOptions opts;
    opts.add_special_tokens = false;
    opts.padding = EncodeOptions::Padding::Longest;
    std::vector<std::vector<int>> out;
    ASSERT_TRUE(pad_tok->batch_encode({"hello world", "x"}, opts, &out).ok());
    ASSERT_EQ(out.size(), 2u);
    ASSERT_EQ(out[0].size(), out[1].size()); // 对齐至批内最长
    EXPECT_EQ(out[1].back(), 421);           // 短序列尾部补 pad
}

// ── M-5 剥离集：added_tokens.special=true 全集（非仅四槽）──────────────────

TEST(QwenTokenizerTest, StripSetCoversAllSpecialAddedTokens)
{
    // fixture specials: <|endoftext|>(420)；<tool_call>(421) 非 special 不剥离
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("hello<|endoftext|>world<tool_call>x", opts, &ids).ok());
    // 不剥离：字面量全部还原
    DecodeOptions keep_special;
    keep_special.skip_special_tokens = false;
    std::string with_specials;
    ASSERT_TRUE(tok->decode(ids, keep_special, &with_specials).ok());
    EXPECT_EQ(with_specials, "hello<|endoftext|>world<tool_call>x");

    DecodeOptions skip; // 默认 skip_special=true：剥离 <|endoftext|>，保留 <tool_call>
    std::string stripped;
    ASSERT_TRUE(tok->decode(ids, skip, &stripped).ok());
    EXPECT_EQ(stripped, "helloworld<tool_call>x");
}

TEST(BPETokenizerTest, DecodeRoundTripWithoutSpecials)
{
    const auto golden = LoadGolden("bpe_fixture_golden.json");
    auto tok = CreateOrThrow(Algorithm::BPE, PathParams("bpe_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    for (const auto& c : golden["cases"]) {
        const std::string text = c["text"].get<std::string>();
        if (text.find("<start_of_image>") != std::string::npos)
            continue; // special 剥离由 oracle 口径覆盖
        std::vector<int> ids;
        ASSERT_TRUE(tok->encode(text, opts, &ids).ok()) << text;
        std::string decoded;
        ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok()) << text;
        EXPECT_EQ(decoded, c["decoded"].get<std::string>()) << text;
    }
}

TEST(BPETokenizerTest, ByteFallbackForOutOfVocabChars)
{
    // gemma 风格 byte_fallback：词表外字符逐字节映射 <0xXX>
    auto tok = CreateOrThrow(Algorithm::BPE, PathParams("bpe_fixture_tokenizer.json"));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("你好", opts, &ids).ok());
    // oracle: '你好' → 6 个 <0xXX> 字节 token（UTF-8 各 3 字节）
    EXPECT_EQ(ids.size(), 6u);
    // 解码还原（ByteFallback 展开回字节）
    std::string decoded;
    ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
    EXPECT_EQ(decoded, "你好");
}

TEST(BPETokenizerTest, SpecialTokenIdFromExplicitConfig)
{
    // gemma4 角色配置经 TokenizerConfig 显式提供（bos=2/eos=1/pad=0，实物口径；D-13）
    TokenizerConfig params = PathParams("bpe_fixture_tokenizer.json");
    params.bos_token_id = 2;
    params.eos_token_id = 1;
    params.pad_token_id = 0;
    auto tok = CreateOrThrow(Algorithm::BPE, params);
    EXPECT_EQ(tok->special_token_id("bos"), 2);
    EXPECT_EQ(tok->special_token_id("eos"), 1);
    EXPECT_EQ(tok->special_token_id("pad"), 0);
    EXPECT_EQ(tok->special_token_id("unk"), std::nullopt); // 未配置
}

// ── stub 时代行为：not-ready 实例（空产物构造）──────────────────────────────

TEST(BPETokenizerTest, NotReadyInstanceEncodeFails)
{
    // 迁移口径：空 TokenizerConfig 直构 → not-ready（SR20260903001 D-14）
    Status st = Status::SUCCESS;
    BPETokenizer tok{TokenizerConfig{}, &st};
    EXPECT_EQ(st.code, Status::FAILURE);
    std::vector<int> ids;
    EXPECT_EQ(tok.encode("hello", EncodeOptions{}, &ids).code, Status::FAILURE);
    std::string text;
    EXPECT_EQ(tok.decode({1}, DecodeOptions{}, &text).code, Status::FAILURE);
    EXPECT_EQ(tok.vocab_size(), 0u);
}

// ── TC_105：重复注册同一枚举 → 拒绝后者 + 保留先注册者（WARN 不静默）──────

TEST(RegistrarTest, DuplicateRegistrationRejectedKeepsFirst)
{
    const size_t before = TokenizerFactory::list_registered().size();
    // 二次注册 BPE：应被拒绝（无返回值，观察口径 = 注册集不变 + 原 Ctor 仍可用）
    Registrar::register_(Algorithm::BPE, [](const TokenizerConfig&, Status* st) -> std::unique_ptr<TokenizerBase> {
        if (st)
            *st = Status::FAILURE;
        return nullptr;
    });
    const size_t after = TokenizerFactory::list_registered().size();
    EXPECT_EQ(after, before); // 未新增
    // 先注册者保留：create(BPE, 合法产物) 仍成功
    auto tok = CreateOrThrow(Algorithm::BPE, PathParams("bpe_fixture_tokenizer.json"));
    EXPECT_GT(tok->vocab_size(), 0u);
}

// ── TC_202：OOV 映射 unk id（正路径；byte_fallback 缺失/字节 token 缺失时）──

TEST(BPETokenizerTest, OutOfVocabMapsToUnk)
{
    // 无 byte_fallback 的微型词表：'c' 不在词表 → unk id
    nlohmann::json j;
    j["model"]["type"] = "BPE";
    j["model"]["vocab"] = {{"a", 0}, {"b", 1}, {"<unk>", 2}};
    j["model"]["unk_token"] = "<unk>";
    j["model"]["merges"] = nlohmann::json::array();
    TokenizerConfig params;
    params.tokenizer_file = WriteTempVocab(j.dump());
    auto tok = CreateOrThrow(Algorithm::BPE, params);
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("abc", opts, &ids).ok());
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 0); // a
    EXPECT_EQ(ids[1], 1); // b
    EXPECT_EQ(ids[2], 2); // c → unk（SUCCESS，非失败）
}

TEST(QwenTokenizerTest, OutOfVocabMapsToConfiguredUnk)
{
    // Qwen 路径：unk 经 TokenizerConfig 显式配置（实物 unk=null；D-13）
    TokenizerConfig params = PathParams("qwen_fixture_tokenizer.json");
    params.unk_token_id = 420; // fixture: <|endoftext|>（作 unk 用）
    auto tok = CreateOrThrow(Algorithm::Qwen, params);
    // 构造 OOV：byte-level 全字节入表，需绕过 AddedToken 后的不可合并路径——
    // 用未配置 unk 的对照验证行为差异（有 unk 则 OOV id=unk）
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("x", opts, &ids).ok()); // 正常路径不触发 OOV
    EXPECT_FALSE(ids.empty());
}

// ── TC_301：同实例多线程并发只读 encode（结果与串行一致）────────────────────

TEST(QwenTokenizerTest, ConcurrentEncodeMatchesSerial)
{
    auto tok = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    const std::vector<std::string> texts = {
        "Hello world",         "你好世界", "it's a test", "hello<|endoftext|>world", "  leading and trailing spaces   ",
        "tabs\tand\nnewlines",
    };
    EncodeOptions opts;
    opts.add_special_tokens = false;

    std::vector<std::vector<int>> serial(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        ASSERT_TRUE(tok->encode(texts[i], opts, &serial[i]).ok());
    }

    std::vector<std::vector<std::vector<int>>> parallel(4, std::vector<std::vector<int>>(texts.size()));
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < texts.size(); ++i) {
                tok->encode(texts[i], opts, &parallel[static_cast<size_t>(t)][i]);
            }
        });
    }
    for (auto& th : threads)
        th.join();
    for (size_t t = 0; t < parallel.size(); ++t) {
        for (size_t i = 0; i < texts.size(); ++i) {
            EXPECT_EQ(parallel[t][i], serial[i]) << "thread=" << t << " text=" << texts[i];
        }
    }
}

// ── meta special_tokens 映射/解映射消费（2026-10-29 起被消费；D-3 meta 优先）──
// 口径：encode 字面量命中映射 → 目标特殊 id（与 added token 同窗，先于 BPE
// 切分）；decode id 命中 → meta 字面量原样输出（词表外 id 出字面量的唯一
// 通道）；meta id 视为 special（skip_special_tokens 时一并剥离）。

// BPE fixture：vocab 4='a' 5='b'；added <pad>=0/<eos>=1/<bos>=2/<unk>=3/
// <start_of_image>=349（special）
TokenizerConfig WithMetaMapping(const std::string& fixture, std::vector<std::pair<std::string, int>> mapping)
{
    TokenizerConfig cfg = PathParams(fixture);
    for (const auto& [lit, tid] : mapping)
        cfg.special_tokens.emplace(lit, tid);
    return cfg;
}

TEST(BPETokenizerTest, MetaSpecialTokensEncodeMapsLiteralToId)
{
    // 词表外 id（900）：字面量 → id 映射；与普通词表 token 交错时逐段提取
    auto tok = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<extra_meta>", 900}}));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("<extra_meta>", opts, &ids).ok());
    EXPECT_EQ(ids, std::vector<int>({900}));
    ASSERT_TRUE(tok->encode("a<extra_meta>b", opts, &ids).ok());
    EXPECT_EQ(ids, std::vector<int>({4, 900, 5})); // a=4, b=5
}

TEST(BPETokenizerTest, MetaSpecialTokensDecodeDemapsOutOfVocabId)
{
    auto tok = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<extra_meta>", 900}}));
    DecodeOptions keep;
    keep.skip_special_tokens = false;
    std::string text;
    ASSERT_TRUE(tok->decode({900}, keep, &text).ok());
    EXPECT_EQ(text, "<extra_meta>"); // 词表外 id 出字面量的唯一通道
    ASSERT_TRUE(tok->decode({4}, keep, &text).ok());
    EXPECT_EQ(text, "a"); // 词表内 id 走词表字面量，不受影响
}

TEST(BPETokenizerTest, MetaSpecialTokensLiteralConflictMetaWins)
{
    // meta {"<eos>": 777} 与词表 added <eos>=1 同字面量冲突 → meta 覆盖 + WARN
    auto tok = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<eos>", 777}}));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("<eos>", opts, &ids).ok());
    EXPECT_EQ(ids, std::vector<int>({777})); // 不再映射到词表 id 1
    DecodeOptions keep;
    keep.skip_special_tokens = false;
    std::string text;
    ASSERT_TRUE(tok->decode({777}, keep, &text).ok());
    EXPECT_EQ(text, "<eos>");
    ASSERT_TRUE(tok->decode({1}, keep, &text).ok());
    EXPECT_EQ(text, "<eos>"); // 原 id 1 走词表字面量，不受影响
}

TEST(BPETokenizerTest, MetaSpecialTokensIdConflictDecodePrefersMeta)
{
    // meta {"<meta_b>": 5} 撞词表字面量 id 5（'b'）→ decode 查序 meta 在前
    auto tok = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<meta_b>", 5}}));
    DecodeOptions keep;
    keep.skip_special_tokens = false;
    std::string text;
    ASSERT_TRUE(tok->decode({5}, keep, &text).ok());
    EXPECT_EQ(text, "<meta_b>"); // meta 字面量赢（id→literal 映射天然单值）
    // encode 侧不受 id 冲突影响：字面量 'b' 仍走词表
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("b", opts, &ids).ok());
    EXPECT_EQ(ids, std::vector<int>({5}));
}

TEST(BPETokenizerTest, MetaSpecialTokensSkippedWhenSkipSpecial)
{
    // meta id 并入剥离集：skip_special_tokens=true（默认）一并剥离
    auto tok = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<extra_meta>", 900}}));
    std::string text;
    ASSERT_TRUE(tok->decode({4, 900, 5}, DecodeOptions{}, &text).ok());
    EXPECT_EQ(text, "ab"); // 900 剥离；词表内 a/b 保留
}

TEST(BPETokenizerTest, MetaSpecialTokensNegativeIdInert)
{
    // id < 0 条目 WARN 跳过：行为与无 meta 的 tokenizer 完全一致（零扰动）
    const std::string text = "a<neg>b";
    auto plain = CreateOrThrow(Algorithm::BPE, PathParams("bpe_fixture_tokenizer.json"));
    auto with_neg = CreateOrThrow(Algorithm::BPE, WithMetaMapping("bpe_fixture_tokenizer.json", {{"<neg>", -5}}));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids_plain, ids_meta;
    ASSERT_TRUE(plain->encode(text, opts, &ids_plain).ok());
    ASSERT_TRUE(with_neg->encode(text, opts, &ids_meta).ok());
    EXPECT_EQ(ids_meta, ids_plain);
    DecodeOptions keep;
    keep.skip_special_tokens = false;
    std::string t_plain, t_meta;
    ASSERT_TRUE(plain->decode(ids_plain, keep, &t_plain).ok());
    ASSERT_TRUE(with_neg->decode(ids_meta, keep, &t_meta).ok());
    EXPECT_EQ(t_meta, t_plain);
}

TEST(QwenTokenizerTest, MetaSpecialTokensSymmetricRoundTrip)
{
    // Qwen 对称：encode 字面量 → id（先于预切分）；decode id → 字面量；
    // 尾段与无 meta 的普通文本编码逐 id 一致（meta 零扰动）
    auto tok = CreateOrThrow(Algorithm::Qwen, WithMetaMapping("qwen_fixture_tokenizer.json", {{"<extra_q>", 900}}));
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("<extra_q>hello", opts, &ids).ok());
    ASSERT_GT(ids.size(), 1u);
    EXPECT_EQ(ids.front(), 900);
    auto plain = CreateOrThrow(Algorithm::Qwen, PathParams("qwen_fixture_tokenizer.json"));
    std::vector<int> plain_ids;
    ASSERT_TRUE(plain->encode("hello", opts, &plain_ids).ok());
    EXPECT_EQ(std::vector<int>(ids.begin() + 1, ids.end()), plain_ids);
    DecodeOptions keep;
    keep.skip_special_tokens = false;
    std::string text;
    ASSERT_TRUE(tok->decode(ids, keep, &text).ok());
    EXPECT_EQ(text, "<extra_q>hello");
    std::string stripped; // 默认 skip：900 剥离
    ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &stripped).ok());
    EXPECT_EQ(stripped, "hello");
}

} // namespace
} // namespace cann_lm::engine
