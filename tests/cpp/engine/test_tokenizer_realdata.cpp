// Tokenizer realdata UT（SR20260728001 v0.7，M-8 B 层）。
// 真实产物（tmp/qwen、tmp/gemma 的 tokenizer.json，本地放置、gitignore）驱动的
// 真实规模一致性测试：golden 由 Python oracle（tokenizers 库）预生成入库
// （scripts/gen_tokenizer_fixture.py --real）。TC_205b：加载耗时与 RSS 实测
// 记录（U-3 内存系数依据，无硬断言——预算未定）。
// 本目标由 CMake 检测 tmp/ 实物存在才注册（CI 不依赖网络/产物）。

#include "core/component/tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cann_lm::engine {
namespace {

#ifndef TOKENIZER_REALDATA_DIR
#define TOKENIZER_REALDATA_DIR "tmp"
#endif
#ifndef TOKENIZER_FIXTURE_DIR
#define TOKENIZER_FIXTURE_DIR "tests/fixtures/tokenizer"
#endif

TokenizerConfig RealParams(const std::string& rel)
{ // SR20260903001 D-14
    TokenizerConfig cfg;
    cfg.tokenizer_file = std::string(TOKENIZER_REALDATA_DIR) + "/" + rel;
    return cfg;
}

nlohmann::json LoadGolden(const std::string& rel)
{
    std::ifstream f(std::string(TOKENIZER_FIXTURE_DIR) + "/" + rel);
    EXPECT_TRUE(f.good()) << "golden missing: " << rel << "（重跑 scripts/gen_tokenizer_fixture.py --real）";
    nlohmann::json j;
    f >> j;
    return j;
}

// RSS（KB，aarch64 Linux /proc/self/statm 第 2 列）
long ResidentKb()
{
    std::ifstream f("/proc/self/statm");
    unsigned long size = 0, resident = 0;
    f >> size >> resident;
    return static_cast<long>(resident * 4); // 页大小 4KB（aarch64 常规配置）
}

// ── Qwen 真实产物（12.8MB，vocab 248070）────────────────────────────────────

TEST(QwenRealDataTest, GoldenIdsMatchOracleRealArtifact)
{
    // TC_203 B 层 + TC_001：真实规模逐 id 一致；vocab_size 实物口径 248070
    const auto golden = LoadGolden("qwen_real_golden.json");
    const long rss_before = ResidentKb();
    const auto t0 = std::chrono::steady_clock::now();

    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::Qwen, RealParams("qwen/tokenizer.json"), &tok);
    ASSERT_TRUE(st.ok()) << "code=" << st.code;

    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
                             .count();
    const long rss_after = ResidentKb();
    // TC_205b：加载耗时与 RSS 实测记录（U-3 依据，预算未定无硬断言）
    std::printf("[realdata] qwen load: %lldms, RSS delta: %ldKB (vocab=%zu)\n", static_cast<long long>(load_ms),
                rss_after - rss_before, tok->vocab_size());

    EXPECT_EQ(tok->vocab_size(), 248070u); // base 248044 + added 26（≠config 248320）
    EXPECT_EQ(tok->vocab_size(), golden["vocab_size"].get<size_t>());

    EncodeOptions opts;
    opts.add_special_tokens = false; // TC_203 口径：裸流水线逐 id 比对
    for (const auto& c : golden["cases"]) {
        const std::string text = c["text"].get<std::string>();
        std::vector<int> ids;
        ASSERT_TRUE(tok->encode(text, opts, &ids).ok()) << text;
        EXPECT_EQ(ids, c["ids"].get<std::vector<int>>()) << "text=" << text;
        std::string decoded;
        ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
        EXPECT_EQ(decoded, c["decoded"].get<std::string>()) << "text=" << text;
    }
}

TEST(QwenRealDataTest, AddedTokenSpecialsOnRealArtifact)
{
    // 真实 26 added_tokens（14 special）的提取与剥离
    std::unique_ptr<TokenizerBase> tok;
    ASSERT_TRUE(TokenizerFactory::create(Algorithm::Qwen, RealParams("qwen/tokenizer.json"), &tok).ok());
    EncodeOptions opts;
    opts.add_special_tokens = false;
    std::vector<int> ids;
    ASSERT_TRUE(tok->encode("hello <|im_end|> world", opts, &ids).ok());
    // oracle: ['hello', ' ', '<|im_end|>', ' ', 'world'] —— <|im_end|> 为单一 id
    ASSERT_GE(ids.size(), 3u);
    std::string decoded;
    ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
    EXPECT_EQ(decoded, "hello  world"); // <|im_end|> 被剥离，空格保留
}

// ── gemma4 真实产物（32MB，vocab 262144）────────────────────────────────────

TEST(GemmaRealDataTest, GoldenIdsMatchOracleRealArtifact)
{
    // TC_203 B 层 + TC_001b/205：真实规模逐 id 一致；vocab 262144（model.type=BPE 已验）
    const auto golden = LoadGolden("gemma_real_golden.json");
    const long rss_before = ResidentKb();
    const auto t0 = std::chrono::steady_clock::now();

    std::unique_ptr<TokenizerBase> tok;
    Status st = TokenizerFactory::create(Algorithm::BPE, RealParams("gemma/tokenizer.json"), &tok);
    ASSERT_TRUE(st.ok()) << "code=" << st.code;

    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
                             .count();
    const long rss_after = ResidentKb();
    std::printf("[realdata] gemma load: %lldms, RSS delta: %ldKB (vocab=%zu)\n", static_cast<long long>(load_ms),
                rss_after - rss_before, tok->vocab_size());

    EXPECT_EQ(tok->vocab_size(), 262144u);
    EXPECT_EQ(tok->vocab_size(), golden["vocab_size"].get<size_t>());

    EncodeOptions opts;
    opts.add_special_tokens = false;
    for (const auto& c : golden["cases"]) {
        const std::string text = c["text"].get<std::string>();
        std::vector<int> ids;
        ASSERT_TRUE(tok->encode(text, opts, &ids).ok()) << text;
        EXPECT_EQ(ids, c["ids"].get<std::vector<int>>()) << "text=" << text;
        std::string decoded;
        ASSERT_TRUE(tok->decode(ids, DecodeOptions{}, &decoded).ok());
        EXPECT_EQ(decoded, c["decoded"].get<std::string>()) << "text=" << text;
    }
}

} // namespace
} // namespace cann_lm::engine
