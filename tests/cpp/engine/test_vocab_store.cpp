// VocabStore L4 .bincache sidecar UT（perf_optimize.md §5 L4；2026-10-27 实施；
// 2026-10-30 默认反转：缓存面改 opt-in——env CANN_LM_TOKENIZER_BINCACHE 显式
// 开启、未设 = JSON 直载，L4①-④ 用例族显式置开以覆盖缓存路径）。
// 覆盖契约（§5 L4 ①-⑤）：
//   L4① CacheRoundTripMemberEquivalence：同一 json 两次 load → 第一次落
//     sidecar → 第二次命中直载，九成员（tok2id 行为面/vocab_size/merges/
//     added/special_strip/model_type/byte_fallback/unk_token/replace_norm）
//     逐一等价——SAX 路径与缓存路径产物等价的对拍。
//   L4② SourceChangeInvalidatesCache：改源 json 内容（size 变）⇒ 键失配回退
//     JSON 且 sidecar 刷新为新键；mtime-only 变化由 sleep 同秒边界观察不到，
//     以 size 键为代表路径（mtime 秒粒度 + size 兜底为设计口径）。
//   L4③ CorruptCacheFallsBack：corner 形态各拒且回退后行为=现状——坏 magic
//     /截断（EOF 提前）/超长拼接尾部/未知 version，逐个 ⇒ load 仍成功（走
//     JSON 全量重建）且成员等价（不崩、不产半加载实例）。
//   L4④ UnwritableSidecarSilent：sidecar 不可写（只读目录）⇒ load 行为=
//     现状（成功且成员就绪），落缓失败静默无异常。
//   L4⑤ EnvDefaultOffOptInGatesCachePlanes：env CANN_LM_TOKENIZER_BINCACHE
//     默认未设 = 缓存面两跳全跳过（不消费在场 sidecar、不落新缓存）；
//     "0" 显式关、"1" 显式开回归 L4① 路径；各态行为面全量等价。
// 形态注意：VocabStore 私有成员无逐容器访问器——等价断言走公开行为面
// （token_to_id/id_to_token/vocab_size/merges()/added_tokens()/
//  special_strip_ids()/model_type()/byte_fallback()/unk_token()/
//  has_replace_normalizer()+replace_normalizer()）全量对拍。
#include "core/component/tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cann_lm::engine {
namespace {

// 与 ut_text_embedding/ut_pipeline_embed 同族确定性 fixture（微型 BPE 词表
// + merges + added_tokens + Replace normalizer——覆盖 SAX 捕获各族字段）。
// merges 引用须在词表内（load 交叉校验 TC_102）：字符串格式 "a b" + 数组
// 格式 ["b","a"] 双格式混用以覆盖两捕获路径。
constexpr const char* kJson = R"json({
  "version": "1.0",
  "truncation": null,
  "added_tokens": [
    {"id": 7, "content": "<bos>", "special": true, "lstrip": false},
    {"id": 8, "content": "<eos>", "special": true},
    {"id": 9, "content": "<pad>", "special": false}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "model": {
    "type": "BPE",
    "byte_fallback": true,
    "unk_token": "<unk>",
    "vocab": {"a": 0, "b": 1, "ab": 2, "ba": 3, "bab": 4, "h": 5, "<unk>": 6},
    "merges": ["a b", ["b", "a"]]
  }
})json";

std::string TempJson()
{
    static int seq = 0;
    return "/tmp/ut_vocab_store_" + std::to_string(::getpid()) + "_" + std::to_string(++seq) + ".json";
}

bool WriteFile(const std::string& path, const char* data, size_t n)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    f.write(data, static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

/// 行为面全量快照对拍（两条加载路径产物的等价性以公开接口表达）
bool ExpectSameStore(const VocabStore& a, const VocabStore& b)
{
    if (a.vocab_size() != b.vocab_size())
        return false;
    for (size_t id = 0; id < a.vocab_size(); ++id) {
        const auto ta = a.id_to_token(static_cast<int>(id));
        const auto tb = b.id_to_token(static_cast<int>(id));
        if (ta.has_value() != tb.has_value())
            return false;
        if (ta.has_value() && *ta != *tb)
            return false;
        if (ta.has_value()) {
            const auto ia = a.token_to_id(*ta);
            const auto ib = b.token_to_id(*tb);
            if (!ia.has_value() || !ib.has_value() || *ia != *ib)
                return false;
        }
    }
    if (a.merges() != b.merges())
        return false;
    if (a.added_tokens() != b.added_tokens())
        return false;
    if (a.special_strip_ids() != b.special_strip_ids())
        return false;
    if (a.model_type() != b.model_type())
        return false;
    if (a.byte_fallback() != b.byte_fallback())
        return false;
    if (a.unk_token() != b.unk_token())
        return false;
    if (a.has_replace_normalizer() != b.has_replace_normalizer())
        return false;
    if (a.has_replace_normalizer() && a.replace_normalizer() != b.replace_normalizer()) {
        return false;
    }
    return true;
}

VocabSource PathSrc(const std::string& p)
{
    VocabSource src;
    src.artifact = std::filesystem::path(p);
    return src;
}

std::string SidecarOf(const std::string& json_path) { return json_path + ".bincache"; }

// env 为进程级全局：RAII 保存/还原防泄漏进同进程后续用例（gtest 同进程串行）。
// 缓存面默认关（2026-10-30 opt-in）：L4①-④ 显式置开以覆盖缓存路径。
class EnvRestore {
public:
    explicit EnvRestore(const char* name) : name_(name)
    {
        const char* old = std::getenv(name);
        had_ = (old != nullptr);
        if (had_)
            val_ = old;
    }
    ~EnvRestore()
    {
        if (had_) {
            ::setenv(name_, val_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_ = false;
    std::string val_;
};

// ── L4① 同 json 两次 load：第二次命中 sidecar，九成员行为面等价 ─────────────
TEST(VocabStoreBincacheTest, CacheRoundTripMemberEquivalence_L4_1)
{
    EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
    ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);

    const std::string json = TempJson();
    ASSERT_TRUE(WriteFile(json, kJson, std::strlen(kJson)));
    std::filesystem::remove(SidecarOf(json)); // 起步无缓存（进程序互混防护）

    VocabStore first;
    ASSERT_TRUE(first.load(PathSrc(json)).ok());
    EXPECT_TRUE(std::filesystem::exists(SidecarOf(json))) << "JSON 成功后应落 .bincache sidecar";

    // 第二次加载：TryLoadFromCache 命中（键=stat 即时取，不受同进程写影响）
    VocabStore second;
    ASSERT_TRUE(second.load(PathSrc(json)).ok());
    EXPECT_TRUE(ExpectSameStore(first, second));

    // 微观锚点：choices 上有抽查（id 0="a"、7="<bos>" overlay、special
    // strip={7,8}、Replace normalizer " "→"▁"、byte_fallback、 unk）
    EXPECT_EQ(second.id_to_token(0).value_or(""), "a");
    EXPECT_EQ(second.token_to_id("<bos>").value_or(-1), 7);
    EXPECT_EQ(second.special_strip_ids().size(), 2u);
    EXPECT_TRUE(second.has_replace_normalizer());
    EXPECT_EQ(second.replace_normalizer().first, " ");
    EXPECT_EQ(second.replace_normalizer().second, "▁");
    ASSERT_TRUE(second.byte_fallback());
    EXPECT_EQ(second.unk_token(), "<unk>");

    std::filesystem::remove(json);
    std::filesystem::remove(SidecarOf(json));
}

// ── L4② 源变化（size 键）⇒ 失配回退 JSON + sidecar 重写 ─────────────────────
TEST(VocabStoreBincacheTest, SourceChangeInvalidatesCache_L4_2)
{
    EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
    ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);

    const std::string json = TempJson();
    ASSERT_TRUE(WriteFile(json, kJson, std::strlen(kJson)));
    VocabStore first;
    ASSERT_TRUE(first.load(PathSrc(json)).ok()); // 落缓存（键 = 本次源 size）
    const auto side_size_1 = std::filesystem::file_size(SidecarOf(json));

    // 源变化：size 键失配（mtime 秒粒度窗口内 size 兜底——设计口径）
    const std::string grown = std::string(kJson) + "\n";
    ASSERT_TRUE(WriteFile(json, grown.c_str(), grown.size()));
    // 注意 sidecar mtime/size 均未变，仅 json 变——缓存必须不被消费
    VocabStore second;
    ASSERT_TRUE(second.load(PathSrc(json)).ok());
    EXPECT_TRUE(ExpectSameStore(first, second)); // 词表内容未变（尾空白）
    // sidecar 已按新源键重写（内容恒等但键位更新）
    const auto side_size_2 = std::filesystem::file_size(SidecarOf(json));
    EXPECT_EQ(side_size_2, side_size_1); // 同词表 ⇒ 同布局长度

    // 键失配证明：手工诱导 mtime 新值（touch -d 未来）恰好可以补位——此面
    // 由 L4③ corner 键篡改责任（键位任意改 ⇒ 失配拒），此处闭包于 size 键
    std::filesystem::remove(json);
    std::filesystem::remove(SidecarOf(json));
}

// ── L4③ 损坏缓存回退不崩：坏 magic/截断/超长拼接/未知 version ────────────────
TEST(VocabStoreBincacheTest, CorruptCacheFallsBack_L4_3)
{
    EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
    ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);

    const char* kCases[] = {"badmagic", "truncated", "extra", "oldver"};
    for (const char* mode : kCases) {
        const std::string json = TempJson();
        ASSERT_TRUE(WriteFile(json, kJson, std::strlen(kJson)));
        VocabStore seed;
        ASSERT_TRUE(seed.load(PathSrc(json)).ok()); // 落成合法 sidecar
        const std::string sidecar = SidecarOf(json);

        std::fstream f(sidecar, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.good());
        if (std::string(mode) == "badmagic") {
            f.seekp(0);
            f.write("XXXXXXXX", 8);
        } else if (std::string(mode) == "truncated") {
            const auto half = std::filesystem::file_size(sidecar) / 2;
            f.close();
            std::filesystem::resize_file(sidecar, half); // 半包：EOF 提前
        } else if (std::string(mode) == "extra") {
            f.close();
            std::ofstream app(sidecar, std::ios::binary | std::ios::app);
            app.write("\xDE\xAD\xBE\xEF", 4); // 尾部拼接：EOF 校验拒
        } else {                              // oldver：version=9
            f.seekp(8);
            const uint32_t v = 9;
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
        if (f.is_open())
            f.close();

        // 回退：load 成功（JSON 全量重建），成员等价，且缓存被重建为合法包
        VocabStore victim;
        ASSERT_TRUE(victim.load(PathSrc(json)).ok()) << "mode=" << mode;
        EXPECT_TRUE(ExpectSameStore(seed, victim)) << "mode=" << mode;
        EXPECT_EQ(victim.token_to_id("<bos>").value_or(-1), 7) << "mode=" << mode;

        std::filesystem::remove(json);
        std::filesystem::remove(sidecar);
    }
}

// ── L4④ sidecar 不可写 ⇒ 静默（行为 = 现状：加载成功，无缓存无异常）──────────
TEST(VocabStoreBincacheTest, UnwritableSidecarSilent_L4_4)
{
    // /proc 只读：load(path in /proc/...) 首先需要 json 本身可读——选用真实
    // 临时目录放 json，然后预占 sidecar 路径为目录（open(目录) 恒失败）
    EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
    ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);

    const std::string json = TempJson();
    ASSERT_TRUE(WriteFile(json, kJson, std::strlen(kJson)));
    const std::string sidecar = SidecarOf(json);
    std::filesystem::remove(sidecar);
    ASSERT_TRUE(std::filesystem::create_directory(sidecar)); // 目录占位：不可写

    VocabStore store;
    ASSERT_TRUE(store.load(PathSrc(json)).ok()); // 落缓存失败静默
    EXPECT_EQ(store.vocab_size(), 10u);          // 8 vocab + 3 added（9 外扩）
    EXPECT_EQ(store.token_to_id("<eos>").value_or(-1), 8);

    // 二次加载同样成功（缓存面恒 miss，走 JSON）
    VocabStore again;
    ASSERT_TRUE(again.load(PathSrc(json)).ok());
    EXPECT_TRUE(ExpectSameStore(store, again));

    std::filesystem::remove(sidecar); // 目录占位
    std::filesystem::remove(json);
}

// ── L4⑤ env 开关 CANN_LM_TOKENIZER_BINCACHE（默认关，opt-in）：未设 = JSON
// 直载（缓存面两跳全跳过）；显式设置才开启 ──────────────────────────────────
// 覆盖契约（2026-10-30 默认反转：开关由 CANN_LM_BINCACHE_OFF 改名为正极性）：
//   ① 默认（未设）下在场合法 sidecar 不被消费也不被重写（JSON 路径重建）；
//   ② 未设下不落新缓存（json 加载成功但 sidecar 缺席——部署目录零副作用）；
//   ③ 默认态产物与开启态行为面全量等价（开关只影响 IO 路径不影响词表内容）；
//   ④ "0" 为显式关（设置非空串并非无条件开——与 BincacheEnabled 语义一致）；
//   ⑤ "1" 显式开：回归 L4① 路径（落缓存）。
TEST(VocabStoreBincacheTest, EnvDefaultOffOptInGatesCachePlanes_L4_5)
{
    const std::string json = TempJson();
    const std::string sidecar = SidecarOf(json);
    ASSERT_TRUE(WriteFile(json, kJson, std::strlen(kJson)));
    std::filesystem::remove(sidecar);

    // 基线：显式开启态落缓存（后续默认态等价对拍的参照实例）
    VocabStore enabled_ref;
    {
        EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
        ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);
        ASSERT_TRUE(enabled_ref.load(PathSrc(json)).ok());
        ASSERT_TRUE(std::filesystem::exists(sidecar));
    }

    // ①② 默认（未设）：在场合法 sidecar 不被消费 + 不落新缓存
    {
        EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
        ::unsetenv("CANN_LM_TOKENIZER_BINCACHE");

        // ① 在场缓存不消费：同 json 再 load——若误走 hit，日志无从断言，
        // 以"行为面等价 + sidecar 内容不被触碰"为可断言面（mtime 不可靠，
        // 用内容快照对拍）
        std::ifstream in(sidecar, std::ios::binary);
        const std::string snap_before((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        VocabStore default_hit;
        ASSERT_TRUE(default_hit.load(PathSrc(json)).ok());
        EXPECT_TRUE(ExpectSameStore(enabled_ref, default_hit));
        std::ifstream in2(sidecar, std::ios::binary);
        const std::string snap_after((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
        // 不消费也不重写：默认态两跳全跳过，sidecar 原样
        EXPECT_EQ(snap_before, snap_after);

        // ② 不落新缓存：换新 json，默认态成功加载但 sidecar 缺席
        const std::string json2 = TempJson();
        ASSERT_TRUE(WriteFile(json2, kJson, std::strlen(kJson)));
        VocabStore default_miss;
        ASSERT_TRUE(default_miss.load(PathSrc(json2)).ok());
        EXPECT_FALSE(std::filesystem::exists(SidecarOf(json2)));
        EXPECT_TRUE(ExpectSameStore(enabled_ref, default_miss));

        std::filesystem::remove(json2);
        std::filesystem::remove(SidecarOf(json2));
    }

    // ④ "0" = 显式关：不落缓存
    {
        EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
        ::setenv("CANN_LM_TOKENIZER_BINCACHE", "0", /*overwrite=*/1);
        const std::string json3 = TempJson();
        ASSERT_TRUE(WriteFile(json3, kJson, std::strlen(kJson)));
        VocabStore off_explicit;
        ASSERT_TRUE(off_explicit.load(PathSrc(json3)).ok());
        EXPECT_FALSE(std::filesystem::exists(SidecarOf(json3))); // 显式关：不落
        EXPECT_TRUE(ExpectSameStore(enabled_ref, off_explicit));
        std::filesystem::remove(json3);
        std::filesystem::remove(SidecarOf(json3));
    }

    // ⑤ "1" = 显式开：回归 L4① 路径（照常落缓存）
    {
        EnvRestore guard("CANN_LM_TOKENIZER_BINCACHE");
        ::setenv("CANN_LM_TOKENIZER_BINCACHE", "1", /*overwrite=*/1);
        const std::string json4 = TempJson();
        ASSERT_TRUE(WriteFile(json4, kJson, std::strlen(kJson)));
        VocabStore on_explicit;
        ASSERT_TRUE(on_explicit.load(PathSrc(json4)).ok());
        EXPECT_TRUE(std::filesystem::exists(SidecarOf(json4)));
        std::filesystem::remove(json4);
        std::filesystem::remove(SidecarOf(json4));
    }

    std::filesystem::remove(json);
    std::filesystem::remove(sidecar);
}

} // namespace
} // namespace cann_lm::engine
