// ═══════════════════════════════════════════════════════════════════════════
// TextEmbedding 单元测试（SR20260828001，覆盖 SPEC 第 8 章 TC_001~TC_113）。
// host 侧三件事：加载（两模式/两粒度）、查表（行 + scale）、填工位（两路径）。
// NPU 侧反量化不在 UT 范围（后端依赖）。
// mock 工位 bundle 经 CreateTensorBuffer 按 desc 构造（桩后端无副作用，走真实单例）。
// gemma4 identity 用例（TC_401~TC_405 + TC_406）按 Q12 合并文件布局
// （docs/gemma4_token_identity.md：identity 段后随主表段、同文件同 load_mode）。
// 2026-10-09 设备事故修正适配：InMemory 表/scale 一律堆内存（unique_ptr<char[]>），
// TensorBuffer 仅用于模型输入/输出工位（本文件无"NNTensor 存表"截获 stub）；
// 随契约删除的 Table()/Scale() 访问器断言改为行为级——装载形态以
// LoadMode()/TableType() 表达，数据驻留以"Load 后截断盘上文件仍可
// Gather/FillInputs"证明（InFlash 表留文件的直读证据由 TC_111 承担）。
// 工位别名组（TC_501~TC_511）：问题 1 方案 b（2026-10-10）——om 侧实名
// embeddings 作 input_embed 的别名（规范名优先、别名兜底；Four 消费点：
// FillInputs 两路径 + FindEmbedSlot/IsEmbedSlotName 助手 + debug inject）。
// 2026-10-14 床证扩组（TC_508~TC_511）：scale 工位别名 embeddings_scale /
// per_layer_token_identity_scale（c42 新转换器正拼）。identity scale 工位名自
// 2026-10-19 拼写契约迁移起收口该正拼单名（Q1/Q8 契约拼写工位不再探测）。
// ═══════════════════════════════════════════════════════════════════════════
#include "core/component/text_embedding/text_embedding.h"
#include "utils/data_type_utils.h" // fp16_t：宿主无 __fp16 保证（x86_64 GCC≤11），软件位级半精度

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace cann_lm::engine {
namespace {

constexpr size_t kVocab = 16;
constexpr size_t kHidden = 8;
constexpr size_t kCount = 3; // ids = {3, 0, 15}
constexpr size_t kWidth = 4; // identity 行宽（刻意 ≠ kHidden，防两表混用）

/// 确定性 INT8 表值：行 v 列 h → (v*7 + h*3 - 50)，落在 int8 范围。
int8_t TableValue(size_t v, size_t h)
{
    return static_cast<int8_t>(static_cast<int>(v) * 7 + static_cast<int>(h) * 3 - 50);
}

/// 确定性 INT8 identity 表值：行 v 列 w → (v*3 + w - 10)（gemma4 identity 用）。
int8_t IdentityValue(size_t v, size_t w)
{
    return static_cast<int8_t>(static_cast<int>(v) * 3 + static_cast<int>(w) - 10);
}

/// 确定性 Fp32 identity scale（PerToken）：行 v → (0.25 + 0.5*v)。
float IdentityScale(size_t v) { return 0.25f + 0.5f * static_cast<float>(v); }

std::string TempPath(const char* tag)
{
    static std::atomic<int> seq{0};
    return "/tmp/ut_text_embedding_" + std::to_string(::getpid()) + "_" + tag + "_" + std::to_string(seq.fetch_add(1)) +
           ".bin";
}

bool WriteFile(const std::string& path, const void* data, size_t n)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<size_t>(f.tellp()) == n;
}

bool RemoveFile(const std::string& path) { return ::remove(path.c_str()) == 0; }

/// 经 C-04（CreateTensorBuffer 代理）构造 mock 工位 buffer
/// （NDTensorDesc 现接口：tensorName + vector<int32_t> dims，无独立 size——
/// 尺寸由 dims×dtype 派生，与真 NNRt 协议一致）。
std::shared_ptr<ITensorBuffer> MakeBuf(const std::string& name, DataType dt, std::vector<int64_t> dims)
{
    TensorDesc d;
    d.tensorName = name;
    d.dataType = dt;
    d.format = Format::ND;
    d.dims.assign(dims.begin(), dims.end());
    std::shared_ptr<ITensorBuffer> out;
    auto st = TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &out);
    return st.ok() ? out : nullptr;
}

/// INT8 主表 [16,8] + Fp32 PerToken 主 scale [16] 的 Q12 合并文件 fixture
/// （docs/gemma4_token_identity.md §3）：identity 段 INT8 [16,4] 紧随主表段
/// （表文件 192B = 128 + 64，identity 段偏移 = vocab×hidden×主表元素），identity
/// scale 段 Fp32 [16] 紧随主 scale 段（scale 文件 128B = 64 + 64）。
class TextEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        table_.resize(kVocab * kHidden);
        for (size_t v = 0; v < kVocab; ++v)
            for (size_t h = 0; h < kHidden; ++h)
                table_[v * kHidden + h] = TableValue(v, h);
        for (size_t v = 0; v < kVocab; ++v)
            scale_[v] = 0.5f + static_cast<float>(v);

        // identity 段数据（并入主表文件后段——Q12；裸 dump 行优先）
        identity_.resize(kVocab * kWidth);
        for (size_t v = 0; v < kVocab; ++v)
            for (size_t w = 0; w < kWidth; ++w)
                identity_[v * kWidth + w] = IdentityValue(v, w);
        for (size_t v = 0; v < kVocab; ++v)
            identity_scale_[v] = IdentityScale(v);

        merged_table_.reserve(table_.size() + identity_.size());
        merged_table_.insert(merged_table_.end(), table_.begin(), table_.end());
        merged_table_.insert(merged_table_.end(), identity_.begin(), identity_.end());
        merged_scale_.reserve(scale_.size() + identity_scale_.size());
        merged_scale_.insert(merged_scale_.end(), scale_.begin(), scale_.end());
        merged_scale_.insert(merged_scale_.end(), identity_scale_.begin(), identity_scale_.end());

        table_path_ = TempPath("table");
        scale_path_ = TempPath("scale");
        ASSERT_TRUE(WriteFile(table_path_, merged_table_.data(), merged_table_.size()));
        ASSERT_TRUE(WriteFile(scale_path_, merged_scale_.data(), merged_scale_.size() * sizeof(float)));
    }

    void TearDown() override
    {
        RemoveFile(table_path_);
        RemoveFile(scale_path_);
    }

    EmbeddingConfig Int8PerTokenConfig(EmbeddingLoadMode mode) const
    {
        EmbeddingConfig c;
        c.table_path = table_path_;
        c.scale_path = scale_path_;
        c.table_type = EmbeddingType::Int8;
        c.scale_type = EmbeddingScaleType::Fp32;
        c.quant_mode = EmbeddingQuantMode::PerToken;
        c.vocab_size = kVocab;
        c.hidden_dim = kHidden;
        c.load_mode = mode;
        return c;
    }

    /// gemma4 per_layer_token_identity 配置（Q12 合并布局）：identity 段位于
    /// table_path_ 主表段之后（偏移 = vocab×hidden×主表元素——主表字段必填，
    /// 经 Int8PerTokenConfig 继承），scale 段位于 scale 文件主 scale 段之后；
    /// load_mode 同时管辖主表与 identity 源（同文件两种模式的组合不复存在）。
    EmbeddingConfig IdentityConfig(EmbeddingLoadMode mode) const
    {
        EmbeddingConfig c = Int8PerTokenConfig(mode);
        c.token_identity_type = EmbeddingType::Int8;
        c.token_identity_scale_type = EmbeddingScaleType::Fp32;
        return c;
    }

    /// Embedding 路径四工位填位 + 逐值断言（TC_403/TC_403b 共用：既有 embed/scale
    /// 两工位照填 + identity/scale 落位；调用前须已 Load 主表与 identity）。
    /// 参数化目标对象（TC_206 移动赋值后对被移入对象复用本断言）。
    void FillEmbeddingPathAndExpectAllSlots(TextEmbedding& e)
    {
        const TokenId ids[kCount] = {3, 0, 15};
        const size_t idx[kCount] = {3, 0, 15};
        // 主表工位 INT8 同型直拷；identity INT8 → FLOAT32 工位展开；scale Fp32 → FLOAT32
        auto slot_e = MakeBuf("input_embed", DataType::INT8, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
        auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
        std::unordered_map<std::string, size_t> name_index{
            {"input_embed", 0}, {"embed_scales", 1}, {kTokenIdentitySlotName, 2}, {kTokenIdentityScaleSlotName, 3}};
        ASSERT_TRUE(e.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

        const auto* emb_rows = static_cast<const int8_t*>(slot_e->GetData());
        for (size_t i = 0; i < kCount; ++i)
            for (size_t h = 0; h < kHidden; ++h)
                EXPECT_EQ(emb_rows[i * kHidden + h], TableValue(idx[i], h));
        const auto* emb_sc = static_cast<const float*>(slot_s->GetData());
        for (size_t i = 0; i < kCount; ++i)
            EXPECT_FLOAT_EQ(emb_sc[i], 0.5f + idx[i]);
        const auto* rows = static_cast<const float*>(ident->GetData());
        for (size_t i = 0; i < kCount; ++i)
            for (size_t w = 0; w < kWidth; ++w)
                EXPECT_FLOAT_EQ(rows[i * kWidth + w], static_cast<float>(IdentityValue(idx[i], w)));
        const auto* sc = static_cast<const float*>(iscale->GetData());
        for (size_t i = 0; i < kCount; ++i)
            EXPECT_FLOAT_EQ(sc[i], IdentityScale(idx[i]));
    }

    std::vector<int8_t> table_ = std::vector<int8_t>(kVocab * kHidden);
    std::vector<float> scale_ = std::vector<float>(kVocab);
    std::vector<int8_t> identity_ = std::vector<int8_t>(kVocab * kWidth);
    std::vector<float> identity_scale_ = std::vector<float>(kVocab);
    std::vector<int8_t> merged_table_; // 主表段 + identity 段（即磁盘布局）
    std::vector<float> merged_scale_;  // 主 scale 段 + identity scale 段
    std::string table_path_, scale_path_;
    TextEmbedding emb_;
};

// ── 正常路径 TC_001~TC_010 ──────────────────────────────────────────────────

// TC_001：PerToken InMemory 加载成功。2026-10-09 契约变化：InMemory 表/scale
// 改存堆内存，原 Table()/Scale() 访问器已删——存在性/尺寸断言以行为级等价替。
// 2026-10-28 L2 再变更：InMemory 表载体重写为 mmap 只读映射（原"装载后截断
// 盘上 fixture 证明驻堆"与 MAP_PRIVATE 载体互斥——截断收缩 inode 后访问
// 越尾页 SIGBUS，且"数据不依赖文件"对 mmap 载体本就不成立）；驻留证明
// 改为 Gather 全词表逐字节命中（映射建立即数据就绪），scale 段仍堆载体。
TEST_F(TextEmbeddingTest, LoadPerTokenInMemorySuccess_TC001)
{
    auto s = emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory));
    ASSERT_TRUE(s.ok()) << s.code;
    EXPECT_TRUE(emb_.Loaded());
    EXPECT_EQ(emb_.TableType(), EmbeddingType::Int8);
    EXPECT_EQ(emb_.LoadMode(), EmbeddingLoadMode::InMemory);

    // 行为级（替原 Table()/Scale() 非 null + GetSize）：全 16 行 + 全 16 scale
    // 逐字节对 fixture，兼证主表段/scale 段两端与全量消费
    std::vector<TokenId> ids(kVocab);
    for (size_t v = 0; v < kVocab; ++v)
        ids[v] = static_cast<TokenId>(v);
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kVocab, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kVocab});
    ASSERT_TRUE(emb_.Gather(ids.data(), kVocab, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t v = 0; v < kVocab; ++v) {
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[v * kHidden + h], TableValue(v, h)) << "v=" << v << " h=" << h;
        EXPECT_FLOAT_EQ(sc[v], 0.5f + static_cast<float>(v));
    }
}

// TC_002：PerGroup（group_size=2，groups=4）加载成功。2026-10-09 契约变化：
// 原 Scale()->GetSize()==vocab*groups*4B（256B 全量消费）断言以行为级等价替——
// Gather 末词表项收取末 4 组 scale + 末行 8B，均与 fixture 末段逐字节一致
// （末段取到即 scale 文件全量读入堆内存；PerGroup 查表口径另见 TC_004）。
TEST_F(TextEmbeddingTest, LoadPerGroupSuccess_TC002)
{
    std::vector<float> gs(kVocab * 4);
    for (size_t i = 0; i < gs.size(); ++i)
        gs[i] = 0.25f * static_cast<float>(i);
    const std::string spath = TempPath("gscale");
    ASSERT_TRUE(WriteFile(spath, gs.data(), gs.size() * 4));

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_path = spath;
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 2;
    auto s = emb_.Load(c);
    ASSERT_TRUE(s.ok()) << s.code;
    EXPECT_TRUE(emb_.Loaded());

    const TokenId last = static_cast<TokenId>(kVocab - 1);
    auto out_e = MakeBuf("out_embed", DataType::INT8, {1, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {4});
    ASSERT_TRUE(emb_.Gather(&last, 1, out_e, out_s).ok());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t g = 0; g < 4; ++g)
        EXPECT_FLOAT_EQ(sc[g], gs[(kVocab - 1) * 4 + g]) << "g=" << g;
    const auto* row = static_cast<const int8_t*>(out_e->GetData());
    for (size_t h = 0; h < kHidden; ++h)
        EXPECT_EQ(row[h], TableValue(kVocab - 1, h)) << "h=" << h;
    RemoveFile(spath);
}

// TC_003：PerToken gather 逐字节正确 + scale 收集正确
TEST_F(TextEmbeddingTest, GatherPerTokenCorrectness_TC003)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    ASSERT_TRUE(emb_.Gather(ids, kCount, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const size_t idx[kCount] = {3, 0, 15};
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h)) << "i=" << i << " h=" << h;
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_004：PerGroup gather 的 scale 收集（ids={5} → 下标 5×4+0..3）
TEST_F(TextEmbeddingTest, GatherPerGroupScales_TC004)
{
    std::vector<float> gs(kVocab * 4);
    for (size_t i = 0; i < gs.size(); ++i)
        gs[i] = 0.25f * static_cast<float>(i);
    const std::string spath = TempPath("gscale");
    ASSERT_TRUE(WriteFile(spath, gs.data(), gs.size() * 4));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_path = spath;
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 2;
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId id = 5;
    auto out_e = MakeBuf("out_embed", DataType::INT8, {1, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {4});
    ASSERT_TRUE(emb_.Gather(&id, 1, out_e, out_s).ok());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t g = 0; g < 4; ++g)
        EXPECT_FLOAT_EQ(sc[g], gs[5 * 4 + g]);
    RemoveFile(spath);
}

// TC_005：Embedding 路径 FillInputs（INT8 存储 → FLOAT16 工位展开 + scale → FLOAT32）
TEST_F(TextEmbeddingTest, FillInputsEmbeddingPath_TC005)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
    auto s = emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount);
    ASSERT_TRUE(s.ok()) << s.code;

    const auto* rows = static_cast<const fp16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(static_cast<float>(rows[i * kHidden + h]), static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_006：TokenId 路径无需 Load（INT32 工位）
TEST_F(TextEmbeddingTest, FillInputsTokenIdNoLoad_TC006)
{
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::INT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    auto s = emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount);
    ASSERT_TRUE(s.ok()) << s.code;
    const auto* v = static_cast<const int32_t*>(slot_e->GetData());
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 15);
}

// TC_010：TokenId 路径 INT64 工位
TEST_F(TextEmbeddingTest, FillInputsTokenIdInt64_TC010)
{
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::INT64, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount).ok());
    const auto* v = static_cast<const int64_t*>(slot_e->GetData());
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 15);
}

// TC_007：同型免转换直填（Fp16 存储 → FLOAT16 工位，逐字节相等）
TEST_F(TextEmbeddingTest, SameTypeDirectCopy_TC007)
{
    std::vector<fp16_t> t(kVocab * kHidden);
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = fp16_t(static_cast<float>(i) * 0.5f - 8.0f);
    const std::string tpath = TempPath("fp16table");
    ASSERT_TRUE(WriteFile(tpath, t.data(), t.size() * 2));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = tpath;
    c.table_type = EmbeddingType::Fp16;
    c.scale_type = EmbeddingScaleType::None;
    c.scale_path.clear();
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
    const auto* rows = static_cast<const fp16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h].bits_, t[idx[i] * kHidden + h].bits_);
    RemoveFile(tpath);
}

// TC_007b：Fp32 主表同型直填（FLOAT32 工位）+ Fp32 scale → FLOAT16 工位软件
// 转换（转换矩阵补全：Fp32 运算面此前未覆盖；fixture 值取 1/8 整倍数，fp16 精确）
TEST_F(TextEmbeddingTest, Fp32TableAndScaleConversions_TC007b)
{
    std::vector<float> t(kVocab * kHidden);
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = 0.125f * static_cast<float>(i) - 8.0f;
    std::vector<float> s(kVocab);
    for (size_t v = 0; v < kVocab; ++v)
        s[v] = 0.25f + 0.125f * static_cast<float>(v);
    const std::string tpath = TempPath("fp32table");
    const std::string spath = TempPath("fp32scale");
    ASSERT_TRUE(WriteFile(tpath, t.data(), t.size() * 4));
    ASSERT_TRUE(WriteFile(spath, s.data(), s.size() * 4));

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = tpath;
    c.scale_path = spath;
    c.table_type = EmbeddingType::Fp32; // scale_type 已是 Fp32，quant PerToken
    ASSERT_TRUE(emb_.Load(c).ok());
    EXPECT_EQ(emb_.TableType(), EmbeddingType::Fp32);

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT32, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT16, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

    const auto* rows = static_cast<const float*>(slot_e->GetData()); // Fp32 同型 memcpy
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(rows[i * kHidden + h], t[idx[i] * kHidden + h]) << "i=" << i << " h=" << h;
    const auto* sc = static_cast<const fp16_t*>(slot_s->GetData()); // Fp32→F16 位级
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_EQ(sc[i].bits_, fp16_t(s[idx[i]]).bits_) << "i=" << i;
    RemoveFile(tpath);
    RemoveFile(spath);
}

// TC_007c：Fp16 主表 → FLOAT32 工位值展开 + Fp16 scale 双工位类型（FLOAT32
// 展开 / FLOAT16 同型直拷）——转换矩阵补全（此前仅同型 FLOAT16 一路，TC_007）
TEST_F(TextEmbeddingTest, Fp16TableAndScaleConversions_TC007c)
{
    std::vector<fp16_t> t(kVocab * kHidden);
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = fp16_t(0.125f * static_cast<float>(i % 32) - 2.0f);
    std::vector<fp16_t> s(kVocab);
    for (size_t v = 0; v < kVocab; ++v)
        s[v] = fp16_t(0.25f + 0.125f * static_cast<float>(v));
    const std::string tpath = TempPath("fp16table2");
    const std::string spath = TempPath("fp16scale");
    ASSERT_TRUE(WriteFile(tpath, t.data(), t.size() * 2));
    ASSERT_TRUE(WriteFile(spath, s.data(), s.size() * 2));

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = tpath;
    c.scale_path = spath;
    c.table_type = EmbeddingType::Fp16;
    c.scale_type = EmbeddingScaleType::Fp16;
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    // 一：embed FLOAT32（Fp16→F32 值展开）+ scale FLOAT32（Fp16→F32 值展开）
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT32, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
        std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
        ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
        const auto* rows = static_cast<const float*>(slot_e->GetData());
        for (size_t i = 0; i < kCount; ++i)
            for (size_t h = 0; h < kHidden; ++h)
                EXPECT_FLOAT_EQ(rows[i * kHidden + h], static_cast<float>(t[idx[i] * kHidden + h]))
                    << "i=" << i << " h=" << h;
        const auto* sc = static_cast<const float*>(slot_s->GetData());
        for (size_t i = 0; i < kCount; ++i)
            EXPECT_FLOAT_EQ(sc[i], static_cast<float>(s[idx[i]])) << "i=" << i;
    }
    // 二：embed FLOAT16（同型）+ scale FLOAT16（同型直拷，位级相等）
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT16, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
        std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
        ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
        const auto* rows = static_cast<const fp16_t*>(slot_e->GetData());
        for (size_t i = 0; i < kCount; ++i)
            for (size_t h = 0; h < kHidden; ++h)
                EXPECT_EQ(rows[i * kHidden + h].bits_, t[idx[i] * kHidden + h].bits_) << "i=" << i << " h=" << h;
        const auto* sc = static_cast<const fp16_t*>(slot_s->GetData());
        for (size_t i = 0; i < kCount; ++i)
            EXPECT_EQ(sc[i].bits_, s[idx[i]].bits_) << "i=" << i;
    }
    RemoveFile(tpath);
    RemoveFile(spath);
}

// TC_008：InFlash 加载与 gather（与 InMemory 逐字节一致）。2026-10-09 契约变化：
// 原 Table()==nullptr（无整表内存映像）/Scale()!=nullptr（scale 恒内存态）断言
// 以行为级等价替——前者由 LoadMode()==InFlash 表达 + TC_111（装载后截断表文件
// Gather → FAILURE）承担直读证据；后者以"表文件完好、仅截断 scale 文件后
// Gather 仍成功且 scale 逐值正确"证明 D-09（scale 于 Load 期读入堆内存）。
TEST_F(TextEmbeddingTest, InFlashLoadGather_TC008)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    EXPECT_TRUE(emb_.Loaded());
    EXPECT_EQ(emb_.LoadMode(), EmbeddingLoadMode::InFlash);

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    ASSERT_TRUE(emb_.Gather(ids, kCount, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h));
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);

    // 行为级（替原 Scale()!=nullptr）：仅废掉盘上 scale 文件，表行仍走 fd pread，
    // Gather 照常成功且行/scale 均逐值正确
    ASSERT_EQ(0, ::truncate(scale_path_.c_str(), 0));
    auto out_e2 = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s2 = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    ASSERT_TRUE(emb_.Gather(ids, kCount, out_e2, out_s2).ok());
    const auto* rows2 = static_cast<const int8_t*>(out_e2->GetData());
    const auto* sc2 = static_cast<const float*>(out_s2->GetData());
    for (size_t i = 0; i < kCount; ++i) {
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows2[i * kHidden + h], TableValue(idx[i], h));
        EXPECT_FLOAT_EQ(sc2[i], 0.5f + idx[i]);
    }
}

// TC_009：InFlash 模式 FillInputs（与 TC_005 InMemory 同配置逐字节一致）
TEST_F(TextEmbeddingTest, InFlashFillInputs_TC009)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
    const size_t idx[kCount] = {3, 0, 15};
    const auto* rows = static_cast<const fp16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(static_cast<float>(rows[i * kHidden + h]), static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// ── 异常路径 TC_101~TC_113 ──────────────────────────────────────────────────

// TC_101：表文件缺失 → INVALID_ARGUMENT，可重 Load
TEST_F(TextEmbeddingTest, MissingTableFile_TC101)
{
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = "/nonexistent/dir/table.bin";
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.Loaded());
    EXPECT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok()); // 可恢复
}

// TC_101b：参数与文件缺失防御——参数校验先于文件系统（vocab=0/hidden=0/
// 空 path）；InFlash 表缺失；主 scale 缺失；LoadTokenIdentity scale 段缺失。
// 全部保持未加载态（与 TC_101 单一"InMemory 表缺失"口径互补）。
TEST_F(TextEmbeddingTest, InvalidArgsAndMissingFiles_TC101b)
{
    const auto c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    auto bad_vocab = c;
    bad_vocab.vocab_size = 0;
    EXPECT_EQ(emb_.Load(bad_vocab).code, Status::INVALID_ARGUMENT);
    auto bad_hidden = c;
    bad_hidden.hidden_dim = 0;
    EXPECT_EQ(emb_.Load(bad_hidden).code, Status::INVALID_ARGUMENT);
    auto bad_path = c;
    bad_path.table_path.clear();
    EXPECT_EQ(emb_.Load(bad_path).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.Loaded());

    // InFlash 表文件缺失（TC_101 仅 InMemory 侧扫）
    auto flash_missing = c;
    flash_missing.load_mode = EmbeddingLoadMode::InFlash;
    flash_missing.table_path = "/nonexistent/dir/table.bin";
    EXPECT_EQ(emb_.Load(flash_missing).code, Status::INVALID_ARGUMENT);

    // 主 scale 文件缺失（表文件本身合法）→ 拒且保持未加载态
    auto no_scale = c;
    no_scale.scale_path = "/nonexistent/dir/scale.bin";
    EXPECT_EQ(emb_.Load(no_scale).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.Loaded());

    // LoadTokenIdentity：主 scale 文件缺失（scale 段无处读取）
    auto ident_ns = IdentityConfig(EmbeddingLoadMode::InMemory);
    ident_ns.scale_path = "/nonexistent/dir/iscale.bin";
    EXPECT_EQ(emb_.LoadTokenIdentity(ident_ns, kWidth).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.IdentityLoaded());
}

// TC_102：主表段过短 → INVALID_ARGUMENT；超长放行（Q12 合并文件布局——
// identity 段后随主表段属常态，不再以整文件长校验，"主段完整即可"）
TEST_F(TextEmbeddingTest, FileSizeMismatch_TC102)
{
    // 过短：主表段差 1 字节（128B - 1）
    const std::string short_path = TempPath("short");
    ASSERT_TRUE(WriteFile(short_path, merged_table_.data(), table_.size() - 1));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = short_path;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);

    // 超长一：合并文件本体（identity 段 64B 后随）→ 放行
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    EXPECT_TRUE(emb_.Loaded());

    // 超长二：尾字节另加 → 仍放行
    const std::string long_path = TempPath("long");
    std::vector<int8_t> extra = merged_table_;
    extra.push_back(0);
    ASSERT_TRUE(WriteFile(long_path, extra.data(), extra.size()));
    c.table_path = long_path;
    ASSERT_TRUE(emb_.Load(c).ok());
    // 超长文件 Gather 仅读主段（首行、尾行各验一次——尾部行不越过后段）
    const TokenId edges[2] = {0, static_cast<TokenId>(kVocab - 1)};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {2, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {2});
    ASSERT_TRUE(emb_.Gather(edges, 2, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    for (size_t h = 0; h < kHidden; ++h)
        EXPECT_EQ(rows[h], TableValue(0, h));
    for (size_t h = 0; h < kHidden; ++h)
        EXPECT_EQ(rows[kHidden + h], TableValue(kVocab - 1, h));

    // InFlash 侧：fstat < 主段长 → 拒；≥ 主段长（后段后随/尾字节）→ 放行
    c.table_path = short_path;
    c.load_mode = EmbeddingLoadMode::InFlash;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    c.table_path = long_path;
    EXPECT_TRUE(emb_.Load(c).ok());
    // 主 scale 文件同口径（超长 = scale 段后随，放行——TC_001 已隐式覆盖；短读拒换路径验一次）
    EmbeddingConfig cs = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    const std::string short_scale = TempPath("shortscale");
    ASSERT_TRUE(WriteFile(short_scale, merged_scale_.data(), scale_.size() * sizeof(float) - 1));
    cs.scale_path = short_scale;
    EXPECT_EQ(emb_.Load(cs).code, Status::INVALID_ARGUMENT);

    RemoveFile(short_path);
    RemoveFile(long_path);
    RemoveFile(short_scale);
}

// TC_103：PerGroup 参数非法（group_size=0 / 不整除 hidden）
TEST_F(TextEmbeddingTest, PerGroupInvalidParams_TC103)
{
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 0;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    c.group_size = 3; // 8 % 3 != 0
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    c.group_size = kHidden + 1;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
}

// TC_104：scale 配置自相矛盾（None 且 path 非空）
TEST_F(TextEmbeddingTest, ScaleConfigContradiction_TC104)
{
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_type = EmbeddingScaleType::None;
    // scale_path 非空 → INVALID_ARGUMENT
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
}

// TC_105：tokenId 越界 → 整批拒绝，out 不被部分写入
TEST_F(TextEmbeddingTest, TokenIdOutOfRangeRejectsBatch_TC105)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 16, 1}; // 16 ≥ vocab
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    std::memset(out_e->GetData(), 0xAB, out_e->GetSize());
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);
    const auto* p = static_cast<const uint8_t*>(out_e->GetData());
    for (size_t i = 0; i < out_e->GetSize(); ++i)
        EXPECT_EQ(p[i], 0xAB); // 未写入
}

// TC_106：未加载先查（Gather / Embedding 路径 FillInputs → INVALID_STATE；TokenId 豁免）
TEST_F(TextEmbeddingTest, NotLoadedRejectsQueries_TC106)
{
    const TokenId ids[kCount] = {1, 2, 3};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_STATE);

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code, Status::INVALID_STATE);
}

// TC_107：工位缺名（input_embed / embed_scales）→ NOT_FOUND
TEST_F(TextEmbeddingTest, MissingSlotName_TC107)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> empty_index{};
    EXPECT_EQ(emb_.FillInputs(inputs, empty_index, EmbeddingPath::Embedding, ids, kCount).code, Status::NOT_FOUND);

    // input_embed 有、embed_scales 缺（scale_type ≠ None）→ NOT_FOUND
    std::unordered_map<std::string, size_t> only_embed{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, only_embed, EmbeddingPath::Embedding, ids, kCount).code, Status::NOT_FOUND);
}

// TC_108：不支持的转换组合（Int4 存储 / Fp32→FLOAT16 工位）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, UnsupportedConversion_TC108)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto slot_e = MakeBuf("input_embed", DataType::INT64, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_ARGUMENT); // Int8 → INT64 超矩阵

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_type = EmbeddingType::Int4;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT); // U-3
}

// TC_109：输出容量不足 → INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, InsufficientOutputCapacity_TC109)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    // 容量不足经 dims 缺行构造（尺寸由 dims×dtype 派生：3×7=21 < 3×8=24）
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden - 1});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);
}

// TC_109b：容量不足与工位类型不支持全域拒绝（TC_109 仅扫 Gather embed 侧）：
// Gather out_scales 容量不足；TokenId 路径 FLOAT32 工位/INT32 容量不足；
// Embedding 路径 embed 工位容量不足、embed_scales 类型非法/容量不足（本用例
// 不加载 identity，主表口径；identity 工位防御见 TC_405b）。
TEST_F(TextEmbeddingTest, InsufficientCapacityAndSlotTypes_TC109b)
{
    const TokenId ids[kCount] = {1, 2, 3};
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());

    // Gather：out_scales 容量不足（PerToken 每 token 1 float，少 1）
    auto out_e_f = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s_f = MakeBuf("out_scales", DataType::FLOAT32, {kCount - 1});
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e_f, out_s_f).code, Status::INVALID_ARGUMENT);

    // FillInputs TokenId：FLOAT32 工位（仅 INT32/INT64）→ 拒
    {
        auto tid = MakeBuf("input_embed", DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{tid};
        std::unordered_map<std::string, size_t> ni{{"input_embed", 0}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::TokenId, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
    // TokenId：INT32 工位容量不足 → 拒
    {
        auto tid = MakeBuf("input_embed", DataType::INT32, {kCount - 1});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{tid};
        std::unordered_map<std::string, size_t> ni{{"input_embed", 0}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::TokenId, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
    // Embedding：embed 工位容量不足（FLOAT16 dims 缺行）→ 拒
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden - 1});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
        std::unordered_map<std::string, size_t> ni{{"input_embed", 0}, {"embed_scales", 1}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::Embedding, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
    // Embedding：embed_scales 类型非法（INT8 工位）→ 拒
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::INT8, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
        std::unordered_map<std::string, size_t> ni{{"input_embed", 0}, {"embed_scales", 1}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::Embedding, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
    // Embedding：embed_scales 容量不足（FLOAT16 dims {kCount-1}）→ 拒
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT16, {kCount - 1});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
        std::unordered_map<std::string, size_t> ni{{"input_embed", 0}, {"embed_scales", 1}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::Embedding, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
}

// TC_110：NULL 入参（ids=nullptr 且 count>0）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, NullIdsRejected_TC110)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(nullptr, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, nullptr, kCount).code,
              Status::INVALID_ARGUMENT);
}

// TC_111：InFlash 运行期文件损坏（截断）→ FAILURE
TEST_F(TextEmbeddingTest, InFlashRuntimeCorruption_TC111)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    ASSERT_EQ(0, ::truncate(table_path_.c_str(),
                            static_cast<off_t>(table_.size() / 2))); // Load 后截断
    const TokenId ids[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::FAILURE);
}

// TC_112：空 buffer 入参（Gather 空 shared_ptr / FillInputs 含空 buffer）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, NullBufferRejected_TC112)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(ids, kCount, nullptr, out_s).code, Status::INVALID_ARGUMENT);

    std::vector<std::shared_ptr<ITensorBuffer>> inputs{nullptr}; // 工位 buffer 为空
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_ARGUMENT);
}

// TC_113：FromSection 未知枚举字符串 / 空 weights（不加载信号）——SR20260903001 D-12
// （identity 两读取类型直参为空串 = 缺省友好；主表空与之无关）
TEST(FromSectionTest, UnknownStringAndEmptyWeights_TC113)
{
    EmbeddingConfig out;
    // 空 weights → SUCCESS + 不加载信号
    EXPECT_TRUE(EmbeddingConfig::FromSection("", "", "", "", "", "", "", "", 16, 8, &out).ok());
    EXPECT_TRUE(out.table_path.empty());
    // 未知 table_type → INVALID_ARGUMENT
    EXPECT_EQ(EmbeddingConfig::FromSection("/tmp/table.bin", "", "fp8", "", "", "", "", "", 16, 8, &out).code,
              Status::INVALID_ARGUMENT);
    // nullptr out → INVALID_ARGUMENT
    EXPECT_EQ(EmbeddingConfig::FromSection("/tmp/table.bin", "", "fp8", "", "", "", "", "", 16, 8, nullptr).code,
              Status::INVALID_ARGUMENT);
}

// TC_113b：FromSection 主表枚举解析变体（fp32/fp16/int4、per_group、load_mode
// 缺省）与未知枚举三维度拒绝（scale_type/quant_mode/load_mode——TC_113 仅覆盖
// table_type 维度）。per_group 无组大小组段 → group_size 恒 0（Load 期拒绝，TC_209）
TEST(FromSectionTest, ParsesVariantsAndRejectsUnknownEnums_TC113b)
{
    EmbeddingConfig out;
    // 合法变体一：fp32 / fp32 scale / per_group / in_memory
    ASSERT_TRUE(
        EmbeddingConfig::FromSection("t.bin", "s.bin", "fp32", "fp32", "per_group", "in_memory", "", "", 8, 4, &out)
            .ok());
    EXPECT_EQ(out.table_type, EmbeddingType::Fp32);
    EXPECT_EQ(out.scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(out.quant_mode, EmbeddingQuantMode::PerGroup);
    EXPECT_EQ(out.load_mode, EmbeddingLoadMode::InMemory);
    EXPECT_EQ(out.group_size, 0u);
    // 合法变体二：fp16 / scale 空缺省 None / quant 空缺省 PerToken
    ASSERT_TRUE(EmbeddingConfig::FromSection("t2.bin", "", "fp16", "", "", "", "", "", 2, 2, &out).ok());
    EXPECT_EQ(out.table_type, EmbeddingType::Fp16);
    EXPECT_EQ(out.scale_type, EmbeddingScaleType::None);
    EXPECT_EQ(out.quant_mode, EmbeddingQuantMode::PerToken);
    // int4 可解析（Load 期拒绝——U-3 口径）
    ASSERT_TRUE(EmbeddingConfig::FromSection("t3.bin", "", "int4", "", "", "", "", "", 2, 2, &out).ok());
    EXPECT_EQ(out.table_type, EmbeddingType::Int4);
    // 未知枚举：scale_type / quant_mode / load_mode → INVALID_ARGUMENT（段降级）
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "s.bin", "fp32", "fp64", "", "", "", "", 8, 4, &out).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "s.bin", "fp32", "fp32", "per_x", "", "", "", 8, 4, &out).code,
              Status::INVALID_ARGUMENT);
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "s.bin", "fp32", "fp32", "", "fast", "", "", 8, 4, &out).code,
              Status::INVALID_ARGUMENT);
}

// FromSection 合法解析（全字段；per_group 无组大小 → Load 期拒绝口径维持，TC_209）
TEST(FromSectionTest, ParsesAllFields)
{
    EmbeddingConfig out;
    ASSERT_TRUE(EmbeddingConfig::FromSection("t.bin", "s.bin", "int8", "fp16", "per_token", "in_flash", "fp32", "fp16",
                                             100, 64, &out)
                    .ok());
    EXPECT_EQ(out.table_type, EmbeddingType::Int8);
    EXPECT_EQ(out.scale_type, EmbeddingScaleType::Fp16);
    EXPECT_EQ(out.quant_mode, EmbeddingQuantMode::PerToken);
    EXPECT_EQ(out.load_mode, EmbeddingLoadMode::InFlash);
    EXPECT_EQ(out.vocab_size, 100u);
    EXPECT_EQ(out.hidden_dim, 64u);
}

// TC_406：FromSection identity 读取类型键解析（gemma4，docs/gemma4_token_identity.md §7；
// Q12 合并布局——仅解析读取类型，路径随独立文件删除）
TEST(FromSectionTest, TokenIdentityFields_TC406)
{
    EmbeddingConfig out;
    // 未配置（两类型直参空）→ 结构缺省（Q10：type 缺省 fp16，scale_type 缺省 fp16）
    ASSERT_TRUE(EmbeddingConfig::FromSection("", "", "", "", "", "", "", "", 16, 8, &out).ok());
    EXPECT_EQ(out.token_identity_type, EmbeddingType::Fp16);
    EXPECT_EQ(out.token_identity_scale_type, EmbeddingScaleType::Fp16);

    // 核心语义：主表空（TokenId 部署形态）+ identity 类型键显式 → 读取类型照常解析
    // （配置层独立；host 加载/查表由 pipeline 按 Loaded() 门控跳过——Q11/Q12）
    ASSERT_TRUE(EmbeddingConfig::FromSection("", "", "", "", "", "", "int8", "fp32", 16, 8, &out).ok());
    EXPECT_TRUE(out.table_path.empty());
    EXPECT_EQ(out.token_identity_type, EmbeddingType::Int8);
    EXPECT_EQ(out.token_identity_scale_type, EmbeddingScaleType::Fp32);

    // 全键显式（int8 identity + fp32 scale）；主表不受 identity 键影响
    ASSERT_TRUE(EmbeddingConfig::FromSection("t.bin", "s.bin", "int8", "fp32", "per_token", "in_memory", "int8", "fp32",
                                             100, 64, &out)
                    .ok());
    EXPECT_EQ(out.token_identity_type, EmbeddingType::Int8);
    EXPECT_EQ(out.token_identity_scale_type, EmbeddingScaleType::Fp32);
    EXPECT_EQ(out.table_path, "t.bin");

    // scale_type "none" → 非法（Q12："none" 不再表达"不启用"——启用信号已随
    // 独立路径键删除，改为图工位判定 + pipeline 依图回填 None）
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "", "int8", "", "", "", "", "none", 16, 8, &out).code,
              Status::INVALID_ARGUMENT);
    // 未知 identity type → INVALID_ARGUMENT（段降级）
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "", "int8", "", "", "", "fp8", "", 16, 8, &out).code,
              Status::INVALID_ARGUMENT);
    // 未知 identity scale_type → INVALID_ARGUMENT
    EXPECT_EQ(EmbeddingConfig::FromSection("t.bin", "", "int8", "", "", "", "", "fp64", 16, 8, &out).code,
              Status::INVALID_ARGUMENT);
    // "int4" 可解析（LoadTokenIdentity 期拒绝——U-3 口径）
    ASSERT_TRUE(EmbeddingConfig::FromSection("t.bin", "", "int8", "", "", "", "int4", "", 16, 8, &out).ok());
    EXPECT_EQ(out.token_identity_type, EmbeddingType::Int4);
}

// ── 边界 TC_201~TC_204 ──────────────────────────────────────────────────────

// TC_201：count=0 → SUCCESS，不写任何字节
TEST_F(TextEmbeddingTest, CountZeroWritesNothing_TC201)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    std::memset(out_e->GetData(), 0xCD, out_e->GetSize());
    EXPECT_TRUE(emb_.Gather(nullptr, 0, out_e, out_s).ok());
    const auto* p = static_cast<const uint8_t*>(out_e->GetData());
    for (size_t i = 0; i < out_e->GetSize(); ++i)
        EXPECT_EQ(p[i], 0xCD);
}

// TC_202：scale_type=None（无 scale 文件；Gather 跳过 out_scales；FillInputs
// 跳过 embed_scales）。2026-10-09 契约变化：原 Scale()==nullptr 断言以行为级
// 等价替——"None = 无 scale 数据"双证：①非空 out_scales 照常成功且哨兵一字节
// 不变（跳过写入，非清零）；②out_scales=nullptr 亦成功（None 豁免容量校验）。
TEST_F(TextEmbeddingTest, ScaleNoneSkipsScales_TC202)
{
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_type = EmbeddingScaleType::None;
    c.scale_path.clear();
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId ids[kCount] = {1, 2, 3};
    const size_t idx[kCount] = {1, 2, 3};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    std::memset(out_e->GetData(), 0x77, out_e->GetSize());
    std::memset(out_s->GetData(), 0x5A, out_s->GetSize());
    EXPECT_TRUE(emb_.Gather(ids, kCount, out_e, out_s).ok());
    // None → out_scales 整块不被触碰；out_e 行照写（查表确已执行，非空成功）
    const auto* ps = static_cast<const uint8_t*>(out_s->GetData());
    for (size_t i = 0; i < out_s->GetSize(); ++i)
        EXPECT_EQ(ps[i], 0x5A);
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h)) << "i=" << i << " h=" << h;
    // nullptr out_scales 同样成功（scale_type=None 时跳过写入——原路径保留）
    EXPECT_TRUE(emb_.Gather(ids, kCount, out_e, /*out_scales=*/nullptr).ok());

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}}; // 无 embed_scales 亦成功
    EXPECT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
}

// TC_203：vocab 边界 id（vocab-1 成功、vocab 拒绝）
TEST_F(TextEmbeddingTest, VocabBoundaryId_TC203)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ok_id = static_cast<TokenId>(kVocab - 1);
    const TokenId bad_id = static_cast<TokenId>(kVocab);
    auto out_e = MakeBuf("out_embed", DataType::INT8, {1, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {1});
    EXPECT_TRUE(emb_.Gather(&ok_id, 1, out_e, out_s).ok());
    EXPECT_EQ(emb_.Gather(&bad_id, 1, out_e, out_s).code, Status::INVALID_ARGUMENT);
}

// TC_204：失败 Load 后修正配置再 Load 成功
TEST_F(TextEmbeddingTest, ReloadAfterFailure_TC204)
{
    EmbeddingConfig bad = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    bad.vocab_size = 99; // 尺寸校验必败
    EXPECT_FALSE(emb_.Load(bad).ok());
    EXPECT_FALSE(emb_.Loaded());
    EXPECT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    EXPECT_TRUE(emb_.Loaded());
}

// TC_205：移动构造移交加载态，原对象置未加载态（DESIGN M-01 §3；表源/scale
// 堆内存与 identity 第二表源状态一并移交）。行为级：移入方能全量 Gather，
// 移出方转 INVALID_STATE。
TEST_F(TextEmbeddingTest, MoveCtorTransfersLoadedState_TC205)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    TextEmbedding moved(std::move(emb_)); // 移动构造
    EXPECT_FALSE(emb_.Loaded());          // 原对象置未加载态
    EXPECT_FALSE(emb_.IdentityLoaded());
    EXPECT_TRUE(moved.Loaded());
    EXPECT_TRUE(moved.IdentityLoaded());
    EXPECT_EQ(moved.TableType(), EmbeddingType::Int8);
    EXPECT_EQ(moved.LoadMode(), EmbeddingLoadMode::InMemory);

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_STATE);
    ASSERT_TRUE(moved.Gather(ids, kCount, out_e, out_s).ok()); // 数据全量随移动移交
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t i = 0; i < kCount; ++i) {
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h)) << "i=" << i << " h=" << h;
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
    }
}

// TC_206：移动赋值移交（InFlash fd 所有权 + identity 状态随源移交）——移入方
// 移动后 FillInputs 四工位落位仍逐字节正确（fd 未因移交失效）
TEST_F(TextEmbeddingTest, MoveAssignTransfersFdBackedState_TC206)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InFlash), kWidth).ok());
    TextEmbedding assigned;
    EXPECT_FALSE(assigned.Loaded());
    assigned = std::move(emb_); // 移动赋值（fd 所有权随源移交）
    EXPECT_FALSE(emb_.Loaded());
    EXPECT_FALSE(emb_.IdentityLoaded());
    EXPECT_TRUE(assigned.Loaded()); // InFlash fd 移交后仍可读
    EXPECT_TRUE(assigned.IdentityLoaded());
    EXPECT_EQ(assigned.LoadMode(), EmbeddingLoadMode::InFlash);
    FillEmbeddingPathAndExpectAllSlots(assigned); // 四工位落位逐值正确
}

// ── 并发 TC_301~TC_302 ──────────────────────────────────────────────────────

// TC_301：4 线程并发 Gather（InMemory），结果与串行一致
TEST_F(TextEmbeddingTest, ConcurrentGatherInMemory_TC301)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
            auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
            for (int iter = 0; iter < 50; ++iter) {
                if (!emb_.Gather(ids, kCount, out_e, out_s).ok()) {
                    mismatches++;
                    continue;
                }
                const auto* rows = static_cast<const int8_t*>(out_e->GetData());
                const auto* sc = static_cast<const float*>(out_s->GetData());
                for (size_t i = 0; i < kCount; ++i) {
                    for (size_t h = 0; h < kHidden; ++h)
                        if (rows[i * kHidden + h] != TableValue(idx[i], h))
                            mismatches++;
                    if (sc[i] != 0.5f + idx[i])
                        mismatches++;
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    EXPECT_EQ(mismatches.load(), 0);
}

// TC_302：4 线程共享 InFlash fd 并发 Gather，互不干扰
TEST_F(TextEmbeddingTest, ConcurrentGatherInFlash_TC302)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden});
            auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount});
            for (int iter = 0; iter < 50; ++iter) {
                if (!emb_.Gather(ids, kCount, out_e, out_s).ok()) {
                    mismatches++;
                    continue;
                }
                const auto* rows = static_cast<const int8_t*>(out_e->GetData());
                const auto* sc = static_cast<const float*>(out_s->GetData());
                for (size_t i = 0; i < kCount; ++i) {
                    for (size_t h = 0; h < kHidden; ++h)
                        if (rows[i * kHidden + h] != TableValue(idx[i], h))
                            mismatches++;
                    if (sc[i] != 0.5f + idx[i])
                        mismatches++;
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    EXPECT_EQ(mismatches.load(), 0);
}

// ── gemma4 per_layer_token_identity TC_401~TC_405（docs/gemma4_token_identity.md §7；
//    Q12 合并文件布局：identity 段后随主表段、scale 段后随主 scale 段）──

/// 预置 identity 已加载态 + 四工位输入 bundle（ids 槽类型 int32）。
struct IdentitySlots {
    std::vector<std::shared_ptr<ITensorBuffer>> inputs;
    std::unordered_map<std::string, size_t> name_index;
};

IdentitySlots MakeIdentitySlots(DataType ids_type, DataType ident_type, DataType scale_type)
{
    IdentitySlots s;
    auto ids = MakeBuf("input_embed", ids_type, {kCount});
    auto ident = MakeBuf(kTokenIdentitySlotName, ident_type, {kCount, kWidth});
    auto iscale = MakeBuf(kTokenIdentityScaleSlotName, scale_type, {kCount});
    s.inputs = {ids, ident, iscale};
    s.name_index = {{"input_embed", 0}, {kTokenIdentitySlotName, 1}, {kTokenIdentityScaleSlotName, 2}};
    return s;
}

// TC_401：LoadTokenIdentity 参数/合并文件段校验（Q12——identity 段偏移可计算、
// 启用时表文件总长须精确 = 主表段 + identity 段，短/超长均拒；scale 段同法）；
// 失败保持 identity 未加载态
TEST_F(TextEmbeddingTest, TokenIdentityLoadValidation_TC401)
{
    const auto c = IdentityConfig(EmbeddingLoadMode::InMemory);

    // 参数校验：row_width=0 / vocab=0 / hidden_dim=0（偏移计算需要）/ 表文件缺失
    EXPECT_EQ(emb_.LoadTokenIdentity(c, 0).code, Status::INVALID_ARGUMENT);
    auto bad_vocab = c;
    bad_vocab.vocab_size = 0;
    EXPECT_EQ(emb_.LoadTokenIdentity(bad_vocab, kWidth).code, Status::INVALID_ARGUMENT);
    auto bad_hidden = c;
    bad_hidden.hidden_dim = 0;
    EXPECT_EQ(emb_.LoadTokenIdentity(bad_hidden, kWidth).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.IdentityLoaded());
    auto missing = c;
    missing.table_path = "/nonexistent/dir/merged.bin";
    EXPECT_EQ(emb_.LoadTokenIdentity(missing, kWidth).code, Status::INVALID_ARGUMENT);

    // 合并文件 identity 段短 1 字节（192-1）：InMemory 段短读拒绝
    const std::string short_path = TempPath("merged_short");
    ASSERT_TRUE(WriteFile(short_path, merged_table_.data(), merged_table_.size() - 1));
    auto shortc = c;
    shortc.table_path = short_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(shortc, kWidth).code, Status::INVALID_ARGUMENT);

    // 合并文件尾多 1 字节（192+1）→ 总长超长拒绝
    const std::string long_path = TempPath("merged_long");
    std::vector<int8_t> tailed = merged_table_;
    tailed.push_back(0);
    ASSERT_TRUE(WriteFile(long_path, tailed.data(), tailed.size()));
    auto longc = c;
    longc.table_path = long_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(longc, kWidth).code, Status::INVALID_ARGUMENT);

    // Int4（U-3）
    auto int4 = c;
    int4.token_identity_type = EmbeddingType::Int4;
    EXPECT_EQ(emb_.LoadTokenIdentity(int4, kWidth).code, Status::INVALID_ARGUMENT);

    // 图带 scale 工位（want_scale）而主 scale 缺失 → 无处读取（防御；
    // pipeline InitEmbedding_ 依图校验应已致命——对齐 TC_104 口径）
    auto no_scale = c;
    no_scale.scale_type = EmbeddingScaleType::None;
    no_scale.scale_path.clear();
    EXPECT_EQ(emb_.LoadTokenIdentity(no_scale, kWidth).code, Status::INVALID_ARGUMENT);

    // scale 段短 1 字节（scale 文件 128-1）/ 超长 1 字节（128+1）→ 均拒
    const std::string scale_short_path = TempPath("iscale_short");
    ASSERT_TRUE(WriteFile(scale_short_path, merged_scale_.data(), merged_scale_.size() * sizeof(float) - 1));
    auto sc_short = c;
    sc_short.scale_path = scale_short_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(sc_short, kWidth).code, Status::INVALID_ARGUMENT);
    const std::string scale_long_path = TempPath("iscale_long");
    std::vector<float> scale_tailed = merged_scale_;
    scale_tailed.push_back(0.f);
    ASSERT_TRUE(WriteFile(scale_long_path, scale_tailed.data(), scale_tailed.size() * sizeof(float)));
    auto sc_long = c;
    sc_long.scale_path = scale_long_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(sc_long, kWidth).code, Status::INVALID_ARGUMENT);

    EXPECT_FALSE(emb_.IdentityLoaded()); // 此前所有失败均保持未加载态

    // 成功（fixture 正身 = 精确合并布局）+ IdentityLoaded；主表状态不受影响（第二表源独立）
    ASSERT_TRUE(emb_.LoadTokenIdentity(c, kWidth).ok());
    EXPECT_TRUE(emb_.IdentityLoaded());
    EXPECT_FALSE(emb_.Loaded());

    // InFlash 侧：fstat 总长须精确（短/超长均拒）
    const auto flash = IdentityConfig(EmbeddingLoadMode::InFlash);
    auto flash_short = flash;
    flash_short.table_path = short_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(flash_short, kWidth).code, Status::INVALID_ARGUMENT);
    auto flash_long = flash;
    flash_long.table_path = long_path;
    EXPECT_EQ(emb_.LoadTokenIdentity(flash_long, kWidth).code, Status::INVALID_ARGUMENT);

    RemoveFile(short_path);
    RemoveFile(long_path);
    RemoveFile(scale_short_path);
    RemoveFile(scale_long_path);
}

// TC_402：TokenId 部署形态不查表（2026-09-22 变更）——identity 已加载也零触碰
TEST_F(TextEmbeddingTest, TokenIdentitySkippedOnTokenIdPath_TC402)
{
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    EXPECT_FALSE(emb_.Loaded()); // TokenId 路径分派（表在模型权重中）

    const TokenId ids[kCount] = {3, 0, 15};
    auto s = MakeIdentitySlots(DataType::INT32, DataType::FLOAT32, DataType::FLOAT32);
    std::memset(s.inputs[1]->GetData(), 0xAB, s.inputs[1]->GetSize());
    std::memset(s.inputs[2]->GetData(), 0xAB, s.inputs[2]->GetSize());
    EXPECT_TRUE(emb_.FillInputs(s.inputs, s.name_index, EmbeddingPath::TokenId, ids, kCount).ok());

    // input_ids 照常落位；identity/scale 保持原值（TokenId 形态由图内自理）
    const auto* v = static_cast<const int32_t*>(s.inputs[0]->GetData());
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 15);
    const auto* p1 = static_cast<const uint8_t*>(s.inputs[1]->GetData());
    for (size_t i = 0; i < s.inputs[1]->GetSize(); ++i)
        EXPECT_EQ(p1[i], 0xAB);
    const auto* p2 = static_cast<const uint8_t*>(s.inputs[2]->GetData());
    for (size_t i = 0; i < s.inputs[2]->GetSize(); ++i)
        EXPECT_EQ(p2[i], 0xAB);
}

// TC_403：Embedding 路径既有工位照填 + identity 两工位同时落位（identity 唯一执行路径；
// Q12 后主表与 identity 同文件同 load_mode——InMemory 全链）
TEST_F(TextEmbeddingTest, TokenIdentityFilledOnEmbeddingPath_TC403)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    EXPECT_TRUE(emb_.Loaded());
    EXPECT_TRUE(emb_.IdentityLoaded());
    FillEmbeddingPathAndExpectAllSlots(emb_);
}

// TC_403b：identity 源 InFlash（Q12 文件内段偏移——行读取基数 = 主表段 128B，
// 不从文件头起）；主表 InFlash 同批成立，两 FlashTableSource 各按段基址读行，
// 落位值与 InMemory（TC_403）逐字节一致。2026-10-09 契约变化：原 Table()==nullptr
// 断言删——主表侧"留文件直读"加载形态以 LoadMode()==InFlash 表达，数据正确性由
// 四工位落位逐值校验承担（直读行为证据在 TC_111）。
TEST_F(TextEmbeddingTest, TokenIdentityInFlashSegmentBase_TC403b)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InFlash), kWidth).ok());
    EXPECT_TRUE(emb_.Loaded());
    EXPECT_TRUE(emb_.IdentityLoaded());
    EXPECT_EQ(emb_.LoadMode(), EmbeddingLoadMode::InFlash);
    FillEmbeddingPathAndExpectAllSlots(emb_);
}

// TC_404：Embedding 路径 identity ids 越界 → 整批拒绝（TC_105 口径）；
// identity 落位于 embed 填充前执行，identity/scale/embed 工位均不被污染
TEST_F(TextEmbeddingTest, TokenIdentityOutOfRangeRejectsBatch_TC404)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    const TokenId ids[kCount] = {3, 16, 1}; // 16 ≥ vocab
    auto slot_e = MakeBuf("input_embed", DataType::INT8, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
    auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
    std::unordered_map<std::string, size_t> name_index{
        {"input_embed", 0}, {"embed_scales", 1}, {kTokenIdentitySlotName, 2}, {kTokenIdentityScaleSlotName, 3}};
    std::memset(ident->GetData(), 0xAB, ident->GetSize());
    std::memset(iscale->GetData(), 0xAB, iscale->GetSize());
    std::memset(slot_e->GetData(), 0xAB, slot_e->GetSize());
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_ARGUMENT);
    const auto* p1 = static_cast<const uint8_t*>(ident->GetData());
    for (size_t i = 0; i < ident->GetSize(); ++i)
        EXPECT_EQ(p1[i], 0xAB);
    const auto* p2 = static_cast<const uint8_t*>(iscale->GetData());
    for (size_t i = 0; i < iscale->GetSize(); ++i)
        EXPECT_EQ(p2[i], 0xAB);
    const auto* pe = static_cast<const uint8_t*>(slot_e->GetData());
    for (size_t i = 0; i < slot_e->GetSize(); ++i)
        EXPECT_EQ(pe[i], 0xAB);
}

// TC_405：图不带 identity 工位（旧图）→ NOT_FOUND；identity 未启用（未 Load）时零行为
TEST_F(TextEmbeddingTest, TokenIdentitySlotMissing_TC405)
{
    // 未启用：identity 未加载 → 不触碰 identity 逻辑（SUCCESS）
    {
        const TokenId ids[kCount] = {1, 2, 3};
        auto slot_e = MakeBuf("input_embed", DataType::INT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
        std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
        EXPECT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount).ok());
    }
    // 启用（含主表，Embedding 形态）但图缺 per_layer_token_identity 工位 → NOT_FOUND；
    // TokenId 路径不触碰 identity（TC_402 已覆盖），此口径只在 Embedding 路径可达
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    {
        const TokenId ids[kCount] = {1, 2, 3};
        auto slot_e = MakeBuf("input_embed", DataType::INT8, {kCount, kHidden});
        auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32,
                              {kCount}); // 带 scale 工位、缺 identity 工位
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, iscale};
        std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {kTokenIdentityScaleSlotName, 1}};
        EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code, Status::NOT_FOUND);
    }
}

// TC_405b：identity/scale 工位容量与形态防御（图带工位而容量不足/空 buffer；
// TC_405 扫"工位缺名"，本用例扫"工位在但不合格"）：identity 容量不足、
// scale 容量不足、scale 缺名（NOT_FOUND）、scale 空 buffer（INVALID_ARGUMENT）
TEST_F(TextEmbeddingTest, IdentitySlotCapacityAndMissing_TC405b)
{
    const TokenId ids[kCount] = {1, 2, 3};
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    const std::unordered_map<std::string, size_t> ni_all{
        {"input_embed", 0}, {"embed_scales", 1}, {kTokenIdentitySlotName, 2}, {kTokenIdentityScaleSlotName, 3}};

    // identity 工位容量不足（dims 缺行 {kCount, kWidth-1}）→ 拒
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth - 1});
        auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
        EXPECT_EQ(emb_.FillInputs(inputs, ni_all, EmbeddingPath::Embedding, ids, kCount).code,
                  Status::INVALID_ARGUMENT);
    }
    // scale 工位容量不足（{kCount-1}）→ 拒
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
        auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount - 1});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
        EXPECT_EQ(emb_.FillInputs(inputs, ni_all, EmbeddingPath::Embedding, ids, kCount).code,
                  Status::INVALID_ARGUMENT);
    }
    // scale 工位缺名（identity 工位照常）→ NOT_FOUND（写零字节，查找先于写入）
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident};
        const std::unordered_map<std::string, size_t> ni_no_scale{
            {"input_embed", 0}, {"embed_scales", 1}, {kTokenIdentitySlotName, 2}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni_no_scale, EmbeddingPath::Embedding, ids, kCount).code, Status::NOT_FOUND);
    }
    // scale 工位在 name_index 而槽位为空 buffer → INVALID_ARGUMENT
    {
        auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden});
        auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
        auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, nullptr};
        EXPECT_EQ(emb_.FillInputs(inputs, ni_all, EmbeddingPath::Embedding, ids, kCount).code,
                  Status::INVALID_ARGUMENT);
    }
}

// ── 工位别名（问题 1 方案 b，2026-10-10 + scale 别名 2026-10-14）TC_501~TC_511 ─
// omc 常见导出名 embeddings 兜底规范名 input_embed（docs/failed_problem.md
// 问题 1；日志实证 [6] name=embeddings, FLOAT16 [1,1,128,1536]）。规范名优先、
// 别名兜底；两者并存取规范名；均缺 → 各消费点按原有口径报错。
// 2026-10-14 床证补全（TC_508~TC_511）：omc 新转换器（1014 gemma4 c42）将
// scale 工位导出正拼实名 embeddings_scale / per_layer_token_identity_scale。
// 2026-10-19 拼写契约迁移：identity scale 工位名收口为正拼
// per_layer_token_identity_scale（老转换器 …sacle 拼写工位不再探测）；embed 侧
// 双名（embeddings/embeddings_scale）不受影响。

// TC_501：FindEmbedSlot 解析语义（四消费点共用的工位定位助手）——规范名优先；
// 规范名缺、别名在 → 兜底命中；两名并存取规范名；均缺 → end()（各消费点
// 自行按口径报错，助手不吞错误）。
TEST_F(TextEmbeddingTest, FindEmbedSlotResolutionSemantics_TC501)
{
    using NameIndex = std::unordered_map<std::string, std::size_t>;

    const NameIndex alias_only{{kEmbedSlotAlias, 3}};
    auto it = FindEmbedSlot(alias_only);
    ASSERT_NE(it, alias_only.end());
    EXPECT_EQ(it->second, 3u);

    const NameIndex canonical_only{{kEmbedSlotName, 7}};
    it = FindEmbedSlot(canonical_only);
    ASSERT_NE(it, canonical_only.end());
    EXPECT_EQ(it->second, 7u);

    // 两名并存取规范名（优先级是名字级判定）
    const NameIndex both{{kEmbedSlotName, 7}, {kEmbedSlotAlias, 3}};
    it = FindEmbedSlot(both);
    ASSERT_NE(it, both.end());
    EXPECT_EQ(it->first, std::string(kEmbedSlotName));
    EXPECT_EQ(it->second, 7u);

    // 均缺 → end()（空表与纯非工位名同口径）
    const NameIndex empty_index{};
    EXPECT_EQ(FindEmbedSlot(empty_index), empty_index.end());
    const NameIndex unrelated{{"position_ids", 0}};
    EXPECT_EQ(FindEmbedSlot(unrelated), unrelated.end());
}

// TC_502：IsEmbedSlotName 谓词语义（model_executor 维度修正按名遍历的判定
// 基础）——规范名/别名命中；近似名、其他工位名、空串不命中。
TEST_F(TextEmbeddingTest, IsEmbedSlotNameMatchesCanonicalAndAlias_TC502)
{
    EXPECT_TRUE(IsEmbedSlotName(kEmbedSlotName));
    EXPECT_TRUE(IsEmbedSlotName(kEmbedSlotAlias));
    EXPECT_FALSE(IsEmbedSlotName("input_ids"));
    EXPECT_FALSE(IsEmbedSlotName("input_embeds")); // 近似名不命中
    EXPECT_FALSE(IsEmbedSlotName("embedding"));    // 单数不命中
    EXPECT_FALSE(IsEmbedSlotName(""));
}

// TC_503：TokenId 路径别名解析——name_index 仅含 "embeddings" → 照常落位。
// 事故正向镜像：图工位实名 embeddings 时 prefill FillInputs 不再 NOT_FOUND(4)。
TEST_F(TextEmbeddingTest, FillInputsTokenIdViaAlias_TC503)
{
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::INT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{kEmbedSlotAlias, 0}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount).ok());

    const auto* v = static_cast<const int32_t*>(slot_e->GetData());
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 15);
}

// TC_504：Embedding 路径别名解析——事故形态镜像（图工位 embeddings、FLOAT16
// embed 工位）。别名解析 + INT8 存储行展开 + scale 收集全链照常，逐值
// ground-truth 与 TC_005（规范名基线）一致。
TEST_F(TextEmbeddingTest, FillInputsEmbeddingViaAlias_TC504)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::FLOAT16, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{kEmbedSlotAlias, 0}, {"embed_scales", 1}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

    const auto* rows = static_cast<const fp16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(static_cast<float>(rows[i * kHidden + h]), static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_505：两名并存的行为级证明——规范名工位落位（逐值 ground-truth），别名
// 工位零触碰（哨兵字节保持）。别名兜底仅是"缺规范名"时的路径，不参与竞争。
TEST_F(TextEmbeddingTest, FillInputsCanonicalWinsOverAlias_TC505)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto canon = MakeBuf(kEmbedSlotName, DataType::FLOAT16, {kCount, kHidden});
    auto alias = MakeBuf(kEmbedSlotAlias, DataType::FLOAT16, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    std::memset(alias->GetData(), 0xA5, alias->GetSize());
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{canon, alias, slot_s};
    std::unordered_map<std::string, size_t> name_index{{kEmbedSlotName, 0}, {kEmbedSlotAlias, 1}, {"embed_scales", 2}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

    const auto* rows = static_cast<const fp16_t*>(canon->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(static_cast<float>(rows[i * kHidden + h]), static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
    const auto* pa = static_cast<const uint8_t*>(alias->GetData());
    for (size_t i = 0; i < alias->GetSize(); ++i)
        EXPECT_EQ(pa[i], 0xA5); // 别名工位未被触碰
}

// TC_506：别名解析成功后既有校验位次不变（"其余校验不变"契约）——
// ① 别名下标越界 → NOT_FOUND（与规范名缺名同一口径，TC_107）；
// ② 别名解析的 TokenId 工位类型非法（FLOAT32）→ INVALID_ARGUMENT（非
//   NOT_FOUND，证明别名命中后仍走原类型校验分支）；
// ③ Embedding 路径别名命中而 embed_scales 缺名 → NOT_FOUND（别名只作用于
//   embed 工位）且 embed 工位零写入（scale 查找先于表行写入）。
TEST_F(TextEmbeddingTest, FillInputsAliasKeepsValidationOrder_TC506)
{
    const TokenId ids[kCount] = {1, 2, 3};
    // ① 别名在 name_index 而下标 ≥ inputs.size()
    {
        auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::INT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
        std::unordered_map<std::string, size_t> ni{{kEmbedSlotAlias, 3}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::TokenId, ids, kCount).code, Status::NOT_FOUND);
    }
    // ② 别名解析的 TokenId 工位类型非法
    {
        auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::FLOAT32, {kCount});
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
        std::unordered_map<std::string, size_t> ni{{kEmbedSlotAlias, 0}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::TokenId, ids, kCount).code, Status::INVALID_ARGUMENT);
    }
    // ③ Embedding 路径：embed 经别名命中、embed_scales 缺名
    {
        ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
        auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::FLOAT16, {kCount, kHidden});
        std::memset(slot_e->GetData(), 0xA5, slot_e->GetSize());
        std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
        std::unordered_map<std::string, size_t> ni{{kEmbedSlotAlias, 0}};
        EXPECT_EQ(emb_.FillInputs(inputs, ni, EmbeddingPath::Embedding, ids, kCount).code, Status::NOT_FOUND);
        const auto* p = static_cast<const uint8_t*>(slot_e->GetData());
        for (size_t i = 0; i < slot_e->GetSize(); ++i)
            EXPECT_EQ(p[i], 0xA5); // 查找先于写入
    }
}

// TC_507：规范名在而不合法时不回退别名——FindEmbedSlot 优先级是名字级
// （头契约"两者并存取规范名"），不依下标合法性二次择路：规范名下标越界即
// NOT_FOUND，别名工位即使可用也不被落位。
TEST_F(TextEmbeddingTest, FillInputsNoAliasFallbackWhenCanonicalPresent_TC507)
{
    const TokenId ids[kCount] = {3, 0, 15};
    auto alias = MakeBuf(kEmbedSlotAlias, DataType::INT32, {kCount});
    std::memset(alias->GetData(), 0xA5, alias->GetSize());
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{alias};
    std::unordered_map<std::string, size_t> name_index{{kEmbedSlotName, 5}, {kEmbedSlotAlias, 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount).code, Status::NOT_FOUND);
    const auto* p = static_cast<const uint8_t*>(alias->GetData());
    for (size_t i = 0; i < alias->GetSize(); ++i)
        EXPECT_EQ(p[i], 0xA5); // 均未写入
}

// ── scale 工位别名（2026-10-14 床证补全）TC_508~TC_511 ───────────────────────
// omc 新转换器（1014 gemma4 c42）将 scale 工位导出为正拼实名 embeddings_scale
// / per_layer_token_identity_scale；embed 侧引擎规范名 embed_scales、别名
// embeddings_scale 双名解析（规范名优先、别名兜底）；identity scale 侧自
// 2026-10-19 拼写契约迁移起仅正拼单名（老转换器 …sacle 拼写工位不再探测）。

// TC_508：FindEmbedScaleSlot 解析语义——规范名优先；规范名缺、别名在 → 兜底
// 命中；两名并存取规范名；均缺 → end()。
TEST_F(TextEmbeddingTest, FindEmbedScaleSlotResolutionSemantics_TC508)
{
    using NameIndex = std::unordered_map<std::string, std::size_t>;

    const NameIndex alias_only{{kEmbedScaleSlotAlias, 3}};
    auto it = FindEmbedScaleSlot(alias_only);
    ASSERT_NE(it, alias_only.end());
    EXPECT_EQ(it->second, 3u);

    const NameIndex canonical_only{{kEmbedScaleSlotName, 7}};
    it = FindEmbedScaleSlot(canonical_only);
    ASSERT_NE(it, canonical_only.end());
    EXPECT_EQ(it->second, 7u);

    // 两名并存取规范名（优先级是名字级判定）
    const NameIndex both{{kEmbedScaleSlotName, 7}, {kEmbedScaleSlotAlias, 3}};
    it = FindEmbedScaleSlot(both);
    ASSERT_NE(it, both.end());
    EXPECT_EQ(it->first, std::string(kEmbedScaleSlotName));
    EXPECT_EQ(it->second, 7u);

    // 均缺 → end()
    const NameIndex empty_index{};
    EXPECT_EQ(FindEmbedScaleSlot(empty_index), empty_index.end());
    const NameIndex unrelated{{"position_ids", 0}};
    EXPECT_EQ(FindEmbedScaleSlot(unrelated), unrelated.end());
}

// TC_509：FindIdentityScaleSlot 解析语义——单名解析（per_layer_token_identity_scale，
// 2026-10-19 契约迁移后无别名）；命中返回、缺 → end()；邻近工位名不误命中。
TEST_F(TextEmbeddingTest, FindIdentityScaleSlotResolutionSemantics_TC509)
{
    using NameIndex = std::unordered_map<std::string, std::size_t>;

    const NameIndex canonical_only{{kTokenIdentityScaleSlotName, 9}};
    auto it = FindIdentityScaleSlot(canonical_only);
    ASSERT_NE(it, canonical_only.end());
    EXPECT_EQ(it->second, 9u);

    const NameIndex empty_index{};
    EXPECT_EQ(FindIdentityScaleSlot(empty_index), empty_index.end());
    const NameIndex unrelated{{kTokenIdentitySlotName, 0}};
    EXPECT_EQ(FindIdentityScaleSlot(unrelated), unrelated.end());
}

// TC_510：IsEmbedScaleSlotName 谓词语义（model_executor 维度修正按名遍历的
// 判定基础）——规范名/别名命中；近似名不命中。
TEST_F(TextEmbeddingTest, IsEmbedScaleSlotNameMatchesCanonicalAndAlias_TC510)
{
    EXPECT_TRUE(IsEmbedScaleSlotName(kEmbedScaleSlotName));
    EXPECT_TRUE(IsEmbedScaleSlotName(kEmbedScaleSlotAlias));
    EXPECT_FALSE(IsEmbedScaleSlotName("embeddings_scales")); // 尾多 s 不命中
    EXPECT_FALSE(IsEmbedScaleSlotName("embed_scale"));       // 单数不命中
    EXPECT_FALSE(IsEmbedScaleSlotName("input_embed"));
    EXPECT_FALSE(IsEmbedScaleSlotName(""));
}

// TC_511：床证事故镜像——1014 gemma4 c42 图实名形态（embeddings +
// embeddings_scale + per_layer_token_identity + per_layer_token_identity_scale）
// Embedding 路径四工位落位 + 逐值 ground-truth 与规范名基线（TC_403）一致；
// 修复前该形态 prefill FillInputs NOT_FOUND(4)。identity scale 工位名与
// kTokenIdentityScaleSlotName 同名（c42 正拼即引擎唯一规范名）。
TEST_F(TextEmbeddingTest, FillInputsEmbeddingViaScaleAlias_TC511)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    // c42 图实名四工位：embed INT8 直拷；identity INT8 → FLOAT32 展开；两 scale
    // Fp32 → FLOAT32 工位
    auto slot_e = MakeBuf(kEmbedSlotAlias, DataType::INT8, {kCount, kHidden});
    auto slot_s = MakeBuf(kEmbedScaleSlotAlias, DataType::FLOAT32, {kCount});
    auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
    auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
    std::unordered_map<std::string, size_t> name_index{
        {kEmbedSlotAlias, 0}, {kEmbedScaleSlotAlias, 1}, {kTokenIdentitySlotName, 2}, {kTokenIdentityScaleSlotName, 3}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

    const auto* emb_rows = static_cast<const int8_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(emb_rows[i * kHidden + h], TableValue(idx[i], h));
    const auto* emb_sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(emb_sc[i], 0.5f + idx[i]);
    const auto* rows = static_cast<const float*>(ident->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t w = 0; w < kWidth; ++w)
            EXPECT_FLOAT_EQ(rows[i * kWidth + w], static_cast<float>(IdentityValue(idx[i], w)));
    const auto* sc = static_cast<const float*>(iscale->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], IdentityScale(idx[i]));
}

// ── L2 mmap 段源组（perf_optimize.md §3.2/§5：InMemory 载体 mmap 化对账）─────
// 2026-10-28 实施：InMemory 表源由 new[]+f.read 堆载体换 mmap 只读映射
// （detail::MmapTableSource；主表段 require_exact=false 超长放行沿用 Q12、
// identity 段 require_exact=true 短/超长均拒——原"读毕恰为文件尾"语义逐字
// 保留，B-10 几何闭环）。本组锁四契约：行等价（全词表 ReadRow/RowPtr 双
// 通道逐字节命中）、截断拒绝（map 前拒，无 SIGBUS 窗口）、非 0 段偏映射的
// 行地址算（identity 段 offset=128）、Load 端到端（InMemory 全链
// Loaded/FillInputs 不回退）。

// TC_L2_1：行等价——InMemory mmap 载体全词表 Gather 逐字节命中
// （ReadRow=memcpy 通道；映射建立即数据就绪）。
TEST_F(TextEmbeddingTest, MmapRowEquivalenceTC_L2_1)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    std::vector<TokenId> ids(kVocab);
    for (size_t v = 0; v < kVocab; ++v)
        ids[v] = static_cast<TokenId>(v);
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kVocab, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kVocab});
    ASSERT_TRUE(emb_.Gather(ids.data(), kVocab, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t v = 0; v < kVocab; ++v) {
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[v * kHidden + h], TableValue(v, h)) << "v=" << v << " h=" << h;
        EXPECT_FLOAT_EQ(sc[v], 0.5f + static_cast<float>(v));
    }
}

// TC_L2_2：段长不足 ⇒ INVALID_ARGUMENT（map 前拒——无 SIGBUS 窗口）；
// 主表段 require_exact=false：超长（identity 段后随）放行，文件尾多字节放行。
TEST_F(TextEmbeddingTest, MmapTruncatedRejectsBeforeMapTC_L2_2)
{
    // 主表段差 1 字节
    const std::string short_path = TempPath("l2_short");
    ASSERT_TRUE(WriteFile(short_path, merged_table_.data(), table_.size() - 1));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = short_path;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.Loaded());

    // 合并布局本体（192B，identity 段后随）与尾多 1 字节均放行（Q12 超长）
    const std::string long_path = TempPath("l2_long");
    std::vector<int8_t> extra = merged_table_;
    extra.push_back(0);
    ASSERT_TRUE(WriteFile(long_path, extra.data(), extra.size()));
    c.table_path = long_path;
    ASSERT_TRUE(emb_.Load(c).ok());
    const TokenId edges[2] = {0, static_cast<TokenId>(kVocab - 1)};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {2, kHidden});
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {2});
    ASSERT_TRUE(emb_.Gather(edges, 2, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    for (size_t h = 0; h < kHidden; ++h)
        EXPECT_EQ(rows[h], TableValue(0, h));
    for (size_t h = 0; h < kHidden; ++h)
        EXPECT_EQ(rows[kHidden + h], TableValue(kVocab - 1, h));

    RemoveFile(short_path);
    RemoveFile(long_path);
}

// TC_L2_3：非 0 段偏映射行地址——identity 段 offset=128B（主表段之后），
// LoadTokenIdentity InMemory 走 MmapTableSource 段偏建映射；行 0 = 映射首行，
// 3 行样本值与 fixture 逐值对（行宽 4 != kHidden 交叉防两表混用）。
// 消费经 FillTokenIdentitySlot_（Embedding 路径，RowPtr 直读通道）。
TEST_F(TextEmbeddingTest, MmapNonZeroSegmentOffsetTC_L2_3)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    ASSERT_TRUE(emb_.IdentityLoaded());

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::INT8, {kCount, kHidden});
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount});
    auto ident = MakeBuf(kTokenIdentitySlotName, DataType::FLOAT32, {kCount, kWidth});
    auto iscale = MakeBuf(kTokenIdentityScaleSlotName, DataType::FLOAT32, {kCount});
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s, ident, iscale};
    std::unordered_map<std::string, size_t> name_index{
        {"input_embed", 0}, {"embed_scales", 1}, {kTokenIdentitySlotName, 2}, {kTokenIdentityScaleSlotName, 3}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());

    // 主表/主 scale 照常（行 0 基址 = 文件头）
    const auto* emb_rows = static_cast<const int8_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(emb_rows[i * kHidden + h], TableValue(idx[i], h));
    const auto* emb_sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(emb_sc[i], 0.5f + idx[i]);
    // identity 段：行 0 = offset 128 处（IdentityValue 表），段偏换算正确
    const auto* rows = static_cast<const float*>(ident->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t w = 0; w < kWidth; ++w)
            EXPECT_FLOAT_EQ(rows[i * kWidth + w], static_cast<float>(IdentityValue(idx[i], w)));
    const auto* sc = static_cast<const float*>(iscale->GetData());
    for (size_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(sc[i], IdentityScale(idx[i]));
}

// TC_L2_4：Load 端到端不回退——InMemory 主表 + identity 双段 mmap 全链
// （TC_403 同链），四工位落位逐值断言复用权威助手。
TEST_F(TextEmbeddingTest, MmapLoadEndToEndTC_L2_4)
{
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    ASSERT_TRUE(emb_.LoadTokenIdentity(IdentityConfig(EmbeddingLoadMode::InMemory), kWidth).ok());
    EXPECT_TRUE(emb_.Loaded());
    EXPECT_TRUE(emb_.IdentityLoaded());
    FillEmbeddingPathAndExpectAllSlots(emb_);
}

} // namespace
} // namespace cann_lm::engine
