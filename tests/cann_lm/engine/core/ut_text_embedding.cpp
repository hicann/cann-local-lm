// ═══════════════════════════════════════════════════════════════════════════
// TextEmbedding 单元测试（SR20260828001，覆盖 SPEC 第 8 章 TC_001~TC_113）。
// host 侧三件事：加载（两模式/两粒度）、查表（行 + scale）、填工位（两路径）。
// NPU 侧反量化不在 UT 范围（后端依赖）。
// mock 工位 bundle 经 TensorManager 代理（hiai::CreateNDTensorBuffer）按 desc 构造。
// ═══════════════════════════════════════════════════════════════════════════
#include "core/component/text_embedding/text_embedding.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace cann_lm::engine {
namespace {

constexpr size_t kVocab  = 16;
constexpr size_t kHidden = 8;
constexpr size_t kCount  = 3;   // ids = {3, 0, 15}

/// 确定性 INT8 表值：行 v 列 h → (v*7 + h*3 - 50)，落在 int8 范围。
int8_t TableValue(size_t v, size_t h) {
    return static_cast<int8_t>(static_cast<int>(v) * 7 + static_cast<int>(h) * 3 - 50);
}

std::string TempPath(const char* tag) {
    static std::atomic<int> seq{0};
    return "/tmp/ut_text_embedding_" + std::to_string(::getpid()) + "_" + tag +
           "_" + std::to_string(seq.fetch_add(1)) + ".bin";
}

bool WriteFile(const std::string& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<size_t>(f.tellp()) == n;
}

bool RemoveFile(const std::string& path) { return ::remove(path.c_str()) == 0; }

/// IEEE 754 binary16 ↔ float（软件 fp16，不依赖 __fp16 关键字——OHOS musl
/// 交叉编译下 __fp16 直接引用失效，text_embedding.cpp 亦改软件实现）。
uint16_t FloatToHalfBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000;
    int32_t       exp  = static_cast<int32_t>((u >> 23) & 0xff) - 127 + 15;
    uint32_t      mant = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff) {                       // inf/nan
        return static_cast<uint16_t>(sign | 0x7c00 | (mant ? (mant >> 13) : 0));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);   // 溢出 → inf
    if (exp <= 0) {                                                // 子规格 / 下溢
        if (exp < -10) return static_cast<uint16_t>(sign);        // → 0
        mant |= 0x800000;
        const uint32_t shift    = static_cast<uint32_t>(14 - exp);
        uint32_t       half_m   = mant >> shift;
        const uint32_t rem      = mant & ((1u << shift) - 1);
        const uint32_t halfway  = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_m & 1))) ++half_m;
        return static_cast<uint16_t>(sign | half_m);
    }
    uint32_t       half_m  = mant >> 13;
    const uint32_t rem     = mant & 0x1fff;
    const uint32_t halfway = 0x1000;
    if (rem > halfway || (rem == halfway && (half_m & 1))) ++half_m;
    if (half_m > 0x3ff) { half_m = 0; ++exp; }                     // 进位至 exponent
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_m);
}

float HalfBitsToFloat(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t       mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {                                                   // 子规格
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; --e; }
            mant &= 0x3ff;
            u = sign | (static_cast<uint32_t>(127 - 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {                                      // inf/nan
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// 经 TensorManager 代理（hiai::CreateNDTensorBuffer）构造 mock 工位 buffer。
/// NDTensorDesc 无 .size 字段——字节数由 dims×dtype 推导，bytes 参数仅作
/// 调用点自文档化（不再参与 desc 构造）。
std::shared_ptr<ITensorBuffer> MakeBuf(const std::string& name, DataType dt,
                                       std::vector<int32_t> dims, size_t bytes) {
    (void)bytes;
    TensorDesc d;
    d.tensorName = name;
    d.dataType   = dt;
    d.format     = Format::ND;
    d.dims       = std::move(dims);
    std::shared_ptr<ITensorBuffer> out;
    auto st = TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &out);
    return st.ok() ? out : nullptr;
}

/// INT8 表 [16,8] + Fp32 PerToken scale [16] 的文件 fixture。
class TextEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        table_.resize(kVocab * kHidden);
        for (size_t v = 0; v < kVocab; ++v)
            for (size_t h = 0; h < kHidden; ++h)
                table_[v * kHidden + h] = TableValue(v, h);
        for (size_t v = 0; v < kVocab; ++v) scale_[v] = 0.5f + static_cast<float>(v);
        table_path_ = TempPath("table");
        scale_path_ = TempPath("scale");
        ASSERT_TRUE(WriteFile(table_path_, table_.data(), table_.size()));
        ASSERT_TRUE(WriteFile(scale_path_, scale_.data(), scale_.size() * sizeof(float)));
    }
    void TearDown() override {
        RemoveFile(table_path_);
        RemoveFile(scale_path_);
    }

    EmbeddingConfig Int8PerTokenConfig(EmbeddingLoadMode mode) const {
        EmbeddingConfig c;
        c.table_path  = table_path_;
        c.scale_path  = scale_path_;
        c.table_type  = EmbeddingType::Int8;
        c.scale_type  = EmbeddingScaleType::Fp32;
        c.quant_mode  = EmbeddingQuantMode::PerToken;
        c.vocab_size  = kVocab;
        c.hidden_dim  = kHidden;
        c.load_mode   = mode;
        return c;
    }

    std::vector<int8_t>  table_ = std::vector<int8_t>(kVocab * kHidden);
    std::vector<float>   scale_ = std::vector<float>(kVocab);
    std::string          table_path_, scale_path_;
    TextEmbedding        emb_;
};

// ── 正常路径 TC_001~TC_010 ──────────────────────────────────────────────────

// TC_001：PerToken InMemory 加载成功
TEST_F(TextEmbeddingTest, LoadPerTokenInMemorySuccess_TC001) {
    auto s = emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory));
    ASSERT_TRUE(s.ok()) << s.code;
    EXPECT_TRUE(emb_.Loaded());
    // Table()/Scale()（曾暴露存储 TensorBuffer）已删除：InMemory 存储现为堆内存，
    // 不再经 TensorBuffer 暴露（2026-10-09 设备事故修正）。改以 TableType()/LoadMode()
    // 判别加载态与运行模式，替代整表/scale buffer 的存在性与尺寸断言。
    EXPECT_EQ(emb_.TableType(), EmbeddingType::Int8);
    EXPECT_EQ(emb_.LoadMode(), EmbeddingLoadMode::InMemory);
}

// TC_002：PerGroup（group_size=2，groups=4）加载成功
TEST_F(TextEmbeddingTest, LoadPerGroupSuccess_TC002) {
    std::vector<float> gs(kVocab * 4);
    for (size_t i = 0; i < gs.size(); ++i) gs[i] = 0.25f * static_cast<float>(i);
    const std::string spath = TempPath("gscale");
    ASSERT_TRUE(WriteFile(spath, gs.data(), gs.size() * 4));

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_path = spath;
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 2;
    auto s = emb_.Load(c);
    EXPECT_TRUE(s.ok());
    // scale buffer 已不再暴露（Table()/Scale() 删除）；PerGroup scale 字节数
    // （vocab×4 组×4 字节）的正确性由 TC_004 gather 逐值校验承载。
    RemoveFile(spath);
}

// TC_003：PerToken gather 逐字节正确 + scale 收集正确
TEST_F(TextEmbeddingTest, GatherPerTokenCorrectness_TC003) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    ASSERT_TRUE(emb_.Gather(ids, kCount, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const size_t idx[kCount] = {3, 0, 15};
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h)) << "i=" << i << " h=" << h;
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_004：PerGroup gather 的 scale 收集（ids={5} → 下标 5×4+0..3）
TEST_F(TextEmbeddingTest, GatherPerGroupScales_TC004) {
    std::vector<float> gs(kVocab * 4);
    for (size_t i = 0; i < gs.size(); ++i) gs[i] = 0.25f * static_cast<float>(i);
    const std::string spath = TempPath("gscale");
    ASSERT_TRUE(WriteFile(spath, gs.data(), gs.size() * 4));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_path = spath;
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 2;
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId id = 5;
    auto out_e = MakeBuf("out_embed", DataType::INT8, {1, kHidden}, kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {4}, 16);
    ASSERT_TRUE(emb_.Gather(&id, 1, out_e, out_s).ok());
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t g = 0; g < 4; ++g) EXPECT_FLOAT_EQ(sc[g], gs[5 * 4 + g]);
    RemoveFile(spath);
}

// TC_005：Embedding 路径 FillInputs（INT8 存储 → FLOAT16 工位展开 + scale → FLOAT32）
TEST_F(TextEmbeddingTest, FillInputsEmbeddingPath_TC005) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount]  = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
    auto s = emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount);
    ASSERT_TRUE(s.ok()) << s.code;

    const auto* rows = static_cast<const uint16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(HalfBitsToFloat(rows[i * kHidden + h]),
                            static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_006：TokenId 路径无需 Load（INT32 工位）
TEST_F(TextEmbeddingTest, FillInputsTokenIdNoLoad_TC006) {
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::INT32, {kCount}, kCount * 4);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    auto s = emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount);
    ASSERT_TRUE(s.ok()) << s.code;
    const auto* v = static_cast<const int32_t*>(slot_e->GetData());
    EXPECT_EQ(v[0], 3); EXPECT_EQ(v[1], 0); EXPECT_EQ(v[2], 15);
}

// TC_010：TokenId 路径 INT64 工位
TEST_F(TextEmbeddingTest, FillInputsTokenIdInt64_TC010) {
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::INT64, {kCount}, kCount * 8);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::TokenId, ids, kCount).ok());
    const auto* v = static_cast<const int64_t*>(slot_e->GetData());
    EXPECT_EQ(v[0], 3); EXPECT_EQ(v[1], 0); EXPECT_EQ(v[2], 15);
}

// TC_007：同型免转换直填（Fp16 存储 → FLOAT16 工位，逐字节相等）
TEST_F(TextEmbeddingTest, SameTypeDirectCopy_TC007) {
    std::vector<uint16_t> t(kVocab * kHidden);
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = FloatToHalfBits(static_cast<float>(i) * 0.5f - 8.0f);
    const std::string tpath = TempPath("fp16table");
    ASSERT_TRUE(WriteFile(tpath, t.data(), t.size() * 2));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = tpath;
    c.table_type = EmbeddingType::Fp16;
    c.scale_type = EmbeddingScaleType::None;
    c.scale_path.clear();
    ASSERT_TRUE(emb_.Load(c).ok());

    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount]  = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
    // 同型免转换直填：FLOAT16 表 → FLOAT16 工位逐字节拷贝，bit 模式严格相等
    const auto* rows = static_cast<const uint16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], t[idx[i] * kHidden + h]);
    RemoveFile(tpath);
}

// TC_008：InFlash 加载与 gather（与 InMemory 逐字节一致）
TEST_F(TextEmbeddingTest, InFlashLoadGather_TC008) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    EXPECT_TRUE(emb_.Loaded());
    // InFlash 无整表内存映像——Table() 已删除，改以 LoadMode() 判别 InFlash 模式
    //（InMemory 整表入堆，InFlash 留文件按行 pread，两者皆 Loaded()==true）。
    EXPECT_EQ(emb_.LoadMode(), EmbeddingLoadMode::InFlash);

    const TokenId ids[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    ASSERT_TRUE(emb_.Gather(ids, kCount, out_e, out_s).ok());
    const auto* rows = static_cast<const int8_t*>(out_e->GetData());
    const size_t idx[kCount] = {3, 0, 15};
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_EQ(rows[i * kHidden + h], TableValue(idx[i], h));
    const auto* sc = static_cast<const float*>(out_s->GetData());
    for (size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// TC_009：InFlash 模式 FillInputs（与 TC_005 InMemory 同配置逐字节一致）
TEST_F(TextEmbeddingTest, InFlashFillInputs_TC009) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    auto slot_s = MakeBuf("embed_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e, slot_s};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}, {"embed_scales", 1}};
    ASSERT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
    const size_t idx[kCount] = {3, 0, 15};
    const auto* rows = static_cast<const uint16_t*>(slot_e->GetData());
    for (size_t i = 0; i < kCount; ++i)
        for (size_t h = 0; h < kHidden; ++h)
            EXPECT_FLOAT_EQ(HalfBitsToFloat(rows[i * kHidden + h]),
                            static_cast<float>(TableValue(idx[i], h)));
    const auto* sc = static_cast<const float*>(slot_s->GetData());
    for (size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(sc[i], 0.5f + idx[i]);
}

// ── 异常路径 TC_101~TC_113 ──────────────────────────────────────────────────

// TC_101：表文件缺失 → INVALID_ARGUMENT，可重 Load
TEST_F(TextEmbeddingTest, MissingTableFile_TC101) {
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = "/nonexistent/dir/table.bin";
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    EXPECT_FALSE(emb_.Loaded());
    EXPECT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());   // 可恢复
}

// TC_102：文件尺寸不符（表过短 / 过长）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, FileSizeMismatch_TC102) {
    const std::string short_path = TempPath("short");
    ASSERT_TRUE(WriteFile(short_path, table_.data(), table_.size() - 1));
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_path = short_path;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);

    const std::string long_path = TempPath("long");
    std::vector<int8_t> extra = table_;
    extra.push_back(0);
    ASSERT_TRUE(WriteFile(long_path, extra.data(), extra.size()));
    c.table_path = long_path;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);

    // InFlash 侧 fstat 校验
    c.table_path = short_path;
    c.load_mode = EmbeddingLoadMode::InFlash;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    RemoveFile(short_path); RemoveFile(long_path);
}

// TC_103：PerGroup 参数非法（group_size=0 / 不整除 hidden）
TEST_F(TextEmbeddingTest, PerGroupInvalidParams_TC103) {
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.quant_mode = EmbeddingQuantMode::PerGroup;
    c.group_size = 0;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    c.group_size = 3;   // 8 % 3 != 0
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
    c.group_size = kHidden + 1;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
}

// TC_104：scale 配置自相矛盾（None 且 path 非空）
TEST_F(TextEmbeddingTest, ScaleConfigContradiction_TC104) {
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_type = EmbeddingScaleType::None;
    // scale_path 非空 → INVALID_ARGUMENT
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);
}

// TC_105：tokenId 越界 → 整批拒绝，out 不被部分写入
TEST_F(TextEmbeddingTest, TokenIdOutOfRangeRejectsBatch_TC105) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 16, 1};   // 16 ≥ vocab
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    std::memset(out_e->GetData(), 0xAB, out_e->GetSize());
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);
    const auto* p = static_cast<const uint8_t*>(out_e->GetData());
    for (size_t i = 0; i < out_e->GetSize(); ++i) EXPECT_EQ(p[i], 0xAB);   // 未写入
}

// TC_106：未加载先查（Gather / Embedding 路径 FillInputs → INVALID_STATE；TokenId 豁免）
TEST_F(TextEmbeddingTest, NotLoadedRejectsQueries_TC106) {
    const TokenId ids[kCount] = {1, 2, 3};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_STATE);

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_STATE);
}

// TC_107：工位缺名（input_embed / embed_scales）→ NOT_FOUND
TEST_F(TextEmbeddingTest, MissingSlotName_TC107) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> empty_index{};
    EXPECT_EQ(emb_.FillInputs(inputs, empty_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::NOT_FOUND);

    // input_embed 有、embed_scales 缺（scale_type ≠ None）→ NOT_FOUND
    std::unordered_map<std::string, size_t> only_embed{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, only_embed, EmbeddingPath::Embedding, ids, kCount).code,
              Status::NOT_FOUND);
}

// TC_108：不支持的转换组合（Int4 存储 / Fp32→FLOAT16 工位）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, UnsupportedConversion_TC108) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto slot_e = MakeBuf("input_embed", DataType::INT64, {kCount, kHidden}, kCount * kHidden * 8);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_ARGUMENT);   // Int8 → INT64 超矩阵

    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.table_type = EmbeddingType::Int4;
    EXPECT_EQ(emb_.Load(c).code, Status::INVALID_ARGUMENT);   // U-3
}

// TC_109：输出容量不足 → INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, InsufficientOutputCapacity_TC109) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    // NDTensorDesc 无 .size 字段（字节数由 dims×dtype 推导），无法再以
    // dims={kCount,kHidden}+size=23 构造欠 1 字节 buffer；改减一维使容量真正
    // 不足：{kCount,kHidden-1} = 21 字节 < 需 24 字节。
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden - 1}, kCount * (kHidden - 1));
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);
}

// TC_110：NULL 入参（ids=nullptr 且 count>0）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, NullIdsRejected_TC110) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    EXPECT_EQ(emb_.Gather(nullptr, kCount, out_e, out_s).code, Status::INVALID_ARGUMENT);

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, nullptr, kCount).code,
              Status::INVALID_ARGUMENT);
}

// TC_111：InFlash 运行期文件损坏（截断）→ FAILURE
TEST_F(TextEmbeddingTest, InFlashRuntimeCorruption_TC111) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    ASSERT_EQ(0, ::truncate(table_path_.c_str(),
                            static_cast<off_t>(table_.size() / 2)));   // Load 后截断
    const TokenId ids[kCount] = {3, 0, 15};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    EXPECT_EQ(emb_.Gather(ids, kCount, out_e, out_s).code, Status::FAILURE);
}

// TC_112：空 buffer 入参（Gather 空 shared_ptr / FillInputs 含空 buffer）→ INVALID_ARGUMENT
TEST_F(TextEmbeddingTest, NullBufferRejected_TC112) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {1, 2, 3};
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    EXPECT_EQ(emb_.Gather(ids, kCount, nullptr, out_s).code, Status::INVALID_ARGUMENT);

    std::vector<std::shared_ptr<ITensorBuffer>> inputs{nullptr};   // 工位 buffer 为空
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_EQ(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).code,
              Status::INVALID_ARGUMENT);
}

// TC_113：FromSection 未知枚举字符串 / 空 weights（不加载信号）——SR20260903001 D-12
TEST(FromSectionTest, UnknownStringAndEmptyWeights_TC113) {
    EmbeddingConfig out;
    // 空 weights → SUCCESS + 不加载信号（identity_type/identity_scale_type 传空串）
    EXPECT_TRUE(EmbeddingConfig::FromSection("", "", "", "", "", "", "", "", 16, 8, &out).ok());
    EXPECT_TRUE(out.table_path.empty());
    // 未知 table_type → INVALID_ARGUMENT
    EXPECT_EQ(EmbeddingConfig::FromSection("/tmp/table.bin", "", "fp8", "", "", "", "", "",
                                           16, 8, &out).code,
              Status::INVALID_ARGUMENT);
    // nullptr out → INVALID_ARGUMENT
    EXPECT_EQ(EmbeddingConfig::FromSection("/tmp/table.bin", "", "fp8", "", "", "", "", "",
                                           16, 8, nullptr).code,
              Status::INVALID_ARGUMENT);
}

// FromSection 合法解析（全字段；per_group 无组大小 → Load 期拒绝口径维持，TC_209）
TEST(FromSectionTest, ParsesAllFields) {
    EmbeddingConfig out;
    ASSERT_TRUE(EmbeddingConfig::FromSection("t.bin", "s.bin", "int8", "fp16",
                                             "per_token", "in_flash", "", "",
                                             100, 64, &out).ok());
    EXPECT_EQ(out.table_type, EmbeddingType::Int8);
    EXPECT_EQ(out.scale_type, EmbeddingScaleType::Fp16);
    EXPECT_EQ(out.quant_mode, EmbeddingQuantMode::PerToken);
    EXPECT_EQ(out.load_mode, EmbeddingLoadMode::InFlash);
    EXPECT_EQ(out.vocab_size, 100u);
    EXPECT_EQ(out.hidden_dim, 64u);
}

// ── 边界 TC_201~TC_204 ──────────────────────────────────────────────────────

// TC_201：count=0 → SUCCESS，不写任何字节
TEST_F(TextEmbeddingTest, CountZeroWritesNothing_TC201) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
    std::memset(out_e->GetData(), 0xCD, out_e->GetSize());
    EXPECT_TRUE(emb_.Gather(nullptr, 0, out_e, out_s).ok());
    const auto* p = static_cast<const uint8_t*>(out_e->GetData());
    for (size_t i = 0; i < out_e->GetSize(); ++i) EXPECT_EQ(p[i], 0xCD);
}

// TC_202：scale_type=None（无 scale 文件；Gather 跳过 out_scales；FillInputs 跳过 embed_scales）
TEST_F(TextEmbeddingTest, ScaleNoneSkipsScales_TC202) {
    EmbeddingConfig c = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    c.scale_type = EmbeddingScaleType::None;
    c.scale_path.clear();
    ASSERT_TRUE(emb_.Load(c).ok());
    // scale_type=None 无 scale——Scale() 已删除不再暴露 scale buffer；
    // None 路径的正确性由下方 Gather(out_scales=nullptr) 成功承载。

    const TokenId ids[kCount] = {1, 2, 3};
    auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
    EXPECT_TRUE(emb_.Gather(ids, kCount, out_e, /*out_scales=*/nullptr).ok());

    auto slot_e = MakeBuf("input_embed", DataType::FLOAT16, {kCount, kHidden}, kCount * kHidden * 2);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{slot_e};
    std::unordered_map<std::string, size_t> name_index{{"input_embed", 0}};
    EXPECT_TRUE(emb_.FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids, kCount).ok());
}

// TC_203：vocab 边界 id（vocab-1 成功、vocab 拒绝）
TEST_F(TextEmbeddingTest, VocabBoundaryId_TC203) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ok_id = static_cast<TokenId>(kVocab - 1);
    const TokenId bad_id = static_cast<TokenId>(kVocab);
    auto out_e = MakeBuf("out_embed", DataType::INT8, {1, kHidden}, kHidden);
    auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {1}, 4);
    EXPECT_TRUE(emb_.Gather(&ok_id, 1, out_e, out_s).ok());
    EXPECT_EQ(emb_.Gather(&bad_id, 1, out_e, out_s).code, Status::INVALID_ARGUMENT);
}

// TC_204：失败 Load 后修正配置再 Load 成功
TEST_F(TextEmbeddingTest, ReloadAfterFailure_TC204) {
    EmbeddingConfig bad = Int8PerTokenConfig(EmbeddingLoadMode::InMemory);
    bad.vocab_size = 99;   // 尺寸校验必败
    EXPECT_FALSE(emb_.Load(bad).ok());
    EXPECT_FALSE(emb_.Loaded());
    EXPECT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    EXPECT_TRUE(emb_.Loaded());
}

// ── 并发 TC_301~TC_302 ──────────────────────────────────────────────────────

// TC_301：4 线程并发 Gather（InMemory），结果与串行一致
TEST_F(TextEmbeddingTest, ConcurrentGatherInMemory_TC301) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InMemory)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount]  = {3, 0, 15};

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
            auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
            for (int iter = 0; iter < 50; ++iter) {
                if (!emb_.Gather(ids, kCount, out_e, out_s).ok()) { mismatches++; continue; }
                const auto* rows = static_cast<const int8_t*>(out_e->GetData());
                const auto* sc = static_cast<const float*>(out_s->GetData());
                for (size_t i = 0; i < kCount; ++i) {
                    for (size_t h = 0; h < kHidden; ++h)
                        if (rows[i * kHidden + h] != TableValue(idx[i], h)) mismatches++;
                    if (sc[i] != 0.5f + idx[i]) mismatches++;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    EXPECT_EQ(mismatches.load(), 0);
}

// TC_302：4 线程共享 InFlash fd 并发 Gather，互不干扰
TEST_F(TextEmbeddingTest, ConcurrentGatherInFlash_TC302) {
    ASSERT_TRUE(emb_.Load(Int8PerTokenConfig(EmbeddingLoadMode::InFlash)).ok());
    const TokenId ids[kCount] = {3, 0, 15};
    const size_t idx[kCount]  = {3, 0, 15};

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            auto out_e = MakeBuf("out_embed", DataType::INT8, {kCount, kHidden}, kCount * kHidden);
            auto out_s = MakeBuf("out_scales", DataType::FLOAT32, {kCount}, kCount * 4);
            for (int iter = 0; iter < 50; ++iter) {
                if (!emb_.Gather(ids, kCount, out_e, out_s).ok()) { mismatches++; continue; }
                const auto* rows = static_cast<const int8_t*>(out_e->GetData());
                const auto* sc = static_cast<const float*>(out_s->GetData());
                for (size_t i = 0; i < kCount; ++i) {
                    for (size_t h = 0; h < kHidden; ++h)
                        if (rows[i * kHidden + h] != TableValue(idx[i], h)) mismatches++;
                    if (sc[i] != 0.5f + idx[i]) mismatches++;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    EXPECT_EQ(mismatches.load(), 0);
}

}  // namespace
}  // namespace cann_lm::engine
