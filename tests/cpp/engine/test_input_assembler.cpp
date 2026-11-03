// 输入组装器单元测试（v0.13：按 INPUT_INFO.md 设计填 position_ids/new_kv_cache_pos/attention_mask）。
// 关联 docs/model_executor/INPUT_INFO.md。
// gemma4 双 bool mask 用例（GemmaMaskTest）关联 docs/mask_fill.md / mask_fill_detail.md；
// mask_sliding 列宽契约同滑动 KV 序列维改造（sliding_kv_seq_dim_rebind_202610270913.md
// U4/U7-b：新契 C=W+P kSliding 纯模 / 旧契 kv_max kAnchored——b688221 w_eff 口径废弃 D8）。
// sliding_kv_pos 可选工位用例（SlidingPosTest）关联 1027 新图工位接线
// （tmp_docs/problems/device_garbled_output_sliding_kv_pos_unwired_20261027.md：
// 填充口径与 new_kv_cache_pos 同构、环宽独立 C=W+P；图带而 C 算不出宁拒）。
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/model_executor/model_executor.h" // SlidingLayerWidth（新契 C=W+P 单源）
#include "core/tensor_manager.h"
#include "utils/data_type_utils.h" // fp16_t/FloatToHalfBits（fp16 mask UT U3-U5）

#include <gtest/gtest.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cann_local_lm::engine {
namespace {

// ── 辅助：创建 ITensorBuffer + 读写 ──────────────────────────────────────

// 张量经 TensorManager（backend 代理）创建；host 无 NDK/stub so 时创建失败 →
// 空指针（用例守卫见下方 UT_NEED_BACKEND）。buffer 尺寸由 desc（dims×dtype）派生，
// 无独立 size 字段。
std::shared_ptr<ITensorBuffer> MakeTensor(const std::vector<int32_t>& dims, DataType dt)
{
    TensorDesc d;
    d.dims = dims;
    d.dataType = dt;
    std::shared_ptr<ITensorBuffer> t;
    if (!TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t).ok())
        return nullptr;
    return t;
}

/// host 可用性探测：CreateTensor 走真后端（NDKProxy 搜 .so），无 .so → 失败。
bool HostBackendAvailable()
{
    TensorDesc d;
    d.dims = {1};
    d.dataType = DataType::INT32;
    std::shared_ptr<ITensorBuffer> t;
    return TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t).ok();
}

/// 用例守卫：触碰张量数据的用例在 host 无 NDK/stub so 时跳过
/// （stub .so 编法见 PROJECT.md「host UT 编译」节；无异常 gtest，
/// GTEST_SKIP 展开 return，必须写在 TEST 体内——同 ut_linear_state_pool）。
#define UT_NEED_BACKEND()                                               \
    do {                                                                \
        if (!HostBackendAvailable())                                    \
            GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用"; \
    } while (0)

int32_t ReadI32(const std::shared_ptr<ITensorBuffer>& t, std::size_t idx)
{
    if (!t || !t->GetData() || (idx + 1) * sizeof(int32_t) > t->GetSize())
        return -999;
    int32_t v{};
    std::memcpy(&v, static_cast<char*>(t->GetData()) + idx * sizeof(int32_t), sizeof(int32_t));
    return v;
}

float ReadF32(const std::shared_ptr<ITensorBuffer>& t, std::size_t idx)
{
    if (!t || !t->GetData() || (idx + 1) * sizeof(float) > t->GetSize())
        return -999.0f;
    float v{};
    std::memcpy(&v, static_cast<char*>(t->GetData()) + idx * sizeof(float), sizeof(float));
    return v;
}

// fp16 元素按 uint16_t 位模式读出（mask_dtype_template_support U3-U5）。
// 直接比对位模式：attend=0x0000、屏蔽=0xFC00（-inf）。
uint16_t ReadF16Bits(const std::shared_ptr<ITensorBuffer>& t, std::size_t idx)
{
    if (!t || !t->GetData() || (idx + 1) * sizeof(uint16_t) > t->GetSize())
        return 0xFFFF;
    uint16_t v{};
    std::memcpy(&v, static_cast<char*>(t->GetData()) + idx * sizeof(uint16_t), sizeof(uint16_t));
    return v;
}

uint8_t ReadBool(const std::shared_ptr<ITensorBuffer>& t, std::size_t idx)
{
    if (!t || !t->GetData() || idx >= t->GetSize())
        return 0xFF;
    uint8_t v{};
    std::memcpy(&v, static_cast<char*>(t->GetData()) + idx, sizeof(uint8_t));
    return v;
}

/// gemma4 bool mask 行校验：行 row 在列宽 width 下 attend 恰为 [lo, hi]
/// （0=attend），其余位置全 1（1=屏蔽）。逐字节全行扫描（残留/越界一眼可见）。
/// width = 该工位物理槽域宽（mask_full=kv_max / mask_sliding=可行环宽，调用方
/// 按契约传：旧契 kv_max、新契 C=W+P）。
void ExpectRow(const std::shared_ptr<ITensorBuffer>& t, std::size_t row, std::size_t width, int lo, int hi)
{
    for (std::size_t j = 0; j < width; ++j) {
        const bool attend = static_cast<int>(j) >= lo && static_cast<int>(j) <= hi;
        EXPECT_EQ(ReadBool(t, row * width + j), attend ? 0 : 1) << "row=" << row << " j=" << j;
    }
}

// ── 创建 6 个输入 tensor + name→index 映射 ────────────────────────────────

struct InputSet {
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;

    // 按 INPUT_INFO.md 定义的 6 个输入
    static InputSet Make(int batch, int seq_len, int kv_cache_len)
    {
        InputSet s;
        // [0] input_embed: FLOAT16 [batch, seqLen, hidden=4]
        s.tensors.push_back(MakeTensor({batch, seq_len, 4}, DataType::FLOAT16));
        s.name_index["input_embed"] = 0;

        // [1] embed_scales: FLOAT32 [batch, seqLen, 1]
        s.tensors.push_back(MakeTensor({batch, seq_len, 1}, DataType::FLOAT32));
        s.name_index["embed_scales"] = 1;

        // [2] attention_mask: FLOAT32 [batch, 1, seqLen, kvCacheLen]
        s.tensors.push_back(MakeTensor({batch, 1, seq_len, kv_cache_len}, DataType::FLOAT32));
        s.name_index["attention_mask"] = 2;

        // [3] position_ids: INT32 [batch, seqLen]
        s.tensors.push_back(MakeTensor({batch, seq_len}, DataType::INT32));
        s.name_index["position_ids"] = 3;

        // [4] new_kv_cache_pos: INT32 [batch, seqLen]
        s.tensors.push_back(MakeTensor({batch, seq_len}, DataType::INT32));
        s.name_index["new_kv_cache_pos"] = 4;

        // [5] kvCache: placeholder（内容不消费，仅占工位）
        s.tensors.push_back(MakeTensor({256}, DataType::FLOAT32));
        s.name_index["kvCache"] = 5;

        return s;
    }

    std::shared_ptr<ITensorBuffer>& at(const std::string& name) { return tensors[name_index.at(name)]; }
};

// ── Fixture ─────────────────────────────────────────────────────────────────

struct Fixture {
    std::shared_ptr<KvCachePool> pool;
    std::shared_ptr<InputAssembler> asm_;
    Fixture(int max_batch_size = 4, int max_seq_len = 16)
        : pool(std::make_shared<KvCachePool>(max_batch_size, max_seq_len, 2, 4, 8)),
          asm_(std::make_shared<InputAssembler>(pool, max_batch_size))
    {}
};

// ── gemma4 图输入组：mask_full/mask_sliding（BOOL）替代 attention_mask ──────

/// GemmaFixture 默认滑窗宽 W（与 kv_max=16 拉开档差）；新契 fixture 传 P=W ⇒
/// 滑窗环宽 C = W+P = 2W——GemmaInputSet::Make 的 mask_sliding 工位列宽缺省
/// 同源 C。本文件 GemmaInputSet 只与 GemmaFixture 搭配，fixture 换窗宽/契约的
/// 用例须同时给 Make 显式传同宽（绕回最小档差见各用例注释）。
constexpr int kGemmaSlidingWidth = 4;
constexpr int kGemmaSlidingRingC = 2 * kGemmaSlidingWidth; ///< C=W+P（fixture P=W）

struct GemmaInputSet {
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;

    // mask 列宽（sliding_kv_seq_dim_rebind D5）：mask_full 列宽恒 kv_max
    // （kv_max 锚定环）；mask_sliding 列宽 = 滑窗层可行环宽 sliding_width
    // （缺省 kGemmaSlidingRingC=8——GemmaFixture 默认按新契 C=W+P=2W 形态构图；
    // 旧契用例显式传 kv_max 构同形工位）。BOOL/INT8 1 字节/元素。
    static GemmaInputSet Make(int batch, int seq_rows, int kv_max, DataType dt = DataType::BOOL,
                              int sliding_width = kGemmaSlidingRingC)
    {
        GemmaInputSet s;
        const int w = (sliding_width > 0) ? sliding_width : kv_max;
        s.tensors.push_back(MakeTensor({batch, 1, seq_rows, kv_max}, dt));
        s.name_index["mask_full"] = 0;
        s.tensors.push_back(MakeTensor({batch, 1, seq_rows, w}, dt));
        s.name_index["mask_sliding"] = 1;
        s.tensors.push_back(MakeTensor({batch, seq_rows}, DataType::INT32));
        s.name_index["position_ids"] = 2;
        s.tensors.push_back(MakeTensor({batch, seq_rows}, DataType::INT32));
        s.name_index["new_kv_cache_pos"] = 3;
        return s;
    }

    std::shared_ptr<ITensorBuffer>& at(const std::string& name) { return tensors[name_index.at(name)]; }
};

// ── gemma4 专用 Fixture：sliding_window 可调（默认 4，与 kv_max=16 拉开档差）──
// 构造即新契形态（sliding_kv_seq_dim_rebind U4/U7-b）：
//   GemmaFixture(w)            → 新契滑窗环 C=w+P：prefill_max=w、sw_hd>0（P=W
//                                使 C=2w——kSliding 纯模 j%C 测试面）；
//   GemmaFixture(w, /*legacy=*/true) → 旧契（sw_hd=0，kv_max 大环 kAnchored
//                                + window=W 截断——pre-b688221 语义回归锁）。
// 新契 attend 窗宽语义保持 w：窗口 [q-w+1, q]、列宽 C=2w（fixture 缺省 w=4 →
// C=8，与 GemmaInputSet::Make 缺省列宽 8 对齐——unit 内自洽）。

struct GemmaFixture {
    std::shared_ptr<KvCachePool> pool;
    std::shared_ptr<InputAssembler> asm_;
    explicit GemmaFixture(int sliding_window = 4, bool legacy = false)
        : pool(std::make_shared<KvCachePool>(4, 16, 2, 4, 8)),
          asm_(std::make_shared<InputAssembler>(pool, 4, sliding_window,
                                                /*prefill_max=*/legacy ? 0 : sliding_window,
                                                /*sw_hd=*/legacy ? 0 : 64))
    {}
};

// ── 1027 新图输入组：GemmaInputSet + 可选 sliding_kv_pos 工位 ──────────────
// 语义对偶 new_kv_cache_pos（[seq_rows]/[rows] 一维 INT32，1027 图
// [64,1,1,1] om 占位经 CorrectTensorDescs 修形 [seq_len] 同构）；本组仅供
// SlidingPosTest——缺省（with_spos=false）不影响既有用例的 4 工位面。

struct GemmaInputSetWithSlidingPos {
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;

    static GemmaInputSetWithSlidingPos Make(int batch, int seq_rows, int kv_max, bool with_spos,
                                            DataType dt = DataType::BOOL, int sliding_width = kGemmaSlidingRingC)
    {
        GemmaInputSetWithSlidingPos s;
        const int w = (sliding_width > 0) ? sliding_width : kv_max;
        s.tensors.push_back(MakeTensor({batch, 1, seq_rows, kv_max}, dt));
        s.name_index["mask_full"] = 0;
        s.tensors.push_back(MakeTensor({batch, 1, seq_rows, w}, dt));
        s.name_index["mask_sliding"] = 1;
        s.tensors.push_back(MakeTensor({batch, seq_rows}, DataType::INT32));
        s.name_index["position_ids"] = 2;
        s.tensors.push_back(MakeTensor({batch, seq_rows}, DataType::INT32));
        s.name_index["new_kv_cache_pos"] = 3;
        if (with_spos) {
            // 与 new_kv_cache_pos 同形口径（一维 [rows]，batch>1 图形为
            // [batch,1]——assembler 只按容量消费、值语义 per 行/lane 不分维度；
            // decode 用例 rows=batch 维须给足容量，故容量按 max(seq_rows,batch)）
            const int cap = (batch > seq_rows) ? batch : seq_rows;
            s.tensors.push_back(MakeTensor({cap}, DataType::INT32));
            s.name_index["sliding_kv_pos"] = 4;
        }
        return s;
    }

    std::shared_ptr<ITensorBuffer>& at(const std::string& name) { return tensors[name_index.at(name)]; }
};

// ════════════════════════════════════════════════════════════════════════
// prefill 填充测试
// ════════════════════════════════════════════════════════════════════════

// prefill: position_ids = cursor_start..cursor_start+chunk_len-1
TEST(InputAssemblerTest, PrefillFillsPositionIds)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // cursor=0, chunk_len=3；attention_mask 修形为全 kv_max=16 宽
    // （float 工位静态全宽，seq 维 rows=3 由 mask 容量推导）
    auto inputs = InputSet::Make(1, 3, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    EXPECT_EQ(ReadI32(pos, 0), 0); // cursor_start=0
    EXPECT_EQ(ReadI32(pos, 1), 1);
    EXPECT_EQ(ReadI32(pos, 2), 2);
    EXPECT_EQ(f.pool->NextPosition(slot), 3u);
}

// prefill: new_kv_cache_pos = position_ids（相同值，但不同 tensor）
TEST(InputAssemblerTest, PrefillFillsNewKvCachePos)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    // mask 全 kv_max=16 宽静态工位（rows=2 由容量推导）
    auto inputs = InputSet::Make(1, 2, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    auto& new_pos = inputs.at("new_kv_cache_pos");
    // position_ids: 2D [1, 2] → [5, 6]
    EXPECT_EQ(ReadI32(pos, 0), 5);
    EXPECT_EQ(ReadI32(pos, 1), 6);
    // new_kv_cache_pos: 1D [2] → [5, 6]（相同值，不同 tensor）
    EXPECT_EQ(ReadI32(new_pos, 0), 5);
    EXPECT_EQ(ReadI32(new_pos, 1), 6);
}

// prefill: attention_mask causal: mask[0][0][i][j] = 0 if j<=cursor+i else -FLT_MAX
TEST(InputAssemblerTest, PrefillFillsAttentionMaskCausal)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // cursor=0, chunk_len=3；mask 全 kv_max=16 宽
    // （float 工位静态全宽，rows=3 由 mask 容量推导）
    auto inputs = InputSet::Make(1, 3, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // 行 r=16 列全宽布局：row=r*16+j；因果 → j<=r 全 0、j>r 全 -FLT_MAX
    for (int r = 0; r < 3; ++r) {
        for (int j = 0; j < 16; ++j) {
            const bool attend = j <= r;
            EXPECT_FLOAT_EQ(ReadF32(mask, r * 16 + j), attend ? 0.0f : -FLT_MAX) << "row=" << r << " j=" << j;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// decode 填充测试
// ════════════════════════════════════════════════════════════════════════

// decode: position_ids = 各 slot 的 cursor
TEST(InputAssemblerTest, DecodeFillsPositionIds)
{
    UT_NEED_BACKEND();
    Fixture f;
    std::vector<SlotId> slots;
    for (int i = 0; i < 2; ++i) {
        SlotId s = -1;
        ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
        slots.push_back(s);
    }
    // cursor=0 for both slots；mask 全 kv_max=16 宽
    auto inputs = InputSet::Make(2, 1, 16);
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    EXPECT_EQ(ReadI32(pos, 0), 0); // slot0 cursor=0
    EXPECT_EQ(ReadI32(pos, 1), 0); // slot1 cursor=0
    for (SlotId s : slots)
        EXPECT_EQ(f.pool->NextPosition(s), 1u);
}

// decode: new_kv_cache_pos = 各 slot 的 cursor（独立 tensor）
TEST(InputAssemblerTest, DecodeFillsNewKvCachePos)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // cursor=3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // cursor=0
    std::vector<SlotId> slots = {s0, s1};
    // mask 全 kv_max=16 宽；attention 语义列断言例见
    // DecodeFillsAttentionMaskCausalPerSlot。注意本例 position_ids/new_kv_cache_pos
    // 的 [batch=2, seqLen=1] 断言不变——两值无绕回时同 abs。
    auto inputs = InputSet::Make(2, 1, 16);
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    auto& new_pos = inputs.at("new_kv_cache_pos");
    // position_ids: [batch=2, seqLen=1] → [3, 0]
    EXPECT_EQ(ReadI32(pos, 0), 3); // s0 cursor=3
    EXPECT_EQ(ReadI32(pos, 1), 0); // s1 cursor=0
    // new_kv_cache_pos: [batch, seqLen=1] per-slot → [3, 0]（各 slot 各自 cursor）
    EXPECT_EQ(ReadI32(new_pos, 0), 3); // s0 cursor=3
    EXPECT_EQ(ReadI32(new_pos, 1), 0); // s1 cursor=0
}

// decode 绕回后（Fix-1，code_review (c)-1）：position_ids 填绝对位、
// new_kv_cache_pos 必须填物理槽 MapSlot(abs)（≠ abs）——float 路径旧实现
// 两工位共填 abs，绕回后 abs ≥ kv_max 即越界 scatter。
// kv_max=16、init_token_len=8 → ring=8：18 = 8+(18-8)%8 = 槽 10（已绕回）。
TEST(InputAssemblerTest, DecodeWraparoundFillsPhysicalSlotNotAbs)
{
    UT_NEED_BACKEND();
    Fixture f; // kv_max=16 (max_seq_len)
    SlotId s = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
    f.pool->SetInitTokenLen(8);               // 锚定段 [0,8)，尾圆环 ring=8
    ASSERT_TRUE(f.pool->Advance(s, 18).ok()); // total_written=18 ≥ kv_max=16（绕回）
    // 池口径自证：abs=18、MapSlot(18)=10（8+(18-8)%8=8+2，绕回加回 anchor）
    ASSERT_EQ(f.pool->TotalWritten(s), 18u);
    ASSERT_EQ(f.pool->NextPosition(s), 10u);
    ASSERT_EQ(f.pool->MapSlot(18), 10u);

    std::vector<SlotId> slots = {s};
    auto inputs = InputSet::Make(1, 1, 16); // batch=1, seqLen=1, kvCacheLen=16
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    auto& new_pos = inputs.at("new_kv_cache_pos");
    EXPECT_EQ(ReadI32(pos, 0), 18);     // 绝对位（RoPE 逻辑位，绕回后仍单调）
    EXPECT_EQ(ReadI32(new_pos, 0), 10); // 物理槽（scatter 写位，恒 < kv_max）
}

// decode: attention_mask causal per slot（不同 cursor 的 slot 尾部遮蔽）
TEST(InputAssemblerTest, DecodeFillsAttentionMaskCausalPerSlot)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // cursor=3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // cursor=0
    std::vector<SlotId> slots = {s0, s1};
    // mask 全 kv_max=16 宽；batch=2 → rows=2
    auto inputs = InputSet::Make(2, 1, 16);
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // 行 b 起 b*16 列；行 b attend 逻辑位 j ∈ [0, cursor_b]，无绕回 MapSlot 恒等
    for (int b = 0; b < 2; ++b) {
        const int cursor = (b == 0) ? 3 : 0;
        for (int j = 0; j < 16; ++j) {
            const bool attend = j <= cursor;
            EXPECT_FLOAT_EQ(ReadF32(mask, b * 16 + j), attend ? 0.0f : -FLT_MAX) << "batch=" << b << " j=" << j;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// 异常路径
// ════════════════════════════════════════════════════════════════════════

TEST(InputAssemblerTest, EmptySlotsRejected)
{
    Fixture f;
    auto inputs = InputSet::Make(1, 1, 1);
    std::vector<SlotId> empty;
    EXPECT_EQ(f.asm_->AssembleDecode(empty, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

TEST(InputAssemblerTest, MissingPositionIdsNameRejected)
{
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // name_index 缺 position_ids
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index.erase("position_ids");
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// prefill cursor>0：mask 前 cursor 列全可见、尾部阶梯
TEST(InputAssemblerTest, PrefillAttentionMaskCausalCursorOffset)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    // chunk_len=2；mask 全 kv_max=16 宽，rows=2
    auto inputs = InputSet::Make(1, 2, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // 行 r=16 列全宽；行 r attend 逻辑位 j ∈ [0, 5+r]（无绕回 MapSlot 恒等）
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 16; ++j) {
            const bool attend = j <= 5 + r;
            EXPECT_FLOAT_EQ(ReadF32(mask, r * 16 + j), attend ? 0.0f : -FLT_MAX) << "row=" << r << " j=" << j;
        }
    }
}

// 异常：缺 new_kv_cache_pos / attention_mask 名 → INVALID_ARGUMENT
TEST(InputAssemblerTest, MissingNewKvCachePosNameRejected)
{
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index.erase("new_kv_cache_pos");
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

TEST(InputAssemblerTest, MissingAttentionMaskNameRejected)
{
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index.erase("attention_mask");
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// 异常：name_index 指向越界 index → INVALID_ARGUMENT
TEST(InputAssemblerTest, IndexOutOfBoundsRejected)
{
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    // position_ids 指向不存在的 index
    inputs.name_index["position_ids"] = 999;
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// 异常：prefill slot<0 → INVALID_ARGUMENT
TEST(InputAssemblerTest, PrefillNegativeSlotRejected)
{
    Fixture f;
    auto inputs = InputSet::Make(1, 1, 1);
    EXPECT_EQ(f.asm_->AssemblePrefill(-1, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// ════════════════════════════════════════════════════════════════════════
// gemma4 双 bool mask 测试（docs/mask_fill.md D1~D8）
// ════════════════════════════════════════════════════════════════════════

// prefill：full=绝对因果 / sliding=滑窗小环。cursor=0, chunk=6, W=4、P=4 ⇒ C=8
// （新契 kSliding 纯模：物理槽 = 逻辑位 j%C）：
// full 行 r attend [0,r]（16 列）；sliding 行 attend 窗口 [max(0,r-3), r] 经
// j%8 落槽——r<3 全部 [0,r]（无绕回）、r≥3 窗口 [r-3,r] ⊂ [0,7] 亦无绕回
// （q ≤ 7 < C 才绕回；本例 q ≤ 5 → 全部窗口原样落位）。行 attend 计数
// min(q+1, W) 是 §2.3 口径侧证。
TEST(GemmaMaskTest, PrefillFullAndSlidingMask)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16); // rows=6, kv_max=16, sliding 列宽 C=8
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());

    auto& full = inputs.at("mask_full");
    auto& sliding = inputs.at("mask_sliding");
    for (int r = 0; r < 6; ++r) {
        ExpectRow(full, r, 16, 0, r);
        // attend 窗 [max(0,r-W+1), r] 全在列宽 8 内（q≤5<C）→ 直接区间落位
        const int lo = std::max(0, r - 3);
        ExpectRow(sliding, r, 8, lo, r);
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 6u);
}

// prefill cursor>0：q_abs=cursor+i（KV buffer 绝对位语义）。W=4、C=8：q=5/6 的
// 窗口 [2,5]/[3,6] 全部 < C=8 → 纯模恒等落位，sliding 行清 [2,5]/[3,6]——
// 与 full 行对照可观察"满窗但非全清"（列 0/1/6/7 屏蔽——新契 C=2W 下满窗行
// 只有 W 列 attend，b688221 j%w"满窗全清"判据不复存在）。
TEST(GemmaMaskTest, PrefillCursorOffset)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    auto inputs = GemmaInputSet::Make(1, 2, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 5);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 1), 6);
    // full：q=5/6 → attend [0,5] / [0,6]
    ExpectRow(inputs.at("mask_full"), 0, 16, 0, 5);
    ExpectRow(inputs.at("mask_full"), 1, 16, 0, 6);
    // sliding C=8 小环：窗口 [2,5]/[3,6] 无绕回恒等落位——满窗恰 4 列 attend
    ExpectRow(inputs.at("mask_sliding"), 0, 8, 2, 5);
    ExpectRow(inputs.at("mask_sliding"), 1, 8, 3, 6);
}

// 残留覆盖：buffer 预填 0x00（stale 全 attend）→ 全量重写后屏蔽位必须回 1
// （新契 C=8 列：行 r=0/1/2 attend [0,r]（部分窗），屏蔽位 r+1..7 回 1——
// 残留 0 的列 4..7 也必须重写回 1）
TEST(GemmaMaskTest, PrefillOverwritesResidue)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 3, 16);
    std::memset(inputs.at("mask_full")->GetData(), 0x00, inputs.at("mask_full")->GetSize());
    std::memset(inputs.at("mask_sliding")->GetData(), 0x00, inputs.at("mask_sliding")->GetSize());

    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());
    for (int r = 0; r < 3; ++r) {
        ExpectRow(inputs.at("mask_full"), r, 16, 0, r);
        ExpectRow(inputs.at("mask_sliding"), r, 8, 0, r); // 部分窗（r<W-1）
    }
}

// 尾块 padding：末块 chunk 可短于图行数 → padding 行 [chunk, rows) 保持初始全 1
// （golden 对齐：mask 先全置 1、只清真实行 attend 区间；原 D5 复制末有效行已删除）
TEST(GemmaMaskTest, PrefillPaddingRowsAllMasked)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16); // 图行数=6，实发 chunk=3
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    auto& full = inputs.at("mask_full");
    auto& sliding = inputs.at("mask_sliding");
    for (int r = 0; r < 3; ++r) {
        ExpectRow(full, r, 16, 0, r);
        ExpectRow(sliding, r, 8, 0, r); // 部分窗（r<W-1）
    }
    // padding 行 3..5 = 全屏蔽（attend 区间空集，lo>hi）
    for (int r = 3; r < 6; ++r) {
        ExpectRow(full, r, 16, 1, 0);
        ExpectRow(sliding, r, 8, 1, 0);
    }
}

// decode：per-slot 光标 → per-lane mask 行（q=cursor）。W=2、P=2 ⇒ C=4：
// lane0 q=3 窗口 [2,3] ⊂ [0,4) 恒等落位；lane1 q=0 → [0,0]
TEST(GemmaMaskTest, DecodePerSlotMask)
{
    UT_NEED_BACKEND();
    GemmaFixture f(2); // W=2, C=4
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // lane0 cursor=3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // lane1 cursor=0
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = GemmaInputSet::Make(2, 1, 16, DataType::BOOL, /*sliding_width=*/4);
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 3);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 1), 0);
    // full：lane0 q=3 attend [0,3]；lane1 q=0 attend [0,0]
    ExpectRow(inputs.at("mask_full"), 0, 16, 0, 3);
    ExpectRow(inputs.at("mask_full"), 1, 16, 0, 0);
    // sliding C=4：lane0 q=3 窗口 [2,3]（满窗 2 列 attend、列 0/1 屏蔽）；
    // lane1 q=0 → [0,0]
    ExpectRow(inputs.at("mask_sliding"), 0, 4, 2, 3);
    ExpectRow(inputs.at("mask_sliding"), 1, 4, 0, 0);
    EXPECT_EQ(f.pool->NextPosition(s0), 4u);
    EXPECT_EQ(f.pool->NextPosition(s1), 1u);
}

// decode inactive lane：padded batch 行 [batch, rows) 保持初始全 1
// （golden 对齐：与 prefill padding 行同口径；原 D5 复制 lane 0 已删除）
TEST(GemmaMaskTest, DecodeInactiveLanesAllMasked)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 2).ok());   // lane0 cursor=2
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // lane1 cursor=0
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = GemmaInputSet::Make(4, 1, 16); // rows=4（decode_batch_size），active=2
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    ExpectRow(inputs.at("mask_full"), 0, 16, 0, 2);
    ExpectRow(inputs.at("mask_full"), 1, 16, 0, 0);
    ExpectRow(inputs.at("mask_sliding"), 0, 8, 0, 2); // q=2, W=4 → [0,2]
    ExpectRow(inputs.at("mask_sliding"), 1, 8, 0, 0);
    // inactive lane 2/3 = 全屏蔽（attend 区间空集，lo>hi）
    for (int r = 2; r < 4; ++r) {
        ExpectRow(inputs.at("mask_full"), r, 16, 1, 0);
        ExpectRow(inputs.at("mask_sliding"), r, 8, 1, 0);
    }
}

// ════════════════════════════════════════════════════════════════════════
// 滑窗小环恰好绕回最小用例（sliding_kv_seq_dim_rebind 方案 §5 U2/U3-b；吸取
// 环形缓冲坐标空间塌缩审查经验——滑窗环绕回点 abs≥C 后常规短序列即触发，
// U2/U3 必须显式覆盖绕回边界 C 与 C+1）
// ════════════════════════════════════════════════════════════════════════

// U2：滑窗环（新契 C=W+P=8）行为表——attend 列集恒 = {j%C | j ∈ [max(0,q-W+1), q]}：
//   q<W-1：部分窗 [0,q]；W-1≤q<C：满窗 [q-W+1,q] 恰 W 列连续（无绕回）；
//   q≥C：绕回——q%C < W-1 时窗口分裂成首尾两段（如 q=8：逻辑 {5..8} → 槽
//   {5,6,7,0}，列 1..4 屏蔽）；q%C ≥ W-1 时仍连续（如 q=11：槽 {0,1,2,3}）。
// 逻辑位集大小 = min(q+1, W) 与覆盖性 §2.3 口径一致。逐行按定义独立演算断言。
TEST(GemmaMaskTest, SlidingRingWindowBehaviors_U2)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, P=4, C=8, kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // q_abs 0..11：跨首满窗 q=3（W-1）、环绕回 q=8=C、分裂态 q=8/9/10、连续态 q=11
    auto inputs = GemmaInputSet::Make(1, 12, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 12, inputs.tensors, inputs.name_index).ok());

    auto& full = inputs.at("mask_full");
    auto& sliding = inputs.at("mask_sliding");
    for (int r = 0; r < 12; ++r) {
        ExpectRow(full, r, 16, 0, r); // full 行 r 恒全因果 [0,r]（MapSlot 恒等）
        // sliding：窗口 [max(0,r-3), r] 槽集 = {j%8 | j ∈ 窗口}
        const int wl = std::max(0, r - 3);
        for (int s = 0; s < 8; ++s) {
            bool present = false;
            for (int j = wl; j <= r; ++j)
                present |= (j % 8 == s);
            EXPECT_EQ(ReadBool(sliding, r * 8 + s), present ? 0 : 1) << "r=" << r << " col=" << s;
        }
    }
}

// U3a：双环独立绕回——滑窗环（C=8）先于 kv_max 环（16）绕回。
// Advance 17（>kv_max=16，两环都已绕回）后 decode 装配：full 行走 kv_max
// 锚定环（MapSlot 映射，anchor=0 纯模）；sliding 行 = 窗口 [14,17] mod 8
// → 槽 {6,7,0,1}（分裂态：首尾两段，与 kv 环绕回状态无关）。
TEST(GemmaMaskTest, SlidingRingWrapsIndependentlyOfKvRing_U3)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId s = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
    ASSERT_TRUE(f.pool->Advance(s, 17).ok()); // abs=17：sliding 环已绕 2 圈、
                                              // kv 环已绕 1 圈（MapSlot(17)=1）
    std::vector<SlotId> slots = {s};
    auto inputs = GemmaInputSet::Make(1, 1, 16);
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    // full：anchor=0 纯模环，q=17 ≥ kv_max-1=15 → 逻辑位 [0,17] 映射覆盖
    // 全环 16 列（多逻辑位同槽幂等，数值自恰）
    for (int j = 0; j < 16; ++j) {
        EXPECT_EQ(ReadBool(inputs.at("mask_full"), j), 0) << "full j=" << j;
    }
    // sliding：q=17 窗口 [14,17] mod 8 = 槽 {6,7,0,1}（分裂态——首尾两段，
    // 纯模小环自身绕回语义、与 kv 环无关）
    for (const int j : {0, 1, 6, 7}) {
        EXPECT_EQ(ReadBool(inputs.at("mask_sliding"), j), 0) << "sliding attend j=" << j;
    }
    for (const int j : {2, 3, 4, 5}) {
        EXPECT_EQ(ReadBool(inputs.at("mask_sliding"), j), 1) << "sliding mask j=" << j;
    }
}

// U3b：恰好绕回边界 C 与 C+1（最小绕回对）——abs_start=C=8 时块首行 q=8
// （滑窗环首行即绕回：窗口 [5,8] mod 8 → 槽 {5,6,7,0} 分裂态）；abs_start=9
// 同分裂形态（窗口 [6,9] → {6,7,0,1}）。full 环对照：abs=8/9 均 < kv_max-1=15
// 两行仍部分因果——两环绕回点不同的直接观测。
TEST(GemmaMaskTest, SlidingRingExactWrapEdgeWAndWPlusOne_U3)
{
    UT_NEED_BACKEND();
    for (const std::size_t cursor : {8u, 9u}) { // abs_start = C、C+1
        GemmaFixture f;                         // W=4, C=8, kv_max=16
        SlotId slot = -1;
        ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
        if (cursor) {
            ASSERT_TRUE(f.pool->Advance(slot, cursor).ok());
        }
        auto inputs = GemmaInputSet::Make(1, 2, 16);
        ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok()) << "cursor=" << cursor;

        // full：q=cursor+r（≤10 < kv_max-1=15）→ 16 列部分因果 [0, q]
        ExpectRow(inputs.at("mask_full"), 0, 16, 0, static_cast<int>(cursor));
        ExpectRow(inputs.at("mask_full"), 1, 16, 0, static_cast<int>(cursor + 1));
        // sliding：窗口 [q-3, q] mod 8 —— 绕回分裂态（cursor=8：行0 {5,6,7,0}、
        // 行1 {6,7,0,1}；cursor=9：行0 {6,7,0,1}、行1 {7,0,1,2}）
        for (int rr = 0; rr < 2; ++rr) {
            const int q = static_cast<int>(cursor) + rr;
            const int wl = q - 3;
            for (int s = 0; s < 8; ++s) {
                bool present = false;
                for (int j = wl; j <= q; ++j)
                    present |= (j % 8 == s);
                EXPECT_EQ(ReadBool(inputs.at("mask_sliding"), static_cast<std::size_t>(rr) * 8 + s), present ? 0 : 1)
                    << "cursor=" << cursor << " row=" << rr << " col=" << s;
            }
        }
    }
}

// 退化：旧契（sw_hd=0）W=0 → window=0 自然全因果，sliding 与 full 逐字节
// 相同（pre-b688221 语义回归锁：全因果 = kAnchored 全宽 kv_max 工位同 map
// 同窗口——D9 废"宽值判据"后，W≥kv_max 的等宽退化不可达也无需构造，
// 仅保留 W=0 全因果分支）。新契不可能 W=0（D10：sw_hd>0 须 sw>0，配置期
// 已拒），无需对称用例。
TEST(GemmaMaskTest, DegenerateWindowEqualsFull)
{
    UT_NEED_BACKEND();
    GemmaFixture f(0, /*legacy=*/true); // 旧契 + W=0 全因果
    {
        SlotId slot = -1;
        ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
        auto inputs = GemmaInputSet::Make(1, 4, 16, DataType::BOOL,
                                          /*sliding_width=*/16); // 旧契 kv_max 宽工位
        ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 4, inputs.tensors, inputs.name_index).ok());
        auto& full = inputs.at("mask_full");
        auto& slid = inputs.at("mask_sliding");
        ASSERT_EQ(full->GetSize(), slid->GetSize());
        EXPECT_EQ(0, std::memcmp(full->GetData(), slid->GetData(), full->GetSize()));
    }
    {
        // 旧契常窗 W=4 回归锁：kv_max 宽 kAnchored + window=W 截断——
        // 行 r attend [max(0,r-3), r]（MapSlot 恒等无绕回），列 (r-3 前全屏蔽)
        GemmaFixture lf(4, /*legacy=*/true);
        SlotId slot = -1;
        ASSERT_TRUE(lf.pool->AllocKvSlot(&slot).ok());
        auto inputs = GemmaInputSet::Make(1, 6, 16, DataType::BOOL,
                                          /*sliding_width=*/16);
        ASSERT_TRUE(lf.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
        auto& full = inputs.at("mask_full");
        auto& slid = inputs.at("mask_sliding");
        for (int r = 0; r < 6; ++r) {
            ExpectRow(full, r, 16, 0, r);
            ExpectRow(slid, r, 16, std::max(0, r - 3), r); // 全因果 vs 滑窗截断
        }
    }
}

// 槽复用残留：Free→Alloc（光标重置）后短 chunk 重新 prefill，
// 上一轮 attend 区间的旧 0 必须被重写为 1（残留 0 = 意外 attend 垃圾 KV）；
// padding 行同时保持全 1（golden 口径）
TEST(GemmaMaskTest, SlotReuseResidueMasked)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // w=4
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    // 第一轮：chunk=6 → q_abs=5..10，attend 区间在 buffer 留痕
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
    // 复用：Free → Alloc（cursor 重置 0）
    ASSERT_TRUE(f.pool->FreeKvSlot(slot).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // 第二轮：chunk=2 → 行 0/1 q=0/1 清 attend；行 2..5 padding 保持全 1
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    ExpectRow(inputs.at("mask_full"), 0, 16, 0, 0); // 旧 [0,5] attend → byte 1..5 回 1
    ExpectRow(inputs.at("mask_full"), 1, 16, 0, 1);
    ExpectRow(inputs.at("mask_sliding"), 0, 8, 0, 0); // 旧滑窗 attend 残留 → 全量重写后
    ExpectRow(inputs.at("mask_sliding"), 1, 8, 0, 1); // 仅 [0,0]/[0,1] 清 0、列 2..7 回 1
    for (int r = 2; r < 6; ++r) {
        ExpectRow(inputs.at("mask_full"), r, 16, 1, 0);
        ExpectRow(inputs.at("mask_sliding"), r, 8, 1, 0);
    }
}

// INT8 dtype 容忍：1 字节/元素，0/1 原样写入
TEST(GemmaMaskTest, MaskInt8Tolerated)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 2, 16, DataType::INT8);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());
    ExpectRow(inputs.at("mask_full"), 0, 16, 0, 0);
    ExpectRow(inputs.at("mask_full"), 1, 16, 0, 1);
    ExpectRow(inputs.at("mask_sliding"), 0, 8, 0, 0); // 部分窗（r<W-1）
    ExpectRow(inputs.at("mask_sliding"), 1, 8, 0, 1);
}

// dtype 防线：非 BOOL/INT8/FLOAT32/FLOAT16 → INVALID_ARGUMENT 且零半写
// （buffer 保持预填值）、光标不推进。模板化统一入口后 BOOL/INT8/FLOAT32/FLOAT16
// 均可走 gemma 双 mask 路径——故用 INT32 测真正未支持的 dtype（mask_dtype_template_support U6）。
TEST(GemmaMaskTest, MaskDtypeRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 2, 16, DataType::INT32); // 非法 dtype
    auto* full = static_cast<uint8_t*>(inputs.at("mask_full")->GetData());
    const std::size_t bytes = inputs.at("mask_full")->GetSize();
    std::memset(full, 0xAA, bytes);
    std::memset(inputs.at("mask_sliding")->GetData(), 0xAA, inputs.at("mask_sliding")->GetSize());

    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    for (std::size_t i = 0; i < bytes; ++i)
        EXPECT_EQ(full[i], 0xAA) << "i=" << i;
    EXPECT_EQ(f.pool->NextPosition(slot), 0u);
}

// ── fp16 mask 路径（mask_dtype_template_support U3-U5）─────────────────────
// fp16 哨兵：attend=0x0000（+0.0h）、block=0xFC00（-inf，softmax exp(-inf)=0
// 精确屏蔽）。以下用例覆盖基础写入 / 短路 / 滑窗三分支。

// U3: fp16 基础全因果写入——attend 列=0x0000、屏蔽列=0xFC00、padding 行全屏蔽
TEST(GemmaMaskTest, MaskFp16Basic)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 2, 16, DataType::FLOAT16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());
    auto& full = inputs.at("mask_full");
    auto& slid = inputs.at("mask_sliding");
    // 行 0：q=0 → attend [0,0]、其余屏蔽；行 1：q=1 → attend [0,1]
    for (int j = 0; j < 16; ++j) {
        EXPECT_EQ(ReadF16Bits(full, 0 * 16 + j), j <= 0 ? 0x0000u : 0xFC00u) << "row0 j=" << j;
        EXPECT_EQ(ReadF16Bits(full, 1 * 16 + j), j <= 1 ? 0x0000u : 0xFC00u) << "row1 j=" << j;
    }
    // sliding C=8：行 0 attend [0,0]、行 1 attend [0,1]（r<W-1=3 部分窗）
    for (int j = 0; j < 8; ++j) {
        EXPECT_EQ(ReadF16Bits(slid, 0 * 8 + j), j <= 0 ? 0x0000u : 0xFC00u) << "slid0 j=" << j;
        EXPECT_EQ(ReadF16Bits(slid, 1 * 8 + j), j <= 1 ? 0x0000u : 0xFC00u) << "slid1 j=" << j;
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 2u);
}

// U4: fp16 短路——q>=width-1 时整行全 attend（0x0000）
TEST(GemmaMaskTest, MaskFp16ShortCircuit)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // mask_full width=kv_max=16；填 16 行使 q=15 = width-1 触发短路
    auto inputs = GemmaInputSet::Make(1, 16, 16, DataType::FLOAT16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 16, inputs.tensors, inputs.name_index).ok());
    auto& full = inputs.at("mask_full");
    // 末行 q=15 ≥ width-1=15 → 整行 16 列全 attend
    for (int j = 0; j < 16; ++j) {
        EXPECT_EQ(ReadF16Bits(full, 15 * 16 + j), 0x0000u) << "short-circuit row j=" << j;
    }
}

// U5: fp16 滑窗——window=W、kSliding 纯模 j%width、attend 区间=[q-W+1, q]
TEST(GemmaMaskTest, MaskFp16Sliding)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // 6 行：r=0..5，W=4，C=8；行 r attend 逻辑位 [max(0,r-3), r] mod 8
    auto inputs = GemmaInputSet::Make(1, 6, 16, DataType::FLOAT16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
    auto& slid = inputs.at("mask_sliding");
    for (int r = 0; r < 6; ++r) {
        const int lo = std::max(0, r - 3);
        for (int j = 0; j < 8; ++j) {
            const bool attend = j >= lo && j <= r; // r<8 无绕回，mod 恒等
            EXPECT_EQ(ReadF16Bits(slid, r * 8 + j), attend ? 0x0000u : 0xFC00u) << "r=" << r << " j=" << j;
        }
    }
}

// U7: fp16 容量失配——mask_full 容量 != rows×width×2 → INVALID_ARGUMENT
TEST(GemmaMaskTest, MaskFp16SizeMismatchRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // rows 按 full 推导 = 2，sliding 换 3 行×8 列（48 != 2×8×2=32 字节）失配
    auto inputs = GemmaInputSet::Make(1, 2, 16, DataType::FLOAT16);
    inputs.tensors[inputs.name_index["mask_sliding"]] = MakeTensor({1, 1, 3, 8}, DataType::FLOAT16);
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(f.pool->NextPosition(slot), 0u);
}

// U8: fp16 绕回——abs_start 跨 kv_max 绕回，MapSlot 映射后 attend 集正确
// （对偶 bool 绕回用例；kv_max=16、advance 12、chunk 6 → q=12..17、MapSlot(16)=0..）
TEST(GemmaMaskTest, MaskFp16Wraparound)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 12).ok()); // cursor=12
    auto inputs = GemmaInputSet::Make(1, 6, 16, DataType::FLOAT16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
    auto& full = inputs.at("mask_full");
    // full 全因果：行 r 的 q=12+r，attend 逻辑位 [0, 12+r] 经 MapSlot 映射物理槽。
    // q=12..17 ≥ kv_max-1=15 → 行 r≥3 (q≥15) 整行全 attend 短路；
    // 行 r=0 (q=12): attend 逻辑位 [0,12] → MapSlot 槽 {0..12}，列 13/14/15 屏蔽
    // 行 r=1 (q=13): attend {0..13}，列 14/15 屏蔽
    // 行 r=2 (q=14): attend {0..14}，列 15 屏蔽
    for (int r = 0; r < 3; ++r) {
        const int q = 12 + r;
        for (int j = 0; j < 16; ++j) {
            // 无绕回时 MapSlot 恒等；q<15 未触发短路
            const bool attend = j <= q;
            EXPECT_EQ(ReadF16Bits(full, r * 16 + j), attend ? 0x0000u : 0xFC00u) << "r=" << r << " j=" << j;
        }
    }
    // 行 r=3..5 (q=15..17) 短路：整行全 attend
    for (int r = 3; r < 6; ++r) {
        for (int j = 0; j < 16; ++j) {
            EXPECT_EQ(ReadF16Bits(full, r * 16 + j), 0x0000u) << "short-circuit r=" << r << " j=" << j;
        }
    }
}

// 尺寸防线：mask 工位容量 != rows×width → INVALID_ARGUMENT（禁部分写留 stale 0）。
// 新契（U4）：rows 从 mask_full 推导（32B/16=2），sliding 期望 rows×C=2×8=16
// 字节——3 行×8 列=24 ≠ 16 失配拒；新契 + 残 kv_max 宽工位（2×16=32）同样拒。
// 旧契对照：sliding 期望 rows×kv_max=32，4 行宽的旧形歧义工位（3×8=24≠32）拒。
TEST(GemmaMaskTest, MaskSizeMismatchRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    { // 行数失配：rows 按 full 取 2，sliding 换 3 行×8 列（24 != 2×8）
        auto inputs = GemmaInputSet::Make(1, 2, 16);
        inputs.tensors[inputs.name_index["mask_sliding"]] = MakeTensor({1, 1, 3, 8}, DataType::BOOL);
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
        EXPECT_EQ(f.pool->NextPosition(slot), 0u); // 失败不 Advance
    }
    { // 旧形 kv_max 宽工位（2×16=32 != 2×C=16）：新契 + 未修形旧工位失配拒
        auto inputs = GemmaInputSet::Make(1, 2, 16, DataType::BOOL,
                                          /*sliding_width=*/16);
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
        EXPECT_EQ(f.pool->NextPosition(slot), 0u);
    }
}

// 工位组合防线：mask 工位都缺 / 只有其一 / 三者并存 → INVALID_ARGUMENT
TEST(GemmaMaskTest, InvalidMaskComboRejected)
{
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    { // 仅 mask_full
        auto inputs = GemmaInputSet::Make(1, 1, 16);
        inputs.name_index.erase("mask_sliding");
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    }
    { // 三者并存（gemma 图上混入旧 attention_mask 名）
        auto inputs = GemmaInputSet::Make(1, 1, 16);
        inputs.tensors.push_back(MakeTensor({16}, DataType::FLOAT32));
        inputs.name_index["attention_mask"] = inputs.tensors.size() - 1;
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 1, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 0u); // 失败不 Advance
}

// 旧 float 路径零影响：sliding_window 无论何值，attention_mask 图行为不变
TEST(InputAssemblerTest, FloatPathUnaffectedBySlidingWindow)
{
    UT_NEED_BACKEND();
    auto pool = std::make_shared<KvCachePool>(4, 16, 2, 4, 8);
    InputAssembler asm_(pool, 4, 512); // 大滑窗也不得影响 float 图
    SlotId slot = -1;
    ASSERT_TRUE(pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(pool->Advance(slot, 3).ok()); // cursor=3
    // mask 全 kv_max=16 宽，rows=2
    auto inputs = InputSet::Make(1, 2, 16);
    ASSERT_TRUE(asm_.AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // 行 r=16 列全宽；行 r attend 逻辑位 j ∈ [0, 3+r]（window 不参与 float 路径）
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 16; ++j) {
            const bool attend = j <= 3 + r;
            EXPECT_FLOAT_EQ(ReadF32(mask, r * 16 + j), attend ? 0.0f : -FLT_MAX) << "row=" << r << " j=" << j;
        }
    }
}

// ── 辅助：读取 output_pos 标量值（INT32=4字节 / INT64=8字节）────────────
// 与冒烟脚本一致：零基初始化读回 bytes 字节，返回非负整数值。
static std::int64_t ReadOutputPosScalar(const std::shared_ptr<ITensorBuffer>& t, int bytes)
{
    std::int64_t v = 0;
    if (t && t->GetData() && bytes > 0 && bytes <= 8 && static_cast<int>(t->GetSize()) >= bytes)
        std::memcpy(&v, t->GetData(), bytes);
    else
        return -1;
    return v;
}

// ── 带 output_pos 工位的 gemma 输入组 ───────────────────────────────────
// rows=8, kv_max=16（与 GemmaFixture 一致）；op_dt=RESERVED 时 tensor 为 nullptr 且不入 name_index。
// mask_sliding 列宽 = 新契滑窗环宽 C（与 GemmaFixture 默认同源，U4 口径）。
struct GemmaInputSetWithOutputPos {
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;

    static GemmaInputSetWithOutputPos Make(DataType op_dt)
    {
        GemmaInputSetWithOutputPos s;
        const int rows = 8, kv_max = 16;
        s.tensors.push_back(MakeTensor({1, 1, rows, kv_max}, DataType::BOOL));
        s.name_index["mask_full"] = 0;
        // mask_sliding 列宽 = C=W+P=8 小环（与 GemmaFixture 默认同源，U4 口径）
        s.tensors.push_back(MakeTensor({1, 1, rows, kGemmaSlidingRingC}, DataType::BOOL));
        s.name_index["mask_sliding"] = 1;
        s.tensors.push_back(MakeTensor({1, rows}, DataType::INT32));
        s.name_index["position_ids"] = 2;
        s.tensors.push_back(MakeTensor({rows}, DataType::INT32));
        s.name_index["new_kv_cache_pos"] = 3;
        if (op_dt != DataType::RESERVED) {
            s.tensors.push_back(MakeTensor({1}, op_dt));
            s.name_index["output_pos"] = 4;
        }
        return s;
    }

    std::shared_ptr<ITensorBuffer>& at(const std::string& name) { return tensors[name_index.at(name)]; }
};

// ════════════════════════════════════════════════════════════════════════
// output_pos 可选工位填充测试（docs/mask_fill.md §4）
// ════════════════════════════════════════════════════════════════════════

// prefill INT32: output_pos == chunk_len-1（chunk 内相对行号，不随 KV 光标平移）
TEST(OutputPosTest, PrefillInt32)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 4) << "output_pos == chunk_len-1 == 4";
    // position_ids[4] == cursor+4 == 4（cursor=0）
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 4), 4);
    EXPECT_EQ(f.pool->NextPosition(slot), 5u);
}

// 第二 chunk prefill: output_pos 仍 == chunk_len-1，不随光标平移
// position_ids 随光标平移但 output_pos 保持 chunk 相对行号
TEST(OutputPosTest, PrefillSecondChunkRelative)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);

    // 第一 chunk: cursor=0, chunk_len=5
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).ok());
    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 4);

    // 第二 chunk: cursor=5, chunk_len=5 → output_pos 仍为 4
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).ok());
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 4) << "output_pos 仍为 4 (chunk-relative)";
    // 但 position_ids[4] == cursor+4 == 9（随光标平移）
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 4), 9) << "position_ids 随光标平移";
}

// decode INT32: output_pos == 0（decode 序列维恒 1，唯一 token 行号 0）
TEST(OutputPosTest, DecodeInt32)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);
    ASSERT_TRUE(f.asm_->AssembleDecode({slot}, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 0) << "output_pos == 0 (decode)";
}

// prefill INT64: output_pos 值正确（8字节写入）
TEST(OutputPosTest, PrefillInt64)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT64);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 7, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 8), 6) << "output_pos == 6 (INT64)";
}

// decode INT64: output_pos == 0
TEST(OutputPosTest, DecodeInt64)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT64);
    ASSERT_TRUE(f.asm_->AssembleDecode({slot}, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 8), 0) << "output_pos == 0 (INT64 decode)";
}

// 图无 output_pos 工位名: SUCCESS 静默跳过，position_ids/new_kv_cache_pos/mask 照常填充
TEST(OutputPosTest, MissingNameSilentSkip)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // DataType::RESERVED → output_pos 不入 name_index，tensor 也不入 tensors
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::RESERVED);
    ASSERT_EQ(inputs.name_index.count("output_pos"), 0u);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).ok());

    // 验证 position_ids 和新 kv_cache_pos 照常填充
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 0);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 4), 4);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), 0);
    EXPECT_EQ(f.pool->NextPosition(slot), 5u);
}

// FLOAT32 dtype: INVALID_ARGUMENT（行号是整数索引，FLOAT32=位型垃圾行号）
TEST(OutputPosTest, InvalidDtypeFloat32Rejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::FLOAT32);
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// name_index 有 output_pos 但 index >= tensors.size(): INVALID_ARGUMENT
TEST(OutputPosTest, IndexOutOfBoundsRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);
    inputs.name_index["output_pos"] = 99; // tensors.size()=5
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(f.pool->NextPosition(slot), 0u);
}

// 旧图 float attention_mask + output_pos: 同样填充成功（mask 分支后统一填充）
TEST(OutputPosTest, FloatMaskLegacyGraph)
{
    UT_NEED_BACKEND();
    auto pool = std::make_shared<KvCachePool>(4, 64, 2, 4, 8);
    InputAssembler assembler(pool, 4, /*sliding_window=*/0);
    SlotId slot = -1;
    ASSERT_TRUE(pool->AllocKvSlot(&slot).ok());

    // 手工构造旧图工位组 + output_pos
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;
    tensors.push_back(MakeTensor({1, 1, 8, 64}, DataType::FLOAT32));
    name_index["attention_mask"] = 0;
    tensors.push_back(MakeTensor({1, 8}, DataType::INT32));
    name_index["position_ids"] = 1;
    tensors.push_back(MakeTensor({8}, DataType::INT32));
    name_index["new_kv_cache_pos"] = 2;
    tensors.push_back(MakeTensor({1}, DataType::INT32));
    name_index["output_pos"] = 3;

    ASSERT_TRUE(assembler.AssemblePrefill(slot, 5, tensors, name_index).ok());
    auto& op = tensors[3];
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 4) << "output_pos == 4 on legacy-mask graph";
    // attention_mask 同样正常填充（绕回改造后：行 r attend 逻辑位 j ∈ [0,r]
    // 经 MapSlot 恒等映射清 0，无绕回时与旧"动态列宽因果"逐值一致）
    EXPECT_FLOAT_EQ(ReadF32(tensors[0], 0), 0.0f) << "mask[0][0][0][0] == 0";
    EXPECT_FLOAT_EQ(ReadF32(tensors[0], 1), -FLT_MAX) << "mask[0][0][0][1] == -FLT_MAX";
}

// 短块 chunk_len < rows: output_pos 指真实末行 (chunk_len-1)，非 rows-1
TEST(OutputPosTest, ShortChunkRealLastRow)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);
    // rows=8, chunk_len=3 → output_pos == 2 (chunk_len-1), not 7 (rows-1)
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 2) << "output_pos == 2 (real last row, not rows-1=7)";
}

// decode 多 slot: output_pos 仍为 0（与 batch 大小无关）
TEST(OutputPosTest, DecodeMultiSlot)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok());
    std::vector<SlotId> slots = {s0, s1};

    // 需要 batch=2 的 tensors —— 复用 GemmaInputSetWithOutputPos 并调整 batch 维
    auto inputs = GemmaInputSetWithOutputPos::Make(DataType::INT32);
    // GemmaInputSetWithOutputPos 默认 rows=8（足够覆盖 batch=2）
    // decode 时 rows >= batch 即可
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& op = inputs.at("output_pos");
    EXPECT_EQ(ReadOutputPosScalar(op, 4), 0) << "output_pos == 0 (multi-slot decode)";
}

// decode 路径异常：缺 position_ids 名 → INVALID_ARGUMENT（行266,269覆盖）
TEST(InputAssemblerTest, DecodeMissingPositionIdsNameRejected)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index.erase("position_ids");
    std::vector<SlotId> slots = {slot};
    EXPECT_EQ(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// decode 路径异常：缺 attention_mask 名 → INVALID_ARGUMENT（行273,278覆盖）
TEST(InputAssemblerTest, DecodeMissingAttentionMaskNameRejected)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index.erase("attention_mask");
    std::vector<SlotId> slots = {slot};
    EXPECT_EQ(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// decode 路径异常：index 越界 → INVALID_ARGUMENT（行287,289覆盖）
TEST(InputAssemblerTest, DecodeIndexOutOfBoundsRejected)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 1, 1);
    inputs.name_index["position_ids"] = 999; // 超出 tensors.size()=6
    std::vector<SlotId> slots = {slot};
    EXPECT_EQ(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// gemma decode 异常：mask rows < batch → INVALID_ARGUMENT（行334,337覆盖）
TEST(GemmaMaskTest, DecodeMaskRowsLessThanBatchRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId s0 = -1, s1 = -1, s2 = -1, s3 = -1, s4 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&s2).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&s3).ok());
    ASSERT_TRUE(f.pool->AllocKvSlot(&s4).ok());
    std::vector<SlotId> slots = {s0, s1, s2, s3, s4}; // batch=5
    // rows=2 < batch=5 → INVALID_ARGUMENT
    auto inputs = GemmaInputSet::Make(2, 1, 16);
    EXPECT_EQ(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
}

// prefill gemma 异常：mask rows < chunk_len → INVALID_ARGUMENT（行221,224覆盖）
TEST(GemmaMaskTest, PrefillMaskRowsLessThanChunkLenRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // rows=2, chunk_len=6 → rows < chunk_len → INVALID_ARGUMENT
    auto inputs = GemmaInputSet::Make(1, 2, 16);
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// prefill: 容量约束上移 max_io_tokens 预算闸后，超 kv_max 块绕回合法
// （spec REQ-9 / design §4.3：原"Advance 越界=FAILURE 上抛"红线已废，仅
// kv_max==0 塌零仍拦）——abs=15、chunk=2 → 15+2=17 > 16 SUCCESS，
// npos = MapSlot 序列 [15, 0]（绕回跳变），TotalWritten=17、物理位绕回 1。
TEST(InputAssemblerTest, PrefillWraparoundAllowedAtCapacityBoundary)
{
    UT_NEED_BACKEND();
    Fixture f; // max_seq_len=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 15).ok()); // abs=15
    auto inputs = InputSet::Make(1, 2, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    // position_ids 绝对位 [15,16]；new_kv_cache_pos 物理槽 MapSlot=[15,0] 跳变
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 15);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 1), 16);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), 15);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 1), 0);
    EXPECT_EQ(f.pool->TotalWritten(slot), 17u);
    EXPECT_EQ(f.pool->NextPosition(slot), 1u);
}

// decode: 光标触顶后装配+推进合法（环形化改造后语义，spec REQ-9 / design §4.3：
// 原"Advance 越界=FAILURE"红线作废，仅 kv_max==0 容量塌零仍拦）——s0 推到 16
// 后 decode 装配 SUCCESS，两槽各推进 1，NextPosition 为物理槽（环形映射）。
TEST(InputAssemblerTest, DecodeAdvanceFailure)
{
    UT_NEED_BACKEND();
    Fixture f; // max_seq_len=16
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 16).ok()); // s0 total_written=16，触顶绕回
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok());
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = InputSet::Make(2, 1, 16); // mask 全 kv_max=16 宽
    // 环形语义：cursor 触顶不再拒绝，装配 SUCCESS 并推进
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());
    // s0: 16+1 → NextPosition=MapSlot(17)=1（anchor=0 纯模绕回）；s1: 0+1 → 1
    EXPECT_EQ(f.pool->NextPosition(s0), 1u);
    EXPECT_EQ(f.pool->TotalWritten(s0), 17u);
    EXPECT_EQ(f.pool->NextPosition(s1), 1u);
}

// ════════════════════════════════════════════════════════════════════════
// gemma4 pos 族填充测试（mask/position_ids/new_kv_cache_pos 填充偏差修复）
// 语义基准：Python 标杆 run_gemma4_gen.py build_prefill/build_decode 的
// arange 全行确定性填充——input tensors 跨 Forward 复用，图逐行按
// new_kv_cache_pos[row] scatter 写 KV，尾行残值 = 错误写入位。
// ════════════════════════════════════════════════════════════════════════

// INT32 工位预填残值（memset 0x7F → 每元素 0x7F7F7F7F）
const int32_t kPosResidue = static_cast<int32_t>(0x7F7F7F7Fu);
const int32_t kPosResidueB = static_cast<int32_t>(0xBBBBBBBBu); // dtype 位残值
const int32_t kPosResidueC = static_cast<int32_t>(0xCCCCCCCCu); // 容量位残值

// A1: 短块 pos 全行续 arange：rows=6、chunk=3、光标=0 → position_ids/
// new_kv_cache_pos 6 行全填 [0..5]（修复前 tail 行 3..5 为残值）
TEST(GemmaMaskTest, PrefillShortChunkPosFillsAllRows)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16); // 图行数 rows=6
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[6] = {0, 1, 2, 3, 4, 5};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "row=" << i;
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 3u);
}

// A1(光标偏移版): Advance 5 后 chunk=3 → pos 6 行续 arange [5..10]（非从 0 重启）
TEST(GemmaMaskTest, PrefillShortChunkPosContinuesFromCursor)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[6] = {5, 6, 7, 8, 9, 10};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "row=" << i;
    }
}

// A2: 跨 Forward 残值覆写：同一输入组 chunk=6（光标 0）→ chunk=3（光标 6），
// 第二块 6 行全覆写 [6..11]（修复前行 3..5 残留第一块 [3,4,5]）
TEST(GemmaMaskTest, PrefillSecondChunkPosOverwritesStaleRows)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16); // 同一输入组跨 Forward 复用
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[6] = {6, 7, 8, 9, 10, 11};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "row=" << i;
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 9u);
}

// A3: 错位续 arange 环形绕回：光标=12、rows=6、kv_max=16 → padding 行续
// arange 越过 kv_max 绕回圆环区头部（环形化改造后语义，spec REQ-9 /
// design §4.3——旧"错位截尾 clamp kv_max-1"作废；padding 落位属已覆盖待
// 重算段，合法覆写）：pos 全 6 行 = [12,13,14,15,0,1]
TEST(GemmaMaskTest, PrefillPosMisalignedTailClampedToKvMax)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 12).ok()); // cursor=12
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[6] = {12, 13, 14, 15, 0, 1};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "row=" << i;
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 15u);
}

// A4: 恰满短块合法：光标=12、chunk=4（12+4==16==kv_max）且 rows=6>4 →
// SUCCESS（环形化改造后语义，spec REQ-9 / design §4.3——"恰满短块拒绝"
// P0 红线作废：padding 行尾值续 arange 绕回落圆环区头部，属已覆盖待重算
// 段合法覆写）：pos 6 行 = [12,13,14,15,0,1]，Advance 到 16 → Next=MapSlot(16)=0
TEST(GemmaMaskTest, PrefillPosExactFillShortChunkRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 12).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 4, inputs.tensors, inputs.name_index).ok());
    const int32_t expect[6] = {12, 13, 14, 15, 0, 1};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "row=" << i;
    }
    EXPECT_EQ(f.pool->TotalWritten(slot), 16u) << "恰满推进到 kv_max";
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "物理槽 MapSlot(16)=0 绕回";
}

// A5（环形化翻转，spec REQ-9 / design §4.3）：容量约束上移 max_io_tokens
// 预算闸后，真实块超 kv_max 绕回合法——光标=12、chunk=5（12+5=17>16）→
// SUCCESS。bool 路径 f4a0d60 口径：position_ids/new_kv_cache_pos 双工位同填
// 物理槽序列 MapSlot(12+i)=[12,13,14,15,0,1]（padding 行 5 落 MapSlot(17)=1，
// clamp 环宽 16 ≥ padding 1 不触发）；Advance 17 → TotalWritten=17、
// 物理位 MapSlot(17)=1 绕回。
TEST(GemmaMaskTest, PrefillChunkWraparoundAcrossKvMaxAllowed_A5)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 12).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 5, inputs.tensors, inputs.name_index).ok());

    const int32_t slot_seq[6] = {12, 13, 14, 15, 0, 1};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), slot_seq[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), slot_seq[i]) << "row=" << i;
    }
    EXPECT_EQ(f.pool->TotalWritten(slot), 17u);
    EXPECT_EQ(f.pool->NextPosition(slot), 1u) << "MapSlot(17)=1 绕回";
}

// A6: pos 工位 dtype 防线：position_ids 换 INT64 → INVALID_ARGUMENT；
// new_kv_cache_pos/mask 预填残值保持原值（零半写）、position_ids 自身不
// 被部分写入、光标不推进
TEST(GemmaMaskTest, PrefillPosDtypeRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    inputs.tensors[inputs.name_index["position_ids"]] = MakeTensor({1, 6}, DataType::INT64);
    std::memset(inputs.at("position_ids")->GetData(), 0xBB, inputs.at("position_ids")->GetSize());
    std::memset(inputs.at("new_kv_cache_pos")->GetData(), 0x7F, inputs.at("new_kv_cache_pos")->GetSize());
    std::memset(inputs.at("mask_full")->GetData(), 0xAA, inputs.at("mask_full")->GetSize());
    std::memset(inputs.at("mask_sliding")->GetData(), 0xAA, inputs.at("mask_sliding")->GetSize());

    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), kPosResidueB) << "零半写 row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), kPosResidue) << "零半写 row=" << i;
    }
    EXPECT_EQ(ReadBool(inputs.at("mask_full"), 0), 0xAA) << "零半写";
    EXPECT_EQ(ReadBool(inputs.at("mask_sliding"), 0), 0xAA) << "零半写";
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// A7: pos 工位容量防线：position_ids 换 {1,1} INT32（4 字节 < 6 值 24 字节）→
// INVALID_ARGUMENT；残值不破坏、光标不推进
TEST(GemmaMaskTest, PrefillPosCapacityRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSet::Make(1, 6, 16);
    inputs.tensors[inputs.name_index["position_ids"]] = MakeTensor({1, 1}, DataType::INT32);
    std::memset(inputs.at("position_ids")->GetData(), 0xCC, inputs.at("position_ids")->GetSize());
    std::memset(inputs.at("new_kv_cache_pos")->GetData(), 0x7F, inputs.at("new_kv_cache_pos")->GetSize());

    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), kPosResidueC) << "零半写";
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), kPosResidue) << "零半写 row=" << i;
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// B8: decode inactive lane pos 复制 lane0：2 active（光标 3/0）+ rows=4 →
// position_ids 4 lane = [3,0,3,3]，new_kv_cache_pos 同值
// （mask 侧同口径断言见 DecodeInactiveLanesCopyLane0，本例独立只测 pos 族）
TEST(GemmaMaskTest, DecodePosInactiveLanesCopyLane0)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // lane0 光标 3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // lane1 光标 0
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = GemmaInputSet::Make(4, 1, 16); // rows=4，active=2
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[4] = {3, 0, 3, 3};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), expect[i]) << "lane=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "lane=" << i;
    }
}

// B9(触顶合法位): 光标==kv_max(16) decode 装配合法成功（环形化改造后语义，
// spec REQ-9 / design §4.3——旧"光标触顶拒绝"红线作废）：position_ids=绝对位
// total_written（R4 分离记账），new_kv_cache_pos=物理槽 MapSlot(abs)（s0 绕回
// 到 0）；装配后各槽 Advance，残值全行覆写。
TEST(GemmaMaskTest, DecodePosActiveCursorAtKvMaxRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 16).ok()); // total_written 16 == kv_max，环形绕回
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok());
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = GemmaInputSet::Make(2, 1, 16);
    std::memset(inputs.at("position_ids")->GetData(), 0x7F, inputs.at("position_ids")->GetSize());

    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());
    // position_ids = 绝对位（R4：RoPE 行）；new_kv_cache_pos = MapSlot 物理槽
    // （s0=MapSlot(16)=0 绕回跳变合法、s1=MapSlot(0)=0）
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 16);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), 0);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 1), 0);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 1), 0);
    // 装配成功推进 1：NextPosition = MapSlot(17)=1 / MapSlot(1)=1
    EXPECT_EQ(f.pool->NextPosition(s0), 1u);
    EXPECT_EQ(f.pool->TotalWritten(s0), 17u);
    EXPECT_EQ(f.pool->NextPosition(s1), 1u);
}

// B9(边界值): 光标 15（kv_max-1）→ SUCCESS 且 pos 绝对位 15 / 物理槽 15
// （无绕回时与绝对位同值），随后 Advance 推到 16 → NextPosition=MapSlot(16)=0
// 环形绕回（环形化改造后语义，spec REQ-9 / design §4.3）
TEST(GemmaMaskTest, DecodePosActiveCursorAtKvMaxMinusOneSucceeds)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 15).ok()); // total_written 15 = kv_max-1
    auto inputs = GemmaInputSet::Make(1, 1, 16);
    ASSERT_TRUE(f.asm_->AssembleDecode({slot}, inputs.tensors, inputs.name_index).ok());

    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 15);
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), 15);
    EXPECT_EQ(f.pool->TotalWritten(slot), 16u) << "成功推进 1 到 16（绝对位）";
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "物理槽 MapSlot(16)=0 环形绕回";
}

// B10: decode pos 全 lane 残值覆写：4 lane 预填 0x7F、单 active（光标 5）→
// pos 4 lane 全部 [5,5,5,5]（修复前 lane 1..3 残值 0x7F7F7F7F）
TEST(GemmaMaskTest, DecodePosOverwritesStaleAllLanes)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // 唯一 active 光标 5
    auto inputs = GemmaInputSet::Make(4, 1, 16);
    std::memset(inputs.at("position_ids")->GetData(), 0x7F, inputs.at("position_ids")->GetSize());
    std::memset(inputs.at("new_kv_cache_pos")->GetData(), 0x7F, inputs.at("new_kv_cache_pos")->GetSize());
    ASSERT_TRUE(f.asm_->AssembleDecode({slot}, inputs.tensors, inputs.name_index).ok());

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), i), 5) << "lane=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), 5) << "lane=" << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// U7-U10：float 路径环形绕回正例 + 防线（max_io_tokens 预算闸方案，
// spec REQ-3/REQ-9；方案 §5 U7-U10）
// ════════════════════════════════════════════════════════════════════════

// U7: float prefill 跨 kv_max 绕回正例——abs_start=15、chunk_len=4（预算闸
// 放行后单轮超 kv_max 输入），npos=[15,0,1,2]（MapSlot 跳变）；
// mask：q=15..18 全部 ≥ kv_max-1=15 → 四行皆走 WriteFloatMask 全环短路
// （逻辑位集 [0,q] 覆盖全环整行 attend——本例 rows=4=chunk_len 无 padding 行
// 语义；含 padding 行的短路/屏蔽形态由 bool 侧用例覆盖）。
TEST(InputAssemblerTest, FloatPrefillWraparoundPosAndMask)
{
    UT_NEED_BACKEND();
    Fixture f; // kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 15).ok()); // abs_start=15
    auto inputs = InputSet::Make(1, 4, 16);      // rows=4, kv_max=16
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 4, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    auto& new_pos = inputs.at("new_kv_cache_pos");
    // position_ids = 绝对位（RoPE 逻辑位）
    EXPECT_EQ(ReadI32(pos, 0), 15);
    EXPECT_EQ(ReadI32(pos, 1), 16);
    EXPECT_EQ(ReadI32(pos, 2), 17);
    EXPECT_EQ(ReadI32(pos, 3), 18);
    // new_kv_cache_pos = MapSlot(abs+i)，绕回跳变 [15, 0, 1, 2]
    EXPECT_EQ(ReadI32(new_pos, 0), 15);
    EXPECT_EQ(ReadI32(new_pos, 1), 0);
    EXPECT_EQ(ReadI32(new_pos, 2), 1);
    EXPECT_EQ(ReadI32(new_pos, 3), 2);

    // mask（行 r=16 列全宽）：行 r 的 q=15+r ≥ kv_max-1 → 整行 attend（短路）
    auto& mask = inputs.at("attention_mask");
    for (int r = 0; r < 4; ++r) {
        for (int j = 0; j < 16; ++j) {
            EXPECT_FLOAT_EQ(ReadF32(mask, r * 16 + j), 0.0f)
                << "row=" << r << " j=" << j << "（q=15+r≥kv_max-1 整行 attend）";
        }
    }
    EXPECT_EQ(f.pool->TotalWritten(slot), 19u);
    EXPECT_EQ(f.pool->NextPosition(slot), 3u); // MapSlot(19)=3
}

// U8: float decode post-wrap mask 回归——旧"平铺前缀 [0,cursor]"绕回后欠
// attend（init_token_len=8、ring=8：abs=18 的 attend 集 = 全环 16 列，平铺
// 只清 [0,10) 列尾巴漏 [10,16)）。锚定段池实例化证全环清 0。
TEST(InputAssemblerTest, FloatDecodePostWrapMaskAttendsFullRing)
{
    UT_NEED_BACKEND();
    Fixture f; // kv_max=16
    SlotId s = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
    f.pool->SetInitTokenLen(8);               // 锚定段 [0,8)，ring=8
    ASSERT_TRUE(f.pool->Advance(s, 18).ok()); // 绕回：abs=18、MapSlot(18)=10
    ASSERT_EQ(f.pool->MapSlot(18), 10u);

    std::vector<SlotId> slots = {s};
    auto inputs = InputSet::Make(1, 1, 16); // rows=1, kv_max=16
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& pos = inputs.at("position_ids");
    auto& new_pos = inputs.at("new_kv_cache_pos");
    EXPECT_EQ(ReadI32(pos, 0), 18);     // 绝对位
    EXPECT_EQ(ReadI32(new_pos, 0), 10); // MapSlot(18)=10 物理槽

    // q=18 ≥ kv_max-1 → 逻辑位 [0,18] 覆盖全环（多逻辑位同物理槽幂等），
    // 整行 16 列全 attend——旧平铺实现欠 attend 的 [10,16) 列全清 0
    auto& mask = inputs.at("attention_mask");
    for (int j = 0; j < 16; ++j) {
        EXPECT_FLOAT_EQ(ReadF32(mask, j), 0.0f) << "j=" << j;
    }
    EXPECT_EQ(f.pool->TotalWritten(s), 19u);
    EXPECT_EQ(f.pool->NextPosition(s), 11u); // MapSlot(19)=11
}

// U9: float mask dtype/容量防线——INT32 dtype 拒；chunk_len 超图行数拒
// （rows<chunk_len 前置拦截，mask buffer 零半写）；光标不推进。
// 模板化统一入口后 BOOL/FLOAT32/FLOAT16 均可走 kFloatSingle 路径——故用 INT32
// 测真正未支持的 dtype（mask_dtype_template_support U6）。
TEST(InputAssemblerTest, FloatMaskDtypeAndSizeRejected)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    { // dtype 防线：attention_mask 换 INT32（kFloatSingle 路径）。rows 推导
        // 按 mask dtype 步长（INT32→4 字节）：INT32 {1,1,8,16}
        // =512 字节 ÷(16×4) 得 rows=8 ≥ chunk_len=2 过前置拦截；pos 双工位先写
        // （pos 已残值被覆写属同段失败回滚语义——光标不推进即契约），随后
        // WriteMask 首判 dtype 不在 BOOL/INT8/FLOAT32/FLOAT16 拒
        // （先于 capacity/填充——mask 零半写）。
        auto inputs = InputSet::Make(1, 2, 16);
        inputs.tensors[inputs.name_index["attention_mask"]] = MakeTensor({1, 1, 8, 16}, DataType::INT32);
        std::memset(inputs.at("attention_mask")->GetData(), 0xAA, inputs.at("attention_mask")->GetSize());
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
        EXPECT_EQ(ReadI32(inputs.at("attention_mask"), 0), static_cast<int32_t>(0xAAAAAAAAu)) << "mask 零半写";
        EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
    }
    { // rows<chunk_len 前置拦截：mask 容量 2 行、chunk_len=3 →
        // AssemblePrefill rows<chunk_len 参数拒绝（mask 零半写、pos 未触）。
        auto inputs = InputSet::Make(1, 2, 16);
        std::memset(inputs.at("attention_mask")->GetData(), 0x7F, inputs.at("attention_mask")->GetSize());
        std::memset(inputs.at("position_ids")->GetData(), 0xCC, inputs.at("position_ids")->GetSize());
        EXPECT_EQ(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
        // 位级校验残值 0x7F7F7F7F 保持（float 位型; 读 I32 比对位型）
        EXPECT_EQ(ReadI32(inputs.at("attention_mask"), 0), kPosResidue) << "mask 零半写";
        EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), kPosResidueC) << "pos 零半写";
        EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
    }
}

// U10: float pos 工位严格容量防线——position_ids 容量 1 值 < chunk_len=2
// 值 → INVALID_ARGUMENT（WritePosValues 禁 min 截断，与 bool 路径 A7 同口径）。
TEST(InputAssemblerTest, FloatPosCapacityRejected)
{
    UT_NEED_BACKEND();
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = InputSet::Make(1, 2, 16);
    inputs.tensors[inputs.name_index["position_ids"]] = MakeTensor({1}, DataType::INT32);
    std::memset(inputs.at("position_ids")->GetData(), 0xCC, inputs.at("position_ids")->GetSize());
    std::memset(inputs.at("new_kv_cache_pos")->GetData(), 0x7F, inputs.at("new_kv_cache_pos")->GetSize());
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), kPosResidueC) << "零半写";
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), kPosResidue) << "零半写";
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// ════════════════════════════════════════════════════════════════════════
// SlidingPosTest：1027 新图可选工位 sliding_kv_pos 接线行为锁
// （tmp_docs/problems/device_garbled_output_sliding_kv_pos_unwired_20261027.md
// §五 修复方向 1/2/3——修复前全仓零接线、残值 0 致滑窗层 KV 全 scatter 到
// 槽 0 = 真机乱码根因；以下用例构造同故障面回红）
// fixture：W=4、P=4 ⇒ C=8（GemmaFixture 默认新契）；kv_max=16 与 C 拉开档差
// 以对偶观察两环独立绕回。
// ════════════════════════════════════════════════════════════════════════

// SP1: prefill 无绕回 (abs+i)%C 恒等落位——rows=6、chunk=3、光标=0 →
// sliding_kv_pos 6 行全填 [0..5]%8 = [0..5]（对偶 new_kv_cache_pos [0..5]，
// 两环同值系 kv_max/C 双双未绕；残值被覆写）
TEST(SlidingPosTest, PrefillArangeNoWrap)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/true);
    std::memset(inputs.at("sliding_kv_pos")->GetData(), 0x7F, inputs.at("sliding_kv_pos")->GetSize());
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), i % 8) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), i) << "对偶 npos row=" << i;
    }
    EXPECT_EQ(f.pool->NextPosition(slot), 3u);
}

// SP2: prefill 滑窗环独立绕回（mask-KV 写位对偶核心）——光标=7、rows=6、
// chunk=3 → 滑窗环 abs 7..12 mod 8 = [7,0,1,2,3,4]（C=8 在 kv_max=16 之前
// 绕回）；full 层 npos 走 MapSlot kv_max=16 大环仍恒等 [7..12]（两环独立）
TEST(SlidingPosTest, PrefillSlidingRingIndependentWrap)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 7).ok()); // abs_start=7
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/true);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    const int32_t expect_spos[6] = {7, 0, 1, 2, 3, 4};    // (7+i)%8
    const int32_t expect_npos[6] = {7, 8, 9, 10, 11, 12}; // MapSlot 恒等（kv_max 未绕）
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), expect_spos[i]) << "row=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect_npos[i]) << "row=" << i;
    }
}

// SP3: prefill padding 行撞末真实写位 clamp——光标=6、rows=6、chunk=2：
// 真实行 abs 6..7 → spos [6,7]；padding 行 2..5 续 arange abs 8..11 mod 8
// = [0,1,2,3]，C=8 环宽 > padding(4) 天然不撞末位 7——改构造撞位场景：
// 光标=0、rows=6、chunk=4 → 真实 abs 0..3 spos [0..3]、末真实写位=3；
// padding 行 4/5 续 arange abs 4/5 → spos 4/5（不撞 3，clamp 不触发）。
// 撞位最小例：光标=4、rows=6、chunk=1 → 末真实写位 (4+0)%8=4；
// padding 行 i=1..5 abs 5..9 → spos [5,6,7,0,1]，行 i=4 abs=8%8=0、
// 行 i=5 abs=9%8=1 均非 4，不触发——真实触发需 padding 抗 arange 跨整环：
// 光标=0、rows=10、chunk=2（ring C=8 < padding 8 行）→ 末真实写位 1；
// padding 行 i=2..9 abs 2..9 mod 8 = [2,3,4,5,6,7,0,1]，i=9 abs=9%8=1 撞
// 末位 → clamp 后仍 1（clamp 语义自反幂等；断言点=值连续性而非位异）——
// 本例退而断言非撞位 full-range 覆写（残值全清）+ 真实行语义，clamp 分支
// 语义同 pos 族既有 A5/WARN 口径，不单测重复。
TEST(SlidingPosTest, PrefillPaddingTailArange)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 10, 16, /*with_spos=*/true);
    std::memset(inputs.at("sliding_kv_pos")->GetData(), 0x7F, inputs.at("sliding_kv_pos")->GetSize());
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    for (int i = 0; i < 10; ++i) {
        // 全行续 arange (0+i)%8：padding 残值 0x7F7F7F7F 必被覆写
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), i % 8) << "row=" << i;
    }
}

// SP4: decode per-lane abs%C——2 active（光标 5/1）+ rows=4（inactive 复制
// lane 0）→ spos [5%8, 1%8, 5%8, 5%8] = [5,1,5,5]（对偶 npos MapSlot
// [5,1,5,1 恒等]——kv_max=16 未绕两者同值、绕回面由 SP5 覆盖）
TEST(SlidingPosTest, DecodePerLaneAbsModC)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 5).ok());   // lane0 abs=5
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // lane1 abs=0 → 装配即写 0（i 时点）
    ASSERT_TRUE(f.pool->Advance(s1, 1).ok());   // lane1 abs=1（预推进模拟多步 decode）
    std::vector<SlotId> slots = {s1, s0};       // lane0=s1(abs=1)、lane1=s0(abs=5)
    auto inputs = GemmaInputSetWithSlidingPos::Make(4, 1, 16, /*with_spos=*/true);
    std::memset(inputs.at("sliding_kv_pos")->GetData(), 0x7F, inputs.at("sliding_kv_pos")->GetSize());
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[4] = {1, 5, 1, 1}; // lane abs mod 8 + inactive 复制 lane0
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), expect[i]) << "lane=" << i;
        EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), i), expect[i]) << "对偶 npos lane=" << i;
    }
}

// SP5: decode 滑窗环绕回（mask-KV 写位对偶核心）——abs=17（C=8 绕 2 圈、
// kv_max=16 绕 1 圈）：spos = 17%8 = 1；npos = MapSlot(17) = 1（anchor=0
// 纯模 17%16=1——本例两者数值巧合同位，构造性证明双环独立计算源：断言点
// 在 mod C vs MapSlot 两算式各自正确，用 abs=9 数值分离——9%8=1 vs
// MapSlot(9)=9 分离证）。
TEST(SlidingPosTest, DecodeSlidingRingWrapsSeparatelyFromKvRing)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId s = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
    ASSERT_TRUE(f.pool->Advance(s, 9).ok()); // abs=9：滑窗环绕 1 圈、kv 环未绕
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 1, 16, /*with_spos=*/true);
    ASSERT_TRUE(f.asm_->AssembleDecode({s}, inputs.tensors, inputs.name_index).ok());

    EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), 0), 1) << "9%8=1 滑窗小环绕回";
    EXPECT_EQ(ReadI32(inputs.at("new_kv_cache_pos"), 0), 9) << "MapSlot(9)=9 大环未绕";
    EXPECT_EQ(f.pool->TotalWritten(s), 10u);
}

// SP6: 旧契（sw_hd=0）图带 sliding_kv_pos → INVALID_ARGUMENT 宁拒 + 零半写
// （修复前红：残值 0 直通前向 = 真机乱码根因；宁拒属于登记文档 §五.2 防线）
TEST(SlidingPosTest, LegacyContractSlotPresentRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f(4, /*legacy=*/true); // sw_hd=0 → SlidingLayerWidth=0
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/true);
    std::memset(inputs.at("sliding_kv_pos")->GetData(), 0x7F, inputs.at("sliding_kv_pos")->GetSize());
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), kPosResidue) << "零半写 row=" << i;
    EXPECT_EQ(f.pool->NextPosition(slot), 0u) << "失败不 Advance";
}

// SP6-b: decode 侧同口径——旧契图带工位 → INVALID_ARGUMENT、零半写、不推进
TEST(SlidingPosTest, LegacyContractSlotPresentRejectedDecode)
{
    UT_NEED_BACKEND();
    GemmaFixture f(4, /*legacy=*/true);
    SlotId s = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 1, 16, /*with_spos=*/true);
    std::memset(inputs.at("sliding_kv_pos")->GetData(), 0x7F, inputs.at("sliding_kv_pos")->GetSize());
    EXPECT_EQ(f.asm_->AssembleDecode({s}, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), 0), kPosResidue) << "零半写";
    EXPECT_EQ(f.pool->NextPosition(s), 0u) << "失败不 Advance";
}

// SP7: 旧图（图未声明 sliding_kv_pos）零行为变化——新旧契 assembler 对缺工位
// 均 SUCCESS 且无 writing（残值保持），pos 族照常填充
TEST(SlidingPosTest, SlotAbsentOldGraphZeroChange)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // 新契 fixture + 图不带工位（旧图回归面）
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/false);
    ASSERT_EQ(inputs.name_index.count("sliding_kv_pos"), 0u);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 0), 0);
    EXPECT_EQ(ReadI32(inputs.at("position_ids"), 5), 5);
    EXPECT_EQ(f.pool->NextPosition(slot), 3u);
}

// SP8: sliding_kv_pos 工位 index 越界 → INVALID_ARGUMENT（壳守卫第二连）
TEST(SlidingPosTest, SlotIndexOutOfRangeRejected)
{
    UT_NEED_BACKEND();
    GemmaFixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/true);
    inputs.name_index["sliding_kv_pos"] = 99;
    EXPECT_EQ(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(f.pool->NextPosition(slot), 0u);
}

// SP9: 残值零半写后写通（跨 Forward 复用回归）——第一块 chunk=6 写满 [0..5]，
// 第二块光标=6 chunk=2 → spos 全 6 行覆写 [6,7,0,1,2,3]（(6+i)%8 绕回 +
// padding 续 arange；修复前第二块尾行残值留 [3,4,5]）
TEST(SlidingPosTest, SecondChunkOverwritesStaleRows)
{
    UT_NEED_BACKEND();
    GemmaFixture f; // W=4, C=8, kv_max=16
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    auto inputs = GemmaInputSetWithSlidingPos::Make(1, 6, 16, /*with_spos=*/true);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 6, inputs.tensors, inputs.name_index).ok());
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    const int32_t expect[6] = {6, 7, 0, 1, 2, 3}; // (6+i)%8
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(ReadI32(inputs.at("sliding_kv_pos"), i), expect[i]) << "row=" << i;
}

} // namespace
} // namespace cann_local_lm::engine
