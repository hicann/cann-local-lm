// TEST-WRITE(RED)：输入组装器单元测试（v0.13：按 INPUT_INFO.md 设计填 position_ids/new_kv_cache_pos/attention_mask）。
// 关联 docs/model_executor/INPUT_INFO.md。
// RED 原因：InputAssembler 接口需增 name→index 映射；当前接口无此参数 + 不填充 pos/mask。
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/tensor_manager.h"

#include <gtest/gtest.h>
#include <cfloat>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cann_lm::engine {
namespace {

// ── 辅助：创建 ITensorBuffer + 读写 ──────────────────────────────────────

std::shared_ptr<ITensorBuffer> MakeTensor(std::size_t bytes)
{
    TensorDesc d;
    d.dims = {static_cast<int32_t>(bytes / 4)};
    d.dataType = DataType::FLOAT32;
    return hiai::CreateNDTensorBuffer(d);
}

std::shared_ptr<ITensorBuffer> MakeTensor(const std::vector<int32_t>& dims, DataType dt, std::size_t /*bytes*/)
{
    TensorDesc d;
    d.dims = dims;
    d.dataType = dt;
    return hiai::CreateNDTensorBuffer(d);
}

void WriteI32(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& values)
{
    if (!t || !t->GetData())
        return;
    std::size_t bytes = std::min(values.size() * sizeof(int32_t), t->GetSize());
    if (bytes)
        std::memcpy(t->GetData(), values.data(), bytes);
}

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

// ── 创建 6 个输入 tensor + name→index 映射 ────────────────────────────────

struct InputSet {
    std::vector<std::shared_ptr<ITensorBuffer>> tensors;
    std::unordered_map<std::string, size_t> name_index;

    // 按 INPUT_INFO.md 定义的 6 个输入
    static InputSet Make(int batch, int seq_len, int kv_cache_len)
    {
        InputSet s;
        // [0] input_embed: FLOAT16 [batch, seqLen, hidden=4] → batch*seqLen*4*2 bytes
        s.tensors.push_back(MakeTensor(batch * seq_len * 4 * 2));
        s.name_index["input_embed"] = 0;

        // [1] embed_scales: FLOAT32 [batch, seqLen, 1] → batch*seqLen*4 bytes
        s.tensors.push_back(MakeTensor(batch * seq_len * 4));
        s.name_index["embed_scales"] = 1;

        // [2] attention_mask: FLOAT32 [batch, 1, seqLen, kvCacheLen] → batch*1*seqLen*kvCacheLen*4
        s.tensors.push_back(MakeTensor(batch * 1 * seq_len * kv_cache_len * 4));
        s.name_index["attention_mask"] = 2;

        // [3] position_ids: INT32 [batch, seqLen] → batch*seqLen*4
        s.tensors.push_back(MakeTensor({batch, seq_len}, DataType::INT32, batch * seq_len * 4));
        s.name_index["position_ids"] = 3;

        // [4] new_kv_cache_pos: INT32 [batch, seqLen] → batch*seqLen*4
        s.tensors.push_back(MakeTensor({batch, seq_len}, DataType::INT32, batch * seq_len * 4));
        s.name_index["new_kv_cache_pos"] = 4;

        // [5] kvCache: placeholder
        s.tensors.push_back(MakeTensor(1024));
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

// ════════════════════════════════════════════════════════════════════════
// prefill 填充测试
// ════════════════════════════════════════════════════════════════════════

// prefill: position_ids = cursor_start..cursor_start+chunk_len-1
TEST(InputAssemblerTest, PrefillFillsPositionIds)
{
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // cursor=0, chunk_len=3
    auto inputs = InputSet::Make(1, 3, 3); // batch=1, seqLen=3, kvCacheLen=0+3=3
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
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    auto inputs = InputSet::Make(1, 2, 7);      // batch=1, seqLen=2, kvCacheLen=5+2=7
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
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    // cursor=0, chunk_len=3, kvCacheLen=3
    auto inputs = InputSet::Make(1, 3, 3);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 3, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // mask[0][0][0][j]: j<=0 → 0, j>0 → -FLT_MAX
    EXPECT_FLOAT_EQ(ReadF32(mask, 0), 0.0f);     // [0,0,0,0] j=0<=0 → 0
    EXPECT_FLOAT_EQ(ReadF32(mask, 1), -FLT_MAX); // [0,0,0,1] j=1>0
    EXPECT_FLOAT_EQ(ReadF32(mask, 2), -FLT_MAX); // [0,0,0,2] j=2>0
    // mask[0][0][1][j]: j<=1 → 0
    EXPECT_FLOAT_EQ(ReadF32(mask, 3), 0.0f);     // [0,0,1,0] j=0<=1 → 0
    EXPECT_FLOAT_EQ(ReadF32(mask, 4), 0.0f);     // [0,0,1,1] j=1<=1 → 0
    EXPECT_FLOAT_EQ(ReadF32(mask, 5), -FLT_MAX); // [0,0,1,2] j=2>1
    // mask[0][0][2][j]: j<=2 → 0
    EXPECT_FLOAT_EQ(ReadF32(mask, 6), 0.0f); // [0,0,2,0]
    EXPECT_FLOAT_EQ(ReadF32(mask, 7), 0.0f); // [0,0,2,1]
    EXPECT_FLOAT_EQ(ReadF32(mask, 8), 0.0f); // [0,0,2,2]
}

// ════════════════════════════════════════════════════════════════════════
// decode 填充测试
// ════════════════════════════════════════════════════════════════════════

// decode: position_ids = 各 slot 的 cursor
TEST(InputAssemblerTest, DecodeFillsPositionIds)
{
    Fixture f;
    std::vector<SlotId> slots;
    for (int i = 0; i < 2; ++i) {
        SlotId s = -1;
        ASSERT_TRUE(f.pool->AllocKvSlot(&s).ok());
        slots.push_back(s);
    }
    // cursor=0 for both slots
    auto inputs = InputSet::Make(2, 1, 1); // batch=2, seqLen=1, kvCacheLen=1
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
    Fixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // cursor=3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // cursor=0
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = InputSet::Make(2, 1, 4); // batch=2, seqLen=1, kvCacheLen=max(3+1,0+1)=4
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

// decode: attention_mask causal per slot（不同 cursor 的 slot 尾部遮蔽）
TEST(InputAssemblerTest, DecodeFillsAttentionMaskCausalPerSlot)
{
    Fixture f;
    SlotId s0 = -1, s1 = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&s0).ok());
    ASSERT_TRUE(f.pool->Advance(s0, 3).ok());   // cursor=3
    ASSERT_TRUE(f.pool->AllocKvSlot(&s1).ok()); // cursor=0
    std::vector<SlotId> slots = {s0, s1};
    auto inputs = InputSet::Make(2, 1, 4); // batch=2, seqLen=1, kvCacheLen=4
    ASSERT_TRUE(f.asm_->AssembleDecode(slots, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // mask[b][0][0][j]: batch=2, kvCacheLen=4
    // slot0 (cursor=3): j<=3 → 0, j>3 → -FLT_MAX
    EXPECT_FLOAT_EQ(ReadF32(mask, 0), 0.0f); // [0,0,0,0] j=0<=3
    EXPECT_FLOAT_EQ(ReadF32(mask, 1), 0.0f); // [0,0,0,1] j=1<=3
    EXPECT_FLOAT_EQ(ReadF32(mask, 2), 0.0f); // [0,0,0,2] j=2<=3
    EXPECT_FLOAT_EQ(ReadF32(mask, 3), 0.0f); // [0,0,0,3] j=3<=3
    // slot1 (cursor=0): j<=0 → 0, j>0 → -FLT_MAX
    EXPECT_FLOAT_EQ(ReadF32(mask, 4), 0.0f);     // [1,0,0,0] j=0<=0
    EXPECT_FLOAT_EQ(ReadF32(mask, 5), -FLT_MAX); // [1,0,0,1] j=1>0
    EXPECT_FLOAT_EQ(ReadF32(mask, 6), -FLT_MAX); // [1,0,0,2] j=2>0
    EXPECT_FLOAT_EQ(ReadF32(mask, 7), -FLT_MAX); // [1,0,0,3] j=3>0
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
    Fixture f;
    SlotId slot = -1;
    ASSERT_TRUE(f.pool->AllocKvSlot(&slot).ok());
    ASSERT_TRUE(f.pool->Advance(slot, 5).ok()); // cursor=5
    // chunk_len=2, kvCacheLen=5+2=7
    auto inputs = InputSet::Make(1, 2, 7);
    ASSERT_TRUE(f.asm_->AssemblePrefill(slot, 2, inputs.tensors, inputs.name_index).ok());

    auto& mask = inputs.at("attention_mask");
    // mask[0][0][0][j]: visible_upto=5 → j<=5 全 0, j=6 → -FLT_MAX
    for (int j = 0; j <= 5; ++j)
        EXPECT_FLOAT_EQ(ReadF32(mask, j), 0.0f) << "row0 j=" << j;
    EXPECT_FLOAT_EQ(ReadF32(mask, 6), -FLT_MAX);
    // mask[0][0][1][j]: visible_upto=6 → j<=6 全 0
    for (int j = 0; j <= 6; ++j)
        EXPECT_FLOAT_EQ(ReadF32(mask, 7 + j), 0.0f) << "row1 j=" << j;
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

} // namespace
} // namespace cann_lm::engine
