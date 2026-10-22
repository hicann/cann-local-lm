// InputAssembler 实现（v0.13：按 name→index 填 position_ids/new_kv_cache_pos/attention_mask；
// gemma4 双 bool mask 扩展见 docs/mask_fill.md / mask_fill_detail.md）。
// gemma4 可选工位 output_pos（logits 裁剪行号）填充见 docs/mask_fill.md §4——图带则填。
// 设计来源：docs/model_executor/INPUT_INFO.md、docs/mask_fill.md
#include "core/component/input_assembler/input_assembler.h"

#include "infra/log.h"
#include "utils/data_type_utils.h"   // DataTypeName（gemma4 mask/output_pos dtype 校验日志）

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>

namespace cann_lm::engine {

namespace {

/// 按 name 查 index；不存在返 -1
std::ptrdiff_t find_index(const std::unordered_map<std::string, std::size_t>& ni, const std::string& name) {
    auto it = ni.find(name);
    return it != ni.end() ? static_cast<std::ptrdiff_t>(it->second) : -1;
}

/// 写 int32 数组到 ITensorBuffer（min 截断）
void write_i32(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& values) {
    if (!t || !t->GetData()) return;
    std::size_t bytes = std::min(values.size() * sizeof(int32_t), t->GetSize());
    if (bytes) std::memcpy(t->GetData(), values.data(), bytes);
}

/// 写 float 数组到 ITensorBuffer（min 截断）
void write_f32(const std::shared_ptr<ITensorBuffer>& t, const std::vector<float>& values) {
    if (!t || !t->GetData()) return;
    std::size_t bytes = std::min(values.size() * sizeof(float), t->GetSize());
    if (bytes) std::memcpy(t->GetData(), values.data(), bytes);
}

/// gemma4 bool mask 写入（docs/mask_fill.md §5.1）：整 buffer 全量重写——
/// 先全置 1（True=屏蔽），再逐行清 attend 区间 [lo, q] 为 0（False=attend）。
/// q_abs[r] = 行 r 的 query 在 KV buffer 的绝对写入位。
/// window=0 → 全因果（mask_full）；window>0 且 < kv_max → 滑窗（mask_sliding，
/// attend [max(0,q-window+1), q]）；window >= kv_max 退化为全因果（与 full 同值）。
/// 校验（D2/D6）：dtype 限 BOOL/INT8（1 字节/元素，INT8 WARN）；GetSize 须严格
/// == rows×kv_max——bool mask 部分写 = stale 0x00 = 意外 attend 垃圾 KV，禁截断。
/// q 夹取 kv_max-1：光标触顶时 Advance 红线随后报错，但 mask 写入先于 Advance
/// 发生，不得越界写行尾。
Status WriteBoolMask(const std::shared_ptr<ITensorBuffer>& t,
                     const std::vector<int32_t>& q_abs,
                     std::size_t kv_max, int window) {
    if (!t || !t->GetData() || kv_max == 0) return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    if (dt != DataType::BOOL && dt != DataType::INT8) {
        CANN_LLM_LOGE("assembler.mask: gemma mask dtype must be BOOL/INT8, got %s",
                      DataTypeName(dt));
        return Status::INVALID_ARGUMENT;
    }
    if (dt == DataType::INT8) {
        CANN_LLM_LOGW("assembler.mask: gemma mask dtype INT8 (expect BOOL), write 0/1 as-is");
    }
    const std::size_t expect = q_abs.size() * kv_max;
    if (t->GetSize() != expect) {
        CANN_LLM_LOGE("assembler.mask: gemma mask size mismatch "
                      "(rows=%zu × kv_max=%zu = %zu bytes, got %zu)",
                      q_abs.size(), kv_max, expect, t->GetSize());
        return Status::INVALID_ARGUMENT;
    }
    auto* base = static_cast<uint8_t*>(t->GetData());
    std::memset(base, 1, expect);   // True = 屏蔽
    const bool banded = (window > 0) && (static_cast<std::size_t>(window) < kv_max);
    const int32_t q_cap = static_cast<int32_t>(kv_max - 1);
    for (std::size_t r = 0; r < q_abs.size(); ++r) {
        const int32_t q = std::min(q_abs[r], q_cap);
        const int32_t lo = banded ? std::max<int32_t>(0, q - window + 1) : 0;
        std::memset(base + r * kv_max + static_cast<std::size_t>(lo), 0,
                    static_cast<std::size_t>(q - lo + 1));   // False = attend
    }
    return Status::SUCCESS;
}

/// mask 工位模式（docs/mask_fill.md §5.4）：按三个工位名在场情况分派。
enum class MaskMode {
    kFloatSingle,   ///< 旧路径：单 attention_mask（float，qwen）
    kGemmaBool,     ///< gemma4：mask_full + mask_sliding（bool）
    kInvalid,       ///< 都缺 / 只有其一 / 三者并存 → INVALID_ARGUMENT
};

MaskMode DetectMaskMode(std::ptrdiff_t idx_attn, std::ptrdiff_t idx_full,
                        std::ptrdiff_t idx_sliding) {
    const bool has_attn    = idx_attn >= 0;
    const bool has_full    = idx_full >= 0;
    const bool has_sliding = idx_sliding >= 0;
    if (has_attn && !has_full && !has_sliding) return MaskMode::kFloatSingle;
    if (!has_attn && has_full && has_sliding) return MaskMode::kGemmaBool;
    return MaskMode::kInvalid;
}

/// output_pos 工位写入（gemma4 logits 裁剪行号；dtype 以 om 工位读出为准）：
/// INT32/INT64 单元素直写；其他 dtype → INVALID_ARGUMENT（行号是整数索引，
/// 妥协写入 = 位型垃圾行号，宁拒勿错）。容量按 min 语义（>= 单元素即可；
/// GetSize 是字节容量可含对齐填充，与 write_i32 口径一致）。
Status WriteOutputPos(const std::shared_ptr<ITensorBuffer>& t, std::int64_t value) {
    if (!t || !t->GetData()) return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    switch (dt) {
        case DataType::INT32: {
            if (t->GetSize() < sizeof(int32_t)) return Status::INVALID_ARGUMENT;
            const int32_t v = static_cast<int32_t>(value);
            std::memcpy(t->GetData(), &v, sizeof(v));
            return Status::SUCCESS;
        }
        case DataType::INT64: {
            if (t->GetSize() < sizeof(int64_t)) return Status::INVALID_ARGUMENT;
            std::memcpy(t->GetData(), &value, sizeof(value));
            return Status::SUCCESS;
        }
        default:
            CANN_LLM_LOGE("assembler.output_pos: dtype must be INT32/INT64, got %s",
                          DataTypeName(dt));
            return Status::INVALID_ARGUMENT;
    }
}

/// 可选工位 output_pos 填充（docs/mask_fill.md §4 工位家族）：图带则填行号，
/// 不带跳过（旧图 / output_pos 常量折叠图零行为变化）。在场但 index 越界 /
/// buffer 空 → 报错上抛：带着残值行号前向 = prefill 采到错误位置的 logits。
Status FillOutputPos(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                     const std::unordered_map<std::string, std::size_t>& name_index,
                     std::int64_t value) {
    const std::ptrdiff_t idx = find_index(name_index, "output_pos");
    if (idx < 0) return Status::SUCCESS;   // 图未声明该工位：无填充需求
    if (static_cast<std::size_t>(idx) >= tensors.size()) {
        CANN_LLM_LOGE("assembler.output_pos: slot index out of range "
                      "(index=%d, tensors=%zu)",
                      static_cast<int>(idx), tensors.size());
        return Status::INVALID_ARGUMENT;
    }
    return WriteOutputPos(tensors[idx], value);
}

}  // namespace

InputAssembler::InputAssembler(std::shared_ptr<KvCachePool> pool, int max_batch_size,
                               int sliding_window)
    : pool_(std::move(pool)), max_batch_size_(max_batch_size),
      sliding_window_(sliding_window > 0 ? sliding_window : 0) {}

Status InputAssembler::AssemblePrefill(SlotId slot, std::size_t chunk_len,
                                        std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                                        const std::unordered_map<std::string, std::size_t>& name_index) {
    if (!pool_) return Status::INVALID_STATE;
    if (slot < 0) return Status::INVALID_ARGUMENT;
    if (chunk_len == 0) return Status::INVALID_ARGUMENT;

    // 按 name 查找工位
    auto idx_pos     = find_index(name_index, "position_ids");
    auto idx_new_pos = find_index(name_index, "new_kv_cache_pos");
    auto idx_attn    = find_index(name_index, "attention_mask");
    auto idx_full    = find_index(name_index, "mask_full");
    auto idx_sliding = find_index(name_index, "mask_sliding");
    if (idx_pos < 0 || idx_new_pos < 0) {
        CANN_LLM_LOGE(
            "executor.assemble.prefill: missing input slot(s) (position_ids=%d, new_kv_cache_pos=%d)",
            static_cast<int>(idx_pos), static_cast<int>(idx_new_pos));
        return Status::INVALID_ARGUMENT;
    }
    MaskMode mode = DetectMaskMode(idx_attn, idx_full, idx_sliding);
    if (mode == MaskMode::kInvalid) {
        CANN_LLM_LOGE(
            "executor.assemble.prefill: invalid mask slots (attention_mask=%d, mask_full=%d, "
            "mask_sliding=%d; need exactly one of: attention_mask | mask_full+mask_sliding)",
            static_cast<int>(idx_attn), static_cast<int>(idx_full),
            static_cast<int>(idx_sliding));
        return Status::INVALID_ARGUMENT;
    }
    auto index_ok = [&](std::ptrdiff_t i) {
        return i >= 0 && static_cast<std::size_t>(i) < tensors.size();
    };
    const bool mask_index_ok = (mode == MaskMode::kFloatSingle)
                                   ? index_ok(idx_attn)
                                   : (index_ok(idx_full) && index_ok(idx_sliding));
    if (!index_ok(idx_pos) || !index_ok(idx_new_pos) || !mask_index_ok) {
        CANN_LLM_LOGE(
            "executor.assemble.prefill: slot index out of range (tensors=%zu)", tensors.size());
        return Status::INVALID_ARGUMENT;
    }

    // 从 KV 池获取光标（Advance 前）
    std::size_t cursor = pool_->NextPosition(slot);
    std::size_t kv_cache_len = cursor + chunk_len;

    // position_ids: [1, chunk_len] = cursor, cursor+1, ..., cursor+chunk_len-1
    std::vector<int32_t> pos_values(chunk_len);
    for (std::size_t i = 0; i < chunk_len; ++i)
        pos_values[i] = static_cast<int32_t>(cursor + i);
    write_i32(tensors[idx_pos], pos_values);

    // new_kv_cache_pos: [chunk_len] 1D = 相同值
    write_i32(tensors[idx_new_pos], pos_values);

    if (mode == MaskMode::kFloatSingle) {
        // attention_mask: [1, 1, chunk_len, kv_cache_len] causal（现状，qwen，零变化）
        // mask[i][j] = 0 if j <= cursor+i else -FLT_MAX
        std::vector<float> mask_values(chunk_len * kv_cache_len, -FLT_MAX);
        for (std::size_t i = 0; i < chunk_len; ++i) {
            std::size_t visible_upto = cursor + i;
            for (std::size_t j = 0; j <= visible_upto && j < kv_cache_len; ++j) {
                mask_values[i * kv_cache_len + j] = 0.0f;
            }
        }
        write_f32(tensors[idx_attn], mask_values);
    } else {
        // gemma4 双 bool mask（docs/mask_fill.md §5.2）：[1,1,rows,kv_max] 静态全 buffer。
        // rows = 工位容量/kv_max（=图行数，自洽不依赖 dims 修正，WriteBoolMask 复验
        // 严格相等）；padding 行 [chunk_len, rows) 复制最后有效行（D5：保证每行有
        // 非空 attend 区间，避免全屏蔽行进融合算子）。
        const std::size_t kv_max = static_cast<std::size_t>(pool_->kv_cache_max_len());
        if (kv_max == 0 || !tensors[idx_full]) return Status::INVALID_STATE;
        const std::size_t rows = tensors[idx_full]->GetSize() / kv_max;
        if (rows < chunk_len) {
            CANN_LLM_LOGE(
                "executor.assemble.prefill: gemma mask rows=%zu < chunk_len=%zu (kv_max=%zu)",
                rows, chunk_len, kv_max);
            return Status::INVALID_ARGUMENT;
        }
        std::vector<int32_t> q_abs(rows, static_cast<int32_t>(cursor + chunk_len - 1));
        for (std::size_t i = 0; i < chunk_len; ++i)
            q_abs[i] = static_cast<int32_t>(cursor + i);
        Status mst = WriteBoolMask(tensors[idx_full], q_abs, kv_max, /*window=*/0);
        if (!mst.ok()) return mst;
        mst = WriteBoolMask(tensors[idx_sliding], q_abs, kv_max, sliding_window_);
        if (!mst.ok()) return mst;
    }

    // output_pos（可选工位）：prefill 行号 = 本 chunk 末 token 行（chunk_len-1）。
    // 行号相对本 chunk 序列维（token 连续落位自行 0 起，FillInputs/mask 同基准），
    // 不随光标平移；短块时指向真实末行而非 stale padding 行。
    Status ost = FillOutputPos(tensors, name_index,
                               static_cast<std::int64_t>(chunk_len - 1));
    if (!ost.ok()) return ost;

    // 推进光标（KV 池管理绕回）
    Status adv_st = pool_->Advance(slot, chunk_len);
    if (!adv_st.ok()) {
        CANN_LLM_LOGE(
            "executor.assemble.prefill: kv advance failed (slot=%d, kv capacity exceeded?)",
            static_cast<int>(slot));
        return adv_st;
    }
    return Status::SUCCESS;
}

Status InputAssembler::AssembleDecode(const std::vector<SlotId>& slots,
                                      std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                                      const std::unordered_map<std::string, std::size_t>& name_index) {
    if (!pool_) return Status::INVALID_STATE;
    if (slots.empty()) return Status::INVALID_ARGUMENT;

    // 按 name 查找工位
    auto idx_pos     = find_index(name_index, "position_ids");
    auto idx_new_pos = find_index(name_index, "new_kv_cache_pos");
    auto idx_attn    = find_index(name_index, "attention_mask");
    auto idx_full    = find_index(name_index, "mask_full");
    auto idx_sliding = find_index(name_index, "mask_sliding");
    if (idx_pos < 0 || idx_new_pos < 0) {
        CANN_LLM_LOGE(
            "executor.assemble.decode: missing input slot(s) (position_ids=%d, new_kv_cache_pos=%d)",
            static_cast<int>(idx_pos), static_cast<int>(idx_new_pos));
        return Status::INVALID_ARGUMENT;
    }
    MaskMode mode = DetectMaskMode(idx_attn, idx_full, idx_sliding);
    if (mode == MaskMode::kInvalid) {
        CANN_LLM_LOGE(
            "executor.assemble.decode: invalid mask slots (attention_mask=%d, mask_full=%d, "
            "mask_sliding=%d; need exactly one of: attention_mask | mask_full+mask_sliding)",
            static_cast<int>(idx_attn), static_cast<int>(idx_full),
            static_cast<int>(idx_sliding));
        return Status::INVALID_ARGUMENT;
    }
    auto index_ok = [&](std::ptrdiff_t i) {
        return i >= 0 && static_cast<std::size_t>(i) < tensors.size();
    };
    const bool mask_index_ok = (mode == MaskMode::kFloatSingle)
                                   ? index_ok(idx_attn)
                                   : (index_ok(idx_full) && index_ok(idx_sliding));
    if (!index_ok(idx_pos) || !index_ok(idx_new_pos) || !mask_index_ok) {
        CANN_LLM_LOGE(
            "executor.assemble.decode: slot index out of range (tensors=%zu)", tensors.size());
        return Status::INVALID_ARGUMENT;
    }

    int batch = static_cast<int>(slots.size());

    // 各 slot 光标（Advance 前）
    std::vector<std::size_t> cursors(slots.size());
    std::size_t max_kv_len = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (slots[i] < 0) return Status::INVALID_ARGUMENT;
        cursors[i] = pool_->NextPosition(slots[i]);
        max_kv_len = std::max(max_kv_len, cursors[i] + 1);
    }

    // position_ids: [batch, 1] = 各 slot cursor
    std::vector<int32_t> pos_values(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        pos_values[i] = static_cast<int32_t>(cursors[i]);
    write_i32(tensors[idx_pos], pos_values);

    // new_kv_cache_pos: [batch, seqLen=1] per-slot = 各 slot cursor（per-slot，非共享）
    std::vector<int32_t> new_pos_values(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        new_pos_values[i] = static_cast<int32_t>(cursors[i]);
    write_i32(tensors[idx_new_pos], new_pos_values);

    if (mode == MaskMode::kFloatSingle) {
        // attention_mask: [batch, 1, 1, max_kv_len] causal per slot（现状，qwen，零变化）
        // mask[b][j] = 0 if j <= cursors[b] else -FLT_MAX
        std::vector<float> mask_values(batch * max_kv_len, -FLT_MAX);
        for (std::size_t i = 0; i < slots.size(); ++i) {
            for (std::size_t j = 0; j <= cursors[i] && j < max_kv_len; ++j) {
                mask_values[i * max_kv_len + j] = 0.0f;
            }
        }
        write_f32(tensors[idx_attn], mask_values);
    } else {
        // gemma4 双 bool mask（docs/mask_fill.md §5.3）：[batch,1,rows,kv_max] 静态全 buffer。
        // rows = 工位容量/kv_max（=图行数=decode_batch_size，自洽不依赖 dims 修正，
        // WriteBoolMask 复验严格相等）；inactive lane [batch, rows) 复制 lane 0
        //（D5：padded batch 槽位缺席不产生全屏蔽行进融合算子）。
        const std::size_t kv_max = static_cast<std::size_t>(pool_->kv_cache_max_len());
        if (kv_max == 0 || !tensors[idx_full]) return Status::INVALID_STATE;
        const std::size_t rows = tensors[idx_full]->GetSize() / kv_max;
        if (rows < slots.size()) {
            CANN_LLM_LOGE(
                "executor.assemble.decode: gemma mask rows=%zu < batch=%zu (kv_max=%zu)",
                rows, slots.size(), kv_max);
            return Status::INVALID_ARGUMENT;
        }
        std::vector<int32_t> q_abs(rows, static_cast<int32_t>(cursors[0]));
        for (std::size_t i = 0; i < slots.size(); ++i)
            q_abs[i] = static_cast<int32_t>(cursors[i]);
        Status mst = WriteBoolMask(tensors[idx_full], q_abs, kv_max, /*window=*/0);
        if (!mst.ok()) return mst;
        mst = WriteBoolMask(tensors[idx_sliding], q_abs, kv_max, sliding_window_);
        if (!mst.ok()) return mst;
    }

    // output_pos（可选工位）：decode 序列维恒 1，唯一 token 行号 0
    Status ost = FillOutputPos(tensors, name_index, 0);
    if (!ost.ok()) return ost;

    // 各 active 槽推进 1
    for (SlotId s : slots) {
        Status st = pool_->Advance(s, 1);
        if (!st.ok()) {
            CANN_LLM_LOGE(
                "executor.assemble.decode: kv advance failed (slot=%d, kv capacity exceeded?)",
                static_cast<int>(s));
            return st;
        }
    }
    return Status::SUCCESS;
}

}  // namespace cann_lm::engine
