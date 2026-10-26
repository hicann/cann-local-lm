// InputAssembler 实现（v0.13：按 name→index 填 position_ids/new_kv_cache_pos/attention_mask；
// gemma4 双 bool mask 扩展见 docs/mask_fill.md / mask_fill_detail.md）。
// gemma4 可选工位 output_pos（logits 裁剪行号）填充见 docs/mask_fill.md §4——图带则填。
// 设计来源：docs/model_executor/INPUT_INFO.md、docs/mask_fill.md
#include "core/component/input_assembler/input_assembler.h"

#include "infra/log.h"
#include "utils/data_type_utils.h" // DataTypeName（gemma4 mask/output_pos dtype 校验日志）

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>

namespace cann_lm::engine {

namespace {

/// 按 name 查 index；不存在返 -1
std::ptrdiff_t find_index(const std::unordered_map<std::string, std::size_t>& ni, const std::string& name)
{
    auto it = ni.find(name);
    return it != ni.end() ? static_cast<std::ptrdiff_t>(it->second) : -1;
}

/// 写 int32 数组到 ITensorBuffer（min 截断）
void write_i32(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& values)
{
    if (!t || !t->GetData())
        return;
    std::size_t bytes = std::min(values.size() * sizeof(int32_t), t->GetSize());
    if (bytes)
        std::memcpy(t->GetData(), values.data(), bytes);
}

/// 写 float 数组到 ITensorBuffer（min 截断）
void write_f32(const std::shared_ptr<ITensorBuffer>& t, const std::vector<float>& values)
{
    if (!t || !t->GetData())
        return;
    std::size_t bytes = std::min(values.size() * sizeof(float), t->GetSize());
    if (bytes)
        std::memcpy(t->GetData(), values.data(), bytes);
}

/// pos 族工位写入（position_ids/new_kv_cache_pos；gemma 图 58b5d1f 约定 INT32）：
/// dtype 防线 + 显式容量校验，禁止 min 截断——om 图逐行按 new_kv_cache_pos[row]
/// scatter 写 KV（input tensors 跨 Forward 复用），截断 = 尾行保残值被当 KV 写入位，
/// 带病运行必覆写真实 KV / 越界写。宁拒勿错（与 WriteBoolMask 同拒绝口径）。
Status WritePosValues(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& values)
{
    if (!t || !t->GetData())
        return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    if (dt != DataType::INT32) {
        CANN_LLM_LOGE("assembler.pos: pos slot dtype must be INT32, got %s", DataTypeName(dt));
        return Status::INVALID_ARGUMENT;
    }
    const std::size_t need = values.size() * sizeof(int32_t);
    if (t->GetSize() < need) {
        CANN_LLM_LOGE("assembler.pos: pos slot capacity %zu bytes < %zu values "
                      "(need %zu bytes)",
                      t->GetSize(), values.size(), need);
        return Status::INVALID_ARGUMENT;
    }
    std::memcpy(t->GetData(), values.data(), need);
    return Status::SUCCESS;
}

/// gemma4 bool mask 写入（golden 对齐语义；PromptCache 环形列映射 design §4.3）：
/// 整 buffer 全量重写——先全置 1（True=屏蔽），再仅对**有效行**（q_abs 所列行）
/// 清 attend 区间为 0（False=attend）；padding 行 [q_abs.size(), rows) 无条目、
/// 保持初始全 1（完全屏蔽。原 D5「复制最后有效行」已删除——golden dump 实测
/// padding 行全 1；有效行对所有 padding 写入槽位全遮蔽、padding 行输出无人
/// gather，全 1 不影响有效结果）。
/// q_abs[r] = 有效行 r 的 query 绝对写入位；attend 窗口 [lo, q] 是**逻辑位**
/// 区间（window=0 → lo=0 全因果；0<window<kv_max → lo=q-window+1 滑窗；
/// window ≥ kv_max 退化全因果），逐逻辑位 j 经 map_slot(j) 映射物理槽清 0
/// ——环形绕回下列轴=物理槽、窗口逻辑位集散射清零（多圈时同一物理槽被
/// 多逻辑位清零，幂等无害；锚定段 [0, init_token_len) 在窗口内自然清出）。
/// map_slot 为空（池缺）⇒ 恒等映射（防御，正常接线恒非空）。
/// 校验（D2/D6）：dtype 限 BOOL/INT8（1 字节/元素，INT8 WARN）；GetSize 须严格
/// == rows×kv_max（rows=图工位全宽行数，与有效行数解耦）——bool mask 部分写
/// = stale 0x00 = 意外 attend 垃圾 KV，禁截断。
Status WriteBoolMask(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& q_abs, std::size_t rows,
                     std::size_t kv_max, int window, const KvCachePool* pool)
{
    if (!t || !t->GetData() || kv_max == 0)
        return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    if (dt != DataType::BOOL && dt != DataType::INT8) {
        CANN_LLM_LOGE("assembler.mask: gemma mask dtype must be BOOL/INT8, got %s", DataTypeName(dt));
        return Status::INVALID_ARGUMENT;
    }
    if (dt == DataType::INT8) {
        CANN_LLM_LOGW("assembler.mask: gemma mask dtype INT8 (expect BOOL), write 0/1 as-is");
    }
    const std::size_t expect = rows * kv_max;
    if (t->GetSize() != expect) {
        CANN_LLM_LOGE("assembler.mask: gemma mask size mismatch "
                      "(rows=%zu × kv_max=%zu = %zu bytes, got %zu)",
                      rows, kv_max, expect, t->GetSize());
        return Status::INVALID_ARGUMENT;
    }
    if (q_abs.size() > rows) {
        CANN_LLM_LOGE("assembler.mask: valid rows %zu > slot rows %zu", q_abs.size(), rows);
        return Status::INVALID_ARGUMENT;
    }
    auto* base = static_cast<uint8_t*>(t->GetData());
    std::memset(base, 1, expect); // True = 屏蔽；padding 行由此保持全 1
    const bool banded = (window > 0) && (static_cast<std::size_t>(window) < kv_max);
    for (std::size_t r = 0; r < q_abs.size(); ++r) {
        const int32_t q = q_abs[r];
        const int32_t lo = banded ? std::max<int32_t>(0, q - window + 1) : 0;
        for (int32_t j = lo; j <= q; ++j) {
            const std::size_t slot = pool ? pool->MapSlot(static_cast<std::size_t>(j)) : j;
            base[r * kv_max + slot] = 0; // False = attend（逻辑位→物理槽）
        }
    }
    return Status::SUCCESS;
}

/// mask 工位模式（docs/mask_fill.md §5.4）：按三个工位名在场情况分派。
enum class MaskMode {
    kFloatSingle, ///< 旧路径：单 attention_mask（float，qwen）
    kGemmaBool,   ///< gemma4：mask_full + mask_sliding（bool）
    kInvalid,     ///< 都缺 / 只有其一 / 三者并存 → INVALID_ARGUMENT
};

MaskMode DetectMaskMode(std::ptrdiff_t idx_attn, std::ptrdiff_t idx_full, std::ptrdiff_t idx_sliding)
{
    const bool has_attn = idx_attn >= 0;
    const bool has_full = idx_full >= 0;
    const bool has_sliding = idx_sliding >= 0;
    if (has_attn && !has_full && !has_sliding)
        return MaskMode::kFloatSingle;
    if (!has_attn && has_full && has_sliding)
        return MaskMode::kGemmaBool;
    return MaskMode::kInvalid;
}

/// output_pos 工位写入（gemma4 logits 裁剪行号；dtype 以 om 工位读出为准）：
/// INT32/INT64 单元素直写；其他 dtype → INVALID_ARGUMENT（行号是整数索引，
/// 妥协写入 = 位型垃圾行号，宁拒勿错）。容量按 min 语义（>= 单元素即可；
/// GetSize 是字节容量可含对齐填充，与 write_i32 口径一致）。
Status WriteOutputPos(const std::shared_ptr<ITensorBuffer>& t, std::int64_t value)
{
    if (!t || !t->GetData())
        return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    switch (dt) {
        case DataType::INT32: {
            if (t->GetSize() < sizeof(int32_t))
                return Status::INVALID_ARGUMENT;
            const int32_t v = static_cast<int32_t>(value);
            std::memcpy(t->GetData(), &v, sizeof(v));
            return Status::SUCCESS;
        }
        case DataType::INT64: {
            if (t->GetSize() < sizeof(int64_t))
                return Status::INVALID_ARGUMENT;
            std::memcpy(t->GetData(), &value, sizeof(value));
            return Status::SUCCESS;
        }
        default:
            CANN_LLM_LOGE("assembler.output_pos: dtype must be INT32/INT64, got %s", DataTypeName(dt));
            return Status::INVALID_ARGUMENT;
    }
}

/// 可选工位 output_pos 填充（docs/mask_fill.md §4 工位家族）：图带则填行号，
/// 不带跳过（旧图 / output_pos 常量折叠图零行为变化）。在场但 index 越界 /
/// buffer 空 → 报错上抛：带着残值行号前向 = prefill 采到错误位置的 logits。
Status FillOutputPos(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                     const std::unordered_map<std::string, std::size_t>& name_index, std::int64_t value)
{
    const std::ptrdiff_t idx = find_index(name_index, "output_pos");
    if (idx < 0)
        return Status::SUCCESS; // 图未声明该工位：无填充需求
    if (static_cast<std::size_t>(idx) >= tensors.size()) {
        CANN_LLM_LOGE("assembler.output_pos: slot index out of range "
                      "(index=%d, tensors=%zu)",
                      static_cast<int>(idx), tensors.size());
        return Status::INVALID_ARGUMENT;
    }
    return WriteOutputPos(tensors[idx], value);
}

} // namespace

InputAssembler::InputAssembler(std::shared_ptr<KvCachePool> pool, int max_batch_size, int sliding_window)
    : pool_(std::move(pool)), max_batch_size_(max_batch_size), sliding_window_(sliding_window > 0 ? sliding_window : 0)
{}

Status InputAssembler::AssemblePrefill(SlotId slot, std::size_t chunk_len,
                                       std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                                       const std::unordered_map<std::string, std::size_t>& name_index)
{
    if (!pool_)
        return Status::INVALID_STATE;
    if (slot < 0)
        return Status::INVALID_ARGUMENT;
    if (chunk_len == 0)
        return Status::INVALID_ARGUMENT;

    // 按 name 查找工位
    auto idx_pos = find_index(name_index, "position_ids");
    auto idx_new_pos = find_index(name_index, "new_kv_cache_pos");
    auto idx_attn = find_index(name_index, "attention_mask");
    auto idx_full = find_index(name_index, "mask_full");
    auto idx_sliding = find_index(name_index, "mask_sliding");
    if (idx_pos < 0 || idx_new_pos < 0) {
        CANN_LLM_LOGE("executor.assemble.prefill: missing input slot(s) (position_ids=%d, new_kv_cache_pos=%d)",
                      static_cast<int>(idx_pos), static_cast<int>(idx_new_pos));
        return Status::INVALID_ARGUMENT;
    }
    MaskMode mode = DetectMaskMode(idx_attn, idx_full, idx_sliding);
    if (mode == MaskMode::kInvalid) {
        CANN_LLM_LOGE("executor.assemble.prefill: invalid mask slots (attention_mask=%d, mask_full=%d, "
                      "mask_sliding=%d; need exactly one of: attention_mask | mask_full+mask_sliding)",
                      static_cast<int>(idx_attn), static_cast<int>(idx_full), static_cast<int>(idx_sliding));
        return Status::INVALID_ARGUMENT;
    }
    auto index_ok = [&](std::ptrdiff_t i) { return i >= 0 && static_cast<std::size_t>(i) < tensors.size(); };
    const bool mask_index_ok = (mode == MaskMode::kFloatSingle) ? index_ok(idx_attn) :
                                                                  (index_ok(idx_full) && index_ok(idx_sliding));
    if (!index_ok(idx_pos) || !index_ok(idx_new_pos) || !mask_index_ok) {
        CANN_LLM_LOGE("executor.assemble.prefill: slot index out of range (tensors=%zu)", tensors.size());
        return Status::INVALID_ARGUMENT;
    }

    // 从 KV 池获取写入记账（Advance 前；R4 分离记账——PromptCache design §4.3）：
    //   abs_start = total_written：本块首 token 绝对位（position_ids 语义源）。
    //   new_kv_cache_pos 语义源 = abs→槽逐位映射（MapSlot(abs+i)，下方按模式
    //   分别填）；无绕回时与旧 cursor 同值（现状兼容）。
    const std::size_t abs_start = pool_->TotalWritten(slot);
    const std::size_t kv_max = static_cast<std::size_t>(pool_->kv_cache_max_len());

    // 真实块容量红线（spec REQ-9 单次写入不自身绕回的前置等价）：装配 + 图
    // 前向先于 Advance 发生，真实 token 写入位绝对值越 kv_max ⇒ 单轮输入
    // 超缓冲（绕回语义不支持单轮输入自身绕回），宁早败。Submit 前置校验
    // （ids.size() ≤ kv_max）正常时本红线不触发（防御深度）。
    if (kv_max == 0 || abs_start + chunk_len > kv_max) {
        CANN_LLM_LOGE("executor.assemble.prefill: chunk overflow kv capacity "
                      "(total_written=%zu, chunk_len=%zu, kv_max=%zu)",
                      abs_start, chunk_len, kv_max);
        return Status::FAILURE;
    }

    // 图行数（om 图 seq 维静态值）：bool 路径按 mask 工位容量推导（与下方 mask
    // 同口径，WriteBoolMask 复验严格相等）；float 路径（qwen 旧图）保持现状
    // 按 chunk_len 填充（零变化——旧图无全行 scatter 消费约定，不作扩面）。
    std::size_t rows = chunk_len;
    if (mode == MaskMode::kGemmaBool) {
        if (!tensors[idx_full])
            return Status::INVALID_STATE;
        rows = tensors[idx_full]->GetSize() / kv_max;
        if (rows < chunk_len) {
            CANN_LLM_LOGE("executor.assemble.prefill: gemma mask rows=%zu < chunk_len=%zu (kv_max=%zu)", rows,
                          chunk_len, kv_max);
            return Status::INVALID_ARGUMENT;
        }
    }

    if (mode == MaskMode::kGemmaBool) {
        // 恰满短块红线（环形语义解除，design §4.3 posfix 衔接）：原 P0 红线防
        // "padding 行尾值 clamp kv_max-1 砸真实 KV 末位"；环形语义下本块真实末
        // 写位为绝对位 abs_start+chunk_len-1 ≤ kv_max-1（上方红线把关），padding
        // 行尾值续 arange 越过 kv_max 即绕回落圆环区头部——该位置属"已覆盖
        // 待重算"段（REQ-4 绕回判定保证），是合法覆写，不再拒绝。
        //
        // position_ids / new_kv_cache_pos 全行确定性填充（P0，Python 标杆
        // run_gemma4_gen.py build_prefill：arange(start_pos, start_pos+seq) 一行
        // 不遗——input_tensors_ 跨 Forward 复用，图逐行按 new_kv_cache_pos[row]
        // scatter 写 KV，尾行残值 = 上一 chunk/请求的写入位 → 覆写本请求前面的
        // 真实 KV / 越界写）。
        //   - 真实行 [0, chunk_len)：position_ids = 绝对位 abs_start+i（R4）；
        //     new_kv_cache_pos = MapSlot(abs_start+i)（物理槽，环形跳变合法值，
        //     f4a0d60 全行确定性重写防线下无残值）。
        //   - padding 行 [chunk_len, rows)：续 arange 落圆环区未来未写位，
        //     后续真实写入前被 mask 不可达、写入时即被覆盖（同 Python 语义）。
        // 错位配置（kv_max 非 rows 整倍数，abs_start+rows 绝对值越 kv_max）：
        // 尾值 clamp 到 MapSlot(abs_start+chunk_len-1)（本块末真实写位；多行
        // 同位互写无实害）+ WARN。
        std::vector<int32_t> pos_values(rows);
        const std::size_t last_real_slot = pool_->MapSlot(abs_start + chunk_len - 1);
        bool clamped = false;
        for (std::size_t i = 0; i < rows; ++i) {
            std::size_t p = pool_->MapSlot(abs_start + i); // 尾行续 arange（环形）
            if (i >= chunk_len && p == last_real_slot) {
                // 绕回小环且 ring 宽 < rows-chunk_len：padding 覆满一圈后首槽
                // 重新对上末真实写位 → clamp 防砸（ring ≥ rows-chunk_len 的常
                // 规配置天然不触发）。
                p = last_real_slot;
                clamped = true;
            }
            pos_values[i] = static_cast<int32_t>(p);
        }
        if (clamped) {
            CANN_LLM_LOGW("executor.assemble.prefill: ring width < padding rows (kv_max=%zu, "
                          "rows=%zu, chunk_len=%zu) — padding tail clamped to last real slot "
                          "%zu",
                          kv_max, rows, chunk_len, last_real_slot);
        }
        Status pst = WritePosValues(tensors[idx_pos], pos_values);
        if (!pst.ok())
            return pst;
        pst = WritePosValues(tensors[idx_new_pos], pos_values);
        if (!pst.ok())
            return pst;
    } else {
        // float（qwen 旧图）现状零变化：position_ids: [1, chunk_len] =
        // 绝对位 abs..abs+chunk_len-1；new_kv_cache_pos 1D = 相同值（无绕回时
        // 与物理槽同值；上方红线已保证全景不越 kv_max，直线填充成立）。
        std::vector<int32_t> pos_values(chunk_len);
        for (std::size_t i = 0; i < chunk_len; ++i)
            pos_values[i] = static_cast<int32_t>(abs_start + i);
        write_i32(tensors[idx_pos], pos_values);
        write_i32(tensors[idx_new_pos], pos_values);
    }

    if (mode == MaskMode::kFloatSingle) {
        // attention_mask: [1, 1, chunk_len, kv_cache_len] causal（现状，qwen，零变化）
        // mask[i][j] = 0 if j <= abs_start+i else -FLT_MAX
        const std::size_t kv_cache_len = abs_start + chunk_len;
        std::vector<float> mask_values(chunk_len * kv_cache_len, -FLT_MAX);
        for (std::size_t i = 0; i < chunk_len; ++i) {
            std::size_t visible_upto = abs_start + i;
            for (std::size_t j = 0; j <= visible_upto && j < kv_cache_len; ++j) {
                mask_values[i * kv_cache_len + j] = 0.0f;
            }
        }
        write_f32(tensors[idx_attn], mask_values);
    } else {
        // gemma4 双 bool mask（golden 标杆 run_gemma4_gen.py / 新 golden_meta 数据
        // 复核）：[1,1,rows,kv_max] 静态全 buffer，列轴 = 物理槽。q_abs 填**绝对位**
        // （WriteBoolMask 内部按 attend 窗口 [max(0, q-window+1), q] 的逻辑位逐位
        // MapSlot 映射到物理槽集——design §4.3 mask 逻辑列→物理槽映射；无绕回时
        // 与旧值逐位一致，行为兼容）。rows 已按 mask 工位容量推导；mask 先全置 1、
        // 只清真实行 attend 区间——padding 行 [chunk_len, rows) 保持初始全 1
        // （原 D5「复制最后有效行」已删除：golden dump 实测 padding 行全 1，
        // 复制语义是本引擎独有的非必要偏差；有效行对所有 padding 写入槽位全遮
        // 蔽 + padding 行输出无人 gather，全 1 不影响有效结果）。
        std::vector<int32_t> q_abs(chunk_len);
        for (std::size_t i = 0; i < chunk_len; ++i)
            q_abs[i] = static_cast<int32_t>(abs_start + i);
        // P2 观测：图带 mask_sliding 工位但 meta 无滑窗宽（sliding_window=0）→
        // 滑窗层静默退化为全因果（值与 Python 标杆 w<=0 退化分支同，但属配置
        // 缺失信号）；WARN 一次（池生命周期内跨 prefill/decode 图去重）。
        if (sliding_window_ == 0 && !warned_window_) {
            warned_window_ = true;
            CANN_LLM_LOGW("assembler.mask: graph has mask_sliding slot but sliding_window=0 "
                          "(meta_json missing?) — sliding layers degrade to full causal");
        }
        Status mst = WriteBoolMask(tensors[idx_full], q_abs, rows, kv_max, /*window=*/0, pool_.get());
        if (!mst.ok())
            return mst;
        mst = WriteBoolMask(tensors[idx_sliding], q_abs, rows, kv_max, sliding_window_, pool_.get());
        if (!mst.ok())
            return mst;
    }

    // output_pos（可选工位）：prefill 行号 = 本 chunk 末 token 行（chunk_len-1）。
    // 行号相对本 chunk 序列维（token 连续落位自行 0 起，FillInputs/mask 同基准），
    // 不随光标平移；短块时指向真实末行而非 stale padding 行。
    Status ost = FillOutputPos(tensors, name_index, static_cast<std::int64_t>(chunk_len - 1));
    if (!ost.ok())
        return ost;

    // 推进光标（KV 池管理绕回）
    Status adv_st = pool_->Advance(slot, chunk_len);
    if (!adv_st.ok()) {
        CANN_LLM_LOGE("executor.assemble.prefill: kv advance failed (slot=%d, kv capacity exceeded?)",
                      static_cast<int>(slot));
        return adv_st;
    }
    return Status::SUCCESS;
}

Status InputAssembler::AssembleDecode(const std::vector<SlotId>& slots,
                                      std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                                      const std::unordered_map<std::string, std::size_t>& name_index)
{
    if (!pool_)
        return Status::INVALID_STATE;
    if (slots.empty())
        return Status::INVALID_ARGUMENT;

    // 按 name 查找工位
    auto idx_pos = find_index(name_index, "position_ids");
    auto idx_new_pos = find_index(name_index, "new_kv_cache_pos");
    auto idx_attn = find_index(name_index, "attention_mask");
    auto idx_full = find_index(name_index, "mask_full");
    auto idx_sliding = find_index(name_index, "mask_sliding");
    if (idx_pos < 0 || idx_new_pos < 0) {
        CANN_LLM_LOGE("executor.assemble.decode: missing input slot(s) (position_ids=%d, new_kv_cache_pos=%d)",
                      static_cast<int>(idx_pos), static_cast<int>(idx_new_pos));
        return Status::INVALID_ARGUMENT;
    }
    MaskMode mode = DetectMaskMode(idx_attn, idx_full, idx_sliding);
    if (mode == MaskMode::kInvalid) {
        CANN_LLM_LOGE("executor.assemble.decode: invalid mask slots (attention_mask=%d, mask_full=%d, "
                      "mask_sliding=%d; need exactly one of: attention_mask | mask_full+mask_sliding)",
                      static_cast<int>(idx_attn), static_cast<int>(idx_full), static_cast<int>(idx_sliding));
        return Status::INVALID_ARGUMENT;
    }
    auto index_ok = [&](std::ptrdiff_t i) { return i >= 0 && static_cast<std::size_t>(i) < tensors.size(); };
    const bool mask_index_ok = (mode == MaskMode::kFloatSingle) ? index_ok(idx_attn) :
                                                                  (index_ok(idx_full) && index_ok(idx_sliding));
    if (!index_ok(idx_pos) || !index_ok(idx_new_pos) || !mask_index_ok) {
        CANN_LLM_LOGE("executor.assemble.decode: slot index out of range (tensors=%zu)", tensors.size());
        return Status::INVALID_ARGUMENT;
    }

    int batch = static_cast<int>(slots.size());
    const std::size_t kv_max = static_cast<std::size_t>(pool_->kv_cache_max_len());

    // 各 slot 写入记账（Advance 前；R4 分离记账）：abs[i] = total_written
    // （绝对位，position_ids 语义）；cursors[i] = write_pos = MapSlot(abs[i])
    // （物理槽，new_kv_cache_pos 语义）。无绕回时两值相等（现状兼容）。
    std::vector<std::size_t> abs_pos(slots.size());
    std::vector<std::size_t> cursors(slots.size());
    std::size_t max_kv_len = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (slots[i] < 0)
            return Status::INVALID_ARGUMENT;
        abs_pos[i] = pool_->TotalWritten(slots[i]);
        cursors[i] = pool_->NextPosition(slots[i]);
        max_kv_len = std::max(max_kv_len, cursors[i] + 1);
    }

    // kv_max = 0 防御（环形化后光标触顶不再拒绝——R5 绕回即语义；仅容量
    // 塌零这一构造异常态仍拦，与 prefill 侧同口径）。
    if (kv_max == 0) {
        CANN_LLM_LOGE("executor.assemble.decode: kv_max=0 (pool malformed)");
        return Status::FAILURE;
    }

    std::size_t rows = slots.size(); // float 路径 = active 槽数（现状零变化）
    if (mode == MaskMode::kGemmaBool) {
        if (!tensors[idx_full])
            return Status::INVALID_STATE;
        rows = tensors[idx_full]->GetSize() / kv_max;
        if (rows < slots.size()) {
            CANN_LLM_LOGE("executor.assemble.decode: gemma mask rows=%zu < batch=%zu (kv_max=%zu)", rows, slots.size(),
                          kv_max);
            return Status::INVALID_ARGUMENT;
        }
        // position_ids / new_kv_cache_pos 全 lane 确定性填充（P1，Python 标杆
        // run_gemma4_gen.py build_decode 全 batch 维确定性输入）：R4 分离记账下
        // position_ids = 绝对位 abs[i]（单调逻辑长，RoPE 行）；
        // new_kv_cache_pos = 物理槽 cursors[i] = MapSlot(abs[i])（逐 lane
        // scatter 写位，环形跳变合法）。inactive lane [slots.size(), rows) 复制
        // lane 0（pos 族确定性填充，防残值 lane 被图 scatter 消费成随机写位；
        // mask 侧 padding lane 保持全 1，两者口径不同——pos 是写入位须合法，
        // mask 全屏蔽不参与有效计算）。input_tensors_ 跨 Forward 复用 +
        // 图逐 lane 按 new_kv_cache_pos 消费 scatter 写位 → 残值 lane = 随机
        // KV 写入位。inactive lane 写入落自身 KV lane 的未来区，复由槽复用
        // 清零/光标覆写兜底，无实害。
        std::vector<int32_t> pos_values(rows);
        std::vector<int32_t> npos_values(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            const std::size_t idx = (i < slots.size()) ? i : 0;
            pos_values[i] = static_cast<int32_t>(abs_pos[idx]);
            npos_values[i] = static_cast<int32_t>(cursors[idx]);
        }
        Status pst = WritePosValues(tensors[idx_pos], pos_values);
        if (!pst.ok())
            return pst;
        pst = WritePosValues(tensors[idx_new_pos], npos_values);
        if (!pst.ok())
            return pst;
    } else {
        // float（qwen 旧图）现状零变化（无绕回时绝对位==物理槽；prefill 侧
        // 红线保证单请求全景 ≤ kv_max，直线值成立）：position_ids: [batch, 1]
        // = 各 slot 绝对位；new_kv_cache_pos: [batch, 1] per-slot = 各 slot
        // 物理槽（非共享）。
        std::vector<int32_t> pos_values(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
            pos_values[i] = static_cast<int32_t>(abs_pos[i]);
        write_i32(tensors[idx_pos], pos_values);
        write_i32(tensors[idx_new_pos], pos_values);
    }

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
        // gemma4 双 bool mask（golden 对齐语义）：[batch,1,rows,kv_max] 静态全
        // buffer，列轴=物理槽。q_abs 填**绝对位**（环形窗口逻辑位集→物理槽映射
        // 在 WriteBoolMask 内，design §4.3）。rows 已按 mask 工位容量推导
        // （=图行数=decode_batch_size，自洽不依赖 dims 修正，WriteBoolMask 复验
        // 严格相等）；mask 先全置 1、只清 active lane 的 attend 区间——
        // inactive lane [slots.size(), rows) 保持初始全 1（完全屏蔽，原 D5
        // 「复制 lane 0」已删除，与 prefill padding 行同口径）。
        std::vector<int32_t> q_abs(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
            q_abs[i] = static_cast<int32_t>(abs_pos[i]);
        if (sliding_window_ == 0 && !warned_window_) {
            warned_window_ = true;
            CANN_LLM_LOGW("assembler.mask: graph has mask_sliding slot but sliding_window=0 "
                          "(meta_json missing?) — sliding layers degrade to full causal");
        }
        Status mst = WriteBoolMask(tensors[idx_full], q_abs, rows, kv_max, /*window=*/0, pool_.get());
        if (!mst.ok())
            return mst;
        mst = WriteBoolMask(tensors[idx_sliding], q_abs, rows, kv_max, sliding_window_, pool_.get());
        if (!mst.ok())
            return mst;
    }

    // output_pos（可选工位）：decode 序列维恒 1，唯一 token 行号 0
    Status ost = FillOutputPos(tensors, name_index, 0);
    if (!ost.ok())
        return ost;

    // 各 active 槽推进 1
    for (SlotId s : slots) {
        Status st = pool_->Advance(s, 1);
        if (!st.ok()) {
            CANN_LLM_LOGE("executor.assemble.decode: kv advance failed (slot=%d, kv capacity exceeded?)",
                          static_cast<int>(s));
            return st;
        }
    }
    return Status::SUCCESS;
}

} // namespace cann_lm::engine
