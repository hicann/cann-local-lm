// InputAssembler 实现（v0.13：按 name→index 填 position_ids/new_kv_cache_pos/attention_mask；
// gemma4 双 bool mask 扩展见 docs/mask_fill.md / mask_fill_detail.md）。
// gemma4 可选工位 output_pos（logits 裁剪行号）填充见 docs/mask_fill.md §4——图带则填。
// 设计来源：docs/model_executor/INPUT_INFO.md、docs/mask_fill.md
#include "core/component/input_assembler/input_assembler.h"

#include "core/component/model_executor/model_executor.h" // SlidingLayerWidth/PrefillMaxLen（新契 C=W+P 单源）
#include "infra/log.h"
#include "utils/data_type_utils.h" // DataTypeName（gemma4 mask/output_pos dtype 校验日志）

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <limits> // std::numeric_limits（MaskTraits<fp16_t> 屏蔽哨兵 -inf）
#include <optional>
#include <utility>

namespace cann_local_lm::engine {

namespace {

/// 按 name 查 index；不存在返 -1
std::ptrdiff_t find_index(const std::unordered_map<std::string, std::size_t>& name_index, const std::string& name)
{
    auto it = name_index.find(name);
    return it != name_index.end() ? static_cast<std::ptrdiff_t>(it->second) : -1;
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

/// bool mask 写入映射环（sliding_kv_seq_dim_rebind D5）：
/// kAnchored = kv_max 锚定环——物理槽经 pool->MapSlot(j)（锚定式绕回），
/// mask_full / 旧契 mask_sliding 消费；kSliding = 滑窗小环——物理槽 =
/// 逻辑位纯模 j % width（新契 mask_sliding 消费）。kSliding 纯模不锚定：
/// 任一 width 连续逻辑位 mod width 互不重复（新契 width=C=W+P>W，窗口 W 列
/// mod C 恰 W 列、不满环；锚定/收窄才会在绕回窗口内产生双逻辑位撞槽）。
enum class MaskRing { kAnchored, kSliding };

/// mask 哨兵值 traits（模板化统一入口 S1，mask_dtype_template_support 方案）：
/// 每种 dtype 的 attend/block 常量 + 元素步长。新增 dtype 只加一个特化即可
/// 扩展 mask 支持（R2）。哨兵值是 dtype 强相关的常量，traits 特化是 C++ 惯用法。
template <typename T>
struct MaskTraits; // 主模板不定义——仅特化可用，未特化 dtype 编译期即拒

template <>
struct MaskTraits<uint8_t> { // BOOL / INT8 共用 1 字节位模式
    static uint8_t Attend() { return 0; }
    static uint8_t Block() { return 1; }
    static constexpr std::size_t kElem = 1;
};

template <>
struct MaskTraits<float> { // FLOAT32
    static float Attend() { return 0.0f; }
    static float Block() { return -FLT_MAX; }
    static constexpr std::size_t kElem = sizeof(float);
};

template <>
struct MaskTraits<fp16_t> { // FLOAT16（data_type_utils.h 软件 half 类型）
    // R4：屏蔽哨兵用 -inf 位模式（0xFC00）而非 -65504.0h——softmax exp(-inf)=0
    // 是数学精确屏蔽；-65504 经 fp16→fp32 累加仍有非零残值，长序列下 attend
    // 集污染风险。
    static fp16_t Attend() { return fp16_t(0.0f); }
    static fp16_t Block() { return fp16_t(-std::numeric_limits<float>::infinity()); }
    static constexpr std::size_t kElem = sizeof(fp16_t); // = 2
};

/// mask 写入模板（模板化统一入口 S2，承载原 WriteBoolMask/WriteFloatMask
/// 同构流程）：整 buffer 填屏蔽哨兵 → 逐有效行按 attend 窗口 [lo, q] 逐逻辑位
/// 经 ring 映射落物理槽写 attend 哨兵。padding 行 [q_abs.size(), rows) 保持
/// 全屏蔽。window=0 全因果（lo=0）；window>0 且 ≤width 滑窗（lo=q-window+1）；
/// window > width 退化全因果（banded=false）。ring=kAnchored 走 pool->MapSlot
/// （kv_max 锚定环），kSliding 走 j%width 纯模（滑窗小环）。非滑窗且
/// q>=width-1 时逻辑位集覆盖全环，整行全 attend 短路。
/// 校验：GetSize 须严格 == rows × width × kElem——部分写 = stale 哨兵混合 =
/// 错误 attend 集，禁截断（与旧两函数同拒绝口径）。
template <typename T>
Status WriteMaskTyped(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& q_abs, std::size_t rows,
                      std::size_t width, int window, MaskRing ring, const KvCachePool* pool)
{
    if (!t || !t->GetData() || width == 0)
        return Status::INVALID_STATE;
    constexpr std::size_t kElem = MaskTraits<T>::kElem;
    const std::size_t expect = rows * width * kElem;
    if (t->GetSize() != expect) {
        CANN_LLM_LOGE("assembler.mask: size mismatch "
                      "(rows=%zu × width=%zu × %zu bytes = %zu, got %zu)",
                      rows, width, kElem, expect, t->GetSize());
        return Status::INVALID_ARGUMENT;
    }
    if (q_abs.size() > rows) {
        CANN_LLM_LOGE("assembler.mask: valid rows %zu > slot rows %zu", q_abs.size(), rows);
        return Status::INVALID_ARGUMENT;
    }
    // 防御复验：分带窗口须不超列宽（新契 width=C=W+P 恒 > window=W、旧契
    // width=kv_max 承载 window=W——sliding_kv_seq_dim_rebind D5）。window >
    // width 照抄旧退化口径（banded=false 全因果），不静默错窗。
    const bool banded = (window > 0) && (static_cast<std::size_t>(window) <= width);
    auto* base = static_cast<T*>(t->GetData());
    const T block_val = MaskTraits<T>::Block();
    const T attend_val = MaskTraits<T>::Attend();
    std::fill(base, base + rows * width, block_val); // 屏蔽；padding 行由此保持
    for (std::size_t r = 0; r < q_abs.size(); ++r) {
        const int32_t q = q_abs[r];
        if (q < 0)
            continue; // 防御：负绝对位（不可达）不 attend
        T* row = base + r * width;
        if (!banded && static_cast<std::size_t>(q) >= width - 1) {
            // 逻辑位集覆盖全环（绕回满圈 / 全因果长序列）：整行全 attend，短路
            std::fill(row, row + width, attend_val);
            continue;
        }
        const int32_t lo = banded ? std::max<int32_t>(0, q - window + 1) : 0;
        for (int32_t j = lo; j <= q; ++j) {
            const std::size_t slot = (ring == MaskRing::kSliding) ? static_cast<std::size_t>(j) % width :
                                                                    (pool ? pool->MapSlot(static_cast<std::size_t>(j)) :
                                                                            static_cast<std::size_t>(j));
            row[slot] = attend_val; // attend（逻辑位→物理槽）
        }
    }
    return Status::SUCCESS;
}

/// mask 写入统一入口（模板化统一入口 S3）：按工位 dtype 分派到对应
/// WriteMaskTyped 实例化。新增 dtype 只在此 switch 加一个 case（R1）。
/// BOOL/INT8 共用 uint8_t 实例化（INT8 WARN 同旧路径）；FLOAT16 走 fp16_t
/// 软件类型（data_type_utils.h）；未列出 dtype → INVALID_ARGUMENT。
Status WriteMask(const std::shared_ptr<ITensorBuffer>& t, const std::vector<int32_t>& q_abs, std::size_t rows,
                 std::size_t width, int window, MaskRing ring, const KvCachePool* pool)
{
    if (!t || !t->GetData() || width == 0)
        return Status::INVALID_STATE;
    const DataType dt = t->GetTensorDesc().dataType;
    switch (dt) {
        case DataType::BOOL:
        case DataType::INT8:
            if (dt == DataType::INT8) {
                CANN_LLM_LOGW("assembler.mask: dtype INT8 (expect BOOL), write 0/1 as-is");
            }
            return WriteMaskTyped<uint8_t>(t, q_abs, rows, width, window, ring, pool);
        case DataType::FLOAT32:
            return WriteMaskTyped<float>(t, q_abs, rows, width, window, ring, pool);
        case DataType::FLOAT16:
            return WriteMaskTyped<fp16_t>(t, q_abs, rows, width, window, ring, pool);
        default:
            CANN_LLM_LOGE("assembler.mask: unsupported dtype %s "
                          "(supported: BOOL/INT8/FLOAT32/FLOAT16)",
                          DataTypeName(dt));
            return Status::INVALID_ARGUMENT;
    }
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

/// mask 工位择位（bool 双 mask 取 mask_full / float 单 mask 取 attention_mask；
/// AssemblePrefill/AssembleDecode rows 推导共用口径——与 WriteBoolMask/
/// WriteFloatMask 复验严格相等）。
std::ptrdiff_t PickMaskSlot(MaskMode mode, std::ptrdiff_t idx_full, std::ptrdiff_t idx_attn)
{
    return mode == MaskMode::kGemmaBool ? idx_full : idx_attn;
}

/// mask 元素步长（字节）：按工位 dtype 单源推导（R5，替代旧按 MaskMode 间接
/// 推导）。GetSize 是字节数——先除元素步长再除 kv_max，否则 float 行数虚大
/// 4 倍（UT host 首跑实证）。fp16 = 2 字节同理。
std::size_t ElemSize(MaskMode /*mode*/, DataType dt) { return DataTypeElemSize(dt); }

/// 可选工位 sliding_kv_pos（新图滑窗层专用 scatter 写位）填充值序列构造：
/// 与 new_kv_cache_pos 成对（同 per-token 写位形态），仅环不同——full 层走
/// pool->MapSlot（kv_max 锚定环），滑窗层走纯模 j % C（C = W+P，kSliding 同源
/// 口径，WriteBoolMask kSliding 分支/mask_sliding 列轴逐值对偶）。
/// 每行 i 填 (abs_start + i) % C；count = 图行数（prefill padding 行续 arange
/// 落小环未来未写位，与 new_kv_cache_pos padding 口径同构）；padding 行与
/// 本块末真实写位撞槽时 clamp 到末真实写位（ring 宽 < padding 行数时防砸，
/// 与 new_kv_cache_pos 的 clamp 分支同构——两环独立各算各的末位）。
/// C=0（旧契滑窗环不可算——sw_hd=0 或 W/P 非正防御态）→ nullopt：构值层不
/// 拒——"图带工位但配置无新契"的 INVALID_ARGUMENT 宁拒守卫在 WriteSlidingPos
/// 壳单源承担。return：values[i] = 行 i 的滑窗层 KV 写位（< C）。
std::optional<std::vector<int32_t>> SlidingPosValues(std::size_t abs_start, std::size_t count, std::size_t chunk_len,
                                                     std::size_t C)
{
    if (C == 0)
        return std::nullopt;
    std::vector<int32_t> v(count);
    const std::size_t last_real = (abs_start + chunk_len - 1) % C;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t p = (abs_start + i) % C;
        if (i >= chunk_len && p == last_real && chunk_len > 0) {
            // padding 行绕满小环后首槽重新对上末真实写位 → clamp（C ≥ 1 恒
            // 可达；与 anchored 环 clamp 同构，多行同位互写无实害）
            p = last_real;
        }
        v[i] = static_cast<int32_t>(p);
    }
    return v;
}

/// 可选工位 sliding_kv_pos 落写壳（prefill/decode 单源共享）：守卫三连——
/// 图未声明（idx<0）→ SUCCESS 跳过（旧图零行为变化）；index 越界 →
/// INVALID_ARGUMENT；C==0（图带工位但新契配置缺失：sw_head_dim=0 /
/// sliding_window<=0 / prefill_len 空 → SlidingLayerWidth=0）→
/// INVALID_ARGUMENT——工位在场但残值未写 = 图把整块滑窗层 KV 全 scatter 到
/// 槽 0（真机乱码根因，tmp_docs/problems/…sliding_kv_pos_unwired），宁拒。
/// 通过后 WritePosValues 落写（dtype INT32/容量防线内置）。
Status WriteSlidingPos(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                       const std::unordered_map<std::string, std::size_t>& name_index,
                       const std::vector<int32_t>& values, std::size_t C)
{
    const std::ptrdiff_t idx = find_index(name_index, "sliding_kv_pos");
    if (idx < 0)
        return Status::SUCCESS; // 图未声明该工位：无填充需求
    if (static_cast<std::size_t>(idx) >= tensors.size()) {
        CANN_LLM_LOGE("assembler.sliding_pos: slot index out of range "
                      "(index=%d, tensors=%zu)",
                      static_cast<int>(idx), tensors.size());
        return Status::INVALID_ARGUMENT;
    }
    if (C == 0) {
        CANN_LLM_LOGE("assembler.sliding_pos: slot present but sliding ring width "
                      "unavailable (sliding_window_head_dim unset / sliding_window<=0 "
                      "/ prefill_len empty) — refusing to forward residue write "
                      "positions");
        return Status::INVALID_ARGUMENT;
    }
    return WritePosValues(tensors[idx], values);
}

} // namespace

InputAssembler::InputAssembler(std::shared_ptr<KvCachePool> pool, int max_batch_size, int sliding_window,
                               int prefill_max, int sliding_window_head_dim)
    : pool_(std::move(pool)),
      max_batch_size_(max_batch_size),
      sliding_window_(sliding_window > 0 ? sliding_window : 0),
      prefill_max_(prefill_max > 0 ? prefill_max : 0),
      sw_head_dim_(sliding_window_head_dim > 0 ? sliding_window_head_dim : 0)
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

    // 容量塌零防御（TC_602 契约）：池 kv_max=0 构造异常态仍拦，与 decode 侧
    // 同口径。容量硬约束整体上移至 GenerateAsync 提交期 max_io_tokens 预算闸
    // （spec REQ-9）——bool/float 两 mask 路径均放行绕回（CorrectTensorDescs
    // 把 attention_mask 修形为全 kv_max 宽静态工位，图容量即圆环大小，与参考
    // 实现 FillMask 同构逐列映射支持绕回）。
    if (kv_max == 0) {
        CANN_LLM_LOGE("executor.assemble.prefill: kv_max=0 pool, chunk overflow "
                      "(total_written=%zu, chunk_len=%zu)",
                      abs_start, chunk_len);
        return Status::FAILURE;
    }

    // 图行数（om 图 seq 维静态值）：按 mask 工位容量推导（bool/float 择工位，
    // 与下方 mask 同口径，WriteBoolMask/WriteFloatMask 复验严格相等）——
    // float 与 bool 同承环形语义，均全 buffer 确定性重写。
    // 择位/元素步长口径提入 PickMaskSlot()/ElemSize()（prefill/decode 共用）。
    if (!tensors[PickMaskSlot(mode, idx_full, idx_attn)]) {
        return Status::INVALID_STATE;
    }
    const DataType mask_dt = tensors[PickMaskSlot(mode, idx_full, idx_attn)]->GetTensorDesc().dataType;
    const std::size_t rows = tensors[PickMaskSlot(mode, idx_full, idx_attn)]->GetSize() /
                             (kv_max * ElemSize(mode, mask_dt));
    if (rows < chunk_len) {
        CANN_LLM_LOGE("executor.assemble.prefill: mask rows=%zu < chunk_len=%zu (kv_max=%zu)", rows, chunk_len, kv_max);
        return Status::INVALID_ARGUMENT;
    }

    if (mode == MaskMode::kGemmaBool) {
        // position_ids / new_kv_cache_pos 全行确定性填充（P0，Python 标杆
        // run_gemma4_gen.py build_prefill：arange(start_pos, start_pos+seq) 一行
        // 不遗——input_tensors_ 跨 Forward 复用，图逐行按 new_kv_cache_pos[row]
        // scatter 写 KV，尾行残值 = 上一 chunk/请求的写入位 → 覆写本请求前面的
        // 真实 KV / 越界写）。
        //   - 真实行 [0, chunk_len)：position_ids = 绝对位 abs_start+i（R4）；
        //     new_kv_cache_pos = MapSlot(abs_start+i)（物理槽，环形跳变合法值，
        //     f4a0d60 全行确定性重写防线下无残值）。
        //   - padding 行 [chunk_len, rows)：续 arange（MapSlot 环形）落圆环
        //     未来未写位，后续真实写入前被 mask 不可达、写入时即被覆盖。
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
        Status pos_status = WritePosValues(tensors[idx_pos], pos_values);
        if (!pos_status.ok())
            return pos_status;
        pos_status = WritePosValues(tensors[idx_new_pos], pos_values);
        if (!pos_status.ok())
            return pos_status;
    } else {
        // float（qwen 旧图）与 bool 同承环形语义：
        // position_ids: [1, chunk_len] = 绝对位 abs..abs+chunk_len-1（RoPE
        // 逻辑位，R4）；new_kv_cache_pos 1D = MapSlot(abs+i)（物理槽——图逐行
        // scatter 写位，绕回后取环形跳变值；无绕回时与绝对位同值，现状兼容）。
        // 双工位分埋，杜绝旧"同值直填"在绕回后越界 scatter。
        std::vector<int32_t> pos_values(chunk_len);
        std::vector<int32_t> npos_values(chunk_len);
        for (std::size_t i = 0; i < chunk_len; ++i) {
            pos_values[i] = static_cast<int32_t>(abs_start + i);
            npos_values[i] = static_cast<int32_t>(pool_->MapSlot(abs_start + i));
        }
        Status pos_status = WritePosValues(tensors[idx_pos], pos_values);
        if (!pos_status.ok())
            return pos_status;
        pos_status = WritePosValues(tensors[idx_new_pos], npos_values);
        if (!pos_status.ok())
            return pos_status;
    }

    // 可选工位 sliding_kv_pos（新图滑窗层专用 scatter 写位）：值 = (abs+i)%C
    // 逐行写小环物理位（填充口径与 new_kv_cache_pos 同构——真实行 + padding
    // 行续 arange 落小环未来未写位、撞末真实写位 clamp；环宽独立为 C=W+P，
    // kSliding 纯模，与 mask_sliding 列轴同源）。守卫（旧图无工位跳过 / 图带
    // 而 C 算不出宁拒）与落写由 WriteSlidingPos 壳单源承担。
    const std::size_t sliding_C = (sw_head_dim_ > 0) ? SlidingLayerWidth(sliding_window_, prefill_max_) : 0;
    Status sliding_pos_status = Status::SUCCESS;
    if (sliding_C > 0) {
        // C>0（新契）才构值：C==0 时跳过构值直落壳（工位在场 → 壳内 C==0
        // 宁拒；工位缺席 → 壳 SUCCESS，均零构值开销）
        auto slide_values = SlidingPosValues(abs_start, rows, chunk_len, sliding_C);
        if (slide_values)
            sliding_pos_status = WriteSlidingPos(tensors, name_index, *slide_values, sliding_C);
    } else {
        sliding_pos_status = WriteSlidingPos(tensors, name_index, {}, sliding_C);
    }
    if (!sliding_pos_status.ok())
        return sliding_pos_status;

    if (mode == MaskMode::kFloatSingle) {
        // attention_mask: [1,1,rows,kv_max] 静态全宽（CorrectTensorDescs 修形
        // 恒全 kv_max 宽）。随同 bool 路径走 WriteFloatMask
        // ——列轴=物理槽，行 r attend 逻辑位 j ∈ [0, abs_start+r] 经 MapSlot
        // 映射清 0；有序写入次序不变量成立（行序=块内 token 序=逻辑绝对位序），
        // 绕回覆写后行 i ≥ 新主写入行 j' 时 attend 的已是新主（同参考实现
        // FillMask 逐列 GetRollbackPos 映射，绕回后同 chunk 内不同 token 可见
        // 窗口长度均不同）。无绕回时与旧"动态列宽因果"逐值一致。
        std::vector<int32_t> q_abs(chunk_len);
        for (std::size_t i = 0; i < chunk_len; ++i)
            q_abs[i] = static_cast<int32_t>(abs_start + i);
        Status mask_status = WriteMask(tensors[idx_attn], q_abs, rows, kv_max,
                                       /*window=*/0, MaskRing::kAnchored, pool_.get());
        if (!mask_status.ok())
            return mask_status;
    } else {
        // gemma4 双 bool mask（golden 标杆 run_gemma4_gen.py / 新 golden_meta 数据
        // 复核）：[1,1,rows,kv_max] 静态全 buffer，列轴 = 物理槽。q_abs 填**绝对位**
        // （WriteBoolMask 内部按 attend 窗口 [max(0, q-window+1), q] 的逻辑位逐位
        // MapSlot 映射到物理槽集——design §4.3 mask 逻辑列→物理槽映射；无绕回时
        // 与无绕回时逐值一致，行为兼容）。rows 已按 mask 工位容量推导；mask 先
        // 全置 1、只清真实行 attend 区间——padding 行 [chunk_len, rows) 保持
        // 初始全 1（有效行对所有 padding 写入槽位全遮蔽 + padding 行输出无人
        // gather，全 1 不影响有效结果；golden dump 实测 padding 行全 1）。
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
        // mask_full：全因果，kv_max 锚定环（零语义变化）。
        Status mask_status = WriteMask(tensors[idx_full], q_abs, rows, kv_max, /*window=*/0, MaskRing::kAnchored,
                                       pool_.get());
        if (!mask_status.ok())
            return mask_status;
        // mask_sliding（sliding_kv_seq_dim_rebind D5/D9，mask-KV 对偶）：
        // 新契（sw_head_dim>0）：列宽 = C = W+P，kSliding 纯模环 j%C——与图侧
        // 写位同映射；旧契：滑窗层 KV 与 full 层同槽域，列宽 kv_max、kAnchored
        // MapSlot 环 + window=W 截断（pre-b688221 字面语义；sw=0 时 window=0
        // 自然全因果，与 mask_full 同值）。分派只看配置 flag，不做宽度相等
        // 比较（D9——"C 恰等 kv_max"的合法滑窗会被值判据误判成全因果）。
        if (sw_head_dim_ > 0) {
            const std::size_t width = SlidingLayerWidth(sliding_window_, prefill_max_);
            mask_status = WriteMask(tensors[idx_sliding], q_abs, rows, width, sliding_window_, MaskRing::kSliding,
                                    pool_.get());
        } else {
            mask_status = WriteMask(tensors[idx_sliding], q_abs, rows, kv_max, sliding_window_, MaskRing::kAnchored,
                                    pool_.get());
        }
        if (!mask_status.ok())
            return mask_status;
    }

    // output_pos（可选工位）：prefill 行号 = 本 chunk 末 token 行（chunk_len-1）。
    // 行号相对本 chunk 序列维（token 连续落位自行 0 起，FillInputs/mask 同基准），
    // 不随光标平移；短块时指向真实末行而非 stale padding 行。
    Status output_pos_status = FillOutputPos(tensors, name_index, static_cast<std::int64_t>(chunk_len - 1));
    if (!output_pos_status.ok())
        return output_pos_status;

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

    const std::size_t kv_max = static_cast<std::size_t>(pool_->kv_cache_max_len());

    // 各 slot 写入记账（Advance 前；R4 分离记账）：abs[i] = total_written
    // （绝对位，position_ids 语义）；cursors[i] = write_pos = MapSlot(abs[i])
    // （物理槽，new_kv_cache_pos 语义）。无绕回时两值相等（现状兼容）。
    std::vector<std::size_t> abs_pos(slots.size());
    std::vector<std::size_t> cursors(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (slots[i] < 0)
            return Status::INVALID_ARGUMENT;
        abs_pos[i] = pool_->TotalWritten(slots[i]);
        cursors[i] = pool_->NextPosition(slots[i]);
    }

    // kv_max = 0 防御（环形化后光标触顶不再拒绝——R5 绕回即语义；仅容量
    // 塌零这一构造异常态仍拦，与 prefill 侧同口径）。
    if (kv_max == 0) {
        CANN_LLM_LOGE("executor.assemble.decode: kv_max=0 (pool malformed)");
        return Status::FAILURE;
    }

    // 图行数（mask 工位容量推导，bool/float 按模式择工位；WriteBoolMask/
    // WriteFloatMask 复验严格相等）。float decode 行数 = decode batch 维
    // （≥ active 槽数；inactive lane 由 mask 全屏蔽行兜底）。择位/元素步长
    // 口径提入 PickMaskSlot()/ElemSize()（prefill/decode 共用）。
    auto mask_slot = tensors[PickMaskSlot(mode, idx_full, idx_attn)];
    if (!mask_slot)
        return Status::INVALID_STATE;
    const DataType mask_dt = mask_slot->GetTensorDesc().dataType;
    std::size_t rows = mask_slot->GetSize() / (kv_max * ElemSize(mode, mask_dt));
    if (rows < slots.size()) {
        CANN_LLM_LOGE("executor.assemble.decode: mask rows=%zu < batch=%zu (kv_max=%zu)", rows, slots.size(), kv_max);
        return Status::INVALID_ARGUMENT;
    }
    if (mode == MaskMode::kGemmaBool) {
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
        Status pos_status = WritePosValues(tensors[idx_pos], pos_values);
        if (!pos_status.ok())
            return pos_status;
        pos_status = WritePosValues(tensors[idx_new_pos], npos_values);
        if (!pos_status.ok())
            return pos_status;
    } else {
        // float（qwen 旧图）：R4 分离记账与 gemma bool 路径同口径——
        // position_ids: [batch, 1] = 各 slot 绝对位 abs（RoPE 逻辑位）；
        // new_kv_cache_pos: [batch, 1] per-slot = 物理槽 cursors[i] =
        // MapSlot(abs[i])（scatter 写位，绕回值恒 < kv_max）。无绕回时两值
        // 相等（旧图现状兼容）；绕回后 abs ≥ kv_max，物理槽经 MapSlot 收敛
        // 回环内（REQ-9 预算闸下 decode 步数只受 max_io_tokens 总预算约束）。
        std::vector<int32_t> pos_values(slots.size());
        std::vector<int32_t> npos_values(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i) {
            pos_values[i] = static_cast<int32_t>(abs_pos[i]);
            npos_values[i] = static_cast<int32_t>(cursors[i]);
        }
        write_i32(tensors[idx_pos], pos_values);
        write_i32(tensors[idx_new_pos], npos_values);
    }

    // 可选工位 sliding_kv_pos（新图滑窗层专用 scatter 写位）：decode per lane
    // 单值 = abs % C（与 new_kv_cache_pos 全 lane 确定性填充同构——active lane
    // 本槽位、inactive lane 复制 lane 0 = 写入位须合法原则；环宽独立 C=W+P
    // kSliding 纯模，与 mask_sliding 列轴同源）。守卫与落写同 prefill 单源壳。
    // decode 各 lane abs 异构，不适用 SlidingPosValues 单一起点 arange——
    // per-lane 直算后走壳。
    {
        const std::size_t sliding_C = (sw_head_dim_ > 0) ? SlidingLayerWidth(sliding_window_, prefill_max_) : 0;
        std::vector<int32_t> slide_values(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            const std::size_t idx = (i < slots.size()) ? i : 0;
            slide_values[i] = (sliding_C > 0) ? static_cast<int32_t>(abs_pos[idx] % sliding_C) : 0;
        }
        Status sliding_status = WriteSlidingPos(tensors, name_index, slide_values, sliding_C);
        if (!sliding_status.ok())
            return sliding_status;
    }

    if (mode == MaskMode::kFloatSingle) {
        // attention_mask: [batch,1,rows,kv_max] 静态全宽。
        // 弃旧"平铺前缀 [0, cursor]"（绕回后欠 attend——真实在线可达缺口），
        // 改 WriteFloatMask 与 bool/gemma 同口径：q_abs = 绝对位 abs_pos[i]，
        // 行 i attend 逻辑位 j ∈ [0, abs_pos[i]] 经 MapSlot 映射物理槽清 0。
        // 无绕回时与旧平铺逐值一致（abs < kv_max 时 attend 集 = [0, abs]，
        // MapSlot 恒等，行宽 attend 到 abs+1 = cursor+1 列）。
        std::vector<int32_t> q_abs(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
            q_abs[i] = static_cast<int32_t>(abs_pos[i]);
        Status mask_status = WriteMask(tensors[idx_attn], q_abs, rows, kv_max,
                                       /*window=*/0, MaskRing::kAnchored, pool_.get());
        if (!mask_status.ok())
            return mask_status;
    } else {
        // gemma4 双 bool mask（golden 对齐语义）：[batch,1,rows,kv_max] 静态全
        // buffer，列轴=物理槽。q_abs 填**绝对位**（环形窗口逻辑位集→物理槽映射
        // 在 WriteBoolMask 内，design §4.3）。rows 已按 mask 工位容量推导
        // （=图行数=decode_batch_size，自洽不依赖 dims 修正，WriteBoolMask 复验
        // 严格相等）；mask 先全置 1、只清 active lane 的 attend 区间——
        // inactive lane [slots.size(), rows) 保持初始全 1（完全屏蔽，
        // 与 prefill padding 行同口径）。
        std::vector<int32_t> q_abs(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
            q_abs[i] = static_cast<int32_t>(abs_pos[i]);
        if (sliding_window_ == 0 && !warned_window_) {
            warned_window_ = true;
            CANN_LLM_LOGW("assembler.mask: graph has mask_sliding slot but sliding_window=0 "
                          "(meta_json missing?) — sliding layers degrade to full causal");
        }
        Status mask_status = WriteMask(tensors[idx_full], q_abs, rows, kv_max, /*window=*/0, MaskRing::kAnchored,
                                       pool_.get());
        if (!mask_status.ok())
            return mask_status;
        // mask_sliding：列宽/映射环同 prefill 侧口径（sliding_kv_seq_dim_rebind
        // D5/D9）——新契 C=W+P 小环 kSliding 纯模 j%C；旧契 kv_max 大环
        // kAnchored + window=W 截断（pre-b688221 语义；sw=0 自然全因果）。
        if (sw_head_dim_ > 0) {
            const std::size_t width = SlidingLayerWidth(sliding_window_, prefill_max_);
            mask_status = WriteMask(tensors[idx_sliding], q_abs, rows, width, sliding_window_, MaskRing::kSliding,
                                    pool_.get());
        } else {
            mask_status = WriteMask(tensors[idx_sliding], q_abs, rows, kv_max, sliding_window_, MaskRing::kAnchored,
                                    pool_.get());
        }
        if (!mask_status.ok())
            return mask_status;
    }

    // output_pos（可选工位）：decode 序列维恒 1，唯一 token 行号 0
    Status output_pos_status = FillOutputPos(tensors, name_index, 0);
    if (!output_pos_status.ok())
        return output_pos_status;

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

} // namespace cann_local_lm::engine
