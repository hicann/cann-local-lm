// KvCachePool 实现（Cache_pool_simplify.md R1-R3+T1）。
// 构造期：槽位/层表/期望值记账（不申请 tensor）；tensor 显存由首个执行器
// Load 时经 EnsureKvTensor 按 om 图工位 desc 惰性创建（R1/T1），双图一致性
// 由 EnsureKvTensor R2 复验把关——与 LinearStatePool::EnsureStateTensor
// 同构 verify-or-create（创建即整只清零）。
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "infra/log.h"

#include <cstring>
#include <string>
#include <utility>

namespace cann_lm::engine {

namespace {
/// dims 列表 → "a, b, c"（日志用；与 state pool 日志风格一致）
std::string FormatDimsForLog(const std::vector<int32_t>& dims)
{
    std::string s;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += std::to_string(dims[i]);
    }
    return s;
}
} // namespace

KvCachePool::KvCachePool(int max_batch_size, int kv_cache_max_len, int num_hidden_layers, int num_kv_heads,
                         int head_dim, const std::vector<std::string>& layer_types, int sliding_window)
    : max_batch_size_(max_batch_size > 0 ? max_batch_size : 0),
      kv_cache_max_len_(kv_cache_max_len > 0 ? kv_cache_max_len : 0),
      num_hidden_layers_(num_hidden_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      sliding_window_(sliding_window),
      slots_(static_cast<std::size_t>(max_batch_size > 0 ? max_batch_size : 0))
{
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].id = static_cast<SlotId>(i);
        slots_[i].total_written = 0;
        slots_[i].write_pos = 0;
        slots_[i].active = false;
    }
    // Qwen3.5 稀疏化（support_qwen3.5.md §4.3）：layer_types 空 → 全部层稠密建 KV
    // （现状行为，零行为变化）；非空且长度 == num_hidden_layers → linear_attention
    // 层不建、full_attention / sliding_attention 层建（升序全局层号）；长度不符 →
    // WARN 回退稠密（防御）。
    if (layer_types.empty() || static_cast<int>(layer_types.size()) != num_hidden_layers_) {
        if (!layer_types.empty()) {
            CANN_LLM_LOGW("kv_pool: layer_types size (%zu) != num_hidden_layers (%d) — "
                          "回退稠密池（全部层建 KV）",
                          layer_types.size(), num_hidden_layers_);
        }
        kv_layer_ids_.reserve(static_cast<std::size_t>(num_hidden_layers_ > 0 ? num_hidden_layers_ : 0));
        for (int i = 0; i < num_hidden_layers_; ++i)
            kv_layer_ids_.push_back(i);
    } else {
        for (int i = 0; i < num_hidden_layers_; ++i) {
            if (layer_types[static_cast<std::size_t>(i)] != "linear_attention") {
                kv_layer_ids_.push_back(i); // full/sliding_attention（linear 不建池）
            }
        }
    }
    // 预计算每建池层 HF 期望 dims（R3 比对值，kv_layer_ids_ 序）：
    // expect = [batch_size, head_num, max_kv_len, 末维]（HF 序），
    // 末维 = sliding_attention 层且 sliding_window>0 时取 sliding_window、
    // 其余层取 head_dim（gemma4 滑窗异构 KV 期望口径——仅作比对期望，
    // 不再用于建 tensor，R1：om 图工位 desc 为 tensor 实形唯一权威）。
    const bool typed_pool = !layer_types.empty() && static_cast<int>(layer_types.size()) == num_hidden_layers_;
    expect_dims_.reserve(kv_layer_ids_.size());
    for (const int lay : kv_layer_ids_) {
        const int end_dim = (typed_pool && sliding_window_ > 0 &&
                             layer_types[static_cast<std::size_t>(lay)] == "sliding_attention") ?
                                sliding_window_ :
                                head_dim_;
        expect_dims_.push_back({max_batch_size_, num_kv_heads_, kv_cache_max_len_, end_dim});
    }
    // tensor 槽位预置全空——EnsureKvTensor 逐层落位（T1：显存创建后移到
    // 首个执行器 Load 期）。全 linear 层（kv_layer_ids_ 空）空池静默成立。
    tensors_.assign(kv_layer_ids_.size() * 2, nullptr);
    CANN_LLM_LOGD("kv_pool.init: lazy-create armed (kv_layers=%zu/%d, kv_len=%d, kv_heads=%d, "
                  "batch=%d, head_dim=%d, sliding=%d) — tensors created by first executor Load",
                  kv_layer_ids_.size(), num_hidden_layers_, kv_cache_max_len_, num_kv_heads_, max_batch_size_,
                  head_dim_, sliding_window);
}

int KvCachePool::LayerPos(int layer) const noexcept
{
    // 升序小表，线性足够（与 TensorAt 同映射）
    for (std::size_t i = 0; i < kv_layer_ids_.size(); ++i) {
        if (kv_layer_ids_[i] == layer)
            return static_cast<int>(i);
    }
    return -1;
}

Status KvCachePool::EnsureKvTensor(int layer, bool is_value, const TensorDesc& desc)
{
    // desc 合法性（图 desc 防御）：dims 非空逐维 >0、dtype 非 RESERVED
    if (layer < 0 || desc.dims.empty() || desc.dataType == DataType::RESERVED) {
        return Status::INVALID_ARGUMENT;
    }
    for (const int32_t dim : desc.dims) {
        if (dim <= 0)
            return Status::INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    // 层号 → 池内下标（linear 层/越界不在池内）
    const int pos = LayerPos(layer);
    if (pos < 0) {
        CANN_LLM_LOGE("kv_pool.ensure_tensor: layer %d is not a KV-pool layer "
                      "(pool kv layers = %d)",
                      layer, NumKvLayers());
        return Status::FAILURE; // 资源类口径（Load 侧层型校验在前，此为防御）
    }
    auto& slot = tensors_[static_cast<std::size_t>(2 * pos + (is_value ? 1 : 0))];
    const char* kv_name = is_value ? "V" : "K";

    // ── R3：HF 期望序 [batch,heads,len,dim] 与图 dims 比对（先于建形成败判定；
    //    创建/复验路径共用。只认 HF 序逐位比对，不兼容旧约定序——2026-10-15 裁定）──
    if (desc.dims.size() != 4) {
        // dims 数 ≠ 4：无法逐位比对 → 告警一次（池生命周期去重）后照常直建
        // （2026-10-15 裁定）
        static constexpr uint32_t kWarnDimsCount = 1u << 0;
        if (!(warned_.fetch_or(kWarnDimsCount, std::memory_order_relaxed) & kWarnDimsCount)) {
            CANN_LLM_LOGW("kv_pool.ensure_tensor: non-4D KV slot desc (layer=%d, %s, "
                          "ndims=%zu, dims=[%s]) — 跳过 HF 期望比对，按图 desc 直建（每池"
                          "只告本次）",
                          layer, kv_name, desc.dims.size(), FormatDimsForLog(desc.dims).c_str());
        }
    } else {
        const auto& expect = expect_dims_[static_cast<std::size_t>(pos)];
        // kv_len（dims[2]）/ batch（dims[0]）失配 → 致命终止（2026-10-15 裁定）。
        // 判次序：先报 batch 还是 kv_len 只影响日志文案，两维任一失配即 fatal。
        if (desc.dims[2] != expect[2]) {
            CANN_LLM_LOGE("kv_pool.ensure_tensor: kv_len dim mismatch (layer=%d, %s) — "
                          "expect %d (meta kv_cache_max_len), graph dims[%d]；光标容量/"
                          "mask/Advance 越界全线以 meta 为准，图短于 meta 将致光标推进"
                          "超图缓冲——Load 终止（2026-10-15 裁定）",
                          layer, kv_name, expect[2], static_cast<int>(desc.dims[2]));
            return Status::INVALID_ARGUMENT;
        }
        if (desc.dims[0] != expect[0]) {
            CANN_LLM_LOGE("kv_pool.ensure_tensor: batch dim mismatch (layer=%d, %s) — "
                          "expect %d (meta batch_num 槽位数), graph dims[%d]；图 batch < "
                          "meta 槽位数时 decode 满并发 KV 写入压错位/越 tensor——"
                          "Load 终止（2026-10-15 裁定）",
                          layer, kv_name, expect[0], static_cast<int>(desc.dims[0]));
            return Status::INVALID_ARGUMENT;
        }
        // 其余维（heads dims[1]/末维 dims[3]）失配 → 告警，每类池内去重一次，
        // 不终止（om 图自洽，host 不直接寻址 KV tensor；om 为形状权威）
        uint32_t cfg = 0;
        if (desc.dims[1] != expect[1])
            cfg |= 1u << 1; // heads 维
        if (desc.dims[3] != expect[3])
            cfg |= 1u << 2; // 末维
        const uint32_t seen = warned_.fetch_or(cfg, std::memory_order_relaxed);
        if (cfg & ~seen) {
            CANN_LLM_LOGW("kv_pool.ensure_tensor: %s dim differs from meta expectation "
                          "(layer=%d, %s) — expect HF [batch=%d, heads=%d, kv_len=%d, "
                          "dim=%d], graph dims=[%s]；om 图为形状权威，继续（每池同类只"
                          "告本次）",
                          (cfg & (1u << 1)) ? ((cfg & (1u << 2)) ? "heads+end" : "heads") : "end", layer, kv_name,
                          expect[0], expect[1], expect[2], expect[3], FormatDimsForLog(desc.dims).c_str());
        }
    }

    // ── R2：已建（首图 Load 已创建）→ 复验 dims+dtype 严格一致（次图把关）──
    if (slot) {
        const auto& cur = slot->GetTensorDesc();
        if (cur.dims != desc.dims || cur.dataType != desc.dataType) {
            CANN_LLM_LOGE("kv_pool.ensure_tensor: layer %d %s dims/dtype mismatch "
                          "(existing vs graph desc) — 双图同层 KV 布局不一致",
                          layer, kv_name);
            return Status::INVALID_ARGUMENT;
        }
        return Status::SUCCESS; // 一致 → 次图复用
    }

    // ── R1：未建 → 按 om 图工位 desc 直建（命名约定不变）──
    TensorDesc d = desc; // dims/dtype/format 以图工位 desc 为权威
    d.tensorName = std::string(is_value ? "kv_cache_v_" : "kv_cache_k_") + std::to_string(layer);
    std::vector<std::shared_ptr<ITensorBuffer>> created;
    Status st = TensorManager::Instance().CreateKvCacheTensorWithDescs({d}, &created);
    if (!st.ok()) {
        CANN_LLM_LOGE("kv_pool.ensure_tensor: layer %d %s create failed (st=%d)", layer, kv_name, st.code);
        return st; // 该层不落位（未建可重试，all-or-nothing 天然保留）
    }
    // 创建即整只清零（仅新建路径；与 EnsureStateTensor 同口径——2026-10-15
    // 裁定）：mask 纪律万一失效时残值=0 而非随机数；0 通道冷页预触对 ECC/带宽
    // 友好；两池口径统一。复用路径不重复清——运行期 lane 生命周期仍靠 cursor
    // 覆写 + mask。
    std::shared_ptr<ITensorBuffer> t = std::move(created[0]);
    std::memset(t->GetData(), 0, t->GetSize());
    CANN_LLM_LOGI("kv_pool.ensure_tensor: layer %d %s tensor created dims=[%s], dtype=%d, "
                  "bytes=%zu — om graph desc 直建",
                  layer, kv_name, FormatDimsForLog(desc.dims).c_str(), static_cast<int>(desc.dataType), t->GetSize());
    slot = std::move(t);
    return Status::SUCCESS;
}

Status KvCachePool::AllocKvSlot(SlotId* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lk(mtx_);
    // 1) 优先取首个非 active 槽
    for (auto& s : slots_) {
        if (!s.active) {
            s.active = true;
            s.total_written = 0;
            s.write_pos = 0;
            s.alloc_seq = ++next_seq_;
            *out = s.id;
            return Status::SUCCESS;
        }
    }
    // 2) 池满 → 驱逐最早分配的 active 槽（真正的 FIFO，按 alloc_seq 最小者）
    SlotInfo* victim = nullptr;
    for (auto& s : slots_) {
        if (s.active) {
            if (!victim || s.alloc_seq < victim->alloc_seq) {
                victim = &s;
            }
        }
    }
    if (victim) {
        victim->active = false;
        victim->total_written = 0; // 驱逐重置
        victim->write_pos = 0;
    }
    // 3) 复用刚释放的槽
    for (auto& s : slots_) {
        if (!s.active) {
            s.active = true;
            s.total_written = 0;
            s.write_pos = 0;
            s.alloc_seq = ++next_seq_;
            *out = s.id;
            return Status::SUCCESS;
        }
    }
    return Status::FAILURE; // 无槽（max_batch_size<=0）
}

Status KvCachePool::FreeKvSlot(SlotId id)
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return Status::SUCCESS; // 幂等
    slots_[static_cast<std::size_t>(id)].active = false;
    // 记账不清零（保留以备观测）；下次 Alloc 会重置
    return Status::SUCCESS; // 重复 Free 幂等
}

Status KvCachePool::Advance(SlotId id, std::size_t n)
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return Status::FAILURE;
    auto& s = slots_[static_cast<std::size_t>(id)];
    if (!s.active)
        return Status::FAILURE;
    // 环形化（R5/ADR-0001）：越 kv_max 绕回不报错——原"越界即 FAILURE"
    // 质量红线作废，容量硬约束前移到 Submit 前置校验（spec REQ-9，
    // ids.size() ≤ kv_max 单次全量须一次装入）。
    s.total_written += n;
    s.write_pos = MapSlot(s.total_written);
    return Status::SUCCESS;
}

std::size_t KvCachePool::NextPosition(SlotId id) const
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return 0;
    return slots_[static_cast<std::size_t>(id)].write_pos;
}

std::size_t KvCachePool::TotalWritten(SlotId id) const
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return 0;
    return slots_[static_cast<std::size_t>(id)].total_written;
}

Status KvCachePool::RollbackTo(SlotId id, std::size_t target_total_written)
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return Status::FAILURE;
    auto& s = slots_[static_cast<std::size_t>(id)];
    if (target_total_written > s.total_written) {
        CANN_LLM_LOGE("kv_pool.rollback: refuse forward move (slot=%d, cur=%zu, target=%zu)", id, s.total_written,
                      target_total_written);
        return Status::INVALID_ARGUMENT;
    }
    s.total_written = target_total_written;
    s.write_pos = MapSlot(s.total_written);
    return Status::SUCCESS;
}

std::size_t KvCachePool::MapSlot(std::size_t pos) const noexcept
{
    const std::size_t kv_max = static_cast<std::size_t>(kv_cache_max_len_);
    if (kv_max == 0)
        return 0;
    if (pos < kv_max)
        return pos;
    // 锚定式绕回（R5，同参考实现 GetRollbackPos）：
    //   [0, init_token_len) 锚定段不动；[init_token_len, kv_max) 尾圆环绕圈。
    // 退化保护：init_token_len >= kv_max（全缓冲锚定，几何不成立）→ 纯模绕回。
    const std::size_t anchor = init_token_len_ < kv_max ? init_token_len_ : 0;
    const std::size_t ring = kv_max - anchor;
    if (ring == 0)
        return anchor == 0 ? 0 : kv_max - 1;
    return anchor + (pos - anchor) % ring;
}

void KvCachePool::SetInitTokenLen(std::size_t len) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    init_token_len_ = len;
}

void KvCachePool::ResetLedgerOnSlot(SlotId id) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return;
    auto& s = slots_[static_cast<std::size_t>(id)];
    s.total_written = 0;
    s.write_pos = 0;
    // R13"一切重建"：init_token_len 是可选本（锚定段 = 本 Context 首轮
    // prompt 段）的链延续参数，miss 重建链整体归零（下轮全量 prefill 后
    // 由 pipeline 依 REQ-3 判定表重设）。WritePosValues 等消费方经
    // MapSlot 只读本值，回零即刻生效。
    init_token_len_ = 0;
}

Status KvCachePool::ResumeSlot(SlotId id, std::size_t target_total_written) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size())
        return Status::FAILURE;
    auto& s = slots_[static_cast<std::size_t>(id)];
    if (target_total_written > s.total_written) {
        CANN_LLM_LOGE("kv_pool.resume: refuse forward move (slot=%d, cur=%zu, target=%zu)", static_cast<int>(id),
                      s.total_written, target_total_written);
        return Status::INVALID_ARGUMENT;
    }
    s.total_written = target_total_written;
    s.write_pos = MapSlot(s.total_written);
    s.active = true;
    s.alloc_seq = ++next_seq_; // lane FIFO 序刷新（本池单 batch 单 lane，防御口径）
    return Status::SUCCESS;
}

Status KvCachePool::Snapshot(std::vector<SlotInfo>* out) const
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lk(mtx_);
    *out = slots_;
    return Status::SUCCESS;
}

std::size_t KvCachePool::TensorCount() const noexcept
{
    // 口径修订（T1）：tensors_ 预置空槽，数非空项——构造后/未加载 = 0；
    // 任一图 Load 成功后 = 2×NumKvLayers()（Load 侧工位计数校验保证全层 Ensure 过）
    std::size_t n = 0;
    for (const auto& t : tensors_) {
        if (t)
            ++n;
    }
    return n;
}

int KvCachePool::NumKvLayers() const noexcept { return static_cast<int>(kv_layer_ids_.size()); }

const std::vector<int>& KvCachePool::KvLayerIds() const noexcept { return kv_layer_ids_; }

std::shared_ptr<ITensorBuffer> KvCachePool::TensorAt(int layer, int kv) const noexcept
{
    if (layer < 0 || kv < 0 || kv > 1)
        return nullptr;
    const int pos = LayerPos(layer);
    if (pos < 0)
        return nullptr; // 非建池层 / 未声明
    const std::size_t idx = static_cast<std::size_t>(pos) * 2 + static_cast<std::size_t>(kv);
    return idx < tensors_.size() ? tensors_[idx] : nullptr;
}

std::shared_ptr<ITensorBuffer> KvCachePool::KTensor(int layer) const noexcept { return TensorAt(layer, /*kv=*/0); }

std::shared_ptr<ITensorBuffer> KvCachePool::VTensor(int layer) const noexcept { return TensorAt(layer, /*kv=*/1); }

} // namespace cann_lm::engine
