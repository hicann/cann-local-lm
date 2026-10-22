// KvCachePool 实现（GREEN）。DESIGN M-02 / C-04：KV tensor 经 TensorManager
// 构造时单次批量申请；槽位/光标记账本模块自持。
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "infra/log.h"

#include <limits>
#include <utility>

namespace cann_lm::engine {

KvCachePool::KvCachePool(int max_batch_size, int kv_cache_max_len, int num_hidden_layers,
                         int num_kv_heads, int head_dim, DataType dtype,
                         const std::vector<std::string>& layer_types,
                         int sliding_window)
    : max_batch_size_(max_batch_size > 0 ? max_batch_size : 0),
      kv_cache_max_len_(kv_cache_max_len > 0 ? kv_cache_max_len : 0),
      num_hidden_layers_(num_hidden_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      dtype_(dtype),
      slots_(static_cast<std::size_t>(max_batch_size > 0 ? max_batch_size : 0)) {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].id = static_cast<SlotId>(i);
        slots_[i].cursor = 0;
        slots_[i].active = false;
    }
    // Qwen3.5 稀疏化（support_qwen3.5.md §4.3）：layer_types 空 → 全部层稠密建 KV
    // （现状行为，零行为变化）；非空且长度 == num_hidden_layers → linear_attention
    // 层不建、full_attention / sliding_attention 层建（升序全局层号）；长度不符 →
    // WARN 回退稠密（防御）。
    if (layer_types.empty() ||
        static_cast<int>(layer_types.size()) != num_hidden_layers_) {
        if (!layer_types.empty()) {
            CANN_LLM_LOGW(
                "kv_pool: layer_types size (%zu) != num_hidden_layers (%d) — "
                "回退稠密池（全部层建 KV）",
                layer_types.size(), num_hidden_layers_);
        }
        kv_layer_ids_.reserve(static_cast<std::size_t>(
            num_hidden_layers_ > 0 ? num_hidden_layers_ : 0));
        for (int i = 0; i < num_hidden_layers_; ++i) kv_layer_ids_.push_back(i);
    } else {
        for (int i = 0; i < num_hidden_layers_; ++i) {
            if (layer_types[static_cast<std::size_t>(i)] != "linear_attention") {
                kv_layer_ids_.push_back(i);   // full/sliding_attention（linear 不建池）
            }
        }
    }
    // 构造期申请全部 KV tensor（C-04；all-or-nothing）。gemma4 滑窗异构 KV：
    // sliding_attention 层末维 = sliding_window、其余层末维 = head_dim
    //（meta layer_types 权威）；连续同末维层合批建形——无滑窗异构时保持
    // 单批建形不变（零行为变化）
    auto& tm = TensorManager::Instance();
    const bool typed_pool = !layer_types.empty() &&
        static_cast<int>(layer_types.size()) == num_hidden_layers_;
    const auto end_dim_of = [&](std::size_t pos) -> int {
        const int lay = kv_layer_ids_[pos];
        return (typed_pool && sliding_window > 0 &&
                layer_types[static_cast<std::size_t>(lay)] == "sliding_attention")
                   ? sliding_window
                   : head_dim_;
    };
    int alloc_code = 0;   // Status::SUCCESS
    if (kv_layer_ids_.empty()) {   // 全 linear 层（无建池层）→ 与旧行为一致显式失败
        alloc_code = static_cast<int>(Status::INVALID_ARGUMENT);
    }
    for (std::size_t seg_begin = 0; seg_begin < kv_layer_ids_.size();) {
        std::size_t seg_end = seg_begin + 1;
        while (seg_end < kv_layer_ids_.size() &&
               end_dim_of(seg_end) == end_dim_of(seg_begin)) {
            ++seg_end;
        }
        const std::vector<int> seg_ids(
            kv_layer_ids_.begin() + static_cast<std::ptrdiff_t>(seg_begin),
            kv_layer_ids_.begin() + static_cast<std::ptrdiff_t>(seg_end));
        const std::vector<int32_t> seg_dims = {kv_cache_max_len_, num_kv_heads_,
                                               max_batch_size_, end_dim_of(seg_begin)};
        std::vector<std::shared_ptr<ITensorBuffer>> seg_ts;
        Status st = tm.CreateKvCacheTensor(seg_ids, seg_dims, dtype_, &seg_ts);
        if (!st.ok()) {
            alloc_code = static_cast<int>(st.code);
            break;
        }
        tensors_.insert(tensors_.end(), seg_ts.begin(), seg_ts.end());
        seg_begin = seg_end;
    }
    CANN_LLM_LOGD(
        "kv_pool.alloc: CreateKvCacheTensor (st=%d, kv_layers=%zu/%d, kv_len=%d, "
        "kv_heads=%d, batch=%d, head_dim=%d, sliding=%d, dtype_=%d)",
        alloc_code, kv_layer_ids_.size(), num_hidden_layers_, kv_cache_max_len_,
        num_kv_heads_, max_batch_size_, head_dim_, sliding_window, dtype_);

    if (alloc_code != 0) {
        CANN_LLM_LOGE(
            "kv_pool.alloc: CreateKvCacheTensor failed (st=%d, kv_layers=%zu/%d, kv_len=%d, "
            "kv_heads=%d, batch=%d, head_dim=%d, sliding=%d)",
            alloc_code, kv_layer_ids_.size(), num_hidden_layers_, kv_cache_max_len_,
            num_kv_heads_, max_batch_size_, head_dim_, sliding_window);
        tensors_.clear();   // 空池继续，槽位记账不受影响（all-or-nothing）
    }
    layout_locked_.assign(kv_layer_ids_.size(), false);
}

Status KvCachePool::EnsureKvLayout(int layer, const std::vector<int32_t>& dims, DataType dtype) {
    // 入参合法性（图 desc 防御）：layer >= 0、dims 非空逐维 > 0、dtype 非 RESERVED
    if (layer < 0 || dims.empty() || dtype == DataType::RESERVED) return Status::INVALID_ARGUMENT;
    for (const int32_t dim : dims) {
        if (dim <= 0) return Status::INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    // 构造期申请失败（空池）→ 无从对齐（绑定侧 pool tensor missing 收口）
    if (tensors_.empty()) return Status::SUCCESS;
    // 层号 → 池内下标（同 TensorAt 映射；linear 层/越界不在池内）
    std::size_t pos = kv_layer_ids_.size();
    for (std::size_t i = 0; i < kv_layer_ids_.size(); ++i) {
        if (kv_layer_ids_[i] == layer) {
            pos = i;
            break;
        }
    }
    if (pos == kv_layer_ids_.size()) return Status::FAILURE;   // 层不在池内（资源类口径）
    const auto& cur = tensors_[2 * pos]->GetTensorDesc();
    if (cur.dataType != dtype) {
        CANN_LLM_LOGE(
            "kv_pool.ensure_layout: dtype mismatch (layer=%d, pool=%d, graph=%d) —— "
            "meta_json kv_cache_type 与 om 图 KV 工位失配",
            layer, static_cast<int>(cur.dataType), static_cast<int>(dtype));
        return Status::INVALID_ARGUMENT;
    }
    // 轴序/维数一致 → no-op（该层首个调用同时锁层，后续同层双图失配由层锁把关）
    if (cur.dims == dims) {
        layout_locked_[pos] = true;
        return Status::SUCCESS;
    }
    if (layout_locked_[pos]) {
        CANN_LLM_LOGE(
            "kv_pool.ensure_layout: layer %d layout locked by first-loaded graph, "
            "dims differ —— 双图同层 KV 布局不一致", layer);
        return Status::INVALID_ARGUMENT;
    }
    // 元素总数一致才可按图轴序重建该层（uint64 累乘 + 溢出防护；真失配显式报错）
    const auto mul = [](uint64_t a, uint64_t b, uint64_t* r) -> bool {
        if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
        *r = a * b;
        return true;
    };
    uint64_t cur_elems = 1, want_elems = 1;
    for (const int32_t d : cur.dims) {
        if (!mul(cur_elems, static_cast<uint64_t>(d), &cur_elems)) {
            return Status::INVALID_ARGUMENT;
        }
    }
    for (const int32_t d : dims) {
        if (!mul(want_elems, static_cast<uint64_t>(d), &want_elems)) {
            return Status::INVALID_ARGUMENT;
        }
    }
    if (cur_elems != want_elems) {
        CANN_LLM_LOGE(
            "kv_pool.ensure_layout: layer %d element count mismatch (pool=%llu, "
            "graph=%llu) —— meta_json 池容量与 om 图失配，不可按图轴序重建",
            layer,
            static_cast<unsigned long long>(cur_elems),
            static_cast<unsigned long long>(want_elems));
        return Status::INVALID_ARGUMENT;
    }
    // 仅重建该层 K/V 两个 tensor（all-or-nothing：失败保留旧 tensor，不破坏现状；
    // 层清单/槽位记账不变；gemma4 滑窗异构 KV 下他层不动；此时未发生任何工位
    // 绑定，重建安全）
    auto& tm = TensorManager::Instance();
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    Status st = tm.CreateKvCacheTensor({layer}, dims, dtype, &ts);
    if (!st.ok()) {
        CANN_LLM_LOGE(
            "kv_pool.ensure_layout: relayout create failed (st=%d, layer=%d)",
            st.code, layer);
        return st;   // 该层旧 K/V 原样保留
    }
    tensors_[2 * pos] = std::move(ts[0]);
    tensors_[2 * pos + 1] = std::move(ts[1]);
    layout_locked_[pos] = true;
    CANN_LLM_LOGI(
        "kv_pool.ensure_layout: layer %d relayout to om graph dims (kv_elems=%llu, "
        "dtype=%d) —— HF 命名族图轴序对齐",
        layer, static_cast<unsigned long long>(want_elems), static_cast<int>(dtype));
    return Status::SUCCESS;
}

Status KvCachePool::AllocKvSlot(SlotId* out) {
    if (!out) return Status::INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lk(mtx_);
    // 1) 优先取首个非 active 槽
    for (auto& s : slots_) {
        if (!s.active) {
            s.active    = true;
            s.cursor    = 0;
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
        victim->active    = false;
        victim->cursor    = 0;   // 驱逐重置
    }
    // 3) 复用刚释放的槽
    for (auto& s : slots_) {
        if (!s.active) {
            s.active    = true;
            s.cursor    = 0;
            s.alloc_seq = ++next_seq_;
            *out = s.id;
            return Status::SUCCESS;
        }
    }
    return Status::FAILURE;   // 无槽（max_batch_size<=0）
}

Status KvCachePool::FreeKvSlot(SlotId id) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size()) return Status::SUCCESS;  // 幂等
    slots_[static_cast<std::size_t>(id)].active = false;
    // cursor 不清零（保留以备观测）；下次 Alloc 会重置
    return Status::SUCCESS;  // 重复 Free 幂等
}

Status KvCachePool::Advance(SlotId id, std::size_t n) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size()) return Status::FAILURE;
    auto& s = slots_[static_cast<std::size_t>(id)];
    if (!s.active) return Status::FAILURE;
    if (s.cursor + n > static_cast<std::size_t>(kv_cache_max_len_)) return Status::FAILURE;  // 越界 🔴
    s.cursor += n;
    return Status::SUCCESS;
}

std::size_t KvCachePool::NextPosition(SlotId id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (id < 0 || static_cast<std::size_t>(id) >= slots_.size()) return 0;
    return slots_[static_cast<std::size_t>(id)].cursor;
}

Status KvCachePool::Snapshot(std::vector<SlotInfo>* out) const {
    if (!out) return Status::INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lk(mtx_);
    *out = slots_;
    return Status::SUCCESS;
}

std::size_t KvCachePool::TensorCount() const noexcept { return tensors_.size(); }

int KvCachePool::NumKvLayers() const noexcept {
    return static_cast<int>(kv_layer_ids_.size());
}

const std::vector<int>& KvCachePool::KvLayerIds() const noexcept {
    return kv_layer_ids_;
}

std::shared_ptr<ITensorBuffer> KvCachePool::TensorAt(int layer, int kv) const noexcept {
    if (layer < 0 || kv < 0 || kv > 1) return nullptr;
    // 全局层号 → kv_layer_ids_ 序（升序小表，线性足够）
    std::size_t pos = kv_layer_ids_.size();
    for (std::size_t i = 0; i < kv_layer_ids_.size(); ++i) {
        if (kv_layer_ids_[i] == layer) {
            pos = i;
            break;
        }
    }
    if (pos == kv_layer_ids_.size()) return nullptr;   // 非 full 层 / 未声明
    const std::size_t idx = pos * 2 + static_cast<std::size_t>(kv);
    return idx < tensors_.size() ? tensors_[idx] : nullptr;
}

std::shared_ptr<ITensorBuffer> KvCachePool::KTensor(int layer) const noexcept {
    return TensorAt(layer, /*kv=*/0);
}

std::shared_ptr<ITensorBuffer> KvCachePool::VTensor(int layer) const noexcept {
    return TensorAt(layer, /*kv=*/1);
}

}  // namespace cann_lm::engine
