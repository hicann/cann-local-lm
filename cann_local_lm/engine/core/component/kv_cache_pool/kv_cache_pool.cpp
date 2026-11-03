// KvCachePool 实现（Cache_pool_simplify.md R1-R3+T1 + ADR-0002
// Session/SetContext：模板池 desc 账本 + 薄壳转发）。
// 构造期：槽位/层表/期望值记账（不申请 tensor、不开显存）。建点双轨
// （ADR-0002）：模板池 Load 期经 EnsureKvDesc 只记 desc 账本（不建 tensor）；
// KV tensor 实建延后到 CreateContext——per-Context 池按模板池已复验 desc
// 逐层 EnsureKvTensor 直建（verify-or-create，创建即整只清零、all-or-nothing）。
// 模板池经 BindFrom 把 tensors_ 薄壳转发指向活动 Context 池列表（Q10 乙）；
// SetContext/DestroyContext 在无在飞窗口切换（Q14 锁纪律）。
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "infra/log.h"

#include <cstring>
#include <string>
#include <utility>

namespace cann_local_lm::engine {

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
                         int head_dim, const std::vector<std::string>& layer_types, int sliding_window,
                         int init_token_len, int prefill_max, int sliding_window_head_dim)
    : max_batch_size_(max_batch_size > 0 ? max_batch_size : 0),
      kv_cache_max_len_(kv_cache_max_len > 0 ? kv_cache_max_len : 0),
      num_hidden_layers_(num_hidden_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      sliding_window_(sliding_window),
      prefill_max_(prefill_max > 0 ? prefill_max : 0),
      sliding_window_head_dim_(sliding_window_head_dim > 0 ? sliding_window_head_dim : 0),
      slots_(static_cast<std::size_t>(max_batch_size > 0 ? max_batch_size : 0))
{
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].id = static_cast<SlotId>(i);
        slots_[i].total_written = 0;
        slots_[i].write_pos = 0;
        slots_[i].active = false;
    }
    // 锚定段构造期定型（meta_json model_config.init_token_len）：
    // 0 = 无锚定段（全缓冲圆环，现状行为零变化）；0 < v < kv_max = 锚定段长
    // 且记入基线（miss 重建恢复本值）。越界（< 0 / ≥ kv_max）= 配置非法 →
    // WARN 降 0（防御深度：解析层已段降级拒绝，正常不可达——同 layer_types
    // 长度不符 WARN 回退口径，不静默采信残值）。
    configured_init_len_ = 0;
    if (init_token_len > 0 && static_cast<int>(init_token_len) < (kv_cache_max_len_ > 0 ? kv_cache_max_len_ : 0)) {
        init_token_len_ = static_cast<std::size_t>(init_token_len);
        configured_init_len_ = init_token_len_;
    } else if (init_token_len != 0) {
        CANN_LLM_LOGW("kv_pool: init_token_len %d out of range (need 0 <= v < kv_max=%d) "
                      "— anchor degraded to 0",
                      init_token_len, kv_cache_max_len_);
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
    // expect = [batch_size, head_num, kv_len, 末维]（HF 序）。
    // 新契（sliding_window_head_dim>0）：sliding_attention 层期望 =
    // [b,h,C=W+P,sw_head_dim]（sliding_kv_seq_dim_rebind D1/D2——滑窗层
    // 专用小环容量 + 新末维）；其余（full 层 / 线性无 KV）= [b,h,kv_max,head_dim]。
    // 旧契（=0）：滑窗层与 full 层同形期望 [b,h,kv_max,head_dim]
    // （pre-b688221 语义；b688221 的"末维=sliding_window"过渡图口径已删 D8
    // ——该图从未产出）。meta 契约与图代错配 → kv_len 维失配致命
    // （CheckKvDescCommon fatal 口径，D7 fail-loud）。仅作比对期望，不用于
    // 建 tensor（R1：om 图工位 desc 为 tensor 实形唯一权威）。
    const bool typed_pool = !layer_types.empty() && static_cast<int>(layer_types.size()) == num_hidden_layers_;
    expect_dims_.reserve(kv_layer_ids_.size());
    for (const int layer : kv_layer_ids_) {
        const bool sw_layer = typed_pool && layer_types[static_cast<std::size_t>(layer)] == "sliding_attention";
        // 末维期望（D2）：新契滑窗层 = sw_head_dim；其余（旧契滑窗层 / full
        // 层）= head_dim（旧契滑窗层与 full 层同形）。
        const int end_dim = (sw_layer && sliding_window_head_dim_ > 0) ? sliding_window_head_dim_ : head_dim_;
        // 序列轴期望（D1/D7）：新契滑窗层 = C = W + P；其余 = kv_max。
        const int seq_dim = (sw_layer && sliding_window_head_dim_ > 0 && prefill_max_ > 0) ?
                                (sliding_window_ + prefill_max_) :
                                kv_cache_max_len_;
        expect_dims_.push_back({max_batch_size_, num_kv_heads_, seq_dim, end_dim});
    }
    // tensor 槽位预置全空——Context 池 = EnsureKvTensor 逐层落位（CreateContext
    // 期，ADR-0002）；模板池 = BindFrom 薄壳转发落位（SetContext 期）。
    // 全 linear 层（kv_layer_ids_ 空）空池静默成立。
    tensors_.assign(kv_layer_ids_.size() * 2, nullptr);
    kv_descs_.assign(kv_layer_ids_.size() * 2, std::nullopt);
    CANN_LLM_LOGD("kv_pool.init: desc-ledger armed (kv_layers=%zu/%d, kv_len=%d, kv_heads=%d, "
                  "batch=%d, head_dim=%d, sliding=%d, sw_head_dim=%d) — tensors created at CreateContext",
                  kv_layer_ids_.size(), num_hidden_layers_, kv_cache_max_len_, num_kv_heads_, max_batch_size_,
                  head_dim_, sliding_window, sliding_window_head_dim_);
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

Status KvCachePool::CheckKvDescCommon(int layer, bool is_value, const TensorDesc& desc, int* pos) const
{
    // desc 合法性（图 desc 防御）：dims 非空逐维 >0、dtype 非 RESERVED
    if (layer < 0 || desc.dims.empty() || desc.dataType == DataType::RESERVED) {
        return Status::INVALID_ARGUMENT;
    }
    for (const int32_t dim : desc.dims) {
        if (dim <= 0)
            return Status::INVALID_ARGUMENT;
    }
    // 层号 → 池内下标（linear 层/越界不在池内）
    const int p = LayerPos(layer);
    if (p < 0) {
        CANN_LLM_LOGE("kv_pool.ensure: layer %d is not a KV-pool layer "
                      "(pool kv layers = %d)",
                      layer, NumKvLayers());
        return Status::FAILURE; // 资源类口径（Load 侧层型校验在前，此为防御）
    }
    if (pos)
        *pos = p;

    // ── R3：HF 期望序 [batch,heads,len,dim] 与图 dims 比对（EnsureKvDesc 记账/
    //    EnsureKvTensor 建形共用。只认 HF 序逐位比对，不兼容旧约定序）──
    if (desc.dims.size() != 4) {
        // dims 数 ≠ 4：无法逐位比对 → 告警一次（池生命周期去重）后照常记账/直建
        static constexpr uint32_t kWarnDimsCount = 1u << 0;
        if (!(warned_.fetch_or(kWarnDimsCount, std::memory_order_relaxed) & kWarnDimsCount)) {
            CANN_LLM_LOGW("kv_pool.ensure: non-4D KV slot desc (layer=%d, %s, "
                          "ndims=%zu, dims=[%s]) — 跳过 HF 期望比对，按图 desc 处理（每池"
                          "只告本次）",
                          layer, is_value ? "V" : "K", desc.dims.size(), FormatDimsForLog(desc.dims).c_str());
        }
        return Status::SUCCESS;
    }
    const auto& expect = expect_dims_[static_cast<std::size_t>(p)];
    // kv_len（dims[2]）/ batch（dims[0]）失配 → 致命终止。
    // 判次序：先报 batch 还是 kv_len 只影响日志文案，两维任一失配即 fatal。
    if (desc.dims[2] != expect[2]) {
        CANN_LLM_LOGE("kv_pool.ensure: kv_len dim mismatch (layer=%d, %s) — "
                      "expect %d (meta kv_cache_max_len), graph dims[%d]；光标容量/"
                      "mask/Advance 越界全线以 meta 为准，图短于 meta 将致光标推进"
                      "超图缓冲——Load 终止",
                      layer, is_value ? "V" : "K", expect[2], static_cast<int>(desc.dims[2]));
        return Status::INVALID_ARGUMENT;
    }
    if (desc.dims[0] != expect[0]) {
        CANN_LLM_LOGE("kv_pool.ensure: batch dim mismatch (layer=%d, %s) — "
                      "expect %d (meta batch_num 槽位数), graph dims[%d]；图 batch < "
                      "meta 槽位数时 decode 满并发 KV 写入压错位/越 tensor——"
                      "Load 终止",
                      layer, is_value ? "V" : "K", expect[0], static_cast<int>(desc.dims[0]));
        return Status::INVALID_ARGUMENT;
    }
    // 其余维（heads dims[1]/末维 dims[3]）失配 → 告警，每类池内去重一次，
    // 不终止（om 图自洽，host 不直接寻址 KV tensor；om 为形状权威）
    uint32_t warn_bits = 0;
    if (desc.dims[1] != expect[1])
        warn_bits |= 1u << 1; // heads 维
    if (desc.dims[3] != expect[3])
        warn_bits |= 1u << 2; // 末维
    const uint32_t seen = warned_.fetch_or(warn_bits, std::memory_order_relaxed);
    if (warn_bits & ~seen) {
        CANN_LLM_LOGW("kv_pool.ensure: %s dim differs from meta expectation "
                      "(layer=%d, %s) — expect HF [batch=%d, heads=%d, kv_len=%d, "
                      "dim=%d], graph dims=[%s]；om 图为形状权威，继续（每池同类只"
                      "告本次）",
                      (warn_bits & (1u << 1)) ? ((warn_bits & (1u << 2)) ? "heads+end" : "heads") : "end", layer,
                      is_value ? "V" : "K", expect[0], expect[1], expect[2], expect[3],
                      FormatDimsForLog(desc.dims).c_str());
    }
    return Status::SUCCESS;
}

Status KvCachePool::EnsureKvDesc(int layer, bool is_value, const TensorDesc& desc)
{
    // 模板池 Load 期记账（ADR-0002）：只记 desc 账本，不建 tensor、不开显存。
    // 前半套共用校验（desc 合法性 + 层表防御 + R3 HF 期望比对）。
    int pos = -1;
    Status st = CheckKvDescCommon(layer, is_value, desc, &pos);
    if (!st.ok())
        return st;
    std::lock_guard<std::mutex> lk(mtx_);
    auto& slot = kv_descs_[static_cast<std::size_t>(2 * pos + (is_value ? 1 : 0))];
    // ── R2：已记账（首图 Load 已录）→ 复验 dims+dtype 严格一致（次图把关）──
    if (slot.has_value()) {
        if (slot->dims != desc.dims || slot->dataType != desc.dataType) {
            CANN_LLM_LOGE("kv_pool.ensure_desc: layer %d %s dims/dtype mismatch "
                          "(existing vs graph desc) — 双图同层 KV 布局不一致",
                          layer, is_value ? "V" : "K");
            return Status::INVALID_ARGUMENT;
        }
        return Status::SUCCESS; // 一致 → 次图复用
    }
    slot = desc; // 首次记账即录（dims+dtype 为契约面；tensorName 不具比对意义）
    CANN_LLM_LOGD("kv_pool.ensure_desc: layer %d %s desc recorded dims=[%s], dtype=%d "
                  "— 模板池记账（tensor 建点延后到 CreateContext）",
                  layer, is_value ? "V" : "K", FormatDimsForLog(desc.dims).c_str(), static_cast<int>(desc.dataType));
    return Status::SUCCESS;
}

std::optional<TensorDesc> KvCachePool::KvDesc(int layer, bool is_value) const
{
    const int pos = LayerPos(layer);
    if (pos < 0)
        return std::nullopt;
    std::lock_guard<std::mutex> lk(mtx_);
    const auto& slot = kv_descs_[static_cast<std::size_t>(2 * pos + (is_value ? 1 : 0))];
    return slot; // 未记账（host/stub 无 Load）→ nullopt
}

bool KvCachePool::DescLedgerComplete() const noexcept
{
    if (kv_descs_.size() != tensors_.size())
        return false;
    for (const auto& d : kv_descs_) {
        if (!d.has_value())
            return false;
    }
    return true;
}

Status KvCachePool::EnsureKvTensor(int layer, bool is_value, const TensorDesc& desc)
{
    // Context 池建 tensor 原语（ADR-0002 建点：CreateContext；原 Load 调用点
    // 已改 EnsureKvDesc 记账）。
    int pos = -1;
    Status st = CheckKvDescCommon(layer, is_value, desc, &pos);
    if (!st.ok())
        return st;
    std::lock_guard<std::mutex> lk(mtx_);
    auto& slot = tensors_[static_cast<std::size_t>(2 * pos + (is_value ? 1 : 0))];
    const char* kv_name = is_value ? "V" : "K";

    // ── R2：已建（含模板池经 BindFrom 转发指向的 tensor 防御口径）→ 复验
    //    dims+dtype 严格一致（次图把关）──
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

    // ── R1：未建 → 按 desc 直建（命名约定不变；desc 取自模板池 KvDesc 账本，
    //    om 图工位 desc 权威经账本传递）──
    TensorDesc d = desc; // dims/dtype/format 以图工位 desc 为权威
    d.tensorName = std::string(is_value ? "kv_cache_v_" : "kv_cache_k_") + std::to_string(layer);
    std::vector<std::shared_ptr<ITensorBuffer>> created;
    Status create_status = TensorManager::Instance().CreateKvCacheTensorWithDescs({d}, &created);
    if (!create_status.ok()) {
        CANN_LLM_LOGE("kv_pool.ensure_tensor: layer %d %s create failed (st=%d)", layer, kv_name, create_status.code);
        return create_status; // 该层不落位（未建可重试，all-or-nothing 天然保留）
    }
    // 创建即整只清零（仅新建路径；与 EnsureStateTensor 同口径）：
    // mask 纪律万一失效时残值=0 而非随机数；0 通道冷页预触对 ECC/带宽
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
    // 质量红线作废，容量硬约束由 GenerateAsync 提交期 max_io_tokens 预算闸
    // 承担（spec REQ-9：输入上界与 KV 缓冲容量解耦，单轮输入超 kv_max 走
    // 环形绕回）。
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
    // R13：锚定段为元数据级常量（构造期 supply，configured_init_len_ 基线），
    // miss 重建 = 恢复配置基线而非归零；配置 0 时与归零等价。WritePosValues
    // 等消费方经 MapSlot 只读本值，恢复即刻生效。
    init_token_len_ = configured_init_len_;
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
    // 口径修订（ADR-0002）：tensors_ 预置空槽，数非空项——构造后/未加载 = 0；
    // Context 池 = CreateContext 期 EnsureKvTensor 全层实建后 = 2×NumKvLayers()；
    // 模板池 = BindFrom 转发登记后 = 活动池实建数（host/stub 零 tensor = 0 合法）
    std::size_t n = 0;
    for (const auto& t : tensors_) {
        if (t)
            ++n;
    }
    return n;
}

Status KvCachePool::BindFrom(const KvCachePool& src)
{
    // 薄壳转发登记（Q10 乙；SetContext 内唯一登记点）。两池同参构造
    // （层表一致）由调用方保证；此处守 size + 同层表防御。
    std::lock_guard<std::mutex> lk(mtx_);
    if (this == &src)
        return Status::SUCCESS; // 自指 no-op（幂等防御）
    if (src.tensors_.size() != tensors_.size() || src.kv_layer_ids_ != kv_layer_ids_) {
        CANN_LLM_LOGE("kv_pool.bind_from: pool layout mismatch (self kv_layers=%zu, "
                      "src kv_layers=%zu) — 两池必须同参构造",
                      kv_layer_ids_.size(), src.kv_layer_ids_.size());
        return Status::INVALID_ARGUMENT;
    }
    // src 零 tensor（host/stub：无 Load → 无 desc 账本 → ContextCreate 零 tensor
    // 记账合法，Q12 裁定 a）→ 按"未指向"处理，仍 SUCCESS（SetContext 恒成功契约）
    if (src.TensorCount() == 0) {
        std::fill(tensors_.begin(), tensors_.end(), nullptr);
        CANN_LLM_LOGD("kv_pool.bind_from: src pool has 0 tensors (host/stub form) — "
                      "shell stays unbound");
        return Status::SUCCESS;
    }
    // 部分建（半建池不允许被激活——CreateContext all-or-nothing 已保证；此为
    // 二次防御，设计文档 §3.2 第 4 点口径）
    for (std::size_t i = 0; i < src.tensors_.size(); ++i) {
        if (!src.tensors_[i]) {
            CANN_LLM_LOGE("kv_pool.bind_from: src pool tensor slot %zu empty — "
                          "partially-built pool cannot be activated",
                          i);
            return Status::FAILURE;
        }
    }
    // 转发登记：模板池 tensors_ 各槽位指向活动池对应槽位（"指向"字面语义——
    // shared_ptr 共享引用，真显存属 Context 池，随其析构释放）
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        tensors_[i] = src.tensors_[i];
    }
    CANN_LLM_LOGI("kv_pool.bind_from: shell now forwards %zu tensors to source pool "
                  "(kv_layers=%d)",
                  tensors_.size(), NumKvLayers());
    return Status::SUCCESS;
}

void KvCachePool::ClearTensors() noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    // 清转发登记（DestroyContext 消费）：回全空"未指向"态；只清指针不动显存
    // （真显存在 Context 池，随其析构释放）。
    std::fill(tensors_.begin(), tensors_.end(), nullptr);
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

} // namespace cann_local_lm::engine
