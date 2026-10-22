#include "pipeline/scheduler.h"

#include "core/component/text_embedding/text_embedding.h"   // SR20260828001：FillInputs 接线
#include "infra/log.h"

#include <utility>

namespace cann_lm::engine {

Scheduler::Scheduler(ModelExecutor* prefill_executor,
                     ModelExecutor* decode_executor,
                     Preprocessor* preprocessor,
                     TextEmbedding* text_embedding,
                     int max_batch_size)
    : prefill_executor_(prefill_executor),
      decode_executor_(decode_executor),
      preprocessor_(preprocessor),
      text_embedding_(text_embedding),
      max_batch_size_(max_batch_size > 0 ? max_batch_size : 1) {
    decode_pool_.resize(max_batch_size_);
}

Scheduler::~Scheduler() = default;

void Scheduler::ReclaimAndFill_() {
    auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;

    // 清理终态/canceled slot
    for (auto& slot : decode_pool_) {
        if (!slot.active) continue;
        if (!slot.req) { slot.active = false; continue; }
        auto s = slot.req->status.load(std::memory_order_acquire);
        if (s == RequestStatus::Completed ||
            s == RequestStatus::Canceled  ||
            s == RequestStatus::Failed) {
            if (pool && slot.kv_slot >= 0) (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        } else if (slot.req->canceled.load(std::memory_order_acquire)) {
            slot.req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            if (pool && slot.kv_slot >= 0) (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
    }

    // 从 pending_decode 补位到空 slot
    for (auto& slot : decode_pool_) {
        if (slot.active) continue;
        if (pending_decode_.empty()) break;
        auto entry = std::move(pending_decode_.front());
        pending_decode_.pop_front();
        slot.req     = std::move(entry.req);
        slot.kv_slot = entry.kv_slot;
        slot.active  = true;
        slot.req->status.store(RequestStatus::Decoding, std::memory_order_release);
    }
}

Status Scheduler::Submit(std::vector<std::vector<TokenId>> prefill_chunks,
                         std::shared_ptr<RequestImpl> req, RequestId* out) {
    if (!req || !out) return Status::INVALID_ARGUMENT;

    RequestId id;
    id.value = next_id_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        PrefillTask task;
        task.id     = id;
        task.req    = req;
        task.chunks = std::move(prefill_chunks);
        prefill_queue_.push_back(std::move(task));
    }

    req->status.store(RequestStatus::Queued, std::memory_order_release);
    cv_.notify_one();
    *out = id;
    return Status::SUCCESS;
}

Status Scheduler::Step(StepOutput* out) {
    if (!out) return Status::INVALID_ARGUMENT;

    std::lock_guard<std::mutex> lk(mtx_);
    ReclaimAndFill_();

    // 检查是否有活任务
    bool has_prefill = !prefill_queue_.empty();
    bool has_decode  = false;
    for (const auto& slot : decode_pool_) {
        if (slot.active) { has_decode = true; break; }
    }

    if (!has_prefill && !has_decode && pending_decode_.empty()) {
        return Status::INVALID_STATE;  // 无活任务 → worker 休眠
    }

    // 检查 executor 是否就绪
    bool prefill_ready = (prefill_executor_ && static_cast<bool>(*prefill_executor_));
    bool decode_ready   = (decode_executor_  && static_cast<bool>(*decode_executor_));

    // 优先 Prefill（有空位 + 队列非空 + executor 就绪）
    if (has_prefill && prefill_ready) {
        auto& task = prefill_queue_.front();

        // 分配 KV 槽（首次处理该 task 时）
        if (task.kv_slot < 0) {
            auto pool = prefill_executor_->Pool();
            if (!pool || !pool->AllocKvSlot(&task.kv_slot).ok()) {
                CANN_LLM_LOGE(
                    "pipeline.step: kv slot alloc failed, request marked failed");
                task.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                prefill_queue_.pop_front();
                return Status::FAILURE;
            }
            // Qwen3.5 线性状态 lane 清零（support_qwen3.5.md §4.7 / §5 D4）：
            // 新序列复用 batch lane 前，清掉该 lane 的 conv/r 状态残留——
            // KV 靠光标覆写天然免疫，状态是原地更新必须显式清理。无状态池
            //（非 Qwen3.5）= no-op；失败仅 WARN（清零失败不阻断调度）。
            if (auto sp = prefill_executor_->StatePool()) {
                (void)sp->ResetLane(task.kv_slot);
            }
        }

        // 捕获 req 副本（防止 pop 后 task 引用悬空）
        auto req = task.req;
        const auto& chunk = task.chunks[task.next_chunk];
        const size_t chunk_len = chunk.size();   // 切块保证非空

        ForwardOptions opts;
        opts.batch     = 1;
        opts.chunk_len = chunk_len;
        opts.slots     = {task.kv_slot};
        opts.dump      = false;

        std::shared_ptr<ITensorBuffer> logits;
        // SR20260828001 §2.2 接线点（prefill）：input_embed 工位改由 FillInputs 填充
        // （embedding/tokenId 两路径皆然，路径按 text_embedding 是否 Loaded 分派）；
        // SetInputData 保留为外部直填通道，Scheduler 不再调用。
        // B-1 修复（2026-09-23，failed_but_not_processed.md §7.1）：FillInputs 失败 =
        // 前向输入工位未被正确填充，带着未填充/残值工位 Forward 只会产出串扰或垃圾
        // logits。升级为请求终态 Failed 并出队，与同函数 kv alloc / debug inject /
        // Forward 失败分支同构（decode 侧 v0.9 语义保留，见 §7.0 B-2 排除项）。
        if (text_embedding_ != nullptr && !chunk.empty()) {
            Status fst = text_embedding_->FillInputs(
                prefill_executor_->InputTensors(),
                prefill_executor_->InputNameIndex(),
                text_embedding_->Loaded() ? EmbeddingPath::Embedding
                                          : EmbeddingPath::TokenId,
                chunk.data(), chunk_len);
            if (!fst.ok()) {
                CANN_LLM_LOGE(
                    "pipeline.step: prefill FillInputs failed (st=%d), "
                    "request marked failed", fst.code);
                req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                auto pool = prefill_executor_->Pool();
                if (pool && task.kv_slot >= 0) (void)pool->FreeKvSlot(task.kv_slot);
                prefill_queue_.pop_front();
                return fst;
            }
        }
        auto st = prefill_executor_->Forward(opts);

        // Forward 失败：请求终态 Failed 并出队（否则 worker 无退避重试 → 挂死 + 日志洪水）
        if (!st.ok()) {
            CANN_LLM_LOGE(
                "pipeline.step: prefill forward failed (st=%d), request marked failed", st.code);
            req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
            auto pool = prefill_executor_->Pool();
            if (pool && task.kv_slot >= 0) (void)pool->FreeKvSlot(task.kv_slot);
            prefill_queue_.pop_front();
            return st;
        }

        task.next_chunk++;

        // chunk 耗尽 → task 出 prefill 队列，req 入 pending_decode（携带 kv_slot）
        if (task.next_chunk >= task.chunks.size()) {
            SlotId kv = task.kv_slot;
            prefill_queue_.pop_front();
            req->status.store(RequestStatus::Decoding, std::memory_order_release);
            pending_decode_.push_back({req, kv});
        }

        out->logits     = prefill_executor_->Logits();   // v0.9：logits 取自 output_tensors_
        out->slots      = {req};  // 使用捕获的副本，非悬空引用
        out->is_prefill = true;
        return Status::SUCCESS;
    }

    // Decode batch（executor 就绪时）
    if (has_decode && decode_ready) {
        auto pool = decode_executor_->Pool();
        // 收集 active slot 的 last token（普通堆内存载体：token id 仅 host 侧
        // FillInputs 消费、不进 NPU，不占 ION——见 preprocessor_mem_use.md）
        std::vector<TokenId> batch_ids;
        std::vector<std::shared_ptr<RequestImpl>> active_reqs;
        std::vector<SlotId> slots;

        for (auto& slot : decode_pool_) {
            if (!slot.active) continue;

            // 取上一步采样的 token 作为 decode 输入
            TokenId last_token = 0;
            if (!slot.req->token_ids.empty()) {
                last_token = slot.req->token_ids.back();
            }

            // 分配不再存在 → 无失败模式（原 CreateTensor 失败分支已移除）
            std::vector<TokenId> input;
            (void)preprocessor_->PrepareDecodeStep(last_token, &input);
            batch_ids.insert(batch_ids.end(), input.begin(), input.end());
            active_reqs.push_back(slot.req);
            slots.push_back(slot.kv_slot);
        }

        if (active_reqs.empty()) {
            return Status::INVALID_STATE;
        }

        ForwardOptions opts;
        opts.batch     = static_cast<int>(active_reqs.size());
        opts.chunk_len = 1;
        opts.slots     = slots;
        opts.dump      = false;

        std::shared_ptr<ITensorBuffer> logits;
        // SR20260828001 §2.2 接线点（decode）：同 prefill 点，FillInputs 取代 SetInputData。
        // v0.9 语义保留：stub 下取首个请求输入（真实后端按 batch 维拼装时改传
        // batch_ids.data() + batch_ids.size()，须核对 decode input 工位 batch 维容量）。
        if (text_embedding_ != nullptr && !batch_ids.empty()) {
            Status fst = text_embedding_->FillInputs(
                decode_executor_->InputTensors(),
                decode_executor_->InputNameIndex(),
                text_embedding_->Loaded() ? EmbeddingPath::Embedding
                                          : EmbeddingPath::TokenId,
                batch_ids.data(), /*count=*/1);
            if (!fst.ok()) {
                CANN_LLM_LOGW(
                    "pipeline.step: decode FillInputs failed (st=%d), "
                    "input_embed slot not filled", fst.code);
            }
        }
        auto st = decode_executor_->Forward(opts);
        if (!st.ok()) {
            // Forward 失败：整批请求终态 Failed 并清池
            // （防 worker 无退避重试 → 挂死 + 日志洪水）
            CANN_LLM_LOGE(
                "pipeline.step: decode forward failed (st=%d), batch requests marked failed",
                st.code);
            for (auto& slot : decode_pool_) {
                if (!slot.active) continue;
                slot.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                if (pool && slot.kv_slot >= 0) (void)pool->FreeKvSlot(slot.kv_slot);
                slot.active = false;
                slot.req.reset();
                slot.kv_slot = -1;
            }
            return st;
        }

        out->logits     = decode_executor_->Logits();   // v0.9：logits 取自 output_tensors_
        out->slots      = std::move(active_reqs);
        out->is_prefill = false;
        return Status::SUCCESS;
    }

    // executor 未就绪但有活任务 → 将 prefill 队首标记 Failed（无法推进）
    if (has_prefill && !prefill_ready) {
        CANN_LLM_LOGE(
            "pipeline.step: prefill executor not ready (model load failed?), request marked failed");
        auto& task = prefill_queue_.front();
        task.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
        prefill_queue_.pop_front();
        return Status::FAILURE;
    }

    if (has_decode && !decode_ready) {
        // decode 池有请求但 executor 未就绪 → 标记 Failed
        CANN_LLM_LOGE(
            "pipeline.step: decode executor not ready (model load failed?), requests marked failed");
        for (auto& slot : decode_pool_) {
            if (!slot.active) continue;
            slot.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
        return Status::FAILURE;
    }

    return Status::INVALID_STATE;
}

Status Scheduler::Cancel(RequestId id) {
    std::lock_guard<std::mutex> lk(mtx_);

    // 查 prefill 队列
    for (auto it = prefill_queue_.begin(); it != prefill_queue_.end(); ++it) {
        if (it->id == id) {
            it->req->canceled.store(true, std::memory_order_release);
            it->req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            prefill_queue_.erase(it);
            return Status::SUCCESS;
        }
    }

    // 查 decode 池
    for (auto& slot : decode_pool_) {
        if (slot.active && slot.id == id) {
            slot.req->canceled.store(true, std::memory_order_release);
            slot.req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            slot.active = false;
            slot.req.reset();
            return Status::SUCCESS;
        }
    }

    // pending_decode 中的请求无 RequestId 索引；
    // 其 canceled 标志由 RequestHandle::Cancel() 直接设置，
    // 在入 decode 池后的 ReclaimAndFill_ 中生效。
    return Status::NOT_FOUND;
}

Status Scheduler::CancelContext(ContextId cid) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;

    // prefill 队列
    for (auto it = prefill_queue_.begin(); it != prefill_queue_.end();) {
        if (it->req->context_id == cid) {
            it->req->canceled.store(true, std::memory_order_release);
            it->req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            if (pool && it->kv_slot >= 0) (void)pool->FreeKvSlot(it->kv_slot);
            it = prefill_queue_.erase(it);
        } else {
            ++it;
        }
    }

    // pending_decode
    for (auto it = pending_decode_.begin(); it != pending_decode_.end();) {
        if (it->req->context_id == cid) {
            it->req->canceled.store(true, std::memory_order_release);
            it->req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            if (pool && it->kv_slot >= 0) (void)pool->FreeKvSlot(it->kv_slot);
            it = pending_decode_.erase(it);
        } else {
            ++it;
        }
    }

    // decode 池
    for (auto& slot : decode_pool_) {
        if (slot.active && slot.req && slot.req->context_id == cid) {
            slot.req->canceled.store(true, std::memory_order_release);
            slot.req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            if (pool && slot.kv_slot >= 0) (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
    }

    return Status::SUCCESS;
}

int Scheduler::ActiveDecodeCount() const noexcept {
    int n = 0;
    for (const auto& slot : decode_pool_) {
        if (slot.active) ++n;
    }
    return n;
}

int Scheduler::PendingPrefillCount() const noexcept {
    return static_cast<int>(prefill_queue_.size());
}

int Scheduler::DecodeCapacity() const noexcept {
    return max_batch_size_;
}

}  // namespace cann_lm::engine
