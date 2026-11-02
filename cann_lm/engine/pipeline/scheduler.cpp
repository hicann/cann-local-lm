#include "pipeline/scheduler.h"

#include "core/component/text_embedding/text_embedding.h" // SR20260828001：FillInputs 接线
#include "infra/log.h"
#include "utils/perf_timer.h" // NowUs/UsToMs：推理耗时观测（perf.* 协议见其头注释）

#include <utility>

namespace cann_lm::engine {

Scheduler::Scheduler(ModelExecutor* prefill_executor, ModelExecutor* decode_executor, Preprocessor* preprocessor,
                     TextEmbedding* text_embedding, int max_batch_size)
    : prefill_executor_(prefill_executor),
      decode_executor_(decode_executor),
      preprocessor_(preprocessor),
      text_embedding_(text_embedding),
      max_batch_size_(max_batch_size > 0 ? max_batch_size : 1)
{
    decode_pool_.resize(max_batch_size_);
}

Scheduler::~Scheduler() = default;

void Scheduler::RollbackCache_(const std::shared_ptr<RequestImpl>& req)
{
    // R10 回退调度器侧统一入口：幂等（闭包首次生效后重灌天然幂等）；
    // 仅 Failed/Canceled 终态请求可进（REQ-8：Completed 记账保留供下轮
    // 复用——由各调用点判定，本入口不复查）；未注入（R3 无台账场景 /
    // 历史遗留直接构造 RequestImpl 的 UT）安全跳过。
    if (req && req->rollback_cache)
        req->rollback_cache();
}

void Scheduler::ReclaimAndFill_()
{
    auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;

    // 清理终态/canceled slot
    for (auto& slot : decode_pool_) {
        if (!slot.active)
            continue;
        if (!slot.req) {
            slot.active = false;
            continue;
        }
        auto s = slot.req->status.load(std::memory_order_acquire);
        if (s == RequestStatus::Completed || s == RequestStatus::Canceled || s == RequestStatus::Failed) {
            // R10 回退仅 Canceled/Failed（REQ-8）：回退 = 恢复提交前快照 =
            // "本轮当未发生"，只服务失败/取消请求的一致性；Completed 成功链
            // 台账/池记账必须保留，供同 Context 下轮比对复用（free the lane,
            // keep the books）。槽位清理（FreeKvSlot/重置）对三类终态照常。
            if (s != RequestStatus::Completed) {
                RollbackCache_(slot.req);
            }
            if (pool && slot.kv_slot >= 0)
                (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        } else if (slot.req->canceled.load(std::memory_order_acquire)) {
            slot.req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            RollbackCache_(slot.req); // R10：cancel 驱逐点 pipeline 不可见，回退随 req 携带
            if (pool && slot.kv_slot >= 0)
                (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
    }

    // 从 pending_decode 补位到空 slot
    for (auto& slot : decode_pool_) {
        if (slot.active)
            continue;
        if (pending_decode_.empty())
            break;
        auto entry = std::move(pending_decode_.front());
        pending_decode_.pop_front();
        // 终态请求不入 decode 池（首 token 终止竞速防御，ut_pipeline_budget
        // U4 定案）：末块出口被 pipeline 侧预算闸/stop_token
        // 终结（SetTerminal Completed）后，请求从 pending_decode_ 取出时已
        // 终态——照旧入池会把终态覆写回 Decoding（下方 status.store），
        // 已完成请求续跑 decode 多采样越界 token。此处直接跳过即可：
        // 槽位/记账已由 pipeline 终态点（ClearActiveGenerate_）+ 本函数
        // 首段终态清理分支按各自职责收尾，无 KV 工位泄漏（kv_slot 在
        // prefill 已绑且 Completed 允许复用账面）。
        if (entry.req->status.load(std::memory_order_acquire) != RequestStatus::Decoding) {
            continue;
        }
        slot.req = std::move(entry.req);
        slot.kv_slot = entry.kv_slot;
        slot.active = true;
        // decode 阶段起点观测（首次入池写一次；重新入池的场景（decode 池
        // 逐出后再入）保持原起点——统计解码整体墙钟，不按入池分段）
        if (slot.req && slot.req->decode_start_us == 0) {
            slot.req->decode_start_us = NowUs();
        }
        slot.req->status.store(RequestStatus::Decoding, std::memory_order_release);
    }
}

Status Scheduler::Submit(std::vector<std::vector<TokenId>> prefill_chunks, std::shared_ptr<RequestImpl> req,
                         RequestId* out)
{
    if (!req || !out)
        return Status::INVALID_ARGUMENT;

    RequestId id;
    id.value = next_id_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        PrefillTask task;
        task.id = id;
        task.req = req;
        task.chunks = std::move(prefill_chunks);
        // 工位直用：预绑定工位随 task 入队，Step 的 kv_slot<0 检查直通
        // （不再 AllocKvSlot 清零）。
        if (req->kv_slot_prefilled >= 0) {
            task.kv_slot = req->kv_slot_prefilled;
        }
        prefill_queue_.push_back(std::move(task));
    }

    req->status.store(RequestStatus::Queued, std::memory_order_release);
    cv_.notify_one();
    *out = id;
    return Status::SUCCESS;
}

Status Scheduler::Step(StepOutput* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;

    std::lock_guard<std::mutex> lk(mtx_);
    ReclaimAndFill_();

    // 检查是否有活任务
    bool has_prefill = !prefill_queue_.empty();
    bool has_decode = false;
    for (const auto& slot : decode_pool_) {
        if (slot.active) {
            has_decode = true;
            break;
        }
    }

    if (!has_prefill && !has_decode && pending_decode_.empty()) {
        return Status::INVALID_STATE; // 无活任务 → worker 休眠
    }

    // 检查 executor 是否就绪
    bool prefill_ready = (prefill_executor_ && static_cast<bool>(*prefill_executor_));
    bool decode_ready = (decode_executor_ && static_cast<bool>(*decode_executor_));

    // 优先 Prefill（有空位 + 队列非空 + executor 就绪）
    if (has_prefill && prefill_ready) {
        auto& task = prefill_queue_.front();

        // 空块守卫（DTS202610220945 Fix B）：空 chunks = 无任何可前向输入，
        // 无 logits 可产出，请求无法推进——空 prompt / 上游回归可达（Fix A 后
        // 命中路径恒 ≥1 真实块，本守卫为防御纵深）。镜像 not-ready 终态模式
        // （本函数尾部分支：Failed + R10 回退 + 出队 + FAILURE）。
        // 不调 FreeKvSlot：kv_slot == -1 = 尚未分配无事可释放；预绑定工位
        // （kv_slot_prefilled ≥ 0）归缓存绑定所有——Free 会置 active=false，
        // 回退闭包只恢复记账不恢复 active，留下被下次 AllocKvSlot 复用清账
        // 的脏窗口（本 PR 不引入该窗口）。
        if (task.chunks.empty()) {
            CANN_LLM_LOGE("pipeline.step: empty prefill chunks (req=%llu), "
                          "request marked failed",
                          static_cast<unsigned long long>(task.id.value));
            task.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
            RollbackCache_(task.req); // R10：回退随 req 携带（幂等）
            prefill_queue_.pop_front();
            return Status::FAILURE;
        }

        // 分配 KV 槽（首次处理该 task 时）。命中请求的预绑定工位
        // （kv_slot ≥ 0，Submit 期随台账登记工位预置）直通——AllocKvSlot
        // 的复用/驱逐路径会清零槽记账，把 GenerateAsync 里 ResumeSlot
        // 刚恢复到 LCP 的 total_written 抹掉（Bug：S2 增量块从绝对位 0
        // 覆写缓存前缀 KV）。
        if (task.kv_slot < 0) {
            auto pool = prefill_executor_->Pool();
            if (!pool || !pool->AllocKvSlot(&task.kv_slot).ok()) {
                CANN_LLM_LOGE("pipeline.step: kv slot alloc failed, request marked failed");
                task.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                // R10 回退载体：失败请求照填 out->slots（pipeline 终态出队
                // 统一回退台账/池记账），失败状态码上抛不变
                out->slots = {task.req};
                prefill_queue_.pop_front();
                return Status::FAILURE;
            }
            // Qwen3.5 线性状态 lane 清零（support_qwen3.5.md §4.7 / §5 D4）：
            // 新序列复用 batch lane 前，清掉该 lane 的 conv/r 状态残留——
            // KV 靠光标覆写天然免疫，状态是原地更新必须显式清理。无状态池
            // （非 Qwen3.5）= no-op；失败仅 WARN（清零失败不阻断调度）。
            if (auto sp = prefill_executor_->StatePool()) {
                (void)sp->ResetLane(task.kv_slot);
            }
        }

        // 捕获 req 副本（防止 pop 后 task 引用悬空）
        auto req = task.req;
        // 实际工位回写（预绑定直通 / Alloc 分配两路统一；worker 单线程
        // 无竞态）：pipeline 首 prefill 完成记账点消费——台账工位记录使
        // miss 请求下轮命中直用时也拿真实工位（多 batch 遗留的取值链）。
        req->kv_slot_actual = task.kv_slot;
        const auto& chunk = task.chunks[task.next_chunk];
        const size_t chunk_len = chunk.size(); // 切块保证非空
        // 首个块记 prefill 阶段起点（末块耗尽点消费）
        if (task.next_chunk == 0 && task.first_chunk_start_us == 0) {
            task.first_chunk_start_us = NowUs();
        }
        const std::int64_t chunk_start = NowUs();

        ForwardOptions opts;
        opts.batch = 1;
        opts.chunk_len = chunk_len;
        opts.slots = {task.kv_slot};
        opts.dump = false;

        std::shared_ptr<ITensorBuffer> logits;
        // SR20260828001 §2.2 接线点（prefill）：input_embed 工位改由 FillInputs 填充
        // （embedding/tokenId 两路径皆然，路径按 text_embedding 是否 Loaded 分派）；
        // SetInputData 保留为外部直填通道，Scheduler 不再调用。
        // B-1：FillInputs 失败 = 前向输入工位未被正确填充，带着未填充/残值工位
        // Forward 只会产出串扰或垃圾 logits。升级为请求终态 Failed 并出队，
        // 与同函数 kv alloc / debug inject / Forward 失败分支同构（decode 侧
        // 语义保留，见 §7.0 B-2 排除项）。
        //
        // golden 对齐（accuracy_requirements.md）：Python 标杆 build_prefill 是
        // 「ids 先补满图行宽（pad_token_id）再整批查表」——embedding 族工位
        // （embeddings/embeddings_scale/per_layer_token_identity/…_scale）的
        // padding 行 = pad_token_id(=0) 表行（golden dump 实测 padding 行互全同
        // 且 == 表第 0 行）。本引擎只填 chunk_len 有效行、尾部留残值会与 golden
        // 不一致。故：chunk 短于图行宽时复制补满 pad_token_id，count = 图行宽；
        // 整块（== 行宽）零复制（常态快路径不受影响）。全行确定性填充与局部
        // padding/续 arange 语义共存：有效行位置不变，mask padding 行全屏蔽
        // 不参与有效计算。pad_token_id 取自 model_config（pad 缺省 0 兜底，
        // meta_json 已带 pad_token_id=0）。
        if (text_embedding_ != nullptr && !chunk.empty()) {
            std::vector<TokenId> fill_ids;
            const TokenId* fill_src = chunk.data();
            std::size_t fill_count = chunk_len;
            ModelConfig mc;
            int graph_rows = 0;
            if (prefill_executor_->GetConfig(&mc).ok() && !mc.prefill_len.empty() && mc.prefill_len[0] > 0) {
                graph_rows = mc.prefill_len[0];
            }
            if (graph_rows > static_cast<int>(chunk_len)) {
                // 短尾块：复制有效 ids + pad_token_id 补满图行宽
                fill_ids.reserve(static_cast<std::size_t>(graph_rows));
                fill_ids.assign(chunk.begin(), chunk.end());
                const TokenId pad = static_cast<TokenId>(mc.pad_token_id);
                fill_ids.resize(static_cast<std::size_t>(graph_rows), pad);
                fill_src = fill_ids.data();
                fill_count = static_cast<std::size_t>(graph_rows);
            }
            Status fill_status = text_embedding_->FillInputs(
                prefill_executor_->InputTensors(), prefill_executor_->InputNameIndex(),
                text_embedding_->Loaded() ? EmbeddingPath::Embedding : EmbeddingPath::TokenId, fill_src, fill_count);
            if (!fill_status.ok()) {
                CANN_LLM_LOGE("pipeline.step: prefill FillInputs failed (st=%d), "
                              "request marked failed",
                              fill_status.code);
                req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                auto pool = prefill_executor_->Pool();
                if (pool && task.kv_slot >= 0)
                    (void)pool->FreeKvSlot(task.kv_slot);
                // R10 回退载体：失败请求照填 out->slots（同上）
                out->slots = {req};
                prefill_queue_.pop_front();
                return fill_status;
            }
        }
        auto st = prefill_executor_->Forward(opts);

        // Forward 失败：请求终态 Failed 并出队（否则 worker 无退避重试 → 挂死 + 日志洪水）
        if (!st.ok()) {
            CANN_LLM_LOGE("pipeline.step: prefill forward failed (st=%d), request marked failed", st.code);
            req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
            auto pool = prefill_executor_->Pool();
            if (pool && task.kv_slot >= 0)
                (void)pool->FreeKvSlot(task.kv_slot);
            // R10 回退载体：失败请求照填 out->slots（同上）
            out->slots = {req};
            prefill_queue_.pop_front();
            return st;
        }

        // 每轮 prefill 完成观测（本轮 forward 时长 = FillInputs + assembler
        // 填充 + Forward；req id 归因，块号/块长供多块序列对账）
        CANN_LLM_LOGD("perf.prefill.chunk: req=%llu chunk=%zu/%zu chunk_len=%zu elapsed=%.3f ms",
                      static_cast<unsigned long long>(task.id.value), task.next_chunk, task.chunks.size(), chunk_len,
                      UsToMs(NowUs() - chunk_start));

        task.next_chunk++;

        // chunk 耗尽 → task 出 prefill 队列，req 入 pending_decode（携带 kv_slot）
        if (task.next_chunk >= task.chunks.size()) {
            // prefill 整体完成观测（自首块起点累计，含批间调度等待；
            // 须在 pop_front 之前——pop 后 task 引用失效）。
            // 吞吐分子 = 本轮实际前向 tokens（全量 − 命中复用段，与切块
            // 区间 [reusable, len) 同口径；命中复用段不计入）
            const double prefill_ms = UsToMs(NowUs() - task.first_chunk_start_us);
            const std::size_t prefill_fed = req->prompt_ids.size() - req->reusable_len;
            CANN_LLM_LOGI("perf.prefill.total: req=%llu chunks=%zu input_tokens=%zu "
                          "reusable=%zu elapsed=%.3f ms speed=%.1f tokens/s",
                          static_cast<unsigned long long>(task.id.value), task.chunks.size(), req->prompt_ids.size(),
                          req->reusable_len, prefill_ms, prefill_ms > 0.0 ? prefill_fed * 1000.0 / prefill_ms : 0.0);
            SlotId kv = task.kv_slot;
            prefill_queue_.pop_front();
            req->status.store(RequestStatus::Decoding, std::memory_order_release);
            pending_decode_.push_back({req, kv});
            // 末块完成点旗（采样/记账唯一点）：中间块出口不进本分支，
            // 恒 false——pipeline RunLoop 依此跳过采样与台账记账。
            out->is_prefill_last = true;
        }

        out->logits = prefill_executor_->Logits(); // v0.9：logits 取自 output_tensors_
        out->slots = {req};                        // 使用捕获的副本，非悬空引用
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
            if (!slot.active)
                continue;

            // 取上一步采样的 token 作为 decode 输入
            TokenId last_token = 0;
            if (!slot.req->token_ids.empty()) {
                last_token = slot.req->token_ids.back();
            }

            // PrepareDecodeStep 在本调用点恒成功（唯一失败面是 out 空指针，
            // 此处恒传栈上有效对象——其内无其他失败路径）
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
        opts.batch = static_cast<int>(active_reqs.size());
        opts.chunk_len = 1;
        opts.slots = slots;
        opts.dump = false;
        const std::int64_t decode_step_start = NowUs();

        std::shared_ptr<ITensorBuffer> logits;
        // SR20260828001 §2.2 接线点（decode）：同 prefill 点，FillInputs 取代 SetInputData。
        // 批 lane 全填（P1 修复，Python 标杆 run_gemma4_gen.py build_decode 全 batch 维
        // 确定性输入）：input 工位是静态 [decode_batch,1,*]
        // （跨 Forward 复用），count=1 只填 lane 0——并发 ≥2 时其余 active lane 读
        // 残值 embedding 必错；inactive lane 复制 lane 0（首个 active 请求 token，
        // 与 assembler pos 族的 padded-lane 复制同口径），全图行确定性无
        // stale。图 batch 维 = KV 池槽数（decode_batch_size 同源）。
        // mask 工位不受此处影响：padding lane 保持全屏蔽（golden 口径）。
        if (text_embedding_ != nullptr && !batch_ids.empty()) {
            std::vector<TokenId> fill_ids = batch_ids;
            const int graph_batch = pool ? pool->max_batch_size() : static_cast<int>(fill_ids.size());
            if (graph_batch > static_cast<int>(fill_ids.size())) {
                fill_ids.resize(static_cast<std::size_t>(graph_batch), fill_ids.front());
            }
            Status fill_status = text_embedding_->FillInputs(
                decode_executor_->InputTensors(), decode_executor_->InputNameIndex(),
                text_embedding_->Loaded() ? EmbeddingPath::Embedding : EmbeddingPath::TokenId, fill_ids.data(),
                static_cast<std::size_t>(fill_ids.size()));
            if (!fill_status.ok()) {
                CANN_LLM_LOGW("pipeline.step: decode FillInputs failed (st=%d), "
                              "input_embed slot not filled",
                              fill_status.code);
            }
        }
        auto st = decode_executor_->Forward(opts);
        if (!st.ok()) {
            // Forward 失败：整批请求终态 Failed 并清池
            // （防 worker 无退避重试 → 挂死 + 日志洪水）
            CANN_LLM_LOGE("pipeline.step: decode forward failed (st=%d), batch requests marked failed", st.code);
            // R10 回退载体：整批失败请求照填 out->slots（pipeline 终态出队
            // 统一回退台账/池记账），失败状态码上抛不变
            out->slots = std::move(active_reqs);
            for (auto& slot : decode_pool_) {
                if (!slot.active)
                    continue;
                slot.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                if (pool && slot.kv_slot >= 0)
                    (void)pool->FreeKvSlot(slot.kv_slot);
                slot.active = false;
                slot.req.reset();
                slot.kv_slot = -1;
            }
            return st;
        }

        // 每轮 decode 完成观测（本批 forward 时长 = FillInputs + assembler 填充
        // + Forward；batch = 本批活跃请求数。RequestImpl 无请求编号面，首个
        // 请求以 context id 归因（批次逐轮同构，其余请求见各自终态日志）。
        CANN_LLM_LOGD("perf.decode.step: batch=%zu ctx0=%llu elapsed=%.3f ms", active_reqs.size(),
                      static_cast<unsigned long long>(active_reqs.front()->context_id.value),
                      UsToMs(NowUs() - decode_step_start));

        out->logits = decode_executor_->Logits(); // v0.9：logits 取自 output_tensors_
        out->slots = std::move(active_reqs);
        out->is_prefill = false;
        return Status::SUCCESS;
    }

    // executor 未就绪但有活任务 → 将 prefill 队首标记 Failed（无法推进）
    if (has_prefill && !prefill_ready) {
        CANN_LLM_LOGE("pipeline.step: prefill executor not ready (model load failed?), request marked failed");
        auto& task = prefill_queue_.front();
        task.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
        RollbackCache_(task.req); // R10：Scheduler 内部终态点，闭包通道回退
        prefill_queue_.pop_front();
        return Status::FAILURE;
    }

    if (has_decode && !decode_ready) {
        // decode 池有请求但 executor 未就绪 → 标记 Failed
        CANN_LLM_LOGE("pipeline.step: decode executor not ready (model load failed?), requests marked failed");
        auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;
        for (auto& slot : decode_pool_) {
            if (!slot.active)
                continue;
            slot.req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
            RollbackCache_(slot.req); // R10：同上
            if (pool && slot.kv_slot >= 0)
                (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
        return Status::FAILURE;
    }

    return Status::INVALID_STATE;
}

Status Scheduler::Cancel(RequestId id)
{
    std::lock_guard<std::mutex> lk(mtx_);
    auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;

    // 查 prefill 队列
    for (auto it = prefill_queue_.begin(); it != prefill_queue_.end(); ++it) {
        if (it->id == id) {
            it->req->canceled.store(true, std::memory_order_release);
            it->req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            RollbackCache_(it->req); // R10：驱逐点 pipeline 不可见 req，回退随 req 携带
            if (pool && it->kv_slot >= 0)
                (void)pool->FreeKvSlot(it->kv_slot);
            prefill_queue_.erase(it);
            return Status::SUCCESS;
        }
    }

    // 查 decode 池
    for (auto& slot : decode_pool_) {
        if (slot.active && slot.id == id) {
            slot.req->canceled.store(true, std::memory_order_release);
            slot.req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            RollbackCache_(slot.req);
            if (pool && slot.kv_slot >= 0)
                (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
            return Status::SUCCESS;
        }
    }

    // pending_decode 中的请求无 RequestId 索引；
    // 其 canceled 标志由 RequestHandle::Cancel() 直接设置，
    // 在入 decode 池后的 ReclaimAndFill_ 中生效。
    return Status::NOT_FOUND;
}

Status Scheduler::CancelContext(ContextId cid)
{
    std::lock_guard<std::mutex> lk(mtx_);
    auto pool = decode_executor_ ? decode_executor_->Pool() : nullptr;

    // prefill 队列
    for (auto it = prefill_queue_.begin(); it != prefill_queue_.end();) {
        if (it->req->context_id == cid) {
            it->req->canceled.store(true, std::memory_order_release);
            it->req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
            RollbackCache_(it->req);
            if (pool && it->kv_slot >= 0)
                (void)pool->FreeKvSlot(it->kv_slot);
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
            RollbackCache_(it->req);
            if (pool && it->kv_slot >= 0)
                (void)pool->FreeKvSlot(it->kv_slot);
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
            RollbackCache_(slot.req);
            if (pool && slot.kv_slot >= 0)
                (void)pool->FreeKvSlot(slot.kv_slot);
            slot.active = false;
            slot.req.reset();
            slot.kv_slot = -1;
        }
    }

    return Status::SUCCESS;
}

int Scheduler::ActiveDecodeCount() const noexcept
{
    int n = 0;
    for (const auto& slot : decode_pool_) {
        if (slot.active)
            ++n;
    }
    return n;
}

bool Scheduler::HasActiveTasks()
{
    std::lock_guard<std::mutex> lk(mtx_);
    // prefill 队列 / pending_decode / decode 池 active 任一非空 = 有活任务
    // （S4 在飞口径第二支——active_generates_ 标记清除与最终出队间的小窗
    // 兜底；SetContext 持 pipeline context_mtx_ 调用，与 Submit 同互斥面）。
    if (!prefill_queue_.empty() || !pending_decode_.empty())
        return true;
    for (const auto& slot : decode_pool_) {
        if (slot.active)
            return true;
    }
    return false;
}

int Scheduler::PendingPrefillCount() const noexcept { return static_cast<int>(prefill_queue_.size()); }

int Scheduler::DecodeCapacity() const noexcept { return max_batch_size_; }

} // namespace cann_lm::engine
