#pragma once

#include "core/context.h"
#include "core/component/model_executor/model_executor.h"
#include "pipeline/preprocessor.h"
#include "core/tensor_manager.h" // ITensorBuffer（v0.12：替代 core/tensor.h）
#include "api/cpp/request_handle.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace cann_local_lm::engine {

class TextEmbedding; // SR20260828001：构造注入的非拥有指针（定义见 text_embedding.h）

struct RequestId {
    uint64_t value = 0;
    bool operator==(const RequestId& other) const noexcept { return value == other.value; }
    explicit operator bool() const noexcept { return value != 0; }
};

/// 单请求共享状态（上半 RequestHandle 持 shared_ptr 指向此，见上半 §5）。
/// Scheduler / 后处理写，用户线程经 RequestHandle 读（线程安全见 pipeline_executor.h）。
struct RequestImpl {
    std::atomic<RequestStatus> status{RequestStatus::Queued};
    std::atomic<bool> canceled{false};
    SamplingParams sampling_params; // 纯采样旋钮（sample 模块消费）
    ContextId context_id;
    StreamCallback callback; // 可空 = 非流式

    // ── 请求级终止参数（自 RequestOptions 拷入；另有 meta eos_token_id 停止
    //    在引擎侧 RunLoop 直判，不落请求字段）──
    /// 生成上限。≤0 = 默认 1024（RunLoop 判定处取默认）。
    int max_tokens = 0;
    /// id 终止集：生成 token 命中 → StopToken 终止。
    std::vector<TokenId> stop_token_ids;
    /// 字面量终止集：文本尾部命中 → StopString 终止（截去停止子串）。
    /// 依赖 detokenize 不剥离 special token（skip_special_tokens 已移出推理链路）。
    std::vector<std::string> stop_strings;
    /// REQ-9 单请求总 token 预算（输入+生成共享；GenerateAsync 提交期拷入
    /// meta 的 max_io_tokens）。每 token 采样追加后复验：prompt_len +
    /// token_ids.size() >= io_budget ⇒ MaxTokens 终止（生成总量 =
    /// max_io_tokens − prompt_len）。与请求级 max_tokens 两闸正交、先到先停。
    /// 默认 SIZE_MAX = 无预算（生产单生产者前提：GenerateAsync 恒拷入 > 0；
    /// 测试直构 RequestImpl 绕过拷入路径时不再首 token 即静默截断）。
    std::size_t io_budget = SIZE_MAX;

    // ── PromptCache 请求级记账（design §4.4；worker 单线程写，无并发面）──
    /// R10 回退快照：提交时台账副本（请求 Failed/Cancel 终态回退重灌）。
    std::vector<TokenId> history_snapshot;
    /// 提交时池 total_written 快照（与 history_snapshot 同点拷贝；终态
    /// RollbackTo 回退目标）。
    std::size_t pool_total_written_snapshot = 0;
    /// 本轮全量输入 ids（REQ-2.1：首个 chunk prefill 完成点一次性追加台账；
    /// R10 回退亦用其首 reusable 位对齐——快照语义已含）。
    std::vector<TokenId> prompt_ids;
    /// 复用长（MatchCachedTokens 出参；REQ-2.1 台账追加起点 == 提交时台账长,
    /// prefill 完成点从 reusable 追加到全量长）。
    std::size_t reusable_len = 0;
    /// 预绑定 KV 工位：≥0 = 命中请求的台账登记工位（GenerateAsync 按台账
    /// KvSlotId 预置，scheduler 直用不再 AllocKvSlot——Alloc 的复用/驱逐
    /// 路径清零 total_written 会把 ResumeSlot 刚恢复的记账抹掉）。
    /// -1 = miss/普通请求，走既有 AllocKvSlot 分配路径。
    int kv_slot_prefilled = -1;
    /// 实际占用工位（Step 期回写，命中预绑定/Alloc 分配两路皆覆盖；worker
    /// 单线程无竞态）：pipeline 首个 prefill 完成记账点读此值回写台账
    /// SetKvSlotId——使工位记录对 miss 请求也真实（而非恒 0 假设）。
    int kv_slot_actual = -1;
    /// 台账记账已落位标记（首个 prefill chunk 完成点一次性；无 PromptCache
    /// 台账的场景（R3）恒 false，防重入）。
    bool history_prefill_applied = false;
    /// R10 终态回退闭包（pipeline GenerateAsync 期注入：台账 Restore +
    /// 池 RollbackTo；仅 Canceled/Failed 终态可调——Completed 不进回退，
    /// 成功链记账保留供下轮复用，调用点负责判定）。Scheduler 内部终态化点
    /// （Cancel 驱逐 / 就绪性检查 Fail / ReclaimAndFill_ 清理）调用——
    /// 这些点 pipeline 不再可见请求，回退只能随 req 携带。幂等（多次调用
    /// 仅首次生效——回退本质 = 恢复提交前快照，天然幂等）。
    std::function<void()> rollback_cache;

    // ── 推理耗时观测（worker 单线程写，无并发面）──
    /// decode 阶段起点（steady_clock us；ReclaimAndFill_ 首次入 decode 池
    /// 时写入，已在池/统计跳过重写）。0 = 尚未进入 decode 阶段（失败/
    /// 取消路径无观测意义，不补写）。Completed 终态点（RunLoop）消费。
    std::int64_t decode_start_us = 0;

    // 输出（受 mutex 保护）
    std::vector<TokenId> token_ids;
    std::string text;
    FinishReason finish_reason{FinishReason::MaxTokens};

    // Wait/TryGetResult 同步原语
    std::mutex mtx;
    std::condition_variable cv;

    /// worker 线程调：设终态并唤醒 Wait
    void SetTerminal(RequestStatus s, FinishReason fr)
    {
        finish_reason = fr;
        {
            std::lock_guard<std::mutex> lk(mtx);
            status.store(s, std::memory_order_release);
        }
        cv.notify_all();
    }
};

/// 单次 iteration 的前向产出（Scheduler 填，pipeline 后处理消费）。
struct StepOutput {
    std::shared_ptr<ITensorBuffer> logits;           // batched logits（N slot）
    std::vector<std::shared_ptr<RequestImpl>> slots; // slot i → 请求
    bool is_prefill = false;
    // prefill 末块（或单块）完成点：chunk 耗尽出队分支置位。诉求：中间块
    // 出口 logits 是对本块末位 prompt 真值的预测（下一块输入带真值），采样/
    // 记账/token_ids/text 均不适用——pipeline 以此旗短路（对齐参考实现
    // "块循环层纯 Forward"分界，详见 tmp_docs/fix_prefill_sample_cache_update_
    // 202610222233.md §3 S3）。失败/未就绪出口恒 false。
    bool is_prefill_last = false;
};

/// 核心组件（需求 #3）：FIFO 任务队列、decode 池（容量 = max_batch_size）、
/// iteration 级组批决策、交错策略（有空位先处理 Prefill）。
/// 不自驱跨阶段，由 pipeline 的 worker 线程循环调 Step 驱动（见 pipeline_executor.h）。
class Scheduler {
public:
    /// text_embedding：非拥有指针（与 preprocessor 同型）。SR20260828001 §2.2 接线点——
    /// prefill/decode 两输入准备点经其 FillInputs 填 input_embed 工位（embedding/tokenId
    /// 两路径皆然），取代旧 SetInputData 调用。可为 nullptr（无 embedding 组件的场景）。
    Scheduler(ModelExecutor* prefill_executor, ModelExecutor* decode_executor, Preprocessor* preprocessor,
              TextEmbedding* text_embedding, int max_batch_size);
    ~Scheduler();

    /// 提交请求：已预处理的 prefill chunks + 共享状态 RequestImpl。返回内部 RequestId。
    /// req->kv_slot_prefilled ≥ 0 时（PromptCache 命中请求）task 直接预绑定该
    /// 工位，Step 跳过 AllocKvSlot（工位直用，防分配路径清零恢复记账）。
    Status Submit(std::vector<std::vector<TokenId>> prefill_chunks, std::shared_ptr<RequestImpl> req, RequestId* out);

    /// 单次迭代驱动（worker 线程循环调）：
    ///   决策本 iteration 跑 Prefill（有空位优先）还是 Decode batch
    ///   → 准备输入（prefill：从队列取下个 chunk；decode：Preprocessor.PrepareDecodeStep 各 active 请求 last token）
    ///   → Executor.Forward → 填 StepOutput（logits + slot→req 映射）
    ///   无活任务返回 Status::INVALID_STATE（worker 可休眠，见 pipeline_executor.h）
    Status Step(StepOutput* out);

    /// 取消单请求（标记 canceled，当前 iteration 后退出池；部分 token 丢弃见需求 #3）
    Status Cancel(RequestId);
    /// 取消某 Context 所有在飞请求
    Status CancelContext(ContextId);

    /// Scheduler 尚有活任务判断（ADR-0002 S4 在飞口径第二支）：prefill 队列
    /// 非空 或 decode 池有 active 或 pending_decode 非空。请求一经完全出队
    /// （终态化）即不在册。SetContext 消费——检查与切换在 pipeline
    /// context_mtx_ 临界区内，取 scheduler_mtx_ 无嵌套问题。
    bool HasActiveTasks();

    // 状态查询（可观测性）
    int ActiveDecodeCount() const noexcept;   // decode 池 active 数
    int PendingPrefillCount() const noexcept; // 待 prefill chunk 数
    int DecodeCapacity() const noexcept;      // = max_batch_size

private:
    ModelExecutor* prefill_executor_;
    ModelExecutor* decode_executor_;
    Preprocessor* preprocessor_;
    TextEmbedding* text_embedding_ = nullptr; // 非拥有（pipeline 传入）
    int max_batch_size_;

    // ── 调度状态（mutex 保护）──
    struct PrefillTask {
        RequestId id;
        std::shared_ptr<RequestImpl> req;
        std::vector<std::vector<TokenId>> chunks;
        size_t next_chunk = 0; // 下一个待 prefill 的 chunk 下标
        SlotId kv_slot = -1;   // prefill 开始时从 KV 池分配
        // 首个 chunk Forward 起点观测（steady_clock us；0 = 尚未起步）。
        // 末块耗尽点消费后输出 prefill 整体耗时（含中间被 decode 批穿插的
        // 等待时间——单请求串行场景即纯 prefill 时长）。
        std::int64_t first_chunk_start_us = 0;
    };

    struct DecodeSlot {
        RequestId id{};
        std::shared_ptr<RequestImpl> req;
        TokenId last_token = 0;
        SlotId kv_slot = -1; // KV 池槽（从 prefill 继承）
        bool active = false;
    };

    /// prefill 完成后待入 decode 池的请求（携带 KV 槽）
    struct PendingEntry {
        std::shared_ptr<RequestImpl> req;
        SlotId kv_slot = -1;
    };

    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<uint64_t> next_id_{1};
    std::deque<PrefillTask> prefill_queue_;   // FIFO
    std::vector<DecodeSlot> decode_pool_;     // 容量 = max_batch_size
    std::deque<PendingEntry> pending_decode_; // prefill 完成待入 decode 池

    /// 内部：清理终态/canceled slot，补位 pending_decode → 空位
    void ReclaimAndFill_();
    /// 内部：R10 终态回退（req 携带闭包；幂等——Completed 场景 pipeline
    /// 注入 no-op，未注入（历史遗留直接构造 RequestImpl 的 UT）安全跳过）
    static void RollbackCache_(const std::shared_ptr<RequestImpl>& req);

    friend class AutoregressivePipelineExecutor; // pipeline 需唤醒 cv_ + 访问 decode_pool_
};

} // namespace cann_local_lm::engine
