#pragma once

#include "core/context.h"
#include "api/cpp/engine_executor.h"   // EngineOptions / ContextOptions
#include "core/component/tokenizer/tokenizer.h"
#include "core/component/text_embedding/text_embedding.h"
#include "core/component/sampler/sampler.h"
#include "core/component/model_executor/model_executor.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/linear_state_pool/linear_state_pool.h"   // Qwen3.5 状态池
#include "pipeline/preprocessor.h"
#include "pipeline/scheduler.h"
#include "pipeline/history_prompt_manager.h"
#include "api/cpp/request_handle.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace cann_lm::engine {

// ── Pipeline 类型枚举（工厂选择派生 pipeline）──────────────────────────────
// 当前唯一：Autoregressive（Prefill/Decode 两阶段 + Continuous Batching）；
// 未来其他 pipeline（非自回归等）经 PipelineExecutorBase 派生 + 工厂生成。
enum class PipelineType { Autoregressive /*, ...future */ };

/// L2 编排骨架的抽象基类（C6 PIMPL impl 的多态根，约束 #3）。
/// 转发上半公开 API impl；EngineExecutor::Impl 经 PipelineExecutorFactory 持有
/// unique_ptr<PipelineExecutorBase>。各派生 pipeline 实现这组虚函数。
class PipelineExecutorBase {
public:
    virtual ~PipelineExecutorBase() = default;

    virtual Status CreateContext(Context* out) = 0;
    virtual Status DestroyContext(ContextId) = 0;
    virtual Status ConfigureContext(ContextId, const ContextOptions&) = 0;
    virtual Status CancelContext(ContextId) = 0;
    virtual Status Generate(const Prompt&, const Context&, const SamplingParams&,
                            const RequestOptions&, Result* out) = 0;
    virtual Status GenerateAsync(const Prompt&, const Context&, const SamplingParams&,
                                 const RequestOptions&, RequestHandle* out,
                                 StreamCallback callback) = 0;
};

// ── 工厂（按 PipelineType 选择派生构造，约束 #3）────────────────────────────
class PipelineExecutorFactory {
public:
    // 错误: Status::INVALID_ARGUMENT（未知类型 / out 为空）
    //     | 执行器 Load 失败码（配置了模型图位而加载失败——构造内已终止初始化
    //       并完成卸载，失败实例不写入 *out；2026-09-22）
    // 注：组件由 pipeline 构造内部创建（D-08：不再自 L1 注入）；
    //     meta_json 解析为构造内内联步骤（D-10），失败降级吸收、不影响本返回值。
    static Status create(PipelineType type,
                         const ModelArtifact& artifact,
                         const EngineOptions& options,
                         std::unique_ptr<PipelineExecutorBase>* out);
};

/// 自回归 pipeline 派生（当前唯一实现）：Prefill/Decode 两阶段分离 + Continuous Batching。
/// 持有全部共享组件与 L2 模块；拥有 worker 线程循环驱动 Scheduler::Step + 后处理（M-07）；
/// 拥有 ContextImpl/RequestImpl/HistoryPromptManager 注册表。
/// SR20260903001：单一 ModelArtifact（prefill/decode 双图位 + prefill/decode
/// 双权重位——权重路径拆分见 docs/seperate_pd_weights.md §4.1）构造；meta_json
/// 三段经构造内内联解析步骤（D-10）直填三模块配置结构体；双执行器各 Load 各自
/// 图位与各自权重路径（D-03）。
class AutoregressivePipelineExecutor final : public PipelineExecutorBase {
public:
    /// 构造内完成：解析步骤（含降级 WARN）→ 组件创建 → KV 池定容 →
    /// 双执行器 Load（先 prefill 后 decode，2026-09-23 换序——prefill 图为形状权威，
    /// 与 token_identity「prefill 权威」约定统一）→ InitEmbedding_ → worker 启动。
    /// 加载失败即终止（2026-09-22）：模型图位已配置（omc path/buffer 任一非空）时，
    /// 任一执行器 Load 失败（含权重 InitWeights，如 NPU 配额耗尽）→ 终止初始化
    /// （跳过 embedding 加载与 worker 启动）→ 卸载双执行器（已加载图/权重即刻
    /// 释放回配额）→ 记入 InitStatus() 由工厂上抛（ExecutorCreate 报错）。
    /// 双图位全空 = 无模型部署形态（stub-era）：跳过加载，生命周期 API 照常，
    /// Generate 报 not-ready（原行为）。
    AutoregressivePipelineExecutor(const ModelArtifact& artifact,
                                   const EngineOptions& options);
    ~AutoregressivePipelineExecutor() override;

    /// 构造期初始化结果（工厂消费）。SUCCESS = 初始化完整完成；
    /// 非 SUCCESS = 已终止初始化并完成卸载，实例不应被交出使用。
    Status InitStatus() const noexcept { return init_status_; }

    // —— 上半 API 的 impl ——
    Status CreateContext(Context* out) override;
    Status DestroyContext(ContextId) override;
    Status ConfigureContext(ContextId, const ContextOptions&) override;
    Status CancelContext(ContextId) override;
    Status Generate(const Prompt&, const Context&, const SamplingParams&,
                    const RequestOptions&, Result* out) override;
    Status GenerateAsync(const Prompt&, const Context&, const SamplingParams&,
                         const RequestOptions&, RequestHandle* out,
                         StreamCallback callback) override;

    // ── 配置解析观测点（SR20260903001 M-08 §8：TC_001–003/010/203 等 UT 载体。
    //     只读、构造后稳定；反映内联解析步骤产出，与模型加载成败无关——host 无模型
    //     场景下 GetConfig（executor 侧）不可用，故经 parsed_ 直观测）──
    const ModelConfig&     ParsedModelConfig() const;      ///< TC_001/003/204/206
    const TokenizerConfig& ParsedTokenizerConfig() const;  ///< TC_001/004/010/207
    const EmbeddingConfig& ParsedEmbeddingConfig() const;  ///< TC_001/005/102/209/210
    int                    ParsedPrefillChunkSize() const; ///< TC_002/203（优先级链终值）
    int                    ParsedDecodeBatchSize() const;  ///< batch 来源（batch_num > override > 4）
    DataType               ParsedKvCacheDtype() const;     ///< kv_cache_type 解析终值（缺省/未知 FLOAT16）
    const KvCachePool*     ParsedKvPool() const;           ///< KV 池只读观测（形状接线 UT 载体）
    /// Qwen3.5 线性状态池只读观测（support_qwen3.5.md §4.6；UT 载体。
    /// 非 Qwen3.5 / 无 linear 层 → nullptr（未创建，零行为变化））
    const LinearStatePool* ParsedStatePool() const;
    bool                   TokenizerReady() const;         ///< TC_010/104/107（词表可用性）

private:
    // ① 内联解析步骤产物（D-10：.cpp 内定义的私有嵌套结构；构造期解析、构造后只读）
    struct ParsedConfigs;
    std::unique_ptr<ParsedConfigs> parsed_;
    // 构造期初始化结果（默认 SUCCESS；执行器 Load 失败时置错误码，
    // 见 InitStatus()——工厂据此拒交实例并上抛）
    Status init_status_;
    // ② engine 内部多态模块（D-08：构造内创建；词表路径 = options.tokenizer_path
    //    优先，回退 meta_json tokenizer_config.tokenizer_file；算法 = options.tokenizer_type）
    std::unique_ptr<TokenizerBase>  tokenizer_;
    // ③ engine 内部 embedding 模块（按值持有，InitEmbedding_ 经 embedding_config 段 Load）
    TextEmbedding                   text_embedding_;
    // ④ engine 内部具体类（按值持有，无多态需求）
    Sampler                         sampler_;

    // L2 模块（D-03：prefill/decode 双执行器各 Load 各自图位；
    //          Load 时自调 CreateModelManager() 自持 MM）
    ModelExecutor    prefill_executor_;   // PrefillExecutor 实例（构造期空依赖，Load 后就绪）
    ModelExecutor    decode_executor_;    // DecodeExecutor 实例
    std::shared_ptr<KvCachePool> kv_pool_;  // 共享 KV 池（按 kv_cache_max_len/num_hidden_layers 定容）
    // Qwen3.5 线性状态池（support_qwen3.5.md §4.6；可空 = 非 Qwen3.5 / 无
    // linear 层——池不创建、全链路 no-op，零行为变化）。tensor 惰性创建：
    // 构造期仅层清单，prefill 图先 Load 时按图 desc 创建（形状权威，§5 D1），
    // decode 图 Load 校验复用；lane 复用清零经 Scheduler（ResetLane，§5 D4）。
    std::shared_ptr<LinearStatePool> state_pool_;
    Preprocessor     preprocessor_;
    Scheduler        scheduler_;

    // Context / History 注册表 + 并发检查（受 context_mtx_）
    std::mutex                                  context_mtx_;
    std::map<ContextId, Context>                contexts_;
    std::map<ContextId, std::shared_ptr<HistoryPromptManager>> histories_;
    std::set<ContextId>                         active_generates_;   // 同 Context 并发 Generate 检查
    std::atomic<uint64_t>                       next_context_id_{1};

    // worker 线程：循环 Step + 后处理，无活任务时休眠
    void RunLoop();
    /// SR20260903001：embedding_config 段直供（U-8 通道取代，D-12）——主表路径非空
    /// （embedding 路径）才 Load；失败不阻断 init（保持未加载态，tokenId 路径仍可用）。
    /// gemma4 追加（docs/gemma4_token_identity.md §5.2；2026-09-22 变更）：identity
    /// host 查表只服务 Embedding 部署形态——主表为空或 Load 失败（降级 TokenId）时
    /// identity 由图内自理，本流整体跳过（零行为变化）；Embedding 形态下路径空自然
    /// 短路；行宽自执行器工位 dims 推导 + 配置/图互通校验。
    /// B-10 修复（2026-09-23，failed_but_not_processed.md §7.2）：以图为准——图声明
    /// 了 identity/sacle 工位而表/配置缺失、prefill/decode 行宽不一致、加载失败 ⇒
    /// 构造致命（返回错误码，构造函数卸载双执行器并记入 InitStatus），不再 WARN 后
    /// 带未填充工位前向。主表降级（D-11）与 TokenId/stub 形态行为不变。
    Status InitEmbedding_(const EmbeddingConfig& config);
    void ClearActiveGenerate_(ContextId cid);  // 请求终态后清除并发标记
    std::thread              worker_;
    std::atomic<bool>        stop_{false};
    std::condition_variable  worker_cv_;           // worker 唤醒/休眠
    std::mutex               worker_mtx_;          // 配合 worker_cv_
};

}  // namespace cann_lm::engine
