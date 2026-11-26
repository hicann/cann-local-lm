# LLM Engine 架构设计 - 设计规格说明书

简体中文 | [English](../../en/architecture_all/SPEC.md)

## 1. 概述

本章说明背景、设计决策与设计目标（1.1–1.3）。

### 1.1 背景

### 1.2 设计决策

| 决策项         | 结论                                        |
| -------------- | ------------------------------------------- |
| 重构策略       | 完全重写                                    |
| 调用方         | C/C++应用层                                 |
| 平台           | Kirin优先+可移植                            |
| 模型范围       | 纯文本LLM优先，可扩展多模态                 |
| 核心特性（v1） | KV Cache优化                                |
| 旧API兼容      | 不需要                                      |
| 状态管理       | Session分层（默认单Session，多Session可选） |
| 异步模型       | 异步优先，同步=wait包装                     |
| API语言        | C++ API优先 + C绑定可选                     |
| 参数风格       | 结构体分组 + Optional扩展                   |
| 架构方案       | 分层解耦架构                                |

### 1.3 设计目标

- 将所有平台依赖隔离在Platform HAL中
- 用显式状态机替代散落的 `atomic<bool>` 标志位
- 支持扩展（投机解码、多模态）而无需修改已有代码

---

## 2. 架构总览

本章描述分层结构、依赖规则、目录结构与代码量估算（2.1–2.4）。

### 2.1 分层结构

```
+----------------------------------------------------------+
|  API层（API Layer）                                       |
|  Engine, Session, GenerateParams, GenerateResult         |
|  C++ API为主要接口，C绑定为薄映射层                        |
+----------------------------------------------------------+
|  编排层（Orchestration Layer）                             |
|  Scheduler, PrefillController, DecodeController          |
|  请求调度、异步队列、prefill/decode流程编排                |
+----------------------------------------------------------+
|  组件层（Component Layer）                                 |
|  共享组件（Engine级）：ModelRunner, Tokenizer, Sampler,   |
|                        Embedding                          |
|  会话组件（Session级）：KVCacheManager, TensorManager,    |
|                        PromptManager                      |
+----------------------------------------------------------+
|  平台HAL（Platform Hardware Abstraction Layer）            |
|  IModelLoader, ITensorAllocator, IExecutor, IPerfCtrl    |
|  Kirin实现: hiai::IModelManager, INDTensorBuffer, FFRT   |
|  测试实现: MockModelLoader, MockTensorAllocator           |
+----------------------------------------------------------+
```

组件分为两级：
- **Engine 级（共享）**：ModelRunner、Tokenizer、Embedding、Sampler — 无状态或状态可复用，多 Session 共享
- **Session 级（独立）**：KVCacheManager、TensorManager、PromptManager — 各 Session 有独立的 KV Cache 和 tensor 池

Session 级组件通过 SessionContext 结构体打包，编排层的 Controller 在 Run 时接收 SessionContext 引用：

```cpp
namespace llm {

// Session 级组件的打包结构，由 Session 持有
struct SessionContext {
    KVCacheManager& kvCacheMgr;
    TensorManager& tensorMgr;
    PromptManager& promptMgr;
};

}  // namespace llm
```

### 2.2 依赖规则

- **单向依赖**：上层依赖下层，下层不可依赖上层。需要反向通知时通过回调/观察者模式。
- **组件间不直接依赖**：组件彼此不直接调用，由编排层协调它们协作。
- **Platform HAL是唯一的平台耦合点**：所有hiai::*调用封装在这一层。

### 2.3 目录结构

```
llm_engine/
+-- api/                            # API层
|   +-- engine.h/.cpp               # Engine类
|   +-- session.h/.cpp              # Session类
|   +-- generate_params.h           # GenerateParams及子结构体
|   +-- generate_result.h           # GenerateResult, ProfilingData
|   +-- callbacks.h                 # GenerateCallbacks, 事件结构体
|   +-- config.h                    # EngineConfig, SessionConfig等
|   +-- c_binding/                  # C绑定层（可选）
|       +-- llm_engine_c.h/.cpp
|
+-- orchestration/                  # 编排层
|   +-- scheduler.h/.cpp            # Scheduler
|   +-- infer_request.h/.cpp        # InferRequest, 状态机
|   +-- prefill_controller.h/.cpp   # PrefillController
|   +-- decode_controller.h/.cpp    # DecodeController, DecodeSlot
|   +-- schedule_strategy.h         # IScheduleStrategy接口
|
+-- component/                      # 组件层
|   +-- model_runner.h/.cpp         # ModelRunner
|   +-- tensor_manager.h/.cpp       # TensorManager（通用输入输出tensor管家）
|   +-- kvcache_manager.h/.cpp      # KVCacheManager
|   +-- tokenizer.h/.cpp            # Tokenizer基类+工厂
|   +-- sampler.h/.cpp              # Sampler
|   +-- embedding.h/.cpp            # Embedding
|   +-- prompt_manager.h/.cpp       # PromptManager
|   +-- tokenizer/                  # 各分词器实现
|       +-- bpe_tokenizer.h/.cpp
|       +-- spm_tokenizer.h/.cpp
|       +-- qwen_tokenizer.h/.cpp
|       +-- glm_tokenizer.h/.cpp
|
+-- platform/                       # 平台HAL
|   +-- hal_types.h                 # Tensor, TensorDesc, DataType等
|   +-- i_model_loader.h            # 纯虚接口
|   +-- i_tensor_allocator.h
|   +-- i_executor.h
|   +-- i_perf_controller.h
|   +-- platform_factory.h/.cpp     # 工厂
|   +-- kirin/                      # Kirin平台实现
|       +-- kirin_model_loader.h/.cpp
|       +-- kirin_tensor_allocator.h/.cpp
|       +-- kirin_executor.h/.cpp
|       +-- kirin_perf_controller.h/.cpp
|
+-- common/                         # 跨层公共定义
|   +-- status.h                    # Status枚举
|   +-- types.h                     # RequestId等类型别名
|   +-- log.h                       # 日志宏
|
+-- CMakeLists.txt
+-- BUILD.gn
```

### 2.4 代码量估算

| 模块 | 头文件 | 源文件 | 估算行数 |
|------|--------|--------|----------|
| api/ | 7 | 6 | ~1,500 |
| orchestration/ | 5 | 5 | ~2,000 |
| component/ | 14 | 14 | ~5,000 |
| platform/ | 9 | 5 | ~1,500 |
| common/ | 3 | 0 | ~200 |
| **合计** | ~38 | ~30 | **~10,200** |

新架构：约10,200行。

---

## 3. API层

本章定义 Engine、Session、请求参数结构体、结果与回调、C 绑定层（3.1–3.5）。

### 3.1 Engine

```cpp
namespace llm {

class Engine {
public:
    // 工厂方法创建引擎
    static std::expected<std::unique_ptr<Engine>, Status> Create(const EngineConfig& config);

    // Session管理
    Session* DefaultSession();                                    // 便利方法：默认Session
    std::expected<Session*, Status> CreateSession(const SessionConfig& config);
    Status DestroySession(Session* session);

    // 引擎级操作
    Status Suspend();
    Status Resume();
    Status SetPerfMode(PerfMode mode);

    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    Engine() = default;
};

}  // namespace llm
```

说明：`DefaultSession()` 返回引擎自动创建的默认Session，生命周期由Engine管理。简单场景下无需显式创建Session即可使用。

### 3.2 Session

```cpp
namespace llm {

class Session {
public:
    // 异步推理（核心接口）
    RequestId GenerateAsync(const GenerateParams& params, GenerateCallbacks callbacks);

    // 同步推理（内部调GenerateAsync + wait）
    std::expected<GenerateResult, Status> Generate(const GenerateParams& params);

    // 批量推理
    RequestId GenerateBatchAsync(const BatchGenerateParams& batchParams, GenerateCallbacks callbacks);
    std::expected<BatchGenerateResult, Status> GenerateBatch(const BatchGenerateParams& batchParams);

    // 请求控制
    Status Cancel(RequestId id);
    Status Terminate();

    // KV Cache操作（仅在 Idle 状态允许调用，Inferring 状态返回 Status::InvalidState）
    std::expected<KVCacheSnapshot, Status> ExportKVCache();
    Status ImportKVCache(const KVCacheSnapshot& snapshot);

    // Prompt Cache（仅在 Idle 状态允许调用）
    Status SetPrefixPrompt(std::string prefix);
    void ClearPrefixPrompt();

    // 权重更新
    Status UpdateWeight(const std::string& name, const void* data, size_t size, size_t offset);
    Status UpdateLora(const LoraConfig& config);

    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

private:
    Session() = default;
};

}  // namespace llm
```

### 3.3 请求参数结构体

```cpp
namespace llm {

// 采样配置
struct SamplingConfig {
    bool doSample = false;
    int32_t topK = 50;
    float topP = 0.9f;
    float temperature = 0.7f;
    float repetitionPenalty = 1.0f;
    uint64_t seed = 0;
};

// 生成配置
struct GenerationConfig {
    int32_t maxTokens = 1024;
    std::vector<std::string> stopSequences;
    int32_t callbackFreq = 1;       // 每N个token回调一次
};

// KV Cache配置（Optional，不设置则使用默认行为）
struct KVCacheConfig {
    bool sparseEnabled = false;
    SparseMode sparseMode = SparseMode::None;
    std::optional<SparseParams> sparseParams;
};

// 投机配置（Optional，未来扩展）
struct SpeculativeConfig {
    SpecType type = SpecType::Autoregressive;
    int32_t gamma = 5;                              // 候选token数
    float draftThreshold = 0.1f;
    std::optional<SpecTreeConfig> treeConfig;        // 树型投机特有
};

// 多模态输入（Optional，未来扩展）
struct MultimodalInput {
    // 多模态支持时定义
};

// 推理请求参数（聚合）
struct GenerateParams {
    // 输入
    std::string promptText;                                  // 文本输入
    std::optional<std::vector<int32_t>> promptTokenIds;      // 或直接传token ids

    // 配置
    SamplingConfig sampling;
    GenerationConfig generation;
    std::optional<KVCacheConfig> kvcache;

    // 高级特性（Optional，不设置则不启用）
    std::optional<SpeculativeConfig> speculative;            // 投机推理
    std::optional<MultimodalInput> multimodal;               // 多模态
};

// 批量推理参数
struct BatchGenerateParams {
    std::vector<GenerateParams> items;
    BatchStrategy strategy = BatchStrategy::DecodeOnly;      // 仅Decode阶段batch
};

// 引擎配置（创建Engine时传入）
struct EngineConfig {
    std::string modelPath;               // 模型路径
    std::string weightPath;              // 权重路径（可选，支持延迟加载）
    PlatformType platform = PlatformType::Kirin;
    PerfMode perfMode = PerfMode::Middle;
    std::optional<KirinPlatformConfig> kirinConfig;  // Kirin特定参数
};

// Session配置
struct SessionConfig {
    int32_t batchSize = 1;
    int32_t maxSeqLen = 4096;
    bool kvcacheSwitchable = false;      // KV Cache可换出
};

// LoRA配置
struct LoraConfig {
    std::string loraPath;                // 文件路径
    const void* loraBuffer = nullptr;    // 或内存buffer
    size_t loraBufferSize = 0;
};

}  // namespace llm
```

### 3.4 结果与回调

```cpp
namespace llm {

// 生成结果
struct GenerateResult {
    std::string text;
    std::vector<int32_t> tokenIds;
    ProfilingData profiling;
    Status status;
};

// 批量生成结果
struct BatchGenerateResult {
    std::vector<GenerateResult> items;
};

// 性能数据
struct ProfilingData {
    float totalTimeMs = 0;
    float prefillTimeMs = 0;
    float decodeTimeMs = 0;
    uint32_t inputTokenCount = 0;
    uint32_t outputTokenCount = 0;
    uint32_t cachedTokenCount = 0;
};

// 回调集合
struct GenerateCallbacks {
    std::function<void(const TokenEvent&)> onToken;          // 流式token回调
    std::function<void(const PrefillEvent&)> onPrefillDone;  // Prefill完成
    std::function<void(const GenerateResult&)> onDone;       // 全部完成
    std::function<void(Status)> onError;                     // 失败
};

// Token事件
struct TokenEvent {
    std::string text;
    int32_t tokenId;
    uint32_t tokenCount;      // 已生成token数
};

// Prefill事件
struct PrefillEvent {
    float prefillTimeMs;
    uint32_t inputTokenCount;
};

}  // namespace llm
```

### 3.5 C绑定层（可选）

C绑定层是薄映射层，仅对需要C ABI的调用方（如鸿蒙NAPI）提供。一对一包装C++ API调用，无额外逻辑。

```c
extern "C" {

LLMEngine* LLMEngine_Create(const LLMEngineConfig* config);
LLMStatus LLMEngine_Destroy(LLMEngine* engine);
LLMStatus LLMEngine_Generate(LLMEngine* engine, LLMSession* session,
                              const LLMGenerateParams* params, LLMGenerateResult* outResult);
// ... 仅对外需要暴露的接口，约20-30个函数

}  // extern "C"
```

---

## 4. 编排层

本章说明 Scheduler、InferRequest、Prefill/Decode 控制器与调度策略（4.1–4.5）。

### 4.1 Scheduler

```cpp
namespace llm::orchestration {

struct SchedulerConfig {
    int32_t maxQueueSize = 64;
    int32_t maxDecodeSlots = 1;    // 同时decode的批次数
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler();

    // 提交异步请求（核心入口）
    RequestId Submit(std::shared_ptr<InferRequest> request);

    // 批量提交异步请求
    RequestId SubmitBatch(std::shared_ptr<BatchInferRequest> batchRequest);

    Status Cancel(RequestId id);
    Status Shutdown();

    // 生命周期控制（Suspend/Resume使用）
    Status Pause();            // 停止从队列取新请求，等待当前请求完成
    Status Resume();           // 恢复从队列取请求

    // 调度策略访问接口
    std::shared_ptr<InferRequest> TryDequeue();
    bool IsQueueEmpty() const;
    bool HasActiveDecode() const;
    bool HasFreeSlot() const;
    void PrefillForSlot(std::shared_ptr<InferRequest> request, int32_t slot);

private:
    void RunLoop();                                     // 工作线程主循环
    std::shared_ptr<InferRequest> PickNext();           // 取下一个请求
    Status ExecuteRequest(std::shared_ptr<InferRequest> request);

    std::queue<std::shared_ptr<InferRequest>> queue_;
    std::mutex queueMtx_;
    std::condition_variable queueCv_;
    std::thread workerThread_;
    std::atomic<bool> running_{false};

    PrefillController* prefillCtrl_;
    DecodeController* decodeCtrl_;
};

}  // namespace llm::orchestration
```

### 4.2 InferRequest

```cpp
namespace llm {

// 推理请求对象，贯穿整个推理生命周期
class InferRequest {
public:
    RequestId id;
    GenerateParams params;
    GenerateCallbacks callbacks;

    // 请求状态机
    enum class State : uint8_t {
        Pending,       // 在队列中等待
        Prefilling,    // 正在Prefill
        Decoding,      // 正在Decode
        Drafting,      // 投机推理：draft阶段
        Verifying,     // 投机推理：verify阶段
        Completed,     // 完成
        Cancelled,     // 被取消
        Failed,        // 失败
    };
    std::atomic<State> state{State::Pending};

    // 推理过程中的中间数据
    std::vector<int32_t> inputTokenIds;       // token化后的输入
    std::vector<int32_t> outputTokenIds;      // 已生成的token
    std::string outputText;                    // 已生成的文本

    // 性能数据
    ProfilingData profiling;
    Status errorStatus;
};

// 批量请求：将多个InferRequest合并用于batch decode
class BatchInferRequest {
public:
    RequestId id;
    std::vector<std::shared_ptr<InferRequest>> subRequests;
    BatchStrategy strategy;
};

}  // namespace llm
```

### 4.3 PrefillController

```cpp
namespace llm::orchestration {

class PrefillController {
public:
    PrefillController(ModelRunner& modelRunner, Tokenizer& tokenizer,
                      Embedding& embedding, Sampler& sampler);

    // 执行Prefill（同步，在Scheduler工作线程中调用）
    // SessionContext 在每次调用时传入，包含该 Session 独立组件的引用
    Status Run(InferRequest& request, SessionContext& ctx);

private:
    ModelRunner& modelRunner_;
    Tokenizer& tokenizer_;
    Embedding& embedding_;
    Sampler& sampler_;
};

}  // namespace llm::orchestration
```

PrefillController::Run 内部流程：
1. tokenizer_.Encode(request.params.promptText) → request.inputTokenIds
2. promptManager.GetPrefixInfo() → prefixKVLen（可能为 nullopt）
3. 对输入 token 分块，逐块执行 Prefill 循环（见 8.2.2 分块 Prefill）
4. 采样 firstToken，tokenizer_.Decode({firstToken}) → tokenText
5. request.outputTokenIds.push_back(firstToken)
6. request.callbacks.onPrefillDone(...)

### 4.4 DecodeController

```cpp
namespace llm::orchestration {

class DecodeController {
public:
    DecodeController(ModelRunner& modelRunner, Embedding& embedding,
                     Tokenizer& tokenizer, Sampler& sampler);

    // 单batch Decode循环
    Status RunSingle(InferRequest& request, SessionContext& ctx);

    // 带策略的多batch Decode循环
    Status RunBatchWithStrategy(std::vector<std::shared_ptr<InferRequest>>& initialRequests,
                                IScheduleStrategy& strategy, Scheduler& scheduler);

private:
    ModelRunner& modelRunner_;
    Embedding& embedding_;
    Tokenizer& tokenizer_;
    Sampler& sampler_;

    bool CheckStop(const InferRequest& request);  // EOS / maxTokens / stopSeq

    struct DecodeSlot {
        int32_t index;
        std::shared_ptr<InferRequest> request;
        SessionContext* ctx;              // 每个-slot 绑定各自的 SessionContext
        SlotState state;  // Active, Completed, Empty
    };
};

}  // namespace llm::orchestration
```

### 4.5 调度策略

```cpp
namespace llm::orchestration {

class IScheduleStrategy {
public:
    virtual ~IScheduleStrategy() = default;

    // 当Decode中有slot空出时调用
    // 返回nullopt表示不填充，返回InferRequest表示用该请求填充
    virtual std::optional<std::shared_ptr<InferRequest>> OnSlotFree(
        int32_t freedSlot, int32_t totalSlots, Scheduler& scheduler) = 0;

    // 当新请求到达时调用
    // 返回true表示立即处理，false表示排队等待
    virtual bool OnRequestArrived(
        std::shared_ptr<InferRequest> request, Scheduler& scheduler) = 0;

    // 判断Decode是否应该结束
    virtual bool ShouldStopDecode(
        const std::vector<SlotState>& slotStates, Scheduler& scheduler) = 0;
};

// 简单策略：先收集满batch个请求，统一prefill，统一decode，slot空出不补充
class SimpleBatchStrategy : public IScheduleStrategy { /* ... */ };

// 连续批处理策略：当slot空出时，检查队列是否有待处理请求，有则prefill后填入slot
class ContinuousBatchStrategy : public IScheduleStrategy { /* ... */ };

}  // namespace llm::orchestration
```

---

## 5. 组件层

本章覆盖 ModelRunner、TensorManager、KVCacheManager、Tokenizer、Sampler、Embedding、PromptManager 与组件交互规则（5.1–5.8）。

### 5.1 ModelRunner

```cpp
namespace llm::component {

class ModelRunner {
public:
    explicit ModelRunner(platform::IModelLoader& loader, platform::IExecutor& executor);
    ~ModelRunner();

    // 初始化：加载模型图和权重
    Status Init(const ModelRunnerConfig& config);

    // 执行Prefill
    Status RunPrefill(const StepTensors& stepTensors,
                      const std::vector<platform::Tensor>& kvCacheTensors);

    // 执行单batch Decode
    Status RunDecode(int32_t tokenId, int32_t posStart,
                     const StepTensors& stepTensors,
                     const std::vector<platform::Tensor>& kvCacheTensors);

    // 执行多batch Decode
    Status RunDecodeBatch(const std::vector<int32_t>& tokenIds,
                          const std::vector<int32_t>& posStarts,
                          const StepTensors& stepTensors,
                          const std::vector<platform::Tensor>& kvCacheTensors);

    // 权重管理
    Status UpdateWeight(const std::string& name, const void* data, size_t size, size_t offset);
    Status UpdateLora(const LoraConfig& config);

    // 性能模式
    Status SetPerfMode(PerfMode mode);

    // 资源控制
    Status FreeResource();
    Status RestoreResource();

private:
    platform::IModelLoader& modelLoader_;
    platform::IExecutor& executor_;

    // 模型实例（按batch分组，可能持有batch=1和batch=N两个实例）
    struct ModelInstance {
        platform::ModelHandle handle;
        int32_t batchSize;
    };
    std::vector<ModelInstance> models_;

    // TensorRole → NPU 模型图输入顺序的映射
    // 在 Init 时从模型元数据或配置中加载，描述模型图输入张量的排列顺序
    // 如某模型的输入顺序为 [PositionIds, InputIds, AttentionMask, LLMConfig, KV0, KV1, ...]
    // 则 inputOrder_ = {PositionIds, InputIds, AttentionMask, LLMConfig, KVCacheKey, ...}
    std::vector<TensorRole> inputOrder_;
};

struct ModelRunnerConfig {
    std::string modelPath;
    std::string weightPath;
    int32_t maxSeqLen = 4096;
    bool initWithoutWeight = false;     // 支持延迟加载权重
    bool supportBatchDecode = false;
    int32_t batchDecodeSize = 1;
    // NPU 模型图输入顺序（不同模型图的输入张量排列不同）
    // 如不设置，Init 时尝试从模型元数据自动推断
    std::optional<std::vector<TensorRole>> inputOrder;
};

}  // namespace llm::component
```

### 5.2 TensorManager

管理非 KV Cache 的输入/输出 tensor 生命周期。**与 KVCacheManager 明确分工**：TensorManager 管理"每步重填的临时 tensor"（inputIds、attentionMask、logits 等），KVCacheManager 管理"跨步累积的状态 tensor"（KV Cache）。

**设计决策**：方案B —— KVCacheManager 独立拥有 KV tensor，TensorManager 只管非 KV tensor。理由：
- KV Cache 是跨步累积的状态，与每步重填的临时 tensor 生命周期不同
- KV Cache 的 Export/Import/SwapOut/SwapIn 是特有操作，自包含在一个组件中更干净
- 两个组件各自持有 ITensorAllocator 引用，独立分配，便于分别追踪内存用量

```cpp
namespace llm::component {

// Tensor 语义角色标识
enum class TensorRole : uint16_t {
    // 输入 - 通用
    InputIds,
    InputEmbedding,
    AttentionMask,
    PositionIds,
    MropeSin,
    MropeCos,
    // 输入 - 运行时配置
    LLMConfig,
    OutputPos,
    // 输入 - 稀疏/投机/MoE
    HistoryActivationMask,
    EagleFeature,
    MoeMapInfo,
    // 输出
    Logits,
    HiddenState,
    Query,
};

// 单个推理步骤所需的 tensor 集合（不含 KV Cache，KV 由 KVCacheManager 提供）
struct StepTensors {
    std::unordered_map<TensorRole, std::vector<platform::Tensor>> inputs;
    std::unordered_map<TensorRole, std::vector<platform::Tensor>> outputs;
};

class TensorManager {
public:
    explicit TensorManager(platform::ITensorAllocator& allocator);
    ~TensorManager();

    // ========== 初始化 ==========

    // 根据模型参数配置分配 tensor 池
    Status Init(const TensorManagerConfig& config);

    // ========== 获取 Tensor ==========

    // 获取指定 seqLen + batch 的输入输出 tensor 集合
    // 首次请求该 seqLen 时触发分配，后续直接返回缓存
    std::expected<StepTensors, Status> GetStepTensors(int32_t seqLen, int32_t batchSize);

    // 获取下一份 embedding 输入缓冲（支持 Prefill 流水线双缓冲）
    std::expected<platform::Tensor, Status> GetNextEmbeddingInput(int32_t seqLen);

    // ========== 数据填充辅助 ==========

    // 填写运行时配置 tensor（posStart, totalInputNum, validLen 等）
    Status FillLLMConfig(const platform::Tensor& configTensor,
                         int32_t posStart, int32_t totalInputNum,
                         int32_t validLen, int32_t batchSize);

    // 填写 attention mask
    // totalKvLen 为本次推理的总 KV 长度：
    //   Prefill: existingKV + seqLen（已有 KV + 本轮新增）
    //   Decode:  existingKV + 1（已有 KV + 本步写入的 1 个位置）
    Status FillAttentionMask(const platform::Tensor& maskTensor,
                            int32_t seqLen, int32_t totalKvLen,
                            int32_t batchSize);

    // ========== 资源管理 ==========

    // 释放所有 tensor 缓冲（Suspend）
    Status ReleaseBuffers();

    // 重建所有 tensor 缓冲（Resume）
    Status RebuildBuffers();

    // 获取当前 tensor 总内存占用（不含 KV Cache）
    size_t GetMemoryUsage() const;

private:
    platform::ITensorAllocator& allocator_;

    // 按 seqLen 索引的 tensor 缓存池
    std::unordered_map<int32_t, StepTensors> tensorPool_;
    std::mutex poolMutex_;

    // Prefill embedding 双缓冲（每个 seqLen 2 份）
    std::unordered_map<int32_t,
        std::array<platform::Tensor, 2>> embeddingPipeline_;
    std::unordered_map<int32_t, size_t> pipelineIndex_;  // 当前使用哪一份

    TensorManagerConfig config_;
    bool initialized_ = false;
    bool buffersFreed_ = false;
};

struct TensorManagerConfig {
    // 模型参数
    int32_t hiddenSize = 4096;
    int32_t vocabSize = 32000;
    int32_t vocabAlignSize = 32000;
    int32_t maxSeqLen = 4096;       // AttentionMask 的第二维度按 maxSeqLen 分配，运行时 actualLen ≤ maxSeqLen
    int32_t batchSize = 1;

    // Prefill 配置
    std::vector<int32_t> prefillBlockSeqLens;  // 如 {32, 64, 128, 256, 512}
    int32_t decodeBlockSeqLen = 1;

    // 特性开关
    bool useExternalEmbedding = false;
    bool useMrope = false;
    bool enableSparse = false;
    bool enableLora = false;
    int32_t mhcNumStream = 1;
    int32_t prefillPipelineCount = 2;  // Prefill embedding 缓冲数
};

}  // namespace llm::component
```

#### TensorManager 与 KVCacheManager 的协作

两个组件不直接交互，由编排层协调：

```
PrefillController::Run(request):
  // 从两个组件分别获取 tensor
  auto stepTensors = tensorMgr.GetStepTensors(seqLen, batchSize);
  auto& kvTensors  = kvCacheMgr.GetTensors();  // KV Cache tensor

  // 填充数据（kvCacheLen 从 KVCacheManager 获取）
  tensorMgr.FillAttentionMask(stepTensors, seqLen, kvCacheMgr.GetCurrentPos(), batch);
  tensorMgr.FillLLMConfig(stepTensors, posStart, totalInputNum, validLen, batch);

  // 合并传入 ModelRunner
  modelRunner.RunPrefill(stepTensors, kvTensors, ...);
```

#### 与当前实现的对比

| 维度 | 当前实现 | 新设计 |
|------|---------|--------|
| 分配时机 | Init 时分配所有 seqLen 变体 | 首次请求时按需分配，减少启动内存峰值 |
| Tensor 语义标识 | 按 InputIndex 位序枚举 | 按 TensorRole 枚举，解耦模型图输入顺序 |
| KV Cache 管理 | 散布在 TensorManager + ModelExecutor | KVCacheManager 独立拥有，TensorManager 不碰 |
| Embedding 流水线 | 硬编码 2 份缓冲 | prefillPipelineCount 可配置 |
| 外部 KV Cache | 全局指针 + batch 切换 | 由 KVCacheManager 负责，不经过 TensorManager |

### 5.3 KVCacheManager

这是v1的核心特性。

```cpp
namespace llm::component {

class KVCacheManager {
public:
    explicit KVCacheManager(platform::ITensorAllocator& allocator);
    ~KVCacheManager();

    // ========== 生命周期 ==========

    // 初始化KV Cache（根据模型参数分配空间）
    Status Init(const KVCacheConfig& config);

    // ========== Prefill相关 ==========

    // 准备Prefill的KV Cache位置
    Status PreparePrefill(int32_t seqLen);

    // Prefill完成后更新KV Cache的有效长度
    Status CommitPrefill(int32_t actualLen);

    // ========== Decode相关 ==========

    // 获取当前KV Cache已使用的位置（decode从这里续写）
    int32_t GetCurrentPos() const;

    // Decode一步后KV Cache位置+1
    void AdvancePos();

    // ========== 导出/导入 ==========

    // 导出KV Cache快照（用于Prompt Cache / 端云协同）
    std::expected<KVCacheSnapshot, Status> Export() const;

    // 导入KV Cache快照
    Status Import(const KVCacheSnapshot& snapshot);

    // ========== 稀疏KV Cache ==========

    // 设置稀疏模式
    Status SetSparseMode(SparseMode mode);

    // 获取稀疏掩码
    std::expected<platform::Tensor, Status> GetSparseMask(int32_t seqLen) const;

    // ========== 动态扩容 ==========

    // 扩展KV Cache容量
    Status Expand(int32_t newMaxSeqLen);

    // ========== Tensor 访问 ==========

    // 获取 KV Cache tensor（供编排层传入 ModelRunner）
    const std::vector<KVCacheLayer>& GetTensors() const;

    // ========== 资源管理 ==========

    // 交换出内存（suspend场景）
    Status SwapOut();

    // 交换回内存（resume场景）
    Status SwapIn();

    // 获取当前KV Cache占用的内存大小
    size_t GetMemoryUsage() const;

private:
    platform::ITensorAllocator& allocator_;

    // KV Cache tensor（key + value，每层各一份）
    struct KVCacheLayer {
        Tensor keyCache;
        Tensor valueCache;
    };
    std::vector<KVCacheLayer> layers_;

    int32_t maxSeqLen_ = 0;
    int32_t currentPos_ = 0;
    int32_t numLayers_ = 0;
    int32_t numKVHeads_ = 0;        // GQA场景下 numKVHeads < numHeads
    int32_t headDim_ = 0;

    // 稀疏相关
    SparseMode sparseMode_ = SparseMode::None;
    std::unique_ptr<SparseKVCacheHelper> sparseHelper_;

    // 换出缓存
    std::vector<uint8_t> swappedData_;
};

struct KVCacheConfig {
    int32_t maxSeqLen = 4096;
    int32_t numLayers = 32;
    int32_t numKVHeads = 32;
    int32_t headDim = 128;
    bool switchable = false;        // 是否支持换出
    int32_t dtypeSize = 2;          // fp16=2, fp32=4
};

struct KVCacheSnapshot {
    std::vector<uint8_t> data;      // 序列化的KV Cache数据
    int32_t seqLen;                 // 有效长度
    int32_t numLayers;
    int32_t numKVHeads;
    int32_t headDim;
};

}  // namespace llm::component
```

### 5.4 Tokenizer

```cpp
namespace llm::component {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // 工厂方法
    static std::expected<std::unique_ptr<Tokenizer>, Status>
        Create(const std::string& modelPath, TokenizerType type);

    // 文本 → token ids
    std::expected<std::vector<int32_t>, Status> Encode(const std::string& text);

    // token ids → 文本
    std::expected<std::string, Status> Decode(const std::vector<int32_t>& ids);

    // 获取词表信息
    int32_t GetVocabSize() const;
    int32_t GetEosTokenId() const;
    int32_t GetPadTokenId() const;
    int32_t GetBosTokenId() const;

protected:
    virtual Status DoEncode(const std::string& text, std::vector<int32_t>& ids) = 0;
    virtual Status DoDecode(const std::vector<int32_t>& ids, std::string& text) = 0;
};

enum class TokenizerType { BPE, SentencePiece, Qwen, GLM };

}  // namespace llm::component
```

### 5.5 Sampler

```cpp
namespace llm::component {

class Sampler {
public:
    // 从logits中采样一个token
    int32_t Sample(const Tensor& logits, const SamplingConfig& config);

private:
    int32_t SampleGreedy(const float* logits, int32_t vocabSize);
    int32_t SampleTopK(const float* logits, int32_t vocabSize, int32_t k, float temperature);
    int32_t SampleTopP(const float* logits, int32_t vocabSize, float p, float temperature);
    void ApplyRepetitionPenalty(float* logits, int32_t vocabSize,
                                 const std::vector<int32_t>& inputIds, float penalty);
};

}  // namespace llm::component
```

### 5.6 Embedding

```cpp
namespace llm::component {

class Embedding {
public:
    explicit Embedding(platform::ITensorAllocator& allocator);
    ~Embedding();

    // 初始化：加载embedding权重
    Status Init(const std::string& modelPath);

    // token ids → embedding tensor
    std::expected<Tensor, Status> Run(const std::vector<int32_t>& tokenIds);

    // 是否支持外部embedding（某些模型允许跳过内部embedding层）
    bool SupportsExternalEmbedding() const;

private:
    platform::ITensorAllocator& allocator_;
    Tensor embeddingWeight_;      // 词表权重矩阵
    bool supportsExternal_ = false;
};

}  // namespace llm::component
```

### 5.7 PromptManager

管理前缀提示缓存和历史 Prompt Cache。**Session 级组件**，每个 Session 独立拥有，持有同 Session 的 KVCacheManager 引用。

```cpp
namespace llm::component {

class PromptManager {
public:
    explicit PromptManager(KVCacheManager& kvCacheMgr, Tokenizer& tokenizer);
    ~PromptManager();

    // 设置前缀提示（如system prompt）
    // 内部调用 tokenizer.Encode + KVCacheManager 执行一次 Prefill 将前缀写入 KV Cache
    Status SetPrefixPrompt(const std::string& prefix);
    void ClearPrefixPrompt();

    // 获取前缀提示对应的token ids和KV Cache长度
    struct PrefixInfo {
        std::vector<int32_t> tokenIds;
        int32_t kvCacheLen;  // 前缀已缓存的KV长度
    };
    std::optional<PrefixInfo> GetPrefixInfo() const;

    // 保存当前prompt的KV Cache（Prompt Cache）
    Status SavePromptCache(const std::string& promptKey);

    // 加载之前保存的prompt cache
    std::expected<int32_t, Status> LoadPromptCache(const std::string& promptKey);

private:
    KVCacheManager& kvCacheMgr_;    // 同一 Session 的 KVCacheManager
    Tokenizer& tokenizer_;          // Engine 级共享
    std::optional<PrefixInfo> prefixInfo_;

    // Prompt Cache存储
    struct PromptCacheEntry {
        std::string key;
        std::vector<uint8_t> kvCacheData;
        int32_t seqLen;
    };
    std::unordered_map<std::string, PromptCacheEntry> promptCache_;
};

}  // namespace llm::component
```

### 5.8 组件交互规则

组件之间永远不直接调用。编排层通过局部变量（tensor、id、pos）传递中间数据来协调组件协作。

**Engine 级组件**（ModelRunner、Tokenizer、Embedding、Sampler）被所有 Session 共享。
**Session 级组件**（KVCacheManager、TensorManager、PromptManager）打包在 SessionContext 中，每次推理时传入 Controller 的 Run 方法。
Controller 构造时只绑定 Engine 级组件引用，Run 时再接收 SessionContext。

TensorManager 与 KVCacheManager 不直接交互，由编排层分别从两者获取 tensor 后组装传入 ModelRunner。

Prefill流程示例（Controller 内部）：
1. tokenizer_.Encode(prompt) → tokenIds
2. ctx.promptMgr.GetPrefixInfo() → prefixKVLen
3. 对输入分块，每块：
   a. embedding_.Run(chunkTokenIds) → embeddingTensor
   b. ctx.tensorMgr.GetStepTensors(chunkSeqLen, batch) → stepTensors
   c. ctx.tensorMgr.FillAttentionMask(mask, chunkSeqLen, totalKvLen, batch)  // totalKvLen = ctx.kvCacheMgr.GetCurrentPos() + chunkSeqLen
   d. ctx.tensorMgr.FillLLMConfig(config, posStart, ...)
   e. ctx.kvCacheMgr.PreparePrefill(chunkSeqLen)
   f. ctx.kvCacheMgr.GetTensors() → kvTensors
   g. modelRunner_.RunPrefill(stepTensors, kvTensors) → logits
   h. ctx.kvCacheMgr.CommitPrefill(chunkSeqLen)
4. sampler_.Sample(logits, config) → firstToken
5. tokenizer_.Decode({firstToken}) → tokenText

---

## 6. 平台HAL

本章定义平台 HAL 的基础类型、核心接口、平台工厂及 Kirin/Mock 实现（6.1–6.5）。

### 6.1 基础类型

```cpp
namespace llm::platform {

// Tensor描述
struct TensorDesc {
    std::vector<int32_t> shape;       // 如 [batch, seq, hidden]
    DataType dtype;                   // FP16, FP32, INT8, INT4
    size_t GetElementCount() const;
    size_t GetByteSize() const;
};

enum class DataType : uint8_t { FP32, FP16, INT8, INT4, INT32 };

// Tensor（包装底层buffer，上层不感知具体分配方式）
class Tensor {
public:
    void* Data();
    const void* Data() const;
    size_t Size() const;
    const TensorDesc& Desc() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// 模型句柄（不透明）
using ModelHandle = void*;

// 执行结果
struct ExecuteResult {
    Status status;
    std::vector<Tensor> outputs;
};

// 性能模式
enum class PerfMode : uint8_t { Low, Middle, High, ExtremeHigh };

}  // namespace llm::platform
```

### 6.2 核心接口

```cpp
namespace llm::platform {

// 模型加载器
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // 加载模型（batch维度在此时确定）
    virtual std::expected<ModelHandle, Status> Load(
        const std::string& modelPath, int32_t batchSize) = 0;

    // 仅加载图结构（延迟加载权重）
    virtual std::expected<ModelHandle, Status> LoadGraph(
        const std::string& modelPath, int32_t batchSize) = 0;

    // 加载权重到已有的图
    virtual Status LoadWeight(ModelHandle handle, const std::string& weightPath) = 0;

    // 卸载模型
    virtual Status Unload(ModelHandle handle) = 0;
};

// Tensor分配器
class ITensorAllocator {
public:
    virtual ~ITensorAllocator() = default;

    virtual std::expected<Tensor, Status> Allocate(const TensorDesc& desc) = 0;
    virtual std::expected<std::vector<Tensor>, Status> AllocateBatch(
        const std::vector<TensorDesc>& descs) = 0;
    virtual size_t GetMemoryUsage() const = 0;
    virtual void Reset() = 0;
};

// 模型执行器
class IExecutor {
public:
    virtual ~IExecutor() = default;

    virtual std::expected<ExecuteResult, Status> Execute(
        ModelHandle model, const std::vector<Tensor>& inputs) = 0;
    virtual Status UpdateWeight(ModelHandle model, const std::string& weightName,
                                const void* data, size_t size, size_t offset) = 0;
    virtual Status UpdateLora(ModelHandle model, const LoraConfig& config) = 0;
    virtual Status SetPerfMode(PerfMode mode) = 0;
};

// 性能控制器
class IPerfController {
public:
    virtual ~IPerfController() = default;

    virtual Status SetPerfMode(PerfMode mode) = 0;
    virtual Status BindCPU(int32_t coreId) = 0;
    virtual PerfMode GetCurrentMode() const = 0;
};

}  // namespace llm::platform
```

### 6.3 平台工厂

```cpp
namespace llm::platform {

class PlatformFactory {
public:
    struct PlatformInstances {
        std::unique_ptr<IModelLoader> modelLoader;
        std::unique_ptr<ITensorAllocator> tensorAllocator;
        std::unique_ptr<IExecutor> executor;
        std::unique_ptr<IPerfController> perfController;
    };

    static std::expected<PlatformInstances, Status> Create(PlatformType type);
};

enum class PlatformType : uint8_t {
    Kirin,      // Kirin NPU
    Mock,       // 测试用
    // 未来: Cpu, Gpu, ...
};

}  // namespace llm::platform
```

### 6.4 Kirin平台实现

使用 hiai::IModelManager、hiai::INDTensorBuffer、hiai::PerfModeManager、FFRT、lm_engine_mmgr 实现 IModelLoader、ITensorAllocator、IExecutor、IPerfController。所有 hiai::* 头文件包含仅在 platform/kirin/ 目录中。

### 6.5 Mock实现（测试用）

不依赖NPU硬件实现HAL接口，用于全流程单元测试。

---

## 7. 错误处理与状态管理

本章说明错误处理与状态管理（7.1–7.2）。

### 7.1 错误处理

使用 `std::expected<T, Status>` + 状态码，不用异常（NPU/嵌入式场景不适宜异常）。

```cpp
namespace llm {

enum class Status : int32_t {
    OK = 0,

    // 通用错误
    InvalidArgument,
    OutOfMemory,
    InternalError,

    // 模型相关
    ModelLoadFailed,
    ModelNotInitialized,
    ModelExecuteFailed,
    WeightUpdateFailed,

    // KV Cache相关
    KVCacheAllocFailed,
    KVCacheExportFailed,
    KVCacheImportFailed,
    KVCacheExpandFailed,

    // Tokenizer相关
    TokenizerInitFailed,
    EncodeFailed,
    DecodeFailed,

    // 请求相关
    RequestCancelled,
    RequestQueueFull,
    InvalidRequest,

    // 平台相关
    PlatformNotSupported,
    PlatformInitFailed,
};

}  // namespace llm
```

使用方式：
```cpp
// 返回值可能失败的操作用 std::expected
auto engine = Engine::Create(config);          // std::expected<unique_ptr<Engine>, Status>
if (!engine) { /* handle engine.error() */ }

auto result = session->Generate(params);        // std::expected<GenerateResult, Status>
if (!result) { /* handle result.error() */ }

// 不可能失败的操作直接返回值
int32_t pos = kvCacheMgr.GetCurrentPos();       // 直接返回
```

### 7.2 状态管理

用显式状态机替代散落的 `atomic<bool>` 标志位。

**引擎状态：**
```
Uninitialized → Ready          （初始化成功）
Ready → Suspended              （挂起）
Suspended → Ready              （恢复）
Ready → Shutdown               （销毁）
Suspended → Shutdown           （销毁）
```

**Session状态：**
```
Idle → Inferring               （GenerateAsync提交）
Inferring → Idle               （推理完成/失败）
Inferring → Cancelled          （取消）
Cancelled → Idle               （清理完成）
Any → Destroyed                （DestroySession）
```

Session 操作约束：
- ExportKVCache / ImportKVCache / SetPrefixPrompt：仅在 Idle 状态允许，Inferring 状态调用返回 Status::InvalidState
- GenerateAsync / Generate：仅在 Idle 状态允许
- Cancel：仅在 Inferring 状态允许

**请求状态：**
```
Pending → Prefilling           （Scheduler开始处理）
Prefilling → Decoding          （Prefill完成）
Decoding → Completed           （正常完成）
Decoding → Cancelled           （取消）
Any → Failed                   （错误）
```

状态转换使用 `compare_exchange_strong` 保证原子性，防止竞态条件。

---

## 8. 关键流程

本章给出引擎初始化、首次推理、多轮对话、KV/Prompt Cache、Batch Decode、挂起恢复与数据流全景（8.1–8.8）。

### 8.1 引擎初始化（模型加载）

```
Engine::Create(engineConfig)
│
├─ 1. PlatformFactory::Create(config.platform)    // 如 PlatformType::Kirin
│     → 返回 PlatformInstances:
│       ├── IModelLoader      (KirinModelLoader: 封装 hiai::IModelManager)
│       ├── ITensorAllocator  (KirinTensorAllocator: 封装 NPU 内存分配)
│       ├── IExecutor         (KirinExecutor: 封装 hiai::IModel::Execute)
│       └── IPerfController   (KirinPerfController: 封装调频/CPU绑定)
│
├─ 2. 创建 Engine 级共享组件（多 Session 共享）
│     ├── ModelRunner(modelLoader, executor)
│     │     └─ modelRunner.Init(modelRunnerConfig)
│     │         ├─ modelLoader.Load(modelPath, batchSize=1)     → ModelHandle
│     │         └─ modelLoader.Load(modelPath, batchSize=N)     → ModelHandle (如果支持batch)
│     │
│     ├── Tokenizer::Create(modelPath, config.tokenizerType)    → 工厂创建具体子类
│     ├── Embedding(tensorAllocator)
│     │     └─ embedding.Init(modelPath)                        → 加载 embedding 权重矩阵
│     └── Sampler()
│
├─ 3. 创建编排层（Controller 只绑定 Engine 级组件，Session 级组件在 Run 时传入）
│     ├── Scheduler(schedulerConfig)
│     │     └─ 启动工作线程 RunLoop()
│     ├── PrefillController(modelRunner, tokenizer, embedding, sampler)
│     └── DecodeController(modelRunner, embedding, tokenizer, sampler)
│
├─ 4. 创建默认 Session（Session 级组件在此创建）
│     └─ DefaultSession()
│         └─ new Session(id, scheduler, engineSharedComponents)
│             ├─ 创建独立的 KVCacheManager(tensorAllocator)
│             │     └─ kvCacheMgr.Init(kvCacheConfig)
│             │         → 按 numLayers × numKVHeads × headDim × maxSeqLen
│             │           分配每层的 key/value cache tensor
│             ├─ 创建独立的 TensorManager(tensorAllocator)
│             │     └─ tensorMgr.Init(tensorManagerConfig)  // 仅记录配置，不分配
│             ├─ 创建独立的 PromptManager(kvCacheMgr, tokenizer)
│             └─ 组装 SessionContext { kvCacheMgr, tensorMgr, promptMgr }
│
└─ 5. EngineState: Uninitialized → Ready
```

关键点：
- 模型加载通过 ModelRunner → IModelLoader → KirinModelLoader 完成，KirinModelLoader 内部调用 hiai::IModelManager::LoadFromModelBuffer 加载 NPU 模型图+权重
- Engine 级组件（ModelRunner、Tokenizer、Embedding、Sampler）被所有 Session 共享
- Session 级组件（KVCacheManager、TensorManager、PromptManager）每个 Session 独立，打包在 SessionContext 中
- TensorManager 此时不分配具体 tensor，仅记录配置，后续按需分配

### 8.2 首次推理（同步 Generate）

用户调用：
```cpp
llm::GenerateParams params;
params.promptText = "你好，请介绍一下你自己";
params.sampling = {.doSample = true, .topK = 50, .topP = 0.9f, .temperature = 0.7f};
params.generation = {.maxTokens = 1024, .callbackFreq = 1};
auto result = session->Generate(params);
```

#### 8.2.1 请求提交

```
Session::Generate(params)
│
├─ 创建 InferRequest
│     ├── id = nextRequestId++
│     ├── params = 用户传入的 GenerateParams
│     ├── callbacks = { onDone: [promise](result) { promise.set_value(result); } }
│     ├── state = Pending
│     ├── inputTokenIds = []
│     ├── outputTokenIds = []
│     └── outputText = ""
│
├─ sessionState: Idle → Inferring
│
├─ scheduler.Submit(request)
│     └─ 入队 queue_，通知 queueCv_
│
└─ return future.get()   // 阻塞等待结果
```

#### 8.2.2 Prefill 阶段（Scheduler 工作线程）

Prefill 采用**分块策略**：长输入按 `prefillBlockSeqLens` 配置拆分为多个 chunk 逐步执行。每个 chunk 独立执行一次模型推理，逐步填充 KV Cache。分块的好处：
- 避免 seqLen 过大时一次性分配超大 tensor 导致 OOM
- 支持 CPU/NPU 流水线双缓冲（embedding 查表与 NPU 计算并行）
- 分块大小匹配 NPU 的最优计算粒度

```
Scheduler::ExecuteRequest(request)
│
├─ request.state: Pending → Prefilling
│
└─ PrefillController::Run(request, ctx)
   │
   ├─ 1. Tokenizer 编码
   │     tokenizer_.Encode(request.params.promptText)
   │     → "你好，请介绍一下你自己" → [3546, 898, ...] = request.inputTokenIds
   │
   ├─ 2. 前缀提示检查
   │     ctx.promptMgr.GetPrefixInfo()
   │     → 如果之前设置过 system prompt，返回已缓存的 tokenIds 和 KV 位置
   │     → 此处首次，返回 nullopt
   │
   ├─ 3. 输入分块
   │     将 inputTokenIds 按 prefillBlockSeqLens 分块
   │     如输入 2000 tokens，配置 {32, 64, 128, 256, 512}:
   │       chunk[0]: tokens[0:32],   chunkSeqLen=32
   │       chunk[1]: tokens[32:96],  chunkSeqLen=64
   │       chunk[2]: tokens[96:224], chunkSeqLen=128
   │       chunk[3]: tokens[224:480],chunkSeqLen=256
   │       chunk[4]: tokens[480:992],chunkSeqLen=512
   │       chunk[5]: tokens[992:2000],chunkSeqLen=1008  (剩余部分)
   │
   └─ 4. 对每个 chunk 执行 Prefill 循环:
      │
      │  posStart = ctx.kvCacheMgr.GetCurrentPos()   ← 本 chunk 写入 KV 的起始位
      │  existingKV = posStart                         ← 已有 KV 长度
      │
      ├─ 4a. Embedding 查表
      │     embedding_.Run(chunkTokenIds)
      │     → 从 embeddingWeight_ 矩阵按 token id 查表
      │     → 返回 embeddingTensor: shape=[1, chunkSeqLen, hiddenSize]
      │
      ├─ 4b. 获取非 KV 的输入输出 tensor
      │     ctx.tensorMgr.GetStepTensors(chunkSeqLen, batchSize=1)
      │     → 首次请求该 seqLen，触发按需分配：
      │       ├── InputIds tensor      [1, chunkSeqLen]
      │       ├── AttentionMask tensor [1, 1, chunkSeqLen, maxSeqLen]
      │       ├── PositionIds tensor   [1, chunkSeqLen]
      │       ├── LLMConfig tensor     [1, N]
      │       ├── OutputPos tensor     [1]
      │       └── Logits tensor        [1, chunkSeqLen, vocabAlignSize]
      │
      ├─ 4c. 填充 Attention Mask
      │     totalKvLen = existingKV + chunkSeqLen   ← 已有 KV + 本轮新增
      │     ctx.tensorMgr.FillAttentionMask(
      │         stepTensors.inputs[TensorRole::AttentionMask],
      │         chunkSeqLen, totalKvLen, batchSize=1)
      │     → 因果掩码：当前 chunk 中位置 i 可看到 [0, existingKV+i] 范围
      │     → 首次 chunk，existingKV=0，totalKvLen=chunkSeqLen，为下三角因果掩码
      │
      ├─ 4d. 填充运行时配置
      │     ctx.tensorMgr.FillLLMConfig(
      │         stepTensors.inputs[TensorRole::LLMConfig],
      │         posStart, totalInputNum, validLen, batchSize=1)
      │
      ├─ 4e. 准备 KV Cache 位置
      │     ctx.kvCacheMgr.PreparePrefill(chunkSeqLen)
      │     → 内部记录预填区间 [currentPos_, currentPos_ + chunkSeqLen)
      │
      ├─ 4f. 获取 KV Cache tensor
      │     ctx.kvCacheMgr.GetTensors() → kvTensors
      │
      ├─ 4g. 模型执行 Prefill
      │     modelRunner_.RunPrefill(stepTensors, kvTensors)
      │     │
      │     └─ 内部流程：
      │        ├─ 按 inputOrder_ 从 stepTensors + kvTensors 组装 inputs vector
      │        ├─ executor.Execute(modelHandle, inputs)
      │        │     └─ KirinExecutor: Tensor→INDTensorBuffer→hiai::IModel::Execute→NPU
      │        └─ 从 ExecuteResult.outputs 中提取 logits tensor
      │
      │     ★ KV Cache 被写入：
      │       keyCache[layer][posStart:posStart+chunkSeqLen] ← 该 chunk 各位置的 key
      │       valueCache[layer][posStart:posStart+chunkSeqLen] ← 该 chunk 各位置的 value
      │
      ├─ 4h. 提交本 chunk 的 Prefill 结果
      │     ctx.kvCacheMgr.CommitPrefill(chunkSeqLen)
      │     → currentPos_ += chunkSeqLen
      │
      └─ 继续下一 chunk（回到 4a）
         // 流水线优化：当前 chunk 的 NPU 计算（4g）与下一 chunk 的 Embedding 查表（4a）可并行
         // 通过 GetNextEmbeddingInput 双缓冲机制实现

   最后一个 chunk 完成后：

   ├─ 5. 采样第一个输出 token
   │     sampler_.Sample(logits, request.params.sampling)
   │     → 取最后一个 chunk 最后一位置的 logits
   │     → doSample=true → SampleTopK → 选出 token id 3520
   │
   ├─ 6. Tokenizer 解码
   │     tokenizer_.Decode({3520}) → "我"
   │
   ├─ 7. 记录输出
   │     request.outputTokenIds.push_back(3520)
   │     request.outputText += "我"
   │
   └─ 8. 回调通知
        request.callbacks.onPrefillDone({prefillTimeMs=xx, inputTokenCount=2000})
```

Prefill 阶段 KV Cache 变化（以 2000 tokens、5 个 chunks 为例）：
```
初始:      keyCache[layer][....................................] pos=0
Chunk 0 后: keyCache[layer][████................................] pos=32
Chunk 1 后: keyCache[layer][██████████..........................] pos=96
Chunk 2 后: keyCache[layer][██████████████████..................] pos=224
  ...
最后 chunk: keyCache[layer][████████████████████████████████████] pos=2000
```

#### 8.2.3 Decode 阶段（自回归循环）

```
DecodeController::RunSingle(request, ctx)
│
├─ request.state: Prefilling → Decoding
│
└─ while (!CheckStop(request)):
   │
   │  CheckStop 检查:
   │  ├── outputTokenIds 最后一个是 EOS token? → 否
   │  ├── outputTokenIds.size() >= maxTokens(1024)? → 否
   │  └── 命中 stopSequences? → 否
   │
   ├─ 1. 获取 Decode 步骤的 tensor
   │     ctx.tensorMgr.GetStepTensors(seqLen=1, batchSize=1)
   │     → 返回 seqLen=1 对应的 stepTensors
   │
   ├─ 2. 输入准备（根据 useExternalEmbedding 分两路）
   │
   │   路径 A：useExternalEmbedding = false（模型内部做 embedding）
   │     stepTensors.inputs[TensorRole::InputIds].Data() = lastTokenId
   │     // 直接将 token id 写入 InputIds tensor
   │
   │   路径 B：useExternalEmbedding = true（外部 Embedding 查表）
   │     embedding_.Run({lastTokenId}) → embedTensor
   │     stepTensors.inputs[TensorRole::InputEmbedding] = embedTensor
   │     // 将 embedding 结果写入 InputEmbedding tensor
   │
   ├─ 3. 填充运行时配置
   │     posStart = ctx.kvCacheMgr.GetCurrentPos()
   │     ctx.tensorMgr.FillLLMConfig(
   │         configTensor,
   │         posStart, totalInputNum=1, validLen=posStart+1, batchSize=1)
   │
   ├─ 4. 填充 Attention Mask
   │     totalKvLen = posStart + 1   ← 已有 KV + 本步写入的 1 个位置
   │     ctx.tensorMgr.FillAttentionMask(
   │         maskTensor, seqLen=1, totalKvLen, batchSize=1)
   │
   ├─ 5. 获取 KV Cache tensor
   │     ctx.kvCacheMgr.GetTensors() → kvTensors
   │
   ├─ 6. 执行 Decode
   │     modelRunner_.RunDecode(lastTokenId, posStart, stepTensors, kvTensors)
   │     → executor.Execute(modelHandle, inputs)
   │     → NPU 对 1 个 token 做 decode 计算
   │
   │     ★ KV Cache 被扩展：
   │       keyCache[layer][posStart] ← 新 token 的 key 向量
   │       valueCache[layer][posStart] ← 新 token 的 value 向量
   │
   ├─ 7. 推进 KV Cache 位置
   │     ctx.kvCacheMgr.AdvancePos()
   │     → currentPos_++（10 → 11 → 12 → ...）
   │
   ├─ 8. 采样输出 token
   │     sampler_.Sample(logits, request.params.sampling)
   │     → 从 logits[0,0,:] 中选出 newTokenId
   │
   ├─ 9. Tokenizer 解码
   │     tokenText = tokenizer_.Decode({newTokenId})
   │     // 将 token id 解码为文本
   │
   ├─ 10. 记录输出
   │     request.outputTokenIds.push_back(newTokenId)
   │     request.outputText += tokenText
   │
   └─ 11. 回调通知（每 callbackFreq 个 token 回调一次）
        request.callbacks.onToken({text=tokenText, tokenId=newTokenId, tokenCount=N})
```

Decode 循环中 KV Cache 逐步增长：
```
Step 0: [██████████............] pos=10  ← Prefill 后
Step 1: [██████████████........] pos=11  ← 写入 "是" 的 KV
Step 2: [██████████████████....] pos=12  ← 写入 "一" 的 KV
  ...
Step N: [██████████████████████] pos=10+N ← 持续累积
```

#### 8.2.4 推理完成

```
when CheckStop() returns true:
│
├─ request.state: Decoding → Completed
│
├─ 结果组装
│     GenerateResult {
│       text = "我是一个AI助手...",
│       tokenIds = [3520, 3214, ...],
│       profiling = {totalTimeMs, prefillTimeMs, decodeTimeMs,
│                    inputTokenCount, outputTokenCount, cachedTokenCount},
│       status = Status::OK
│     }
│
├─ 回调通知
│     request.callbacks.onDone(result)
│     → future.set_value(result)  // 解除用户线程阻塞
│
└─ sessionState: Inferring → Idle
```

### 8.3 多轮对话（KV Cache 复用）

第二轮请求复用已有 KV Cache：

```cpp
llm::GenerateParams params2;
params2.promptText = "你能做什么？";
auto result2 = session->Generate(params2);
```

```
PrefillController::Run(request2)
│
├─ 1. tokenizer.Encode("你能做什么？") → newTokenIds (如 6 个 token)
│
├─ 2. ctx.kvCacheMgr.GetCurrentPos() → 52  ← 上一轮 prefill(10) + decode(42) 的累积
│
├─ 3. embedding.Run(newTokenIds) → embeddingTensor
│
├─ 4. ctx.tensorMgr.GetStepTensors(seqLen=6, batchSize=1)
│
├─ 5. FillAttentionMask:
│     seqLen=6, totalKvLen=52+6=58
│     → 新的 6 个 token 可以看到 58 个位置（52 已有 + 6 新增）
│
├─ 6. FillLLMConfig:
│     posStart=52, totalInputNum=6, validLen=58
│
├─ 7. ctx.kvCacheMgr.PreparePrefill(6)
│     → 在 pos=52 位置开始写入
│
├─ 8. modelRunner.RunPrefill(stepTensors, kvTensors)
│     → NPU 只计算新增 6 个 token，已有 52 个 token 的 KV 不重复计算
│
├─ 9. ctx.kvCacheMgr.CommitPrefill(6)
│     → currentPos_ = 58
│
└─ 10. sampler.Sample → firstToken
```

KV Cache 跨轮累积：
```
第一轮后:         [█████████████████████████........] pos=52  (10+42)
第二轮 Prefill后: [██████████████████████████████......] pos=58  (52+6)
第二轮 Decode后:  [██████████████████████████████████..] pos=58+N
```

关键：第二轮只计算新增 6 个 token 的 Prefill，已有 52 个 token 的 KV Cache 直接复用，不重新计算。

### 8.4 KV Cache 全生命周期管理

```
┌─────────────────────────────────────────────────────────────┐
│                KVCacheManager 状态变化                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Session 创建                                                │
│    Init(config)                                              │
│      → 分配 key/value tensor (每层各一份)                     │
│      → currentPos_ = 0, maxSeqLen_ = 4096                   │
│                                                             │
│  Generate #1 (Session: Idle → Inferring → Idle)             │
│    PreparePrefill(10)  → 记录区间 [0, 10)                    │
│    CommitPrefill(10)   → currentPos_ = 10                   │
│    AdvancePos() × 42  → currentPos_ = 52                    │
│                                                             │
│  Generate #2 (Session: Idle → Inferring → Idle)             │
│    PreparePrefill(6)   → 记录区间 [52, 58)                   │
│    CommitPrefill(6)    → currentPos_ = 58                   │
│    AdvancePos() × 30  → currentPos_ = 88                    │
│                                                             │
│  ExportKVCache() ← 仅 Session Idle 时允许                    │
│    → 序列化所有层的 KV 数据 + 元信息到 KVCacheSnapshot        │
│    → 用于保存对话状态                                        │
│                                                             │
│  ImportKVCache(snapshot) ← 仅 Session Idle 时允许           │
│    → 从快照恢复 KV 数据和位置                                 │
│    → 用于恢复之前的对话                                      │
│                                                             │
│  Expand(8192)                                                │
│    → 当 currentPos 接近 maxSeqLen 时触发                     │
│    → 重新分配更大的 KV tensor，拷贝旧数据                     │
│    → maxSeqLen_: 4096 → 8192                                │
│                                                             │
│  SetSparseMode(SparseMode::Attention)                        │
│    → 启用稀疏注意力                                          │
│    → GetSparseMask() 返回稀疏掩码，只保留重要 token 的注意力  │
│                                                             │
│  SwapOut()  [Suspend]                                        │
│    → KV Cache tensor 数据备份到 CPU 内存 (swappedData_)      │
│    → 释放 NPU tensor，回收 NPU 显存                          │
│                                                             │
│  SwapIn()   [Resume]                                         │
│    → 重新分配 NPU tensor                                     │
│    → 从 swappedData_ 恢复数据到 NPU                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 Prompt Cache

场景：固定 system prompt 前缀，多轮对话不重复计算。

```
session->SetPrefixPrompt("你是一个专业的助手，请用中文回答。")
│
├─ promptManager.SetPrefixPrompt(prefix)
│     ├─ tokenizer.Encode(prefix) → prefixTokenIds (如 15 个 token)
│     ├─ 临时 Generate 一次，计算 prefix 的 KV Cache
│     └─ 保存 prefixInfo_ = {tokenIds: [...], kvCacheLen: 15}

// 后续每次 Generate:
PrefillController::Run(request)
│
├─ promptManager.GetPrefixInfo() → prefixInfo (kvCacheLen=15)
│     // KV Cache 前 15 个位置已是 system prompt 的 KV
│     // 只需 Prefill 新增的用户输入
│
├─ kvCacheMgr.GetCurrentPos() → 15  ← 从 system prompt 的 KV 末尾继续
├─ PreparePrefill(userInputLen)     ← 只 Prefill 用户输入，不重复计算 prefix
└─ ... 其余同正常流程
```

场景：保存完整对话状态：

```
auto snapshot = session->ExportKVCache()
│
└─ kvCacheMgr.Export()
      → 序列化所有层 KV 数据 + {seqLen, numLayers, numKVHeads, headDim}
      → 返回 KVCacheSnapshot {data: vector<uint8_t>, seqLen=88, ...}

// 另一次会话中恢复:
session->ImportKVCache(snapshot)
│
└─ kvCacheMgr.Import(snapshot)
      → 反序列化，恢复所有层 KV 数据到 tensor
      → currentPos_ = snapshot.seqLen (88)
      → 对话可无缝继续
```

### 8.6 Batch Decode

```
session->GenerateBatchAsync(batchParams)
  → 创建 BatchInferRequest
  → scheduler.SubmitBatch()
     ═══ 工作线程 ═══
     → PrefillController::Run(request1)  → slot0
     → PrefillController::Run(request2)  → slot1
     → DecodeController::RunBatchWithStrategy(requests, strategy, scheduler)
        while (!strategy.ShouldStopDecode()):
          modelRunner.RunDecodeBatch([token0, token1])
          → 各slot独立采样 → checkStop
          → slot完成时: strategy.OnSlotFree() → 检查队列
```

### 8.7 挂起/恢复

```
engine->Suspend()
│
├─ 1. EngineState: Ready → Suspended (CAS 原子转换)
│
├─ 2. scheduler.Pause()
│     → 设置 paused_ = true
│     → 工作线程完成当前请求后不再取新请求
│     → 等待当前正在执行的 InferRequest 完成
│
├─ 3. 对每个 Session:
│     ├─ ctx.kvCacheMgr.SwapOut()
│     │     → 遍历 layers_，将 key/value tensor 的数据
│     │       拷贝到 swappedData_ (CPU 侧 vector<uint8_t>)
│     │     → 释放 NPU tensor
│     │
│     ├─ ctx.tensorMgr.ReleaseBuffers()
│     │     → 清空 tensorPool_ 和 embeddingPipeline_
│     │     → 释放输入/输出 tensor 占用的 NPU 内存
│     │
│     └─ modelRunner.FreeResource()
│           → modelLoader.Unload(modelHandle)
│           → 卸载 NPU 模型图
│
└─ 4. NPU 显存已清空，可给其他应用使用

engine->Resume()
│
├─ 1. 对每个 Session:
│     ├─ modelRunner.RestoreResource()
│     │     → modelLoader.Load(modelPath, batchSize) → 新 ModelHandle
│     │
│     ├─ ctx.tensorMgr.RebuildBuffers()
│     │     → 根据之前用过的 seqLen 重新分配输入/输出 tensor
│     │
│     └─ ctx.kvCacheMgr.SwapIn()
│           → 重新分配 NPU tensor
│           → 从 swappedData_ 恢复 KV Cache 数据
│           → currentPos_ 不变，对话可继续
│
├─ 2. scheduler.Resume()
│     → 设置 paused_ = false
│     → 工作线程恢复从队列取请求
│
└─ 3. EngineState: Suspended → Ready
```

恢复后 KV Cache 完整保留对话历史，Generate 可继续从 lastPos 开始。

### 8.8 全景数据流图

```
用户代码         API层              编排层                  组件层               平台HAL
───────         ──────             ──────                  ──────               ────────

Engine.Create(config)
  │                                Scheduler ←────────── PlatformFactory
  │                                                        .Create(Kirin)
  │                                │                       ↓
  │                                │                [ModelLoader]
  │                                │                [TensorAlloc]
  │                                │                [Executor]
  │                                │                [PerfCtrl]
  │                                │
  │                                ├─ Engine级共享组件:
  │                                │   ├─ ModelRunner ←── IModelLoader.Load()
  │                                │   │                  IExecutor
  │                                │   ├─ Tokenizer
  │                                │   ├─ Embedding  ←── ITensorAllocator
  │                                │   └─ Sampler
  │                                │
  │                                └─ PrefillController(modelRunner, tokenizer,
  │                                    embedding, sampler)
  │                                   DecodeController(modelRunner, embedding,
  │                                    tokenizer, sampler)
  │
session.Generate(params)          Session级组件:
  │                                ├─ KVCacheManager ←─ ITensorAllocator
  │                                ├─ TensorManager  ←─ ITensorAllocator
  │                                ├─ PromptManager(kvCacheMgr, tokenizer)
  │                                └─ SessionContext{kvCacheMgr, tensorMgr, promptMgr}
  │
  ├─ InferRequest ──→ Scheduler.Submit(request, ctx)
  │                     │
  │                     ├─ PrefillController.Run(request, ctx)
  │                     │    │
  │                     │    ├─ Tokenizer.Encode() → tokenIds
  │                     │    ├─ [分块循环] 对每个 chunk:
  │                     │    │   ├─ Embedding.Run(chunkIds) → embedTensor
  │                     │    │   ├─ ctx.tensorMgr.GetStepTensors() → stepTensors
  │                     │    │   ├─ ctx.tensorMgr.FillMask/Config()
  │                     │    │   ├─ ctx.kvCacheMgr.GetTensors() → kvTensors
  │                     │    │   ├─ ModelRunner.RunPrefill(step, kv)
  │                     │    │   │    └─ IExecutor.Execute() → NPU → logits
  │                     │    │   └─ ctx.kvCacheMgr.CommitPrefill()
  │                     │    ├─ Sampler.Sample(logits) → firstToken
  │                     │    └─ Tokenizer.Decode({firstToken}) → tokenText
  │                     │
  │                     ├─ DecodeController.RunSingle(request, ctx)
  │                     │    │
  │                     │    └─ while(!stop):
  │                     │         ├─ [useExternalEmbed?]
  │                     │         │   Embedding.Run({tokenId}) → embedTensor
  │                     │         ├─ ctx.tensorMgr.GetStepTensors(1,1)
  │                     │         ├─ ctx.tensorMgr.FillMask/Config()
  │                     │         ├─ ModelRunner.RunDecode(step, kv)
  │                     │         │    └─ IExecutor.Execute() → NPU → logits
  │                     │         ├─ ctx.kvCacheMgr.AdvancePos()
  │                     │         ├─ Sampler.Sample(logits) → tokenId
  │                     │         ├─ Tokenizer.Decode({tokenId}) → text
  │                     │         └─ callback.onToken()
  │                     │
  │                     └─ callback.onDone(result)
  │
  └─ return result ← future.get()
```

---

## 9. 扩展性：投机推理

不需要结构性调整。扩展方式为"新增Controller + 新增Component"。

| 层 | 改动 | 量级 |
|----|------|------|
| API | 无（SpeculativeConfig已预留） | 0 |
| 编排层 | 新增 SpeculativeDecodeController | 新增约1个类 |
| 编排层 | Scheduler路由加一个if分支 | 约5行 |
| 组件层 | 新增 SpecTree 组件 | 新增约1个类 |
| 组件层 | 其他组件不动 | 0 |
| 平台HAL | 不动 | 0 |
| 状态管理 | RequestState增加2个状态（Drafting, Verifying） | 约2行 |

SpeculativeDecodeController 管理 draft/verify 循环，使用两个 ModelRunner 实例（draft + target），各自持有独立的 KVCacheManager。SpecTree 管理树型/EAGLE变体的候选树。

投机推理流程：
```
SpeculativeDecodeController::Run(request, config)
  while (!CheckStop):
    DraftPhase():
      for i in 0..gamma:
        draftEmbedding.Run(lastToken)          → draftEmbed
        draftModelRunner.RunDecode(...)         → draftLogits
        sampler.Sample(draftLogits)             → draftToken
        draftKVCacheMgr.AdvancePos()
        draftTokens.push(draftToken)

    VerifyPhase():
      targetEmbedding.Run(draftTokens)          → targetEmbed
      targetModelRunner.RunPrefill(...)          → targetLogits（batch验证）
      对比 draftLogits 和 targetLogits → acceptCount
      targetKVCacheMgr.CommitVerify(acceptCount)
      清理 draftKVCacheMgr（回退被拒绝的位置）

    request.outputTokenIds 追加接受的tokens
```

---

## 10. 当前架构与新架构对比

| 维度 | 当前架构 | 新架构 |
|------|---------|--------|
| 总代码量 | 约43,800行 | 约10,200行 |
| C-API函数 | 110+个 | C++约15个方法 + C绑定约20-30个 |
| Context Set/Get | 80+个 | 5个分组结构体 |
| 回调Set函数 | 6个 | 1个GenerateCallbacks结构体 |
| InitOption Set | 7个 | 1个EngineConfig结构体 |
| God Class（>1000行） | 4个 | 0个 |
| 状态管理 | 10+处散落的 `atomic<bool>` | 3个显式状态机 |
| 平台耦合 | hiai::*散布在30+个文件 | 封装在platform/kirin/目录 |
| 异步路径 | 同步包装异步（线程+队列） | 异步优先，同步=wait |
| 可测试性 | 依赖NPU硬件 | Mock HAL可脱离NPU测试全流程 |
| 新增投机推理 | 修改5+个已有文件 | 新增2个类，不改已有代码 |
| 新增多模态 | 修改8+个已有文件 | 新增Controller + Component |
