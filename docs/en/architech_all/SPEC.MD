# LLM Engine Architecture Design - Design Specification

English | [简体中文](../../zh/architech_all/SPEC.MD)

## 1. Overview

### 1.1 Background

### 1.2 Design Decisions

| Decision item  | Conclusion                                  |
| -------------- | ------------------------------------------- |
| Refactoring strategy | Complete rewrite                       |
| Caller         | C/C++ application layer                     |
| Platform       | Kirin first + portable                      |
| Model scope    | Text-only LLM first, extensible to multimodal |
| Core feature (v1) | KV Cache optimization                    |
| Legacy API compatibility | Not needed                       |
| State management | Session layering (default single Session, multi-Session optional) |
| Async model    | Async first, sync = wait wrapper            |
| API language   | C++ API first + optional C binding          |
| Parameter style | Struct grouping + Optional extension       |
| Architecture scheme | Layered decoupled architecture          |

### 1.3 Design Goals

- Isolate all platform dependencies in the Platform HAL
- Replace scattered `atomic<bool>` flags with explicit state machines
- Support extensions (speculative decoding, multimodal) without modifying existing code

---

## 2. Architecture Overview

### 2.1 Layered Structure

```
+----------------------------------------------------------+
|  API Layer                                                |
|  Engine, Session, GenerateParams, GenerateResult          |
|  C++ API is the primary interface, C binding is a thin mapping layer |
+----------------------------------------------------------+
|  Orchestration Layer                                      |
|  Scheduler, PrefillController, DecodeController           |
|  Request scheduling, async queue, prefill/decode flow orchestration |
+----------------------------------------------------------+
|  Component Layer                                          |
|  Shared components (Engine level): ModelRunner, Tokenizer,|
|                        Sampler, Embedding                 |
|  Session components (Session level): KVCacheManager,      |
|                        TensorManager, PromptManager       |
+----------------------------------------------------------+
|  Platform HAL (Platform Hardware Abstraction Layer)       |
|  IModelLoader, ITensorAllocator, IExecutor, IPerfCtrl     |
|  Kirin implementation: hiai::IModelManager, INDTensorBuffer, FFRT |
|  Test implementation: MockModelLoader, MockTensorAllocator|
+----------------------------------------------------------+
```

Components are divided into two levels:
- **Engine level (shared)**: ModelRunner, Tokenizer, Embedding, Sampler — stateless or state reusable, shared across multiple Sessions
- **Session level (independent)**: KVCacheManager, TensorManager, PromptManager — each Session has its own KV Cache and tensor pool

Session-level components are bundled in the SessionContext struct; the orchestration layer's Controllers receive a SessionContext reference at Run time:

```cpp
namespace llm {

// Bundle struct of Session-level components, held by Session
struct SessionContext {
    KVCacheManager& kvCacheMgr;
    TensorManager& tensorMgr;
    PromptManager& promptMgr;
};

}  // namespace llm
```

### 2.2 Dependency Rules

- **One-way dependency**: upper layers depend on lower layers; lower layers must not depend on upper layers. When reverse notification is needed, use callbacks/observer pattern.
- **Components do not depend on each other directly**: components do not call each other; the orchestration layer coordinates their collaboration.
- **Platform HAL is the only platform coupling point**: all hiai::* calls are encapsulated in this layer.

### 2.3 Directory Structure

```
llm_engine/
+-- api/                            # API Layer
|   +-- engine.h/.cpp               # Engine class
|   +-- session.h/.cpp              # Session class
|   +-- generate_params.h           # GenerateParams and sub-structs
|   +-- generate_result.h           # GenerateResult, ProfilingData
|   +-- callbacks.h                 # GenerateCallbacks, event structs
|   +-- config.h                    # EngineConfig, SessionConfig, etc.
|   +-- c_binding/                  # C binding layer (optional)
|       +-- llm_engine_c.h/.cpp
|
+-- orchestration/                  # Orchestration Layer
|   +-- scheduler.h/.cpp            # Scheduler
|   +-- infer_request.h/.cpp        # InferRequest, state machine
|   +-- prefill_controller.h/.cpp   # PrefillController
|   +-- decode_controller.h/.cpp    # DecodeController, DecodeSlot
|   +-- schedule_strategy.h         # IScheduleStrategy interface
|
+-- component/                      # Component Layer
|   +-- model_runner.h/.cpp         # ModelRunner
|   +-- tensor_manager.h/.cpp       # TensorManager (general I/O tensor manager)
|   +-- kvcache_manager.h/.cpp      # KVCacheManager
|   +-- tokenizer.h/.cpp            # Tokenizer base class + factory
|   +-- sampler.h/.cpp              # Sampler
|   +-- embedding.h/.cpp            # Embedding
|   +-- prompt_manager.h/.cpp       # PromptManager
|   +-- tokenizer/                  # Tokenizer implementations
|       +-- bpe_tokenizer.h/.cpp
|       +-- spm_tokenizer.h/.cpp
|       +-- qwen_tokenizer.h/.cpp
|       +-- glm_tokenizer.h/.cpp
|
+-- platform/                       # Platform HAL
|   +-- hal_types.h                 # Tensor, TensorDesc, DataType, etc.
|   +-- i_model_loader.h            # Pure virtual interface
|   +-- i_tensor_allocator.h
|   +-- i_executor.h
|   +-- i_perf_controller.h
|   +-- platform_factory.h/.cpp     # Factory
|   +-- kirin/                      # Kirin platform implementation
|       +-- kirin_model_loader.h/.cpp
|       +-- kirin_tensor_allocator.h/.cpp
|       +-- kirin_executor.h/.cpp
|       +-- kirin_perf_controller.h/.cpp
|
+-- common/                         # Cross-layer common definitions
|   +-- status.h                    # Status enum
|   +-- types.h                     # RequestId and other type aliases
|   +-- log.h                       # Log macros
|
+-- CMakeLists.txt
+-- BUILD.gn
```

### 2.4 Code Volume Estimate

| Module | Header files | Source files | Estimated lines |
|------|--------|--------|----------|
| api/ | 7 | 6 | ~1,500 |
| orchestration/ | 5 | 5 | ~2,000 |
| component/ | 14 | 14 | ~5,000 |
| platform/ | 9 | 5 | ~1,500 |
| common/ | 3 | 0 | ~200 |
| **Total** | ~38 | ~30 | **~10,200** |

New architecture: approximately 10,200 lines.

---

## 3. API Layer

### 3.1 Engine

```cpp
namespace llm {

class Engine {
public:
    // Factory method to create the engine
    static std::expected<std::unique_ptr<Engine>, Status> Create(const EngineConfig& config);

    // Session management
    Session* DefaultSession();                                    // Convenience method: default Session
    std::expected<Session*, Status> CreateSession(const SessionConfig& config);
    Status DestroySession(Session* session);

    // Engine-level operations
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

Note: `DefaultSession()` returns the default Session automatically created by the engine, whose lifetime is managed by Engine. In simple scenarios, you can use it without explicitly creating a Session.

### 3.2 Session

```cpp
namespace llm {

class Session {
public:
    // Async inference (core interface)
    RequestId GenerateAsync(const GenerateParams& params, GenerateCallbacks callbacks);

    // Sync inference (internally calls GenerateAsync + wait)
    std::expected<GenerateResult, Status> Generate(const GenerateParams& params);

    // Batch inference
    RequestId GenerateBatchAsync(const BatchGenerateParams& batchParams, GenerateCallbacks callbacks);
    std::expected<BatchGenerateResult, Status> GenerateBatch(const BatchGenerateParams& batchParams);

    // Request control
    Status Cancel(RequestId id);
    Status Terminate();

    // KV Cache operations (only allowed in Idle state; returns Status::InvalidState in Inferring state)
    std::expected<KVCacheSnapshot, Status> ExportKVCache();
    Status ImportKVCache(const KVCacheSnapshot& snapshot);

    // Prompt Cache (only allowed in Idle state)
    Status SetPrefixPrompt(std::string prefix);
    void ClearPrefixPrompt();

    // Weight update
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

### 3.3 Request Parameter Structs

```cpp
namespace llm {

// Sampling config
struct SamplingConfig {
    bool doSample = false;
    int32_t topK = 50;
    float topP = 0.9f;
    float temperature = 0.7f;
    float repetitionPenalty = 1.0f;
    uint64_t seed = 0;
};

// Generation config
struct GenerationConfig {
    int32_t maxTokens = 1024;
    std::vector<std::string> stopSequences;
    int32_t callbackFreq = 1;       // Callback once every N tokens
};

// KV Cache config (Optional; uses default behavior if unset)
struct KVCacheConfig {
    bool sparseEnabled = false;
    SparseMode sparseMode = SparseMode::None;
    std::optional<SparseParams> sparseParams;
};

// Speculative config (Optional, future extension)
struct SpeculativeConfig {
    SpecType type = SpecType::Autoregressive;
    int32_t gamma = 5;                              // Number of candidate tokens
    float draftThreshold = 0.1f;
    std::optional<SpecTreeConfig> treeConfig;        // Specific to tree-based speculation
};

// Multimodal input (Optional, future extension)
struct MultimodalInput {
    // Defined when multimodal is supported
};

// Inference request params (aggregated)
struct GenerateParams {
    // Input
    std::string promptText;                                  // Text input
    std::optional<std::vector<int32_t>> promptTokenIds;      // Or pass token ids directly

    // Config
    SamplingConfig sampling;
    GenerationConfig generation;
    std::optional<KVCacheConfig> kvcache;

    // Advanced features (Optional; not enabled if unset)
    std::optional<SpeculativeConfig> speculative;            // Speculative inference
    std::optional<MultimodalInput> multimodal;               // Multimodal
};

// Batch inference params
struct BatchGenerateParams {
    std::vector<GenerateParams> items;
    BatchStrategy strategy = BatchStrategy::DecodeOnly;      // Batch only in Decode phase
};

// Engine config (passed in when creating Engine)
struct EngineConfig {
    std::string modelPath;               // Model path
    std::string weightPath;              // Weight path (optional, supports lazy loading)
    PlatformType platform = PlatformType::Kirin;
    PerfMode perfMode = PerfMode::Middle;
    std::optional<KirinPlatformConfig> kirinConfig;  // Kirin-specific params
};

// Session config
struct SessionConfig {
    int32_t batchSize = 1;
    int32_t maxSeqLen = 4096;
    bool kvcacheSwitchable = false;      // KV Cache swappable
};

// LoRA config
struct LoraConfig {
    std::string loraPath;                // File path
    const void* loraBuffer = nullptr;    // Or memory buffer
    size_t loraBufferSize = 0;
};

}  // namespace llm
```

### 3.4 Results and Callbacks

```cpp
namespace llm {

// Generation result
struct GenerateResult {
    std::string text;
    std::vector<int32_t> tokenIds;
    ProfilingData profiling;
    Status status;
};

// Batch generation result
struct BatchGenerateResult {
    std::vector<GenerateResult> items;
};

// Profiling data
struct ProfilingData {
    float totalTimeMs = 0;
    float prefillTimeMs = 0;
    float decodeTimeMs = 0;
    uint32_t inputTokenCount = 0;
    uint32_t outputTokenCount = 0;
    uint32_t cachedTokenCount = 0;
};

// Callback set
struct GenerateCallbacks {
    std::function<void(const TokenEvent&)> onToken;          // Streaming token callback
    std::function<void(const PrefillEvent&)> onPrefillDone;  // Prefill done
    std::function<void(const GenerateResult&)> onDone;       // All done
    std::function<void(Status)> onError;                     // Error
};

// Token event
struct TokenEvent {
    std::string text;
    int32_t tokenId;
    uint32_t tokenCount;      // Number of generated tokens
};

// Prefill event
struct PrefillEvent {
    float prefillTimeMs;
    uint32_t inputTokenCount;
};

}  // namespace llm
```

### 3.5 C Binding Layer (Optional)

The C binding layer is a thin mapping layer, provided only for callers that need a C ABI (e.g., HarmonyOS NAPI). It wraps C++ API calls one-to-one with no additional logic.

```c
extern "C" {

LLMEngine* LLMEngine_Create(const LLMEngineConfig* config);
LLMStatus LLMEngine_Destroy(LLMEngine* engine);
LLMStatus LLMEngine_Generate(LLMEngine* engine, LLMSession* session,
                              const LLMGenerateParams* params, LLMGenerateResult* outResult);
// ... only the interfaces that need to be exposed externally, about 20-30 functions

}  // extern "C"
```

---

## 4. Orchestration Layer

### 4.1 Scheduler

```cpp
namespace llm::orchestration {

struct SchedulerConfig {
    int32_t maxQueueSize = 64;
    int32_t maxDecodeSlots = 1;    // Number of batches decoding simultaneously
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler();

    // Submit async request (core entry)
    RequestId Submit(std::shared_ptr<InferRequest> request);

    // Batch submit async request
    RequestId SubmitBatch(std::shared_ptr<BatchInferRequest> batchRequest);

    Status Cancel(RequestId id);
    Status Shutdown();

    // Lifecycle control (used by Suspend/Resume)
    Status Pause();            // Stop taking new requests from the queue, wait for current request to complete
    Status Resume();           // Resume taking requests from the queue

    // Schedule strategy access interface
    std::shared_ptr<InferRequest> TryDequeue();
    bool IsQueueEmpty() const;
    bool HasActiveDecode() const;
    bool HasFreeSlot() const;
    void PrefillForSlot(std::shared_ptr<InferRequest> request, int32_t slot);

private:
    void RunLoop();                                     // Worker thread main loop
    std::shared_ptr<InferRequest> PickNext();           // Pick next request
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

// Inference request object, spans the entire inference lifecycle
class InferRequest {
public:
    RequestId id;
    GenerateParams params;
    GenerateCallbacks callbacks;

    // Request state machine
    enum class State : uint8_t {
        Pending,       // Waiting in queue
        Prefilling,    // Prefilling
        Decoding,      // Decoding
        Drafting,      // Speculative inference: draft phase
        Verifying,     // Speculative inference: verify phase
        Completed,     // Completed
        Cancelled,     // Cancelled
        Failed,        // Failed
    };
    std::atomic<State> state{State::Pending};

    // Intermediate data during inference
    std::vector<int32_t> inputTokenIds;       // Tokenized input
    std::vector<int32_t> outputTokenIds;      // Generated tokens
    std::string outputText;                    // Generated text

    // Profiling data
    ProfilingData profiling;
    Status errorStatus;
};

// Batch request: merge multiple InferRequests for batch decode
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

    // Execute Prefill (synchronous, called in the Scheduler worker thread)
    // SessionContext is passed in on each call, containing references to that Session's independent components
    Status Run(InferRequest& request, SessionContext& ctx);

private:
    ModelRunner& modelRunner_;
    Tokenizer& tokenizer_;
    Embedding& embedding_;
    Sampler& sampler_;
};

}  // namespace llm::orchestration
```

Internal flow of PrefillController::Run:
1. tokenizer_.Encode(request.params.promptText) → request.inputTokenIds
2. promptManager.GetPrefixInfo() → prefixKVLen (may be nullopt)
3. Chunk the input tokens, execute the Prefill loop per chunk (see 8.2.2 Chunked Prefill)
4. Sample firstToken, tokenizer_.Decode({firstToken}) → tokenText
5. request.outputTokenIds.push_back(firstToken)
6. request.callbacks.onPrefillDone(...)

### 4.4 DecodeController

```cpp
namespace llm::orchestration {

class DecodeController {
public:
    DecodeController(ModelRunner& modelRunner, Embedding& embedding,
                     Tokenizer& tokenizer, Sampler& sampler);

    // Single-batch Decode loop
    Status RunSingle(InferRequest& request, SessionContext& ctx);

    // Multi-batch Decode loop with strategy
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
        SessionContext* ctx;              // Each slot binds its own SessionContext
        SlotState state;  // Active, Completed, Empty
    };
};

}  // namespace llm::orchestration
```

### 4.5 Scheduling Strategy

```cpp
namespace llm::orchestration {

class IScheduleStrategy {
public:
    virtual ~IScheduleStrategy() = default;

    // Called when a slot frees up in Decode
    // Return nullopt to not fill; return InferRequest to fill with that request
    virtual std::optional<std::shared_ptr<InferRequest>> OnSlotFree(
        int32_t freedSlot, int32_t totalSlots, Scheduler& scheduler) = 0;

    // Called when a new request arrives
    // Return true to process immediately, false to queue
    virtual bool OnRequestArrived(
        std::shared_ptr<InferRequest> request, Scheduler& scheduler) = 0;

    // Determine whether Decode should stop
    virtual bool ShouldStopDecode(
        const std::vector<SlotState>& slotStates, Scheduler& scheduler) = 0;
};

// Simple strategy: first collect a full batch of requests, prefill uniformly, decode uniformly, do not refill freed slots
class SimpleBatchStrategy : public IScheduleStrategy { /* ... */ };

// Continuous batching strategy: when a slot frees up, check whether the queue has pending requests; if so, prefill and fill into the slot
class ContinuousBatchStrategy : public IScheduleStrategy { /* ... */ };

}  // namespace llm::orchestration
```

---

## 5. Component Layer

### 5.1 ModelRunner

```cpp
namespace llm::component {

class ModelRunner {
public:
    explicit ModelRunner(platform::IModelLoader& loader, platform::IExecutor& executor);
    ~ModelRunner();

    // Initialize: load model graph and weights
    Status Init(const ModelRunnerConfig& config);

    // Execute Prefill
    Status RunPrefill(const StepTensors& stepTensors,
                      const std::vector<platform::Tensor>& kvCacheTensors);

    // Execute single-batch Decode
    Status RunDecode(int32_t tokenId, int32_t posStart,
                     const StepTensors& stepTensors,
                     const std::vector<platform::Tensor>& kvCacheTensors);

    // Execute multi-batch Decode
    Status RunDecodeBatch(const std::vector<int32_t>& tokenIds,
                          const std::vector<int32_t>& posStarts,
                          const StepTensors& stepTensors,
                          const std::vector<platform::Tensor>& kvCacheTensors);

    // Weight management
    Status UpdateWeight(const std::string& name, const void* data, size_t size, size_t offset);
    Status UpdateLora(const LoraConfig& config);

    // Performance mode
    Status SetPerfMode(PerfMode mode);

    // Resource control
    Status FreeResource();
    Status RestoreResource();

private:
    platform::IModelLoader& modelLoader_;
    platform::IExecutor& executor_;

    // Model instances (grouped by batch; may hold both batch=1 and batch=N instances)
    struct ModelInstance {
        platform::ModelHandle handle;
        int32_t batchSize;
    };
    std::vector<ModelInstance> models_;

    // Mapping of TensorRole → NPU model graph input order
    // Loaded from model metadata or config at Init; describes the order of model graph input tensors
    // e.g. a model's input order is [PositionIds, InputIds, AttentionMask, LLMConfig, KV0, KV1, ...]
    // then inputOrder_ = {PositionIds, InputIds, AttentionMask, LLMConfig, KVCacheKey, ...}
    std::vector<TensorRole> inputOrder_;
};

struct ModelRunnerConfig {
    std::string modelPath;
    std::string weightPath;
    int32_t maxSeqLen = 4096;
    bool initWithoutWeight = false;     // Supports lazy weight loading
    bool supportBatchDecode = false;
    int32_t batchDecodeSize = 1;
    // NPU model graph input order (different model graphs have different input tensor arrangements)
    // If unset, Init attempts to auto-infer from model metadata
    std::optional<std::vector<TensorRole>> inputOrder;
};

}  // namespace llm::component
```

### 5.2 TensorManager

Manages the lifecycle of non-KV-Cache input/output tensors. **Clear division of labor with KVCacheManager**: TensorManager manages "temporary tensors refilled each step" (inputIds, attentionMask, logits, etc.), while KVCacheManager manages "state tensors accumulated across steps" (KV Cache).

**Design decision**: Option B — KVCacheManager independently owns KV tensors, TensorManager manages only non-KV tensors. Rationale:
- KV Cache is a state accumulated across steps, with a different lifecycle from temporary tensors refilled each step
- KV Cache's Export/Import/SwapOut/SwapIn are unique operations, cleaner when self-contained in one component
- Both components hold their own ITensorAllocator reference, allocating independently, making it easier to track memory usage separately

```cpp
namespace llm::component {

// Tensor semantic role identifier
enum class TensorRole : uint16_t {
    // Input - general
    InputIds,
    InputEmbedding,
    AttentionMask,
    PositionIds,
    MropeSin,
    MropeCos,
    // Input - runtime config
    LLMConfig,
    OutputPos,
    // Input - sparse/speculative/MoE
    HistoryActivationMask,
    EagleFeature,
    MoeMapInfo,
    // Output
    Logits,
    HiddenState,
    Query,
};

// Set of tensors needed for a single inference step (excluding KV Cache, which is provided by KVCacheManager)
struct StepTensors {
    std::unordered_map<TensorRole, std::vector<platform::Tensor>> inputs;
    std::unordered_map<TensorRole, std::vector<platform::Tensor>> outputs;
};

class TensorManager {
public:
    explicit TensorManager(platform::ITensorAllocator& allocator);
    ~TensorManager();

    // ========== Initialization ==========

    // Allocate tensor pool according to model params config
    Status Init(const TensorManagerConfig& config);

    // ========== Get Tensor ==========

    // Get the input/output tensor set for the specified seqLen + batch
    // Allocation triggered on first request for this seqLen; subsequent calls return cached directly
    std::expected<StepTensors, Status> GetStepTensors(int32_t seqLen, int32_t batchSize);

    // Get the next embedding input buffer (supports Prefill pipeline double buffering)
    std::expected<platform::Tensor, Status> GetNextEmbeddingInput(int32_t seqLen);

    // ========== Data Fill Helpers ==========

    // Fill the runtime config tensor (posStart, totalInputNum, validLen, etc.)
    Status FillLLMConfig(const platform::Tensor& configTensor,
                         int32_t posStart, int32_t totalInputNum,
                         int32_t validLen, int32_t batchSize);

    // Fill attention mask
    // totalKvLen is the total KV length for this inference:
    //   Prefill: existingKV + seqLen (existing KV + new this round)
    //   Decode:  existingKV + 1 (existing KV + 1 position written this step)
    Status FillAttentionMask(const platform::Tensor& maskTensor,
                            int32_t seqLen, int32_t totalKvLen,
                            int32_t batchSize);

    // ========== Resource Management ==========

    // Release all tensor buffers (Suspend)
    Status ReleaseBuffers();

    // Rebuild all tensor buffers (Resume)
    Status RebuildBuffers();

    // Get current total tensor memory usage (excluding KV Cache)
    size_t GetMemoryUsage() const;

private:
    platform::ITensorAllocator& allocator_;

    // Tensor cache pool indexed by seqLen
    std::unordered_map<int32_t, StepTensors> tensorPool_;
    std::mutex poolMutex_;

    // Prefill embedding double buffer (2 per seqLen)
    std::unordered_map<int32_t,
        std::array<platform::Tensor, 2>> embeddingPipeline_;
    std::unordered_map<int32_t, size_t> pipelineIndex_;  // Which one is currently in use

    TensorManagerConfig config_;
    bool initialized_ = false;
    bool buffersFreed_ = false;
};

struct TensorManagerConfig {
    // Model params
    int32_t hiddenSize = 4096;
    int32_t vocabSize = 32000;
    int32_t vocabAlignSize = 32000;
    int32_t maxSeqLen = 4096;       // AttentionMask's second dimension is allocated per maxSeqLen; at runtime actualLen ≤ maxSeqLen
    int32_t batchSize = 1;

    // Prefill config
    std::vector<int32_t> prefillBlockSeqLens;  // e.g. {32, 64, 128, 256, 512}
    int32_t decodeBlockSeqLen = 1;

    // Feature switches
    bool useExternalEmbedding = false;
    bool useMrope = false;
    bool enableSparse = false;
    bool enableLora = false;
    int32_t mhcNumStream = 1;
    int32_t prefillPipelineCount = 2;  // Prefill embedding buffer count
};

}  // namespace llm::component
```

#### Collaboration between TensorManager and KVCacheManager

The two components do not interact directly; the orchestration layer coordinates them:

```
PrefillController::Run(request):
  // Get tensors from the two components separately
  auto stepTensors = tensorMgr.GetStepTensors(seqLen, batchSize);
  auto& kvTensors  = kvCacheMgr.GetTensors();  // KV Cache tensor

  // Fill data (kvCacheLen obtained from KVCacheManager)
  tensorMgr.FillAttentionMask(stepTensors, seqLen, kvCacheMgr.GetCurrentPos(), batch);
  tensorMgr.FillLLMConfig(stepTensors, posStart, totalInputNum, validLen, batch);

  // Merge and pass to ModelRunner
  modelRunner.RunPrefill(stepTensors, kvTensors, ...);
```

#### Comparison with current implementation

| Dimension | Current implementation | New design |
|------|---------|--------|
| Allocation timing | Allocate all seqLen variants at Init | Allocate on demand on first request, reducing startup memory peak |
| Tensor semantic identifier | Enumerate by InputIndex position | Enumerate by TensorRole, decoupling from model graph input order |
| KV Cache management | Scattered in TensorManager + ModelExecutor | KVCacheManager owns it independently, TensorManager does not touch it |
| Embedding pipeline | Hard-coded 2 buffers | prefillPipelineCount configurable |
| External KV Cache | Global pointer + batch switching | Handled by KVCacheManager, not through TensorManager |

### 5.3 KVCacheManager

This is a core feature of v1.

```cpp
namespace llm::component {

class KVCacheManager {
public:
    explicit KVCacheManager(platform::ITensorAllocator& allocator);
    ~KVCacheManager();

    // ========== Lifecycle ==========

    // Initialize KV Cache (allocate space according to model params)
    Status Init(const KVCacheConfig& config);

    // ========== Prefill related ==========

    // Prepare KV Cache positions for Prefill
    Status PreparePrefill(int32_t seqLen);

    // Update KV Cache valid length after Prefill completes
    Status CommitPrefill(int32_t actualLen);

    // ========== Decode related ==========

    // Get the current used position of KV Cache (decode continues writing from here)
    int32_t GetCurrentPos() const;

    // KV Cache position +1 after one Decode step
    void AdvancePos();

    // ========== Export/Import ==========

    // Export KV Cache snapshot (for Prompt Cache / device-cloud collaboration)
    std::expected<KVCacheSnapshot, Status> Export() const;

    // Import KV Cache snapshot
    Status Import(const KVCacheSnapshot& snapshot);

    // ========== Sparse KV Cache ==========

    // Set sparse mode
    Status SetSparseMode(SparseMode mode);

    // Get sparse mask
    std::expected<platform::Tensor, Status> GetSparseMask(int32_t seqLen) const;

    // ========== Dynamic Expansion ==========

    // Expand KV Cache capacity
    Status Expand(int32_t newMaxSeqLen);

    // ========== Tensor Access ==========

    // Get KV Cache tensor (for orchestration layer to pass to ModelRunner)
    const std::vector<KVCacheLayer>& GetTensors() const;

    // ========== Resource Management ==========

    // Swap out memory (suspend scenario)
    Status SwapOut();

    // Swap in memory (resume scenario)
    Status SwapIn();

    // Get current KV Cache memory usage
    size_t GetMemoryUsage() const;

private:
    platform::ITensorAllocator& allocator_;

    // KV Cache tensor (key + value, one each per layer)
    struct KVCacheLayer {
        Tensor keyCache;
        Tensor valueCache;
    };
    std::vector<KVCacheLayer> layers_;

    int32_t maxSeqLen_ = 0;
    int32_t currentPos_ = 0;
    int32_t numLayers_ = 0;
    int32_t numKVHeads_ = 0;        // In GQA, numKVHeads < numHeads
    int32_t headDim_ = 0;

    // Sparse related
    SparseMode sparseMode_ = SparseMode::None;
    std::unique_ptr<SparseKVCacheHelper> sparseHelper_;

    // Swapped-out cache
    std::vector<uint8_t> swappedData_;
};

struct KVCacheConfig {
    int32_t maxSeqLen = 4096;
    int32_t numLayers = 32;
    int32_t numKVHeads = 32;
    int32_t headDim = 128;
    bool switchable = false;        // Whether swappable
    int32_t dtypeSize = 2;          // fp16=2, fp32=4
};

struct KVCacheSnapshot {
    std::vector<uint8_t> data;      // Serialized KV Cache data
    int32_t seqLen;                 // Valid length
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

    // Factory method
    static std::expected<std::unique_ptr<Tokenizer>, Status>
        Create(const std::string& modelPath, TokenizerType type);

    // Text → token ids
    std::expected<std::vector<int32_t>, Status> Encode(const std::string& text);

    // token ids → text
    std::expected<std::string, Status> Decode(const std::vector<int32_t>& ids);

    // Get vocab info
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
    // Sample a token from logits
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

    // Initialize: load embedding weights
    Status Init(const std::string& modelPath);

    // token ids → embedding tensor
    std::expected<Tensor, Status> Run(const std::vector<int32_t>& tokenIds);

    // Whether external embedding is supported (some models allow skipping the internal embedding layer)
    bool SupportsExternalEmbedding() const;

private:
    platform::ITensorAllocator& allocator_;
    Tensor embeddingWeight_;      // Vocab weight matrix
    bool supportsExternal_ = false;
};

}  // namespace llm::component
```

### 5.7 PromptManager

Manages prefix prompt caching and historical Prompt Cache. **Session-level component**, owned independently by each Session, holding a reference to the same Session's KVCacheManager.

```cpp
namespace llm::component {

class PromptManager {
public:
    explicit PromptManager(KVCacheManager& kvCacheMgr, Tokenizer& tokenizer);
    ~PromptManager();

    // Set prefix prompt (e.g., system prompt)
    // Internally calls tokenizer.Encode + KVCacheManager to execute one Prefill, writing the prefix into the KV Cache
    Status SetPrefixPrompt(const std::string& prefix);
    void ClearPrefixPrompt();

    // Get the token ids and KV Cache length corresponding to the prefix prompt
    struct PrefixInfo {
        std::vector<int32_t> tokenIds;
        int32_t kvCacheLen;  // KV length already cached for the prefix
    };
    std::optional<PrefixInfo> GetPrefixInfo() const;

    // Save the current prompt's KV Cache (Prompt Cache)
    Status SavePromptCache(const std::string& promptKey);

    // Load a previously saved prompt cache
    std::expected<int32_t, Status> LoadPromptCache(const std::string& promptKey);

private:
    KVCacheManager& kvCacheMgr_;    // KVCacheManager of the same Session
    Tokenizer& tokenizer_;          // Engine-level shared
    std::optional<PrefixInfo> prefixInfo_;

    // Prompt Cache storage
    struct PromptCacheEntry {
        std::string key;
        std::vector<uint8_t> kvCacheData;
        int32_t seqLen;
    };
    std::unordered_map<std::string, PromptCacheEntry> promptCache_;
};

}  // namespace llm::component
```

### 5.8 Component Interaction Rules

Components never call each other directly. The orchestration layer coordinates component collaboration by passing intermediate data through local variables (tensor, id, pos).

**Engine-level components** (ModelRunner, Tokenizer, Embedding, Sampler) are shared by all Sessions.
**Session-level components** (KVCacheManager, TensorManager, PromptManager) are bundled in SessionContext and passed into the Controller's Run method on each inference.
Controllers bind only Engine-level component references at construction, and receive SessionContext at Run time.

TensorManager and KVCacheManager do not interact directly; the orchestration layer obtains tensors from each and assembles them to pass to ModelRunner.

Prefill flow example (inside the Controller):
1. tokenizer_.Encode(prompt) → tokenIds
2. ctx.promptMgr.GetPrefixInfo() → prefixKVLen
3. Chunk the input, for each chunk:
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

## 6. Platform HAL

### 6.1 Basic Types

```cpp
namespace llm::platform {

// Tensor description
struct TensorDesc {
    std::vector<int32_t> shape;       // e.g. [batch, seq, hidden]
    DataType dtype;                   // FP16, FP32, INT8, INT4
    size_t GetElementCount() const;
    size_t GetByteSize() const;
};

enum class DataType : uint8_t { FP32, FP16, INT8, INT4, INT32 };

// Tensor (wraps the underlying buffer; upper layers are unaware of the specific allocation method)
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

// Model handle (opaque)
using ModelHandle = void*;

// Execution result
struct ExecuteResult {
    Status status;
    std::vector<Tensor> outputs;
};

// Performance mode
enum class PerfMode : uint8_t { Low, Middle, High, ExtremeHigh };

}  // namespace llm::platform
```

### 6.2 Core Interfaces

```cpp
namespace llm::platform {

// Model loader
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // Load model (batch dimension determined at this point)
    virtual std::expected<ModelHandle, Status> Load(
        const std::string& modelPath, int32_t batchSize) = 0;

    // Load graph structure only (lazy load weights)
    virtual std::expected<ModelHandle, Status> LoadGraph(
        const std::string& modelPath, int32_t batchSize) = 0;

    // Load weights into an existing graph
    virtual Status LoadWeight(ModelHandle handle, const std::string& weightPath) = 0;

    // Unload model
    virtual Status Unload(ModelHandle handle) = 0;
};

// Tensor allocator
class ITensorAllocator {
public:
    virtual ~ITensorAllocator() = default;

    virtual std::expected<Tensor, Status> Allocate(const TensorDesc& desc) = 0;
    virtual std::expected<std::vector<Tensor>, Status> AllocateBatch(
        const std::vector<TensorDesc>& descs) = 0;
    virtual size_t GetMemoryUsage() const = 0;
    virtual void Reset() = 0;
};

// Model executor
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

// Performance controller
class IPerfController {
public:
    virtual ~IPerfController() = default;

    virtual Status SetPerfMode(PerfMode mode) = 0;
    virtual Status BindCPU(int32_t coreId) = 0;
    virtual PerfMode GetCurrentMode() const = 0;
};

}  // namespace llm::platform
```

### 6.3 Platform Factory

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
    Mock,       // For testing
    // Future: Cpu, Gpu, ...
};

}  // namespace llm::platform
```

### 6.4 Kirin Platform Implementation

Uses hiai::IModelManager, hiai::INDTensorBuffer, hiai::PerfModeManager, FFRT, lm_engine_mmgr to implement IModelLoader, ITensorAllocator, IExecutor, IPerfController. All hiai::* header includes are only in the platform/kirin/ directory.

### 6.5 Mock Implementation (For Testing)

Implements HAL interfaces without depending on NPU hardware, for full-pipeline unit testing.

---

## 7. Error Handling and State Management

### 7.1 Error Handling

Uses `std::expected<T, Status>` + status codes, no exceptions (NPU/embedded scenarios are not suitable for exceptions).

```cpp
namespace llm {

enum class Status : int32_t {
    OK = 0,

    // General errors
    InvalidArgument,
    OutOfMemory,
    InternalError,

    // Model related
    ModelLoadFailed,
    ModelNotInitialized,
    ModelExecuteFailed,
    WeightUpdateFailed,

    // KV Cache related
    KVCacheAllocFailed,
    KVCacheExportFailed,
    KVCacheImportFailed,
    KVCacheExpandFailed,

    // Tokenizer related
    TokenizerInitFailed,
    EncodeFailed,
    DecodeFailed,

    // Request related
    RequestCancelled,
    RequestQueueFull,
    InvalidRequest,

    // Platform related
    PlatformNotSupported,
    PlatformInitFailed,
};

}  // namespace llm
```

Usage:
```cpp
// Operations whose return value may fail use std::expected
auto engine = Engine::Create(config);          // std::expected<unique_ptr<Engine>, Status>
if (!engine) { /* handle engine.error() */ }

auto result = session->Generate(params);        // std::expected<GenerateResult, Status>
if (!result) { /* handle result.error() */ }

// Operations that cannot fail return a value directly
int32_t pos = kvCacheMgr.GetCurrentPos();       // Direct return
```

### 7.2 State Management

Uses explicit state machines to replace scattered `atomic<bool>` flags.

**Engine states:**
```
Uninitialized → Ready          (initialization succeeded)
Ready → Suspended              (suspend)
Suspended → Ready              (resume)
Ready → Shutdown               (destroy)
Suspended → Shutdown           (destroy)
```

**Session states:**
```
Idle → Inferring               (GenerateAsync submitted)
Inferring → Idle               (inference complete/failed)
Inferring → Cancelled          (cancelled)
Cancelled → Idle               (cleanup complete)
Any → Destroyed                (DestroySession)
```

Session operation constraints:
- ExportKVCache / ImportKVCache / SetPrefixPrompt: only allowed in Idle state; calling in Inferring state returns Status::InvalidState
- GenerateAsync / Generate: only allowed in Idle state
- Cancel: only allowed in Inferring state

**Request states:**
```
Pending → Prefilling           (Scheduler starts processing)
Prefilling → Decoding          (Prefill complete)
Decoding → Completed           (normal completion)
Decoding → Cancelled           (cancelled)
Any → Failed                   (error)
```

State transitions use `compare_exchange_strong` to guarantee atomicity and prevent race conditions.

---

## 8. Key Flows

### 8.1 Engine Initialization (Model Loading)

```
Engine::Create(engineConfig)
│
├─ 1. PlatformFactory::Create(config.platform)    // e.g. PlatformType::Kirin
│     → Returns PlatformInstances:
│       ├── IModelLoader      (KirinModelLoader: wraps hiai::IModelManager)
│       ├── ITensorAllocator  (KirinTensorAllocator: wraps NPU memory allocation)
│       ├── IExecutor         (KirinExecutor: wraps hiai::IModel::Execute)
│       └── IPerfController   (KirinPerfController: wraps frequency scaling/CPU binding)
│
├─ 2. Create Engine-level shared components (shared across Sessions)
│     ├── ModelRunner(modelLoader, executor)
│     │     └─ modelRunner.Init(modelRunnerConfig)
│     │         ├─ modelLoader.Load(modelPath, batchSize=1)     → ModelHandle
│     │         └─ modelLoader.Load(modelPath, batchSize=N)     → ModelHandle (if batch is supported)
│     │
│     ├── Tokenizer::Create(modelPath, config.tokenizerType)    → Factory creates concrete subclass
│     ├── Embedding(tensorAllocator)
│     │     └─ embedding.Init(modelPath)                        → Load embedding weight matrix
│     └── Sampler()
│
├─ 3. Create orchestration layer (Controllers bind only Engine-level components; Session-level components are passed in at Run time)
│     ├── Scheduler(schedulerConfig)
│     │     └─ Start worker thread RunLoop()
│     ├── PrefillController(modelRunner, tokenizer, embedding, sampler)
│     └── DecodeController(modelRunner, embedding, tokenizer, sampler)
│
├─ 4. Create default Session (Session-level components are created here)
│     └─ DefaultSession()
│         └─ new Session(id, scheduler, engineSharedComponents)
│             ├─ Create independent KVCacheManager(tensorAllocator)
│             │     └─ kvCacheMgr.Init(kvCacheConfig)
│             │         → Per numLayers × numKVHeads × headDim × maxSeqLen
│             │           allocate key/value cache tensor for each layer
│             ├─ Create independent TensorManager(tensorAllocator)
│             │     └─ tensorMgr.Init(tensorManagerConfig)  // Only records config, does not allocate
│             ├─ Create independent PromptManager(kvCacheMgr, tokenizer)
│             └─ Assemble SessionContext { kvCacheMgr, tensorMgr, promptMgr }
│
└─ 5. EngineState: Uninitialized → Ready
```

Key points:
- Model loading is done via ModelRunner → IModelLoader → KirinModelLoader; KirinModelLoader internally calls hiai::IModelManager::LoadFromModelBuffer to load the NPU model graph + weights
- Engine-level components (ModelRunner, Tokenizer, Embedding, Sampler) are shared by all Sessions
- Session-level components (KVCacheManager, TensorManager, PromptManager) are independent per Session, bundled in SessionContext
- TensorManager does not allocate concrete tensors at this point; it only records config and allocates on demand later

### 8.2 First Inference (Synchronous Generate)

User calls:
```cpp
llm::GenerateParams params;
params.promptText = "Hello, please introduce yourself";
params.sampling = {.doSample = true, .topK = 50, .topP = 0.9f, .temperature = 0.7f};
params.generation = {.maxTokens = 1024, .callbackFreq = 1};
auto result = session->Generate(params);
```

#### 8.2.1 Request Submission

```
Session::Generate(params)
│
├─ Create InferRequest
│     ├── id = nextRequestId++
│     ├── params = user-passed GenerateParams
│     ├── callbacks = { onDone: [promise](result) { promise.set_value(result); } }
│     ├── state = Pending
│     ├── inputTokenIds = []
│     ├── outputTokenIds = []
│     └── outputText = ""
│
├─ sessionState: Idle → Inferring
│
├─ scheduler.Submit(request)
│     └─ Enqueue queue_, notify queueCv_
│
└─ return future.get()   // Block waiting for result
```

#### 8.2.2 Prefill Phase (Scheduler Worker Thread)

Prefill uses a **chunking strategy**: long inputs are split into multiple chunks according to the `prefillBlockSeqLens` config and executed step by step. Each chunk independently executes one model inference, progressively filling the KV Cache. Benefits of chunking:
- Avoids allocating an oversized tensor at once when seqLen is large, preventing OOM
- Supports CPU/NPU pipeline double buffering (embedding lookup parallel with NPU computation)
- Chunk size matches the NPU's optimal computation granularity

```
Scheduler::ExecuteRequest(request)
│
├─ request.state: Pending → Prefilling
│
└─ PrefillController::Run(request, ctx)
   │
   ├─ 1. Tokenizer encode
   │     tokenizer_.Encode(request.params.promptText)
   │     → "Hello, please introduce yourself" → [3546, 898, ...] = request.inputTokenIds
   │
   ├─ 2. Prefix prompt check
   │     ctx.promptMgr.GetPrefixInfo()
   │     → If a system prompt was set previously, returns the cached tokenIds and KV position
   │     → First time here, returns nullopt
   │
   ├─ 3. Input chunking
   │     Split inputTokenIds into chunks per prefillBlockSeqLens
   │     e.g. input 2000 tokens, config {32, 64, 128, 256, 512}:
   │       chunk[0]: tokens[0:32],   chunkSeqLen=32
   │       chunk[1]: tokens[32:96],  chunkSeqLen=64
   │       chunk[2]: tokens[96:224], chunkSeqLen=128
   │       chunk[3]: tokens[224:480],chunkSeqLen=256
   │       chunk[4]: tokens[480:992],chunkSeqLen=512
   │       chunk[5]: tokens[992:2000],chunkSeqLen=1008  (remainder)
   │
   └─ 4. Execute Prefill loop for each chunk:
      │
      │  posStart = ctx.kvCacheMgr.GetCurrentPos()   ← start position where this chunk writes to KV
      │  existingKV = posStart                         ← existing KV length
      │
      ├─ 4a. Embedding lookup
      │     embedding_.Run(chunkTokenIds)
      │     → Lookup from embeddingWeight_ matrix by token id
      │     → Returns embeddingTensor: shape=[1, chunkSeqLen, hiddenSize]
      │
      ├─ 4b. Get non-KV input/output tensors
      │     ctx.tensorMgr.GetStepTensors(chunkSeqLen, batchSize=1)
      │     → First request for this seqLen, triggers on-demand allocation:
      │       ├── InputIds tensor      [1, chunkSeqLen]
      │       ├── AttentionMask tensor [1, 1, chunkSeqLen, maxSeqLen]
      │       ├── PositionIds tensor   [1, chunkSeqLen]
      │       ├── LLMConfig tensor     [1, N]
      │       ├── OutputPos tensor     [1]
      │       └── Logits tensor        [1, chunkSeqLen, vocabAlignSize]
      │
      ├─ 4c. Fill Attention Mask
      │     totalKvLen = existingKV + chunkSeqLen   ← existing KV + new this round
      │     ctx.tensorMgr.FillAttentionMask(
      │         stepTensors.inputs[TensorRole::AttentionMask],
      │         chunkSeqLen, totalKvLen, batchSize=1)
      │     → Causal mask: position i in the current chunk can see range [0, existingKV+i]
      │     → First chunk, existingKV=0, totalKvLen=chunkSeqLen, is a lower-triangular causal mask
      │
      ├─ 4d. Fill runtime config
      │     ctx.tensorMgr.FillLLMConfig(
      │         stepTensors.inputs[TensorRole::LLMConfig],
      │         posStart, totalInputNum, validLen, batchSize=1)
      │
      ├─ 4e. Prepare KV Cache positions
      │     ctx.kvCacheMgr.PreparePrefill(chunkSeqLen)
      │     → Internally records the prefill interval [currentPos_, currentPos_ + chunkSeqLen)
      │
      ├─ 4f. Get KV Cache tensor
      │     ctx.kvCacheMgr.GetTensors() → kvTensors
      │
      ├─ 4g. Model execute Prefill
      │     modelRunner_.RunPrefill(stepTensors, kvTensors)
      │     │
      │     └─ Internal flow:
      │        ├─ Assemble inputs vector from stepTensors + kvTensors per inputOrder_
      │        ├─ executor.Execute(modelHandle, inputs)
      │        │     └─ KirinExecutor: Tensor→INDTensorBuffer→hiai::IModel::Execute→NPU
      │        └─ Extract logits tensor from ExecuteResult.outputs
      │
      │     ★ KV Cache is written:
      │       keyCache[layer][posStart:posStart+chunkSeqLen] ← key for each position in this chunk
      │       valueCache[layer][posStart:posStart+chunkSeqLen] ← value for each position in this chunk
      │
      ├─ 4h. Commit this chunk's Prefill result
      │     ctx.kvCacheMgr.CommitPrefill(chunkSeqLen)
      │     → currentPos_ += chunkSeqLen
      │
      └─ Continue to next chunk (back to 4a)
          // Pipeline optimization: the current chunk's NPU computation (4g) can run in parallel with the next chunk's Embedding lookup (4a)
          // Implemented via the GetNextEmbeddingInput double-buffer mechanism

   After the last chunk completes:

   ├─ 5. Sample the first output token
   │     sampler_.Sample(logits, request.params.sampling)
   │     → Take the logits of the last position of the last chunk
   │     → doSample=true → SampleTopK → selected token id 3520
   │
   ├─ 6. Tokenizer decode
   │     tokenizer_.Decode({3520}) → "I"
   │
   ├─ 7. Record output
   │     request.outputTokenIds.push_back(3520)
   │     request.outputText += "I"
   │
   └─ 8. Callback notification
        request.callbacks.onPrefillDone({prefillTimeMs=xx, inputTokenCount=2000})
```

KV Cache changes during the Prefill phase (example with 2000 tokens, 5 chunks):
```
Initial:       keyCache[layer][....................................] pos=0
After Chunk 0: keyCache[layer][████................................] pos=32
After Chunk 1: keyCache[layer][██████████..........................] pos=96
After Chunk 2: keyCache[layer][██████████████████..................] pos=224
  ...
Last chunk:    keyCache[layer][████████████████████████████████████] pos=2000
```

#### 8.2.3 Decode Phase (Autoregressive Loop)

```
DecodeController::RunSingle(request, ctx)
│
├─ request.state: Prefilling → Decoding
│
└─ while (!CheckStop(request)):
   │
   │  CheckStop checks:
   │  ├── Is the last of outputTokenIds an EOS token? → No
   │  ├── outputTokenIds.size() >= maxTokens(1024)? → No
   │  └── Hit stopSequences? → No
   │
   ├─ 1. Get the tensors for the Decode step
   │     ctx.tensorMgr.GetStepTensors(seqLen=1, batchSize=1)
   │     → Returns the stepTensors corresponding to seqLen=1
   │
   ├─ 2. Input preparation (two paths depending on useExternalEmbedding)
   │
   │   Path A: useExternalEmbedding = false (embedding done inside the model)
   │     stepTensors.inputs[TensorRole::InputIds].Data() = lastTokenId
   │     // Write token id directly into InputIds tensor
   │
   │   Path B: useExternalEmbedding = true (external Embedding lookup)
   │     embedding_.Run({lastTokenId}) → embedTensor
   │     stepTensors.inputs[TensorRole::InputEmbedding] = embedTensor
   │     // Write embedding result into InputEmbedding tensor
   │
   ├─ 3. Fill runtime config
   │     posStart = ctx.kvCacheMgr.GetCurrentPos()
   │     ctx.tensorMgr.FillLLMConfig(
   │         configTensor,
   │         posStart, totalInputNum=1, validLen=posStart+1, batchSize=1)
   │
   ├─ 4. Fill Attention Mask
   │     totalKvLen = posStart + 1   ← existing KV + 1 position written this step
   │     ctx.tensorMgr.FillAttentionMask(
   │         maskTensor, seqLen=1, totalKvLen, batchSize=1)
   │
   ├─ 5. Get KV Cache tensor
   │     ctx.kvCacheMgr.GetTensors() → kvTensors
   │
   ├─ 6. Execute Decode
   │     modelRunner_.RunDecode(lastTokenId, posStart, stepTensors, kvTensors)
   │     → executor.Execute(modelHandle, inputs)
   │     → NPU performs decode computation on 1 token
   │
   │     ★ KV Cache is extended:
   │       keyCache[layer][posStart] ← key vector of the new token
   │       valueCache[layer][posStart] ← value vector of the new token
   │
   ├─ 7. Advance KV Cache position
   │     ctx.kvCacheMgr.AdvancePos()
   │     → currentPos_++ (10 → 11 → 12 → ...)
   │
   ├─ 8. Sample output token
   │     sampler_.Sample(logits, request.params.sampling)
   │     → Select newTokenId from logits[0,0,:]
   │
   ├─ 9. Tokenizer decode
   │     tokenText = tokenizer_.Decode({newTokenId})
   │     // Decode token id into text
   │
   ├─ 10. Record output
   │     request.outputTokenIds.push_back(newTokenId)
   │     request.outputText += tokenText
   │
   └─ 11. Callback notification (callback once every callbackFreq tokens)
        request.callbacks.onToken({text=tokenText, tokenId=newTokenId, tokenCount=N})
```

KV Cache grows step by step in the Decode loop:
```
Step 0: [██████████............] pos=10  ← After Prefill
Step 1: [██████████████........] pos=11  ← KV for "am" written
Step 2: [██████████████████....] pos=12  ← KV for "an" written
  ...
Step N: [██████████████████████] pos=10+N ← continuous accumulation
```

#### 8.2.4 Inference Complete

```
when CheckStop() returns true:
│
├─ request.state: Decoding → Completed
│
├─ Assemble result
│     GenerateResult {
│       text = "I am an AI assistant...",
│       tokenIds = [3520, 3214, ...],
│       profiling = {totalTimeMs, prefillTimeMs, decodeTimeMs,
│                    inputTokenCount, outputTokenCount, cachedTokenCount},
│       status = Status::OK
│     }
│
├─ Callback notification
│     request.callbacks.onDone(result)
│     → future.set_value(result)  // Unblock the user thread
│
└─ sessionState: Inferring → Idle
```

### 8.3 Multi-turn Dialogue (KV Cache Reuse)

The second-round request reuses the existing KV Cache:

```cpp
llm::GenerateParams params2;
params2.promptText = "What can you do?";
auto result2 = session->Generate(params2);
```

```
PrefillController::Run(request2)
│
├─ 1. tokenizer.Encode("What can you do?") → newTokenIds (e.g. 6 tokens)
│
├─ 2. ctx.kvCacheMgr.GetCurrentPos() → 52  ← accumulation of previous round's prefill(10) + decode(42)
│
├─ 3. embedding.Run(newTokenIds) → embeddingTensor
│
├─ 4. ctx.tensorMgr.GetStepTensors(seqLen=6, batchSize=1)
│
├─ 5. FillAttentionMask:
│     seqLen=6, totalKvLen=52+6=58
│     → The 6 new tokens can see 58 positions (52 existing + 6 new)
│
├─ 6. FillLLMConfig:
│     posStart=52, totalInputNum=6, validLen=58
│
├─ 7. ctx.kvCacheMgr.PreparePrefill(6)
│     → Start writing at pos=52
│
├─ 8. modelRunner.RunPrefill(stepTensors, kvTensors)
│     → NPU only computes the 6 new tokens; the KV of the existing 52 tokens is not recomputed
│
├─ 9. ctx.kvCacheMgr.CommitPrefill(6)
│     → currentPos_ = 58
│
└─ 10. sampler.Sample → firstToken
```

KV Cache accumulates across rounds:
```
After round 1:          [█████████████████████████........] pos=52  (10+42)
After round 2 Prefill:  [██████████████████████████████......] pos=58  (52+6)
After round 2 Decode:   [██████████████████████████████████..] pos=58+N
```

Key: the second round only computes the Prefill of the 6 new tokens; the KV Cache of the existing 52 tokens is directly reused and not recomputed.

### 8.4 KV Cache Full Lifecycle Management

```
┌─────────────────────────────────────────────────────────────┐
│                KVCacheManager state changes                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Session created                                            │
│    Init(config)                                             │
│      → Allocate key/value tensor (one each per layer)        │
│      → currentPos_ = 0, maxSeqLen_ = 4096                   │
│                                                             │
│  Generate #1 (Session: Idle → Inferring → Idle)             │
│    PreparePrefill(10)  → Record interval [0, 10)            │
│    CommitPrefill(10)   → currentPos_ = 10                   │
│    AdvancePos() × 42  → currentPos_ = 52                    │
│                                                             │
│  Generate #2 (Session: Idle → Inferring → Idle)             │
│    PreparePrefill(6)   → Record interval [52, 58)           │
│    CommitPrefill(6)    → currentPos_ = 58                   │
│    AdvancePos() × 30  → currentPos_ = 88                    │
│                                                             │
│  ExportKVCache() ← only allowed when Session is Idle        │
│    → Serialize all layers' KV data + metadata into KVCacheSnapshot │
│    → For saving conversation state                          │
│                                                             │
│  ImportKVCache(snapshot) ← only allowed when Session is Idle│
│    → Restore KV data and position from snapshot             │
│    → For restoring a previous conversation                  │
│                                                             │
│  Expand(8192)                                               │
│    → Triggered when currentPos approaches maxSeqLen         │
│    → Reallocate a larger KV tensor, copy old data           │
│    → maxSeqLen_: 4096 → 8192                                │
│                                                             │
│  SetSparseMode(SparseMode::Attention)                       │
│    → Enable sparse attention                                │
│    → GetSparseMask() returns a sparse mask, keeping only    │
│      attention to important tokens                          │
│                                                             │
│  SwapOut()  [Suspend]                                       │
│    → KV Cache tensor data backed up to CPU memory (swappedData_) │
│    → Release NPU tensor, reclaim NPU memory                 │
│                                                             │
│  SwapIn()   [Resume]                                        │
│    → Reallocate NPU tensor                                  │
│    → Restore data from swappedData_ to NPU                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 Prompt Cache

Scenario: fixed system prompt prefix, not recomputed across multi-turn dialogues.

```
session->SetPrefixPrompt("You are a professional assistant, please answer in Chinese.")
│
├─ promptManager.SetPrefixPrompt(prefix)
│     ├─ tokenizer.Encode(prefix) → prefixTokenIds (e.g. 15 tokens)
│     ├─ Temporarily Generate once to compute the prefix's KV Cache
│     └─ Save prefixInfo_ = {tokenIds: [...], kvCacheLen: 15}

// On each subsequent Generate:
PrefillController::Run(request)
│
├─ promptManager.GetPrefixInfo() → prefixInfo (kvCacheLen=15)
│     // The first 15 positions of KV Cache are already the system prompt's KV
│     // Only need to Prefill the new user input
│
├─ kvCacheMgr.GetCurrentPos() → 15  ← Continue from the end of the system prompt's KV
├─ PreparePrefill(userInputLen)     ← Only Prefill user input, do not recompute prefix
└─ ... rest same as normal flow
```

Scenario: save complete conversation state:

```
auto snapshot = session->ExportKVCache()
│
└─ kvCacheMgr.Export()
      → Serialize all layers' KV data + {seqLen, numLayers, numKVHeads, headDim}
      → Returns KVCacheSnapshot {data: vector<uint8_t>, seqLen=88, ...}

// Restore in another session:
session->ImportKVCache(snapshot)
│
└─ kvCacheMgr.Import(snapshot)
      → Deserialize, restore all layers' KV data to tensor
      → currentPos_ = snapshot.seqLen (88)
      → Conversation can continue seamlessly
```

### 8.6 Batch Decode

```
session->GenerateBatchAsync(batchParams)
  → Create BatchInferRequest
  → scheduler.SubmitBatch()
     ═══ worker thread ═══
     → PrefillController::Run(request1)  → slot0
     → PrefillController::Run(request2)  → slot1
     → DecodeController::RunBatchWithStrategy(requests, strategy, scheduler)
        while (!strategy.ShouldStopDecode()):
          modelRunner.RunDecodeBatch([token0, token1])
          → Each slot samples independently → checkStop
          → When a slot completes: strategy.OnSlotFree() → check queue
```

### 8.7 Suspend/Resume

```
engine->Suspend()
│
├─ 1. EngineState: Ready → Suspended (CAS atomic transition)
│
├─ 2. scheduler.Pause()
│     → Set paused_ = true
│     → Worker thread stops taking new requests after completing the current one
│     → Wait for the currently executing InferRequest to complete
│
├─ 3. For each Session:
│     ├─ ctx.kvCacheMgr.SwapOut()
│     │     → Iterate layers_, copy key/value tensor data
│     │       to swappedData_ (CPU-side vector<uint8_t>)
│     │     → Release NPU tensor
│     │
│     ├─ ctx.tensorMgr.ReleaseBuffers()
│     │     → Clear tensorPool_ and embeddingPipeline_
│     │     → Release NPU memory occupied by input/output tensors
│     │
│     └─ modelRunner.FreeResource()
│           → modelLoader.Unload(modelHandle)
│           → Unload NPU model graph
│
└─ 4. NPU memory cleared, available for other applications

engine->Resume()
│
├─ 1. For each Session:
│     ├─ modelRunner.RestoreResource()
│     │     → modelLoader.Load(modelPath, batchSize) → new ModelHandle
│     │
│     ├─ ctx.tensorMgr.RebuildBuffers()
│     │     → Reallocate input/output tensors based on previously used seqLens
│     │
│     └─ ctx.kvCacheMgr.SwapIn()
│           → Reallocate NPU tensor
│           → Restore KV Cache data from swappedData_
│           → currentPos_ unchanged, conversation can continue
│
├─ 2. scheduler.Resume()
│     → Set paused_ = false
│     → Worker thread resumes taking requests from the queue
│
└─ 3. EngineState: Suspended → Ready
```

After resuming, the KV Cache fully retains the conversation history; Generate can continue from lastPos.

### 8.8 Full Data Flow Diagram

```
User code        API Layer          Orchestration Layer     Component Layer     Platform HAL
─────────        ─────────          ──────────────────      ──────────────      ────────────

Engine.Create(config)
  │                                Scheduler ←────────── PlatformFactory
  │                                                        .Create(Kirin)
  │                                │                       ↓
  │                                │                [ModelLoader]
  │                                │                [TensorAlloc]
  │                                │                [Executor]
  │                                │                [PerfCtrl]
  │                                │
  │                                ├─ Engine-level shared components:
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
session.Generate(params)          Session-level components:
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
  │                     │    ├─ [Chunking loop] for each chunk:
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

## 9. Extensibility: Speculative Inference

No structural changes needed. The extension approach is "add new Controller + add new Component".

| Layer | Change | Magnitude |
|----|------|------|
| API | None (SpeculativeConfig already reserved) | 0 |
| Orchestration Layer | Add SpeculativeDecodeController | ~1 new class |
| Orchestration Layer | Scheduler routing adds one if branch | ~5 lines |
| Component Layer | Add SpecTree component | ~1 new class |
| Component Layer | Other components unchanged | 0 |
| Platform HAL | Unchanged | 0 |
| State management | RequestState adds 2 states (Drafting, Verifying) | ~2 lines |

SpeculativeDecodeController manages the draft/verify loop, using two ModelRunner instances (draft + target), each holding an independent KVCacheManager. SpecTree manages candidate trees for tree-type/EAGLE variants.

Speculative inference flow:
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
      targetModelRunner.RunPrefill(...)          → targetLogits (batch verification)
      Compare draftLogits and targetLogits → acceptCount
      targetKVCacheMgr.CommitVerify(acceptCount)
      Clean up draftKVCacheMgr (roll back rejected positions)

    Append accepted tokens to request.outputTokenIds
```

---

## 10. Current vs New Architecture Comparison

| Dimension | Current architecture | New architecture |
|------|---------|--------|
| Total code volume | ~43,800 lines | ~10,200 lines |
| C-API functions | 110+ | ~15 C++ methods + ~20-30 C bindings |
| Context Set/Get | 80+ | 5 grouped structs |
| Callback Set functions | 6 | 1 GenerateCallbacks struct |
| InitOption Set | 7 | 1 EngineConfig struct |
| God Class (>1000 lines) | 4 | 0 |
| State management | 10+ scattered `atomic<bool>` | 3 explicit state machines |
| Platform coupling | hiai::* scattered in 30+ files | Encapsulated in platform/kirin/ directory |
| Async path | Sync wrapping async (thread + queue) | Async first, sync = wait |
| Testability | Depends on NPU hardware | Mock HAL can test full pipeline without NPU |
| Add speculative inference | Modify 5+ existing files | Add 2 classes, no changes to existing code |
| Add multimodal | Modify 8+ existing files | Add Controller + Component |
