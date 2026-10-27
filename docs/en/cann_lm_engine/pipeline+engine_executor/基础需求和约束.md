# Basic Requirements and Constraints

English | [简体中文](../../../zh/cann_lm_engine/pipeline+engine_executor/基础需求和约束.md)

> Version: v0.5 (Draft)
> Status: READ stage product (see D3), enters DESIGN stage after finalization
> Change record at end of document

## Purpose

Design the LLM Engine used for autoregressive inference of large language models, focusing on the top-level inference pipeline.

**Scope (this module is responsible for):**
- Autoregressive inference flow orchestration, Prefill/Decode two-stage execution
- Queue scheduling and batching
- Input preprocessing (tokenizer / embedding lookup)
- Post-processing (sampler sampling)
- Model loading, context management
- External C++ API boundary

**Scope (this module is NOT responsible for, provided by underlying runtime encapsulation):**
- KV Cache management
- Hardware device abstraction and tensor operator execution
- RPC / service lifecycle / multi-tenant authentication and other "servitization" governance

**Terms:**
- omc: offline compiled model format, this module loads via file path (graph path + weight path) or already-read buffer.
- `PipelineExecutorBase`: the internal component orchestration body held by `EngineExecutor` via PIMPL (impl, constraint #3: abstract base class + currently only derived `AutoregressivePipelineExecutor` (autoregressive) + `PipelineExecutorFactory` construction), aggregates `Scheduler` / `PrefillExecutor` / `DecodeExecutor` / `Preprocessor` / `Sampler` / `HistoryPromptManager` and other components. In this md, "pipeline" refers to the `PipelineExecutorBase` instance; "Engine holds / Pipeline holds" etc. uniformly refers to `PipelineExecutorBase` holding the corresponding component.
- `EngineExecutor`: public class (PIMPL shell), held by user; internally delegates to `PipelineExecutorBase`.
- **"Engine layer"**: refers to the Engine module/library overall (layer concept, e.g. "Engine layer ModelLoader" refers to the ModelLoader encapsulated by this module), different semantics from "Engine holds" (refers to PipelineExecutorBase) and `EngineExecutor` (class), distinguished by context.

**Form and instance model:**
- Form is **embedded library** (not standalone service). Only exposes C++ API boundary externally; if servitization is needed in the future, it is implemented by outer wrapper, zero intrusion to this module.
- User holds one `EngineExecutor` instance, performs context management and inference calls via `Context` and `Prompt`.
- **Multi-instance supported**: caller can instantiate multiple `EngineExecutor` in the same process (loading different models or different session pools). Engine does not depend on global singleton, resource pool belongs to instance itself.

## Feature Points

### 1. Text / Multimodal embedding Input (encapsulated as `Prompt` structure)
- **Text input**: single `string`, does not introduce multi-turn chat / messages list.
- **Embedding input**: pre-computed tensors of other modalities (e.g. image); **does not include** token embedding path bypassing tokenizer (token embedding is produced by the embedding lookup module in the preprocessing stage, see #7).
- **Mixed input**: same Prompt can carry both text and one or more modality embeddings.
- **Sampling parameters**: independent as `SamplingParams` structure, not in Prompt; Prompt only carries input content.
- **Clean design**: Prompt does not carry request_id / priority / callback handle and other scheduling metadata.
- **Avoid copying**: large tensors passed by `shared_ptr` / view semantics; Prompt movable, shallow-copyable, but internal large objects not deep-copied.
- **Field draft** (illustrative only, refined at design stage):
  ```cpp
  struct ModalityEmbedding {
      ModalityTag tag;                       // modality identifier (image / audio / ...)
      std::shared_ptr<Tensor> embedding;     // shared ownership, avoid copy
  };
  struct Prompt {
      std::string text;
      std::vector<ModalityEmbedding> embeddings;  // can be empty, empty for pure text
  };
  ```
- **Placeholder insertion scheme (decided)**: multimodal embedding marks position via placeholder token in text, replaced in preprocessing stage (see #7).
- **Placeholder ↔ embedding mapping mechanism**: by `Prompt.embeddings` array order ↔ placeholder appearance order in text one-to-one correspondence (regardless of modality tag). I.e. the i-th embedding corresponds to the i-th placeholder. `ModalityEmbedding.tag` is only modality annotation, does not participate in matching.
- **`SamplingParams` field draft** (independent structure, not in Prompt; fields scattered in #2 termination conditions, #8 sampling algorithm are gathered here):
  ```cpp
  struct SamplingParams {
      int max_tokens;                          // see #2 (a)
      std::vector<TokenId> stop_token_ids;     // see #2 (b), includes EOS
      std::vector<std::string> stop_strings;   // see #2 (b)
      float temperature;                        // see #8
      int top_k;                               // see #8
      float top_p;                             // see #8
  };
  ```

### 2. Autoregressive Inference Path: Prefill / Decode Two Stages
- **Two independent `ModelExecutor` instances**: `PrefillExecutor` and `DecodeExecutor` each hold and manage one omc (Prefill omc / Decode omc), instances independent, convenient for each to manage lifecycle and resources.
- **Executor responsibility boundary**: forward shell + KV Cache ringBuffer cursor management + load/unload/LoRA control delegation (per `docs/model_executor/`). **Does not** do sampling (belongs to #8 Sampler), **does not** do KV tensor memory allocation/release (belongs to underlying/`TensorManager`); KV ringBuffer cursor management belongs to this module, conforms to the established boundary in "Purpose").
- **Executor batch interface**:
  - `PrefillExecutor.forward`: **batch=1**, processes one request / one chunk at a time, does not batch with other prefill requests
  - `DecodeExecutor.forward`: **batch dimension can be > 1**, processes multiple active decode requests' current tokens at once (data parallel), upper limit constrained by model metadata `max_batch_size` (see #4)
  - Both are serial interfaces (C4): one forward one batch, no built-in request-level concurrency; batch dimension is data parallel
- **Stage switching ownership**: Prefill → Decode switching, Decode iteration loop driving are uniformly managed by Scheduler (connects with #3 scheduling module), Executor does not self-drive cross-stage.
- **Autoregressive loop**: Decode stage iterates at token level — each step `DecodeExecutor → logits → Sampler → next token`, next token fed back as next step input, until termination.
- **Decode termination conditions** (stops when any met):
  - (a) reach `max_tokens`
  - (b) hit stop token (including EOS) or stop string
  - (c) user actively cancels (cancel)
- **Chunked Prefill**: need to support chunking long prompt, executing in multiple times, to control TTFT and single-step memory peak. **Chunking done by `Preprocessor`** (execution + chunking strategy, based on config), produces multiple chunk tasks; Scheduler is responsible for interleaved scheduling of each chunk with Decode batch (strategy see #3 Continuous Batching).
- **Weight sharing**: Prefill omc and Decode omc may reference same weights (same model different compiled versions). Underlying runtime already supports weight reuse mechanism, Engine layer **does not do special optimization** — just calls load interface for each of the two Executors, deduplication by underlying.

### 3. Queue Scheduling and Asynchronous Inference Interface
- **External API (dual form)**:
  - `Generate(Prompt, Context, SamplingParams)`: synchronous blocking, returns complete `Result`.
  - `GenerateAsync(Prompt, Context, SamplingParams, StreamCallback)`: asynchronous submission, immediately returns `RequestHandle`. `StreamCallback` optional (empty means non-streaming); if provided, triggered after each Decode step completes, user gets that step's token (id + text) in callback.
- **`StreamCallback` semantics**: `using StreamCallback = std::function<void(TokenId, std::string)>`; called after PipelineExecutorBase post-processing produces token each step (see #8). Callback triggered in PipelineExecutorBase internal thread, user should not block or call Engine's synchronous API in callback.
- **`RequestHandle` semantics**: can query status (queued / Prefilling / Decoding / completed / canceled / failed), can cancel, can wait for complete `Result`. `Generate` equivalent to `GenerateAsync` (no callback) + blocking wait.
- **Scheduler responsibility boundary**:
  - Task queuing (FIFO)
  - Stage switching (Prefill → Decode driving, see #2)
  - **Continuous Batching**: iteration-level dynamic batching (see "Continuous Batching mechanism" below)
  - Interleaved scheduling of each chunk task (**chunking done by Preprocessor**, see #7; each chunk treated as independent task enqueued)
  - Concurrency upper limit control (decode pool capacity = `max_batch_size`, see below)
- **Continuous Batching mechanism (from v0.2)**:
  - **decode pool**: capacity = model metadata `max_batch_size` (see #4). At most N requests in active decode state at any time, batched into one `DecodeExecutor.forward` (batch=N).
  - **Request join**: new request Prefill completes → next decode iteration joins decode pool (occupies one empty slot).
  - **Request exit**: termination condition (max_tokens / stop / cancel, see #2) triggers → exits decode pool after current iteration ends (releases slot).
  - **Over-limit handling**: when pending decode request count > empty slot count, excess requests wait in queue until a slot is released (continuous batching core: request lifecycle spans iterations, no need to wait for entire batch to complete).
  - **Interleaving strategy (Prefill and Decode)**: Prefill independent (batch=1, does not fuse forward with Decode). Default scheduling strategy — **process Prefill first when there is an empty slot**: if decode pool has empty slot (active < max_batch_size) and there are pending prefill requests, prioritize executing one Prefill (brings it into decode pool); otherwise execute one Decode batch iteration.
  - **No variable padding**: KV of each request within batch is mutually independent (addresses contiguous but independently managed, not padded to longest within batch); decode om graph batch dimension fixed = `max_batch_size`, when active not full ModelExecutor pads to `max_batch_size` (attentionMask shields padded positions, per `docs/model_executor/` §1); prefill om fixed batch=1 no padding.
  - **Iteration-level batching decision**: Scheduler decides batch members each iteration (constrained by max_batch_size), whether to insert Prefill, logits dispatch to each request within batch (after sampling fed back to each history). This is Scheduler's core responsibility, no separate Batcher split out.
- **Concurrency model (see #2 discussion)**:
  - `ModelExecutor` is **serial interface**, single forward, no built-in request-level concurrency (C4). batch dimension is data parallel, not request-level concurrency.
  - From v0.2 Decode stage supports multi-batch (Continuous Batching); Prefill still batch=1.
  - Concurrency source: multi-instance EngineExecutor (horizontal) + single-instance Continuous Batching (vertical).
- **Cancel semantics (two granularities)**:
  - **`RequestHandle.cancel`**: cancel single request. Already generated partial tokens **discarded**, not returned to user; Handle enters "canceled" terminal state.
  - **Context-level cancel** (triggered via `ContextId`): cancels **all in-flight requests** on that Context, each corresponding RequestHandle enters "canceled" terminal state. Used for session overall termination scenario.
- **Scheduling module ownership**: Scheduler is this module's core component, collaborates with PrefillExecutor / DecodeExecutor; Executor does not self-drive cross-stage, all orchestrated by Scheduler.

### 4. Model Modular Loading
- **Load timing**: `EngineExecutor` loads two omc (Prefill omc + Decode omc) once at initialization. Current version does not support runtime hot-loading / model switching.
- **Load sources (only two)**:
  - (a) File path: includes omc graph path + weight path + **json metadata file path with same name as omc**
  - (b) Already-read buffer: graph buffer + weight buffer + metadata buffer
- **`EngineExecutor` constructor signature draft** (corresponds to two load sources, one each for Prefill + Decode):
  ```cpp
  // Path loading
  EngineExecutor(OmcPath prefillOmc, WeightPath prefillWeight,
                  OmcPath decodeOmc,  WeightPath decodeWeight,
                  const EngineOptions& options = {});
  // Buffer loading
  EngineExecutor(OmcBuffer prefillOmc, WeightBuffer prefillWeight,
                 OmcBuffer decodeOmc,  WeightBuffer decodeWeight,
                 const EngineOptions& options = {});
  ```
  json metadata path derived from omc path (same-name json); in buffer mode metadata buffer passed separately or merged. `EngineOptions` aggregates optional config (e.g. chunk size default, concurrency upper limit, specific fields to be listed at design stage).
- **Engine layer ModelLoader responsibility** (encapsulates loading flow, thin wrapper):
  1. Parse input parameters (path or buffer)
  2. Read json metadata (max batch size / max seq len / dtype / tensor layout, etc., specific fields to be listed at design stage)
  3. Call underlying `ModelManager` to complete actual loading
  4. Hand loading result handle to corresponding `ModelExecutor` for holding
- **Underlying `ModelManager`** (underlying already encapsulated, Engine layer does not re-implement):
  - Holds loaded model instance + all metadata
  - Provides weight reuse / deduplication mechanism (per #2 established: Engine layer just calls load interface once for each of the two Executors, deduplication by underlying)
  - Provides release method for Executor to call at destruction
- **Loading result product**: model instance + metadata held by underlying `ModelManager`; `ModelExecutor` holds reference / handle to it.
- **Error handling**: uses **`Status` (not exception)**. Load failure (file not exists / buffer corrupted / version mismatch / json parse failure, etc.) returned via `Status` (coarse-grained code, details via logs C11), throughout Engine's API style (see C2).
- **Resource release**: RAII style — `ModelExecutor` calls underlying `ModelManager`'s release method at destruction. No explicit `unload` interface provided.

### 5. Context Management
- **Ownership**: Context held by **user**. EngineExecutor provides interfaces:
  - `CreateContext`: creates Context, returns `ContextId` + Context handle
  - `DestroyContext`: destroys Context (releases corresponding HistoryPromptManager, underlying KV handle, etc.)
  - `ConfigureContext(ContextId, ...)`: configures Context configurable fields (see "configurable fields" below)
  - `CancelContext(ContextId)`: Context-level cancel, cancels all in-flight requests on that Context (see #3 Cancel semantics)
- **Configurable fields (current version)**:
  - User custom metadata (string / arbitrary KV)
  - Other configurable items (e.g. that Context's default chunk size override, concurrency upper limit override) **to be listed at design stage**
- **Multi-turn dialogue continuation**: each turn calls `Generate` / `GenerateAsync`, Prompt **only contains this turn's new text**; already generated historical tokens managed by separate `HistoryPromptManager` module (see below).
- **Switching semantics** (supports a + b):
  - (a) Switching between multiple Contexts within same EngineExecutor (multi-session reuses same Engine)
  - (b) Context passes state between Prefill / Decode stages (stage cursor, KV handle, etc.)
  - **Does not support** migrating Context across Engine instances
- **Context only holds references**: internal state held in reference / handle form, data body in underlying (KV) or inside `PipelineExecutorBase` (HistoryPromptManager).
- **KV switching handled in underlying**: multi-session switching KV cache swap-in / swap-out transparently handled by underlying runtime, Engine is unaware, does not participate in scheduling (conforms to "KV in underlying" boundary).
- **Persistence**: current version does not support Context save / restore.

**Context content draft (refined at design stage):**

A. User-visible / configurable (public):
- `ContextId`: identifier, for multi-session switching, state query, Context-level cancel (see #3 Cancel semantics)
- User custom metadata (optional)

B. `PipelineExecutorBase` internal state (Context holds reference / handle):
- KV cache handle: references underlying-managed KV
- `HistoryPromptManager` handle: references `PipelineExecutorBase`-held HistoryPromptManager instance (isolated per Context, see #9)
- Inference stage state: stage (idle / prefilling / decoding / done), sequence cursor (processed token position, for chunked prefill continuation)
- Owning EngineExecutor reference (implicit or explicit)

**New module: `HistoryPromptManager`** (see feature point #9, independent feature point / module, held by `PipelineExecutorBase`)

### 6. External Interface Layer Separation
- **Separation mechanism: PIMPL**. `EngineExecutor` is concrete class, impl hidden in .cpp; public header does not expose any internal types.
- **Public type list** (appears in public header):
  - `EngineExecutor`, `EngineOptions`, `Context`, `ContextId`, `Prompt`, `ModalityEmbedding`, `Tensor` (provided by underlying, forward declared in public header), `ModalityTag`, `TokenId`, `SamplingParams`, `RequestHandle`, `Result`, `StreamCallback`, `Status` (status code struct, with `static constexpr` constants), `Buffer`, `ModelArtifact`, `ModelArtifactPath`
- **Internal type list** (does not appear in public header):
  - `PipelineExecutorBase` (EngineExecutor's PIMPL impl), `Scheduler`, `ModelExecutor` (`PrefillExecutor` / `DecodeExecutor` as its two instances), `Preprocessor`, `HistoryPromptManager`, `Tokenizer` (engine internal polymorphic module `TokenizerBase`+factory+BPE/SP, C8 v1.2), `TextEmbedding` (engine internal embedding module, load+bind), `Sampler` (engine internal concrete class); `IModelManager` is the only L2→L3 abstract interface (created via link-time `CreateModelManager()` factory, see C8 v0.11). Each module's auxiliary types (`ModelHandle` / `ModelMetaInfo` / `ModalityEntry` / `RequestImpl` / `RequestId` / `StepOutput`, etc.) see `04-interfaces` lower half.
- **Optional C API layer** (see `docs/c_api/`): provides `extern "C"` thin wrapper on top of C++ API (opaque handle + POD struct), provides ABI stable boundary for cross-language bindings. Does not replace C++ API; C API public types are C-compatible POD struct + opaque pointers (`CANN_LMEngine_Executor` / `CANN_LMEngine_Request`) + value types (`CANN_LMEngine_ContextId`). See c_api SPEC/DESIGN v0.1.0.
- **Thread model**:
  - `GenerateAsync`: can be called concurrently from multiple threads, API thread-safe (requests safely enqueued)
  - Multiple `Context`: concurrent submission allowed (submission-layer concurrency, execution layer still serial, see #2/#3)
  - Same `Context`: concurrent calls **return error** (explicitly exposes problem, avoids implicit queuing masking bugs)
  - `Generate` (synchronous): implemented per `GenerateAsync` + blocking wait to achieve synchronous effect; concurrent behavior same as `GenerateAsync`
- **ABI stability**: current version does not commit to ABI stability, source-level compatibility sufficient.

### 7. Input Preprocessing Flow
- **Engine internal modules (C8 v1.2, implemented by Engine itself)**:
  - `Tokenizer` (= `TokenizerBase` polymorphic base class + `TokenizerFactory` + BPE/SentencePiece derived self-registration, see `docs/tokenizer/`): text → token id; M-02 holds via factory `unique_ptr<TokenizerBase>`
  - `TextEmbedding`: loads embedding table and binds to model (per `docs/model_executor/` §4.1.1; tokenId lookup is inside om graph, not lookup-produced tensor); M-02 holds by value
  - `Sampler` (engine internal concrete class, no polymorphism requirement): logits + sampling parameters → token id; M-02 holds by value
- **Standalone `Preprocessor` module** coordinates entire preprocessing chain (single responsibility, Scheduler only schedules, does not chain preprocessing).
- **Complete preprocessing chain (Prefill path)**:
  ```
  Prompt.text (contains placeholder tokens) ──Tokenizer──► token ids ─────┐
                                                                                ├─ concat ─► complete token id sequence
  HistoryPromptManager historical token ids ──────────────────────────────┘              │
                                                                                           ▼
  Placeholder replacement: placeholder positions filled with corresponding multimodal embedding ─► mixed sequence
  (placeholder positions are embedding tensors; other positions are token id, lookup inside om graph forward)
                                                                                           │
                                                                                           ▼
  TextEmbedding LoadAndBind (init period loads table + binds model) ─► token id sequence + multimodal placeholder ─► chunking (see below) ─► Executor (om graph forward internal lookup by tokenId)
  ```
- **Placeholder replacement**: multimodal embedding marks position via placeholder token in text, replaced with corresponding modality embedding in preprocessing stage (corresponds to #1 decided scheme). Mapping by `Prompt.embeddings` array order ↔ placeholder appearance order (see #1).
- **Chunked Prefill chunking ownership (decided)**: chunking done by `Preprocessor` (execution + chunking strategy, based on configured chunk size), produces multiple chunk tasks handed to Scheduler; Scheduler only responsible for interleaved scheduling of each chunk with other tasks/Decode.
- **Decode single-step preprocessing (lightweight subset)**: above chain is Prefill input. Decode each step only needs `single token id → input tensor → DecodeExecutor` (tokenId lookup inside om graph forward), is a lightweight subset of Prefill preprocessing chain, provided as fast path by Preprocessor.

### 8. Post-processing Flow (sampling + detokenize + stop detection)
- **Sampler ownership**: engine internal concrete class `Sampler` (C8 v1.2, no polymorphism requirement), `PipelineExecutorBase` holds by value; stateless, not history-aware.
- **Supported algorithms (basic)**: greedy, top-k, top-p, temperature.
- **Not supported**: Beam search, multi-sampling (n > 1), logprobs return, repetition / frequency / presence penalty and other history-related algorithms (because Sampler temporarily not history-aware, see below).
- **Sampler temporarily not history-aware**: Sampler only receives current step logits to produce token id, does not read `HistoryPromptManager`. Therefore penalty-type algorithms not supported, left for future versions.
- **Post-processing flow** (controlled by `PipelineExecutorBase`):
  ```
  Executor outputs logits
        │
        ▼
  Sampler sampling ─► token id
        │
        ├─► PipelineExecutorBase writes to HistoryPromptManager (see #9)
        │
        ▼
  Tokenizer detokenize (streaming) ─► text
        │
        ├─► triggers StreamCallback (if registered, see #3) ─► user gets that step's token
        │
        ▼
  stop detection
        ├─► stop token (including EOS): token id level, compared immediately after Sampler output
        └─► stop string: text level, detected on plaintext after detokenize
        │
        ▼
  decide whether to terminate (see #2 termination conditions)
  ```
- **StreamCallback trigger point**: triggered after each step detokenize produces text, before stop detection (see #3 `StreamCallback` semantics). Whether to callback the step that hits stop is decided at design stage (inclination: do not callback the step that hits stop, to avoid leaking stop marker to user).
- **Output form**: `Result` contains both token id sequence and text (`PipelineExecutorBase` controls the flow from token id to text via Tokenizer).
- **stop string detection details (to be confirmed at design stage)**: streaming detokenize buffer strategy, retention strategy for already-generated buffer part when stop string is hit.


### 9. HistoryPromptManager (Historical Token Management, Independent Module)
- **Positioning**: independent feature point / module, held by `PipelineExecutorBase` (PipelineExecutorBase is EngineExecutor's PIMPL impl, see terms), isolated per Context (one instance per Context).
- **Responsibility**: manages **token id history** accumulated across turns for a single Context, for next-turn prefill concatenation.
- **Storage form**: token id sequence (not text), avoids repeated tokenize and tokenizer non-bijective loss.
- **Write subject**: written internally by `PipelineExecutorBase`. After each Decode step Sampler samples new token, `PipelineExecutorBase` writes to corresponding Context's history (user-unaware).
- **Read usage**: at next-turn prefill, `PipelineExecutorBase` takes historical token id from `HistoryPromptManager` + new token id from this turn's Prompt new text via tokenizer splitting → concatenates into complete input sequence.
- **To be confirmed at design stage**: truncation / sliding window strategy, alignment strategy with underlying KV length.


## Constraints

### C1. Module Encapsulation
Modules with sufficient independence and complexity need separate encapsulation. List of modules needing separate encapsulation in this module:
- `EngineExecutor` (external interface layer, PIMPL)
- `Scheduler`
- `PrefillExecutor`, `DecodeExecutor` (two instances of `ModelExecutor`)
- `Preprocessor` (coordinates `TokenizerBase` / `TextEmbedding` / placeholder positioning / chunking; tokenId lookup inside om graph)
- `HistoryPromptManager`
- Engine internal modules `Tokenizer` (`TokenizerBase`+factory+BPE/SP, held via factory `unique_ptr`) / `TextEmbedding` (load+bind, by value) / `Sampler` (concrete class, by value) (C8 v1.2); L2→L3 injection point `IModelManager` (creates owned MM via link-time `CreateModelManager()` factory, backend determined at compile time, no separate shell class, see C8 v0.11)

### C2. Error Handling
**Uniformly uses `Status` (not exception)**, throughout all public APIs. `Status` is struct, holds `int code`, with a set of `static constexpr` constants (`SUCCESS=0` / `FAILURE=1` / `INVALID_ARGUMENT=2` / `INVALID_STATE=3` / `NOT_FOUND=4` / `CANCELED=5`); detailed failure reason output via logs (C11), status code only gives coarse-grained. Value-returning operation signature `Status Op(..., T* out)` (value returned via out parameter), no-value-returning operation signature `Status Op(...)`. No `StatusOr`, no exceptions.

### C3. Avoid Copying
Large objects (tensors, buffers, historical token sequences) uniformly passed by `shared_ptr` / view semantics; movable, shallow-copyable, internal large object deep-copy prohibited.

### C4. ModelExecutor Serial Interface
`ModelExecutor` is serial interface, single forward, no built-in request-level concurrency. **Batch dimension can be > 1 (data parallel, not request-level concurrency)** — `DecodeExecutor.forward` processes multiple active requests' current tokens at once (Continuous Batching, see #3). Concurrency source is only **multi-instance** (horizontal) + **single-instance Continuous Batching** (vertical).

### C5. Boundary Discipline: Engine Layer Unaware of Lower Layer
Engine layer is unaware of KV Cache management, hardware device abstraction, tensor operator execution, device scheduling. These are provided by underlying runtime encapsulation, Engine layer only calls encapsulation interfaces.

### C6. Interface Separation
PIMPL + public/internal type list separation (see feature point #6). public header does not expose any internal types.

### C7. Multi-instance
Does not depend on global singleton, supports multi-instance. Resource pool belongs to `EngineExecutor` instance itself.

### C8. Lower-layer Encapsulation Modules
- **L2→L3 injection point** (abstract interface, underlying provides implementation, 01 §7.2 injection point): `IModelManager` — created as owned `unique_ptr<IModelManager>` via link-time free function `CreateModelManager()` (D-12, backend determined at compile time); ModelExecutor::Load self-calls factory to create and hold, pipeline unaware of backend type. No longer runtime-injected via `shared_ptr` (v0.11 hard cut).
- **Engine internal modules** (C8 v1.2): `Tokenizer` (polymorphic `TokenizerBase` + factory + BPE/SP derived self-registration, held via factory `unique_ptr`, per `docs/tokenizer/`), `TextEmbedding` (load table + bind model, by value, per `docs/model_executor/` §4.1.1), `Sampler` (concrete class, no polymorphism requirement, by value) — implemented by Engine itself, constructible from test data / factory stubs. `TokenizerBase` has virtual functions (polymorphic dispatch); `TextEmbedding`/`Sampler` no polymorphism requirement.

### C9. Language Standard
**C++17**.

### C10. Dependency Principle
**Minimal dependency**. Only introduce third-party libraries when necessary (e.g. `nlohmann/json` for reading omc same-name json metadata, see #4). **Logging library and testing framework are both self-developed**, do not introduce `spdlog` / `GoogleTest`, etc.

### C11. Logging
Engine layer needs logging (scheduling, loading, error observability), uses **self-developed lightweight logger**.

### C12. Code Style
**TBD** (unified at development stage, recommend synchronously writing to `AGENTS.md` lint/format config).

## Design Process Requirements

### D1. Design Deliverables
Follow the document map already listed in `docs/README.md`, produce in order:
- `01-architecture.md`: architecture overview (design goals, constraints, performance budget, top-level architecture diagram, data flow, layering model)
- `02-components.md`: component responsibilities (each component's responsibility boundary, internal structure, collaboration relationships)
- `03-key-mechanisms.md`: key mechanisms (Continuous Batching / PagedAttention / KV Cache / Prefill-Decode / sampling)
- `04-interfaces.md`: C++ interface contracts (header-level abstraction)

### D2. Design Order
**Interface first**, advance in four steps:
1. First define **public API** (header abstraction of feature point #6 public type list)
2. Then define **each internal module header abstraction** (`PipelineExecutorBase` / Scheduler / ModelExecutor / ModelLoader / Preprocessor / HistoryPromptManager`, etc.)
3. Then define **data flow and component collaboration** (02-components)
4. Finally define **key mechanism details** (03-key-mechanisms)

### D3. Connect with AGENTS.md Workflow
AGENTS.md workflow is `READ → [DESIGN] → TEST-WRITE (RED) → IMPLEMENT (GREEN) → TEST-RUN → REVIEW`. This md's positioning:
- **This md belongs to READ stage product** (basic requirements and constraints, as input for subsequent design)
- Downstream `01-04` design documents belong to **DESIGN stage product**
- This md must be finalized before entering DESIGN stage to produce `01-04`

### D4. This md's Change Management
- This md **uses version number** (see "version" field at document top).
- After finalization, if need to modify requirements / constraints, must **increment version number** and register change content in "change record" at end.
- Major scope changes (e.g. add / remove feature points, modify boundary discipline) increment minor version number; detail clarification increments revision number.

### D5. Review
This md **does not participate** in AGENTS.md's +1 self-review / +2 external review mechanism. Downstream `01-04` design documents participate in review per AGENTS.md.

### D6. Testing Strategy
**Pending**. Self-developed testing framework (see constraint C10) form, test organization, coverage requirements, etc. to be defined before IMPLEMENT stage.

---

## Change Record

| Version | Date | Change content |
|------|------|----------|
| v0.1 | 2026-08-03 | Initial version: finalized purpose, scope, 9 feature points, 12 constraints, 6 design process requirements |
| v0.2 | 2026-08-03 | Consistency review revision: ① chunking ownership finalized to Preprocessor (eliminate #2/#3/#7 conflict); ② define `PipelineExecutor` (= EngineExecutor's PIMPL impl) and unify document-wide "Engine/pipeline" terminology; ③ add Context-level cancel and streaming `StreamCallback` callback; ④ add `EngineExecutor` constructor signature, `SamplingParams` fields, `ConfigureContext` configurable fields; ⑤ error code defined as enum; ⑥ `Tensor`/`ModalityTag`/`TokenId` added to public type list; ⑦ placeholder ↔ embedding mapping mechanism by order; ⑧ add Decode single-step preprocessing fast path; ⑨ C1 remove misleading "shell" wording, D2 add `PipelineExecutor`, C5 unify "Engine layer" |
| v0.3 | 2026-08-03 | Scope extension (Continuous Batching): ① #2 add Executor batch interface (PrefillExecutor batch=1, DecodeExecutor batch>1); ② #3 add Continuous Batching mechanism — decode pool (capacity=`max_batch_size`), request join/exit, over-limit queue, interleaving strategy (process Prefill first when slot available), no padding (each request KV independent), iteration-level batching decision belongs to Scheduler (no Batcher split); ③ delete original "v0.1 does not implement iteration-level dynamic batching / reserve extension point" wording, change to actual definition; ④ C4 add "batch dimension can be >1 (data parallel)"; ⑤ from now design refresh takes latest version as authoritative |
| v0.4 | 2026-08-03 | Error model and buffer construction finalized (written back from 04-interfaces v0.2 design decisions): ① C2 / #4 / #6 public type list change "error code / enum ErrorCode" to `Status` struct (`static constexpr` constants SUCCESS/FAILURE/INVALID_ARGUMENT/INVALID_STATE/NOT_FOUND/CANCELED, value returned via out parameter, no StatusOr); ② #6 public type list adds `Buffer` / `ModelArtifact` / `ModelArtifactPath` (model artifacts for buffer loading and path loading, scheme C) |
| v0.5 | 2026-08-03 | Internal type naming aligned with 04-interfaces lower half (v0.3): ① C8 "lower-layer encapsulation as class" → "as abstract interface (I prefix)", `Tokenizer`/`TextEmbedding`/`Sampler`/`ModelManager`→`ITokenizer`/`ITextEmbedding`/`ISampler`/`IModelManager`, `PipelineExecutor` holds via `shared_ptr<interface>`; ② #6 internal type list merges `PrefillExecutor`/`DecodeExecutor` into `ModelExecutor` (two instances), changes to I prefix, auxiliary types (`ModelHandle`/`ModelMetaInfo`/`ModalityEntry`/`RequestImpl`/`RequestId`/`StepOutput`) point to 04 lower half; ③ #7 / #8 / C1 synchronize I prefix and abstract interface wording |
| v0.6 | 2026-08-13 | C8 v1.1 architecture convergence (linked SPEC v1.1 / 01 v0.6 / 04 v0.5): ① C8 split — `IModelManager` retained as only L2→L3 abstract interface (injection point, `shared_ptr`); `Tokenizer`/`TextEmbedding`/`Sampler` demoted to L2 internal concrete class (remove `I` prefix, no virtual functions, `PipelineExecutor` holds by value, constructible from test data); ② #6 internal type list synchronized; ③ #7 preprocessing / #8 post-processing ownership synchronized (Tokenizer/TextEmbedding/Sampler as concrete classes); ④ testability path adjusted to "IModelManager mock + 3 classes constructible from test data" (NFR-015/TC_NFR_001). |
| v0.7 | 2026-08-13 | C8 v1.2 reconciliation (fix v1.1 overshoot + write back model_executor C3): ① `Tokenizer` re-promoted to engine internal polymorphic module (`ITokenizer`+factory+BPE/SP derived self-registration, per `docs/tokenizer/`), M-02 holds via factory `unique_ptr`; ② `TextEmbedding` re-defined as engine internal embedding module (load table + bind model, per `docs/model_executor/` §4.1.1; tokenId lookup inside om graph, not lookup-produced tensor); ③ `Sampler` kept as engine internal concrete class (no polymorphism requirement); ④ Executor responsibility boundary changed to forward shell + KV ringBuffer cursor + load/unload/LoRA delegation (KV tensor memory belongs to underlying), decode batch pad to `max_batch_size` (attentionMask shields); ⑤ #7 preprocessing flow synchronized (ITokenizer, lookup inside om graph). |

| v0.8 | 2026-08-13 | Source constraint revision (#1/#2/#3): ① #1 Tokenizer error unified — remove module's own `TokenizerError`, encode/decode/create uniformly return project `Status` (C2 already defined, OOV internally maps unk→SUCCESS); ② #2 polymorphic base class `ITokenizer`→`TokenizerBase` (Base suffix replaces I prefix), BPE/SP derived + factory + Registrar synchronized; ③ #3 `PipelineExecutor` itself polymorphized — add abstract base class `PipelineExecutorBase` + `PipelineExecutorFactory` + `PipelineType` enum, current `PipelineExecutor` renamed `AutoregressivePipelineExecutor` (autoregressive, currently only derived). Document-wide `PipelineExecutor`→`PipelineExecutorBase`, `ITokenizer`→`TokenizerBase` (changelog history lines retained). Linked SPEC v0.4.0, DESIGN v0.4.0. |
| v0.9 | 2026-08-14 | Add C API layer (linked SPEC v0.5.0 / c_api SPEC v0.1.0 / DESIGN v0.1.0): provides `extern "C"` thin wrapper on top of C++ API (opaque handle + POD struct), provides ABI stable boundary for cross-language bindings. §#6 supplements C API public type list (`CANN_LMEngine_Executor` / `CANN_LMEngine_Request` opaque handle + POD struct + `CANN_LMEngine_ContextId` value type). |
| v0.11 | 2026-08-19 | Hard cut v0.6 (linked DESIGN v0.6.0): ① C8 update — L2→L3 injection from `shared_ptr<IModelManager>` runtime injection to link-time `CreateModelManager()` factory (D-12, backend determined at compile time); ② `ModelLoader` deleted from internal type list (D-11, responsibility merged into `ModelExecutor::Load`); ③ #6 internal type list, C8 synchronized update. |
