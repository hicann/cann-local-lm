# TensorManager Design Specification

English | [简体中文](../../../zh/cann_lm_engine/tensor_manager/SPEC.md)

## Overview

Design the TensorManager module for the LLM inference engine built from scratch. This module is responsible for managing tensor buffer lifecycle, shape computation and retrieval, supporting both prefill and batch decode stages of autoregressive text generation.

**Scope**: basic text generation (prefill + batch decode), static and dynamic expanding KV Cache, KV Cache Merge mode, external KV Cache injection (specinfer dual model), multi-backend memory allocation (Kirin NPU / Ascend NPU / CPU testing).

**Design goal**: extensibility first — when adding new tensors or KV Cache strategies, no need to modify TensorManager core code.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      TensorManager                        │
│                                                           │
│  Registration layer ────── RegisterTensor / RegisterKVCache           │
│       ↓ FinalizeRegistration → ForwardLayout              │
│                                                           │
│  Allocation layer ────── AllocateStatic / AllocateForPhase          │
│       ↓              AllocateKVCache                      │
│                                                           │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ STATIC bucket │  │ DYNAMIC bucket │  │ STATEFUL bucket   │  │
│  │ (1 shared)   │  │ (per seqLen) │  │ (KV Cache slots) │  │
│  └────────────┘  └─────────────┘  └──────────────────┘  │
│       ↓                ↓                ↓                 │
│  Query layer ────── GetPrefillTensors / GetDecodeTensors       │
│                  GetTensor / GetKVCache                    │
│                                                           │
│  KV Cache operations ─ Reset / Copy / Resize / UpdateLength      │
│               / ResetTensor (InitFn re-run)                 │
│                                                           │
│  External dependencies:                                                │
│    TensorAllocator                  (multi-backend memory abstraction)      │
│    NPUTensorAllocatorExtensions     (NPU-specific optimization capabilities)    │
│    KVCacheStrategy                  (static/dynamic/Merge/external) │
└──────────────────────────────────────────────────────────┘
```

**Core principle**: three bucket types stored separately by lifecycle. Declarative registration design — adding a new tensor only needs one line of Register call, no need to modify TensorManager internals.

---

## 2. Core Types

### 2.1 TensorSpec

Concentrates all knowledge of a tensor in one place declaratively.

```cpp
enum class Lifecycle {
    STATIC,             // one global copy, engine-level lifecycle (llmConfig, scales, moeEma, ...)
    SEQ_LEN_DEPENDENT,  // bucketed by seqLen (inputIds, logits, attentionMask, ...)
    STATEFUL            // managed per batch slot, conversation-level lifecycle (KV Cache, usually via RegisterKVCache)
};

enum class Role {
    INPUT,
    OUTPUT,
    INPUT_AND_OUTPUT    // output reuses input pointer (KV Cache in-place update)
};

struct ShapeContext {
    int seqLen;          // current inference sequence length (prefill: prompt token count; decode: 1)
    int batchSize;       // current batch size (prefill: 1; decode: batchSize)
    int kvCacheLen;      // current KV Cache valid length (for future extension: e.g. incremental attention
                         // dynamic shape; current version all static/dynamic tensor ShapeFn should not depend on this field
                         // to compute allocation shape, should use maxKvCacheLen)
    int maxKvCacheLen;   // KV Cache maximum capacity (attentionMask and other tensors needing capacity-based pre-allocation use this field)
};

using ShapeFn = std::function<std::vector<int64_t>(const ShapeContext&)>;

// STATEFUL bucket-specific ShapeFn: accepts additional layerIdx parameter (C5 correction)
using KVCacheShapeFn = std::function<std::vector<int64_t>(
    const ShapeContext& ctx, int layerIdx)>;

// Initialization function called immediately after allocation (C4 correction)
using InitFn = std::function<void(void* data, size_t bytes, DataType dtype)>;

struct TensorSpec {
    std::string name;           // unique identifier, e.g. "input_ids", "kv_cache_k"
    DataType dtype;             // FLOAT16, INT32, ...
    Lifecycle lifecycle;        // storage bucket type
    Role role;                  // input / output / input-output reuse
    ShapeFn shapeFn;            // Decode stage shape computation callback
    ShapeFn prefillShapeFn;     // optional: Prefill-specific shape (defaults to shapeFn)
    std::string reuseInputName; // when INPUT_AND_OUTPUT: which input tensor to reuse
    InitFn initFn;              // optional: called immediately after allocation, for filling initial values; re-run on Reset
};
```

**Design decisions**:

- **`ShapeFn` receives `ShapeContext` rather than separate parameters**: subsequent new context fields (e.g. `moeExpertNum`) require no signature change
- **`prefillShapeFn` optional**: when not provided both stages share `shapeFn`. Typical scenario: some models need different vocab align in prefill stage (e.g. when `enableLmHeadOpt` logits in prefill are `[B, 1, vocabAlign]` while decode is `[B, 1, vocabAlign]`, the two have different dimension counts), at this point sharing ShapeFn for prefill/decode would error, must provide `prefillShapeFn` and `shapeFn` separately. **attentionMask does not need `prefillShapeFn`** — although prefill/decode shapes look different (`[1,1,S,maxKV]` vs `[B,1,1,maxKV]`), the same formula `ctx.batchSize × 1 × ctx.seqLen × ctx.maxKvCacheLen` can express both (prefill: batchSize=1, seqLen=S; decode: seqLen=1)
- **`reuseInputName`**: decouples reuse declaration from index computation, looks up source tensor by name at assembly time
- **`InitFn`**: initialization function called immediately after allocation. Called after `AllocateStaticTensors` / `AllocateForPhase` allocation; also called when re-allocating after `Reset()` / `ReleaseAll()`; new `ResetTensor(name)` calls it when restoring a single tensor to initial state. KV Cache does not use InitFn (zeroing handled by `KVCacheStrategy::Reset`)
- **`KVCacheShapeFn`**: STATEFUL bucket-specific, accepts `layerIdx` parameter, supports per-layer independent shape

**ShapeFn implementation constraint**: tensor buffer capacity at allocation time must be reserved per `maxKvCacheLen`, cannot depend on `kvCacheLen`. `kvCacheLen` is only maintained at operator runtime via `UpdateSlotLength`, does not participate in allocation-time shape computation. Any ShapeFn whose output buffer needs to be incrementally written during decode (e.g. attentionMask) must use `maxKvCacheLen` as the corresponding dimension size.

**`ShapeContext` filling rules when `AllocateForPhase` called**:

| Field | PREFILL call | DECODE call |
|------|-------------|-------------|
| `seqLen` | passed-in seqLen (e.g. 128) | 1 |
| `batchSize` | 1 | passed-in batchSize |
| `kvCacheLen` | 0 (semantics: KV not yet written, invalid length) | maximum of each slot's current valid length (maintained by caller) |
| `maxKvCacheLen` | engine-configured KV Cache maximum capacity | same as above |

attentionMask's ShapeFn should be written as:

```cpp
auto attentionMaskShapeFn = [](const ShapeContext& ctx) {
    return std::vector<int64_t>{
        ctx.batchSize, 1, ctx.seqLen, ctx.maxKvCacheLen
    };
};
// prefill: ctx.seqLen=128, ctx.maxKvCacheLen=2048 → [1, 1, 128, 2048]
// decode:  ctx.seqLen=1,   ctx.maxKvCacheLen=2048 → [B, 1, 1,   2048]
```

### 2.2 KVCacheRange

Describes the KV Cache tensor's position range in forward vector. Derived by `FinalizeRegistration()` calling `KVCacheStrategy::GetSpecCount(layerNum)`.

```cpp
struct KVCacheRange {
    uint32_t inputStart;   // KV Cache start index in input vector
    uint32_t outputStart;  // KV Cache start index in output vector
    uint32_t count;        // total KV Cache tensor count
                           // non-Merge: layerNum*2
                           // Merge: 1
                           // external injection: decided by ExternalKVCacheStrategy
};
```

### 2.3 KVCacheSlot

Each batch's KV Cache state.

```cpp
struct KVCacheSlot {
    std::vector<NDTensorBufferPtr> layerCaches;  // per-layer buffer
                                                  // non-Merge: [key_0, value_0, key_1, value_1, ...]
                                                  // Merge: [bigBuffer] (1 element)
    int currentLen = 0;   // this slot's current valid length
    int capacity = 0;     // this slot's allocated capacity
};
```

---

## 3. Three-Bucket Storage

| Bucket | Key | Lifecycle | Creation interface | Destruction interface | Typical tensor |
|----|-----|---------|---------|---------|-----------|
| STATIC | tensor name | engine-level | `AllocateStaticTensors()` | `ReleaseAll()` / engine destruction | llmConfig, scales, moeConfig, moeEma, moeLambdaBias, enable_lora |
| DYNAMIC | (seqLen, tensor name) | inference round | `AllocateForPhase()` | `ReleasePhase()` | inputIds, attentionMask, logits, positionIds, history_activation_mask |
| STATEFUL | (slotIdx, layerIdx) | conversation-level | `AllocateKVCache()` | `ResetKVCache()` / `ReleaseAll()` | KV Cache key/value |

**Why not use one `map<seqLen, vector>` for unified storage**:

1. **Memory**: static tensors shared across all seqLen tiers, rather than referenced multiple times via shared_ptr
2. **Independent lifecycle**: KV Cache Reset at conversation end; static tensors released at engine destruction; dynamic tensors releasable between inference rounds. Mixed storage makes it impossible to independently release one category
3. **Extensibility**: adding new KV Cache strategies only affects STATEFUL bucket code path, does not involve static and dynamic allocation logic

---

## 4. ForwardLayout — Assembly Blueprint

Computed once in `FinalizeRegistration()`. Drives all `Get*Tensors` calls, no conditional branches needed.

```cpp
// BucketType corresponds to Lifecycle: at assembly time indicates which bucket to fetch tensor from
enum class BucketType { STATIC, DYNAMIC, STATEFUL };

// KVRole: distinguishes key and value in KV Cache (used when looking up per layer)
enum class KVRole { KEY, VALUE };

struct ForwardLayout {
    struct Entry {
        std::string name;        // tensor semantic name
        BucketType  source;      // STATIC / DYNAMIC / STATEFUL
        std::string sourceName;  // actual tensor name fetched (differs from name when reused)
        int         kvLayerIdx;  // STATEFUL-specific: which layer, -1 means non-KV
        KVRole      kvRole;      // STATEFUL-specific: KEY / VALUE
    };

    std::vector<Entry> inputEntries;
    std::vector<Entry> outputEntries;
    KVCacheRange kvRange;
};
```

**Assembly process in GetDecodeTensors**:

```cpp
for (size_t i = 0; i < layout_.inputEntries.size(); i++) {
    auto& entry = layout_.inputEntries[i];
    result.inputs[i] = ResolveTensor(entry, ctx, activeSlots);
}
```

No need to write if-else by tensor type. Layout encodes each tensor's source. When adding new tensor via `RegisterTensor`, layout auto-updates, assembly loop unchanged.

**`kvRange.count` derivation**: decided by `KVCacheStrategy::GetSpecCount(layerNum)`. Non-Merge strategy returns `layerNum * 2`; Merge strategy returns 1; external injection strategy per external buffer count.

---

## 5. Complete Interface

### 5.1 Registration

| Interface | Purpose |
|------|------|
| `RegisterTensor(spec)` | Register single tensor spec |
| `RegisterKVCache(prefix, layerNum, dtype, shapeFn, batchSize, strategy)` | Batch register KV Cache (expansion count decided by `strategy->GetSpecCount()`) |
| `FinalizeRegistration()` | Validate integrity, derive ForwardLayout, compute KVCacheRange |

**`RegisterKVCache` signature (C5 correction)**:

```cpp
hiai::Status RegisterKVCache(
    const std::string& prefix,
    int layerNum,
    DataType dtype,
    const KVCacheShapeFn& shapeFn,   // ← accepts layerIdx, supports per-layer independent shape
    int batchSize,
    std::shared_ptr<KVCacheStrategy> strategy);
```

**Calling convention**: all registration done once at engine initialization, before allocation. `FinalizeRegistration` must be called after all `Register*`, before all `Allocate*`.

**Multiple attentionMask / multiple newKvCachePos scenarios**: original scheme `InputIndex::attentionMaskIdx` and `InputIndex::newKvCachePosIdx` are both `std::vector<uint32_t>`, one-to-one corresponding with `LLMConfigOption::kvCacheLenList`, for supporting sink + rollback and other multi-sliding-window configurations. This spec expresses via **loop registering multiple independent specs** (named `attention_mask_0`, `attention_mask_1`, etc.), consistent with `RegisterKVCache`'s expansion pattern. Each independent spec's ShapeFn captures its own index `i` via lambda to select corresponding `kvCacheLenList[i]`, no need to extend `RegisterTensor` interface.

### 5.2 Allocation

| Interface | Purpose |
|------|------|
| `AllocateStaticTensors()` | Allocate all STATIC tensors (one of each), call `initFn` after allocation |
| `AllocateForPhase(phase, seqLen, batchSize)` | Allocate DYNAMIC tensors for specified phase/seqLen/batch, call `initFn` after allocation |
| `AllocateKVCache(maxKvCacheLen, batchSize)` | Allocate all KV Cache slots (delegates to strategy) |
| `ReleasePhase(phase, seqLen)` | Release DYNAMIC tensors for specified phase/seqLen |

**Phase enum**:

```cpp
enum class Phase { PREFILL, DECODE };
```

### 5.3 Query

| Interface | Purpose |
|------|------|
| `GetPrefillTensors(seqLen, slotIdx=0)` | Get prefill's complete IO set (batch=1), KV output written to specified slot |
| `GetDecodeTensors(batchSize, activeSlotIndices)` | Get batch decode's complete IO set (seqLen=1) |
| `GetTensor(name, seqLen=0)` | Get single tensor by name |
| `GetKVCache(slotIdx, layerIdx, role)` | Get single KV Cache layer buffer (delegates to `strategy->GetLayerView`) |
| `GetKVCacheSlot(slotIdx)` | Get all buffers of a slot (for serialization) |

**About slotIdx parameter**: Prefill directly writes KV Cache to specified slot's buffer (in-place update). After `model.Run` completes KV data already in that slot, no extra copy step needed. `CopyKVCacheBetweenSlots` only needed when moving data between slots.

**`GetKVCache` implementation**:

```cpp
NDTensorBufferPtr TensorManager::GetKVCache(int slotIdx, int layerIdx, KVRole role) {
    if (slotIdx < 0 || slotIdx >= static_cast<int>(slots_.size())) return nullptr;
    return strategy_->GetLayerView(slots_[slotIdx], layerIdx, role);
}
```

**Core code has no if branches** — all Merge / non-Merge differences absorbed by strategy.

**Return type**:

```cpp
struct ForwardTensors {
    std::vector<NDTensorBufferPtr> inputs;
    std::vector<NDTensorBufferPtr> outputs;
};
```

### 5.4 KV Cache Operations

| Interface | Purpose |
|------|------|
| `ResetKVCache()` | Clear all slots, currentLen set to 0 (delegates to `strategy->Reset`) |
| `ResetKVCacheSlot(slotIdx)` | Clear specified slot |
| `CopyKVCacheBetweenSlots(srcSlotIdx, dstSlotIdx)` | Copy between slots (prefix reuse/multi-turn dialogue) |
| `ResizeKVCache(newKvCacheLen)` | Expand all slots to new capacity (delegates to strategy) |
| `UpdateSlotLength(slotIdx, newLen)` | Update slot valid length (called after each decode step) |
| `GetSlotLength(slotIdx)` | Query slot valid length |
| `GetSlotCapacity(slotIdx)` | Query slot allocated capacity |
| `ResetTensor(name)` | Reset single STATIC/DYNAMIC tensor to initial state (re-run `initFn`) |

### 5.5 Metadata Query

| Interface | Purpose |
|------|------|
| `HasTensor(name, seqLen=0)` | Check if tensor is allocated |
| `GetTensorBytes(name, seqLen=0)` | Get tensor byte count |
| `GetAllocatedPrefillSeqLens()` | List allocated prefill seqLen tiers |
| `GetKVCacheRange()` | Get KV Cache index range in forward vector |
| `GetMemoryUsage()` | Get memory statistics bucketed by category |

```cpp
struct MemoryUsage {
    size_t staticBytes;     // static tensor memory
    size_t dynamicBytes;    // dynamic tensor memory (all seqLen tiers total)
    size_t kvCacheBytes;    // KV Cache memory (all slots total)
};
```

### 5.6 Lifecycle

| Interface | Purpose |
|------|------|
| `ReleaseAll()` | Release all tensors, retain specs (can re-allocate) |
| `Reset()` | Clear all state including specs (back to post-construction state) |

**Total interface count: 28** (original spec 27 + C4 added `ResetTensor`).

---

## 6. Multi-Backend Memory Abstraction

### TensorAllocator Interface

```cpp
class TensorAllocator {
public:
    virtual ~TensorAllocator() = default;

    // Allocate single tensor buffer
    virtual NDTensorBufferPtr Allocate(
        const std::string& name,
        const std::vector<int64_t>& dims,
        DataType dtype) = 0;

    // Batch allocation (non-pure virtual, default implementation loops single; NPU backend can override to batch API)
    virtual std::vector<NDTensorBufferPtr> AllocateBatch(
        const std::vector<std::string>& names,
        const std::vector<std::vector<int64_t>>& dims,
        const std::vector<DataType>& dtypes);

    // NoCopy creation: external provides memory, wraps as tensor buffer
    virtual NDTensorBufferPtr AllocateNoCopy(
        const std::string& name,
        const std::vector<int64_t>& dims,
        DataType dtype,
        void* externalPtr,
        size_t size) = 0;

    // Destroy
    virtual void Deallocate(NDTensorBufferPtr& buffer) = 0;

    // Backend identifier
    virtual std::string GetBackendName() const = 0;
};
```

### Backend List

| Backend | Type | Status | Purpose |
|------|------|------|------|
| `KirinNPUTensorAllocator` | Main path | Current implementation | Kirin NPU production path, encapsulates `HIAI_MR_NDTensorBuffer_*` series APIs. **Single class two compilation forms**: `__OHOS__` macro determines NDK interface or ROM interface, no runtime switch |
| `AscendNPUTensorAllocator` | Main path | Future implementation | Ascend NPU, reserved interface, encapsulates `aclrt*` / `aclCreateTensor*` series APIs |
| `CPUTensorAllocator` | Test path | Optional implementation | Only for `lm_engine_test` compile target, used in unit tests, **does not appear in production build** |
| ~~`GPUTensorAllocator`~~ | — | **Deleted** | YAGNI, no real requirement |

**About Kirin NDK vs ROM**: original scheme `lm_engine_mmgr.h:8-22` uses `#ifdef __OHOS__` to bind same set of API names to NDK or ROM implementation at compile time. This spec retains this mechanism, but encapsulated inside `KirinNPUTensorAllocator` — allocator is same C++ class externally, internally selects which ROM/NDK function to call based on compilation environment. NDK/ROM differences invisible to `TensorManager` and `KVCacheStrategy`.

**Construction injection example**:

```cpp
// Kirin NPU (current)
auto tm = TensorManager(std::make_shared<KirinNPUTensorAllocator>());

// Ascend NPU (future)
auto tm = TensorManager(std::make_shared<AscendNPUTensorAllocator>());

// Unit testing
auto tm = TensorManager(std::make_shared<CPUTensorAllocator>());
```

**NoCopy path**: when external provides memory (e.g. shared KV Cache memory from other process), `AllocateNoCopy` wraps existing pointer as tensor buffer, no data copy occurs.

**CPU Allocator implementation constraint**: must implement its own `INDTensorBuffer` interface (cannot reuse original scheme's `NDTensorBufferImpl`, because it strongly binds NPU ROM handle):

- `CPUTensorBufferImpl`: internal `std::vector<uint8_t>` stores data, implements `GetData() / GetSize() / SetCacheStatus() / GetCacheStatus()` etc.
- `SetCacheStatus` / `GetCacheStatus` returns 0 / no-op in CPU path
- `GetFd()` returns -1 in CPU path (decoder offload not supported)

CPU allocator implementation location: `test/mock/cpu_tensor_allocator.cpp`, **not in production build**.

---

## 7. NPU-Specific Capability Extension

NPU backend may provide `TensorAllocator` subclass also implementing `NPUTensorAllocatorExtensions`. Internal modules like `KVCacheStrategy` get extension interface via `dynamic_cast`; when extension interface unavailable goes fallback or rejects. CPU allocator does not implement extension interface.

### NPUTensorAllocatorExtensions Interface

```cpp
class NPUTensorAllocatorExtensions {
public:
    virtual ~NPUTensorAllocatorExtensions() = default;

    // KV Cache optimized allocation: cross-layer one-time allocation, tile-aligned
    // corresponds to original scheme CreateKVCacheBufferFromNDK
    virtual NDTensorBufferPtr AllocateKVCache(
        const std::vector<int64_t>& dims,
        DataType dtype,
        int kvCacheNum,
        int tileSize) = 0;

    // KV Cache expansion: in-place expand existing buffer
    // corresponds to original scheme ResizeKVCacheBufferFromNDK
    virtual Status Resize(NDTensorBufferPtr& buffer, size_t newSize) = 0;

    // Set/get hardware cache status (for cross-process KV Cache sharing)
    virtual void SetCacheStatus(NDTensorBufferPtr& buffer, uint8_t status) = 0;
    virtual uint8_t GetCacheStatus(const NDTensorBufferPtr& buffer) const = 0;

    // Get file descriptor (for decoder offload cross-process zero-copy)
    // unsupported backends return -1
    virtual int32_t GetFd(const NDTensorBufferPtr& buffer) const = 0;
};
```

### Relationship Between Extension Interface and Core Interface

```cpp
class KirinNPUTensorAllocator
    : public TensorAllocator,
      public NPUTensorAllocatorExtensions {
    // simultaneously implements both sets of interfaces
public:
    // TensorAllocator
    NDTensorBufferPtr Allocate(...) override;
    NDTensorBufferPtr AllocateNoCopy(...) override;
    void Deallocate(NDTensorBufferPtr&) override;
    std::string GetBackendName() const override { return "KirinNPU"; }

    // NPUTensorAllocatorExtensions
    NDTensorBufferPtr AllocateKVCache(...) override;  // calls HIAI_MR NDK API
    Status Resize(NDTensorBufferPtr&, size_t) override;
    void SetCacheStatus(...) override;
    uint8_t GetCacheStatus(...) const override;
    int32_t GetFd(...) const override;
};

class AscendNPUTensorAllocator
    : public TensorAllocator,
      public NPUTensorAllocatorExtensions {
    // future implementation, corresponds to aclrt* API
};

class CPUTensorAllocator : public TensorAllocator {
    // only implements core interface, does not implement Extensions
    // for unit testing
};
```

### Compatibility Matrix

| Allocator | Core interface | Extension interface | Static KV strategy | Dynamic KV strategy | CPU testing |
|----------|---------|---------|-------------|-------------|---------|
| `KirinNPUTensorAllocator` (NDK compile) | ✅ | ✅ | ✅ | ✅ | — |
| `KirinNPUTensorAllocator` (ROM compile) | ✅ | ⚠️ (`AllocateKVCache`/`Resize` returns unsupported) | ✅ | ❌ | — |
| `AscendNPUTensorAllocator` | ✅ | ✅ (future) | ✅ | ✅ (future) | — |
| `CPUTensorAllocator` | ✅ | ❌ | ✅ (fallback) | ❌ (no optimization API) | ✅ |

Note: Kirin ROM compile path `Extensions::AllocateKVCache` and `Resize` return `nullptr` / `ERROR` (consistent with original scheme `from_rom.cpp:144, 156`). `DynamicKVCacheStrategy` should fallback to generic `Allocate` when `ext->AllocateKVCache` returns nullptr, or directly error to let upper layer choose static strategy.

### Performance Notes

`dynamic_cast` in `KVCacheStrategy::Allocate` only happens once (at initialization), not in decode hot path. `Resize` only called when expanding tier (rare), performance negligible.

---

## 8. KV Cache Strategy Pattern

### KVCacheStrategy Interface

```cpp
class KVCacheStrategy {
public:
    virtual ~KVCacheStrategy() = default;

    // Allocate KV Cache buffers to slots
    virtual Status Allocate(TensorAllocator& allocator, int batchSize, int layerNum,
                            int headNum, int headDim, const KVCacheShapeFn& shapeFn,
                            const ShapeContext& ctx, std::vector<KVCacheSlot>& slots) = 0;

    // Expand
    virtual Status Resize(TensorAllocator& allocator, int newKvCacheLen,
                          const KVCacheShapeFn& shapeFn, const ShapeContext& ctx,
                          std::vector<KVCacheSlot>& slots) = 0;

    // Zero
    virtual Status Reset(std::vector<KVCacheSlot>& slots) = 0;

    // Identifier
    virtual std::string GetName() const = 0;

    // Spec count expanded at registration (determines kvRange.count)
    virtual int GetSpecCount(int layerNum) const = 0;

    // Get actual buffer by (layerIdx, role)
    // Merge strategy returns view of big buffer; non-Merge strategy returns slot.layerCaches[layerIdx*2 + role]
    virtual NDTensorBufferPtr GetLayerView(
        const KVCacheSlot& slot, int layerIdx, KVRole role) const = 0;
};
```

### Strategy List

| Strategy | Purpose | spec count | Expand | Merge | External injection |
|------|------|--------|------|-------|---------|
| `StaticKVCacheStrategy` | One-time allocate `maxKvCacheLen` full capacity | `layerNum*2` | ❌ | ❌ | ❌ |
| `DynamicKVCacheStrategy` | Accepts expansion tier list, initial lowest tier | `layerNum*2` | ✅ | ❌ | ❌ |
| `MergedKVCacheStrategy` (static) | All layers merged into 1 big buffer | 1 | ❌ | ✅ | ❌ |
| `MergedKVCacheStrategy` (dynamic) | Merge + dynamic expansion combination | 1 | ✅ | ✅ | ❌ |
| `ExternalKVCacheStrategy` | Accepts external buffer, does not call allocator | decided by external buffer count | ❌ | composable | ✅ |
| `PagedKVCacheStrategy` (future) | vLLM-style paging | TBD | TBD | composable | ❌ |

### StaticKVCacheStrategy

- One-time allocate `maxKvCacheLen` full capacity
- `Resize` returns error (expansion not supported)
- `Reset` zeros all buffers, `currentLen` set to 0
- `GetSpecCount` returns `layerNum * 2`
- `GetLayerView` returns `slot.layerCaches[layerIdx * 2 + (role == KEY ? 0 : 1)]`

```cpp
int StaticKVCacheStrategy::GetSpecCount(int layerNum) const {
    return layerNum * 2;
}

NDTensorBufferPtr StaticKVCacheStrategy::GetLayerView(
    const KVCacheSlot& slot, int layerIdx, KVRole role) const {
    return slot.layerCaches[layerIdx * 2 + (role == KVRole::KEY ? 0 : 1)];
}
```

### DynamicKVCacheStrategy

- Accepts expansion tier list, e.g. `{256, 512, 1024, 2048}`
- Initial allocation uses lowest tier
- `Resize` finds next tier satisfying requested length, allocates new buffer, copies existing data, replaces old buffer
- `Reset` same as static strategy
- Gets extension interface via `dynamic_cast<NPUTensorAllocatorExtensions*>`; rejects when no extension

```cpp
class DynamicKVCacheStrategy : public KVCacheStrategy {
public:
    DynamicKVCacheStrategy(std::vector<int> steps) : steps_(std::move(steps)) {}

    Status Allocate(TensorAllocator& allocator, ...) override {
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        if (ext != nullptr) {
            // NPU path: use optimized allocation
            for (auto& slot : slots) {
                for (int layer = 0; layer < layerNum; layer++) {
                    // layerIdx passed to KVCacheShapeFn, allows per-layer shape independence
                    auto buf = ext->AllocateKVCache(
                        shapeFn(ctx, layer), dtype, 1, tileSize);
                    slot.layerCaches.push_back(buf);
                }
            }
        } else {
            // CPU / backends not supporting extension: fallback to generic Allocate
            for (auto& slot : slots) {
                for (int layer = 0; layer < layerNum; layer++) {
                    auto buf = allocator.Allocate(name, shapeFn(ctx, layer), dtype);
                    slot.layerCaches.push_back(buf);
                }
            }
        }
        return SUCCESS;
    }

    Status Resize(TensorAllocator& allocator, int newKvCacheLen, ...) override {
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        if (ext == nullptr) return ERROR;  // backends not supporting expansion directly reject
        auto it = std::upper_bound(steps_.begin(), steps_.end(), newKvCacheLen);
        if (it == steps_.end()) return ERROR;
        int target = *it;
        for (auto& slot : slots) {
            for (auto& buf : slot.layerCaches) {
                HIAI_EXPECT_EXEC(ext->Resize(buf, target));
            }
        }
        return SUCCESS;
    }

private:
    std::vector<int> steps_;
};
```

### MergedKVCacheStrategy

All layers merged into **1 contiguous big buffer**, internal layout `|L0K|L0V|L1K|L1V|...|LnK|LnV|`, exposed externally as single tensor. Corresponds to original scheme `isKvCacheMerge=true`.

**Merge and Dynamic are two orthogonal dimensions**: Merge controls "1 big buffer vs layerNum*2 independent buffers" (layout dimension), Dynamic controls "whether to expand by tiers" (expansion dimension). Evidence from `tensor_manager.cpp:1319-1332`: `enableDynamicKvCache && isKvCacheMerge` takes effect in same path. `MergedKVCacheStrategy` internally optionally supports expansion (passes tier list at construction).

```cpp
class MergedKVCacheStrategy : public KVCacheStrategy {
public:
    // static capacity (no expansion)
    explicit MergedKVCacheStrategy(uint32_t shareLayerNum = 0)
        : shareLayerNum_(shareLayerNum), dynamicSteps_() {}

    // dynamic expansion (by tiers)
    MergedKVCacheStrategy(uint32_t shareLayerNum, std::vector<int> steps)
        : shareLayerNum_(shareLayerNum), dynamicSteps_(std::move(steps)) {}

    int GetSpecCount(int layerNum) const override { return 1; }

    Status Allocate(TensorAllocator& allocator, int batchSize, int layerNum,
                    int headNum, int headDim, const KVCacheShapeFn& shapeFn,
                    const ShapeContext& ctx, std::vector<KVCacheSlot>& slots) override {
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        int effectiveLayerNum = layerNum - shareLayerNum_;
        int tileSize = 16;  // NZ format; ND format uses headDim
        int kvCacheNum = effectiveLayerNum * 2;

        int initialCapacity = dynamicSteps_.empty()
            ? ctx.maxKvCacheLen
            : dynamicSteps_.front();

        // Merge mode has only 1 spec, layerIdx passed 0 means overall big buffer shape
        for (int s = 0; s < batchSize; s++) {
            KVCacheSlot slot;
            slot.capacity = initialCapacity;
            slot.currentLen = 0;
            if (ext != nullptr) {
                auto buf = ext->AllocateKVCache(
                    shapeFn(ctx, 0), dtype, kvCacheNum, tileSize);
                if (!dynamicSteps_.empty()) {
                    HIAI_EXPECT_EXEC(ext->Resize(buf, initialCapacity));
                }
                slot.layerCaches.push_back(buf);
            } else {
                auto buf = allocator.Allocate("kv_cache_merged", shapeFn(ctx, 0), dtype);
                slot.layerCaches.push_back(buf);
            }
            slots.push_back(std::move(slot));
        }
        return SUCCESS;
    }

    NDTensorBufferPtr GetLayerView(const KVCacheSlot& slot, int layerIdx, KVRole role) const override {
        // big buffer internal layout: |L0K|L0V|L1K|L1V|...|LnK|LnV|
        size_t totalLayers = /* computed from construction-time layerNum - shareLayerNum_ */;
        size_t layerSize = slot.layerCaches[0]->GetSize() / (totalLayers * 2);
        size_t offset = (layerIdx * 2 + (role == KVRole::KEY ? 0 : 1)) * layerSize;
        return MakeBufferView(slot.layerCaches[0], offset, layerSize);
    }

    Status Reset(std::vector<KVCacheSlot>& slots) override {
        for (auto& slot : slots) {
            for (auto& buf : slot.layerCaches) {
                std::memset(buf->GetData(), 0, buf->GetSize());
            }
            slot.currentLen = 0;
        }
        return SUCCESS;
    }

    Status Resize(TensorAllocator& allocator, int newKvCacheLen,
                  const KVCacheShapeFn& shapeFn, const ShapeContext& ctx,
                  std::vector<KVCacheSlot>& slots) override {
        if (dynamicSteps_.empty()) return ERROR;
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        if (ext == nullptr) return ERROR;
        auto it = std::upper_bound(dynamicSteps_.begin(), dynamicSteps_.end(), newKvCacheLen);
        if (it == dynamicSteps_.end()) return ERROR;
        int target = *it;
        for (auto& slot : slots) {
            for (auto& buf : slot.layerCaches) {
                HIAI_EXPECT_EXEC(ext->Resize(buf, target));
            }
            slot.capacity = target;
        }
        return SUCCESS;
    }

    std::string GetName() const override {
        return dynamicSteps_.empty() ? "MergedStatic" : "MergedDynamic";
    }

private:
    uint32_t shareLayerNum_;
    std::vector<int> dynamicSteps_;
};
```

### ExternalKVCacheStrategy

Accepts external buffer (from other process, shared memory, or specinfer target model's generated KV Cache), does not call allocator. Corresponds to original scheme `kvCacheSwitchableFlag_=true`.

**Adopts "construct empty strategy + later `Inject`" pattern** (scheme C), aligns with original scheme timing: `kvCacheSwitchableFlag_=true` set at Init, but actual buffer injected only after target runs.

```cpp
class ExternalKVCacheStrategy : public KVCacheStrategy {
    std::vector<NDTensorBufferPtr> externalBuffers_;  // later injected
public:
    int GetSpecCount(int layerNum) const override { return externalBuffers_.size(); }
    Status Allocate(...) override { /* create empty slot, do not allocate buffer */ }

    // later inject external buffer
    Status Inject(std::vector<NDTensorBufferPtr> buffers) {
        externalBuffers_ = std::move(buffers);
        // write buffers to all slots' layerCaches
    }

    NDTensorBufferPtr GetLayerView(...) override {
        /* index externalBuffers_ by layerIdx */
    }
    Status Resize(...) override { return ERROR; }  // expansion not supported
    Status Reset(...) override { /* do not zero external memory */ return SUCCESS; }
};
```

Caller:

```cpp
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
tm.RegisterKVCache("kv_cache", layerNum, dtype, shapeFn, batchSize, extStrategy);
// ... target model runs complete ...
extStrategy->Inject(targetKvBuffers);  // later injection
```

### Compatibility Matrix

| Allocator / strategy combination | Static capacity | Dynamic expansion | External injection | Merge | Multi-model sharing |
|---------------------|---------|---------|---------|-------|----------|
| `StaticKVCacheStrategy` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `DynamicKVCacheStrategy` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `MergedKVCacheStrategy` (static) | ✅ | ❌ | ❌ | ✅ | ❌ |
| `MergedKVCacheStrategy` (dynamic) | ✅ | ✅ | ❌ | ✅ | ❌ |
| `ExternalKVCacheStrategy` | — | ❌ | ✅ | composable | ✅ specinfer |

### KV Cache Merge Constraints

1. **NZ + multiple kvCacheLen**: not supported (NPU operator layer constraint, hard constraint)
2. **Merge + Dynamic**: the two are not mutually exclusive, can coexist as `MergedKVCacheStrategy` (dynamic tier version)
3. **ND + Merge**: original scheme `pipeline_executor_kvcahe.cpp:113` prohibits, but the restriction is in prefix cache size computation helper function layer, not in allocation/operator layer; after spec rewrite this restriction can be lifted
4. **Merge and multiple attentionMask**: Merge mode KV Cache is 1 big buffer, conflicts with `kvCacheLenList.size() > 1` — **recommend Merge mode `kvCacheLenList.size() == 1`**

These constraints are declared via documentation (not enforced by code), caller guarantees combination legality.

### Selecting Strategy at Registration

```cpp
// Static strategy (non-Merge)
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<StaticKVCacheStrategy>());

// Dynamic expansion strategy (non-Merge)
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<DynamicKVCacheStrategy>({256, 512, 1024, 2048}));

// Merge static strategy
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<MergedKVCacheStrategy>(shareLayerNum));

// Merge dynamic expansion strategy
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<MergedKVCacheStrategy>(shareLayerNum, {256, 512, 1024, 2048}));

// External injection (specinfer dual model)
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4, extStrategy);
// after target runs: extStrategy->Inject(targetKvBuffers);

// Future: paging strategy
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<PagedKVCacheStrategy>(pageSize=16));
```

Adding new KV Cache strategy only requires implementing `KVCacheStrategy` interface, TensorManager core code needs no changes.

### STATEFUL ShapeFn Typical Usage

```cpp
// Same shape per layer (most common)
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [](const ShapeContext& ctx, int layerIdx) {
        return std::vector<int64_t>{
            ctx.batchSize, ctx.maxKvCacheLen, headNum, headDim};
    }, batchSize, strategy);

// Sink + Rollback: first N layers use sink length, rest use maxKvCacheLen
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [sinkLen, sinkLayerNum](const ShapeContext& ctx, int layerIdx) {
        int kvLen = (layerIdx < sinkLayerNum) ? sinkLen : ctx.maxKvCacheLen;
        return std::vector<int64_t>{
            ctx.batchSize, kvLen, headNum, headDim};
    }, batchSize, strategy);

// Multiple kvCacheLenList: map to different capacities per layer
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [kvCacheLenList, layerToLenIdx](const ShapeContext& ctx, int layerIdx) {
        int kvLen = kvCacheLenList[layerToLenIdx[layerIdx]];
        return std::vector<int64_t>{
            ctx.batchSize, kvLen, headNum, headDim};
    }, batchSize, strategy);
```

---

## 9. Initialization and Usage Flow

### 9.1 Engine Initialization

```cpp
// 1. Construct with injected backend allocator
auto tm = TensorManager(std::make_shared<KirinNPUTensorAllocator>());

// 2. Register all tensor specs (including InitFn)
tm.RegisterTensor({
    .name = "moe_map_info_0",
    .lifecycle = STATIC,
    .role = INPUT,
    .shapeFn = [](const ShapeContext& ctx) { return std::vector<int64_t>{expertNum}; },
    .initFn = [](void* data, size_t bytes, DataType) {
        int64_t* p = static_cast<int64_t*>(data);
        std::fill(p, p + bytes / sizeof(int64_t), -1);
    }
});
tm.RegisterTensor({
    .name = "moe_lambda_bias",
    .lifecycle = STATIC,
    .role = INPUT,
    .shapeFn = [](const ShapeContext& ctx) { return std::vector<int64_t>{1}; },
    .initFn = [lambdaBias](void* data, size_t bytes, DataType) {
        CvtFp32ToFp16(&lambdaBias, static_cast<uint16_t*>(data), 1);
    }
});
tm.RegisterTensor({
    .name = "history_activation_mask",
    .lifecycle = SEQ_LEN_DEPENDENT,
    .role = INPUT,
    .shapeFn = [](const ShapeContext& ctx) { return std::vector<int64_t>{ctx.batchSize, intermediateSize}; },
    .initFn = [](void* data, size_t bytes, DataType) {
        int32_t* p = static_cast<int32_t*>(data);
        std::fill(p, p + bytes / sizeof(int32_t), 0);
    }
});

// Multiple attentionMask (loop registration)
for (size_t i = 0; i < kvCacheLenList.size(); i++) {
    tm.RegisterTensor({
        .name = "attention_mask_" + std::to_string(i),
        .lifecycle = SEQ_LEN_DEPENDENT,
        .role = INPUT,
        .shapeFn = [i, kvCacheLenList](const ShapeContext& ctx) {
            return std::vector<int64_t>{
                ctx.batchSize, 1, ctx.seqLen, kvCacheLenList[i]
            };
        }
    });
}

// Register KV Cache (select strategy)
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [kvCacheLenList](const ShapeContext& ctx, int layerIdx) {
        return std::vector<int64_t>{
            ctx.batchSize, ctx.maxKvCacheLen, headNum, headDim};
    }, batchSize, std::make_shared<StaticKVCacheStrategy>());

// 3. Finalize registration (validate, derive layout)
tm.FinalizeRegistration();

// 4. Allocate
tm.AllocateStaticTensors();  // after allocation auto-calls each spec's initFn
tm.AllocateKVCache(maxKvCacheLen, batchSize);
tm.AllocateForPhase(PREFILL, seqLen, batchSize=1);
tm.AllocateForPhase(DECODE, seqLen=1, batchSize);
```

### 9.2 Prefill

```cpp
// Prefill directly writes KV data to slot 0 (via STATEFUL bucket in-place update)
auto tensors = tm.GetPrefillTensors(seqLen, /*slotIdx=*/0);
FillInputIds(tensors, promptTokens);       // get data via GetTensor("input_ids", seqLen)->GetData()
FillAttentionMask(tensors, seqLen, maxKvCacheLen);
// actually fill values per [1,1,S,maxKV] layout, positions beyond S write -FLT_MAX
model.Run(tensors.inputs, tensors.outputs);
// KV data already in slot 0's layerCaches, no extra copy needed
tm.UpdateSlotLength(/*slotIdx=*/0, seqLen);
```

### 9.3 Batch Decode

```cpp
auto tensors = tm.GetDecodeTensors(batchSize, {0, 1, 2, 3});
FillDecodeInputIds(tensors, {tokenId0, tokenId1, ...});
model.Run(tensors.inputs, tensors.outputs);
for (int slot : activeSlots) {
    tm.UpdateSlotLength(slot, tm.GetSlotLength(slot) + 1);
}
```

### 9.4 Multi-turn Dialogue

```cpp
// Turn 2: reuse same slot's KV Cache, no reset needed
auto tensors = tm.GetDecodeTensors(1, {slotIdx});
// ...(same as batch decode)

// Conversation ends: reset slot
tm.ResetKVCacheSlot(slotIdx);
```

### 9.5 Dynamic KV Cache Expansion

```cpp
// During decode, when valid length approaches capacity limit trigger expansion
if (tm.GetSlotLength(slotIdx) >= tm.GetSlotCapacity(slotIdx) - 1) {
    tm.ResizeKVCache(tm.GetSlotCapacity(slotIdx) + 1);  // strategy auto-finds next tier
}
// Example: current capacity=256, strategy tiers={256,512,1024,2048}
// ResizeKVCache(257) → strategy finds nextStep=512, expands to 512
```

### 9.6 specinfer Dual Model KV Cache Sharing

```cpp
// target model and draft model share KV Cache
// 1. target uses normal strategy allocation
auto targetTm = TensorManager(/* KirinNPU allocator */);
targetTm.RegisterKVCache(..., std::make_shared<StaticKVCacheStrategy>());
targetTm.FinalizeRegistration();
targetTm.AllocateKVCache(maxKvCacheLen, batchSize);

// 2. draft uses ExternalKVCacheStrategy, placeholder
auto draftTm = TensorManager(/* KirinNPU allocator */);
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
draftTm.RegisterKVCache(..., extStrategy);
draftTm.FinalizeRegistration();
draftTm.AllocateKVCache(maxKvCacheLen, batchSize);  // creates empty slot

// 3. after target runs prefill
auto targetKvBuffers = targetTm.GetKVCacheSlot(/*slotIdx=*/0);

// 4. inject to draft
extStrategy->Inject(targetKvBuffers);

// 5. draft directly reuses target's KV, no need to recompute
auto draftTensors = draftTm.GetDecodeTensors(batchSize, {0});
model.Run(draftTensors.inputs, draftTensors.outputs);
```

### 9.7 Reset Single Tensor (Corresponds to Original Scheme ResetMoeConfigs)

```cpp
// equivalent to original scheme ResetMoeConfigs
tm.ResetTensor("moe_map_info_0");
tm.ResetTensor("moe_cache_info_0");
tm.ResetTensor("moe_ema");
tm.ResetTensor("moe_lambda_bias");

// or optional batch interface
tm.ResetTensorsByPrefix("moe_");
```

---

## 10. Error Handling

- All methods returning `Status` use `hiai::Status` (SUCCESS / ERROR codes)
- `FinalizeRegistration` validates: no duplicate names, complete role coverage, `INPUT_AND_OUTPUT`'s `reuseInputName` valid, KV Cache registered before finalize
- `GetTensor` / `GetKVCache` returns `nullptr` when not found or not allocated (no exception thrown)
- `AllocateForPhase` skips already-allocated tensors (idempotent)
- `ResizeKVCache` returns error when strategy does not support expansion (static strategy)
- `KVCacheStrategy::Allocate` does not check combination legality (e.g. Merge + multiple kvCacheLen), when caller passes wrong combination operator layer errors

---

## 11. Thread Safety

TensorManager **is not thread-safe by design**.

Reason: inference engine typically uses single scheduling thread to orchestrate prefill and decode. Lock-free design avoids hot path overhead. If concurrent access needed (e.g. async KV Cache loading), external synchronization by caller.

---

## 12. Comparison with Original Implementation

| Dimension | Original implementation | Redesigned |
|------|---------|---------|
| Add new tensor | modify 5+ places | add 1 line Register call |
| Index management | manual `currentIdx++` | auto-derived from registration order |
| Tensor storage | `map<seqLen, vector>` (each seqLen stores full) | three buckets separated by lifecycle |
| KV Cache mode | if-else chain in creation method | strategy pattern (Static / Dynamic / Merged / External / Paged) |
| Memory backend | compile-time NDK/ROM macro switch (only Kirin internal two deployments) | runtime inject TensorAllocator (Kirin NPU / Ascend NPU / CPU test) |
| NPU-specific capabilities | scattered global functions (`CreateKVCacheBuffer` etc.) | `NPUTensorAllocatorExtensions` optional mixin interface |
| Shape computation | scattered in InitialCreateInputTensors various places | each spec centrally declares ShapeFn |
| Shape + initial values | shape scattered + initial value filling scattered (`CreateInputsTensorBuffer` + `InitHistoryActivationMask` + `FillEnableLoraData` + `InitKvCache` + `ResetMoeConfigs`) | each spec centrally declares ShapeFn + InitFn, eliminates scattered filling |
| Output reuses input | hardcoded std::copy at creation | declared via reuseInputName, parsed by layout |
| KV Cache Merge | `isKvCacheMerge` ternary in 5 places | `MergedKVCacheStrategy` absorbs all branches, TensorManager core 0 ifs |
| External injection KV Cache | `kvCacheSwitchableFlag_` + `SetKVCacheTensors` temporary placeholder + override | `ExternalKVCacheStrategy` + later `Inject` |
| Multiple attentionMask | `attentionMaskIdx` vector + loop expansion | caller loops `RegisterTensor`, symmetric with `RegisterKVCache` expansion pattern |
| Per-layer independent KV shape | `inputsDesc_[kvCacheIdx + i].dims` indexed by i | `KVCacheShapeFn(ctx, layerIdx)` accepts layerIdx |
| Public method count | 45+ | 28 |
| Testability | needs complete model description to test | can register test with any spec list, CPUTensorAllocator does not depend on NPU |

---

## 13. Revision History

| Date | Version | Description |
|------|------|------|
| 2026-08-10 | v1.0 | Initial spec, established three-bucket, ForwardLayout, RegisterTensor design |
| 2026-08-16 | v1.1 | Corrections C1-C8: prefill attentionMask shape, ShapeContext semantics, multiple attentionMask, InitFn, external KV Cache + layerIdx, multi-backend objects, NPU extension interface, KV Cache Merge |
| 2026-08-17 | v1.2 | This merged version: merges original spec + correction documents into single authoritative spec |
