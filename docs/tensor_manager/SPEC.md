# TensorManager 设计规格

## 概述

为从零构建的 LLM 推理引擎设计 TensorManager 模块。该模块负责管理 tensor buffer 的生命周期、shape 计算和检索，支持自回归文本生成的 prefill 和 batch decode 两个阶段。

**范围**：基础文本生成（prefill + batch decode）、静态和动态扩容 KV Cache、KV Cache Merge 模式、外部 KV Cache 注入（specinfer 双模型）、多后端内存分配（Kirin NPU / Ascend NPU / CPU 测试）。

**设计目标**：可扩展性优先——新增 tensor 或 KV Cache 策略时，无需修改 TensorManager 核心代码。

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                      TensorManager                        │
│                                                           │
│  注册层 ────── RegisterTensor / RegisterKVCache           │
│       ↓ FinalizeRegistration → ForwardLayout              │
│                                                           │
│  分配层 ────── AllocateStatic / AllocateForPhase          │
│       ↓              AllocateKVCache                      │
│                                                           │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ STATIC 桶   │  │ DYNAMIC 桶   │  │ STATEFUL 桶       │  │
│  │ (1份共享)   │  │ (per seqLen) │  │ (KV Cache slots) │  │
│  └────────────┘  └─────────────┘  └──────────────────┘  │
│       ↓                ↓                ↓                 │
│  查询层 ────── GetPrefillTensors / GetDecodeTensors       │
│                  GetTensor / GetKVCache                    │
│                                                           │
│  KV Cache操作 ─ Reset / Copy / Resize / UpdateLength      │
│               / ResetTensor (InitFn 重跑)                 │
│                                                           │
│  外部依赖:                                                │
│    TensorAllocator                  (多后端内存抽象)      │
│    NPUTensorAllocatorExtensions     (NPU 专属优化能力)    │
│    KVCacheStrategy                  (静态/动态/Merge/外部) │
└──────────────────────────────────────────────────────────┘
```

**核心原则**：三类桶按生命周期分离存储。声明式注册设计——新增 tensor 只需一行 Register 调用，无需修改 TensorManager 内部。

---

## 2. 核心类型

### 2.1 TensorSpec

将一个 tensor 的全部知识集中声明在一处。

```cpp
enum class Lifecycle {
    STATIC,             // 全局一份，引擎级生命周期（llmConfig, scales, moeEma, ...）
    SEQ_LEN_DEPENDENT,  // 按 seqLen 桶存储（inputIds, logits, attentionMask, ...）
    STATEFUL            // 按 batch slot 管理，对话级生命周期（KV Cache，通常走 RegisterKVCache）
};

enum class Role {
    INPUT,
    OUTPUT,
    INPUT_AND_OUTPUT    // 输出复用输入指针（KV Cache 原地更新）
};

struct ShapeContext {
    int seqLen;          // 当前推理序列长度（prefill: prompt token 数；decode: 1）
    int batchSize;       // 当前 batch 大小（prefill: 1；decode: batchSize）
    int kvCacheLen;      // 当前 KV Cache 有效长度（用于未来扩展：如 incremental attention
                         // 的动态 shape；当前版本所有静态/动态 tensor ShapeFn 不应依赖此字段
                         // 来计算分配 shape，应使用 maxKvCacheLen）
    int maxKvCacheLen;   // KV Cache 最大容量（attentionMask 等需要按容量预分配的 tensor 使用此字段）
};

using ShapeFn = std::function<std::vector<int64_t>(const ShapeContext&)>;

// STATEFUL 桶专用 ShapeFn：多接 layerIdx 参数（C5 修正）
using KVCacheShapeFn = std::function<std::vector<int64_t>(
    const ShapeContext& ctx, int layerIdx)>;

// 分配后立即调用的初始化函数（C4 修正）
using InitFn = std::function<void(void* data, size_t bytes, DataType dtype)>;

struct TensorSpec {
    std::string name;           // 唯一标识，如 "input_ids", "kv_cache_k"
    DataType dtype;             // FLOAT16, INT32, ...
    Lifecycle lifecycle;        // 存储桶类型
    Role role;                  // 输入 / 输出 / 输入输出复用
    ShapeFn shapeFn;            // Decode 阶段 shape 计算回调
    ShapeFn prefillShapeFn;     // 可选：Prefill 专用 shape（默认使用 shapeFn）
    std::string reuseInputName; // INPUT_AND_OUTPUT 时：复用哪个输入 tensor
    InitFn initFn;              // 可选：分配后立即调用，用于填入初值；Reset 时重跑
};
```

**设计决策**：

- **`ShapeFn` 接收 `ShapeContext` 而非单独参数**：后续新增上下文字段（如 `moeExpertNum`）无需改签名
- **`prefillShapeFn` 可选**：不提供时两个阶段共用 `shapeFn`。典型场景：某些模型在 prefill 阶段需要不同的 vocab align（如 `enableLmHeadOpt` 时 logits 在 prefill 是 `[B, 1, vocabAlign]` 而 decode 是 `[B, 1, vocabAlign]`，二者维度数不同），此时 prefill/decode 共用 ShapeFn 会出错，必须分别提供 `prefillShapeFn` 和 `shapeFn`。**attentionMask 不需要 `prefillShapeFn`**——虽然 prefill/decode shape 看似不同（`[1,1,S,maxKV]` vs `[B,1,1,maxKV]`），但用同一个公式 `ctx.batchSize × 1 × ctx.seqLen × ctx.maxKvCacheLen` 即可表达（prefill 时 batchSize=1, seqLen=S；decode 时 seqLen=1）
- **`reuseInputName`**：将复用声明与索引计算解耦，组装时按名称查找源 tensor
- **`InitFn`**：分配后立即调用的初始化函数。在 `AllocateStaticTensors` / `AllocateForPhase` 分配后调；`Reset()` / `ReleaseAll()` 后重新分配时也调；新增 `ResetTensor(name)` 让单个 tensor 回到初始状态时调。KV Cache 不走 InitFn（由 `KVCacheStrategy::Reset` 处理清零）
- **`KVCacheShapeFn`**：STATEFUL 桶专用，接 `layerIdx` 参数，支持每层独立 shape

**ShapeFn 实现约束**：tensor 分配时的 buffer 容量必须按 `maxKvCacheLen` 预留，不能依赖 `kvCacheLen`。`kvCacheLen` 仅在算子运行时通过 `UpdateSlotLength` 维护，不参与分配期 shape 计算。任何 ShapeFn 若其输出 buffer 需要在 decode 过程中被增量写入（如 attentionMask），必须用 `maxKvCacheLen` 作为对应维度的尺寸。

**`AllocateForPhase` 调用时 `ShapeContext` 的填充规则**：

| 字段 | PREFILL 调用 | DECODE 调用 |
|------|-------------|-------------|
| `seqLen` | 传入的 seqLen（如 128） | 1 |
| `batchSize` | 1 | 传入的 batchSize |
| `kvCacheLen` | 0（语义：尚未写入 KV，无效长度） | 各 slot 当前有效长度的最大值（由调用方维护） |
| `maxKvCacheLen` | 引擎配置的 KV Cache 最大容量 | 同上 |

attentionMask 的 ShapeFn 应写为：

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

描述 KV Cache tensor 在 forward vector 中的位置区间。由 `FinalizeRegistration()` 调用 `KVCacheStrategy::GetSpecCount(layerNum)` 推导。

```cpp
struct KVCacheRange {
    uint32_t inputStart;   // KV Cache 在输入 vector 中的起始索引
    uint32_t outputStart;  // KV Cache 在输出 vector 中的起始索引
    uint32_t count;        // KV Cache tensor 总数
                           // 非 Merge: layerNum*2
                           // Merge: 1
                           // 外部注入: 由 ExternalKVCacheStrategy 决定
};
```

### 2.3 KVCacheSlot

每个 batch 的 KV Cache 状态。

```cpp
struct KVCacheSlot {
    std::vector<NDTensorBufferPtr> layerCaches;  // 每层 buffer
                                                  // 非 Merge: [key_0, value_0, key_1, value_1, ...]
                                                  // Merge: [bigBuffer] (1 个元素)
    int currentLen = 0;   // 该 slot 当前有效长度
    int capacity = 0;     // 该 slot 已分配容量
};
```

---

## 3. 三桶存储

| 桶 | 键 | 生命周期 | 创建接口 | 销毁接口 | 典型 tensor |
|----|-----|---------|---------|---------|-----------|
| STATIC | tensor 名称 | 引擎级 | `AllocateStaticTensors()` | `ReleaseAll()` / 引擎销毁 | llmConfig, scales, moeConfig, moeEma, moeLambdaBias, enable_lora |
| DYNAMIC | (seqLen, tensor名称) | 推理轮次 | `AllocateForPhase()` | `ReleasePhase()` | inputIds, attentionMask, logits, positionIds, history_activation_mask |
| STATEFUL | (slotIdx, layerIdx) | 对话级 | `AllocateKVCache()` | `ResetKVCache()` / `ReleaseAll()` | KV Cache key/value |

**为什么不用一个 `map<seqLen, vector>` 统一存储**：

1. **内存**：静态 tensor 在所有 seqLen 档位间共享，而非通过 shared_ptr 引用多次
2. **生命周期独立**：KV Cache 在对话结束时 Reset；静态 tensor 引擎销毁时释放；动态 tensor 推理轮次间可释放。混合存储导致无法独立释放某一类
3. **可扩展性**：新增 KV Cache 策略只影响 STATEFUL 桶代码路径，不涉及静态和动态分配逻辑

---

## 4. ForwardLayout — 组装蓝图

在 `FinalizeRegistration()` 中一次性计算。驱动所有 `Get*Tensors` 调用，无需条件分支。

```cpp
// BucketType 与 Lifecycle 对应：组装时指明从哪个桶取 tensor
enum class BucketType { STATIC, DYNAMIC, STATEFUL };

// KVRole：区分 KV Cache 中的 key 和 value（按层查找时使用）
enum class KVRole { KEY, VALUE };

struct ForwardLayout {
    struct Entry {
        std::string name;        // tensor 语义名
        BucketType  source;      // STATIC / DYNAMIC / STATEFUL
        std::string sourceName;  // 实际取的 tensor 名（复用时与 name 不同）
        int         kvLayerIdx;  // STATEFUL 专用：第几层，-1 表示非 KV
        KVRole      kvRole;      // STATEFUL 专用：KEY / VALUE
    };

    std::vector<Entry> inputEntries;
    std::vector<Entry> outputEntries;
    KVCacheRange kvRange;
};
```

**GetDecodeTensors 中的组装过程**：

```cpp
for (size_t i = 0; i < layout_.inputEntries.size(); i++) {
    auto& entry = layout_.inputEntries[i];
    result.inputs[i] = ResolveTensor(entry, ctx, activeSlots);
}
```

不需要按 tensor 类型写 if-else。Layout 编码了每个 tensor 的来源。通过 `RegisterTensor` 新增 tensor 时 layout 自动更新，组装循环不变。

**`kvRange.count` 推导**：由 `KVCacheStrategy::GetSpecCount(layerNum)` 决定。非 Merge 策略返回 `layerNum * 2`；Merge 策略返回 1；外部注入策略按外部 buffer 数决定。

---

## 5. 完整接口

### 5.1 注册

| 接口 | 用途 |
|------|------|
| `RegisterTensor(spec)` | 注册单个 tensor 规格 |
| `RegisterKVCache(prefix, layerNum, dtype, shapeFn, batchSize, strategy)` | 批量注册 KV Cache（展开数由 `strategy->GetSpecCount()` 决定） |
| `FinalizeRegistration()` | 校验完整性，推导 ForwardLayout，计算 KVCacheRange |

**`RegisterKVCache` 签名（C5 修正）**：

```cpp
hiai::Status RegisterKVCache(
    const std::string& prefix,
    int layerNum,
    DataType dtype,
    const KVCacheShapeFn& shapeFn,   // ← 接 layerIdx，支持每层独立 shape
    int batchSize,
    std::shared_ptr<KVCacheStrategy> strategy);
```

**调用约定**：所有注册在引擎初始化时一次性完成，在分配之前。`FinalizeRegistration` 必须在所有 `Register*` 之后、所有 `Allocate*` 之前调用。

**多 attentionMask / 多 newKvCachePos 场景**：原方案 `InputIndex::attentionMaskIdx` 和 `InputIndex::newKvCachePosIdx` 都是 `std::vector<uint32_t>`，与 `LLMConfigOption::kvCacheLenList` 一一对应，用于支持 sink + rollback 等多滑动窗口配置。本 spec 通过**循环注册多个独立 spec** 表达（命名 `attention_mask_0`、`attention_mask_1` 等），与 `RegisterKVCache` 的展开模式一致。每个独立 spec 的 ShapeFn 通过 lambda 捕获自己的索引 `i` 来选对应的 `kvCacheLenList[i]`，无需扩展 `RegisterTensor` 接口。

### 5.2 分配

| 接口 | 用途 |
|------|------|
| `AllocateStaticTensors()` | 分配所有 STATIC tensor（每种一份），分配后调 `initFn` |
| `AllocateForPhase(phase, seqLen, batchSize)` | 分配指定阶段/seqLen/batch 的 DYNAMIC tensor，分配后调 `initFn` |
| `AllocateKVCache(maxKvCacheLen, batchSize)` | 分配所有 KV Cache slot（委托给策略） |
| `ReleasePhase(phase, seqLen)` | 释放指定阶段/seqLen 的 DYNAMIC tensor |

**Phase 枚举**：

```cpp
enum class Phase { PREFILL, DECODE };
```

### 5.3 查询

| 接口 | 用途 |
|------|------|
| `GetPrefillTensors(seqLen, slotIdx=0)` | 获取 prefill 的完整 IO 集合（batch=1），KV 输出写入指定 slot |
| `GetDecodeTensors(batchSize, activeSlotIndices)` | 获取 batch decode 的完整 IO 集合（seqLen=1） |
| `GetTensor(name, seqLen=0)` | 按名称获取单个 tensor |
| `GetKVCache(slotIdx, layerIdx, role)` | 获取单个 KV Cache 层 buffer（委托给 `strategy->GetLayerView`） |
| `GetKVCacheSlot(slotIdx)` | 获取某个 slot 的全部 buffer（用于序列化） |

**关于 slotIdx 参数**：Prefill 直接将 KV Cache 写入指定 slot 的 buffer（原地更新）。`model.Run` 完成后 KV 数据已在该 slot 中，无需额外拷贝步骤。仅在 slot 间搬运数据时才需要 `CopyKVCacheBetweenSlots`。

**`GetKVCache` 实现**：

```cpp
NDTensorBufferPtr TensorManager::GetKVCache(int slotIdx, int layerIdx, KVRole role) {
    if (slotIdx < 0 || slotIdx >= static_cast<int>(slots_.size())) return nullptr;
    return strategy_->GetLayerView(slots_[slotIdx], layerIdx, role);
}
```

**核心代码无 if 分支**——所有 Merge / 非 Merge 差异由策略吸收。

**返回类型**：

```cpp
struct ForwardTensors {
    std::vector<NDTensorBufferPtr> inputs;
    std::vector<NDTensorBufferPtr> outputs;
};
```

### 5.4 KV Cache 操作

| 接口 | 用途 |
|------|------|
| `ResetKVCache()` | 清空所有 slot，currentLen 置 0（委托给 `strategy->Reset`） |
| `ResetKVCacheSlot(slotIdx)` | 清空指定 slot |
| `CopyKVCacheBetweenSlots(srcSlotIdx, dstSlotIdx)` | slot 间拷贝（前缀复用/多轮对话） |
| `ResizeKVCache(newKvCacheLen)` | 扩容所有 slot 到新容量（委托给策略） |
| `UpdateSlotLength(slotIdx, newLen)` | 更新 slot 有效长度（decode 每步后调用） |
| `GetSlotLength(slotIdx)` | 查询 slot 有效长度 |
| `GetSlotCapacity(slotIdx)` | 查询 slot 已分配容量 |
| `ResetTensor(name)` | 重置单个 STATIC/DYNAMIC tensor 到初始状态（重跑 `initFn`） |

### 5.5 元信息查询

| 接口 | 用途 |
|------|------|
| `HasTensor(name, seqLen=0)` | 检查 tensor 是否已分配 |
| `GetTensorBytes(name, seqLen=0)` | 获取 tensor 字节数 |
| `GetAllocatedPrefillSeqLens()` | 列出已分配的 prefill seqLen 档位 |
| `GetKVCacheRange()` | 获取 KV Cache 在 forward vector 中的索引区间 |
| `GetMemoryUsage()` | 获取按桶分类的内存统计 |

```cpp
struct MemoryUsage {
    size_t staticBytes;     // 静态 tensor 内存
    size_t dynamicBytes;    // 动态 tensor 内存（所有 seqLen 档位合计）
    size_t kvCacheBytes;    // KV Cache 内存（所有 slot 合计）
};
```

### 5.6 生命周期

| 接口 | 用途 |
|------|------|
| `ReleaseAll()` | 释放全部 tensor，保留 spec（可重新分配） |
| `Reset()` | 清空全部状态包括 spec（回到构造后状态） |

**接口总数：28 个**（原 spec 27 个 + C4 新增 `ResetTensor`）。

---

## 6. 多后端内存抽象

### TensorAllocator 接口

```cpp
class TensorAllocator {
public:
    virtual ~TensorAllocator() = default;

    // 分配单个 tensor buffer
    virtual NDTensorBufferPtr Allocate(
        const std::string& name,
        const std::vector<int64_t>& dims,
        DataType dtype) = 0;

    // 批量分配（非纯虚，默认实现走单条循环；NPU 后端可覆盖为批量 API）
    virtual std::vector<NDTensorBufferPtr> AllocateBatch(
        const std::vector<std::string>& names,
        const std::vector<std::vector<int64_t>>& dims,
        const std::vector<DataType>& dtypes);

    // NoCopy 创建：外部提供内存，包装为 tensor buffer
    virtual NDTensorBufferPtr AllocateNoCopy(
        const std::string& name,
        const std::vector<int64_t>& dims,
        DataType dtype,
        void* externalPtr,
        size_t size) = 0;

    // 销毁
    virtual void Deallocate(NDTensorBufferPtr& buffer) = 0;

    // 后端标识
    virtual std::string GetBackendName() const = 0;
};
```

### 后端清单

| 后端 | 类型 | 状态 | 用途 |
|------|------|------|------|
| `KirinNPUTensorAllocator` | 主路径 | 当前实现 | Kirin NPU 生产路径，封装 `HIAI_MR_NDTensorBuffer_*` 系列 API。**单类两种编译形态**：`__OHOS__` 宏决定走 NDK 接口还是 ROM 接口，运行时不切换 |
| `AscendNPUTensorAllocator` | 主路径 | 未来实现 | Ascend NPU，预留接口，封装 `aclrt*` / `aclCreateTensor*` 系列 API |
| `CPUTensorAllocator` | 测试路径 | 可选实现 | 仅用于 `lm_engine_test` 编译目标，单元测试时使用，**不出现在生产构建里** |
| ~~`GPUTensorAllocator`~~ | — | **删除** | YAGNI，无真实需求 |

**关于 Kirin NDK vs ROM**：原方案 `lm_engine_mmgr.h:8-22` 用 `#ifdef __OHOS__` 把同一组 API 名在编译期绑定到 NDK 或 ROM 实现。本 spec 保留这个机制，但封装在 `KirinNPUTensorAllocator` 内部——allocator 对外是同一个 C++ 类，内部根据编译环境选择调用哪个 ROM/NDK 函数。NDK/ROM 差异对 `TensorManager` 和 `KVCacheStrategy` 不可见。

**构造注入示例**：

```cpp
// Kirin NPU（当前）
auto tm = TensorManager(std::make_shared<KirinNPUTensorAllocator>());

// Ascend NPU（未来）
auto tm = TensorManager(std::make_shared<AscendNPUTensorAllocator>());

// 单元测试
auto tm = TensorManager(std::make_shared<CPUTensorAllocator>());
```

**NoCopy 路径**：外部提供内存时（如来自其他进程的共享 KV Cache 内存），`AllocateNoCopy` 将已有指针包装为 tensor buffer，不发生数据拷贝。

**CPU Allocator 的实现约束**：必须自己实现一套 `INDTensorBuffer` 接口（不能复用原方案的 `NDTensorBufferImpl`，因为它强绑 NPU ROM 句柄）：

- `CPUTensorBufferImpl`：内部 `std::vector<uint8_t>` 存储数据，实现 `GetData() / GetSize() / SetCacheStatus() / GetCacheStatus()` 等
- `SetCacheStatus` / `GetCacheStatus` 在 CPU 路径返回 0 / no-op
- `GetFd()` 在 CPU 路径返回 -1（不支持 decoder offload）

CPU allocator 的实现位置：`test/mock/cpu_tensor_allocator.cpp`，**不进生产构建**。

---

## 7. NPU 专属能力扩展

NPU 后端可能提供 `TensorAllocator` 子类同时实现 `NPUTensorAllocatorExtensions`。`KVCacheStrategy` 等内部模块通过 `dynamic_cast` 取扩展接口；扩展接口不可用时走 fallback 或拒绝。CPU allocator 不实现扩展接口。

### NPUTensorAllocatorExtensions 接口

```cpp
class NPUTensorAllocatorExtensions {
public:
    virtual ~NPUTensorAllocatorExtensions() = default;

    // KV Cache 优化分配：跨层一次性分配，按 tile 对齐
    // 对应原方案 CreateKVCacheBufferFromNDK
    virtual NDTensorBufferPtr AllocateKVCache(
        const std::vector<int64_t>& dims,
        DataType dtype,
        int kvCacheNum,
        int tileSize) = 0;

    // KV Cache 扩容：原地扩容已有 buffer
    // 对应原方案 ResizeKVCacheBufferFromNDK
    virtual Status Resize(NDTensorBufferPtr& buffer, size_t newSize) = 0;

    // 设置/获取硬件缓存状态（跨进程共享 KV Cache 用）
    virtual void SetCacheStatus(NDTensorBufferPtr& buffer, uint8_t status) = 0;
    virtual uint8_t GetCacheStatus(const NDTensorBufferPtr& buffer) const = 0;

    // 获取文件描述符（decoder offload 跨进程零拷贝用）
    // 不支持的后端返回 -1
    virtual int32_t GetFd(const NDTensorBufferPtr& buffer) const = 0;
};
```

### 扩展接口与核心接口的关系

```cpp
class KirinNPUTensorAllocator
    : public TensorAllocator,
      public NPUTensorAllocatorExtensions {
    // 同时实现两套接口
public:
    // TensorAllocator
    NDTensorBufferPtr Allocate(...) override;
    NDTensorBufferPtr AllocateNoCopy(...) override;
    void Deallocate(NDTensorBufferPtr&) override;
    std::string GetBackendName() const override { return "KirinNPU"; }

    // NPUTensorAllocatorExtensions
    NDTensorBufferPtr AllocateKVCache(...) override;  // 调 HIAI_MR NDK API
    Status Resize(NDTensorBufferPtr&, size_t) override;
    void SetCacheStatus(...) override;
    uint8_t GetCacheStatus(...) const override;
    int32_t GetFd(...) const override;
};

class AscendNPUTensorAllocator
    : public TensorAllocator,
      public NPUTensorAllocatorExtensions {
    // 未来实现，对应 aclrt* API
};

class CPUTensorAllocator : public TensorAllocator {
    // 只实现核心接口，不实现 Extensions
    // 用于单元测试
};
```

### 兼容性矩阵

| Allocator | 核心接口 | 扩展接口 | 静态 KV 策略 | 动态 KV 策略 | CPU 测试 |
|----------|---------|---------|-------------|-------------|---------|
| `KirinNPUTensorAllocator`（NDK 编译） | ✅ | ✅ | ✅ | ✅ | — |
| `KirinNPUTensorAllocator`（ROM 编译） | ✅ | ⚠️（`AllocateKVCache`/`Resize` 返回 unsupported） | ✅ | ❌ | — |
| `AscendNPUTensorAllocator` | ✅ | ✅（未来） | ✅ | ✅（未来） | — |
| `CPUTensorAllocator` | ✅ | ❌ | ✅（fallback） | ❌（无优化 API） | ✅ |

注意：Kirin ROM 编译路径下 `Extensions::AllocateKVCache` 和 `Resize` 返回 `nullptr` / `ERROR`（与原方案 `from_rom.cpp:144, 156` 一致）。`DynamicKVCacheStrategy` 在 `ext->AllocateKVCache` 返回 nullptr 时应 fallback 到通用 `Allocate`，或直接报错让上层选择静态策略。

### 性能说明

`dynamic_cast` 在 `KVCacheStrategy::Allocate` 中只发生一次（初始化时），不在 decode 热路径。`Resize` 只在扩容档位时调用（罕见），性能可忽略。

---

## 8. KV Cache 策略模式

### KVCacheStrategy 接口

```cpp
class KVCacheStrategy {
public:
    virtual ~KVCacheStrategy() = default;

    // 分配 KV Cache buffer 到 slots
    virtual Status Allocate(TensorAllocator& allocator, int batchSize, int layerNum,
                            int headNum, int headDim, const KVCacheShapeFn& shapeFn,
                            const ShapeContext& ctx, std::vector<KVCacheSlot>& slots) = 0;

    // 扩容
    virtual Status Resize(TensorAllocator& allocator, int newKvCacheLen,
                          const KVCacheShapeFn& shapeFn, const ShapeContext& ctx,
                          std::vector<KVCacheSlot>& slots) = 0;

    // 清零
    virtual Status Reset(std::vector<KVCacheSlot>& slots) = 0;

    // 标识
    virtual std::string GetName() const = 0;

    // 注册时展开的 spec 数（决定 kvRange.count）
    virtual int GetSpecCount(int layerNum) const = 0;

    // 按 (layerIdx, role) 取实际 buffer
    // Merge 策略返回大 buffer 的 view；非 Merge 策略返回 slot.layerCaches[layerIdx*2 + role]
    virtual NDTensorBufferPtr GetLayerView(
        const KVCacheSlot& slot, int layerIdx, KVRole role) const = 0;
};
```

### 策略清单

| 策略 | 用途 | spec 数 | 扩容 | Merge | 外部注入 |
|------|------|--------|------|-------|---------|
| `StaticKVCacheStrategy` | 一次性分配 `maxKvCacheLen` 全量容量 | `layerNum*2` | ❌ | ❌ | ❌ |
| `DynamicKVCacheStrategy` | 接受扩容档位列表，初始最低档位 | `layerNum*2` | ✅ | ❌ | ❌ |
| `MergedKVCacheStrategy`（静态） | 所有层合并为 1 个大 buffer | 1 | ❌ | ✅ | ❌ |
| `MergedKVCacheStrategy`（动态） | Merge + 动态扩容组合 | 1 | ✅ | ✅ | ❌ |
| `ExternalKVCacheStrategy` | 接受外部 buffer，不调 allocator | 由外部 buffer 数决定 | ❌ | 可组合 | ✅ |
| `PagedKVCacheStrategy`（未来） | vLLM 风格分页 | 待定 | 待定 | 可组合 | ❌ |

### StaticKVCacheStrategy

- 一次性分配 `maxKvCacheLen` 全量容量
- `Resize` 返回错误（不支持扩容）
- `Reset` 将所有 buffer 清零，`currentLen` 置 0
- `GetSpecCount` 返回 `layerNum * 2`
- `GetLayerView` 返回 `slot.layerCaches[layerIdx * 2 + (role == KEY ? 0 : 1)]`

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

- 接受扩容档位列表，如 `{256, 512, 1024, 2048}`
- 初始分配使用最低档位
- `Resize` 找到能满足请求长度的下一个档位，分配新 buffer，拷贝已有数据，替换旧 buffer
- `Reset` 同静态策略
- 通过 `dynamic_cast<NPUTensorAllocatorExtensions*>` 取扩展接口；无扩展时拒绝

```cpp
class DynamicKVCacheStrategy : public KVCacheStrategy {
public:
    DynamicKVCacheStrategy(std::vector<int> steps) : steps_(std::move(steps)) {}

    Status Allocate(TensorAllocator& allocator, ...) override {
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        if (ext != nullptr) {
            // NPU 路径：用优化分配
            for (auto& slot : slots) {
                for (int layer = 0; layer < layerNum; layer++) {
                    // layerIdx 传给 KVCacheShapeFn，让每层 shape 可独立
                    auto buf = ext->AllocateKVCache(
                        shapeFn(ctx, layer), dtype, 1, tileSize);
                    slot.layerCaches.push_back(buf);
                }
            }
        } else {
            // CPU / 不支持扩展的后端：fallback 到通用 Allocate
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
        if (ext == nullptr) return ERROR;  // 不支持扩容的后端直接拒绝
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

所有层合并为 **1 个连续大 buffer**，内部布局 `|L0K|L0V|L1K|L1V|...|LnK|LnV|`，对外暴露为单一 tensor。对应原方案 `isKvCacheMerge=true`。

**Merge 与 Dynamic 是两个正交维度**：Merge 控制"1 个大 buffer 还是 layerNum*2 个独立 buffer"（布局维度），Dynamic 控制"是否按档位扩容"（扩容维度）。证据来自 `tensor_manager.cpp:1319-1332`：`enableDynamicKvCache && isKvCacheMerge` 在同一路径生效。`MergedKVCacheStrategy` 内部可选支持扩容（构造时传入档位列表）。

```cpp
class MergedKVCacheStrategy : public KVCacheStrategy {
public:
    // 静态容量（不扩容）
    explicit MergedKVCacheStrategy(uint32_t shareLayerNum = 0)
        : shareLayerNum_(shareLayerNum), dynamicSteps_() {}

    // 动态扩容（按档位）
    MergedKVCacheStrategy(uint32_t shareLayerNum, std::vector<int> steps)
        : shareLayerNum_(shareLayerNum), dynamicSteps_(std::move(steps)) {}

    int GetSpecCount(int layerNum) const override { return 1; }

    Status Allocate(TensorAllocator& allocator, int batchSize, int layerNum,
                    int headNum, int headDim, const KVCacheShapeFn& shapeFn,
                    const ShapeContext& ctx, std::vector<KVCacheSlot>& slots) override {
        auto* ext = dynamic_cast<NPUTensorAllocatorExtensions*>(&allocator);
        int effectiveLayerNum = layerNum - shareLayerNum_;
        int tileSize = 16;  // NZ format；ND format 用 headDim
        int kvCacheNum = effectiveLayerNum * 2;

        int initialCapacity = dynamicSteps_.empty()
            ? ctx.maxKvCacheLen
            : dynamicSteps_.front();

        // Merge 模式下只有 1 个 spec，layerIdx 传 0 表示整体大 buffer 的 shape
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
        // 大 buffer 内部布局：|L0K|L0V|L1K|L1V|...|LnK|LnV|
        size_t totalLayers = /* 由构造时 layerNum - shareLayerNum_ 计算 */;
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

接受外部 buffer（来自其他进程、共享内存、或 specinfer target 模型生成的 KV Cache），不调 allocator。对应原方案 `kvCacheSwitchableFlag_=true`。

**采用"构造空策略 + 后期 `Inject`"模式**（方案 C），对齐原方案时序：`kvCacheSwitchableFlag_=true` 在 Init 时设置，但实际 buffer 在 target 跑完后才注入。

```cpp
class ExternalKVCacheStrategy : public KVCacheStrategy {
    std::vector<NDTensorBufferPtr> externalBuffers_;  // 后期注入
public:
    int GetSpecCount(int layerNum) const override { return externalBuffers_.size(); }
    Status Allocate(...) override { /* 创建空 slot，不分配 buffer */ }

    // 后期注入外部 buffer
    Status Inject(std::vector<NDTensorBufferPtr> buffers) {
        externalBuffers_ = std::move(buffers);
        // 把 buffers 写入所有 slot 的 layerCaches
    }

    NDTensorBufferPtr GetLayerView(...) override {
        /* 按 layerIdx 索引 externalBuffers_ */
    }
    Status Resize(...) override { return ERROR; }  // 不支持扩容
    Status Reset(...) override { /* 不清零外部内存 */ return SUCCESS; }
};
```

调用方：

```cpp
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
tm.RegisterKVCache("kv_cache", layerNum, dtype, shapeFn, batchSize, extStrategy);
// ... target 模型跑完 ...
extStrategy->Inject(targetKvBuffers);  // 后期注入
```

### 兼容性矩阵

| Allocator / 策略组合 | 静态容量 | 动态扩容 | 外部注入 | Merge | 多模型共享 |
|---------------------|---------|---------|---------|-------|----------|
| `StaticKVCacheStrategy` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `DynamicKVCacheStrategy` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `MergedKVCacheStrategy` (静态) | ✅ | ❌ | ❌ | ✅ | ❌ |
| `MergedKVCacheStrategy` (动态) | ✅ | ✅ | ❌ | ✅ | ❌ |
| `ExternalKVCacheStrategy` | — | ❌ | ✅ | 可组合 | ✅ specinfer |

### KV Cache Merge 约束

1. **NZ + 多 kvCacheLen**：不支持（NPU 算子层限制，强约束）
2. **Merge + Dynamic**：二者不互斥，可共存为 `MergedKVCacheStrategy`（动态档位版）
3. **ND + Merge**：原方案 `pipeline_executor_kvcahe.cpp:113` 禁止，但限制在 prefix cache size 计算辅助函数层，不在分配/算子层；spec 重写后该限制可解除
4. **Merge 与多 attentionMask**：Merge 模式下 KV Cache 是 1 个大 buffer，与 `kvCacheLenList.size() > 1` 冲突——**建议 Merge 模式下 `kvCacheLenList.size() == 1`**

这些约束通过文档化声明（不通过代码强制），由调用方保证组合合法。

### 注册时选择策略

```cpp
// 静态策略（非 Merge）
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<StaticKVCacheStrategy>());

// 动态扩容策略（非 Merge）
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<DynamicKVCacheStrategy>({256, 512, 1024, 2048}));

// Merge 静态策略
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<MergedKVCacheStrategy>(shareLayerNum));

// Merge 动态扩容策略
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<MergedKVCacheStrategy>(shareLayerNum, {256, 512, 1024, 2048}));

// 外部注入（specinfer 双模型）
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4, extStrategy);
// target 跑完后：extStrategy->Inject(targetKvBuffers);

// 未来：分页策略
tm.RegisterKVCache("kv_cache", 32, FLOAT16, shapeFn, 4,
    std::make_shared<PagedKVCacheStrategy>(pageSize=16));
```

新增 KV Cache 策略只需实现 `KVCacheStrategy` 接口，TensorManager 核心代码无需改动。

### STATEFUL ShapeFn 典型用法

```cpp
// 每层相同 shape（最常见）
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [](const ShapeContext& ctx, int layerIdx) {
        return std::vector<int64_t>{
            ctx.batchSize, ctx.maxKvCacheLen, headNum, headDim};
    }, batchSize, strategy);

// Sink + Rollback：前 N 层用 sink 长度，其余用 maxKvCacheLen
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [sinkLen, sinkLayerNum](const ShapeContext& ctx, int layerIdx) {
        int kvLen = (layerIdx < sinkLayerNum) ? sinkLen : ctx.maxKvCacheLen;
        return std::vector<int64_t>{
            ctx.batchSize, kvLen, headNum, headDim};
    }, batchSize, strategy);

// 多 kvCacheLenList：按层映射到不同容量
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [kvCacheLenList, layerToLenIdx](const ShapeContext& ctx, int layerIdx) {
        int kvLen = kvCacheLenList[layerToLenIdx[layerIdx]];
        return std::vector<int64_t>{
            ctx.batchSize, kvLen, headNum, headDim};
    }, batchSize, strategy);
```

---

## 9. 初始化与使用流程

### 9.1 引擎初始化

```cpp
// 1. 注入后端分配器构造
auto tm = TensorManager(std::make_shared<KirinNPUTensorAllocator>());

// 2. 注册所有 tensor 规格（含 InitFn）
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

// 多 attentionMask（循环注册）
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

// 注册 KV Cache（选策略）
tm.RegisterKVCache("kv_cache", layerNum, dtype,
    [kvCacheLenList](const ShapeContext& ctx, int layerIdx) {
        return std::vector<int64_t>{
            ctx.batchSize, ctx.maxKvCacheLen, headNum, headDim};
    }, batchSize, std::make_shared<StaticKVCacheStrategy>());

// 3. 完成注册（校验、推导 layout）
tm.FinalizeRegistration();

// 4. 分配
tm.AllocateStaticTensors();  // 分配后自动调每个 spec 的 initFn
tm.AllocateKVCache(maxKvCacheLen, batchSize);
tm.AllocateForPhase(PREFILL, seqLen, batchSize=1);
tm.AllocateForPhase(DECODE, seqLen=1, batchSize);
```

### 9.2 Prefill

```cpp
// Prefill 将 KV 数据直接写入 slot 0（通过 STATEFUL 桶原地更新）
auto tensors = tm.GetPrefillTensors(seqLen, /*slotIdx=*/0);
FillInputIds(tensors, promptTokens);       // 通过 GetTensor("input_ids", seqLen)->GetData() 取 data
FillAttentionMask(tensors, seqLen, maxKvCacheLen);
// 实际填值时按 [1,1,S,maxKV] 布局填，超出 S 的位置写 -FLT_MAX
model.Run(tensors.inputs, tensors.outputs);
// KV 数据已在 slot 0 的 layerCaches 中，无需额外拷贝
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

### 9.4 多轮对话

```cpp
// 第2轮：复用同一 slot 的 KV Cache，无需 reset
auto tensors = tm.GetDecodeTensors(1, {slotIdx});
// ...（同 batch decode）

// 对话结束：重置 slot
tm.ResetKVCacheSlot(slotIdx);
```

### 9.5 动态 KV Cache 扩容

```cpp
// Decode 过程中，有效长度接近容量上限时触发扩容
if (tm.GetSlotLength(slotIdx) >= tm.GetSlotCapacity(slotIdx) - 1) {
    tm.ResizeKVCache(tm.GetSlotCapacity(slotIdx) + 1);  // 策略自动寻找下一个档位
}
// 示例：当前容量=256，策略档位={256,512,1024,2048}
// ResizeKVCache(257) → 策略找到 nextStep=512，扩容到 512
```

### 9.6 specinfer 双模型 KV Cache 共享

```cpp
// target 模型与 draft 模型共享 KV Cache
// 1. target 用正常策略分配
auto targetTm = TensorManager(/* KirinNPU allocator */);
targetTm.RegisterKVCache(..., std::make_shared<StaticKVCacheStrategy>());
targetTm.FinalizeRegistration();
targetTm.AllocateKVCache(maxKvCacheLen, batchSize);

// 2. draft 用 ExternalKVCacheStrategy，占位
auto draftTm = TensorManager(/* KirinNPU allocator */);
auto extStrategy = std::make_shared<ExternalKVCacheStrategy>();
draftTm.RegisterKVCache(..., extStrategy);
draftTm.FinalizeRegistration();
draftTm.AllocateKVCache(maxKvCacheLen, batchSize);  // 创建空 slot

// 3. target 跑完 prefill 后
auto targetKvBuffers = targetTm.GetKVCacheSlot(/*slotIdx=*/0);

// 4. 注入到 draft
extStrategy->Inject(targetKvBuffers);

// 5. draft 直接复用 target 的 KV，无需重复算
auto draftTensors = draftTm.GetDecodeTensors(batchSize, {0});
model.Run(draftTensors.inputs, draftTensors.outputs);
```

### 9.7 Reset 单个 tensor（对应原方案 ResetMoeConfigs）

```cpp
// 原方案 ResetMoeConfigs 的等价
tm.ResetTensor("moe_map_info_0");
tm.ResetTensor("moe_cache_info_0");
tm.ResetTensor("moe_ema");
tm.ResetTensor("moe_lambda_bias");

// 或可选批量接口
tm.ResetTensorsByPrefix("moe_");
```

---

## 10. 错误处理

- 所有返回 `Status` 的方法使用 `hiai::Status`（SUCCESS / ERROR 码）
- `FinalizeRegistration` 校验：名称无重复、角色覆盖完整、`INPUT_AND_OUTPUT` 的 `reuseInputName` 有效、KV Cache 在 finalize 前已注册
- `GetTensor` / `GetKVCache` 未找到或未分配时返回 `nullptr`（不抛异常）
- `AllocateForPhase` 对已分配的 tensor 跳过（幂等）
- `ResizeKVCache` 在策略不支持扩容时（静态策略）返回错误
- `KVCacheStrategy::Allocate` 不检查组合合法性（如 Merge + 多 kvCacheLen），调用方传错时由算子层报错

---

## 11. 线程安全

TensorManager **设计上非线程安全**。

理由：推理引擎通常使用单调度线程编排 prefill 和 decode。无锁设计避免热路径开销。如需并发访问（如异步 KV Cache 加载），由调用方外部同步。

---

## 12. 与原始实现的对比

| 维度 | 原始实现 | 重新设计 |
|------|---------|---------|
| 新增 tensor | 改 5+ 处代码 | 加 1 行 Register 调用 |
| 索引管理 | 手动 `currentIdx++` | 注册顺序自动推导 |
| tensor 存储 | `map<seqLen, vector>`（每个 seqLen 存全量） | 三桶按生命周期分离 |
| KV Cache 模式 | 创建方法中的 if-else 链 | 策略模式（Static / Dynamic / Merged / External / Paged） |
| 内存后端 | 编译期 NDK/ROM 宏切换（仅 Kirin 内部两种部署） | 运行时注入 TensorAllocator（Kirin NPU / Ascend NPU / CPU 测试） |
| NPU 专属能力 | 散落全局函数（`CreateKVCacheBuffer` 等） | `NPUTensorAllocatorExtensions` 可选 mixin 接口 |
| shape 计算 | 散布在 InitialCreateInputTensors 各处 | 每个 spec 集中声明 ShapeFn |
| shape + 初始值 | shape 散布 + 填初值散布（`CreateInputsTensorBuffer` + `InitHistoryActivationMask` + `FillEnableLoraData` + `InitKvCache` + `ResetMoeConfigs`） | 每个 spec 集中声明 ShapeFn + InitFn，消除分散填值 |
| 输出复用输入 | 创建时硬编码 std::copy | 通过 reuseInputName 声明，由 layout 解析 |
| KV Cache Merge | `isKvCacheMerge` 在 5 处三元运算 | `MergedKVCacheStrategy` 吸收所有分支，TensorManager 核心 0 if |
| 外部注入 KV Cache | `kvCacheSwitchableFlag_` + `SetKVCacheTensors` 临时占位 + 覆盖 | `ExternalKVCacheStrategy` + 后期 `Inject` |
| 多 attentionMask | `attentionMaskIdx` vector + 循环展开 | 调用方循环 `RegisterTensor`，与 `RegisterKVCache` 展开模式对称 |
| 每层独立 KV shape | `inputsDesc_[kvCacheIdx + i].dims` 按 i 索引 | `KVCacheShapeFn(ctx, layerIdx)` 接 layerIdx |
| 公开方法数 | 45+ | 28 |
| 可测试性 | 需完整模型描述才能测试 | 可用任意 spec 列表注册测试，CPUTensorAllocator 不依赖 NPU |

---

## 13. 修订历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-08-10 | v1.0 | 初版 spec，确立三桶、ForwardLayout、RegisterTensor 设计 |
| 2026-08-16 | v1.1 | 修正 C1-C8：prefill attentionMask shape、ShapeContext 语义、多 attentionMask、InitFn、外部 KV Cache + layerIdx、多后端对象、NPU 扩展接口、KV Cache Merge |
| 2026-08-17 | v1.2 | 本合并版：将原 spec + 修正文档合并为单一权威 spec |
