# TensorManager Design Document

English | [简体中文](../../../zh/cann_lm_engine/tensor_manager/DESIGN.md)

| Field | Value |
|------|-----|
| Spec ID | — |
| Level | L3 core component (`cann_local_lm/engine/core/tensor_manager.h/.cpp`) |
| Status | **Landed** (v0.12 backend proxy; v0.14 adds KV batch channel `CreateKvCacheTensor`, C-04 landed) |
| Version | 0.2 |

## §1 Scope and Goals

TensorManager is the engine-side **unified tensor application entry** (singleton proxy), providing application channels for input/output tensors and KV cache tensors; divides work with KvCachePool (M-02):

- **TensorManager**: application channel — `CreateTensor` (input/output tensors refilled per step, modelId associates loaded model) and `CreateKvCacheTensor` (KV pool dedicated batch channel, no modelId — pool created before model Load, not bound to specific model); internally delegates to backend `CreateTensorBuffer(TensorDesc, DeviceType)` factory, engine Status mapped via `map_backend_status`
- **KvCachePool**: **semantics and lifecycle** of KV cache tensors accumulated across steps (slots/cursor/eviction/reuse, holds `shared_ptr<ITensorBuffer>` RAII), at construction time **single call** to `CreateKvCacheTensor` to apply for all `2×num_hidden_layers` tensors

**Design decision (D-13 supplement → v0.14 evolution)**: KV cache's unique operations like eviction/reuse are self-contained in KvCachePool; its underlying memory application/release goes through TensorManager unified entry (C-04 landing — replaces v0.1 draft's `AllocKvBuffer`→`AttachBuffer` injection-style division of work, v0.1 draft not implemented).

## §2 Current Status

**Landed** (`core/tensor_manager.h/.cpp`):

- Singleton `Instance()`; `backend_type_ = DetectAvailableDevice()` runtime detection (NDK preferred)
- `CreateTensor(TensorDesc, modelId, out*)`: after `ModelExecutor::Load`, applies for `input_tensors_`/`output_tensors_` by built descs (consumed by model_executor)
- `CreateKvCacheTensor(num_hidden_layers, kv_cache_max_len, num_attention_kv_heads, batch, num_attention_head_dims, DataType, out*)`: **creates all** `2×num_hidden_layers` per-layer K/V tensors in single call (consumed by KvCachePool constructor); unified validation at interface entry (any parameter ≤ 0 / dtype UNDEFINED / out empty → INVALID_ARGUMENT) + uint64 product overflow protection + all-or-nothing (any failure → FAILURE, out produces nothing)
- `DataTypeSize(DataType)`: dtype → byte count inline helper
- `Release(shared_ptr)`: RAII fallback (explicit release optional)

v0.1 draft's `Init/GetStepTensors/AllocKvBuffer/FreeKvBuffer` not implemented, replaced by the above actual form (InputAssembler's per-step temporary tensors are now `input_tensors_` slots applied by `ModelExecutor::Load`, looked up by name→index to fill, not via pooling).

## §3 External Interface (Actual)

```cpp
class TensorManager {
public:
    static TensorManager& Instance();

    /// Apply tensor by desc + modelId (input/output; internally delegates to CreateTensorBuffer).
    Status CreateTensor(const TensorDesc& desc, int modelId,
                        std::shared_ptr<ITensorBuffer>* out);

    /// KV cache pool dedicated batch channel: one call creates all 2×num_hidden_layers tensors.
    /// Dimensions [kv_cache_max_len, num_attention_kv_heads, batch, num_attention_head_dims],
    /// named kv_cache_k_<layer> / kv_cache_v_<layer>, output order [k0,v0,k1,v1,...].
    Status CreateKvCacheTensor(int num_hidden_layers, int kv_cache_max_len,
                               int num_attention_kv_heads, int batch,
                               int num_attention_head_dims, DataType dtype,
                               std::vector<std::shared_ptr<ITensorBuffer>>* out);

    /// Explicit release (optional; RAII also auto-releases).
    Status Release(const std::shared_ptr<ITensorBuffer>& t);
};
```

## §4 Relationship with Other Modules

| Module | Relationship |
|------|------|
| KvCachePool (M-02) | At construction **single call** to `CreateKvCacheTensor` to apply for all KV tensors; slots/cursor/eviction belong to KvCachePool |
| ModelExecutor (M-01) | After Load, applies for `input_tensors_`/`output_tensors_` by built descs + modelId via `CreateTensor` |
| InputAssembler (M-03) | Fills `input_tensors_` slots exposed by ModelExecutor (not via TensorManager) |
| backend | Delegates to `cann_llm_engine::CreateTensorBuffer(TensorDesc, DeviceType)` factory |

## §5 Memory Layout

- KV Cache (v0.14): **per-layer independent K/V tensors** (total `2×num_hidden_layers`), single tensor dimensions `[kv_cache_max_len][num_attention_kv_heads][batch][num_attention_head_dims]`, order `[k0,v0,k1,v1,...]`, named `kv_cache_k_<layer>`/`kv_cache_v_<layer>` (satisfies om binding entire layer K/V per layer); dtype parsed from meta_json `kv_cache_type` (lowercase vocabulary fp16/fp32…, default/unknown falls back to FLOAT16); capacity/shape parameters authoritative from meta_json model_config section
- Input/output tensors: per built descs (backend model definition)

## §6 Landing Plan (v0.1 draft → current state)

1. ~~Define `ITensorAllocator` interface~~ → not adopted; from v0.12 directly via backend `CreateTensorBuffer` factory
2. ~~TensorManager pooled allocation/release~~ → landed as singleton proxy (`CreateTensor`/`CreateKvCacheTensor`), no pooling
3. ~~KvCachePool switches from AttachBuffer to TensorManager injection~~ → **completed (v0.14)**:
   `CreateKvCacheTensor` single batch application; old `AttachBuffer`/`PoolHandle`/`SlotOffset` deleted
4. ~~InputAssembler switches from raw `shared_ptr<Tensor>` to TensorManager acquisition~~ → form evolution:
   After Load apply for slots by descs, assembler fills slots by name (name→index)

## §7 Change Log

| Version | Date | Change |
|------|------|------|
| 0.1 | 2026-08-19 | Initial version: expected/spec state document (`AllocKvBuffer`→`AttachBuffer` injection-style sketch; code not provided). |
| 0.2 | 2026-09-08 | Aligned with actual state: TM is unified tensor application entry (`CreateTensor` + KV batch channel `CreateKvCacheTensor`); C-04 landed — KvCachePool constructor single apply for all `2×num_hidden_layers` per-layer K/V tensors (driven by meta_json kv shape parameters, dims `[kv_cache_max_len][kv_heads][batch][head_dims]`); old `AttachBuffer`/`PoolHandle`/`SlotOffset` deleted. §1/§2/§3/§4/§5/§6 rewritten. |
