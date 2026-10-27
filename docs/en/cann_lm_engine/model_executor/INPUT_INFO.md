# ModelExecutor Model Input Information

English | [简体中文](../../../zh/cann_lm_engine/model_executor/INPUT_INFO.md)

> Source: backend `IBuiltModel::GetInputDescs()` + om graph input definitions + code implementation + user confirmation.
> This document is the **source of truth** for ModelExecutor input filling design (SPEC/DESIGN references this document).

## 1. Model input tensor composition

The model om graph has multiple input tensors, returned by `IBuiltModel::GetInputDescs()` as `vector<TensorDesc>`. Each `TensorDesc` contains `name`, `dataType`, `format`, `dims`, `size`.

ModelExecutor applies for `input_tensors_` (`vector<shared_ptr<ITensorBuffer>>`) based on descs via `TensorManager::Instance().CreateTensor(desc, modelId, &out)` at Load time.

**Slot ordering is undetermined** — each omc may differ, the index needs to be confirmed from `TensorDesc.name`.

### 1.1 Input tensor slots (6, all in input_tensors_)

| # | Name | Type | Dimensions | Filler | Fill timing | Description |
|---|------|------|------|--------|---------|------|
| 1 | input_embed | **Per TensorDesc.dataType** (varies by path, see §2.1) | `[batch, seqLen, hidden]` (TokenId path is `[batch, seqLen]`) | TextEmbedding::FillInputs (SR20260828001 wiring, Scheduler two input preparation points) | Before Forward | embedding input (tokenId path goes through model internal embedding) |
| 2 | embed_scales | FLOAT32 | PerToken: `batch × seqLen × 1`; PerGroup: `batch × seqLen × groups` | embedding module | At input preparation | embedding dequantization parameters (implemented on 2026-08-29, SR20260828001) |
| 3 | attention_mask | FLOAT32 | `batch * 1 * seqLen * kvCacheLen` | InputAssembler | During Forward | Attention matrix: visible=0, invisible=-FLT_MAX |
| 4 | position_ids | INT32 | `batch * seqLen` | InputAssembler | During Forward | Position information corresponding to current tokenId (position encoding) |
| 5 | new_kv_cache_pos | INT32 | `[batch, seqLen]` | InputAssembler | During Forward | Guides the model to generate the position of the next kvCache (per-slot), may wrap around |
| 6 | kvCache | — | `[numLayers, max_batch, max_seqLen, 2(K&V), num_heads, head_dim]` | Shared pool | — | KV Cache pool buffer (applied via TensorManager, as an input_tensors_ element) |

### 1.2 Output tensor

| # | Name | Type | Dimensions | Filler |
|---|------|------|------|--------|
| 1 | logits | FLOAT32 | `[batch, vocabSize]` | Filled by ModelManager::Run |

## 2. Filling rules for each slot

### 2.1 input_embed

- **Filler**: `TextEmbedding::FillInputs` (SR20260828001 §2.2 wiring point — called by Scheduler's prefill/decode two input preparation points; locates slot via C-02 read-only accessor `InputTensors()`/`InputNameIndex()`)
- **Written type**: based on the slot's `TensorDesc.dataType` (D-06), not the file storage type:
  - TokenId path: INT32 / INT64 (token id value written)
  - Embedding path: FLOAT16 / FLOAT32 / INT8 (low-precision row value expansion conversion, no scale multiplication — dequantization is on NPU)
- **Data source**:
  - embedding path: TextEmbedding host gather low-precision row (InMemory memory copy / InFlash row-by-row pread)
  - tokenId path: tokenId sequence, goes through model internal embedding (om graph handles lookup)
- **SetInputData current state**: retained as "external direct-fill channel" API (memcpy semantics unchanged), after SR20260828001 wiring, zero in-repo calls
- **Note**: real backend needs to assemble by batch dimension; current stub only writes the first slot (M5 leftover)

### 2.2 embed_scales

- **Filler**: embedding module (`TextEmbedding::FillInputs` Embedding path, when scale_type ≠ None)
- **Description**: embedding dequantization parameters, filled at input preparation; scale is always in-memory state (loaded during module Load)
- **Dimensions**: PerToken `batch × seqLen × 1`; PerGroup `batch × seqLen × groups` (groups = hidden/group_size)
- **Type**: FLOAT32 (when scale is stored as Fp16, the module does value expansion conversion)
- **Status**: **Implemented** (2026-08-29, SR20260828001; previous "shelved" record is now outdated)

### 2.3 attention_mask

- **Filler**: InputAssembler
- **Dimensions**: `batch * 1 * seqLen * kvCacheLen` (4D, 2nd dimension fixed to 1)
  - prefill: `batch=1, seqLen=chunk_len, kvCacheLen=cursor_before + chunk_len`
  - decode: `batch=active_count, seqLen=1, kvCacheLen=max(cursor_i + 1)`
- **Filling rule** (causal):
  - `mask[b][0][i][j] = 0.0` if `j <= cursor_b_i` (current position and before, visible)
  - `mask[b][0][i][j] = -FLT_MAX` if `j > cursor_b_i` (after, masked)
- **padding masking**: slots with different cursors within decode batch use `max_kv_len = max(cursor_i + 1)` to take max; short slot tails are set to -FLT_MAX
- **Data type**: FLOAT32

### 2.4 position_ids

- **Filler**: InputAssembler
- **Dimensions**: `batch * seqLen`
- **Derivation rule** (position obtained from KV pool):
  - prefill: `pos[i] = cursor_start + i` (i=0..chunk_len-1)
  - decode: `pos[b][0] = cursor_b` (current cursor value of each active slot)
- **Data type**: INT32

### 2.5 new_kv_cache_pos

- **Filler**: InputAssembler (position information obtained from KV pool)
- **Dimensions**: `[batch, seqLen]` (same dimensions as position_ids, but different semantics)
- **Description**: guides the model to generate the position of the next kvCache — the position where the current token is written to the KV pool
- **Difference from position_ids** (cannot be merged, filled into different slots respectively):
  - `position_ids`: `batch × seqLen`, used for position encoding
  - `new_kv_cache_pos`: `batch × seqLen`, used for KV pool write position guidance, **may wrap around**
  - Both currently take the same value in implementation (both derived from each slot cursor), but the semantics are independent, the slots are different, cannot be merged
- **Wrap-around logic**: managed by KV pool — KV pool manages cursor position, obtains the next KV position (including wrap-around) through the KV pool, provided to InputAssembler
- **Fill timing**: cursor is read **before** `Advance` (first obtain position from KV pool, fill into tensor, then advance cursor)

### 2.6 kvCache

- **Filler**: shared KV pool (`KvCachePool`)
- **Application method**: applied via `TensorManager::Instance().CreateTensor(desc, modelId, &out)` — **as an input_tensors_ array element**, submitted to model inference along with other inputs
- **Binding method**: at Load time, applies for ITensorBuffer based on the desc named "kvCache" in `IBuiltModel::GetInputDescs()`; the buffer's underlying layer points to the shared memory of the KV pool (layer-major layout)
- **Description**: prefill/decode two instances share the same pool with zero-copy; pool memory is allocated by TensorManager

## 3. Filling flow (inside Forward)

```
Forward(opts):
  1. FillInputs(...)                  → fill input_embed (+ embed_scales for Embedding path);
                                        Scheduler input preparation point calls TextEmbedding::FillInputs
                                        (stub era name_index is empty → NOT_FOUND ignored, same as old SetInputData behavior)
  2. AssemblePrefill/Decode           → obtain cursor position from KV pool → fill position_ids + new_kv_cache_pos + attention_mask → advance cursor
  3. mm->Run(input_tensors_, output_tensors_)  → MM reads all inputs (including kvCache), fills output[0] (logits)
  4. Logits()                         → returns output_tensors_[0]
```

### 3.1 prefill path (`chunk_len > 1`)

```
cursor_before = pool->NextPosition(slot)           // obtained from KV pool
position_ids = [cursor_before, cursor_before+1, ..., cursor_before+chunk_len-1]
new_kv_cache_pos = position_ids ([1, chunk_len], same value; batch=1)
kvCacheLen = cursor_before + chunk_len
mask[0][0][i][j] = 0.0 if j <= cursor_before+i else -FLT_MAX
pool->Advance(slot, chunk_len)                     // advance cursor (KV pool manages wrap-around)
```

### 3.2 decode path (`chunk_len == 1`)

```
cursors[i] = pool->NextPosition(slots[i])          // obtained from KV pool (including wrap-around)
max_kv_len = max(cursors[i] + 1)
position_ids[i][0] = cursors[i]
new_kv_cache_pos[i][0] = cursors[i]                // per-slot [batch, 1]
mask[i][0][0][j] = 0.0 if j <= cursors[i] else -FLT_MAX
pool->Advance(slots[i], 1)                        // advance cursor for each active slot
```

## 4. name → index mapping

At Load time, the `TensorDesc.name → input_tensors_ index` mapping table is built. InputAssembler looks up the corresponding tensor by name to fill (no longer hardcodes `[1]`/`[2]`).

```cpp
// Built after ModelExecutor Load
std::unordered_map<std::string, size_t> input_name_index_;
for (size_t i = 0; i < in_descs.size(); ++i) {
    input_name_index_[in_descs[i].name] = i;
}
```

InputAssembler receives the name→index mapping, and looks up the `position_ids` / `new_kv_cache_pos` / `attention_mask` slots by name.

## 5. TensorDesc → ITensorBuffer mapping

| TensorDesc field | ITensorBuffer interface | Description |
|----------------|-------------------|------|
| `name` | `GetDesc().name` | om graph input tensor name (e.g. "input_embed") |
| `dataType` | `GetDesc().dataType` | FLOAT16 / FLOAT32 / INT32 (backend `cann_llm_engine::DataType`) |
| `format` | `GetDesc().format` | ND (all inputs of this model are ND format) |
| `dims` | `GetDesc().dims` | int64_t dimension array |
| `size` | `GetSize()` | byte size (0=dynamic shape, TensorBufferImpl defaults to allocate 1024) |

## 6. Known gaps (related leftovers)

| Leftover # | Related input | Issue |
|---------|---------|------|
| #4 | All | When `TensorDesc.size==0`, allocate 1024 bytes (dynamic shape) |
| #5 | All | `malloc` not initialized → unfilled regions are heap garbage |
| #6 | input_embed | `SetInputData` does not check src null (currently zero in-repo calls, only external direct-fill channel retained) |
| M4 | input_embed | prefill/decode distinguished by chunk_len heuristic |
| M5 | input_embed | decode batch>1 only writes the first slot |
| — | kvCache | Need to confirm how KV pool buffer is associated with ITensorBuffer (pool memory pointer) |

## 7. Version

| Version | Date | Change |
|------|------|------|
| 0.1 | 2026-08-26 | Initial version: 6 input slots + filling rules + flow + name→index mapping |
| 0.2 | 2026-08-26 | User confirmation: kvCache is also an input_tensors_ element (applied via TensorManager); position_ids and new_kv_cache_pos cannot be merged; attention_mask is 4D (2nd dimension=1); wrap-around managed by KV pool; embed_scales filled by embedding module (shelved) |
| 0.3 | 2026-08-26 | review v0.13-M1 confirmation: `new_kv_cache_pos` dimension changed from 1D `[seqLen]` to `[batch, seqLen]` (per-slot). Each decode slot has its own cursor (not shared cursors[0]). §1.1/§2.5/§3.1/§3.2 synchronized. |
| 0.4 | 2026-08-29 | U-4 correction (with TextEmbedding implementation/SR20260828001 landing): input_embed type changed to "per TensorDesc.dataType varies by path" (old record FLOAT16 was stub placeholder); embed_scales dimension PerGroup added `×groups` (old record `×1` only holds for PerToken), status changed from "shelved" to "implemented"; input_embed filler changed from SetInputData to TextEmbedding::FillInputs (Scheduler two input preparation points wiring); §3 flow step 1 and §6 leftover table synchronized. |
