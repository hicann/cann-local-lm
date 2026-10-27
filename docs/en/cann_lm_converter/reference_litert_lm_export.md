# LiteRT-LM Export Logic Analysis (Reference Knowledge Base)

English | [简体中文](../../zh/cann_lm_converter/reference_litert_lm_export.md)

> Analysis object: `google-ai-edge/LiteRT-LM` (local source `/mnt/workspace/LiteRT-LM-main/`) + `litert-torch` (local source `/mnt/workspace/litert-torch-main/`)
> Purpose: External reference for our converter export approach.
> Last updated: 2026-07-14 (third round: detailed analysis of litert-torch source)

## 1. What is LiteRT-LM

Google's **production-grade LLM orchestration layer**, running on LiteRT (TFLite successor), targeting Android/iOS/Web/desktop/IoT. **LiteRT-LM itself is not a model exporter**, but "packaging of exported artifacts + runtime orchestration". README's words: "production-ready orchestration layer to run LLMs with LiteRT".

Supports Gemma / Llama / Phi-4 / Qwen etc.; output is a single `.litertlm` file.

## 2. Layered Architecture (confirmed from local source)

```
┌──────────────────────────────────────────────────────────────┐
│ Model export layer (not in this repo — full-repo grep "torch.export/ai_edge_torch/Signature" 0 hits) │
│   HF PyTorch model → ai_edge_torch (torch.export + Signature) → .tflite │
│   Output: one .tflite containing two signature runners: "prefill" + "decode"      │
└──────────────────────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Packaging layer (python/litert_lm_builder/)                            │
│   litert-lm-builder: .tflite + weights + tokenizer + metadata│
│                      → .litertlm container                         │
└──────────────────────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Runtime (runtime/ C++ + python/litert_lm/ ctypes FFI)         │
│   C++ executor: calls prefill/decode two signature runners, manages KV cache │
└──────────────────────────────────────────────────────────────┘
```

**The model export layer does not exist in this repo** (`grep -rIln "ai_edge_torch\|torch.export\|Signature" .` 0 hits). LiteRT-LM only consumes ready-made `.tflite`. Export relies on external `ai_edge_torch` (`torch.export` + Signature API), making KV cache explicit in/out. *(ai_edge_torch source is not in this repo, but its contract can be reverse-inferred from LiteRT-LM runtime's tensor name consumption — see §5.)*

## 3. `.litertlm` bundle structure (confirmed)

`LitertLmFileBuilder` API (`python/litert_lm_builder/litertlm_builder.py`):

| API | Purpose | Comparison with our epair output |
|------|------|------|
| `add_system_metadata(Metadata)` | Key-value metadata (Authors / TargetBackend) | (can go in model_config) |
| `add_llm_metadata(...)` | LLM metadata (config etc.) | `model_config.json` |
| `add_tflite_model(path, model_type=PREFILL_DECODE)` | **Main graph** | `ep_json_with_ascend_ir.json` |
| `add_tflite_weights(...)` | **Independent weights** (graph-weight separation) | `weights_*.bin` + `weight_info.json` |
| `add_sentencepiece_tokenizer` / `add_hf_tokenizer` | tokenizer | tokenizer files |
| `add_generic_binary_data(...)` | Any binary section | — |
| `build(stream)` | Serialize into single `.litertlm` file | (we use a directory; could add packaging step) |

Graph-weight separation is **isomorphic** to our `ep_json` + `weights_*.bin`. The only difference is it packs into a single file.

## 4. `TfLiteModelType` (confirmed, `litertlm_builder.py:208`, external dependency litert framework)

```python
class TfLiteModelType(enum.Enum):
    PREFILL_DECODE = "tf_lite_prefill_decode"      # One .tflite contains prefill + decode two signatures
    ARTISAN_TEXT_DECODER = "tf_lite_artisan_text_decoder"
```

> **Correction of first-round misstatement**: `PREFILL_DECODE` **is not "one graph unifying prefill+decode"**, but **two signature runners in one model file** (`"prefill"` + `"decode"`, see §5). prefill signature has dynamic seq, decode signature has fixed seq=1; KV cache is managed by runtime between the two signatures.

## 5. Model Signature Contract (confirmed from local source, runtime/executor/)

### 5.1 Two signature runners

`runtime/executor/llm_litert_compiled_model_executor.cc:80-81` (external dependency litert framework):

```cpp
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner  = "decode";
```

Runtime calls `compiled_model.FindSignature(signature)` to get the corresponding signature runner, calling prefill or decode by stage. **The two signatures share the same set of input tensor names** (`ModelSignatures` struct, common to prefill/decode).

### 5.2 `ModelSignatures` struct (`litert_compiled_model_executor_utils.h:55`, external dependency litert framework)

```cpp
struct ModelSignatures {
  std::string input_tokens;                       // Required
  std::string input_positions;                     // Required
  std::optional<std::string> input_attn_mask;      // Optional
  std::optional<std::string> input_embeddings;     // Optional (if set, external embedding lookup, no token_ids)
  std::optional<std::string> input_per_layer_embeddings;  // Optional
  std::optional<std::string> input_int32_param;    // Optional
  std::string output_logits;                      // Required (decode use)
};
```

### 5.3 Default tensor names (`litert_compiled_model_executor_utils.cc:71-89`, external dependency litert framework)

Runtime auto-matches by "candidate name list" (supports multiple naming variants):

| Semantic | Default name | Candidates |
|------|--------|------|
| Input tokens | `token_ids` | `token_ids` / `tokens` |
| Input positions | `positions` | `positions` / `input_pos` |
| attention mask | `attn_mask` | `attn_mask` / `mask` |
| Input embedding | `embeddings` | `embeddings` |
| per-layer embedding | `per_layer_embeddings` | `per_layer_embeddings` |
| int32 parameter | `param_tensor` | `param_tensor` |
| Output logits | `logits` | `logits` |

### 5.4 KV cache tensor naming convention (`GetKVCacheRootNames`, `utils.cc:189`, external dependency litert framework)

KV cache is **per-layer pair (k, v) explicit input and explicit output**, named by "root name + layer number". Runtime scans input/output names to auto-detect one of three conventions:

| Convention | Input example | k root name | v root name |
|------|---------|--------|--------|
| Standard (separate k/v) | `kv_cache_k_0` / `kv_cache_k_1` … | `kv_cache_k_` | `kv_cache_v_` |
| Abbreviated | `k_cache_0` / `v_cache_0` … | `k_cache_` | `v_cache_` |
| Merged (k+v same tensor) | `kv_cache_c_0` … | `kv_cache_c_` | `kv_cache_c_` |

If none of the conventions are found, `FailedPreconditionError("No KV cache inputs found.")`. That is, model export **must** expose KV cache in/out by one of these three naming conventions.

## 6. prefill / decode runtime mechanism (confirmed, `llm_litert_compiled_model_executor.cc`)

### 6.1 Dynamic shape (prefill)

prefill supports **dynamic seq_len**: `kDynamicDimValue = -1` (line 82); at runtime `ResizeInputTensor(signature, tensor_name, new_shape)` resizes by current prompt length (line 254, 440). decode has fixed seq=1.

### 6.2 KV cache double buffer + ping-pong

```cpp
input_kv_cache_buffers_  = &kv_cache_buffers_1;   // line 550-552
output_kv_cache_buffers_ = &kv_cache_buffers_2;
...
std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // line 761
```

Each step uses the previous step's output KV buffer as the current step's input (ping-pong), avoiding copies.

### 6.3 prefill↔decode batch conversion (`CopyKVCache`, line 144)

KV cache switches between prefill (batch=B) and decode (batch=B×candidates):
- **First prefill-after-decode**: `SelectAndCopyFrom` selects one batch index from decode's large batch to copy back to prefill batch (`src_index_to_copy_on_prefill`, line 172-178).
- **First decode-after-prefill**: `BroadcastAndCopyFrom` broadcasts batch=1 KV to decode's multi-candidate batch (line 180-186).

`RuntimeState.ran_decode` flag (`llm_executor_io_types.h`) decides "whether to prepare KVCache for prefill or decode".

### 6.4 KV cache interface (`kv_cache_interface.h`)

Abstract `KVCacheInterface`: `GetNumEntries` / `GetBatchSize` / `Serialize`/`Load` / `SelectAndCopyFrom` (by batch) / `BroadcastAndCopyFrom` (broadcast) / `DeepCopy`. Concrete implementations follow model KV layout, managed by executor.

### 6.5 Other runtime behaviors (`interfaces.py` / `engine.py`)

- `max_num_tokens`: KV cache capacity limit (= npu_tuned_export's `kv_cache_max_len`).
- `filter_channel_content_from_kv_cache`: filter thinking/reasoning tokens from KV cache.
- Python is a thin ctypes FFI shell over C++ (`litert_lm_engine_*` series of C functions).

## 7. Three Approaches Comparison (corrected)

| | npu_tuned_export | **LiteRT-LM** | Our current epair |
|---|---|---|---|
| Export format | ONNX | LiteRT FlatBuffer (.tflite) | epair (torch.export→ep_json) |
| Export tool | `torch.onnx.export` | ai_edge_torch (`torch.export`+Signature, **external**, not in this repo) | `torch.export` |
| prefill/decode organization | **Multiple static graphs** (loop export multiple graphs by seq_len) | **One model file with two signature runners** (prefill dynamic seq + decode fixed seq=1) | Not unified (forward has `if past is None` branch, takes one path by sample) |
| cache signature | Pre-allocated buffer + `new_kv_cache_pos` scatter + `output_pos` | KV cache explicit in/out, per-layer `kv_cache_k_N`/`kv_cache_v_N` (three naming conventions) | None (forward returns dict, cache is a list side effect) |
| seq length handling | Fixed seq_len multiple graphs | prefill dynamic (`kDynamicDimValue=-1` + `ResizeInputTensor`), decode fixed 1 | Fixed sample seq_len |
| Graph-weight separation | Yes (ONNX + .pb weights) | Yes (tflite_model + tflite_weights) | Yes (ep_json + weights_*.bin) |
| batch multi-candidate | — | KV cache `BroadcastAndCopyFrom`/`SelectAndCopyFrom` converts between prefill/decode | — |
| Packaging | Directory | **Single .litertlm file** (header+sections) | Directory |
| NPU optimization | do_opt (gemm2matmul / embed quant / lora) | Not verified in this round | None |
| Multimodal | No | Yes (vision + audio executor) | No |

## 8. Implications for our converter (path B, post-correction)

1. **path B (retain torch.export) direction validated**: LiteRT-LM uses `torch.export` (not ONNX). We don't need to switch to ONNX.
2. **"Unification" has two reference paradigms**:
   - **npu_tuned_export paradigm**: single static graph + pre-allocated buffer + `new_kv_cache_pos` scatter + `output_pos` + `attn_mask`/`position_ids` indexing — **one graph** covers prefill+decode.
   - **LiteRT-LM paradigm**: **two signatures** (prefill dynamic seq + decode fixed 1), KV cache explicit in/out, double-buffer ping-pong + batch broadcast/select — **two entries**.
   - For torch.export/epair (without TFLite's signature runner concept), LiteRT-LM's "two signatures" correspond to **exporting two epairs** (prefill epair + decode epair); npu_tuned_export's "single graph + scatter" corresponds to **one epair + dynamic indexing**. Both are feasible; selection TBD.
3. **Adoptable tensor contract** (from LiteRT-LM §5): `token_ids` / `positions` / `attn_mask` / `logits` + per-layer `kv_cache_k_N` / `kv_cache_v_N`. Clear naming, runtime auto-detection — our ExportWrapper can directly inherit, facilitating future LiteRT-LM-style backend integration.
4. **Graph-weight separation already aligned** (ep_json + weights_*.bin), no change needed.
5. **Runtime-side prefill/decode management**: C++ backend needs to track stage (`ran_decode` flag) + KV capacity (`max_num_tokens`) + double-buffer ping-pong + batch broadcast. This is backend's job, but the exported graph's signature must support it (KV cache in/out + position_ids + attn_mask).

## 9. litert-torch export source analysis (`/mnt/workspace/litert-torch-main/`)

### 9.1 Source location

litert-torch (formerly ai_edge_torch) is LiteRT-LM's model export tool, source at `/mnt/workspace/litert-torch-main/litert_torch/`.

### 9.2 .tflite export entry call chain

```
User code
  │
  ▼
litert_torch/__init__.py              → exports convert(), signature()
  │
  ▼
(The flowchart below references litert framework external dependency source, line numbers vary with upstream version)
_convert/interface.py                convert()           ← Python API entry
  │  Creates Converter, calls Converter.convert()
  ▼
_convert/interface.py               Converter.convert() ← adds signatures + triggers conversion
  │  Iterates Signature list
  ▼
_convert/core.py                    convert_signatures()← calls torch.export for each signature
  │  torch.export.export(module, args, kwargs, dynamic_shapes)
  │  → FX passes (optimization)
  ▼
_convert/litert_converter.py         exported_programs_to_flatbuffer() ← MLIR → .tflite
  │  ExportedProgram → MLIR → convert_to_tfl passes → flatbuffer
  ▼
model.py                            LiteRTModel.export(path) ← writes .tflite file
```

Minimal usage (`generative/examples/test_models/convert_toy_model.py`):

```python
import litert_torch

edge_model = (
    litert_torch.signature("prefill", model, sample_kwargs={...})
    .signature("decode", model, sample_kwargs={...})
    .convert()
)
edge_model.export("model.tflite")
```

### 9.3 Prefill/Decode signature definition

litert-torch does not directly export the original model, **always unifies input/output via a standardized wrapper**:

| Signature | Input | Output | seq |
|------|------|------|------|
| `prefill_{N}` or dynamic `prefill` | tokens, input_pos, kv_cache, mask | **only kv_cache** (no logits returned) | dynamic or N |
| `decode` | tokens, input_pos, kv_cache, mask | **kv_cache + logits** | fixed 1 |

Signatures are returned via `get_sample_inputs()` as `dict[signature_name -> (sample_kwargs, dynamic_shapes)]`, iterated by `export_text_prefill_decode_model` to call `converter.add_signature()`.

### 9.4 KV Cache Handling

**Shape:** `[1, batch*num_kv_heads, cache_length, head_dim]` (ts_idx=2) or `[1, batch*num_kv_heads, head_dim, cache_length]` (ts_idx=3, transposed). K and V can have different ts_idx.

**Naming (pytree flatten):** `k_0, v_0, k_1, v_1, ...` (one pair per layer); split-cache variant is `k_0, v_0, k_0_slice, v_0_slice, ...`.

**Update mechanism:**
- Default: `dynamic_update_slice(cache, new_slice, indices)` scatter write
- Split-cache: only returns new slice, merged by independent `CacheUpdate` signature

**Input/output layout:** KV cache is both input and output — zero-initialized full cache tensor passed as input, updated cache returned as output, next step as input (ping-pong swap).

### 9.5 How litert-torch handles inconsistent model input/output

litert-torch's core strategy is **never directly export the original model, always unify input/output via a standardized wrapper**. Three layers of isolation:

#### Layer 1: Fixed wrapper interface

```python
# Regardless of gemma4/qwen3_5, wrapper's forward signature is fixed:
def forward(self, tokens, input_pos, kv_cache, mask):
    inputs = self.adapt_inputs(tokens, input_pos, kv_cache, mask)  # translate differences
    output = self.model(**inputs)                                   # call original model
    return {"kv_cache": output.past_key_values}                    # standardized return

# adapt_inputs internally handles differences:
# - Some models want input_ids, some want inputs_embeds
# - Some want position_ids, some compute internally
# - Some want attention_mask, some use cache_position
```

#### Layer 2: KV cache pytree registration

Turns KV cache from "implicit return value" to "graph explicit input/output":

```
Current approach (ours):
  forward(input_ids, past_key_values=None) → {"past_key_values": updated_kv_cache}
  past_key_values is a Python list object (28 tensor pairs)
  torch.export sees a "black box", cannot expose as graph input/output

litert-torch approach:
  1. Define LiteRTLMCache class holding 28 tensor pairs
  2. Register as pytree, tells torch.export "this object is just a bunch of tensors, can be unpacked"
  3. torch.export sees inputs: k_0, v_0, k_1, v_1, ..., k_27, v_27
  4. torch.export sees outputs: k_0', v_0', k_1', v_1', ..., k_27', v_27'

Analogy:
  Current: ship a sealed box, courier doesn't know what's inside
  litert-torch: unpack into individually labeled items, each independently traceable
```

pytree registration is the step "telling torch.export how to unpack and pack". After registration, regardless of whether the original model uses list, DynamicCache, or other cache types, to torch.export it all becomes `k_0, v_0, k_1, v_1, ...` a bunch of plain tensors.

#### Layer 3: Attention implementation replacement + model patch

```python
# Attention replacement: register custom implementation, replacing model's native attention
model.set_attn_implementation('lrt_transposed_attention')

# Model patch registry: register model-specific handling by model type
@register_model_patch(["gemma4"])
def patch_gemma4(model, export_config):
    # Replace RMSNorm → litert RMSNorm
    # Replace VisionEncoder → litert optimized version
    # Handle per-layer embeddings
```

#### Layer 4: Signature API explicit declaration

```python
# Input explicitly specified by sample_kwargs (not inferred):
converter.add_signature("prefill", module,
    sample_kwargs={"tokens": t, "input_pos": p, "kv_cache": c, "mask": m})

# Output determined by forward's returned dict keys:
# prefill → {"kv_cache": ...}        → output only kv_cache
# decode  → {"kv_cache":..., "logits":...} → output has kv_cache + logits
```

### 9.6 Complete export flow (HF model → .tflite/.litertlm)

```
export(model="hf/repo", output_dir, task="text_generation", prefill_lengths=[128], ...)
  │
  ├─ Build ExportableModuleConfig (dataclass)
  ├─ run_export_tasks(export_tasks, export_config)
  │   ├─ load_model: AutoConfig + AutoModel + model patch
  │   ├─ export_text_prefill_decode_model:
  │   │   ├─ Construct prefill_module, decode_module (standardized wrapper)
  │   │   ├─ converter = Converter()
  │   │   ├─ for sig_name in prefill_module.get_sample_inputs():
  │   │   │     converter.add_signature(name, module, sample_kwargs=..., dynamic_shapes=...)
  │   │   ├─ for sig_name in decode_module.get_sample_inputs():
  │   │   │     converter.add_signature(name, module, sample_kwargs=..., dynamic_shapes=...)
  │   │   ├─ lrt_model = converter.convert()  ← torch.export + FX passes + MLIR → .tflite
  │   │   └─ lrt_model.export("model.tflite")
  │   ├─ [optional] aot_compile_model, export_embedder_model, export_auxiliary_model
  │   ├─ export_tokenizer
  │   └─ [optional] package_model → model.litertlm
  └─ cleanup
```

### 9.7 Key source index

| File | Purpose |
|------|------|
| `litert_torch/__init__.py` | Public API exports |
| `litert_torch/_convert/interface.py` | `Converter` class, `convert()`/`signature()` |
| `litert_torch/_convert/signature.py` | `Signature` dataclass |
| `litert_torch/_convert/core.py` | `convert_signatures()` — calls `torch.export.export` |
| `litert_torch/_convert/litert_converter.py` | `exported_programs_to_flatbuffer()` — MLIR→tflite |
| `litert_torch/generative/export_hf/export.py` | Public `export()` orchestration function |
| `litert_torch/generative/export_hf/core/export_lib.py` | `load_model`, `export_text_prefill_decode_model` |
| `litert_torch/generative/export_hf/core/exportable_module.py` | Default prefill/decode wrapper + `get_sample_inputs` |
| `litert_torch/generative/export_hf/core/exportable_module_config.py` | `ExportableModuleConfig` dataclass |
| `litert_torch/generative/export_hf/core/cache.py` | `LiteRTLMCache` (default, DUS-based) |
| `litert_torch/generative/export_hf/core/split_cache/cache.py` | `LiteRTLMSplitCache` (NPU split cache) |
| `litert_torch/generative/export_hf/core/attention.py` | `lrt_transposed_attention` registration |
| `litert_torch/generative/export_hf/model_ext/patches.py` | Model patch registry |
| `litert_torch/generative/export_hf/core/litert_lm_builder.py` | `package_model` → `.litertlm` |

### 9.8 Improvement suggestions for our epair pipeline

| Improvement | litert-torch approach | Our current status | Suggestion |
|--------|------------------|-------------|------|
| wrapper | `ExportableModule` fixed forward signature + `adapt_inputs()` | `ExportWrapper` only returns logits | Add `adapt_inputs()` to translate model differences |
| KV cache | `LiteRTLMCache` pytree registration, explicit graph I/O | Implicit return dict's `past_key_values` | Register pytree, turn cache into explicit graph input/output |
| attention | Replace with `lrt_transposed_attention` | Model native | Don't change for now, consider for future NPU optimization |
| Model patch | `@register_model_patch` registry | None | Short-term use converter adapter, long-term introduce patch registry |
| Dynamic seq | `torch.export.Dim("prefill_length")` | Fixed sample seq_len | Support dynamic prefill length later |
| Two signatures | `prefill` + `decode` two signatures | Single epair | Later export `prefill_epair` + `decode_epair` |

## 10. Key source index (LiteRT-LM runtime local source)

| File | Purpose | Confirmed content |
|------|------|------|
| `python/litert_lm_builder/litertlm_core.py` | Bundle file format constants | `.litertlm` header schema, section data type, version 1.5.0 |
| `python/litert_lm_builder/litertlm_builder.py` | Packaging API | `LitertLmFileBuilder` + `TfLiteModelType.PREFILL_DECODE` + `add_tflite_model/weights/tokenizer` |
| `docs/litert_lm_builder.md` | Usage | CLI chaining + Python API demo |
| `runtime/executor/llm_litert_compiled_model_executor.{h,cc}` | C++ executor | `kPrefillSignatureRunner="prefill"` / `kDecodeSignatureRunner="decode"`, KV double-buffer ping-pong, batch broadcast/select |
| `runtime/executor/litert_compiled_model_executor_utils.{h,cc}` | Signature/tensor name utilities | `ModelSignatures` struct, default tensor name constants, `GetKVCacheRootNames` (three KV naming conventions) |
| `runtime/executor/llm_executor_io_types.h` | I/O types | `RuntimeState.ran_decode` (prefill/decode stage flag) |
| `runtime/executor/kv_cache_interface.h` | KV cache abstract interface | `SelectAndCopyFrom` / `BroadcastAndCopyFrom` / `Serialize`/`Load` |
| `python/litert_lm/{interfaces,engine}.py` | Runtime FFI | `max_num_tokens`, prefill/decode counts, `filter_channel_content_from_kv_cache` |
