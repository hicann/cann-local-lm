# LiteRT-LM 导出逻辑分析（参考知识库）

> 分析对象：`google-ai-edge/LiteRT-LM`（本地源码 `/mnt/workspace/LiteRT-LM-main/`）+ `litert-torch`（本地源码 `/mnt/workspace/litert-torch-main/`）
> 用途：作为我们 converter 导出方案的外部参考。
> 最后更新：2026-07-14（第三轮：litert-torch 源码详细分析）

## 1. LiteRT-LM 是什么

Google 的**生产级 LLM 编排层**，跑在 LiteRT（TFLite 后继）上，面向 Android/iOS/Web/桌面/IoT。**LiteRT-LM 本身不是模型导出器**，而是"导出后产物的打包 + 运行时编排"。README 原话："production-ready orchestration layer to run LLMs with LiteRT"。

支持 Gemma / Llama / Phi-4 / Qwen 等；产物是 `.litertlm` 单文件。

## 2. 分层架构（本地源码确认）

```
┌──────────────────────────────────────────────────────────────┐
│ 模型导出层（不在本仓库——全仓 grep "torch.export/ai_edge_torch/Signature" 0 命中） │
│   HF PyTorch 模型 → ai_edge_torch (torch.export + Signature) → .tflite │
│   产出: 一个 .tflite 内含两个签名 runner: "prefill" + "decode"      │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 打包层（python/litert_lm_builder/）                            │
│   litert-lm-builder: .tflite + weights + tokenizer + metadata│
│                      → .litertlm 容器                         │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 运行时（runtime/ C++ + python/litert_lm/ ctypes FFI）         │
│   C++ executor: 调 prefill/decode 两个签名 runner、管理 KV cache │
└──────────────────────────────────────────────────────────────┘
```

**模型导出层在本仓不存在**（`grep -rIln "ai_edge_torch\|torch.export\|Signature" .` 0 命中）。LiteRT-LM 只消费现成 `.tflite`。导出靠外部 `ai_edge_torch`（`torch.export` + Signature API），把 KV cache 作为显式 in/out。*（ai_edge_torch 源码不在本仓，但其契约可从 LiteRT-LM runtime 的张量名消费反推——见 §5。）*

## 3. `.litertlm` bundle 结构（确认）

`LitertLmFileBuilder` 的 API（`python/litert_lm_builder/litertlm_builder.py`）：

| API | 作用 | 对照我们 epair 输出 |
|------|------|------|
| `add_system_metadata(Metadata)` | 键值元信息（Authors / TargetBackend） | （可放 model_config） |
| `add_llm_metadata(...)` | LLM 元信息（config 等） | `model_config.json` |
| `add_tflite_model(path, model_type=PREFILL_DECODE)` | **主图** | `ep_json_with_ascend_ir.json` |
| `add_tflite_weights(...)` | **独立权重**（图权分离） | `weights_*.bin` + `weight_info.json` |
| `add_sentencepiece_tokenizer` / `add_hf_tokenizer` | tokenizer | tokenizer 文件 |
| `add_generic_binary_data(...)` | 任意二进制 section | — |
| `build(stream)` | 序列化成单 `.litertlm` 文件 | （我们是目录，可加打包步骤） |

图权分离和我们的 `ep_json` + `weights_*.bin` **同构**。差异只在它打包成单文件。

## 4. `TfLiteModelType`（确认，`litertlm_builder.py:208`）

```python
class TfLiteModelType(enum.Enum):
    PREFILL_DECODE = "tf_lite_prefill_decode"      # 一个 .tflite 含 prefill + decode 两个签名
    ARTISAN_TEXT_DECODER = "tf_lite_artisan_text_decoder"
```

> **修正第一轮的误述**：`PREFILL_DECODE` **不是"一张图统一 prefill+decode"**，而是**一个模型文件里有两个签名 runner**（`"prefill"` + `"decode"`，见 §5）。prefill 签名动态 seq、decode 签名固定 seq=1，KV cache 在两签名间由 runtime 管理。

## 5. 模型签名契约（本地源码确认，runtime/executor/）

### 5.1 两个签名 runner

`runtime/executor/llm_litert_compiled_model_executor.cc:80-81`：

```cpp
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner  = "decode";
```

runtime 通过 `compiled_model.FindSignature(signature)` 取对应签名 runner，按阶段调 prefill 或 decode。**两个签名共用同一套输入张量名**（`ModelSignatures` 结构体，prefill/decode 通用）。

### 5.2 `ModelSignatures` 结构体（`litert_compiled_model_executor_utils.h:55`）

```cpp
struct ModelSignatures {
  std::string input_tokens;                       // 必需
  std::string input_positions;                     // 必需
  std::optional<std::string> input_attn_mask;      // 可选
  std::optional<std::string> input_embeddings;     // 可选（设了就外部做 embedding 查表，不走 token_ids）
  std::optional<std::string> input_per_layer_embeddings;  // 可选
  std::optional<std::string> input_int32_param;    // 可选
  std::string output_logits;                      // 必需（decode 用）
};
```

### 5.3 默认张量名（`litert_compiled_model_executor_utils.cc:71-89`）

runtime 按"候选名列表"自动匹配（支持多种命名变体）：

| 语义 | 默认名 | 候选 |
|------|--------|------|
| 输入 token | `token_ids` | `token_ids` / `tokens` |
| 输入位置 | `positions` | `positions` / `input_pos` |
| attention mask | `attn_mask` | `attn_mask` / `mask` |
| 输入 embedding | `embeddings` | `embeddings` |
| per-layer embedding | `per_layer_embeddings` | `per_layer_embeddings` |
| int32 参数 | `param_tensor` | `param_tensor` |
| 输出 logits | `logits` | `logits` |

### 5.4 KV cache 张量命名约定（`GetKVCacheRootNames`, `utils.cc:189`）

KV cache 是**每层一对 (k, v) 显式输入也显式输出**，按"根名 + 层号"命名。runtime 扫输入/输出名自动探测三种约定之一：

| 约定 | 输入样例 | k 根名 | v 根名 |
|------|---------|--------|--------|
| 标准（分 k/v） | `kv_cache_k_0` / `kv_cache_k_1` … | `kv_cache_k_` | `kv_cache_v_` |
| 简写 | `k_cache_0` / `v_cache_0` … | `k_cache_` | `v_cache_` |
| 合并（k+v 同张量） | `kv_cache_c_0` … | `kv_cache_c_` | `kv_cache_c_` |

找不到任一约定则 `FailedPreconditionError("No KV cache inputs found.")`。即模型导出时**必须**按这三种命名之一暴露 KV cache in/out。

## 6. prefill / decode 运行时机制（确认，`llm_litert_compiled_model_executor.cc`）

### 6.1 动态形状（prefill）

prefill 支持**动态 seq_len**：`kDynamicDimValue = -1`（line 82），运行时 `ResizeInputTensor(signature, tensor_name, new_shape)` 按当前 prompt 长度 resize（line 254, 440）。decode 则固定 seq=1。

### 6.2 KV cache 双缓冲 + 乒乓

```cpp
input_kv_cache_buffers_  = &kv_cache_buffers_1;   // line 550-552
output_kv_cache_buffers_ = &kv_cache_buffers_2;
...
std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // line 761
```

每步把上一步的输出 KV buffer 当作本步输入（乒乓），避免拷贝。

### 6.3 prefill↔decode 的 batch 转换（`CopyKVCache`, line 144）

KV cache 在 prefill（batch=B）和 decode（batch=B×candidates）间切换：
- **首次 prefill-after-decode**：`SelectAndCopyFrom` 从 decode 的大 batch 选一个 batch index 拷回 prefill batch（`src_index_to_copy_on_prefill`，line 172-178）。
- **首次 decode-after-prefill**：`BroadcastAndCopyFrom` 把 batch=1 的 KV 广播到 decode 的多 candidate batch（line 180-186）。

`RuntimeState.ran_decode` 标志位（`llm_executor_io_types.h`）决定"该按 prefill 还是 decode 准备 KVCache"。

### 6.4 KV cache 接口（`kv_cache_interface.h`）

抽象 `KVCacheInterface`：`GetNumEntries` / `GetBatchSize` / `Serialize`/`Load` / `SelectAndCopyFrom`（按 batch 选） / `BroadcastAndCopyFrom`（广播） / ` DeepCopy`。具体实现按模型 KV 布局，由 executor 管。

### 6.5 其它 runtime 行为（`interfaces.py` / `engine.py`）

- `max_num_tokens`：KV cache 容量上限（= npu_tuned_export 的 `kv_cache_max_len`）。
- `filter_channel_content_from_kv_cache`：把 thinking/reasoning token 从 KV cache 过滤掉。
- Python 是 C++ 的 ctypes FFI 薄壳（`litert_lm_engine_*` 系列 C 函数）。

## 7. 三种方案对照（已修正）

| | npu_tuned_export | **LiteRT-LM** | 我们当前 epair |
|---|---|---|---|
| 导出格式 | ONNX | LiteRT FlatBuffer (.tflite) | epair (torch.export→ep_json) |
| 导出工具 | `torch.onnx.export` | ai_edge_torch (`torch.export`+Signature，**外部**，不在本仓) | `torch.export` |
| prefill/decode 组织 | **多静态图**（按 seq_len 循环导多张） | **一个模型文件两个签名 runner**（prefill 动态 seq + decode 固定 seq=1） | 未统一（forward 带 `if past is None` 分支，按 sample 走一条） |
| cache 签名 | 预分配 buffer + `new_kv_cache_pos` scatter + `output_pos` | KV cache 显式 in/out，每层 `kv_cache_k_N`/`kv_cache_v_N`（三种命名约定） | 无（forward 返回 dict，cache 是 list 副作用） |
| seq 长度处理 | 固定 seq_len 多图 | prefill 动态（`kDynamicDimValue=-1` + `ResizeInputTensor`），decode 固定 1 | 固定 sample seq_len |
| 图权分离 | 是（ONNX + .pb 权重） | 是（tflite_model + tflite_weights） | 是（ep_json + weights_*.bin） |
| batch 多候选 | — | KV cache `BroadcastAndCopyFrom`/`SelectAndCopyFrom` 在 prefill/decode 间转换 | — |
| 打包 | 目录 | **单 .litertlm 文件**（header+sections） | 目录 |
| NPU 调优 | do_opt（gemm2matmul / embed 量化 / lora） | 未在本轮核实 | 无 |
| 多模态 | 否 | 是（vision + audio executor） | 否 |

## 8. 对我们 converter 的启示（path B，修正后）

1. **path B（保留 torch.export）方向被验证**：LiteRT-LM 用的就是 `torch.export`（不是 ONNX）。我们不必切 ONNX。
2. **"统一"有两种参考范式**：
   - **npu_tuned_export 范式**：单张静态图 + 预分配 buffer + `new_kv_cache_pos` scatter + `output_pos` + `attn_mask`/`position_ids` 索引——**一张图**覆盖 prefill+decode。
   - **LiteRT-LM 范式**：**两个签名**（prefill 动态 seq + decode 固定 1），KV cache 显式 in/out、双缓冲乒乓 + batch 广播/选择——**两个入口**。
   - 对 torch.export/epair（没有 TFLite 的 signature runner 概念），LiteRT-LM 的"两签名"对应**导出两个 epair**（prefill epair + decode epair）；npu_tuned_export 的"单图 + scatter"对应**一个 epair + 动态索引**。两者都可行，选型待定。
3. **可采纳的张量契约**（来自 LiteRT-LM §5）：`token_ids` / `positions` / `attn_mask` / `logits` + 每层 `kv_cache_k_N` / `kv_cache_v_N`。命名清晰、runtime 自动探测——我们的 ExportWrapper 可直接沿用，便于将来和 LiteRT-LM 风格的后端对接。
4. **图权分离已对齐**（ep_json + weights_*.bin），无需改。
5. **runtime 侧 prefill/decode 管理**：C++ 后端需跟踪阶段（`ran_decode` 标志）+ KV 容量（`max_num_tokens`）+ 双缓冲乒乓 + batch 广播。这是后端的事，但导出图的签名要支持它（KV cache in/out + position_ids + attn_mask）。

## 9. litert-torch 导出源码分析（`/mnt/workspace/litert-torch-main/`）

### 9.1 源码位置

litert-torch（原 ai_edge_torch）是 LiteRT-LM 的模型导出工具，源码在 `/mnt/workspace/litert-torch-main/litert_torch/`。

### 9.2 .tflite 导出入口调用链

```
用户代码
  │
  ▼
litert_torch/__init__.py              → 导出 convert(), signature()
  │
  ▼
_convert/interface.py:274             convert()           ← Python API 入口
  │  创建 Converter，调用 Converter.convert()
  ▼
_convert/interface.py:131            Converter.convert() ← 添加签名 + 触发转换
  │  遍历 Signature 列表
  ▼
_convert/core.py:79                  convert_signatures()← 对每个签名调 torch.export
  │  torch.export.export(module, args, kwargs, dynamic_shapes)
  │  → FX passes (优化)
  ▼
_convert/litert_converter.py:104      exported_programs_to_flatbuffer() ← MLIR → .tflite
  │  ExportedProgram → MLIR → convert_to_tfl passes → flatbuffer
  ▼
model.py:182                         LiteRTModel.export(path) ← 写 .tflite 文件
```

最简用法（`generative/examples/test_models/convert_toy_model.py`）：

```python
import litert_torch

edge_model = (
    litert_torch.signature("prefill", model, sample_kwargs={...})
    .signature("decode", model, sample_kwargs={...})
    .convert()
)
edge_model.export("model.tflite")
```

### 9.3 Prefill/Decode 签名定义

litert-torch 不直接导出原始模型，**始终通过标准化 wrapper 统一输入输出**：

| 签名 | 输入 | 输出 | seq |
|------|------|------|------|
| `prefill_{N}` 或动态 `prefill` | tokens, input_pos, kv_cache, mask | **仅 kv_cache**（不返回 logits） | 动态或 N |
| `decode` | tokens, input_pos, kv_cache, mask | **kv_cache + logits** | 固定 1 |

签名通过 `get_sample_inputs()` 返回 `dict[signature_name -> (sample_kwargs, dynamic_shapes)]`，由 `export_text_prefill_decode_model` 遍历调用 `converter.add_signature()`。

### 9.4 KV Cache 处理

**形状：** `[1, batch*num_kv_heads, cache_length, head_dim]`（ts_idx=2）或 `[1, batch*num_kv_heads, head_dim, cache_length]`（ts_idx=3，转置）。K 和 V 可以有不同的 ts_idx。

**命名（pytree flatten）：** `k_0, v_0, k_1, v_1, ...`（每层一对），split-cache 变体为 `k_0, v_0, k_0_slice, v_0_slice, ...`。

**更新机制：**
- 默认：`dynamic_update_slice(cache, new_slice, indices)` 散射写入
- Split-cache：只返回新 slice，由独立的 `CacheUpdate` 签名合并

**输入输出布局：** KV cache 既是输入也是输出——零初始化全量 cache tensor 作为输入传入，更新后的 cache 作为输出返回，下一步作为输入（乒乓交换）。

### 9.5 litert-torch 如何处理不同模型输入输出不一致

litert-torch 的核心策略是**永远不直接导出原始模型，始终通过标准化 wrapper 统一输入输出**。通过三层隔离：

#### 第 1 层：固定 wrapper 接口

```python
# 不论 gemma4/qwen3/qwen3_5，wrapper 的 forward 签名固定:
def forward(self, tokens, input_pos, kv_cache, mask):
    inputs = self.adapt_inputs(tokens, input_pos, kv_cache, mask)  # 翻译差异
    output = self.model(**inputs)                                   # 调原始模型
    return {"kv_cache": output.past_key_values}                    # 标准化返回

# adapt_inputs 内部处理差异:
# - 有的模型要 input_ids，有的要 inputs_embeds
# - 有的模型要 position_ids，有的自己算
# - 有的模型要 attention_mask，有的用 cache_position
```

#### 第 2 层：KV cache pytree 注册

将 KV cache 从"隐式返回值"变为"图显式输入输出"：

```
当前做法 (我们):
  forward(input_ids, past_key_values=None) → {"past_key_values": updated_kv_cache}
  past_key_values 是一个 Python list 对象 (28 对 tensor)
  torch.export 看到的是"黑盒"，无法暴露为图输入输出

litert-torch 做法:
  1. 定义 LiteRTLMCache 类，装着 28 对 tensor
  2. 注册为 pytree，告诉 torch.export "这个对象就是一堆 tensor，可以拆开"
  3. torch.export 看到的输入: k_0, v_0, k_1, v_1, ..., k_27, v_27
  4. torch.export 看到的输出: k_0', v_0', k_1', v_1', ..., k_27', v_27'

比喻:
  当前: 寄一个密封箱子，快递公司不知道里面是什么
  litert-torch: 拆箱成一个个贴标签的物品，每个都是独立可追踪的
```

pytree 注册就是"告诉 torch.export 怎么拆箱和装箱"的那一步。注册后，无论原始模型用 list、DynamicCache 还是其他 cache 类型，对 torch.export 都变成了 `k_0, v_0, k_1, v_1, ...` 这样一堆普通 tensor。

#### 第 3 层：Attention 实现替换 + 模型 patch

```python
# Attention 替换: 注册自定义实现，替代模型原生 attention
model.set_attn_implementation('lrt_transposed_attention')

# 模型 patch 注册表: 按模型类型注册特有处理
@register_model_patch(["gemma4"])
def patch_gemma4(model, export_config):
    # 替换 RMSNorm → litert RMSNorm
    # 替换 VisionEncoder → litert 优化版
    # 处理 per-layer embeddings
```

#### 第 4 层：Signature API 显式声明

```python
# 输入由 sample_kwargs 显式指定 (不推断):
converter.add_signature("prefill", module,
    sample_kwargs={"tokens": t, "input_pos": p, "kv_cache": c, "mask": m})

# 输出由 forward 返回的 dict keys 决定:
# prefill → {"kv_cache": ...}        → 输出只有 kv_cache
# decode  → {"kv_cache":..., "logits":...} → 输出有 kv_cache + logits
```

### 9.6 完整导出流程 (HF 模型 → .tflite/.litertlm)

```
export(model="hf/repo", output_dir, task="text_generation", prefill_lengths=[128], ...)
  │
  ├─ Build ExportableModuleConfig (dataclass)
  ├─ run_export_tasks(export_tasks, export_config)
  │   ├─ load_model: AutoConfig + AutoModel + 模型 patch
  │   ├─ export_text_prefill_decode_model:
  │   │   ├─ 构造 prefill_module, decode_module (标准化 wrapper)
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

### 9.7 关键源码索引

| 文件 | 作用 |
|------|------|
| `litert_torch/__init__.py` | 公共 API 导出 |
| `litert_torch/_convert/interface.py` | `Converter` 类, `convert()`/`signature()` |
| `litert_torch/_convert/signature.py` | `Signature` dataclass |
| `litert_torch/_convert/core.py` | `convert_signatures()` — 调 `torch.export.export` |
| `litert_torch/_convert/litert_converter.py` | `exported_programs_to_flatbuffer()` — MLIR→tflite |
| `litert_torch/generative/export_hf/export.py` | 公共 `export()` 编排函数 |
| `litert_torch/generative/export_hf/core/export_lib.py` | `load_model`, `export_text_prefill_decode_model` |
| `litert_torch/generative/export_hf/core/exportable_module.py` | 默认 prefill/decode wrapper + `get_sample_inputs` |
| `litert_torch/generative/export_hf/core/exportable_module_config.py` | `ExportableModuleConfig` dataclass |
| `litert_torch/generative/export_hf/core/cache.py` | `LiteRTLMCache` (默认, DUS-based) |
| `litert_torch/generative/export_hf/core/split_cache/cache.py` | `LiteRTLMSplitCache` (NPU split cache) |
| `litert_torch/generative/export_hf/core/attention.py` | `lrt_transposed_attention` 注册 |
| `litert_torch/generative/export_hf/model_ext/patches.py` | 模型 patch 注册表 |
| `litert_torch/generative/export_hf/core/litert_lm_builder.py` | `package_model` → `.litertlm` |

### 9.8 对我们 epair 管线的改进建议

| 改进项 | litert-torch 做法 | 我们当前状态 | 建议 |
|--------|------------------|-------------|------|
| wrapper | `ExportableModule` 固定 forward 签名 + `adapt_inputs()` | `ExportWrapper` 只返回 logits | 增加 `adapt_inputs()` 翻译模型差异 |
| KV cache | `LiteRTLMCache` pytree 注册，显式图 I/O | 隐式返回 dict 中的 `past_key_values` | 注册 pytree，将 cache 变为图显式输入输出 |
| attention | 替换为 `lrt_transposed_attention` | 模型原生 | 暂不改，后续 NPU 优化时考虑 |
| 模型 patch | `@register_model_patch` 注册表 | 无 | 短期用 converter adapter，长期引入 patch 注册表 |
| 动态 seq | `torch.export.Dim("prefill_length")` | 固定 sample seq_len | 后续支持动态 prefill 长度 |
| 两签名 | `prefill` + `decode` 两个签名 | 单个 epair | 后续导出 `prefill_epair` + `decode_epair` |

## 10. 关键源码索引（LiteRT-LM runtime 本地源码）

| 文件 | 作用 | 确认内容 |
|------|------|------|
| `python/litert_lm_builder/litertlm_core.py` | bundle 文件格式常量 | `.litertlm` header schema、section data type、版本 1.5.0 |
| `python/litert_lm_builder/litertlm_builder.py` | 打包 API | `LitertLmFileBuilder` + `TfLiteModelType.PREFILL_DECODE` + `add_tflite_model/weights/tokenizer` |
| `docs/litert_lm_builder.md` | 用法 | CLI chaining + Python API demo |
| `runtime/executor/llm_litert_compiled_model_executor.{h,cc}` | C++ executor | `kPrefillSignatureRunner="prefill"` / `kDecodeSignatureRunner="decode"`、KV 双缓冲乒乓、batch 广播/选择 |
| `runtime/executor/litert_compiled_model_executor_utils.{h,cc}` | 签名/张量名工具 | `ModelSignatures` 结构体、默认张量名常量、`GetKVCacheRootNames`（三种 KV 命名约定） |
| `runtime/executor/llm_executor_io_types.h` | I/O 类型 | `RuntimeState.ran_decode`（prefill/decode 阶段标志） |
| `runtime/executor/kv_cache_interface.h` | KV cache 抽象接口 | `SelectAndCopyFrom` / `BroadcastAndCopyFrom` / `Serialize`/`Load` |
| `python/litert_lm/{interfaces,engine}.py` | runtime FFI | `max_num_tokens`、prefill/decode 计数、`filter_channel_content_from_kv_cache` |
