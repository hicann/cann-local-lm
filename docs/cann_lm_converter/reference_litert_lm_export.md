# LiteRT-LM 导出逻辑分析（参考知识库）

> 分析对象：`google-ai-edge/LiteRT-LM`（main 分支）
> 用途：作为我们 converter "目标导出方案：统一 cache 签名（path B）" 的外部参考。
> 最后更新：2026-07-14

## 1. LiteRT-LM 是什么

Google 的**生产级 LLM 编排层**，跑在 LiteRT（TFLite 后继）上，面向 Android/iOS/Web/桌面/IoT。LiteRT-LM 本身**不是模型导出器**，而是"导出后产物的打包 + 运行时编排"。README 原话："production-ready orchestration layer to run LLMs with LiteRT"。

支持 Gemma / Llama / Phi-4 / Qwen 等；产物是 `.litertlm` 单文件。

## 2. 分层架构（已从源码确认）

```
┌──────────────────────────────────────────────────────────────┐
│ 模型导出层（不在本仓库，外部 ai_edge_torch）                  │
│   HF PyTorch 模型 → torch.export + Signature → .tflite        │
│   产出: 带统一 cache 签名的 PREFILL_DECODE 单图              │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 打包层（本仓库 python/litert_lm_builder/）                    │
│   litert-lm-builder: .tflite + weights + tokenizer + metadata│
│                      → .litertlm 容器                         │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 运行时（本仓库 c/engine.cc + python/litert_lm/ FFI）          │
│   C++ engine: 管理 KV cache、prefill/decode 阶段、采样        │
└──────────────────────────────────────────────────────────────┘
```

- **模型导出层**：本仓库不含。LiteRT-LM 消费现成 `.tflite`，导出靠 Google 的 `ai_edge_torch`（`torch.export` + `Signature` API，把 KV cache 作为显式 in/out）。*（注：ai_edge_torch 源码未在本轮抓取中直接核实，由"builder 只打包 .tflite + Google 工具链常识"推断；结论高置信但未逐行验证。）*
- **打包层**：`python/litert_lm_builder/litertlm_builder.py`（30KB），`LitertLmFileBuilder` 类。
- **运行时**：`c/engine.cc`（C++）+ `python/litert_lm/{engine,interfaces}.py`（ctypes FFI 薄壳）。

## 3. `.litertlm` bundle 结构（已确认）

`LitertLmFileBuilder` 的 API（`litertlm_builder.py`）：

| API | 作用 | 对照我们 epair 输出 |
|------|------|------|
| `add_system_metadata(Metadata)` | 键值元信息（Authors / TargetBackend） | （无直接对应，可放 model_config） |
| `add_llm_metadata(...)` | LLM 元信息（config 等） | `model_config.json` |
| `add_tflite_model(path, model_type=PREFILL_DECODE)` | **主图**（graph） | `ep_json_with_ascend_ir.json` |
| `add_tflite_weights(...)` | **独立权重**（图权分离） | `weights_*.bin` + `weight_info.json` |
| `add_sentencepiece_tokenizer(path)` / `add_hf_tokenizer(...)` | tokenizer | tokenizer 文件 |
| `add_generic_binary_data(...)` | 任意二进制 section | — |
| `build(stream)` | 序列化成单 `.litertlm` 文件 | （我们是目录，可加打包步骤） |

**关键**：LiteRT-LM 把图和权重**分开放**（`add_tflite_model` + `add_tflite_weights`），和我们 epair 的 `ep_json` + `weights_*.bin` **完全一致**。差异只在：它最终打包成单文件，我们是目录。

## 4. `TfLiteModelType` 枚举（已确认，`litertlm_builder.py:208`）

```python
class TfLiteModelType(enum.Enum):
    PREFILL_DECODE = "tf_lite_prefill_decode"      # 一张图统一 prefill + decode
    ARTISAN_TEXT_DECODER = "tf_lite_artisan_text_decoder"
```

**`PREFILL_DECODE` 是核心**：一张导出图同时覆盖 prefill 和 decode。这是通过 KV cache 作为显式输入也作为输出实现的（参考 `npu_tuned_export` 的统一签名思路）——**不是** npu_tuned_export 那种"固定 seq_len + 多静态图"。

## 5. 运行时行为（已确认，`interfaces.py` / `engine.py`）

- `max_num_tokens`：**KV cache 容量上限**（= npu_tuned_export 的 `kv_cache_max_len`）。
- runtime **区分 prefill / decode 阶段**：`BenchmarkInfo` 跟踪 `last_prefill_token_count` / `last_prefill_tokens_per_second` / `last_decode_token_count` / `last_decode_tokens_per_second` / `num_prefill_turns`。
- `filter_channel_content_from_kv_cache`：把 "channel" 内容（如 thinking/reasoning token）从 KV cache 里过滤掉，不持久化。
- Python 层是 C++ 的 ctypes FFI 薄壳（`litert_lm_engine_*` 系列 C 函数）。

## 6. 三种方案对照

| | npu_tuned_export | **LiteRT-LM** | 我们当前 epair |
|---|---|---|---|
| 导出格式 | ONNX | LiteRT FlatBuffer (.tflite) | epair (torch.export→ep_json) |
| 导出工具 | `torch.onnx.export` | `torch.export` + ai_edge_torch Signature | `torch.export` |
| prefill/decode | **多静态图**（按 seq_len 循环） | **一张统一图** (PREFILL_DECODE) | 未统一（forward 带 `if past is None` 分支，按 sample 走一条） |
| cache 签名 | 预分配 buffer + new_kv_cache_pos scatter | KV cache 显式 in/out（具体张量契约在 ai_edge_torch，未核实） | 无（forward 返回 dict，cache 是 list 副作用） |
| 图权分离 | 是（ONNX + 外部 .pb 权重） | 是（tflite_model + tflite_weights） | 是（ep_json + weights_*.bin） |
| 打包 | 目录 | **单 .litertlm 文件**（header+sections） | 目录 |
| NPU 调优 | do_opt（gemm2matmul / embed 量化 / lora） | 未在本轮核实 | 无 |
| 多模态 | 否 | 是（vision + audio） | 否 |

## 7. 对我们 converter 的启示（path B 验证 + 目标）

1. **path B（保留 torch.export + ExportWrapper 统一 cache 签名）方向被验证**：LiteRT-LM 用的就是 `torch.export`（不是 ONNX）+ 一张统一图。我们不必切 ONNX。
2. **目标是"一张 PREFILL_DECODE 图"**，不是 npu_tuned_export 的多静态图。即全量版（预分配 buffer + scatter-write + output_pos）应做成单图覆盖 prefill+decode，而非按 seq_len 导多张。
3. **图权分离我们已对齐**（ep_json + weights_*.bin），和 LiteRT-LM 的 tflite_model + tflite_weights 同构。无需改。
4. **可选打包步骤**：后续可加一个"把 epair 目录打包成单文件"的步骤（类比 `.litertlm`），便于分发；但非必须。
5. **runtime 侧的 prefill/decode 阶段管理**：C++ 后端需跟踪 prefill/decode 阶段 + KV cache 容量（max_num_tokens）。这是后端的事，但导出图的签名要支持它（KV cache in/out + 阶段指示）。
6. **待核实项**：ai_edge_torch 的具体张量契约（KV cache 的 shape/布局、prefill/decode 在单图里怎么区分——靠 dynamic shape 还是 new_kv_cache_pos 这类索引）。建议后续抓 `google-ai-edge/ai-edge-torch` 的 `generative` 模块签名定义来对齐我们的 ExportWrapper 张量契约。

## 8. 关键源码索引（本轮已抓取确认）

| 文件 | 作用 | 确认内容 |
|------|------|------|
| `python/litert_lm_builder/litertlm_core.py` | bundle 文件格式常量 | `.litertlm` header schema、section data type 枚举、版本 1.5.0 |
| `python/litert_lm_builder/litertlm_builder.py` | 打包 API | `LitertLmFileBuilder` + `TfLiteModelType.PREFILL_DECODE` + `add_tflite_model/weights/tokenizer` |
| `python/litert_lm_builder/README.md` + `docs/litert_lm_builder.md` | 用法 | CLI chaining + Python API demo |
| `python/litert_lm/interfaces.py` | runtime 接口 | `max_num_tokens`、prefill/decode 计数、`filter_channel_content_from_kv_cache` |
| `python/litert_lm/engine.py` | runtime FFI | ctypes 调 `litert_lm_engine_*` C 函数 |
| `c/engine.cc` | C++ engine | （未逐行抓取，由接口推断） |
| `npu_tuned_export/*`（gitee 参考） | 对照 | ONNX + 多静态图 + scatter-write attention |
