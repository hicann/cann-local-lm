# CANN-LM Converter Design Documentation (Overview)

English | [简体中文](../../zh/cann_lm_converter/README.md)

> A collection of design documents for the CLI tool that converts HuggingFace models to the epair format. Supports gemma4 / qwen3_5 (extensible later).
>
> The documentation is organized into three modules: **CLI / Model Export / Quantization**, with the rest as reference documents.

## Table of Contents

- [Document Layout](#document-layout)
- [End-to-End Flow Overview](#end-to-end-flow-overview)
- [Model Differences Overview](#model-differences-overview)
- [Accuracy Verification Summary](#accuracy-verification-summary)
- [Module Navigation](#module-navigation)
- [Pending and Legacy Items](#pending-and-legacy-items)
- [Quick Start](#quick-start)

---

<a id="document-layout"></a>
## Document Layout

```
docs/cann_lm_converter/
├── README.md                              ← This file (top-level overview + navigation)
│
├── cli/                                   ← CLI module
│   ├── cli_design.md                      ← CLI design
│   └── INDEX.md                           ← CLI module index
├── export/                                ← Model export module
│   ├── export_design.md                   ← Export design
│   └── INDEX.md                           ← Export module index
├── quant/                                 ← Quantization module
│   ├── quant_design.md                    ← Quantization design
│   └── INDEX.md                           ← Quantization module index
│
├── ─── Reference Documents ───
├── reference_litert_lm_export.md          ← LiteRT-LM external reference
├── reference_npu_op_elimination.md        ← NPU op elimination skill
└── feature_dump_design.md                ← Dump debug capability (per-op intermediate tensor dump at FX node level)
```

<a id="end-to-end-flow-overview"></a>
## End-to-End Flow Overview

The converter exports HF models to epair, which is then loaded and compiled to OM by the C++ backend. The three stages are orchestrated by the CLI, with each stage belonging to a module as follows:

```
cann_lm_converter.py (CLI)                                          ← cli_design.md
    │
    ├── parse_args()  ← --hf-path, --output-dir, --target-soc, --mode,
    │                    [--quant-path | --quant-method gptq]
    ▼
convert(hf_path, output_dir, dtype, ...)
    │
    ▼
loader.load_model(hf_path, dtype, device)                          ← export_design.md
    │   ① Read config.json
    │   ② model_registry.detect_model_type → ModelEntry
    │   ③ build_config (field filtering + __post_init__)
    │   ④ entry.model_cls(config).to(dtype).eval()
    │   ⑤ entry.load_weights_fn (safetensors → key remapping → param.copy_)
    │   ⑥ collect_tokenizer_files
    │   → ModelArtifact
    ▼
(Optional) Quantization swap                                       ← quant_design.md
    │   swap_from_dopt(model, quant_path)
    │   swap_embeddings_from_dopt(sd, model)   # embed auto-detection (when embed_quant is non-empty)
    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)             ← export_design.md
    │   ① register_ge_converters
    │   ② wrapper = artifact.wrapper_cls(model, ...).eval()
    │   ③ wrapper.get_sample_inputs(use_embeddings)
    │   ④ (use_embeddings) _extract_host_side_weights + detach_externalized_lookups
    │   ⑤ torch.export.export(wrapper, sample, strict=False)
    │   ⑥ torch_delegate.export_to_epair(ep) → epair
    │   ⑦ del ep, wrapper, model; gc.collect()       # OOM protection
    │   ⑧ consolidated_weights_save (weights_*.bin + host_weights_*.bin + ep_json)
    │   ⑨ copy_tokenizer_files + save_model_config
    ▼
C++ EpairModelLoader → GetGEGraph → aclgrphBuildModel → .om        ← export_design.md "C++ Loading Side"
```

> For full CLI parameters and call chain details, see [cli_design.md](./cli/cli_design.md); for implementation details of each loading/saving/exporting step, see [export_design.md](./export/export_design.md).

<a id="model-differences-overview"></a>
## Model Differences Overview

The two models have inconsistent interfaces, and the converter smooths out the differences via a registry + adapter. For detailed handling of each difference, see [export_design.md](./export/export_design.md).

### Config Loading Differences

| Model | from_pretrained | from_json_file | to_dict | config_json_key |
|------|:-:|:-:|:-:|---|
| gemma4 | No | No | Yes | `text_config` |
| qwen3_5 | Yes | Yes | No | `text_config` |

> `build_config` uniformly goes through the field-filtering path (taking the sub-dict by `config_json_key` + filtering by dataclass field names + `__post_init__`), and no longer calls `from_pretrained`/`from_json_file`. See [export_design.md](./export/export_design.md) "build_config".

### forward Signature Differences

| Parameter | gemma4 | qwen3_5 |
|------|:------:|:-------:|
| input_ids | Yes | Yes |
| attention_mask | Yes | Yes |
| position_ids | No (internal computation) | No (internal computation) |
| past_key_values | `list` | `SimpleKVCache` |
| inputs_embeds | Yes | Yes |
| labels | Yes | Yes |
| output_hidden_states | Yes | No |
| use_cache | No | Yes |
| shared_kv_states | Yes (unique) | No |
| delta_cache | No | Yes (unique) |

### forward Return Differences

| Return key | gemma4 | qwen3_5 |
| ------------------| ------------| ------------|
| loss | Yes | Yes |
| logits | Yes | Yes |
| past_key_values | Yes | Yes |
| shared_kv_states | Yes (unique) | No |
| hidden_states | Yes | No |
| delta_cache | No | Yes (unique) |

### Weight Key Remapping Differences

| Model | Remapping strategy | Description |
|------|-----------|------|
| gemma4 | `model.language_model.*` → `model.*` | Remove VLM prefix |
| qwen3_5 | `model.language_model.*` → `model.*` | Remove VLM prefix via `_remap_key` |

> For differences in the `ExportWrapper` subclasses of both models (cache type / forward kwargs / PLE handling), see [export_design.md](./export/export_design.md) "Differences between the two subclasses".

<a id="accuracy-verification-summary"></a>
## Accuracy Verification Summary

For all models, the generated token sequences are identical to HF, with accuracy differences within the normal range for float16/bfloat16 (qwen3.5-0.8B max_diff=0.019 / gemma4-E2B max_diff=0.17).

> For error accumulation analysis (ClippableLinear / 175 RMSNorm / PLE accumulation / KV sharing without decay), the bfloat16 attention accuracy strategy under scatter-write, and tie-breaking issues, see [export_design.md](./export/export_design.md) "Accuracy Difference Analysis".

<a id="module-navigation"></a>
## Module Navigation

### Design Documents (Three Modules)

| Document | Scope | Key Content |
|------|------|---------|
| [cli_design.md](./cli/cli_design.md) | `cann_lm_converter.py` | `convert()` programmatic entry / command-line parameters / complete call chain / prefill-decode CLI / environment variables / usage examples |
| [export_design.md](./export/export_design.md) | `model_registry` / `loader` / `saver` / `export_wrapper` / `export/` (incl. C++ `csrc/`) | `ModelArtifact` / `build_config` / `ExportWrapper` base class + two subclasses / unified graph I/O contract / pure embeddings mode / `pack_weights` / `consolidated_weights_save` / `EpairModelLoader` / `GeWeightLoader` lazy loading / `_assert_tensor_metadata` / two-layer GE op registration / prefill-decode separation / new model onboarding / accuracy analysis |
| [quant_design.md](./quant/quant_design.md) | `cann_local_lm/converter/quant/` | W4A8 s8s4 scheme / dopt GPTQ framework / `materialize_int4_per_group_symmetric` / `QuantLinear` / `swap_from_dopt` / `npu::npu_quant_matmul` op / C++ registration / CANN V3/V4 source analysis appendix |
| `embedding_quant_design.md` | embed_tokens quantization topic | W8 host gather / `materialize_int8_per_channel_symmetric` / `swap_embeddings_from_dopt` |

### Module Indices

| Index | Module |
|------|------|
| [INDEX.md](./cli/INDEX.md) | CLI module documents + code index |
| [INDEX.md](./export/INDEX.md) | Export module documents + code index |
| [INDEX.md](./quant/INDEX.md) | Quantization module documents + code index |

### Reference Documents

| Document | Topic |
|------|------|
| [reference_litert_lm_export.md](./reference_litert_lm_export.md) | LiteRT-LM + litert-torch external reference (prefill/decode two signatures / KV cache pytree / wrapper unified I/O) |
| [reference_npu_op_elimination.md](./reference_npu_op_elimination.md) | NPU op elimination skill (host pre-constructed mask / repeat_kv concat / RoPE pre-cached Gather) |
| [feature_dump_design.md](./feature_dump_design.md) | Dump debug capability (per-op intermediate tensor dump at FX node level / `gen_golden_with_dump.py` tool / capability comparison vs torch hook / artifact layout / application scenarios) |

<a id="pending-and-legacy-items"></a>
## Pending and Legacy Items

Grouped by module; see the legacy section of each module's document for details.

- **Export module** — Unified cache signature (full version) / generation_config.json handling / graph export I/O interface / qwen3.5 linear attention / qwen3.5 scatter-write refactoring. See [export_design.md](./export/export_design.md) "Pending and Legacy Items"
- **Quantization module** — dopt functions out of spec need splitting. See [quant_design.md](./quant/quant_design.md) "Known Issues and Legacy"

### Resolved

See the "Resolved" entries in each module's document: [export_design.md](./export/export_design.md) / [quant_design.md](./quant/quant_design.md).

<a id="quick-start"></a>
## Quick Start

See [Quick Start](../QUICKSTART.md) (installation / exporting epair / quantized export / omc compilation / deployment to HarmonyOS devices) and [Build Guide](../build_guide.md) (build prerequisites / one-click build / FAQ).

> For CLI parameter quick reference, see [cli/INDEX.md](./cli/INDEX.md) "Command-Line Parameter Quick Reference"; for full design, see [cli_design.md](./cli/cli_design.md).
