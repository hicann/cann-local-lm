# Model Export Module Index

English | [简体中文](../../../zh/cann_lm_converter/export/INDEX.md)

> `model_registry` / `loader` / `saver` / `export_wrapper` / `export/` (incl. C++ `csrc/`) — exports HF models to the epair format.

## Module Documents

| Document | Content |
|------|------|
| [export_design.md](./export_design.md) | `model_registry` / `loader` / `saver` / `ExportWrapper` base class + two subclasses / unified graph I/O contract / pure embeddings mode / `epair_utils` / C++ `EpairModelLoader` / `GeWeightLoader` lazy loading / `_assert_tensor_metadata` / two-layer GE op registration / prefill-decode separation / new model onboarding / accuracy difference analysis |
| [weight_format_design.md](./weight_format_design.md) | Weight layout conversion (Stage 1 fp16 MatMul→NZ, Stage 2 quantized QuantBatchMatmulV4 weight_q→FractZ, both implemented and verified): `weight_format/` package (`nd2nz` / `pack_int4_nz` / `resolve_format`) + `weight_format_rules.json` config (op_type + format + rule-level enable) + `epair_utils` integration (incl. `_transform_weight_tensor` and `_sync_*` annotation sync) + C++ `MapGeLayoutString` / wildcard guard / **CANN_LM_ENABLE_WEIGHT_TRANS deprecation migration design (finalized 2026-10-20, pending implementation; rule-level enable: MatMul off / QuantBatchMatmulV4 on)** |

## Code Files

### Python

| File | Responsibility |
|------|------|
| `cann_local_lm/models/model_registry.py` | `ModelEntry` + `REGISTRY` + `detect_model_type` + `available_models` |
| `cann_local_lm/converter/loader.py` | `load_model` + `build_config` + `collect_tokenizer_files` |
| `cann_local_lm/converter/saver.py` | `save_epair` / `save_epair_prefill_decode` / `save_model_config` / `copy_tokenizer_files` |
| `cann_local_lm/models/export_wrapper.py` | `ExportWrapper` base class (`forward` + `get_sample_inputs` + `_drive_layers`) |
| `cann_local_lm/models/{gemma4,qwen3_5}/export_wrapper.py` | Per-model `ExportWrapper` subclasses |
| `cann_local_lm/converter/export/epair_utils.py` | `pack_weights` + `consolidated_weights_save` |
| `cann_local_lm/converter/export/weight_format/` | Weight format conversion package (implemented): `nd2nz` (ND→FractNZ, Stage 1) + `pack_int4_nz` (int4pack→FractZ, Stage 2) + `rules.resolve_format` (config parsing, dispatch by op_type) + `load_rules` (see [weight_format_design.md](./weight_format_design.md)) |
| `cann_local_lm/converter/export/weight_format_rules.json` | shipped default config (`name_pattern` + `op_type` + `format` + `pack` + post-migration rule-level `enable`): MatMul `*_proj.weight`→NZ (post-migration `enable:false` no conversion), QuantBatchMatmulV4 `*.weight_q`→Z (`enable:true` mandatory); missing/empty rules → all ND (escape hatch). Post-migration it is the sole authority for format control (replaces env master switch, see design doc "Master Switch Deprecation Design") |
| `cann_local_lm/converter/export/quant_swap/` | Quantization materialization internal package (migrated from `quant/` on 2026-08-31): `materialize.py` (int4 materialization + `transdata_scale`/`pack_int4`) / `quant_linear.py` (QuantLinear); `__init__` re-exports 6 materialization symbols. Orchestration layer `swap.py` promoted to `cann_local_lm/converter/swap.py` (converter root). See [quant_design.md](../quant/quant_design.md) |
| `cann_local_lm/converter/export/ge_converters/ge_converter.py` | `register_ge_converters` (cann_local_lm.* + aten.* → GE op mapping) |
| `cann_local_lm/converter/export/ge_converters/_quant_matmul_converter.py` | `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4` GE converter (see [INDEX.md](../quant/INDEX.md)) |

### C++ (`cann_local_lm/converter/export/csrc/`)

| File | Responsibility |
|------|------|
| `inc/external/torch_delegate/epair_model_loader.h` | C++ public API: `EpairModelLoader` |
| `backend/epair_model_loader.cc` | `EpairModelLoader::Impl` implementation (initialization, Graph construction) |
| `backend/ge_graph_builder.h/.cc` | `GeGraphBuilder`: parses model JSON + builds GE Graph |
| `backend/ge_weight_loader.h/.cc` | `GeWeightLoader`: parses `weight_info.json`, lazy-loads bin |
| `backend/epair_model_parser.h/.cc` | `EpairModelParser`: parses `ep_json_with_ascend_ir.json` |

### Examples (`examples/test_converter/export_llm_model/`)

| File | Responsibility |
|------|------|
| `main.cpp` | Example main for C++ loading epair + compiling OM |
| `reg_register.h` | GE graph compile-time op registration (`REG_OP`) |
| `om_npu_verify/` | OM compile + NPU inference verification toolkit |

## Key Functions / Classes

| Symbol | Purpose |
|------|------|
| `load_model(hf_path, *, dtype, device) -> ModelArtifact` | Load HF model (auto-detect type + config + weights + tokenizer) |
| `build_config(hf_config, entry)` | Unified field-filtering to construct config dataclass |
| `save_epair(artifact, output_dir, ...)` | Save epair (single mode) |
| `save_epair_prefill_decode(artifact, output_dir, ...)` | Save prefill + decode two graphs (shared weights) |
| `ExportWrapper` | Unified I/O contract base class (KV cache in/out + output_pos selection) |
| `pack_weights(contents, *, max_file_size) -> ConsolidatedWeights` | Pack weights into multiple bins |
| `consolidated_weights_save(...)` | Write to disk (weights_*.bin + weight_info.json + ep_json) |
| `nd2nz(tensor, block=16, pad=True)` | `[M,K]` ND row-major → FractNZ physical block `[ceil(M/b),ceil(K/b),b,b]` (implemented+verified, see [weight_format_design.md](./weight_format_design.md)) |
| `pack_int4_nz(tensor)` | int4pack int32 `[K, N//8]` → FractZ `[K//64, N//16, 16, 8]` (implemented, KirinX90 board cos=0.999912) |
| `resolve_format(name, is_quantized)` | Match (name_pattern, op_type) against `weight_format_rules.json`, returns format/block/pad/pack rule (post-migration includes rule-level `enable:false` skip semantics) |
| `_transform_weight_tensor(raw, name, ...)` | Packer weight pre-write transform (dtype view + transdata_scale + layout conversion), shared by v2/PD |
| `_sync_input_desc_layout` / `_sync_weight_scale_dtype` (`epair_utils`) | ep_json annotation sync (Z layout 4D shape / scale dtype uint64), must be co-located with packing (see design doc "Annotation Sync Mechanism") |
| ~~`inline_weight_layout(ep_json_path, names_nz)`~~ | **Deprecated**: pt2 layout enum has no NZ expression; format signal now goes through `weight_info.json`'s `format` field (design doc "ep_json layout annotation") |
| `td::EpairModelLoader` | C++ load epair → build GE Graph |
| `GeWeightLoader` | C++ lazy-load weight bin |
| `register_ge_converters()` | Register cann_local_lm.* + aten.* → GE op mapping |

## Artifact Layout

```
output_dir/
├── ep_json_with_ascend_ir.json     # (single) or prefill_*.json + decode_*.json (prefill-decode)
├── weight_info.json                # weight index
├── weights_0.bin                   # weight data
├── model_config.json               # config + hf_config + use_embeddings flag
├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
└── host_weights_*.bin + host_weights_info.json   # (only use_embeddings=True) host lookup weights
```

> For full description, see [export_design.md](./export_design.md) "Output File Layout".

## Related Modules

- **CLI**: [INDEX.md](../cli/INDEX.md) — `cann_lm_converter.py` entry / `convert()` orchestration
- **Quantization**: [INDEX.md](../quant/INDEX.md) — `QuantLinear` replaces `nn.Linear` / `npu_quant_matmul` op
- **Reference**: [reference_litert_lm_export.md](../reference_litert_lm_export.md)
