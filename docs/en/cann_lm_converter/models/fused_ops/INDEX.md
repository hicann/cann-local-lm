# Fused Operators Module Index

English | [简体中文](../../../../zh/cann_lm_converter/models/fused_ops/INDEX.md)

> `cann_local_lm/models/fused_ops/` — torch.library operator schema / fake-meta / CPU eager implementation + GE mapping.
> This directory is the **documentation side** (design / reference / verification) of the fused_ops module; the code lives in `cann_local_lm/models/fused_ops/`.

## Module Documentation

| Document | Content |
|------|------|
| [fused_ops.md](./fused_ops.md) | Merged from the original `support_ops.md` + `compositte_op.md`: operator overview table (fuse/quant/single) → definition tracing (torch_npu vs torchair) → torchair GE converter mapping → per-model analysis for Gemma-4 / Qwen3.5 (reference implementation survey + modeling replacement + benefit ranking + OM operator table comparison) → platform differences |

## Code Files (`cann_local_lm/models/fused_ops/`)

| File | Responsibility |
|------|------|
| `_library.py` | `NPU_LIB` (`cann_local_lm::` FRAGMENT) + `CANN_LM_LIB` (`cann_local_lm::` DEF) + `safe_define` + `register_cpu_impl` (skipped in stub env)|
| `ops.py` | Aggregated import (registers schema + fake + CPU impl for all fused ops)|
| `chunk_gated_delta_rule.py` | `cann_local_lm::chunk_gated_delta_rule` (Qwen3.5 linear attention core, resolves export segfault)|
| `fused_infer_attention_score.py` | `cann_local_lm::fused_infer_attention_score_v2` (FlashAttention fusion)|
| `rms_norm.py` | `cann_local_lm::rms_norm` |
| `rotary_mul.py` | `cann_local_lm::rotary_mul` |
| `scatter_nd_update.py` | `cann_local_lm::scatter_nd_update` + `_` (KV cache write)|
| `l2_norm.py` | `cann_local_lm::l2norm` |
| `dynamic_quant.py` | `cann_local_lm::dynamic_quant` |
| `quant_matmul.py` | `cann_local_lm::quant_matmul` |

## GE Converter (`cann_local_lm/converter/export/`)

| File | Responsibility |
|------|------|
| `ge_converter.py` | `register_ge_converters`: cann_local_lm built-in converters (quant_matmul / softplus / l2norm / chunk_gated_delta_rule / conv1d / index_copy / scatter_nd_update / index_put / FIA_v2) + import torchair `custom/` (auto-registers npu_rms_norm / npu_rotary_mul, etc.)|
| `_chunk_gated_delta_rule_converter.py` | `cann_local_lm::chunk_gated_delta_rule` → `ge.ChunkGatedDeltaRule` (torchair stub, cann_local_lm override)|
| `_fused_infer_attention_score_v2_converter.py` | `cann_local_lm::fused_infer_attention_score_v2` → `ge.FusedInferAttentionScore` |
| `_scatter_nd_update_converter.py` | `cann_local_lm::scatter_nd_update(_)` → `ge.ScatterNdUpdate` |
| `_index_copy_converter.py` | `aten::index_copy` → `ge.InplaceUpdate + TransposeD` |
| `_conv1d_converter.py` | `aten::conv1d` → GE conv1d |
| `_quant_matmul_converter.py` | `cann_local_lm::quant_matmul` → `ge.QuantBatchMatmulV4` (see [quant/INDEX.md](../../quant/INDEX.md))|

## Related Modules

- **Converter export**: [export_design.md](../../export/export_design.md) — `ExportWrapper` / `consolidated_weights_save` / two-layer GE operator registration
- **CLI**: [cli_design.md](../../cli/cli_design.md) — "Relationship between converter and torch_npu" (root cause of the `_library.py` stub path)
- **Quantization**: [quant_design.md](../../quant/quant_design.md) — `npu_quant_matmul`
- **Model design**: [long_term_design.md](../long_term_design.md) / [short_term_design.md](../short_term_design.md) — unified model scheme
