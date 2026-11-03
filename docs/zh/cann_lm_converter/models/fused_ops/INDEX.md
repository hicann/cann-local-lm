# 融合算子模块索引

简体中文 | [English](../../../../en/cann_lm_converter/models/fused_ops/INDEX.md)

> `cann_local_lm/models/fused_ops/` — torch.library 算子 schema / fake-meta / CPU eager 实现 + GE 映射。
> 本目录是 fused_ops 模块的**文档侧**（设计/参考/验证），代码在 `cann_local_lm/models/fused_ops/`。

## 模块文档

| 文档 | 内容 |
|------|------|
| [fused_ops.md](./fused_ops.md) | 合并自原 `support_ops.md` + `compositte_op.md`：算子总表（fuse/quant/single）→ 定义溯源（torch_npu vs torchair）→ torchair GE converter 映射 → Gemma-4 / Qwen3.5 逐模型（参考实现 survey + modeling 替换 + 收益排序 + OM 算子表对比）→ 平台差异 |

## 代码文件（`cann_local_lm/models/fused_ops/`）

| 文件 | 职责 |
|------|------|
| `_library.py` | `NPU_LIB`（`cann_local_lm::` FRAGMENT）+ `CANN_LM_LIB`（`cann_local_lm::` DEF）+ `safe_define` + `register_cpu_impl`（stub env 跳过）|
| `ops.py` | 聚合 import（注册全部 fused ops 的 schema + fake + CPU impl）|
| `chunk_gated_delta_rule.py` | `cann_local_lm::chunk_gated_delta_rule`（Qwen3.5 线性注意力核心，解决 export segfault）|
| `fused_infer_attention_score.py` | `cann_local_lm::fused_infer_attention_score_v2`（FlashAttention 融合）|
| `rms_norm.py` | `cann_local_lm::rms_norm` |
| `rotary_mul.py` | `cann_local_lm::rotary_mul` |
| `scatter_nd_update.py` | `cann_local_lm::scatter_nd_update` + `_`（KV cache 写入）|
| `l2_norm.py` | `cann_local_lm::l2norm` |
| `dynamic_quant.py` | `cann_local_lm::dynamic_quant` |
| `quant_matmul.py` | `cann_local_lm::quant_matmul` |

## GE converter（`cann_local_lm/converter/export/`）

| 文件 | 职责 |
|------|------|
| `ge_converter.py` | `register_ge_converters`：cann_local_lm 自带 converter（quant_matmul / softplus / l2norm / chunk_gated_delta_rule / conv1d / index_copy / scatter_nd_update / index_put / FIA_v2）+ import torchair `custom/`（自动注册 npu_rms_norm / npu_rotary_mul 等）|
| `_chunk_gated_delta_rule_converter.py` | `cann_local_lm::chunk_gated_delta_rule` → `ge.ChunkGatedDeltaRule`（torchair 空壳，cann_local_lm 覆盖）|
| `_fused_infer_attention_score_v2_converter.py` | `cann_local_lm::fused_infer_attention_score_v2` → `ge.FusedInferAttentionScore` |
| `_scatter_nd_update_converter.py` | `cann_local_lm::scatter_nd_update(_)` → `ge.ScatterNdUpdate` |
| `_index_copy_converter.py` | `aten::index_copy` → `ge.InplaceUpdate + TransposeD` |
| `_conv1d_converter.py` | `aten::conv1d` → GE conv1d |
| `_quant_matmul_converter.py` | `cann_local_lm::quant_matmul` → `ge.QuantBatchMatmulV4`（详见 [quant/INDEX.md](../../quant/INDEX.md)）|

## 关联模块

- **converter 导出**：[export_design.md](../../export/export_design.md) — `ExportWrapper` / `consolidated_weights_save` / GE 算子注册两层
- **CLI**：[cli_design.md](../../cli/cli_design.md) —「converter 与 torch_npu 的关系」（`_library.py` stub 路径根因）
- **量化**：[quant_design.md](../../quant/quant_design.md) — `npu_quant_matmul`
- **模型设计**：[long_term_design.md](../long_term_design.md) / [short_term_design.md](../short_term_design.md) — 模型统一方案
