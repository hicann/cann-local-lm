### fuse_op
ChunkGatedDeltaRule
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md
Rmsnorm
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md
FusedInferAttention
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md
l2norm
RotaryPositionEmbedding
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md
ScatterNdUpdate
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_scatter_nd_update.md

### quant_op
DynamicQuant
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md
QuantBatchMatmulV4
https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md



### single_op
Cast
Mul
splitV
concatD
Transpose
Conv2DV2
StridedSliceV2
Add
Swish
GatherV2
TransData
Sigmoid
MatmulV3
SoftplusV2
Gelu

---

## 融合算子定义定位分析

> 本节分析范围：原 6 个融合算子（ChunkGatedDeltaRule / Rmsnorm / FusedInferAttention / l2norm / RotaryPositionEmbedding / ScatterNdUpdate）。
> DynamicQuant / QuantBatchMatmulV4 用户新归入 fuse_op，对应 `npu_dynamic_quant` / `quant_matmul`，未在本次验证范围内。
> 验证版本：torch_npu 2.7.1.post4（系统 Python `/opt/buildtools/Python-3.11.4/bin/python3`）；converter venv（`/mnt/workspace/torch-delegate-harmony/.venv`，torch_npu absent，torchair 源码编译）。
> 行号为验证时版本快照，升级后可能漂移，以文件路径为准。

### 1. torch_npu 2.7.1.post4（系统 Python）下的定义

共注册 354 个 `npu::` 算子。6 个融合算子命中情况：

| fuse_op | npu:: 注册 | schema 来源 | Meta 定义 | GE raw op |
|---|---|---|---|---|
| ChunkGatedDeltaRule | ❌ 未注册（需 torch_npu ≥2.7.1.post8） | 无 | 无 | 无 |
| Rmsnorm | `npu_rms_norm` | C++ `_C` 扩展 | `op_plugin/meta/_meta_registrations.py`（`npu_rms_norm_meta`） | `auto_generated_ge_raw_ops.py` `REG_OP(RmsNorm)` |
| FusedInferAttention | `npu_fused_infer_attention_score` + `_v2` | C++ `_C` 扩展 | `_meta_registrations.py`（`npu_fused_infer_attention_score`/`_v2_forward`） | — |
| l2norm | ❌ 非 torch_npu 自定义算子 | 无 | 无 | `auto_generated_ge_raw_ops.py` `REG_OP(L2Normalize)`（仅 GE IR 级） |
| RotaryPositionEmbedding | `npu_rotary_mul`（+ `npu_apply_rotary_pos_emb`） | C++ `_C` 扩展 | `_meta_registrations.py`（`npu_rotary_mul_meta`） | `auto_generated_ge_raw_ops.py` `REG_OP(RotaryMul)` |
| ScatterNdUpdate | `npu_scatter_nd_update` + `_` | C++ `_C` 扩展 | `_meta_registrations.py`（`scatter_nd_update_meta`） | `auto_generated_ge_raw_ops.py` `REG_OP(ScatterNdUpdate)` |

注册映射见 `torch_npu/utils/custom_ops.py`（`npu_rotary_mul`/`npu_rms_norm`/`npu_fused_infer_attention_score[_v2]`/`npu_scatter_nd_update`）。

### 2. converter venv 下 torchair 的定义

torch_npu 包 absent；torchair present（`/mnt/workspace/torch-delegate-harmony/.venv/lib/python3.11/site-packages/torchair`）。

| fuse_op                 | venv torchair 定义文件　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 类型　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否自包含（不依赖 torch_npu）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------|
| ChunkGatedDeltaRule     | `_ge_concrete_graph/ge_converter/custom/npu_chunk_gated_delta_rule.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_chunk_gated_delta_rule.default`，且函数体直接 `raise RuntimeError("not supported in GE graph mode")`（空壳） |
| Rmsnorm                 | `_ge_concrete_graph/ge_converter/custom/npu_rms_norm.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_rms_norm`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| FusedInferAttention     | `ops/_npu_fused_infer_attention_score.py` + `_v2.py`（air::）**且** `ge_converter/custom/fused_infer_attention_score.py` + `_v2.py`（npu::） | air:: 自带 `lib.define` schema+Meta（`torch.library.Library("air","FRAGMENT")`）；npu:: 仅为 converter | ⚠️ schema+Meta 自包含（可 trace），但文件顶部仍 `import torch_npu`（converter 部分需要）　　　　　　　　　　　　　　　　　　　　　　　　|
| l2norm                  | 无（仅 `auto_generated_ge_raw_ops.py` 有 GE raw op `L2Normalize`）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE IR 级 REG_OP　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 无 schema/Meta/converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| RotaryPositionEmbedding | `_ge_concrete_graph/ge_converter/custom/npu_rotary_mul.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_rotary_mul`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ScatterNdUpdate         | `_ge_concrete_graph/ge_converter/custom/npu_scatter_nd_update.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_scatter_nd_update`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

### 3. 关键结论

- venv torchair 里 6 个融合算子**只有 FX→GE converter**（`ge_converter/custom/` 下，依赖 `torch.ops.npu.*` schema），而 torch_npu 在 venv 中 absent → 这些 converter **无法注册**（`register_fx_node_ge_converter` 拿不到 schema）。
- **唯一自包含 schema+Meta 的是 FusedInferAttention 的 air:: 版本**（`ops/_npu_fused_infer_attention_score_v2.py` 用 `torch.library.Library("air","FRAGMENT")` + `lib.define`），但其 converter 部分仍 `import torch_npu`。
- **ChunkGatedDeltaRule** 的 converter 是空壳（`raise RuntimeError`），且 schema 需 torch_npu ≥2.7.1.post8（venv 里两个都缺）。
- **l2norm** 在 torchair 完全没有定义，只有 GE raw op `L2Normalize`。
- **根因**：CANN-LM `cann_lm/models/fused_ops/_library.py` 必须在 converter venv 里为这 6 个算子自提供 `npu::`/`cann_lm::` 的 schema+fake-meta，使 `torch.export` 能 trace。详见 [cli_design.md](../../../docs/cann_lm_converter/cli/cli_design.md)「converter 与 torch_npu 的关系」。
