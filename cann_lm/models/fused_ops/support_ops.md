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

| fuse_op                 | npu:: 注册　　　　　　　　　　　　　　　　　　　 | schema 来源　 | Meta 定义　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE raw op　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| --------------------------------------------------| ---------------| -----------------------------------------------------------------------------| ---------------------------------------------------------------------|
| ChunkGatedDeltaRule     | ❌ 未注册（需 torch_npu ≥2.7.1.post8）　　　　　　| 无　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| Rmsnorm                 | `npu_rms_norm`　　　　　　　　　　　　　　　　　 | C++ `_C` 扩展 | `op_plugin/meta/_meta_registrations.py`（`npu_rms_norm_meta`）　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(RmsNorm)`　　　　　　　　　　|
| FusedInferAttention     | `npu_fused_infer_attention_score` + `_v2`　　　　| C++ `_C` 扩展 | `_meta_registrations.py`（`npu_fused_infer_attention_score`/`_v2_forward`） | —　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| l2norm                  | ❌ 非 torch_npu 自定义算子　　　　　　　　　　　　| 无　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(L2Normalize)`（仅 GE IR 级） |
| RotaryPositionEmbedding | `npu_rotary_mul`（+ `npu_apply_rotary_pos_emb`） | C++ `_C` 扩展 | `_meta_registrations.py`（`npu_rotary_mul_meta`）　　　　　　　　　　　　　 | `auto_generated_ge_raw_ops.py` `REG_OP(RotaryMul)`　　　　　　　　　|
| ScatterNdUpdate         | `npu_scatter_nd_update` + `_`　　　　　　　　　　| C++ `_C` 扩展 | `_meta_registrations.py`（`scatter_nd_update_meta`）　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(ScatterNdUpdate)`　　　　　　|

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

### 4. torchair ge converter 产出 GE 算子映射

> FX→GE 转换由 torchair `python/torchair/_ge_concrete_graph/ge_converter/custom/` 下的 converter 完成（注册到 `npu::` 命名空间，命中 `torch.ops.npu.*`）。下表列出各算子 converter 的注册 schema、converter 函数及产出 GE 算子；converter 文件路径下文简写为 `custom/<file>`。
> 命中规则：FX 图里 `torch.ops.npu.*` → 走 `custom/` 版本（函数名无后缀）；`torch.ops.air.*` → 走 `torchair/ops/` 版本（函数名带 `_tensor` 后缀，自带 schema+Meta）。

#### fuse_op

| fuse_op | converter 文件 | 注册 schema (`@register_fx_node_ge_converter`) | converter 函数 | 产出 GE 算子 | 备注 |
|---|---|---|---|---|---|
| ChunkGatedDeltaRule | `custom/npu_chunk_gated_delta_rule.py` | `npu.npu_chunk_gated_delta_rule.default` | `convert_npu_chunk_gated_delta_rule(q,k,v,*,beta?,initial_state?,actual_seq_lengths?,scale?,g?)` | ❌ 空壳 `raise RuntimeError("...not supported in GE graph mode")` | GE 图模式不支持；schema 需 torch_npu ≥2.7.1.post8 |
| Rmsnorm | `custom/npu_rms_norm.py` | `npu.npu_rms_norm.default` | `conveter_npu_rms_norm_default(self,gamma,epsilon=1e-6)` | `ge.RmsNorm(self,gamma,epsilon=)`（1 个） | `@declare_supported`（F16/F32） |
| FusedInferAttention | `custom/fused_infer_attention_score_v2.py`（npu::） | `npu.npu_fused_infer_attention_score_v2.default` | `convert_npu_npu_fused_infer_attention_score_v2(q,k,v,*,query_rope?,key_rope?,pse_shift?,atten_mask?,actual_seq_qlen?,actual_seq_kvlen?,block_table?,dequant_scale_*?,quant_scale_*?,learnable_sink?,input_layout,sparse_mode,block_size,...)` | `ge.FusedInferAttentionScore(...)` → `(out,lse)` | air:: 版在 `torchair/ops/_npu_fused_infer_attention_score_v2.py`（自带 schema+Meta） |
| l2norm | — | — | — | GE raw op `L2Normalize`（仅 IR 级 REG_OP） | torchair 无 converter/schema/Meta |
| RotaryPositionEmbedding | `custom/npu_rotary_mul.py` | `npu.npu_rotary_mul.default` | `conveter_npu_rotary_mul_default(self,r1,r2,rotary_mode='half',rotate?)` | 默认 `ge.RotaryMul(self,r1,r2)`；arch35 或带 `rotate` 时 `ge.RotaryPositionEmbedding(self,r1,r2,mode=,rotate=)` | gemma4 走默认 `ge.RotaryMul` 分支 |
| ScatterNdUpdate | `custom/npu_scatter_nd_update.py` | `npu.npu_scatter_nd_update.default` + `npu.npu_scatter_nd_update_.default` | out-of-place / in-place 两版 | out-of-place: `ge.TensorMove(self)`+`ge.ScatterNdUpdate(copy,idx,upd)`（2 算子）；in-place: `ge.ScatterNdUpdate(self,idx,upd)`+`dont_prune_me`（1 算子） | ⚠️ cann_lm 已自带 `converter/export/_scatter_nd_update_converter.py` 覆盖此 converter（产出相同 `ge.ScatterNdUpdate`，行为可控）；gemma4 已落地（`scatter_kv_to_cache` 构造多维 indices 适配标准语义） |

#### quant_op

| quant_op | converter 文件 | 注册 schema | 产出 GE 算子 | 备注 |
|---|---|---|---|---|
| DynamicQuant | `custom/npu_dynamic_quant.py` | `npu.npu_dynamic_quant.default` + `npu.npu_dynamic_quant_asymmetric.default` | `quant_mode=="pertoken" && dst_type_max≈0`: `ge.DynamicQuant(...)`→`(y,scale)`；其他: `ge.DynamicQuantV2(...)`→`(y,scale,offset)`；int4(dst_type==16) 额外 Const/Shape/Div/ConcatV2/Bitcast/Reshape bitcast 子图 | `@declare_supported`（F16/BF16） |
| QuantBatchMatmulV4 | `custom/quantmatmul.py` | `npu.npu_quant_matmul.default` | 主: `ge.QuantBatchMatmulV4(...)`；备: `ge.QuantBatchMatmulV3(...)`；含 int4/fp4 bitcast+Transpose+Reshape 子图 | ⚠️ cann_lm 实际走**自带 converter** `cann_lm/converter/export/_quant_matmul_converter.py`（去除 torch_npu 依赖、去掉 a8w4 分支），不命中 torchair 此 converter |

#### 关键说明

- **converter 函数名笔误**：torchair 多数 converter 函数名拼写为 `conveter_*`（少一个 r，如 `conveter_npu_rms_norm_default`），系 torchair 历史笔误，非本仓问题，引用时照原拼写。
- **GE 算子名 vs OM 算子名**：converter 产出的 GE 算子（如 `ge.ScatterNdUpdate`/`ge.RmsNorm`）是 IR 阶段名；OM 编译期经融合优化后算子名会变化（`ScatterNdUpdate`/`InplaceUpdate+TensorMove` → OM `ScatterUpdate`+`MemcpyAddrAsync`；`RmsNorm`+残差 Add → OM `InplaceAddRmsNorm`）。详见 AGENTS.md「查看 OM 模型算子」。
- **converter venv 命中情况**：converter venv 不装 torch_npu，`ge_converter/custom/` 里 `import torch_npu` 会失败；但本仓 `cann_lm::`/`npu::` 算子 schema+Meta 由 `_library.py` 自提供，FX 图只命中 `air::` 路径（`_LazyTorchair` 代理不覆盖源码 torchair），不触发 `npu::` converter，故不冲突。
