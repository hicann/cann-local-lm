# 融合算子参考（support_ops + compositte_op 合并）

> **合并自** 原 `cann_lm/models/fused_ops/support_ops.md`（算子注册表 / 定义溯源 / GE converter 映射 / OM 验证）与 `compositte_op.md`（参考实现 survey / modeling 替换方案 / 优先级）。两文档围绕同一组融合算子，前者偏下游（IR/OM 验证），后者偏上游（参考实现 + modeling 替换），合并后按「算子总表 → 定义溯源 → GE 映射 → 逐模型（参考→modeling→优先级→OM 验证）→ 平台差异」组织。

> **数据来源**：
> - 参考实现（CANN 优化版）：Gemma-4 `cann-recipes-infer/models/gemma_4/models/modeling_gemma4.py`；Qwen3.5 `cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py`
> - 本仓实现：`cann_lm/models/gemma4/modeling_gemma4.py` / `cann_lm/models/qwen3_5/modeling_qwen3_5.py`
> - 自定义算子：`cann_lm/models/fused_ops/`（`_library.py` 用 `Library("cann_lm","FRAGMENT")` 注册 `cann_lm::` schema+fake+CPU impl）+ GE 映射 `cann_lm/converter/export/ge_converters/ge_converter.py`（注册 `cann_lm::` converter，覆盖 torchair `custom/` 的 npu:: converter）
> - 验证环境：torch_npu 2.9.0.post8（系统 Python，NPU 实机）；converter venv（torchair 源码编译，torch_npu absent，走 `_library.py` stub 路径）。

参考实现基于 CANN 推理框架（TND/BNSD Paged Attention + MC2 EP + GE Graph），大量使用 `torch_npu` / `torch.ops.npu` 融合算子。**本仓已迁移到 `cann_lm::` 命名空间**（避免与 torch_npu 重复注册冲突，见 `_rotary_mul_converter.py:18` 注释「avoid torch_npu duplicate-registration」）：`modeling_*.py` 调用 `torch.ops.cann_lm.*` 融合算子（含 l2norm），`cann_lm/models/fused_ops/_library.py` 用 `Library("cann_lm","FRAGMENT")` 始终注册 `cann_lm::` schema+fake+CPU impl（`_HAS_TORCH_NPU=False` 硬编码，不依赖 torch_npu），FX→GE 转换由 `cann_lm/converter/export/ge_converters/` 自带 converter 完成（注册 `cann_lm::`，覆盖 torchair `custom/` 的 npu:: converter，见 §3）。NPU 加速走 export→OM（GE 转换），eager 走 CPU impl。

## 目录

1. [算子总表（fuse_op / quant_op / single_op）](#1-算子总表fuse_op--quant_op--single_op)
2. [定义溯源：torch_npu 2.9.0.post8 vs converter venv torchair](#2-定义溯源torch_npu-290post8-vs-converter-venv-torchair)
3. [torchair GE converter 映射](#3-torchair-ge-converter映射)
4. [Gemma-4 融合算子](#4-gemma-4-融合算子) — 4.1 参考实现汇总（15 个）/ 4.2 modeling 替换分析 / 4.3 收益排序 / 4.4 OM 算子表对比
5. [Qwen3.5 融合算子](#5-qwen35-融合算子) — 5.1 参考实现汇总（22 个）/ 5.2 modeling 替换分析 / 5.3 收益排序 / 5.4 OM 算子表对比 / 5.5 平台差异速查

---

## 1. 算子总表（fuse_op / quant_op / single_op）

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

## 2. 定义溯源：torch_npu 2.9.0.post8 vs converter venv torchair

### 融合算子定义定位分析

> 本节分析范围：原 6 个融合算子（ChunkGatedDeltaRule / Rmsnorm / FusedInferAttention / l2norm / RotaryPositionEmbedding / ScatterNdUpdate）。
> DynamicQuant / QuantBatchMatmulV4 用户新归入 fuse_op，对应 `npu_dynamic_quant` / `quant_matmul`，未在本次验证范围内。
> 验证版本：torch_npu 2.9.0.post8（系统 Python `$DOPT_PYTHON`）；converter venv（`$TORCH_DELEGATE_HOME/.venv`，torch_npu absent，torchair 源码编译）。
> 行号为验证时版本快照，升级后可能漂移，以文件路径为准。

### 2.1 torch_npu 2.9.0.post8（系统 Python）下的定义

共注册 354 个 `npu::` 算子。6 个融合算子命中情况：

| fuse_op                 | npu:: 注册　　　　　　　　　　　　　　　　　　　 | schema 来源　 | Meta 定义　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE raw op　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| --------------------------------------------------| ---------------| -----------------------------------------------------------------------------| ---------------------------------------------------------------------|
| ChunkGatedDeltaRule     | ❌ 未注册（需 torch_npu ≥2.9.0.post8）　　　　　　| 无　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| Rmsnorm                 | `npu_rms_norm`　　　　　　　　　　　　　　　　　 | C++ `_C` 扩展 | `op_plugin/meta/_meta_registrations.py`（`npu_rms_norm_meta`）　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(RmsNorm)`　　　　　　　　　　|
| FusedInferAttention     | `npu_fused_infer_attention_score` + `_v2`　　　　| C++ `_C` 扩展 | `_meta_registrations.py`（`npu_fused_infer_attention_score`/`_v2_forward`） | —　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| l2norm                  | ❌ 非 torch_npu 自定义算子　　　　　　　　　　　　| 无　　　　　　| 无　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(L2Normalize)`（仅 GE IR 级） |
| RotaryPositionEmbedding | `npu_rotary_mul`（+ `npu_apply_rotary_pos_emb`） | C++ `_C` 扩展 | `_meta_registrations.py`（`npu_rotary_mul_meta`）　　　　　　　　　　　　　 | `auto_generated_ge_raw_ops.py` `REG_OP(RotaryMul)`　　　　　　　　　|
| ScatterNdUpdate         | `npu_scatter_nd_update` + `_`　　　　　　　　　　| C++ `_C` 扩展 | `_meta_registrations.py`（`scatter_nd_update_meta`）　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(ScatterNdUpdate)`　　　　　　　|

注册映射见 `torch_npu/utils/custom_ops.py`（`npu_rotary_mul`/`npu_rms_norm`/`npu_fused_infer_attention_score[_v2]`/`npu_scatter_nd_update`）。

### 2.2 converter venv 下 torchair 的定义

torch_npu 包 absent；torchair present（`$TORCH_DELEGATE_HOME/.venv/lib/python3.11/site-packages/torchair`）。

> **迁移注脚**：下表为 torchair `custom/` 的 npu:: converter 现状（torchair 自带，描述 torchair 注册的 npu:: schema）。本仓 modeling 已迁 `cann_lm::`（见 §2.3），FX 图命中 `cann_lm::`（非 npu::），故下表 npu:: converter 不命中，由 cann_lm 自带 `ge_converters/` 覆盖（见 §3）。下表保留作 torchair 对照与历史溯源。

| fuse_op                 | venv torchair 定义文件　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 类型　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否自包含（不依赖 torch_npu）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------|
| ChunkGatedDeltaRule     | `_ge_concrete_graph/ge_converter/custom/npu_chunk_gated_delta_rule.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_chunk_gated_delta_rule.default`，且函数体直接 `raise RuntimeError("not supported in GE graph mode")`（空壳） |
| Rmsnorm                 | `_ge_concrete_graph/ge_converter/custom/npu_rms_norm.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_rms_norm`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| FusedInferAttention     | `ops/_npu_fused_infer_attention_score.py` + `_v2.py`（air::）**且** `ge_converter/custom/fused_infer_attention_score.py` + `_v2.py`（npu::） | air:: 自带 `lib.define` schema+Meta（`torch.library.Library("air","FRAGMENT")`）；npu:: 仅为 converter | ⚠️ schema+Meta 自包含（可 trace），但文件顶部仍 `import torch_npu`（converter 部分需要）　　　　　　　　　　　　　　　　　　　　　　　　|
| l2norm                  | 无（仅 `auto_generated_ge_raw_ops.py` 有 GE raw op `L2Normalize`）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE IR 级 REG_OP　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 无 schema/Meta/converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| RotaryPositionEmbedding | `_ge_concrete_graph/ge_converter/custom/npu_rotary_mul.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_rotary_mul`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ScatterNdUpdate         | `_ge_concrete_graph/ge_converter/custom/npu_scatter_nd_update.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ 引用 `torch.ops.npu.npu_scatter_nd_update`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

### 2.3 关键结论

- venv torchair 里 6 个融合算子的 `custom/` converter 注册到 `npu::`（命中 `torch.ops.npu.*`）。**但本仓 modeling 调 `torch.ops.cann_lm.*`（非 npu::），FX 图命中 `cann_lm::`**，故 torchair `custom/` 的 npu:: converter 不命中；由 CANN-LM 自带 `cann_lm/converter/export/ge_converters/` 覆盖（`ge_converter.py:22` import cann_lm ops define `cann_lm::` schema，`:33` import torchair `custom/` 注册 npu:: converter 作 fallback，`:44/54/60/69-70/73/76/82` 用 `register_fx_node_ge_converter(torch.ops.cann_lm.xxx.default)` 注册 cann_lm:: converter）。`_library.py` 用 `NPU_LIB = Library("cann_lm","FRAGMENT")` + `safe_define` 自定义 cann_lm:: schema（见 `rms_norm.py:27`/`rotary_mul.py:27` 等）。
- **唯一自包含 schema+Meta 的是 FusedInferAttention 的 air:: 版本**（`ops/_npu_fused_infer_attention_score_v2.py` 用 `torch.library.Library("air","FRAGMENT")` + `lib.define`），但其 converter 部分仍 `import torch_npu`。
- **ChunkGatedDeltaRule** 的 converter 是空壳（`raise RuntimeError`），且 schema 需 torch_npu ≥2.9.0.post8（venv 里两个都缺）。
- **l2norm** 在 torchair 完全没有定义，只有 GE raw op `L2Normalize`。
- **根因**：CANN-LM `cann_lm/models/fused_ops/_library.py` 始终注册 `cann_lm::` schema+fake+CPU impl（`_HAS_TORCH_NPU=False` 硬编码），使 `torch.export` 能 trace；GE 转换由自带 `ge_converters/` 命中 `cann_lm::`（覆盖 torchair npu:: converter）。详见 [cli_design.md](../../cli/cli_design.md)「converter 与 torch_npu 的关系」。

---

## 3. torchair GE converter 映射

> **FX→GE 转换现由 CANN-LM 自带 `cann_lm/converter/export/ge_converters/` 完成**（`ge_converter.py` 用 `register_fx_node_ge_converter(torch.ops.cann_lm.xxx.default)` 注册 `cann_lm::` converter，命中 `torch.ops.cann_lm.*`）。torchair `custom/` 的 npu:: converter 仅作 fallback/对照（modeling 不调 npu::，故不命中）。下表列出各算子 converter 的注册 schema、converter 函数及产出 GE 算子；cann_lm 自带 converter 文件简写为 `ge_converters/_<file>`，torchair 对照为 `custom/<file>`。
> 命中规则：FX 图里 `torch.ops.cann_lm.*` → 走 `ge_converters/` 自带版本（cann_lm::）；`torch.ops.npu.*`/`torch.ops.air.*` → 走 torchair `custom/`/`ops/` 版本（本仓 modeling 不命中）。

### 3.1 fuse_op

| fuse_op | converter 文件 | 注册 schema (`@register_fx_node_ge_converter`) | converter 函数 | 产出 GE 算子 | 备注 |
|---|---|---|---|---|---|
| ChunkGatedDeltaRule | `custom/npu_chunk_gated_delta_rule.py` | `npu.npu_chunk_gated_delta_rule.default` | `convert_npu_chunk_gated_delta_rule(q,k,v,*,beta?,initial_state?,actual_seq_lengths?,scale?,g?)` | ❌ 空壳 `raise RuntimeError("...not supported in GE graph mode")` | GE 图模式不支持；schema 需 torch_npu ≥2.9.0.post8。cann_lm 自带 `_chunk_gated_delta_rule_converter.py` 覆盖（→`ge.ChunkGatedDeltaRule`） |
| Rmsnorm | `custom/npu_rms_norm.py` | `npu.npu_rms_norm.default` | `conveter_npu_rms_norm_default(self,gamma,epsilon=1e-6)` | `ge.RmsNorm(self,gamma,epsilon=)`（1 个） | `@declare_supported`（F16/F32） |
| FusedInferAttention | `custom/fused_infer_attention_score_v2.py`（npu::） | `npu.npu_fused_infer_attention_score_v2.default` | `convert_npu_npu_fused_infer_attention_score_v2(q,k,v,*,query_rope?,key_rope?,pse_shift?,atten_mask?,actual_seq_qlen?,actual_seq_kvlen?,block_table?,dequant_scale_*?,quant_scale_*?,learnable_sink?,input_layout,sparse_mode,block_size,...)` | `ge.FusedInferAttentionScore(...)` → `(out,lse)` | air:: 版在 `torchair/ops/_npu_fused_infer_attention_score_v2.py`（自带 schema+Meta） |
| l2norm | — | — | — | GE raw op `L2Normalize`（仅 IR 级 REG_OP） | torchair 无 converter/schema/Meta；cann_lm 自带 `_convert_l2norm`（→`ge.L2Normalize`） |
| RotaryPositionEmbedding | `custom/npu_rotary_mul.py` | `npu.npu_rotary_mul.default` | `conveter_npu_rotary_mul_default(self,r1,r2,rotary_mode='half',rotate?)` | 默认 `ge.RotaryMul(self,r1,r2)`；arch35 或带 `rotate` 时 `ge.RotaryPositionEmbedding(self,r1,r2,mode=,rotate=)` | gemma4 走默认 `ge.RotaryMul` 分支（⚠️ OM 终态见 §4.4「RoPE 注」） |
| ScatterNdUpdate | `custom/npu_scatter_nd_update.py` | `npu.npu_scatter_nd_update.default` + `npu.npu_scatter_nd_update_.default` | out-of-place / in-place 两版 | out-of-place: `ge.TensorMove(self)`+`ge.ScatterNdUpdate(copy,idx,upd)`（2 算子）；in-place: `ge.ScatterNdUpdate(self,idx,upd)`+`dont_prune_me`（1 算子） | ⚠️ cann_lm 已自带 `converter/export/ge_converters/_scatter_nd_update_converter.py` 覆盖此 converter（产出相同 `ge.ScatterNdUpdate`，行为可控）；gemma4 已落地（`scatter_kv_to_cache` 构造多维 indices 适配标准语义） |

### 3.2 quant_op

| quant_op | converter 文件 | 注册 schema | 产出 GE 算子 | 备注 |
|---|---|---|---|---|
| DynamicQuant | `custom/npu_dynamic_quant.py` | `npu.npu_dynamic_quant.default` + `npu.npu_dynamic_quant_asymmetric.default` | `quant_mode=="pertoken" && dst_type_max≈0`: `ge.DynamicQuant(...)`→`(y,scale)`；其他: `ge.DynamicQuantV2(...)`→`(y,scale,offset)`；int4(dst_type==16) 额外 Const/Shape/Div/ConcatV2/Bitcast/Reshape bitcast 子图 | `@declare_supported`（F16/BF16） |
| QuantBatchMatmulV4 | `custom/quantmatmul.py` | `npu.npu_quant_matmul.default` | 主: `ge.QuantBatchMatmulV4(...)`；备: `ge.QuantBatchMatmulV3(...)`；含 int4/fp4 bitcast+Transpose+Reshape 子图 | ⚠️ cann_lm 实际走**自带 converter** `cann_lm/converter/export/ge_converters/_quant_matmul_converter.py`（去除 torch_npu 依赖、去掉 a8w4 分支），不命中 torchair 此 converter |

### 3.3 关键说明

- **converter 函数名笔误**：torchair 多数 converter 函数名拼写为 `conveter_*`（少一个 r，如 `conveter_npu_rms_norm_default`），系 torchair 历史笔误，非本仓问题，引用时照原拼写。
- **GE 算子名 vs OM 算子名**：converter 产出的 GE 算子（如 `ge.ScatterNdUpdate`/`ge.RmsNorm`）是 IR 阶段名；OM 编译期经融合优化后算子名会变化（`ScatterNdUpdate`/`InplaceUpdate+TensorMove` → OM `ScatterUpdate`+`MemcpyAddrAsync`；`RmsNorm`+残差 Add → OM `InplaceAddRmsNorm`）。详见 AGENTS.md「查看 OM 模型算子」。
- **converter venv 命中情况**：converter venv 不装 torch_npu，modeling 调 `torch.ops.cann_lm.*`（cann_lm::），FX 图**命中 cann_lm:: converter**（`ge_converters/` 自带，非 torchair npu::）。cann_lm 自带 converter（`_rms_norm_converter.py`/`_rotary_mul_converter.py`/`_scatter_nd_update_converter.py`/`_chunk_gated_delta_rule_converter.py`/`_quant_matmul_converter.py`/`_fused_infer_attention_score_v2_converter.py`）主体逻辑与 torchair `custom/` 的 npu:: 版本一致（直接 emit `ge.*` op，不 import torch_npu），差异仅在注册命名空间（cann_lm:: vs npu::）与算子名（去 `npu_` 前缀）。schema 由 `_library.py` `Library("cann_lm","FRAGMENT")` + `safe_define` 提供，与 torch_npu 无重复注册冲突。

---

## 4. Gemma-4 融合算子

### 4.1 参考实现（cann-recipes-infer）使用的 torch_npu 算子汇总

共 **15** 个融合算子，按功能域分为 5 类。

#### 4.1.1 归一化类（RMSNorm）

| #   | 算子　　　　　　　　　　　　 | 调用位置　　　　　　　　　　| 关键参数　　　　　　　　　　　　　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| -----| ------------------------------| -----------------------------| ----------------------------------------------------------| ---------------------------------------------------------------------------------------------------|
| 1   | `torch_npu.npu_rms_norm`　　 | `Gemma4RMSNorm.forward`　　 | `(x, weight, eps)` 返回 `[normed, ...]`　　　　　　　　　| 融合 RMSNorm：`x * rsqrt(mean(x²)+eps) * weight`，替代 pow/mean/mul 多步　　　　　　　　　　　　　|
| 2   | `torch_npu.npu_add_rms_norm` | `Gemma4RMSNorm.forward_add` | `(x1, x2, weight, eps)` 返回 `[normed, _, residual_sum]` | **融合 add + RMSNorm**：一次算出 `rms_norm(x1+x2)` 和 `x1+x2`，省一次 elementwise add + 一次 norm |

#### 4.1.2 激活类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 3 | `torch_npu.npu_fast_gelu` | `Gemma4MLP.forward`、`_GegluMoEMethod.apply` | `(x)` | `gelu(approximate='tanh')` 的融合实现，graph-safe |

#### 4.1.3 RoPE 旋转位置编码类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 4 | `torch_npu.npu_rotary_mul` | `Gemma4Attention.forward`（sliding 层） | `(q4, cos4, sin4, rotary_mode='half')`，需 4D BSND 输入 | 融合 rotate-half RoPE，用于 sliding 层全 head_dim 旋转 |
| 5 | `torch_npu.npu_apply_rotary_pos_emb` | `Gemma4Attention.forward`（full 层） | `(q_rot, k_rot, cos, sin, layout="TND")` | 融合 partial-rotary RoPE（full 层只旋 rotary_dim 维），支持 TND layout |

#### 4.1.4 Attention 类（FlashAttention + PagedAttention）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 6 | `torch.ops.npu.npu_fused_infer_attention_score_v2`（ge_graph decode 经 `torchair.ops`） | `_attn_tnd_sliding`、`_attn_bnsd_full` | `input_layout="TND"/"BNSD"`、`sparse_mode`、`pre_tokens`、`block_table`、`block_size`、`num_key_value_heads`（GQA） | 融合 QK^T + softmax + AV + PagedAttention，sliding 层用 TND(D=256)+sparse_mode=4，full 层用 BNSD(D=512)+sparse_mode=3/0 |
| 7 | `torch_npu.npu_scatter_nd_update_` | `Gemma4Attention._update_cache` | `(cache.view(-1,N,D), slot, kv)` in-place | 融合 KV cache scatter-write（按 slot_mapping 写入 PagedAttention 缓存） |

#### 4.1.5 MoE 类（本仓 Gemma-4 当前为纯 dense MLP，不含 MoE）

参考实现含 8 个 MoE 算子：`npu_moe_gating_top_k_softmax`（top-k+softmax 路由）、`npu_grouped_matmul`（分组 GEMM）、`npu_geglu`（融合 GEGLU 激活，仅 prefill/eager，graph decode 退化为 manual `npu_fast_gelu*up`）、`npu_moe_init_routing_v2`/`npu_moe_finalize_routing`/`npu_moe_re_routing`（路由初始化/收尾/EP 二次重排）、`npu_moe_distribute_dispatch_v2`/`npu_moe_distribute_combine_v2`（MC2 EP dispatch/combine）。本仓 `modeling_gemma4.py` 为纯 dense MLP（`Gemma4TextMLP`，无 `Gemma4SparseMoeBlock`），**均不适用**；若后续引入 MoE，可对齐参考实现 `Gemma4SparseMoeBlock` 三种路由模式（TP / EP double-routing / MC2 dispatch-combine）。

---

### 4.2 本仓 `modeling_gemma4.py` 可用的融合算子分析

本仓实现调用 `torch.ops.npu.*`（+ `torch.ops.cann_lm::l2norm`）融合算子，以下逐模块给出「当前实现 → 融合算子」的映射。标注 `✅ 已落地+已用` / `✅ 可直接替换` / `❌ 不采用` / `❌ 未落地`。

#### 4.2.1 RMSNorm → `npu_rms_norm`（已落地+已用）/ `npu_add_rms_norm`（不采用）

**当前实现**（`Gemma4RMSNorm.forward`，modeling_gemma4.py:301-303，已无 `_norm` 纯 torch 实现）：
```python
def forward(self, x):
    y, _ = torch.ops.cann_lm.rms_norm(x, self.weight, epsilon=self.eps)
    return y
```

> 原文档此处展示的 `pow(2).mean + pow(-0.5) + mul` 纯 torch `_norm` 已删除（modeling_gemma4.py:288 docstring「via cann_lm::rms_norm (fused)」）。`with_scale=False` 子类（v_norm/router）用 `register_buffer("weight", ones)` 预分配 ones weight 传入。

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `cann_lm::rms_norm` | ✅ 已落地+已用 | modeling 行 302/311 已调用 `torch.ops.cann_lm.rms_norm`（`Gemma4RMSNorm.forward` / `forward_add`）。`with_scale=False` 时传 ones weight（v_norm/router norm 即如此；本仓 `with_scale=False` 用 `register_buffer("weight", ones)` 预分配，modeling_gemma4.py:299） |
| `torch_npu.npu_add_rms_norm` | ❌ 不采用（设计决策） | 融合 residual add + rms_norm。**本仓显式拆为 add + `npu_rms_norm` 两步**（`forward_add`，modeling_gemma4.py:305-312，注释「原用 fused npu_add_rms_norm，现拆为 add + npu_rms_norm（用户要求）」）。参考实现 `Gemma4DecoderLayer` 用融合版，本仓按设计决策走拆分；下方 §4.2.2 的融合建议已被该决策否决，保留仅作设计分析 |

#### 4.2.2 Residual Add + Norm 拓扑（`forward_add` 拆分实现）

**当前实现**（`Gemma4TextDecoderLayer.forward`，modeling_gemma4.py:487-494）：残差 add 与 norm 分两步（`forward_add` 拆为 add + `npu_rms_norm`）。
```python
hidden_states = self.post_attention_layernorm(hidden_states)   # norm(attn_out)
hidden_states = residual + hidden_states                        # add（独立语句）
...
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)   # norm(add 结果)
```

> 参考实现 `Gemma4DecoderLayer.forward`（line 862）用 `npu_add_rms_norm` 把 add+norm 合为单算子；本仓按设计决策拆分（见 §4.2.1 表「不采用」）。

#### 4.2.3 GELU(tanh) → `npu_fast_gelu`

**当前实现**：
- `Gemma4TextMLP.__init__`（line 272）：`self.act_fn = nn.GELU(approximate="tanh")`
- `Gemma4TextDecoderLayer.forward` per_layer_input（line 499）：`F.gelu(gated, approximate="tanh")`

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_fast_gelu` | ✅ 可直接替换 | 参考实现注释明确 `npu_fast_gelu == gelu(approximate='tanh')`，graph-safe。MLP 的 `act_fn(gate)*up` 模式与参考实现 `Gemma4MLP.forward` 一致 |

#### 4.2.4 RoPE → `npu_rotary_mul`（`npu_apply_rotary_pos_emb` 未用）

**已落地**（modeling_gemma4.py）：与图灵参考实现（cann-recipes-infer）一致，按 layer_type 分两条路径：
- **sliding 层**（全旋，`head_dim == cos.shape[-1]`）：`npu_rotary_mul` 单 tensor，q/k 各调一次（`rotary_mode='half'`）。
- **full 层**（proportional partial，`rotary_dim == cos.shape[-1] < head_dim`）：cat 出 q_rot/k_rot（前 `pair_dim` + 中 `half_dim:half_dim+pair_dim`），**`npu_rotary_mul` q/k 各调一次**（`rotary_mode='half'`，非 `npu_apply_rotary_pos_emb`），拆装插回原位（中间和后面保持原值）。见 `_rope_full_pair`（modeling_gemma4.py:119-136）。
- **kv_shared 路径**（q 单独，k 从 shared 来已 RoPE）：用 `apply_rotary_pos_emb_single`（`npu_rotary_mul` + cat/拆装 模拟 partial），避免对已 RoPE 的 k 再次旋转。
- `inv_freq` 对 proportional 不再补 0（长度 = `rope_angles = rotary_dim//2`），`cos/sin` 长度 = `rotary_dim`，由 full 层 cat/拆装消费（与参考实现一致；原"inv_freq 补 0 + 整 head_dim 过 rotary_mul"路径已废弃）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rotary_mul` | ✅ 已落地+已用 | sliding 层（全旋）单 tensor；full 层 + kv_shared 路径（cat/拆装模拟 partial）。GE converter 在 cann_lm `_rotary_mul_converter.py`→`ge.RotaryMul`。modeling 行 107/114/115/126/127/145 调用 `torch.ops.cann_lm.rotary_mul` |
| `torch_npu.npu_apply_rotary_pos_emb`（q+k 配对） | ❌ 未用 | full 层改用 `npu_rotary_mul` + cat/拆装（`_rope_full_pair`），**未调用** `npu_apply_rotary_pos_emb`。原文档「已落地」为 stale |

> ⚠️ `rope_in_place` / `ge.RopeInPlace` **未落地**：`fused_ops/ops.py` 未注册、`ge_converter.py` 无对应 converter，原文档此处为 stale 声明。
>
> ⚠️ RoPE OM 终态与 IR 不一致（待核实）：IR 级 converter 默认产出 `ge.RotaryMul`，但 OM 终态 RoPE 算子为 `RotaryPositionEmbedding`（gemma4 prefill/decode 各 50，非 `RotaryMul`）。详见 §4.4「RoPE 注」——疑似 950DT 被识别为 arch35 或命中带 `rotate` 分支。

#### 4.2.5 Eager Attention → `npu_fused_infer_attention_score_v2`

**已落地**（modeling_gemma4.py）：与图灵参考实现一致，`fused_attention_forward` 按 `layer_type` 调
`npu_fused_infer_attention_score_v2`（BNSD layout），融合 QK^T+softmax+AV+GQA+causal/sliding。
- **sliding 层**（全旋 head_dim=256）：`sparse_mode=4`（sliding_window）+ `pre_tokens=sliding_window`。
- **full 层**（global_head_dim=512）：`sparse_mode=3`（causal）+ `pre_tokens=INT_MAX`（全 attention）。
- **GQA 原生**：`num_query_heads` vs `num_key_value_heads`（消除 `repeat_kv`，原 35 层×2=70 个 `ConcatV2D` 归零）。
- **mask 适配**：cann_lm 的 float mask `[*, q_len, kv_len]` → 融合算子要求 bool `[*, kv_len, kv_len]` 方阵
  （`_pad_mask_to_kv_square` pad q 维到 kv_len，融合算子用 `actual_seq_qlen` 切片前 q_len 行）；`actual_seq_qlen/kvlen` 用 int64。
- **保留**：`eager_attention_forward` + `repeat_kv` + `_prepare_causal_mask`（未删，仅不再调用，作为 fallback/对照）。

| 算子 | 状态 | 说明 |
|------|--------|------|
| `cann_lm::fused_infer_attention_score_v2` | ✅ 已落地 | cann_lm `fused_ops/fused_infer_attention_score.py` 注册 schema+fake+CPU eager（`cann_lm::` 命名空间）；`ge_converters/_fused_infer_attention_score_v2_converter.py` 注册 converter（调 `ge.FusedInferAttentionScore`，40+ 参数全传，None 给不用）；`reg_register.h` 加 `REG_OP(FusedInferAttentionScore)`（OM 编译注册到 GE OperatorFactory）。消除 `BatchMatMul`/`SoftmaxV2`（归零），`ConcatV2D` 从 14 降到 9（repeat_kv 消除） |
| `cann_lm::fused_infer_attention_score` | ❌ 未采用（文档原提） | 原文档提的 cann_lm 自定义 FA 算子（IR 级 fallback），实际改用 npu 命名空间（与图灵一致），未用 cann_lm 命名空间 |

> softcap 部分：参考实现把 `1/sqrt(d)` 吸收进 q_norm（`scale_fa=1.0`），本仓 `scaling=1.0`
> 已一致；logit softcap（`x/cap→tanh→x*cap`）无单一融合算子，`fused_attention_forward` 未传 softcap
> （gemma4 attention 无 softcap，logit softcap 在 `Gemma4ForCausalLM` 末尾，不在 attention 内）。

##### 遗留问题：融合算子引入权重时的支持方案（当前 gemma4 不涉及，量化/sink 场景需关注）

当前 `FusedInferAttentionScore` 在 gemma4 中**无 weight 参数**（只接收 Q/K/V + 配置常量），故 eager→fused 切换可复用旧 `weights_0.bin`（q/k/v/o_proj weight 不变）。**遗留场景**：若融合算子引入可学习 weight（量化 `antiquant_scale`/`antiquant_offset`/`dequant_scale_query`/`dequant_scale_key`/`dequant_scale_value`、或 attention sink `learnable_sink`），则不能复用旧 `.bin`（新 weight 无索引，OM 编译 Const 找不到）。**支持方案**：把新 weight 暴露为 `nn.Parameter`（跟 q_proj 同级），`torch.export` + `consolidated_weights_save` 自动 lift+落盘，无需手动管 `weights_0.bin` 偏移——需同步：① `Attention.__init__` 加 `nn.Parameter`；② `load_gemma4_weights` 加 HF→cann_lm name 映射；③ `fused_attention_forward` 传新 weight；④ `fused_infer_attention_score.py` schema 补可选输入；⑤ `ge_converter.py` converter 传非 None（`REG_OP(FusedInferAttentionScore)` 已有 `learnable_sink` 等 OPTIONAL_INPUT）；⑥ `--save-weights=True` 重新落盘（**不能复用旧 .bin**）。**触发条件**：`fused_attention_forward` 给融合算子传非 None 的 `learnable_sink`/`antiquant_scale`/`dequant_scale_*` 等时；当前 gemma4（纯 bf16，无量化/sink）全传 None，不触发。

#### 4.2.6 KV Cache 写入 → `npu_scatter_nd_update_`

**已落地实现**（`Gemma4TextAttention.forward` + `scatter_kv_to_cache` helper，modeling_gemma4.py）：
```python
cache_k = scatter_kv_to_cache(cache_k, new_kv_cache_pos, key_states)
cache_v = scatter_kv_to_cache(cache_v, new_kv_cache_pos, value_states)
```
`scatter_kv_to_cache` 构造 `[b*n*seq, 3]` 多维坐标 indices (b,n,kv_pos) + `[b*n*seq, d]` updates，命中 `ge.ScatterNdUpdate` 标准多维 scatter 语义（等价 `index_copy(2, kv_pos, kv_new)`，CPU 验证 diff=0）。

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_scatter_nd_update_` | ✅ 已落地 | cann_lm 自带 converter (`converter/export/ge_converters/_scatter_nd_update_converter.py`) 覆盖 torchair，直接产出 `ge.ScatterNdUpdate`（不绕 `aten.index_copy` 的 `InplaceUpdate+Transpose`）。gemma4 经 `scatter_kv_to_cache` 构造多维 indices 适配标准语义。CPU else 分支与 GE `ge.ScatterNdUpdate` 语义一致，export trace 命中 `npu_scatter_nd_update_` |

#### 4.2.7 MoE 算子（不适用）

本仓 `modeling_gemma4.py` 当前为**纯 dense MLP**（`Gemma4TextMLP`，无 `Gemma4SparseMoeBlock`），
故参考实现的 8 个 MoE 算子（`npu_moe_gating_top_k_softmax`、`npu_grouped_matmul`、`npu_geglu`、
`npu_moe_init_routing_v2`、`npu_moe_finalize_routing`、`npu_moe_re_routing`、
`npu_moe_distribute_dispatch_v2`、`npu_moe_distribute_combine_v2`）**均不适用**。若后续本仓
引入 MoE 路径，可逐一对齐参考实现的 `Gemma4SparseMoeBlock` 三种路由模式（TP / EP double-routing /
MC2 dispatch-combine）。

---

### 4.3 收益排序与建议落地顺序

按「单次 forward 调用频次 × 单算子收益」排序：

| 优先级 | 融合算子 | 替换目标 | 状态/预期收益 |
|--------|----------|----------|----------|
| P0 | `cann_lm::fused_infer_attention_score_v2` | `eager_attention_forward` 全流程 | ✅ **已落地+已用**：modeling:259 调用 `torch.ops.cann_lm.fused_infer_attention_score_v2`；消除 repeat_kv+2×matmul+softmax+mask 构造，单层最大 kernel 融合，全模型 N 层 |
| P0 | `cann_lm::rms_norm` | `Gemma4RMSNorm`（原 `_norm` 纯 torch 已删） | ✅ **已落地+已用**：modeling:296/305 调用 `torch.ops.cann_lm.rms_norm`；每层 4-6 个 RMSNorm（input/post_attn/pre_ffn/post_ffn + q/k/v_norm），高频 |
| P1 | `npu_fast_gelu` | `nn.GELU(tanh)` / `F.gelu(tanh)` | ❌ 未落地：modeling 仍用 `nn.GELU`/`F.gelu`（行 438/671）；每层 MLP + per_layer_input 各一次 |
| P1 | `npu_add_rms_norm` | residual add + 下一 norm | ❌ 不采用（设计决策）：拆为 add + `npu_rms_norm`（modeling_gemma4.py:308「用户要求」） |
| P2 | `cann_lm::rotary_mul`（已落地） | RoPE 配对 + cat/拆装 | ✅ **已落地+已用**：sliding/full/kv_shared 全走 `torch.ops.cann_lm.rotary_mul`（modeling:107/114/115/126/127/145）；`npu_apply_rotary_pos_emb` 未用 |
| P2 | `cann_lm::scatter_nd_update_`（已落地） | `index_copy` KV cache 写入 | ✅ **已落地+已用**：modeling 用 `scatter_kv_to_cache`（多维 indices）+ cann_lm `ge.ScatterNdUpdate` converter |

> 注：本仓当前为纯 PyTorch 参考实现，引入 `torch_npu` 需遵循 AGENTS.md 的 stub 约定
> （converter 侧 torchair 编译 + lib 层打桩），NPU 实机验证走系统 Python 的 torch_npu。

---

### 4.4 OM 算子表对比 support_ops（gemma4 无量化 prefill-decode, Ascend950DT_9575）

> 导出 `epair_gemma4_no_quant`（prefill-decode, seq_len=128 / decode=1）→ C++ `export_llm_model` 编译为 OM（`SOC_VERSION=Ascend950DT_9575`，fusion cfg 关 AUB + AddRmsNorm/InplaceAddRmsNorm）。下表为 OM 终态算子 vs 本文 §1 定义。

#### 4.4.1 fuse_op

| fuse_op | OM 算子 | prefill | decode | 命中 |
|---|---|---:|---:|---|
| ChunkGatedDeltaRule | — | 0 | 0 | 模型架构不用（qwen3.5 SSM 才有）|
| Rmsnorm | RmsNorm | 242 | 242 | ✅ |
| FusedInferAttention | FusedInferAttentionScore | 35 | 35 | ✅（每层 1 个）|
| l2norm (L2Normalize) | — | 0 | 0 | 模型架构不用 |
| RotaryPositionEmbedding | RotaryPositionEmbedding | 50 | 50 | ⚠️ 命中（见下「RoPE 注」）|
| ScatterNdUpdate | ScatterNdUpdate | 30 | 30 | ✅ |

#### 4.4.2 quant_op（基线无量化）

| quant_op | OM | prefill | decode | 命中 |
|---|---|---:|---:|---|
| DynamicQuant | — | 0 | 0 | 未量化 |
| QuantBatchMatmulV4 | — | 0 | 0 | 未量化（走 MatMulV3）|

#### 4.4.3 single_op

| single_op | OM 算子 | prefill | decode | 命中 |
|---|---|---:|---:|---|
| Cast | Cast | 66 | 66 | ✅ |
| Mul | Mul | 105 | 105 | ✅ |
| splitV | — | 0 | 0 | 未用（StridedSliceV2 覆盖）|
| concatD | ConcatD | 85 | 85 | ✅ |
| Transpose | Transpose | 100 | 0 | ✅（prefill 独有；decode seq=1 不需）|
| Conv2DV2 | — | 0 | 0 | gemma4 无 conv1d |
| StridedSliceV2 | StridedSliceV2 | 60 | 60 | ✅ |
| Add | Add | 106 | 106 | ✅ |
| Swish | — | 0 | 0 | gemma4 用 Gelu |
| GatherV2 | GatherV2 | 40 | 40 | ✅ |
| TransData | — | 0 | 0 | 未用 |
| Sigmoid | — | 0 | 0 | 未用 |
| MatmulV3 | MatMulV3 | 277 | 277 | ✅ |
| SoftplusV2 | — | 0 | 0 | 未用 |
| Gelu | Gelu | 70 | 70 | ✅ |

#### 4.4.4 OM 框架/融合变体算子（不在 §1 范畴）

| OM 算子 | prefill | decode | 性质 |
|---|---:|---:|---|
| Const | 561 | 559 | 权重/常量（图基础设施）|
| Data | 77 | 77 | 图输入（基础设施）|
| NetOutput | 1 | 1 | 图输出（基础设施）|
| MemcpyAddrAsync | 40 | 40 | 共享层 KV 透传（CANN 基础设施 op）|
| Muls | 2 | 2 | 标量乘（Mul 标量变体，框架算子，保留）|

#### 4.4.5 最终 OM 算子表

**prefill（1947 个 / 18 种）**：Const 561 · MatMulV3 277 · RmsNorm 242 · Add 106 · Mul 105 · Transpose 100 · ConcatD 85 · Data 77 · Gelu 70 · Cast 66 · StridedSliceV2 60 · RotaryPositionEmbedding 50 · MemcpyAddrAsync 40 · GatherV2 40 · FusedInferAttentionScore 35 · ScatterNdUpdate 30 · Muls 2 · NetOutput 1.

**decode（1845 个 / 17 种）**：同 prefill 但无 Transpose（seq=1 不需转置），Const 559。

#### 4.4.6 结论

- **fuse_op**：4/6 命中（Rmsnorm / FusedInferAttention / RotaryPositionEmbedding / ScatterNdUpdate），2 未用是模型架构差异（ChunkGatedDeltaRule SSM、l2norm）。
- **single_op**：9/14 命中，5 未用是架构差异（splitV / Conv2DV2 / Swish / TransData / Sigmoid / SoftplusV2）。
- **quant_op**：0/2（基线无量化，DynamicQuant / QuantBatchMatmulV4 均为 0）。
- **框架算子**（Const / Data / NetOutput / MemcpyAddrAsync / Muls）：本就不属 §1 范畴。
- **未覆盖算子全部消除**：`InplaceAddRmsNorm` = 0（关 `AddRmsNormFusionGraphPass` / `InplaceAddRmsNormFusionPass`，§7）；无 `PartitionedCall`（actual_seq 内联为真 `ge.Const`，不进动态子图）。

#### ⚠️ RoPE 注

§3.1 表记「gemma4 走默认 `ge.RotaryMul` 分支」，但实测 OM 终态 RoPE 算子为 `RotaryPositionEmbedding`（50，非 `RotaryMul`）。按 torchair `npu_rotary_mul` converter，`RotaryPositionEmbedding` 仅在 arch35 或带 `rotate` 时产出——当前命中此分支（疑似 950DT 被识别为 arch35，或 modeling 传了 `rotate`），与文档描述有出入，待核实。

---

## 5. Qwen3.5 融合算子

> 数据来源：`cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py`（3237 行）
> 本仓实现：`cann_lm/models/qwen3_5/modeling_qwen3_5.py`

Qwen3.5 是**混合架构**：full_attention 层（标准 GQA + KV cache）+ linear_attention 层
（GatedDeltaNet 线性注意力 + conv1d + recurrent state）。参考实现的融合算子重点在
**线性注意力**（`npu_chunk_gated_delta_rule` / `npu_recurrent_gated_delta_rule` / `causal_conv1d` /
`FusedRMSNormGated`），这也是本仓 `export_to_epair` segfault 的根因（纯 torch 的
`torch_chunk_gated_delta_rule` Python for 循环展开 19418 节点导致 GE 图爆炸）。

### 5.1 参考实现（cann-recipes-infer）使用的 torch_npu 算子汇总

共 **22** 个融合算子，按功能域分 6 类。线性注意力类（#1-#6）是 Qwen3.5 特有，其余与
Gemma-4 重叠（详见 §4.1，此处只列差异）。

#### 5.1.1 线性注意力类（Qwen3.5 特有，核心）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 1 | `torch_npu.npu_chunk_gated_delta_rule` | `GatedDeltaNet._forward_prefill_with_fused_kernel`（行 1541） | `(query, key, value, beta=, initial_state=, actual_seq_lengths=, scale=, g=)` | **prefill 线性注意力核心**：融合 chunked gated delta rule 全流程（L2norm + chunk decay + intra/inter-chunk recurrence + state update），单算子替代纯 torch 的 ~100 行 Python for 循环 |
| 2 | `torch_npu.npu_recurrent_gated_delta_rule` | `GatedDeltaNet._forward_decode`（行 1646） | `(query, key, value, last_recurrent_state, beta=, scale=, actual_seq_lengths=, ssm_state_indices=, num_accepted_tokens=, g=, gk=)` | **decode 线性注意力核心**：融合 token-by-token 递归（state *= exp(g) → kv_mem → delta → state += → out），支持 speculative decode 的 num_accepted_tokens |
| 3 | `causal_conv1d_fn`（外部注入） | `GatedDeltaNet._forward_prefill`（行 1336） | `(x, weight, bias, activation, seq_idx)` | prefill conv1d 融合：state 滚动 + depthwise conv1d + SiLU 单 kernel |
| 4 | `causal_conv1d_update`（外部注入） | `GatedDeltaNet._forward_decode`（行 1598） | `(x, conv_state, weight, bias, activation)` 位置参数 | decode conv1d 融合：单 token 的 state 滚动 + conv + SiLU |
| 5 | `FusedRMSNormGated`（外部注入） | `GatedDeltaNet.__init__`（行 1153） | `FusedRMSNormGated(head_v_dim, eps, activation, device, dtype)`，forward `(core_attn_out, z)` | 线性注意力的 norm + gate 融合：`rms_norm(x) * silu(gate)` 单 kernel |
| 6 | `ge_safe_softplus`（纯 torch 替代） | `GatedDeltaNet`（行 1518） | `relu(x) + log1p(exp(-abs(x)))` | 非 NPU 算子，但为 ge_graph 避开 `aten::softplus`（用 relu+log1p 表达），本仓可直接借鉴 |

> **启用条件**（行 1178-1181）：`use_npu_chunk_gated_delta_rule = (_PLATFORM_VERSION != "950"
> and hasattr(torch_npu, "npu_chunk_gated_delta_rule"))` —— 即 **950 走纯 torch prefill，
> 910/A3 走融合算子**。

#### 5.1.2 归一化类（RMSNorm，与 Gemma-4 重叠）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 7 | `torch_npu.npu_rms_norm` | `Qwen3_5MoeRMSNormGated.ln_npu`（行 885）、`Qwen3_5MoeRMSNorm.ln_npu`（行 2577） | `(x, weight, eps)[0]` | 融合 RMSNorm |
| 8 | `torch_npu.npu_add_rms_norm` | `Qwen3_5MoeRMSNorm.add_rms_norm_npu`（行 2615） | `(residual, x, gamma, eps)` 返回 `(y, _, residual)` | 融合 residual add + RMSNorm |
| 9 | `torch_npu.npu_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.rms_norm_mx`（行 2588） | `(x, gamma, epsilon, dst_type)` 返回 `(x_quant, x_scale, _)` | RMSNorm + MXFP8 量化输出（供下游量化 linear） |
| 10 | `torch_npu.npu_add_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.add_rms_norm_mx`（行 2605） | `(residual, x, gamma, epsilon, dst_type)` 返回 `(y, x_out, mxscale, _)` | add + RMSNorm + MXFP8 量化 |

#### 5.1.3 激活类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 11 | `torch_npu.npu_swiglu` | `SiLUAndMul.forward`（行 742） | `(x)` | 融合 SiLU+Mul：`silu(gate) * up`（MLP gate_up_proj 后） |
| 12 | `torch_npu.npu_swiglu_mx_quant` | `SiLUAndMul.forward_mx`（行 746） | `(x, group_index, dst_type, activate_left)` 返回 `(out, scale)` | SiLU+Mul + MXFP8 量化（供量化 down_proj） |

#### 5.1.4 Attention 类（full_attention 层）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 13 | `torch_npu.npu_rotary_mul` | `Qwen3_5MoeAttention.forward`（行 1961） | `(q, cos, sin, rotary_mode='half')` | RoPE 融合（与 Gemma-4 一致） |
| 14 | `torch_npu.scatter_update_` | `Qwen3_5MoeAttention.forward`（行 1975） | `(cache, indices, updates, dim=-2)` in-place | KV cache 写入（替代 index_copy） |
| 15 | `torch.ops.npu.npu_fused_infer_attention_score` | `Qwen3_5MoeAttention.forward`（行 1979 prefill / 1998 decode） | prefill: `sparse_mode=3, atten_mask=...`；decode: `actual_seq_lengths_kv=..., atten_mask=None` | FlashAttention 融合（QK^T+softmax+AV+GQA+PagedAttention） |

#### 5.1.5 量化类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 16 | `torch_npu.npu_dynamic_mx_quant` | `_quantize_weight_last_dim`（行 286） | `(weight_2d, dst_type=float8_e4m3fn)` 返回 `(quant, scale)` | 权重在线 MXFP8 量化 |
| 17 | `torch_npu.npu_mm_all_reduce_base` | `qwen3_5_prefill_mm_all_reduce`（行 219） | `(input, weight, hcom, reduce_op="sum")` | MM + AllReduce 融合（仅 950，prefill 的 o_proj/down_proj） |

#### 5.1.6 MoE 类（本仓 Qwen3.5 为 dense，不含 MoE；参考实现 MoE 版有 8 个，与 Gemma-4 §4.1.5 #8-#15 一致，略）

---

### 5.2 本仓 `modeling_qwen3_5.py` 可用的融合算子分析

本仓实现调用 `torch.ops.npu.*`（+ `torch.ops.cann_lm::l2norm`）融合算子，以下逐模块给出「当前实现 → 融合算子」的映射。标注 `✅ 已落地` / `⚠️ 需重构` / `❌ 不适用`。

#### 5.2.1 线性注意力核心（GatedDeltaNet）— **segfault 已解决（chunk_gated_delta_rule 已落地）**

**当前实现**（`Qwen3_5GatedDeltaNet.forward`，modeling_qwen3_5.py）：
- **active 路径**：prefill + decode **共用** `torch.ops.cann_lm.chunk_gated_delta_rule`（行 621），融合算子已落地（`fused_ops/chunk_gated_delta_rule.py` + `_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule`，ge_converter.py:59-60）。OM 终态命中 18 个。
- **fallback（未激活）**：`torch_chunk_gated_delta_rule`（行 264）+ `torch_recurrent_gated_delta_rule`（行 346）为纯 torch 参考实现，保留作对照，不在 export 路径。

| 算子 | 状态 | 说明 |
|------|------|------|
| `torch_npu.npu_chunk_gated_delta_rule` | ✅ 已落地+已用 | segfault 已解决：`fused_ops/chunk_gated_delta_rule.py` 注册 schema+fake+CPU impl（`npu::` 命名空间）；`_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule`（ge_converter.py:55-56）；modeling 行 621 调用。OM 终态命中 18 个 |
| `torch_npu.npu_recurrent_gated_delta_rule` | ❌ 未注册（不需要） | decode **不走 recurrent**：chunk_gated_delta_rule prefill+decode 共用（decode seq=1 时 chunk 内 pad，见 `chunk_gated_delta_rule.py` docstring）。`torch_recurrent_gated_delta_rule`（行 346）仅作纯 torch fallback 保留 |
| `causal_conv1d_fn` / `causal_conv1d_update` | ✅ aten::conv1d 路径已落地 | modeling `torch_causal_conv1d_update`（行 243-260）用 `F.conv1d`+`F.silu`；`ge_converter.py:58-59` 注册 `aten::conv1d` converter（`_conv1d_converter.py`）。未注册独立 `cann_lm::causal_conv1d`（aten 路径已够） |
| `FusedRMSNormGated` | ⚠️ 拆解方案已采用（融合版未落地） | modeling `Qwen3_5RMSNormGated`（行 40-42）= `cann_lm::rms_norm` + `F.silu(gate)*x`（拆解，非融合单 kernel）。`cann_lm::rms_norm`→`ge.RmsNorm` 已用；融合 `rms_norm_gated` 未注册 |

> **平台分支**：参考实现 `npu_chunk_gated_delta_rule` 只在非 950（910/A3）启用，950 走纯 torch。
> 本仓已统一用 `cann_lm::chunk_gated_delta_rule`（910 实机 + export 都走融合算子，segfault 已解决）；
> 950 仍走纯 torch fallback（950 导出 segfault 问题待 950 专属解）。

#### 5.2.2 RMSNorm → `npu_rms_norm`（已落地+已用）/ `npu_add_rms_norm`（未落地）

**当前实现**：`Qwen3_5RMSNorm.forward`（行 22-28）已调用 `torch.ops.cann_lm.rms_norm`（行 25，`1.0 + weight` 偏置约定）；`Qwen3_5RMSNormGated.forward`（行 40-42）= `cann_lm::rms_norm` + `F.silu(gate)*x`。原文档「纯 torch pow」描述为 stale。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rms_norm` | ✅ 已落地+已用 | `fused_ops/rms_norm.py` 注册 schema+fake+CPU impl（`cann_lm::` 命名空间）；GE converter 在 cann_lm `_rms_norm_converter.py`（ge_converter.py:72-73 注册 `torch.ops.cann_lm.rms_norm.default`）→`ge.RmsNorm`。modeling 行 25/41 **已调用** `torch.ops.cann_lm.rms_norm` |
| `cann_lm::add_rms_norm` | ❌ 未落地 | 无 `fused_ops/add_rms_norm.py`、ge_converter.py 无此注册（原文档「已落地」+文件为 stale）。gemma4 已显式拆分为 add + `cann_lm::rms_norm`（modeling_gemma4.py:308 注释「用户要求」），qwen3.5 同设计。若要融合需新建 schema+converter |

#### 5.2.3 RoPE → `npu_rotary_mul`（已落地）

**当前实现**：`Qwen3_5RotaryEmbedding`（行 68）+ attention 用 `torch.ops.cann_lm.rotary_mul`（行 128/129，`rotary_mode='half'`），非 `apply_rotary_pos_emb`。原文档描述为 stale。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rotary_mul` | ✅ 已落地+已用 | `fused_ops/rotary_mul.py` 注册 schema+fake（`cann_lm::` 命名空间）；GE converter 在 cann_lm `_rotary_mul_converter.py`（ge_converter.py:75-76 注册 `torch.ops.cann_lm.rotary_mul.default`）→`ge.RotaryMul`/`ge.RotaryPositionEmbedding`（arch35）。modeling 行 128/129 **已调用** `torch.ops.cann_lm.rotary_mul`。与 Gemma-4 共用 |

#### 5.2.4 KV Cache 写入 → `scatter_update_`（已落地，IR 路径通）

**当前实现**（`SimpleKVCache.update`，行 330-332）：`index_copy(2, new_kv_cache_pos, kv)`。

| 算子 | 状态 | 说明 |
|------|------|------|
| `aten::index_copy` → `ge.InplaceUpdate + TransposeD` | ✅ 已落地 | `ge_converter.py:61-62` 注册 `aten::index_copy` converter（`_index_copy_converter.py`：TensorMove + TransposeD + InplaceUpdate，**非** `ge_converter.py:234` 该行号不存在）。IR 导出路径已通。实机可用 `npu_scatter_update_` |

#### 5.2.5 Full Attention → `npu_fused_infer_attention_score_v2`（op 已落地；启用状态待核实）

**当前实现**：`Qwen3_5Attention`（行 520-569）= eager attention（matmul QK^T + softmax + matmul AV + GQA repeat_kv）。modeling 另有 FIA 辅助函数（行 215）调 `cann_lm::fused_infer_attention_score_v2`（带 decode 正确性注释），但 §5.4 qwen3.5 OM 表显示 FIA=0（全 eager，「decode 修复暂搁置」）—— 二者矛盾，export 实际走哪条路径待核实。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::fused_infer_attention_score_v2` | ✅ op 已落地（启用待核实） | `fused_ops/fused_infer_attention_score.py` 注册 schema+fake（`cann_lm::` 命名空间）；`ge_converter.py:81-82`（`_fused_infer_attention_score_v2_converter.py`）→`ge.FusedInferAttentionScore`。modeling 行 215 有调用 `torch.ops.cann_lm.fused_infer_attention_score_v2`，但 OM 终态 FIA=0（见 §5.4 qwen3.5 表），export 路径待核实 |

#### 5.2.6 激活 → `npu_swiglu`（未落地）

**当前实现**：`Qwen3_5MLP.forward`（行 569）= `F.silu(gate) * up` 纯 torch（或 `gate * silu(up)`）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `npu_swiglu` | ❌ 未落地 | 可注册 `cann_lm::swiglu` + GE 映射 `ge.Swiglu`（查 CANN op_info）。优先级低，纯 torch 的 silu+mul 在 GE 导出时分解为 Swish/Sigmoid+Mul，已能导出 |

---

### 5.3 收益排序与建议落地顺序

按「解决导出 segfault / 实机推理提速」排序：

| 优先级 | 融合算子　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 替换目标　　　　　　　　　　　　　　　　　　 | 状态/预期收益　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------| -----------------------------------------------------------------| ----------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **P0** | `cann_lm::chunk_gated_delta_rule`（→ `ge.ChunkGatedDeltaRule`） | `torch_chunk_gated_delta_rule`（prefill）　　| ✅ **已解决**：`fused_ops/chunk_gated_delta_rule.py` + `_chunk_gated_delta_rule_converter.py`（ge_converter.py:59-60）；modeling 行 621 调用 torch.ops.cann_lm.chunk_gated_delta_rule；OM 命中 18 个。19418 节点 → 1 算子 |
| **P0** | `npu_recurrent_gated_delta_rule`　　　　　　　　　　　　　　　　| `torch_recurrent_gated_delta_rule`（decode） | ❌ **不需要**：decode 走 chunk_gated_delta_rule 共用（chunk 内 pad）；recurrent 仅作 torch fallback 保留，未注册　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P1　　 | `aten::conv1d`（→ `_conv1d_converter.py`）　　　　　　　　　　　| `torch_causal_conv1d_update`　　　　　　　　 | ✅ **已落地**：ge_converter.py:58-59 注册 `aten::conv1d`；modeling 用 `F.conv1d`+`F.silu`（行 259-260）。未建独立 `cann_lm::causal_conv1d`　　　　　　　　　　　　　　　　　　　 |
| P1　　 | `FusedRMSNormGated`　　　　　　　　　　　　　　　　　　　　　　 | `Qwen3_5RMSNormGated`　　　　　　　　　　　　| ⚠️ **拆解方案已采用**：modeling 行 40-42 = `npu_rms_norm` + `F.silu(gate)*x`；融合 `rms_norm_gated` 未注册（拆解已可用）　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P2　　 | `cann_lm::rms_norm`　　　　　　　　　　　　　　　　　　　　　　 | `Qwen3_5RMSNorm` 纯 torch　　　　　　　　　　| ✅ **已落地+已用**：modeling 行 25/41 调用 `torch.ops.cann_lm.rms_norm`；cann_lm `_rms_norm_converter.py`→`ge.RmsNorm`（ge_converter.py:72-73）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P2　　 | `cann_lm::add_rms_norm`　　　　　　　　　　　　　　　　　　　　 | 残差 add + norm 两步　　　　　　　　　　　　 | ❌ **未落地**：无 add_rms_norm.py/converter；gemma4 显式拆分为 add+`cann_lm::rms_norm`（modeling_gemma4.py:308），qwen3.5 同设计　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| P3　　 | `cann_lm::fused_infer_attention_score_v2`　　　　　　　　　　　 | full attention eager　　　　　　　　　　　　 | ⚠️ **op 已落地**（ge_converter.py:81-82），modeling 行 215 有调用 `torch.ops.cann_lm.fused_infer_attention_score_v2`，但 OM 终态 FIA=0（待核实，见 §5.4）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P3　　 | `npu_swiglu`　　　　　　　　　　　　　　　　　　　　　　　　　　| `F.silu(gate) * up`　　　　　　　　　　　　　| ❌ 未落地（纯 torch 分解为 Swish/Sigmoid+Mul 已能导出，OM 命中 Swish=60/Sigmoid=24）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

> **P0 已解决**：`chunk_gated_delta_rule` 是 Qwen3.5 `export_to_epair` segfault 的解（纯 torch for
> 循环展开 19418 节点 → GE 图爆炸）。已按 ① `fused_ops/chunk_gated_delta_rule.py` 注册 schema+fake
> （`cann_lm::` 命名空间）+ CPU impl；② `_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule`
> （ge_converter.py:59-60，查 CANN `aic-ascend910_93-ops-info-*.json`）；③ modeling 行 621 调用
> `torch.ops.cann_lm.chunk_gated_delta_rule`。`recurrent_gated_delta_rule` 未注册（decode 走 chunk 共用）。
> 参考 cann-recipes-infer 的 `npu_chunk_gated_delta_rule` 签名（行 1541）对齐参数。

---

### 5.4 OM 算子表对比 support_ops（qwen3.5 无量化 prefill-decode, Ascend910_9362）

> qwen3.5-0.8B 无量化，attention **全 eager**（FIA decode 修复已验证但暂搁置：prefill 多 chunk 限制 + 方案 B 撞 torch.export 分支常量 lift 崩溃）。SoftplusV2 + L2Normalize 已落地。OM 扁平 so_store=0。

#### 5.4.1 fuse_op

| fuse_op | OM 算子 | decode | 命中 |
|---|---|---:|---|
| ChunkGatedDeltaRule | ChunkGatedDeltaRule | 18 | ✅ |
| Rmsnorm | RmsNorm | 79 | ✅ |
| FusedInferAttention | — | 0 | ❌ 全 eager（decode 修复已验证但暂搁置：prefill 多 chunk 限制）|
| l2norm | L2Normalize | 36 | ✅（cann_lm::l2norm → ge.L2Normalize）|
| RotaryPositionEmbedding | RotaryMul | 12 | ✅ |
| ScatterNdUpdate | ScatterNdUpdate | 12 | ✅ |

#### 5.4.2 quant_op（基线无量化）

| quant_op | OM | decode | 命中 |
|---|---|---:|---|
| DynamicQuant | — | 0 | 未量化 |
| QuantBatchMatmulV4 | — | 0 | 未量化 |

#### 5.4.3 single_op

| single_op | OM 算子 | decode | 命中 |
|---|---|---:|---|
| Cast | Cast | 208 | ✅ |
| Mul | Mul | 84 | ✅（含 Neg→Mul(-1.0) 替换）|
| splitV | SplitVD | 24 | ✅ 同族 |
| concatD | ConcatV2D | 67 | ✅ 同族 |
| Transpose | Transpose | 54 | ✅ |
| Conv2DV2 | Conv2D | 18 | ✅ 同族 |
| StridedSliceV2 | StridedSliceD | 61 | ✅ 同族 |
| Add | Add | 24 | ✅ |
| Swish | Swish | 60 | ✅ |
| GatherV2 | GatherV2 | 4 | ✅（含 Index→index_select→GatherV2 替换）|
| TransData | TransData | 36 | ✅ |
| Sigmoid | Sigmoid | 24 | ✅ |
| MatmulV3 | MatMul | 187 | ✅ 同族 |
| SoftplusV2 | SoftplusV2 | 18 | ✅（cann_lm converter → ge.SoftplusV2，需 reg_register.h 注册）|
| Gelu | — | 0 | ❌ 模型未用 |

#### 5.4.4 OM 框架/分解算子（不在 §1 范畴）

| OM 算子 | decode | 性质 | 可消除? |
|---|---:|---|---|
| Const | 341 | 权重/参数/内联 ge.Const（图基础设施）| ❌ |
| Data | 53 | 图输入（基础设施）| ❌ |
| NetOutput | 1 | 图输出（基础设施）| ❌ |
| Exp | 18 | 线性注意力 g 的 `A_log.exp()`（普通 exp，无 §1 对应）| ❌ g 计算固有 |
| BatchMatMul | 12 | eager attention 的 QK^T + @V（6 full-attn 层×2）| ⚠️ 重启用 FIA 可消除（但 prefill 多 chunk 限制）|
| SoftmaxV2 | 6 | eager attention 的 softmax（6 full-attn 层）| ⚠️ 同上 |
| Muls | 6 | eager attention 的 `*scaling` 标量乘（6 层）| ⚠️ FIA 用 softmax_scale attr 可消除 |

#### 5.4.5 结论

- **fuse_op**：5/8 命中（ChunkGatedDeltaRule / Rmsnorm / l2norm / RotaryPositionEmbedding / ScatterNdUpdate ✅；FusedInferAttention ❌ 全 eager；DynamicQuant / QuantBatchMatmulV4 ❌ 无量化）。
- **single_op**：14/15 命中（Gelu ❌ 模型未用；SoftplusV2 ✅ 已落地需 reg_register.h 注册；Neg→Mul / Index→GatherV2 已替换消除）。
- **框架算子**（Const / Data / NetOutput）：本就不属 §1 范畴。
- **eager 代价**：BatchMatMul ×12 + SoftmaxV2 ×6 + Muls ×6（共 24 个）是全 eager attention 的代价；重启用 FIA 可消除（decode 已验证，prefill 多 chunk 限制待解）。
- **分解基元**：Exp ×18 是线性注意力 g 计算固有（`A_log.exp()`），无 §1 对应，不可消除。
- **扁平无动态子图**：so_store=0、stream_num=1。

---

### 5.5 平台差异速查（Qwen3.5）

| 平台 | `npu_chunk_gated_delta_rule` | fp8/mxfp8 量化 | `npu_mm_all_reduce_base` | MoE MC2 |
|------|------------------------------|----------------|---------------------------|---------|
| **950**（Atlas 950） | ❌ 走纯 torch prefill | ✅ 支持 | ✅ 支持 | group_type=0/3 |
| **910/A3**（非 950） | ✅ 融合算子 | ❌ 不支持 | ❌ A3 不支持 | `fullmesh_v2` |

> 本机 NPU 为 **Ascend910_9362**（非 950），支持 `npu_chunk_gated_delta_rule` 融合算子（参考实现），
> 不支持 fp8/mxfp8 量化。**融合算子已落地**：910 实机推理 + export 都走 `cann_lm::chunk_gated_delta_rule`
> （segfault 已解决，OM 命中 18 个）；950 参考实现走纯 torch fallback，但 950 纯 torch 在导出时
> 仍会 segfault，需 950 专属解或保持 `ge.ChunkGatedDeltaRule` 映射让 950 也能用 GE 算子。
