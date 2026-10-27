# Fused Operators Reference (support_ops + compositte_op merged)

English | [简体中文](../../../../zh/cann_lm_converter/models/fused_ops/fused_ops.md)

> **Merged from** the original `cann_lm/models/fused_ops/support_ops.md` (operator registry / definition tracing / GE converter mapping / OM verification) and `compositte_op.md` (reference implementation survey / modeling replacement scheme / priority). The two docs revolve around the same set of fused operators; the former leans downstream (IR/OM verification), the latter upstream (reference impl + modeling replacement). After merging, organized as "operator overview table → definition tracing → GE mapping → per-model (reference→modeling→priority→OM verification) → platform differences".

> **Data sources**:
> - Reference implementation (CANN optimized): Gemma-4 `cann-recipes-infer/models/gemma_4/models/modeling_gemma4.py`; Qwen3.5 `cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py`
> - This repo's implementation: `cann_lm/models/gemma4/modeling_gemma4.py` / `cann_lm/models/qwen3_5/modeling_qwen3_5.py`
> - Custom operators: `cann_lm/models/fused_ops/` (`_library.py` uses `Library("cann_lm","FRAGMENT")` to register `cann_lm::` schema+fake+CPU impl) + GE mapping `cann_lm/converter/export/ge_converters/ge_converter.py` (registers `cann_lm::` converters, overriding torchair `custom/` npu:: converters)
> - Verification env: torch_npu 2.9.0.post8 (system Python, real NPU); converter venv (torchair source-built, torch_npu absent, goes through `_library.py` stub path).

The reference implementation is based on the CANN inference framework (TND/BNSD Paged Attention + MC2 EP + GE Graph) and heavily uses `torch_npu` / `torch.ops.npu` fused operators. **This repo has migrated to the `cann_lm::` namespace** (to avoid duplicate-registration conflicts with torch_npu, see `_rotary_mul_converter.py:18` comment "avoid torch_npu duplicate-registration"): `modeling_*.py` calls `torch.ops.cann_lm.*` fused operators (including l2norm), `cann_lm/models/fused_ops/_library.py` always registers `cann_lm::` schema+fake+CPU impl via `Library("cann_lm","FRAGMENT")` (`_HAS_TORCH_NPU=False` hardcoded, no torch_npu dependency), and FX→GE conversion is done by `cann_lm/converter/export/ge_converters/` built-in converters (registering `cann_lm::`, overriding torchair `custom/`'s npu:: converters, see §3). NPU acceleration goes through export→OM (GE conversion); eager goes through CPU impl.

## Table of Contents

1. [Operator Overview Table (fuse_op / quant_op / single_op)](#1-operator-overview-table-fuse_op-quant_op-single_op)
2. [Definition Tracing: torch_npu 2.9.0.post8 vs converter venv torchair](#2-definition-tracing-torch_npu-290post8-vs-converter-venv-torchair)
3. [torchair GE converter mapping](#3-torchair-ge-converter-mapping)
4. [Gemma-4 Fused Operators](#4-gemma-4-fused-operators) — 4.1 Reference implementation survey (15 ops) / 4.2 modeling replacement analysis / 4.3 Benefit ranking / 4.4 OM operator table comparison
5. [Qwen3.5 Fused Operators](#5-qwen35-fused-operators) — 5.1 Reference implementation survey (22 ops) / 5.2 modeling replacement analysis / 5.3 Benefit ranking / 5.4 OM operator table comparison / 5.5 platform differences quick reference

---

<a id="1-operator-overview-table-fuse_op-quant_op-single_op"></a>
## 1. Operator Overview Table (fuse_op / quant_op / single_op)

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

<a id="2-definition-tracing-torch_npu-290post8-vs-converter-venv-torchair"></a>
## 2. Definition Tracing: torch_npu 2.9.0.post8 vs converter venv torchair

### Fused Operator Definition Location Analysis

> Scope of this section: the original 6 fused operators (ChunkGatedDeltaRule / Rmsnorm / FusedInferAttention / l2norm / RotaryPositionEmbedding / ScatterNdUpdate).
> DynamicQuant / QuantBatchMatmulV4 are newly categorized by the user into fuse_op, corresponding to `npu_dynamic_quant` / `quant_matmul`, not in this verification scope.
> Verified versions: torch_npu 2.9.0.post8 (system Python `$DOPT_PYTHON`); converter venv (`$TORCH_DELEGATE_HOME/.venv`, torch_npu absent, torchair source-built).
> Line numbers are version snapshots at verification time and may drift after upgrades; refer to file paths as authoritative.

### 2.1 Definitions under torch_npu 2.9.0.post8 (system Python)

A total of 354 `npu::` operators registered. The 6 fused operators hit status:

| fuse_op                 | npu:: registration　　　　　　　　　　　　　　　　　 | schema source　 | Meta definition　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE raw op　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| --------------------------------------------------| ---------------| -----------------------------------------------------------------------------| ---------------------------------------------------------------------|
| ChunkGatedDeltaRule     | ❌ Not registered (needs torch_npu ≥2.9.0.post8)　　　　　　| none　　　　　　| none　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| none　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| Rmsnorm                 | `npu_rms_norm`　　　　　　　　　　　　　　　　　 | C++ `_C` ext | `op_plugin/meta/_meta_registrations.py` (`npu_rms_norm_meta`)　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(RmsNorm)`　　　　　　　　　　|
| FusedInferAttention     | `npu_fused_infer_attention_score` + `_v2`　　　　| C++ `_C` ext | `_meta_registrations.py` (`npu_fused_infer_attention_score`/`_v2_forward`) | —　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| l2norm                  | ❌ Not a torch_npu custom operator　　　　　　　　　　　　| none　　　　　　| none　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(L2Normalize)` (GE IR level only) |
| RotaryPositionEmbedding | `npu_rotary_mul` (+ `npu_apply_rotary_pos_emb`) | C++ `_C` ext | `_meta_registrations.py` (`npu_rotary_mul_meta`)　　　　　　　　　　　　　 | `auto_generated_ge_raw_ops.py` `REG_OP(RotaryMul)`　　　　　　　　　|
| ScatterNdUpdate         | `npu_scatter_nd_update` + `_`　　　　　　　　　　| C++ `_C` ext | `_meta_registrations.py` (`scatter_nd_update_meta`)　　　　　　　　　　　　| `auto_generated_ge_raw_ops.py` `REG_OP(ScatterNdUpdate)`　　　　　　　|

Registration mapping in `torch_npu/utils/custom_ops.py` (`npu_rotary_mul`/`npu_rms_norm`/`npu_fused_infer_attention_score[_v2]`/`npu_scatter_nd_update`).

### 2.2 torchair definitions under converter venv

torch_npu package absent; torchair present (`$TORCH_DELEGATE_HOME/.venv/lib/python3.11/site-packages/torchair`).

> **Migration footnote**: The table below is the current state of torchair `custom/`'s npu:: converters (shipped with torchair, describing the npu:: schema torchair registers). This repo's modeling has migrated to `cann_lm::` (see §2.3); FX graph hits `cann_lm::` (not npu::), so the npu:: converters in the table below don't hit, covered by cann_lm's own `ge_converters/` (see §3). The table is retained as a torchair reference and historical tracing.

| fuse_op                 | venv torchair definition file　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | type　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | self-contained (no torch_npu dependency)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------|
| ChunkGatedDeltaRule     | `_ge_concrete_graph/ge_converter/custom/npu_chunk_gated_delta_rule.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ References `torch.ops.npu.npu_chunk_gated_delta_rule.default`, and the body directly `raise RuntimeError("not supported in GE graph mode")` (stub) |
| Rmsnorm                 | `_ge_concrete_graph/ge_converter/custom/npu_rms_norm.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ References `torch.ops.npu.npu_rms_norm`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| FusedInferAttention     | `ops/_npu_fused_infer_attention_score.py` + `_v2.py` (air::) **and** `ge_converter/custom/fused_infer_attention_score.py` + `_v2.py` (npu::) | air:: self-contained `lib.define` schema+Meta (`torch.library.Library("air","FRAGMENT")`); npu:: is converter only | ⚠️ schema+Meta self-contained (can trace), but file top still `import torch_npu` (converter part needs it)　　　　　　　　　　　　　　　　　　　　　　　　|
| l2norm                  | none (only `auto_generated_ge_raw_ops.py` has GE raw op `L2Normalize`)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | GE IR-level REG_OP　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ No schema/Meta/converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| RotaryPositionEmbedding | `_ge_concrete_graph/ge_converter/custom/npu_rotary_mul.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ References `torch.ops.npu.npu_rotary_mul`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ScatterNdUpdate         | `_ge_concrete_graph/ge_converter/custom/npu_scatter_nd_update.py`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| npu:: FX→GE converter　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| ❌ References `torch.ops.npu.npu_scatter_nd_update`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

### 2.3 Key Conclusions

- In venv torchair, the 6 fused operators' `custom/` converters are registered to `npu::` (hit `torch.ops.npu.*`). **But this repo's modeling calls `torch.ops.cann_lm.*` (not npu::), so the FX graph hits `cann_lm::`**, thus torchair `custom/`'s npu:: converters don't hit; covered by CANN-LM's own `cann_lm/converter/export/ge_converters/` (`ge_converter.py:22` imports cann_lm ops to define `cann_lm::` schema, `:33` imports torchair `custom/` to register npu:: converters as fallback, `:44/54/60/69-70/73/76/82` use `register_fx_node_ge_converter(torch.ops.cann_lm.xxx.default)` to register cann_lm:: converters). `_library.py` uses `NPU_LIB = Library("cann_lm","FRAGMENT")` + `safe_define` to custom-define cann_lm:: schema (see `rms_norm.py:27`/`rotary_mul.py:27`, etc.).
- **The only self-contained schema+Meta is FusedInferAttention's air:: version** (`ops/_npu_fused_infer_attention_score_v2.py` uses `torch.library.Library("air","FRAGMENT")` + `lib.define`), but its converter part still `import torch_npu`.
- **ChunkGatedDeltaRule**'s converter is a stub (`raise RuntimeError`), and the schema needs torch_npu ≥2.9.0.post8 (both missing in venv).
- **l2norm** has no definition in torchair at all, only the GE raw op `L2Normalize`.
- **Root cause**: CANN-LM `cann_lm/models/fused_ops/_library.py` always registers `cann_lm::` schema+fake+CPU impl (`_HAS_TORCH_NPU=False` hardcoded), enabling `torch.export` to trace; GE conversion is done by built-in `ge_converters/` hitting `cann_lm::` (overriding torchair npu:: converters). See [cli_design.md](../../cli/cli_design.md) "Relationship between converter and torch_npu".

---

<a id="3-torchair-ge-converter-mapping"></a>
## 3. torchair GE converter mapping

> **FX→GE conversion is now done by CANN-LM's own `cann_lm/converter/export/ge_converters/`** (`ge_converter.py` uses `register_fx_node_ge_converter(torch.ops.cann_lm.xxx.default)` to register `cann_lm::` converters, hitting `torch.ops.cann_lm.*`). torchair `custom/`'s npu:: converters serve only as fallback/reference (modeling doesn't call npu::, so they don't hit). The table below lists each operator's converter registration schema, converter function, and produced GE op; cann_lm built-in converter files are abbreviated as `ge_converters/_<file>`, torchair reference as `custom/<file>`.
> Hit rule: FX graph `torch.ops.cann_lm.*` → goes through `ge_converters/` built-in version (cann_lm::); `torch.ops.npu.*`/`torch.ops.air.*` → goes through torchair `custom/`/`ops/` version (not hit by this repo's modeling).

### 3.1 fuse_op

| fuse_op | converter file | registration schema (`@register_fx_node_ge_converter`) | converter function | produced GE op | notes |
|---|---|---|---|---|---|
| ChunkGatedDeltaRule | `custom/npu_chunk_gated_delta_rule.py` | `npu.npu_chunk_gated_delta_rule.default` | `convert_npu_chunk_gated_delta_rule(q,k,v,*,beta?,initial_state?,actual_seq_lengths?,scale?,g?)` | ❌ Stub `raise RuntimeError("...not supported in GE graph mode")` | Not supported in GE graph mode; schema needs torch_npu ≥2.9.0.post8. cann_lm built-in `_chunk_gated_delta_rule_converter.py` overrides (→`ge.ChunkGatedDeltaRule`) |
| Rmsnorm | `custom/npu_rms_norm.py` | `npu.npu_rms_norm.default` | `conveter_npu_rms_norm_default(self,gamma,epsilon=1e-6)` | `ge.RmsNorm(self,gamma,epsilon=)` (1 op) | `@declare_supported` (F16/F32) |
| FusedInferAttention | `custom/fused_infer_attention_score_v2.py` (npu::) | `npu.npu_fused_infer_attention_score_v2.default` | `convert_npu_npu_fused_infer_attention_score_v2(q,k,v,*,query_rope?,key_rope?,pse_shift?,atten_mask?,actual_seq_qlen?,actual_seq_kvlen?,block_table?,dequant_scale_*?,quant_scale_*?,learnable_sink?,input_layout,sparse_mode,block_size,...)` | `ge.FusedInferAttentionScore(...)` → `(out,lse)` | air:: version in `torchair/ops/_npu_fused_infer_attention_score_v2.py` (self-contained schema+Meta) |
| l2norm | — | — | — | GE raw op `L2Normalize` (IR-level REG_OP only) | torchair has no converter/schema/Meta; cann_lm built-in `_convert_l2norm` (→`ge.L2Normalize`) |
| RotaryPositionEmbedding | `custom/npu_rotary_mul.py` | `npu.npu_rotary_mul.default` | `conveter_npu_rotary_mul_default(self,r1,r2,rotary_mode='half',rotate?)` | default `ge.RotaryMul(self,r1,r2)`; arch35 or with `rotate` → `ge.RotaryPositionEmbedding(self,r1,r2,mode=,rotate=)` | gemma4 goes through default `ge.RotaryMul` branch (⚠️ OM final state see §4.4 "RoPE note") |
| ScatterNdUpdate | `custom/npu_scatter_nd_update.py` | `npu.npu_scatter_nd_update.default` + `npu.npu_scatter_nd_update_.default` | out-of-place / in-place two versions | out-of-place: `ge.TensorMove(self)`+`ge.ScatterNdUpdate(copy,idx,upd)` (2 ops); in-place: `ge.ScatterNdUpdate(self,idx,upd)`+`dont_prune_me` (1 op) | ⚠️ cann_lm already has built-in `converter/export/ge_converters/_scatter_nd_update_converter.py` overriding this converter (produces same `ge.ScatterNdUpdate`, controllable behavior); gemma4 already landed (`scatter_kv_to_cache` constructs multi-dim indices to fit standard semantics) |

### 3.2 quant_op

| quant_op | converter file | registration schema | produced GE op | notes |
|---|---|---|---|---|
| DynamicQuant | `custom/npu_dynamic_quant.py` | `npu.npu_dynamic_quant.default` + `npu.npu_dynamic_quant_asymmetric.default` | `quant_mode=="pertoken" && dst_type_max≈0`: `ge.DynamicQuant(...)`→`(y,scale)`; others: `ge.DynamicQuantV2(...)`→`(y,scale,offset)`; int4(dst_type==16) extra Const/Shape/Div/ConcatV2/Bitcast/Reshape bitcast subgraph | `@declare_supported` (F16/BF16) |
| QuantBatchMatmulV4 | `custom/quantmatmul.py` | `npu.npu_quant_matmul.default` | main: `ge.QuantBatchMatmulV4(...)`; alt: `ge.QuantBatchMatmulV3(...)`; includes int4/fp4 bitcast+Transpose+Reshape subgraph | ⚠️ cann_lm actually uses its **own converter** `cann_lm/converter/export/ge_converters/_quant_matmul_converter.py` (removes torch_npu dependency, drops a8w4 branch), doesn't hit this torchair converter |

### 3.3 Key Notes

- **converter function name typo**: torchair spells most converter function names as `conveter_*` (missing an r, e.g. `conveter_npu_rms_norm_default`); this is a historical torchair typo, not this repo's issue; quote the original spelling.
- **GE op name vs OM op name**: the GE op produced by the converter (e.g. `ge.ScatterNdUpdate`/`ge.RmsNorm`) is the IR-stage name; after OM compilation fusion optimization the op name changes (`ScatterNdUpdate`/`InplaceUpdate+TensorMove` → OM `ScatterUpdate`+`MemcpyAddrAsync`; `RmsNorm`+residual Add → OM `InplaceAddRmsNorm`). See AGENTS.md "Viewing OM Model Operators".
- **converter venv hit status**: converter venv doesn't install torch_npu; modeling calls `torch.ops.cann_lm.*` (cann_lm::), FX graph **hits cann_lm:: converters** (`ge_converters/` built-in, not torchair npu::). cann_lm built-in converters (`_rms_norm_converter.py`/`_rotary_mul_converter.py`/`_scatter_nd_update_converter.py`/`_chunk_gated_delta_rule_converter.py`/`_quant_matmul_converter.py`/`_fused_infer_attention_score_v2_converter.py`) have the same main logic as torchair `custom/`'s npu:: versions (directly emit `ge.*` op, don't import torch_npu); differences are only in registration namespace (cann_lm:: vs npu::) and op name (drop `npu_` prefix). schema is provided by `_library.py` `Library("cann_lm","FRAGMENT")` + `safe_define`, no duplicate-registration conflict with torch_npu.

---

<a id="4-gemma-4-fused-operators"></a>
## 4. Gemma-4 Fused Operators

### 4.1 Reference implementation (cann-recipes-infer) torch_npu operators survey

A total of **15** fused operators, grouped into 5 categories by function domain.

#### 4.1.1 Normalization (RMSNorm)

| #   | operator　　　　　　　　　　　　 | call site　　　　　　　　　　| key params　　　　　　　　　　　　　　　　　　　　　　　　 | purpose　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| -----| ------------------------------| -----------------------------| ----------------------------------------------------------| ---------------------------------------------------------------------------------------------------|
| 1   | `torch_npu.npu_rms_norm`　　 | `Gemma4RMSNorm.forward`　　 | `(x, weight, eps)` returns `[normed, ...]`　　　　　　　　　| Fused RMSNorm: `x * rsqrt(mean(x²)+eps) * weight`, replaces pow/mean/mul multi-step　　　　　　　　　　　　　|
| 2   | `torch_npu.npu_add_rms_norm` | `Gemma4RMSNorm.forward_add` | `(x1, x2, weight, eps)` returns `[normed, _, residual_sum]` | **Fused add + RMSNorm**: computes `rms_norm(x1+x2)` and `x1+x2` in one op, saves one elementwise add + one norm |

#### 4.1.2 Activation

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 3 | `torch_npu.npu_fast_gelu` | `Gemma4MLP.forward`, `_GegluMoEMethod.apply` | `(x)` | Fused `gelu(approximate='tanh')`, graph-safe |

#### 4.1.3 RoPE Rotary Position Embedding

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 4 | `torch_npu.npu_rotary_mul` | `Gemma4Attention.forward` (sliding layers) | `(q4, cos4, sin4, rotary_mode='half')`, needs 4D BSND input | Fused rotate-half RoPE, for sliding-layer full head_dim rotation |
| 5 | `torch_npu.npu_apply_rotary_pos_emb` | `Gemma4Attention.forward` (full layers) | `(q_rot, k_rot, cos, sin, layout="TND")` | Fused partial-rotary RoPE (full layers rotate only rotary_dim), supports TND layout |

#### 4.1.4 Attention (FlashAttention + PagedAttention)

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 6 | `torch.ops.npu.npu_fused_infer_attention_score_v2` (ge_graph decode via `torchair.ops`) | `_attn_tnd_sliding`, `_attn_bnsd_full` | `input_layout="TND"/"BNSD"`, `sparse_mode`, `pre_tokens`, `block_table`, `block_size`, `num_key_value_heads` (GQA) | Fused QK^T + softmax + AV + PagedAttention; sliding layers use TND(D=256)+sparse_mode=4, full layers use BNSD(D=512)+sparse_mode=3/0 |
| 7 | `torch_npu.npu_scatter_nd_update_` | `Gemma4Attention._update_cache` | `(cache.view(-1,N,D), slot, kv)` in-place | Fused KV cache scatter-write (write into PagedAttention cache by slot_mapping) |

#### 4.1.5 MoE (this repo's Gemma-4 is currently pure dense MLP, no MoE)

The reference implementation has 8 MoE operators: `npu_moe_gating_top_k_softmax` (top-k+softmax routing), `npu_grouped_matmul` (grouped GEMM), `npu_geglu` (fused GEGLU activation, prefill/eager only; graph decode degrades to manual `npu_fast_gelu*up`), `npu_moe_init_routing_v2`/`npu_moe_finalize_routing`/`npu_moe_re_routing` (routing init/finalize/EP secondary re-rank), `npu_moe_distribute_dispatch_v2`/`npu_moe_distribute_combine_v2` (MC2 EP dispatch/combine). This repo's `modeling_gemma4.py` is pure dense MLP (`Gemma4TextMLP`, no `Gemma4SparseMoeBlock`), **none applicable**; if MoE is introduced later, can align with the reference impl `Gemma4SparseMoeBlock`'s three routing modes (TP / EP double-routing / MC2 dispatch-combine).

---

### 4.2 Available fused operators analysis for this repo's `modeling_gemma4.py`

This repo's implementation calls `torch.ops.npu.*` (+ `torch.ops.cann_lm::l2norm`) fused operators; below gives per-module "current implementation → fused operator" mapping. Marked `✅ landed+used` / `✅ directly replaceable` / `❌ not adopted` / `❌ not landed`.

#### 4.2.1 RMSNorm → `npu_rms_norm` (landed+used) / `npu_add_rms_norm` (not adopted)

**Current implementation** (`Gemma4RMSNorm.forward`, modeling_gemma4.py:301-303, the pure-torch `_norm` impl is removed):
```python
def forward(self, x):
    y, _ = torch.ops.cann_lm.rms_norm(x, self.weight, epsilon=self.eps)
    return y
```

> The pure-torch `pow(2).mean + pow(-0.5) + mul` `_norm` shown here in the original doc is deleted (modeling_gemma4.py:288 docstring "via cann_lm::rms_norm (fused)"). The `with_scale=False` subclass (v_norm/router) passes in pre-allocated ones weight via `register_buffer("weight", ones)`.

| operator | applicability | notes |
|------|--------|------|
| `cann_lm::rms_norm` | ✅ landed+used | modeling line 302/311 already calls `torch.ops.cann_lm.rms_norm` (`Gemma4RMSNorm.forward` / `forward_add`). When `with_scale=False`, passes ones weight (v_norm/router norm does this; this repo's `with_scale=False` uses `register_buffer("weight", ones)` pre-allocated, modeling_gemma4.py:299) |
| `torch_npu.npu_add_rms_norm` | ❌ not adopted (design decision) | Fuses residual add + rms_norm. **This repo explicitly splits into add + `npu_rms_norm` two steps** (`forward_add`, modeling_gemma4.py:305-312, comment "originally used fused npu_add_rms_norm, now split into add + npu_rms_norm (user requirement)"). The reference impl `Gemma4DecoderLayer` uses the fused version; this repo splits by design decision; the fusion suggestion in §4.2.2 below is rejected by this decision, retained only as design analysis |

#### 4.2.2 Residual Add + Norm topology (`forward_add` split implementation)

**Current implementation** (`Gemma4TextDecoderLayer.forward`, modeling_gemma4.py:487-494): residual add and norm are two steps (`forward_add` split into add + `npu_rms_norm`).
```python
hidden_states = self.post_attention_layernorm(hidden_states)   # norm(attn_out)
hidden_states = residual + hidden_states                        # add (standalone statement)
...
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)   # norm(add result)
```

> The reference impl `Gemma4DecoderLayer.forward` (line 862) uses `npu_add_rms_norm` to merge add+norm into a single op; this repo splits by design decision (see §4.2.1 table "not adopted").

#### 4.2.3 GELU(tanh) → `npu_fast_gelu`

**Current implementation**:
- `Gemma4TextMLP.__init__` (line 272): `self.act_fn = nn.GELU(approximate="tanh")`
- `Gemma4TextDecoderLayer.forward` per_layer_input (line 499): `F.gelu(gated, approximate="tanh")`

| operator | applicability | notes |
|------|--------|------|
| `torch_npu.npu_fast_gelu` | ✅ directly replaceable | The reference impl comment explicitly states `npu_fast_gelu == gelu(approximate='tanh')`, graph-safe. MLP's `act_fn(gate)*up` pattern matches the reference impl `Gemma4MLP.forward` |

#### 4.2.4 RoPE → `npu_rotary_mul` (`npu_apply_rotary_pos_emb` unused)

**Landed** (modeling_gemma4.py): consistent with the Turing reference impl (cann-recipes-infer), two paths by layer_type:
- **sliding layers** (full rotation, `head_dim == cos.shape[-1]`): `npu_rotary_mul` single tensor, q/k each called once (`rotary_mode='half'`).
- **full layers** (proportional partial, `rotary_dim == cos.shape[-1] < head_dim`): cat out q_rot/k_rot (front `pair_dim` + middle `half_dim:half_dim+pair_dim`), **`npu_rotary_mul` q/k each called once** (`rotary_mode='half'`, not `npu_apply_rotary_pos_emb`), split and insert back (middle and rear keep original values). See `_rope_full_pair` (modeling_gemma4.py:119-136).
- **kv_shared path** (q alone, k from shared already RoPE'd): uses `apply_rotary_pos_emb_single` (`npu_rotary_mul` + cat/split to simulate partial), to avoid re-rotating the already-RoPE'd k.
- `inv_freq` no longer zero-padded for proportional (length = `rope_angles = rotary_dim//2`), `cos/sin` length = `rotary_dim`, consumed by full-layer cat/split (consistent with reference impl; the old "inv_freq zero-pad + full head_dim through rotary_mul" path is deprecated).

| operator | status | notes |
|------|------|------|
| `cann_lm::rotary_mul` | ✅ landed+used | sliding layers (full rotation) single tensor; full layers + kv_shared path (cat/split simulating partial). GE converter in cann_lm `_rotary_mul_converter.py`→`ge.RotaryMul`. modeling lines 107/114/115/126/127/145 call `torch.ops.cann_lm.rotary_mul` |
| `torch_npu.npu_apply_rotary_pos_emb` (q+k paired) | ❌ unused | full layers use `npu_rotary_mul` + cat/split (`_rope_full_pair`), **does not call** `npu_apply_rotary_pos_emb`. The original doc's "landed" is stale |

> ⚠️ `rope_in_place` / `ge.RopeInPlace` **not landed**: `fused_ops/ops.py` not registered, `ge_converter.py` has no corresponding converter; the original doc's statement here is stale.
>
> ⚠️ RoPE OM final state differs from IR (to verify): IR-level converter default produces `ge.RotaryMul`, but OM final state RoPE op is `RotaryPositionEmbedding` (gemma4 prefill/decode each 50, not `RotaryMul`). See §4.4 "RoPE note" — suspected 950DT recognized as arch35 or hitting the `rotate` branch.

#### 4.2.5 Eager Attention → `npu_fused_infer_attention_score_v2`

**Landed** (modeling_gemma4.py): consistent with the Turing reference impl, `fused_attention_forward` calls
`npu_fused_infer_attention_score_v2` (BNSD layout) by `layer_type`, fusing QK^T+softmax+AV+GQA+causal/sliding.
- **sliding layers** (full rotation head_dim=256): `sparse_mode=4` (sliding_window) + `pre_tokens=sliding_window`.
- **full layers** (global_head_dim=512): `sparse_mode=3` (causal) + `pre_tokens=INT_MAX` (full attention).
- **GQA native**: `num_query_heads` vs `num_key_value_heads` (eliminates `repeat_kv`; the original 35 layers×2=70 `ConcatV2D` go to zero).
- **mask adaptation**: cann_lm's float mask `[*, q_len, kv_len]` → fused op requires bool `[*, kv_len, kv_len]` square matrix
  (`_pad_mask_to_kv_square` pads q dim to kv_len; the fused op uses `actual_seq_qlen` to slice the first q_len rows); `actual_seq_qlen/kvlen` uses int64.
- **Retained**: `eager_attention_forward` + `repeat_kv` + `_prepare_causal_mask` (not deleted, just no longer called, as fallback/reference).

| operator | status | notes |
|------|--------|------|
| `cann_lm::fused_infer_attention_score_v2` | ✅ landed | cann_lm `fused_ops/fused_infer_attention_score.py` registers schema+fake+CPU eager (`cann_lm::` namespace); `ge_converters/_fused_infer_attention_score_v2_converter.py` registers converter (calls `ge.FusedInferAttentionScore`, all 40+ params passed, None for unused); `reg_register.h` adds `REG_OP(FusedInferAttentionScore)` (OM compile registers to GE OperatorFactory). Eliminates `BatchMatMul`/`SoftmaxV2` (to zero), `ConcatV2D` from 14 to 9 (repeat_kv elimination) |
| `cann_lm::fused_infer_attention_score` | ❌ not adopted (originally proposed in doc) | The cann_lm custom FA op proposed in the original doc (IR-level fallback); actually uses the npu namespace (consistent with Turing), not the cann_lm namespace |

> softcap part: the reference impl absorbs `1/sqrt(d)` into q_norm (`scale_fa=1.0`); this repo's `scaling=1.0`
> is already consistent; logit softcap (`x/cap→tanh→x*cap`) has no single fused op; `fused_attention_forward`
> doesn't pass softcap (gemma4 attention has no softcap; logit softcap is at the end of `Gemma4ForCausalLM`,
> not inside attention).

##### Legacy issue: support scheme when fused op introduces weights (current gemma4 doesn't involve; quantization/sink scenarios need attention)

Currently `FusedInferAttentionScore` in gemma4 has **no weight params** (only receives Q/K/V + config constants), so eager→fused switch can reuse the old `weights_0.bin` (q/k/v/o_proj weight unchanged). **Legacy scenario**: if the fused op introduces learnable weights (quantization `antiquant_scale`/`antiquant_offset`/`dequant_scale_query`/`dequant_scale_key`/`dequant_scale_value`, or attention sink `learnable_sink`), then the old `.bin` cannot be reused (new weights have no index; OM compile Const can't find them). **Support scheme**: expose new weights as `nn.Parameter` (same level as q_proj); `torch.export` + `consolidated_weights_save` auto lift+persist, no need to manually manage `weights_0.bin` offsets — needs sync: ① `Attention.__init__` add `nn.Parameter`; ② `load_gemma4_weights` add HF→cann_lm name mapping; ③ `fused_attention_forward` pass new weight; ④ `fused_infer_attention_score.py` schema add optional input; ⑤ `ge_converter.py` converter pass non-None (`REG_OP(FusedInferAttentionScore)` already has `learnable_sink` etc. OPTIONAL_INPUT); ⑥ `--save-weights=True` re-persist (**cannot reuse old .bin**). **Trigger condition**: when `fused_attention_forward` passes non-None `learnable_sink`/`antiquant_scale`/`dequant_scale_*` etc. to the fused op; current gemma4 (pure bf16, no quant/sink) passes all None, doesn't trigger.

#### 4.2.6 KV Cache Write → `npu_scatter_nd_update_`

**Landed implementation** (`Gemma4TextAttention.forward` + `scatter_kv_to_cache` helper, modeling_gemma4.py):
```python
cache_k = scatter_kv_to_cache(cache_k, new_kv_cache_pos, key_states)
cache_v = scatter_kv_to_cache(cache_v, new_kv_cache_pos, value_states)
```
`scatter_kv_to_cache` constructs `[b*n*seq, 3]` multi-dim coordinate indices (b,n,kv_pos) + `[b*n*seq, d]` updates, hitting `ge.ScatterNdUpdate` standard multi-dim scatter semantics (equivalent to `index_copy(2, kv_pos, kv_new)`, CPU verified diff=0).

| operator | applicability | notes |
|------|--------|------|
| `torch_npu.npu_scatter_nd_update_` | ✅ landed | cann_lm built-in converter (`converter/export/ge_converters/_scatter_nd_update_converter.py`) overrides torchair, directly producing `ge.ScatterNdUpdate` (not going through `aten.index_copy`'s `InplaceUpdate+Transpose`). gemma4 via `scatter_kv_to_cache` constructs multi-dim indices to fit standard semantics. CPU else branch consistent with GE `ge.ScatterNdUpdate` semantics; export trace hits `npu_scatter_nd_update_` |

#### 4.2.7 MoE Operators (not applicable)

This repo's `modeling_gemma4.py` is currently **pure dense MLP** (`Gemma4TextMLP`, no `Gemma4SparseMoeBlock`),
so the reference impl's 8 MoE operators (`npu_moe_gating_top_k_softmax`, `npu_grouped_matmul`, `npu_geglu`,
`npu_moe_init_routing_v2`, `npu_moe_finalize_routing`, `npu_moe_re_routing`,
`npu_moe_distribute_dispatch_v2`, `npu_moe_distribute_combine_v2`) **are all inapplicable**. If this repo
later introduces MoE, can align one-by-one with the reference impl's `Gemma4SparseMoeBlock` three routing
modes (TP / EP double-routing / MC2 dispatch-combine).

---

### 4.3 Benefit Ranking and Recommended Landing Order

Ranked by "single forward call frequency × per-op benefit":

| Priority | Fused operator | Replacement target | Status/expected benefit |
|--------|----------|----------|----------|
| P0 | `cann_lm::fused_infer_attention_score_v2` | `eager_attention_forward` full flow | ✅ **landed+used**: modeling:259 calls `torch.ops.cann_lm.fused_infer_attention_score_v2`; eliminates repeat_kv+2×matmul+softmax+mask construction, single-layer max kernel fusion, all model N layers |
| P0 | `cann_lm::rms_norm` | `Gemma4RMSNorm` (original `_norm` pure torch deleted) | ✅ **landed+used**: modeling:296/305 calls `torch.ops.cann_lm.rms_norm`; each layer has 4-6 RMSNorms (input/post_attn/pre_ffn/post_ffn + q/k/v_norm), high frequency |
| P1 | `npu_fast_gelu` | `nn.GELU(tanh)` / `F.gelu(tanh)` | ❌ not landed: modeling still uses `nn.GELU`/`F.gelu` (lines 438/671); each layer MLP + per_layer_input once each |
| P1 | `npu_add_rms_norm` | residual add + next norm | ❌ not adopted (design decision): split into add + `npu_rms_norm` (modeling_gemma4.py:308 "user requirement") |
| P2 | `cann_lm::rotary_mul` (landed) | RoPE pairing + cat/split | ✅ **landed+used**: sliding/full/kv_shared all go through `torch.ops.cann_lm.rotary_mul` (modeling:107/114/115/126/127/145); `npu_apply_rotary_pos_emb` unused |
| P2 | `cann_lm::scatter_nd_update_` (landed) | `index_copy` KV cache write | ✅ **landed+used**: modeling uses `scatter_kv_to_cache` (multi-dim indices) + cann_lm `ge.ScatterNdUpdate` converter |

> Note: this repo is currently a pure PyTorch reference impl; introducing `torch_npu` must follow AGENTS.md's
> stub convention (converter-side torchair build + lib-layer stubbing); NPU real-machine verification goes
> through system Python's torch_npu.

---

### 4.4 OM operator table vs support_ops (gemma4 no-quant prefill-decode, Ascend950DT_9575)

> Export `epair_gemma4_no_quant` (prefill-decode, seq_len=128 / decode=1) → C++ `export_llm_model` compiled to OM (`SOC_VERSION=Ascend950DT_9575`, fusion cfg disables AUB + AddRmsNorm/InplaceAddRmsNorm). The table below is OM final-state operators vs this doc's §1 definitions.

#### 4.4.1 fuse_op

| fuse_op | OM op | prefill | decode | hit |
|---|---|---:|---:|---|
| ChunkGatedDeltaRule | — | 0 | 0 | Model architecture unused (qwen3.5 SSM only)|
| Rmsnorm | RmsNorm | 242 | 242 | ✅ |
| FusedInferAttention | FusedInferAttentionScore | 35 | 35 | ✅ (1 per layer)|
| l2norm (L2Normalize) | — | 0 | 0 | Model architecture unused |
| RotaryPositionEmbedding | RotaryPositionEmbedding | 50 | 50 | ⚠️ hit (see "RoPE note" below)|
| ScatterNdUpdate | ScatterNdUpdate | 30 | 30 | ✅ |

#### 4.4.2 quant_op (baseline no quantization)

| quant_op | OM | prefill | decode | hit |
|---|---|---:|---:|---|
| DynamicQuant | — | 0 | 0 | Not quantized |
| QuantBatchMatmulV4 | — | 0 | 0 | Not quantized (goes MatMulV3)|

#### 4.4.3 single_op

| single_op | OM op | prefill | decode | hit |
|---|---|---:|---:|---|
| Cast | Cast | 66 | 66 | ✅ |
| Mul | Mul | 105 | 105 | ✅ |
| splitV | — | 0 | 0 | Unused (StridedSliceV2 covers it)|
| concatD | ConcatD | 85 | 85 | ✅ |
| Transpose | Transpose | 100 | 0 | ✅ (prefill only; decode seq=1 doesn't need it)|
| Conv2DV2 | — | 0 | 0 | gemma4 has no conv1d |
| StridedSliceV2 | StridedSliceV2 | 60 | 60 | ✅ |
| Add | Add | 106 | 106 | ✅ |
| Swish | — | 0 | 0 | gemma4 uses Gelu |
| GatherV2 | GatherV2 | 40 | 40 | ✅ |
| TransData | — | 0 | 0 | Unused |
| Sigmoid | — | 0 | 0 | Unused |
| MatmulV3 | MatMulV3 | 277 | 277 | ✅ |
| SoftplusV2 | — | 0 | 0 | Unused |
| Gelu | Gelu | 70 | 70 | ✅ |

#### 4.4.4 OM framework/fusion variant operators (out of §1 scope)

| OM op | prefill | decode | nature |
|---|---:|---:|---|
| Const | 561 | 559 | Weights/constants (graph infrastructure)|
| Data | 77 | 77 | Graph inputs (infrastructure)|
| NetOutput | 1 | 1 | Graph output (infrastructure)|
| MemcpyAddrAsync | 40 | 40 | Shared-layer KV passthrough (CANN infrastructure op)|
| Muls | 2 | 2 | Scalar mul (Mul scalar variant, framework op, retained)|

#### 4.4.5 Final OM operator table

**prefill (1947 ops / 18 types)**: Const 561 · MatMulV3 277 · RmsNorm 242 · Add 106 · Mul 105 · Transpose 100 · ConcatD 85 · Data 77 · Gelu 70 · Cast 66 · StridedSliceV2 60 · RotaryPositionEmbedding 50 · MemcpyAddrAsync 40 · GatherV2 40 · FusedInferAttentionScore 35 · ScatterNdUpdate 30 · Muls 2 · NetOutput 1.

**decode (1845 ops / 17 types)**: same as prefill but no Transpose (seq=1 doesn't need transpose), Const 559.

#### 4.4.6 Conclusions

- **fuse_op**: 4/6 hit (Rmsnorm / FusedInferAttention / RotaryPositionEmbedding / ScatterNdUpdate); 2 unused are model architecture differences (ChunkGatedDeltaRule SSM, l2norm).
- **single_op**: 9/14 hit; 5 unused are architecture differences (splitV / Conv2DV2 / Swish / TransData / Sigmoid / SoftplusV2).
- **quant_op**: 0/2 (baseline no quantization; DynamicQuant / QuantBatchMatmulV4 both 0).
- **Framework ops** (Const / Data / NetOutput / MemcpyAddrAsync / Muls): inherently out of §1 scope.
- **Uncovered operators all eliminated**: `InplaceAddRmsNorm` = 0 (disabled `AddRmsNormFusionGraphPass` / `InplaceAddRmsNormFusionPass`, §7); no `PartitionedCall` (actual_seq inlined as real `ge.Const`, not into dynamic subgraph).

#### ⚠️ RoPE note

§3.1 table notes "gemma4 goes through default `ge.RotaryMul` branch", but measured OM final state RoPE op is `RotaryPositionEmbedding` (50, not `RotaryMul`). Per torchair `npu_rotary_mul` converter, `RotaryPositionEmbedding` is produced only on arch35 or with `rotate` — currently hits this branch (suspected 950DT recognized as arch35, or modeling passes `rotate`), inconsistent with the doc description, to verify.

---

<a id="5-qwen35-fused-operators"></a>
## 5. Qwen3.5 Fused Operators

> Data source: `cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py` (3237 lines)
> This repo's implementation: `cann_lm/models/qwen3_5/modeling_qwen3_5.py`

Qwen3.5 is a **hybrid architecture**: full_attention layers (standard GQA + KV cache) + linear_attention layers
(GatedDeltaNet linear attention + conv1d + recurrent state). The reference impl's fused operators focus on
**linear attention** (`npu_chunk_gated_delta_rule` / `npu_recurrent_gated_delta_rule` / `causal_conv1d` /
`FusedRMSNormGated`), which is also the root cause of this repo's `export_to_epair` segfault (pure-torch
`torch_chunk_gated_delta_rule` Python for-loop unrolling 19418 nodes causing GE graph explosion).

### 5.1 Reference implementation (cann-recipes-infer) torch_npu operators survey

A total of **22** fused operators, grouped into 6 categories by function domain. The linear attention category
(#1-#6) is Qwen3.5-specific; the rest overlap with Gemma-4 (see §4.1, only differences listed here).

#### 5.1.1 Linear Attention (Qwen3.5-specific, core)

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 1 | `torch_npu.npu_chunk_gated_delta_rule` | `GatedDeltaNet._forward_prefill_with_fused_kernel` (line 1541) | `(query, key, value, beta=, initial_state=, actual_seq_lengths=, scale=, g=)` | **prefill linear attention core**: fuses chunked gated delta rule full flow (L2norm + chunk decay + intra/inter-chunk recurrence + state update), single op replaces ~100 lines of pure-torch Python for-loop |
| 2 | `torch_npu.npu_recurrent_gated_delta_rule` | `GatedDeltaNet._forward_decode` (line 1646) | `(query, key, value, last_recurrent_state, beta=, scale=, actual_seq_lengths=, ssm_state_indices=, num_accepted_tokens=, g=, gk=)` | **decode linear attention core**: fuses token-by-token recursion (state *= exp(g) → kv_mem → delta → state += → out), supports speculative decode's num_accepted_tokens |
| 3 | `causal_conv1d_fn` (externally injected) | `GatedDeltaNet._forward_prefill` (line 1336) | `(x, weight, bias, activation, seq_idx)` | prefill conv1d fusion: state rolling + depthwise conv1d + SiLU single kernel |
| 4 | `causal_conv1d_update` (externally injected) | `GatedDeltaNet._forward_decode` (line 1598) | `(x, conv_state, weight, bias, activation)` positional args | decode conv1d fusion: single-token state rolling + conv + SiLU |
| 5 | `FusedRMSNormGated` (externally injected) | `GatedDeltaNet.__init__` (line 1153) | `FusedRMSNormGated(head_v_dim, eps, activation, device, dtype)`, forward `(core_attn_out, z)` | linear attention norm + gate fusion: `rms_norm(x) * silu(gate)` single kernel |
| 6 | `ge_safe_softplus` (pure-torch replacement) | `GatedDeltaNet` (line 1518) | `relu(x) + log1p(exp(-abs(x)))` | Not an NPU op, but avoids `aten::softplus` for ge_graph (uses relu+log1p expression); this repo can directly borrow |

> **Enable condition** (lines 1178-1181): `use_npu_chunk_gated_delta_rule = (_PLATFORM_VERSION != "950"
> and hasattr(torch_npu, "npu_chunk_gated_delta_rule"))` — i.e. **950 goes pure-torch prefill,
> 910/A3 goes fused operator**.

#### 5.1.2 Normalization (RMSNorm, overlaps with Gemma-4)

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 7 | `torch_npu.npu_rms_norm` | `Qwen3_5MoeRMSNormGated.ln_npu` (line 885), `Qwen3_5MoeRMSNorm.ln_npu` (line 2577) | `(x, weight, eps)[0]` | Fused RMSNorm |
| 8 | `torch_npu.npu_add_rms_norm` | `Qwen3_5MoeRMSNorm.add_rms_norm_npu` (line 2615) | `(residual, x, gamma, eps)` returns `(y, _, residual)` | Fused residual add + RMSNorm |
| 9 | `torch_npu.npu_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.rms_norm_mx` (line 2588) | `(x, gamma, epsilon, dst_type)` returns `(x_quant, x_scale, _)` | RMSNorm + MXFP8 quantized output (for downstream quantized linear) |
| 10 | `torch_npu.npu_add_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.add_rms_norm_mx` (line 2605) | `(residual, x, gamma, epsilon, dst_type)` returns `(y, x_out, mxscale, _)` | add + RMSNorm + MXFP8 quantization |

#### 5.1.3 Activation

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 11 | `torch_npu.npu_swiglu` | `SiLUAndMul.forward` (line 742) | `(x)` | Fused SiLU+Mul: `silu(gate) * up` (after MLP gate_up_proj) |
| 12 | `torch_npu.npu_swiglu_mx_quant` | `SiLUAndMul.forward_mx` (line 746) | `(x, group_index, dst_type, activate_left)` returns `(out, scale)` | SiLU+Mul + MXFP8 quantization (for quantized down_proj) |

#### 5.1.4 Attention (full_attention layers)

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 13 | `torch_npu.npu_rotary_mul` | `Qwen3_5MoeAttention.forward` (line 1961) | `(q, cos, sin, rotary_mode='half')` | RoPE fusion (consistent with Gemma-4) |
| 14 | `torch_npu.scatter_update_` | `Qwen3_5MoeAttention.forward` (line 1975) | `(cache, indices, updates, dim=-2)` in-place | KV cache write (replaces index_copy) |
| 15 | `torch.ops.npu.npu_fused_infer_attention_score` | `Qwen3_5MoeAttention.forward` (line 1979 prefill / 1998 decode) | prefill: `sparse_mode=3, atten_mask=...`; decode: `actual_seq_lengths_kv=..., atten_mask=None` | FlashAttention fusion (QK^T+softmax+AV+GQA+PagedAttention) |

#### 5.1.5 Quantization

| # | operator | call site | key params | purpose |
|---|------|----------|----------|------|
| 16 | `torch_npu.npu_dynamic_mx_quant` | `_quantize_weight_last_dim` (line 286) | `(weight_2d, dst_type=float8_e4m3fn)` returns `(quant, scale)` | Weight online MXFP8 quantization |
| 17 | `torch_npu.npu_mm_all_reduce_base` | `qwen3_5_prefill_mm_all_reduce` (line 219) | `(input, weight, hcom, reduce_op="sum")` | MM + AllReduce fusion (950 only, prefill's o_proj/down_proj) |

#### 5.1.6 MoE (this repo's Qwen3.5 is dense, no MoE; the reference impl's MoE version has 8, same as Gemma-4 §4.1.5 #8-#15, omitted)

---

### 5.2 Available fused operators analysis for this repo's `modeling_qwen3_5.py`

This repo's implementation calls `torch.ops.npu.*` (+ `torch.ops.cann_lm::l2norm`) fused operators; below gives per-module "current implementation → fused operator" mapping. Marked `✅ landed` / `⚠️ needs refactor` / `❌ not applicable`.

#### 5.2.1 Linear Attention Core (GatedDeltaNet) — **segfault resolved (chunk_gated_delta_rule landed)**

**Current implementation** (`Qwen3_5GatedDeltaNet.forward`, modeling_qwen3_5.py):
- **active path**: prefill + decode **share** `torch.ops.cann_lm.chunk_gated_delta_rule` (line 621); the fused op has landed (`fused_ops/chunk_gated_delta_rule.py` + `_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule`, ge_converter.py:59-60). OM final state hits 18.
- **fallback (not activated)**: `torch_chunk_gated_delta_rule` (line 264) + `torch_recurrent_gated_delta_rule` (line 346) are pure-torch reference impls, retained for comparison, not on the export path.

| operator | status | notes |
|------|------|------|
| `torch_npu.npu_chunk_gated_delta_rule` | ✅ landed+used | segfault resolved: `fused_ops/chunk_gated_delta_rule.py` registers schema+fake+CPU impl (`npu::` namespace); `_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule` (ge_converter.py:55-56); modeling line 621 calls it. OM final state hits 18 |
| `torch_npu.npu_recurrent_gated_delta_rule` | ❌ not registered (not needed) | decode **doesn't go recurrent**: chunk_gated_delta_rule shared by prefill+decode (decode seq=1 pads within chunk, see `chunk_gated_delta_rule.py` docstring). `torch_recurrent_gated_delta_rule` (line 346) retained only as pure-torch fallback |
| `causal_conv1d_fn` / `causal_conv1d_update` | ✅ aten::conv1d path landed | modeling `torch_causal_conv1d_update` (lines 243-260) uses `F.conv1d`+`F.silu`; `ge_converter.py:58-59` registers `aten::conv1d` converter (`_conv1d_converter.py`). No standalone `cann_lm::causal_conv1d` registered (aten path suffices) |
| `FusedRMSNormGated` | ⚠️ decomposition scheme adopted (fused version not landed) | modeling `Qwen3_5RMSNormGated` (lines 40-42) = `cann_lm::rms_norm` + `F.silu(gate)*x` (decomposition, not fused single kernel). `cann_lm::rms_norm`→`ge.RmsNorm` already used; fused `rms_norm_gated` not registered |

> **Platform branch**: the reference impl's `npu_chunk_gated_delta_rule` is enabled only on non-950 (910/A3); 950 goes pure torch.
> This repo unified to `cann_lm::chunk_gated_delta_rule` (910 real machine + export both go fused op, segfault resolved);
> 950 still goes pure-torch fallback (950 export segfault issue awaits a 950-specific solution).

#### 5.2.2 RMSNorm → `npu_rms_norm` (landed+used) / `npu_add_rms_norm` (not landed)

**Current implementation**: `Qwen3_5RMSNorm.forward` (lines 22-28) already calls `torch.ops.cann_lm.rms_norm` (line 25, `1.0 + weight` bias convention); `Qwen3_5RMSNormGated.forward` (lines 40-42) = `cann_lm::rms_norm` + `F.silu(gate)*x`. The original doc's "pure torch pow" description is stale.

| operator | status | notes |
|------|------|------|
| `cann_lm::rms_norm` | ✅ landed+used | `fused_ops/rms_norm.py` registers schema+fake+CPU impl (`cann_lm::` namespace); GE converter in cann_lm `_rms_norm_converter.py` (ge_converter.py:72-73 registers `torch.ops.cann_lm.rms_norm.default`)→`ge.RmsNorm`. modeling lines 25/41 **already call** `torch.ops.cann_lm.rms_norm` |
| `cann_lm::add_rms_norm` | ❌ not landed | No `fused_ops/add_rms_norm.py`, no such registration in ge_converter.py (original doc's "landed"+file is stale). gemma4 already explicitly splits into add + `cann_lm::rms_norm` (modeling_gemma4.py:308 comment "user requirement"), qwen3.5 same design. To fuse, need new schema+converter |

#### 5.2.3 RoPE → `npu_rotary_mul` (landed)

**Current implementation**: `Qwen3_5RotaryEmbedding` (line 68) + attention uses `torch.ops.cann_lm.rotary_mul` (lines 128/129, `rotary_mode='half'`), not `apply_rotary_pos_emb`. Original doc description is stale.

| operator | status | notes |
|------|------|------|
| `cann_lm::rotary_mul` | ✅ landed+used | `fused_ops/rotary_mul.py` registers schema+fake (`cann_lm::` namespace); GE converter in cann_lm `_rotary_mul_converter.py` (ge_converter.py:75-76 registers `torch.ops.cann_lm.rotary_mul.default`)→`ge.RotaryMul`/`ge.RotaryPositionEmbedding` (arch35). modeling lines 128/129 **already call** `torch.ops.cann_lm.rotary_mul`. Shared with Gemma-4 |

#### 5.2.4 KV Cache Write → `scatter_update_` (landed, IR path works)

**Current implementation** (`SimpleKVCache.update`, lines 330-332): `index_copy(2, new_kv_cache_pos, kv)`.

| operator | status | notes |
|------|------|------|
| `aten::index_copy` → `ge.InplaceUpdate + TransposeD` | ✅ landed | `ge_converter.py:61-62` registers `aten::index_copy` converter (`_index_copy_converter.py`: TensorMove + TransposeD + InplaceUpdate, **not** `ge_converter.py:234` which doesn't exist). IR export path works. Real machine can use `npu_scatter_update_` |

#### 5.2.5 Full Attention → `npu_fused_infer_attention_score_v2` (op landed; enable status to verify)

**Current implementation**: `Qwen3_5Attention` (lines 520-569) = eager attention (matmul QK^T + softmax + matmul AV + GQA repeat_kv). modeling has another FIA helper (line 215) calling `cann_lm::fused_infer_attention_score_v2` (with decode correctness comment), but §5.4 qwen3.5 OM table shows FIA=0 (all eager, "decode fix temporarily shelved") — the two contradict; which path export actually takes is to verify.

| operator | status | notes |
|------|------|------|
| `cann_lm::fused_infer_attention_score_v2` | ✅ op landed (enable to verify) | `fused_ops/fused_infer_attention_score.py` registers schema+fake (`cann_lm::` namespace); `ge_converter.py:81-82` (`_fused_infer_attention_score_v2_converter.py`)→`ge.FusedInferAttentionScore`. modeling line 215 calls `torch.ops.cann_lm.fused_infer_attention_score_v2`, but OM final state FIA=0 (see §5.4 qwen3.5 table), export path to verify |

#### 5.2.6 Activation → `npu_swiglu` (not landed)

**Current implementation**: `Qwen3_5MLP.forward` (line 569) = `F.silu(gate) * up` pure torch (or `gate * silu(up)`).

| operator | status | notes |
|------|------|------|
| `npu_swiglu` | ❌ not landed | Can register `cann_lm::swiglu` + GE mapping `ge.Swiglu` (check CANN op_info). Low priority; pure-torch silu+mul decomposes to Swish/Sigmoid+Mul in GE export, already exportable |

---

### 5.3 Benefit Ranking and Recommended Landing Order

Ranked by "resolve export segfault / speed up real-machine inference":

| Priority | Fused operator　　　　　　　　　　　　　　　　　　　　　　　　　　　　| Replacement target　　　　　　　　　　　　　　　　　　 | Status/expected benefit　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------| -----------------------------------------------------------------| ----------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **P0** | `cann_lm::chunk_gated_delta_rule` (→ `ge.ChunkGatedDeltaRule`) | `torch_chunk_gated_delta_rule` (prefill)　　| ✅ **resolved**: `fused_ops/chunk_gated_delta_rule.py` + `_chunk_gated_delta_rule_converter.py` (ge_converter.py:59-60); modeling line 621 calls torch.ops.cann_lm.chunk_gated_delta_rule; OM hits 18. 19418 nodes → 1 op |
| **P0** | `npu_recurrent_gated_delta_rule`　　　　　　　　　　　　　　　　| `torch_recurrent_gated_delta_rule` (decode) | ❌ **not needed**: decode goes through shared chunk_gated_delta_rule (pads within chunk); recurrent retained only as torch fallback, not registered　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P1　　 | `aten::conv1d` (→ `_conv1d_converter.py`)　　　　　　　　　　　| `torch_causal_conv1d_update`　　　　　　　　 | ✅ **landed**: ge_converter.py:58-59 registers `aten::conv1d`; modeling uses `F.conv1d`+`F.silu` (lines 259-260). No standalone `cann_lm::causal_conv1d` built　　　　　　　　　　　　　　　　　　　 |
| P1　　 | `FusedRMSNormGated`　　　　　　　　　　　　　　　　　　　　　　 | `Qwen3_5RMSNormGated`　　　　　　　　　　　　| ⚠️ **decomposition scheme adopted**: modeling lines 40-42 = `npu_rms_norm` + `F.silu(gate)*x`; fused `rms_norm_gated` not registered (decomposition works)　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P2　　 | `cann_lm::rms_norm`　　　　　　　　　　　　　　　　　　　　　　 | `Qwen3_5RMSNorm` pure torch　　　　　　　　　　| ✅ **landed+used**: modeling lines 25/41 call `torch.ops.cann_lm.rms_norm`; cann_lm `_rms_norm_converter.py`→`ge.RmsNorm` (ge_converter.py:72-73)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P2　　 | `cann_lm::add_rms_norm`　　　　　　　　　　　　　　　　　　　　 | residual add + norm two steps　　　　　　　　　　　　 | ❌ **not landed**: no add_rms_norm.py/converter; gemma4 explicitly splits into add+`cann_lm::rms_norm` (modeling_gemma4.py:308), qwen3.5 same design　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| P3　　 | `cann_lm::fused_infer_attention_score_v2`　　　　　　　　　　　 | full attention eager　　　　　　　　　　　　 | ⚠️ **op landed** (ge_converter.py:81-82), modeling line 215 calls `torch.ops.cann_lm.fused_infer_attention_score_v2`, but OM final state FIA=0 (to verify, see §5.4)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| P3　　 | `npu_swiglu`　　　　　　　　　　　　　　　　　　　　　　　　　　| `F.silu(gate) * up`　　　　　　　　　　　　　| ❌ not landed (pure-torch decomposes to Swish/Sigmoid+Mul already exportable, OM hits Swish=60/Sigmoid=24)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

> **P0 resolved**: `chunk_gated_delta_rule` is the solution to Qwen3.5 `export_to_epair` segfault (pure-torch for
> loop unrolling 19418 nodes → GE graph explosion). Done via ① `fused_ops/chunk_gated_delta_rule.py` registers
> schema+fake (`cann_lm::` namespace) + CPU impl; ② `_chunk_gated_delta_rule_converter.py`→`ge.ChunkGatedDeltaRule`
> (ge_converter.py:59-60, check CANN `aic-ascend910_93-ops-info-*.json`); ③ modeling line 621 calls
> `torch.ops.cann_lm.chunk_gated_delta_rule`. `recurrent_gated_delta_rule` not registered (decode goes through
> shared chunk). Aligned params with the cann-recipes-infer `npu_chunk_gated_delta_rule` signature (line 1541).

---

### 5.4 OM operator table vs support_ops (qwen3.5 no-quant prefill-decode, Ascend910_9362)

> qwen3.5-0.8B no quantization, attention **all eager** (FIA decode fix verified but temporarily shelved: prefill multi-chunk limit + scheme B hits torch.export branch constant lift crash). SoftplusV2 + L2Normalize landed. OM flat so_store=0.

#### 5.4.1 fuse_op

| fuse_op | OM op | decode | hit |
|---|---|---:|---|
| ChunkGatedDeltaRule | ChunkGatedDeltaRule | 18 | ✅ |
| Rmsnorm | RmsNorm | 79 | ✅ |
| FusedInferAttention | — | 0 | ❌ all eager (decode fix verified but temporarily shelved: prefill multi-chunk limit)|
| l2norm | L2Normalize | 36 | ✅ (cann_lm::l2norm → ge.L2Normalize)|
| RotaryPositionEmbedding | RotaryMul | 12 | ✅ |
| ScatterNdUpdate | ScatterNdUpdate | 12 | ✅ |

#### 5.4.2 quant_op (baseline no quantization)

| quant_op | OM | decode | hit |
|---|---|---:|---|
| DynamicQuant | — | 0 | Not quantized |
| QuantBatchMatmulV4 | — | 0 | Not quantized |

#### 5.4.3 single_op

| single_op | OM op | decode | hit |
|---|---|---:|---|
| Cast | Cast | 208 | ✅ |
| Mul | Mul | 84 | ✅ (includes Neg→Mul(-1.0) replacement)|
| splitV | SplitVD | 24 | ✅ same family |
| concatD | ConcatV2D | 67 | ✅ same family |
| Transpose | Transpose | 54 | ✅ |
| Conv2DV2 | Conv2D | 18 | ✅ same family |
| StridedSliceV2 | StridedSliceD | 61 | ✅ same family |
| Add | Add | 24 | ✅ |
| Swish | Swish | 60 | ✅ |
| GatherV2 | GatherV2 | 4 | ✅ (includes Index→index_select→GatherV2 replacement)|
| TransData | TransData | 36 | ✅ |
| Sigmoid | Sigmoid | 24 | ✅ |
| MatmulV3 | MatMul | 187 | ✅ same family |
| SoftplusV2 | SoftplusV2 | 18 | ✅ (cann_lm converter → ge.SoftplusV2, needs reg_register.h registration)|
| Gelu | — | 0 | ❌ model unused |

#### 5.4.4 OM framework/decomposition operators (out of §1 scope)

| OM op | decode | nature | eliminable? |
|---|---:|---|---|
| Const | 341 | Weights/params/inlined ge.Const (graph infrastructure)| ❌ |
| Data | 53 | Graph inputs (infrastructure)| ❌ |
| NetOutput | 1 | Graph output (infrastructure)| ❌ |
| Exp | 18 | linear attention g's `A_log.exp()` (plain exp, no §1 counterpart)| ❌ g computation inherent |
| BatchMatMul | 12 | eager attention's QK^T + @V (6 full-attn layers×2)| ⚠️ re-enabling FIA can eliminate (but prefill multi-chunk limit)|
| SoftmaxV2 | 6 | eager attention's softmax (6 full-attn layers)| ⚠️ same |
| Muls | 6 | eager attention's `*scaling` scalar mul (6 layers)| ⚠️ FIA uses softmax_scale attr to eliminate |

#### 5.4.5 Conclusions

- **fuse_op**: 5/8 hit (ChunkGatedDeltaRule / Rmsnorm / l2norm / RotaryPositionEmbedding / ScatterNdUpdate ✅; FusedInferAttention ❌ all eager; DynamicQuant / QuantBatchMatmulV4 ❌ no quantization).
- **single_op**: 14/15 hit (Gelu ❌ model unused; SoftplusV2 ✅ landed needs reg_register.h registration; Neg→Mul / Index→GatherV2 replaced and eliminated).
- **Framework ops** (Const / Data / NetOutput): inherently out of §1 scope.
- **eager cost**: BatchMatMul ×12 + SoftmaxV2 ×6 + Muls ×6 (24 total) is the cost of all-eager attention; re-enabling FIA can eliminate (decode verified, prefill multi-chunk limit pending).
- **Decomposition primitives**: Exp ×18 is inherent to linear attention g computation (`A_log.exp()`), no §1 counterpart, not eliminable.
- **Flat no dynamic subgraph**: so_store=0, stream_num=1.

---

### 5.5 Platform Differences Quick Reference (Qwen3.5)

| Platform | `npu_chunk_gated_delta_rule` | fp8/mxfp8 quantization | `npu_mm_all_reduce_base` | MoE MC2 |
|------|------------------------------|----------------|---------------------------|---------|
| **950** (Atlas 950) | ❌ pure-torch prefill | ✅ supported | ✅ supported | group_type=0/3 |
| **910/A3** (non-950) | ✅ fused operator | ❌ not supported | ❌ A3 not supported | `fullmesh_v2` |

> This machine's NPU is **Ascend910_9362** (non-950), supports `npu_chunk_gated_delta_rule` fused operator
> (reference impl), doesn't support fp8/mxfp8 quantization. **Fused operator landed**: 910 real-machine
> inference + export both go through `cann_lm::chunk_gated_delta_rule` (segfault resolved, OM hits 18); 950
> reference impl goes pure-torch fallback, but 950 pure-torch still segfaults at export, needs a 950-specific
> solution or keeping the `ge.ChunkGatedDeltaRule` mapping so 950 can also use the GE op.
