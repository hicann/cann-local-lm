# Gemma-4 融合算子汇总与分析

> 数据来源：
> - 参考实现（CANN 优化版）：`cann-recipes-infer/models/gemma_4/models/modeling_gemma4.py`
> - 本仓实现（纯 PyTorch 参考版）：`cann_lm/models/gemma4/modeling_gemma4.py`

参考实现基于 CANN 推理框架（TND/BNSD Paged Attention + MC2 EP + GE Graph），大量使用
`torch_npu` / `torch.ops.npu` 融合算子；本仓 `modeling_gemma4.py` 为纯 PyTorch 参考实现，
**未引入任何 `torch_npu` 调用**，全部算子用 `torch` / `torch.nn.functional` 表达。本文档先
汇总参考实现用到的全部 torch_npu 算子，再逐模块分析本仓实现可替换的融合算子。

---

## 一、参考实现（cann-recipes-infer）使用的 torch_npu 算子汇总

共 **15** 个融合算子，按功能域分为 5 类。

### 1.1 归一化类（RMSNorm）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 1 | `torch_npu.npu_rms_norm` | `Gemma4RMSNorm.forward` | `(x, weight, eps)` 返回 `[normed, ...]` | 融合 RMSNorm：`x * rsqrt(mean(x²)+eps) * weight`，替代 pow/mean/mul 多步 |
| 2 | `torch_npu.npu_add_rms_norm` | `Gemma4RMSNorm.forward_add` | `(x1, x2, weight, eps)` 返回 `[normed, _, residual_sum]` | **融合 add + RMSNorm**：一次算出 `rms_norm(x1+x2)` 和 `x1+x2`，省一次 elementwise add + 一次 norm |

### 1.2 激活类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 3 | `torch_npu.npu_fast_gelu` | `Gemma4MLP.forward`、`_GegluMoEMethod.apply` | `(x)` | `gelu(approximate='tanh')` 的融合实现，graph-safe |

### 1.3 RoPE 旋转位置编码类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 4 | `torch_npu.npu_rotary_mul` | `Gemma4Attention.forward`（sliding 层） | `(q4, cos4, sin4, rotary_mode='half')`，需 4D BSND 输入 | 融合 rotate-half RoPE，用于 sliding 层全 head_dim 旋转 |
| 5 | `torch_npu.npu_apply_rotary_pos_emb` | `Gemma4Attention.forward`（full 层） | `(q_rot, k_rot, cos, sin, layout="TND")` | 融合 partial-rotary RoPE（full 层只旋 rotary_dim 维），支持 TND layout |

### 1.4 Attention 类（FlashAttention + PagedAttention）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 6 | `torch.ops.npu.npu_fused_infer_attention_score_v2`（ge_graph decode 经 `torchair.ops`） | `_attn_tnd_sliding`、`_attn_bnsd_full` | `input_layout="TND"/"BNSD"`、`sparse_mode`、`pre_tokens`、`block_table`、`block_size`、`num_key_value_heads`（GQA） | 融合 QK^T + softmax + AV + PagedAttention，sliding 层用 TND(D=256)+sparse_mode=4，full 层用 BNSD(D=512)+sparse_mode=3/0 |
| 7 | `torch_npu.npu_scatter_nd_update_` | `Gemma4Attention._update_cache` | `(cache.view(-1,N,D), slot, kv)` in-place | 融合 KV cache scatter-write（按 slot_mapping 写入 PagedAttention 缓存） |

### 1.5 MoE 类（本仓 Gemma-4 当前为纯 dense MLP，不含 MoE）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 8 | `torch_npu.npu_moe_gating_top_k_softmax` | `Gemma4Router.forward` | `(expert_scores, None, k=top_k)` 返回 `[weights, index, _]` | 融合 top-k + softmax 路由 |
| 9 | `torch_npu.npu_grouped_matmul` | `_GegluMoEMethod.apply`（两次：w13 / w2） | `group_list=expert_tokens, group_type=0, split_item=3` | 分组 GEMM，按 expert 分段做 matmul |
| 10 | `torch_npu.npu_geglu` | `_GegluMoEMethod.apply`（prefill/eager 路径） | `(mm1_mm3, -1, 1)` | 融合 GEGLU 激活（gelu_tanh(gate)*up），仅 prefill/eager 用；graph decode 退化为 manual `npu_fast_gelu*up` |
| 11 | `torch_npu.npu_moe_init_routing_v2` | `moe_infer_tp`、`moe_infer_double_routing` | `expert_idx, active_num, expert_num, expert_tokens_num_type=1, quant_mode=-1` | MoE 初始化路由：展开 token×expert、排序、统计每 expert token 数 |
| 12 | `torch_npu.npu_moe_finalize_routing` | `moe_infer_tp`、`moe_infer_double_routing` | `scales=topk_weight, expanded_src_to_dst_row, drop_pad_mode=2` | MoE 收尾路由：按 topk_weight 聚合 expert 输出回原 token 位 |
| 13 | `torch_npu.npu_moe_re_routing` | `moe_infer_double_routing` | `(gathered_tokens, tokens_per_expert_group.view(ep_size,-1))` | EP 二次路由：AllToAll 后按本地 expert 重排序 |
| 14 | `torch_npu.npu_moe_distribute_dispatch_v2` | `moe_infer_dispatch_combine`（EP decode） | `x, expert_ids, group_ep, ep_world_size, comm_alg="fullmesh_v2"` | MC2 EP dispatch：AllToAll 派发 token 到 expert 所在 rank |
| 15 | `torch_npu.npu_moe_distribute_combine_v2` | `moe_infer_dispatch_combine`（EP decode） | `expand_x, expert_ids, assist_info_for_combine, expert_scales` | MC2 EP combine：聚合 expert 输出回原 rank 并加权 |

---

## 二、本仓 `modeling_gemma4.py` 可用的融合算子分析

本仓实现为纯 PyTorch，以下逐模块给出「当前实现 → 可替换融合算子」的映射。标注
`✅ 可直接替换` / `⚠️ 需重构` / `❌ 不适用`。

### 2.1 RMSNorm → `npu_rms_norm` / `npu_add_rms_norm`

**当前实现**（`Gemma4RMSNorm._norm` + `forward`，modeling_gemma4.py:146-154）：
```python
mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
output = x * torch.pow(mean_sq, -0.5)          # 注释称 Gemma-4 用 pow(-0.5) 而非 rsqrt
output = output * self.weight.float()
return output.type_as(x)
```

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_rms_norm` | ✅ 可直接替换 | 与参考实现 `Gemma4RMSNorm.forward` 一一对应。注意：本仓注释强调用 `pow(-0.5)`，需验证 `npu_rms_norm` 内部是否为 `rsqrt`（多数实现 `1/sqrt` 与 `pow(-0.5)` 数值一致，bf16 下无差异）。`with_scale=False` 时传 ones weight 即可（参考实现的 v_norm / router norm 即如此） |
| `torch_npu.npu_add_rms_norm` | ⚠️ 需重构 | 融合 residual add + rms_norm。本仓当前是「先 norm 再 add」的两步（见下文 2.2），需重构为 `forward_add(residual, x)` 一次算出 `norm(residual+x)` 和 `residual+x`，省一次 elementwise add。参考实现 `Gemma4DecoderLayer` 已示范该模式 |

### 2.2 Residual Add + Norm 融合 → `npu_add_rms_norm`

**当前实现**（`Gemma4TextDecoderLayer.forward`，modeling_gemma4.py:487-494）：
```python
hidden_states = self.post_attention_layernorm(hidden_states)   # norm(attn_out)
hidden_states = residual + hidden_states                        # add（独立语句）
...
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)   # norm(add 结果)
```

line 488 的 `residual + hidden_states` 与 line 491 的 `pre_feedforward_layernorm` 可融合为一次
`npu_add_rms_norm(residual, post_attn_normed, weight, eps)` → 返回 `(normed, residual_sum)`，
后续 `residual = residual_sum` 直接复用返回值，省一次独立 add kernel。同理 line 494 的
`residual + hidden_states` 可与下一层 `input_layernorm` 融合。

> 参考实现 `Gemma4DecoderLayer.forward`（line 862）即用 `pre_feedforward_layernorm.forward_add`
> 把 add + norm 合为单算子。

### 2.3 GELU(tanh) → `npu_fast_gelu`

**当前实现**：
- `Gemma4TextMLP.__init__`（line 272）：`self.act_fn = nn.GELU(approximate="tanh")`
- `Gemma4TextDecoderLayer.forward` per_layer_input（line 499）：`F.gelu(gated, approximate="tanh")`

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_fast_gelu` | ✅ 可直接替换 | 参考实现注释明确 `npu_fast_gelu == gelu(approximate='tanh')`，graph-safe。MLP 的 `act_fn(gate)*up` 模式与参考实现 `Gemma4MLP.forward` 一致 |

### 2.4 RoPE → `npu_rotary_mul` / `npu_apply_rotary_pos_emb`

**已落地**（modeling_gemma4.py）：`apply_rotary_pos_emb`（单 tensor）委托 `cann_lm::rotary_mul`
（`rotary_mode='half'`），partial rotary 由调用方 slice+cat 包裹；attention 对 q、k 各调一次
（sliding 全旋 + full partial 旋均走此路径）。原 `apply_rotary_pos_emb_qk`（q+k 配对）及对应
`cann_lm::apply_rotary_pos_emb` 融合算子已删除（attention 未用配对路径，单 tensor 路径覆盖两类层）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rotary_mul`（= `torch_npu.npu_rotary_mul`） | ✅ 已落地 | sliding + full 两类层均用，单 tensor。GE 转换器 → `ge.RotaryMul` |
| `torch_npu.npu_apply_rotary_pos_emb`（q+k 配对） | ❌ 未采用 | 参考 full 层用此配对算子（TND partial）；本仓改用单 tensor `rotary_mul` 统一两类层，配对路径已删 |

> 本仓 `fused_ops/ops.py` 还注册了自定义 `cann_lm::rope_in_place`（GE 算子 `RopeInPlace`），
> 可作为 RoPE 的项目内融合替代（见 converter/export/ge_converter.py）。

### 2.5 Eager Attention → `npu_fused_infer_attention_score_v2`

**当前实现**（`eager_attention_forward`，modeling_gemma4.py:87-129）：
手动 `repeat_kv`（GQA 展开）→ `matmul(Q,K^T)*scale` → `+mask` → `softmax(fp32)` → `matmul(@V)`，
含 softcap（`x/cap → tanh → x*cap`）。

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch.ops.npu.npu_fused_infer_attention_score_v2` | ⚠️ 需重构 | 融合 QK^T+softmax+AV+PA+GQA+sliding window。收益最大：① 取消 `repeat_kv`（融合算子原生支持 `num_key_value_heads` GQA）；② sliding 层用 `sparse_mode=4 + pre_tokens=sliding_window` 直接表达滑窗，省 `_prepare_causal_mask` 构造的 4D mask；③ 支持 `block_table`+`block_size` 做 PagedAttention，省 KV cache 的 index_copy。注意 full 层 head_dim=512 需用 BNSD layout（TND 拒绝 D=512）；sliding 层 D=256 可用 TND |
| `cann_lm::fused_infer_attention_score` | ⚠️ 需重构 | 本仓 `ops.py` 注册的自定义 FA 算子（GE 转换为 `ge.FusedInferAttentionScore`，converter/export/ge_converter.py:82），参数与 npu 版对齐（含 `block_table`/`num_key_value_heads`/`sparse_mode`），可作为 NPU 不可用时的 IR 级 fallback |

> softcap 部分：参考实现把 `1/sqrt(d)` 吸收进 q_norm（`scale_fa=1.0`），本仓 `scaling=1.0`
> 已一致；logit softcap（`x/cap→tanh→x*cap`）无单一融合算子，可保留 elementwise 或后续做
> 自定义融合。

### 2.6 KV Cache 写入 → `npu_scatter_nd_update_`

**当前实现**（`Gemma4TextAttention.forward`，modeling_gemma4.py:391-392）：
```python
cache_k = cache_k.index_copy(2, new_kv_cache_pos, key_states)
cache_v = cache_v.index_copy(2, new_kv_cache_pos, value_states)
```

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_scatter_nd_update_` | ✅ 可直接替换 | 与参考实现 `_update_cache` 一致，按 slot_mapping in-place 写入。本仓 `converter/export/ge_converter.py:234` 也已为 `aten::index_copy` 注册了 GE `InplaceUpdate+Transpose` 转换器，IR 导出路径已通 |

### 2.7 MoE 算子（不适用）

本仓 `modeling_gemma4.py` 当前为**纯 dense MLP**（`Gemma4TextMLP`，无 `Gemma4SparseMoeBlock`），
故参考实现的 8 个 MoE 算子（`npu_moe_gating_top_k_softmax`、`npu_grouped_matmul`、`npu_geglu`、
`npu_moe_init_routing_v2`、`npu_moe_finalize_routing`、`npu_moe_re_routing`、
`npu_moe_distribute_dispatch_v2`、`npu_moe_distribute_combine_v2`）**均不适用**。若后续本仓
引入 MoE 路径，可逐一对齐参考实现的 `Gemma4SparseMoeBlock` 三种路由模式（TP / EP double-routing /
MC2 dispatch-combine）。

---

## 三、收益排序与建议落地顺序

按「单次 forward 调用频次 × 单算子收益」排序：

| 优先级 | 融合算子 | 替换目标 | 预期收益 |
|--------|----------|----------|----------|
| P0 | `npu_fused_infer_attention_score_v2` | `eager_attention_forward` 全流程 | 消除 repeat_kv+2×matmul+softmax+mask 构造，单层最大 kernel 融合，全模型 N 层 |
| P0 | `npu_rms_norm` | `Gemma4RMSNorm._norm` | 每层 4-6 个 RMSNorm（input/post_attn/pre_ffn/post_ffn + q/k/v_norm），高频 |
| P1 | `npu_fast_gelu` | `nn.GELU(tanh)` / `F.gelu(tanh)` | 每层 MLP + per_layer_input 各一次 |
| P1 | `npu_add_rms_norm` | residual add + 下一 norm | 需轻微重构 forward 拓扑，每层 2 处 |
| P2 | `npu_rotary_mul`（已落地） | `apply_rotary_pos_emb` 手动实现 | 每层 q/k 各 1 次，sliding+full 统一走单 tensor `rotary_mul` |
| P2 | `npu_scatter_nd_update_` | `index_copy` KV cache 写入 | 每层 2 处（k/v），IR 路径已通 |

> 注：本仓当前为纯 PyTorch 参考实现，引入 `torch_npu` 需遵循 AGENTS.md 的 stub 约定
> （converter 侧 torchair 编译 + lib 层打桩），NPU 实机验证走系统 Python 的 torch_npu。
