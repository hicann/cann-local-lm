# 融合算子汇总与分析（Gemma-4 + Qwen3.5）

> 数据来源：
> - 参考实现（CANN 优化版）：
>   - Gemma-4：`cann-recipes-infer/models/gemma_4/models/modeling_gemma4.py`
>   - Qwen3.5：`cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py`
> - 本仓实现（纯 PyTorch 参考版）：
>   - `cann_lm/models/gemma4/modeling_gemma4.py`
>   - `cann_lm/models/qwen3_5/modeling_qwen3_5.py`
>   - 自定义算子：`cann_lm/models/fused_ops/` + GE 映射 `cann_lm/converter/export/ge_converter.py`

参考实现基于 CANN 推理框架（TND/BNSD Paged Attention + MC2 EP + GE Graph），大量使用
`torch_npu` / `torch.ops.npu` 融合算子；本仓 `modeling_*.py` 为纯 PyTorch 参考实现，
**未引入任何 `torch_npu` 调用**，全部算子用 `torch` / `torch.nn.functional` 表达，再通过
`cann_lm::` 自定义算子 + `ge_converter.py` 映射到 GE 算子做 IR 导出。本文档分别汇总两模型
参考实现用到的全部 torch_npu 算子，再逐模块分析本仓实现可替换的融合算子。

---

# 第一部分：Gemma-4 融合算子

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

**已落地**（modeling_gemma4.py）：与图灵参考实现（cann-recipes-infer）一致，按 layer_type 分两条路径：
- **sliding 层**（全旋，`head_dim == cos.shape[-1]`）：`npu_rotary_mul` 单 tensor，q/k 各调一次（`rotary_mode='half'`）。
- **full 层**（proportional partial，`rotary_dim == cos.shape[-1] < head_dim`）：cat 出 q_rot/k_rot（前 `pair_dim` + 中 `half_dim:half_dim+pair_dim`），`npu_apply_rotary_pos_emb` q+k 配对（`layout='BSND'`），拆装插回原位（中间和后面保持原值）。
- **kv_shared 路径**（q 单独，k 从 shared 来已 RoPE）：用 `apply_rotary_pos_emb_single`（`npu_rotary_mul` + cat/拆装 模拟 partial），避免对已 RoPE 的 k 再次旋转。
- `inv_freq` 对 proportional 不再补 0（长度 = `rope_angles = rotary_dim//2`），`cos/sin` 长度 = `rotary_dim`，由 full 层 cat/拆装消费（与参考实现一致；原"inv_freq 补 0 + 整 head_dim 过 rotary_mul"路径已废弃）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rotary_mul`（= `torch_npu.npu_rotary_mul`） | ✅ 已落地 | sliding 层（全旋）单 tensor；kv_shared 路径 full 层（cat/拆装模拟 partial）。GE 转换器 → `ge.RotaryMul` |
| `torch_npu.npu_apply_rotary_pos_emb`（q+k 配对） | ✅ 已落地 | full 层（非 kv_shared）配对 + cat/拆装，`layout='BSND'`。GE 转换器 → `ge.ApplyRotaryPosEmb` |

> 本仓 `fused_ops/ops.py` 还注册了自定义 `cann_lm::rope_in_place`（GE 算子 `RopeInPlace`），
> 可作为 RoPE 的项目内融合替代（见 converter/export/ge_converter.py）。

### 2.5 Eager Attention → `npu_fused_infer_attention_score_v2`

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
| `torch.ops.npu.npu_fused_infer_attention_score_v2` | ✅ 已落地 | cann_lm `fused_ops/fused_infer_attention_score.py` 注册 schema+fake+CPU eager（npu 命名空间）；`ge_converter.py` 注册 converter（调 `ge.FusedInferAttentionScore`，40+ 参数全传，None 给不用）；`reg_register.h` 加 `REG_OP(FusedInferAttentionScore)`（OM 编译注册到 GE OperatorFactory）。消除 `BatchMatMul`/`SoftmaxV2`（归零），`ConcatV2D` 从 14 降到 9（repeat_kv 消除） |
| `cann_lm::fused_infer_attention_score` | ❌ 未采用（文档原提） | 原文档提的 cann_lm 自定义 FA 算子（IR 级 fallback），实际改用 npu 命名空间（与图灵一致），未用 cann_lm 命名空间 |

> softcap 部分：参考实现把 `1/sqrt(d)` 吸收进 q_norm（`scale_fa=1.0`），本仓 `scaling=1.0`
> 已一致；logit softcap（`x/cap→tanh→x*cap`）无单一融合算子，`fused_attention_forward` 未传 softcap
> （gemma4 attention 无 softcap，logit softcap 在 `Gemma4ForCausalLM` 末尾，不在 attention 内）。

#### 遗留问题：融合算子引入权重时的支持方案（当前 gemma4 不涉及，量化/sink 场景需关注）

**背景**：当前 `FusedInferAttentionScore` 在 gemma4 中**无 weight 参数**（只接收 Q/K/V 张量 + 配置常量如 `sparse_mode`/`actual_seq_qlen`），因此切换 eager→fused 时**可复用旧 `weights_0.bin`**（q_proj/k_proj/v_proj/o_proj 的 weight 不变，只是中间计算算子从 `BatchMatMul`+`SoftmaxV2` 换成融合算子）。

**遗留场景**：若后续融合算子引入**可学习 weight**（如量化场景的 `antiquant_scale`/`antiquant_offset`/`dequant_scale_query`/`dequant_scale_key`/`dequant_scale_value`，或 attention sink 的 `learnable_sink`），则不能复用旧 `weights_0.bin` —— 新 weight 在旧 `.bin` 里没有，`weight_info.json` 也没索引，OM 编译时 Const 节点找不到权重。

**支持方案**（6 步，把融合算子权重暴露为 `nn.Parameter`，让 torch.export + `consolidated_weights_save` 自动 lift + 落盘）：

| 步骤 | 改动 | 说明 |
|------|------|------|
| 1. 模型定义 | `Attention.__init__` 加 `nn.Parameter`（如 `self.learnable_sink = nn.Parameter(torch.zeros(num_heads))`） | 跟 q_proj 等 nn.Linear 同级，torch.export 自动 lift |
| 2. 权重加载 | `load_gemma4_weights` 加 name 映射（HF → cann_lm） | HF 有则加载，无则用 `__init__` 初始化值 |
| 3. forward 传递 | `fused_attention_forward` 传新 weight（如 `learnable_sink=module.learnable_sink`） | 给融合算子 |
| 4. schema | `fused_ops/fused_infer_attention_score.py` 的 schema 加新输入（`learnable_sink` 等） | 已有部分可选输入（IR 定义含 `learnable_sink`），schema 简化版需补 |
| 5. GE converter | `ge_converter.py` 的 `_convert_fused_infer_attention_score_v2` 传新 weight（不是 None） | `REG_OP(FusedInferAttentionScore)` 已有 `learnable_sink` 等 OPTIONAL_INPUT，无需改 |
| 6. 落盘 | `--save-weights=True`（**不能复用旧 .bin**） | `consolidated_weights_save` 自动把新 weight 存到 `weights_0.bin` + 更新 `weight_info.json` |

**核心思路**：融合算子的 weight 暴露为 `nn.Parameter`（跟 q_proj 同级），让 `torch.export` + `consolidated_weights_save` 自动 lift + 落盘，无需手动管理 `weights_0.bin` 偏移/索引。**关键约束**：引入新 weight 后必须重新落盘（`--save-weights=True`），不能复用旧 `weights_0.bin`。

**触发条件**：当 `fused_attention_forward` 给融合算子传非 None 的 `learnable_sink`/`antiquant_scale`/`dequant_scale_*` 等参数时，需走此方案。当前 gemma4（纯 bf16，无量化，无 sink）全部传 None，不触发。

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
| P2 | `npu_rotary_mul` + `npu_apply_rotary_pos_emb`（已落地） | `apply_rotary_pos_emb` 配对 + cat/拆装 | sliding: `npu_rotary_mul` 单 tensor；full: `npu_apply_rotary_pos_emb` 配对（`layout='BSND'`）；kv_shared: `apply_rotary_pos_emb_single`（`npu_rotary_mul` + cat/拆装 模拟 partial） |
| P2 | `npu_scatter_nd_update_` | `index_copy` KV cache 写入 | 每层 2 处（k/v），IR 路径已通 |

> 注：本仓当前为纯 PyTorch 参考实现，引入 `torch_npu` 需遵循 AGENTS.md 的 stub 约定
> （converter 侧 torchair 编译 + lib 层打桩），NPU 实机验证走系统 Python 的 torch_npu。

---

# 第二部分：Qwen3.5 融合算子

> 数据来源：`cann-recipes-infer/models/qwen3_5/models/modeling_qwen3_5_moe.py`（3237 行）
> 本仓实现：`cann_lm/models/qwen3_5/modeling_qwen3_5.py`

Qwen3.5 是**混合架构**：full_attention 层（标准 GQA + KV cache）+ linear_attention 层
（GatedDeltaNet 线性注意力 + conv1d + recurrent state）。参考实现的融合算子重点在
**线性注意力**（`npu_chunk_gated_delta_rule` / `npu_recurrent_gated_delta_rule` / `causal_conv1d` /
`FusedRMSNormGated`），这也是本仓 `export_to_epair` segfault 的根因（纯 torch 的
`torch_chunk_gated_delta_rule` Python for 循环展开 19418 节点导致 GE 图爆炸）。

## 四、参考实现（cann-recipes-infer）使用的 torch_npu 算子汇总

共 **22** 个融合算子，按功能域分 6 类。线性注意力类（#1-#6）是 Qwen3.5 特有，其余与
Gemma-4 重叠（详见第一部分，此处只列差异）。

### 4.1 线性注意力类（Qwen3.5 特有，核心）

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

### 4.2 归一化类（RMSNorm，与 Gemma-4 重叠）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 7 | `torch_npu.npu_rms_norm` | `Qwen3_5MoeRMSNormGated.ln_npu`（行 885）、`Qwen3_5MoeRMSNorm.ln_npu`（行 2577） | `(x, weight, eps)[0]` | 融合 RMSNorm |
| 8 | `torch_npu.npu_add_rms_norm` | `Qwen3_5MoeRMSNorm.add_rms_norm_npu`（行 2615） | `(residual, x, gamma, eps)` 返回 `(y, _, residual)` | 融合 residual add + RMSNorm |
| 9 | `torch_npu.npu_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.rms_norm_mx`（行 2588） | `(x, gamma, epsilon, dst_type)` 返回 `(x_quant, x_scale, _)` | RMSNorm + MXFP8 量化输出（供下游量化 linear） |
| 10 | `torch_npu.npu_add_rms_norm_dynamic_mx_quant` | `Qwen3_5MoeRMSNorm.add_rms_norm_mx`（行 2605） | `(residual, x, gamma, epsilon, dst_type)` 返回 `(y, x_out, mxscale, _)` | add + RMSNorm + MXFP8 量化 |

### 4.3 激活类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 11 | `torch_npu.npu_swiglu` | `SiLUAndMul.forward`（行 742） | `(x)` | 融合 SiLU+Mul：`silu(gate) * up`（MLP gate_up_proj 后） |
| 12 | `torch_npu.npu_swiglu_mx_quant` | `SiLUAndMul.forward_mx`（行 746） | `(x, group_index, dst_type, activate_left)` 返回 `(out, scale)` | SiLU+Mul + MXFP8 量化（供量化 down_proj） |

### 4.4 Attention 类（full_attention 层）

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 13 | `torch_npu.npu_rotary_mul` | `Qwen3_5MoeAttention.forward`（行 1961） | `(q, cos, sin, rotary_mode='half')` | RoPE 融合（与 Gemma-4 一致） |
| 14 | `torch_npu.scatter_update_` | `Qwen3_5MoeAttention.forward`（行 1975） | `(cache, indices, updates, dim=-2)` in-place | KV cache 写入（替代 index_copy） |
| 15 | `torch.ops.npu.npu_fused_infer_attention_score` | `Qwen3_5MoeAttention.forward`（行 1979 prefill / 1998 decode） | prefill: `sparse_mode=3, atten_mask=...`；decode: `actual_seq_lengths_kv=..., atten_mask=None` | FlashAttention 融合（QK^T+softmax+AV+GQA+PagedAttention） |

### 4.5 量化类

| # | 算子 | 调用位置 | 关键参数 | 作用 |
|---|------|----------|----------|------|
| 16 | `torch_npu.npu_dynamic_mx_quant` | `_quantize_weight_last_dim`（行 286） | `(weight_2d, dst_type=float8_e4m3fn)` 返回 `(quant, scale)` | 权重在线 MXFP8 量化 |
| 17 | `torch_npu.npu_mm_all_reduce_base` | `qwen3_5_prefill_mm_all_reduce`（行 219） | `(input, weight, hcom, reduce_op="sum")` | MM + AllReduce 融合（仅 950，prefill 的 o_proj/down_proj） |

### 4.6 MoE 类（本仓 Qwen3.5 为 dense，不含 MoE；参考实现 MoE 版有 8 个，与 Gemma-4 第一部分 #8-#15 一致，略）

---

## 五、本仓 `modeling_qwen3_5.py` 可用的融合算子分析

本仓实现为纯 PyTorch，以下逐模块给出「当前实现 → 可替换融合算子」的映射。标注
`✅ 已落地` / `✅ 可直接替换` / `⚠️ 需重构（GE 算子）` / `❌ 不适用`。

### 5.1 线性注意力核心（GatedDeltaNet）— **segfault 根因，最高优先级**

**当前实现**（`Qwen3_5GatedDeltaNet.forward`，modeling_qwen3_5.py:415-480）：
- prefill：`torch_chunk_gated_delta_rule`（行 179-256，~78 行 Python for 循环 + 多次 matmul + cumsum + exp + masked_fill + solve_triangular）
- decode：`torch_recurrent_gated_delta_rule`（行 259-323，逐 token for 循环递归）

| 算子 | 适用性 | 说明 |
|------|--------|------|
| `torch_npu.npu_chunk_gated_delta_rule` | ⚠️ 需重构（GE 算子） | **解决 segfault 的关键**：纯 torch prefill 展开约 19418 节点 → `export_to_epair` GE 转换图爆炸 segfault。融合算子把整个 chunked GDR 流程打成单 GE 算子。需：① `fused_ops/` 注册 `cann_lm::chunk_gated_delta_rule`（schema + fake meta）；② `ge_converter.py` 映射到 `ge.ChunkGatedDeltaRule`（查 CANN op_info）；③ modeling 用 `torch.ops.cann_lm.chunk_gated_delta_rule` 替代 `torch_chunk_gated_delta_rule` |
| `torch_npu.npu_recurrent_gated_delta_rule` | ⚠️ 需重构（GE 算子） | decode 路径同上，纯 torch for 循环无法被 torch.export 优化。需注册 `cann_lm::recurrent_gated_delta_rule` + GE 映射 |
| `causal_conv1d_fn` / `causal_conv1d_update` | ⚠️ 需重构（GE 算子） | 当前 `torch_causal_conv1d_update`（行 163-176）= `torch.cat(state,x) + F.conv1d + F.silu`，3 个 kernel launch。可注册 `cann_lm::causal_conv1d` + GE 映射；或直接用 `aten::conv1d`（已有 GE 映射）+ `aten::silu` |
| `FusedRMSNormGated` | ⚠️ 需重构（GE 算子） | 当前 `Qwen3_5RMSNormGated`（行 31-44）= `pow/rsqrt` 纯 torch norm + `F.silu(gate) * x`。可拆解：norm 部分已有 `cann_lm::rms_norm`（→ `ge.RmsNorm`），gate 部分用 `aten::silu` + `aten::mul`；或注册融合 `cann_lm::rms_norm_gated` + GE 映射 `ge.RmsNormGated`（查 CANN 是否有） |

> **平台分支**：参考实现 `npu_chunk_gated_delta_rule` 只在非 950（910/A3）启用，950 走纯 torch。
> 本仓纯 torch 路径在 **910 实机推理**会低效（Python 循环），在 **export_to_epair 导出**会
> segfault（图爆炸）。两条路都需融合算子。

### 5.2 RMSNorm → `npu_rms_norm` / `npu_add_rms_norm`（已落地）

**当前实现**：`Qwen3_5RMSNorm.forward`（行 22-28）= `pow(2).mean + pow(-0.5) + mul` 纯 torch。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rms_norm`（= `npu_rms_norm`） | ✅ 已落地 | `fused_ops/rms_norm.py` 有 CPU eager + fake meta；`ge_converter.py:178` 映射 `ge.RmsNorm`。但 modeling_qwen3_5.py **未调用** `torch.ops.cann_lm.rms_norm`，仍是纯 torch（导出时走 aten 分解为 Rsqrt+Mul+ReduceMean）。可改为调 cann_lm 算子，GE 直接用融合 RmsNorm |
| `cann_lm::add_rms_norm`（= `npu_add_rms_norm`） | ✅ 已落地 | `fused_ops/add_rms_norm.py` + `ge_converter.py:189` 映射 `ge.AddRmsNorm`。modeling_qwen3_5.py 的 `Qwen3_5DecoderLayer.forward`（行 588）残差 add + norm 可改用此融合算子（需重构 forward 拓扑） |

### 5.3 RoPE → `npu_rotary_mul`（已落地）

**当前实现**：`Qwen3_5RotaryEmbedding`（行 68）+ attention 用 `apply_rotary_pos_emb`。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::rotary_mul`（= `npu_rotary_mul`） | ✅ 已落地 | `fused_ops/rotary_mul.py` + `ge_converter.py:202` 映射 `ge.RotaryMul`。与 Gemma-4 共用 |

### 5.4 KV Cache 写入 → `scatter_update_`（已落地，IR 路径通）

**当前实现**（`SimpleKVCache.update`，行 330-332）：`index_copy(2, new_kv_cache_pos, kv)`。

| 算子 | 状态 | 说明 |
|------|------|------|
| `aten::index_copy` → `ge.InplaceUpdate + TransposeD` | ✅ 已落地 | `ge_converter.py:234` 注册 `aten::index_copy` 的 GE 转换器（TensorMove + TransposeD + InplaceUpdate），IR 导出路径已通。实机可用 `npu_scatter_update_` |

### 5.5 Full Attention → `npu_fused_infer_attention_score`（已落地，未启用）

**当前实现**：`Qwen3_5Attention`（行 520-569）= eager attention（matmul QK^T + softmax + matmul AV + GQA repeat_kv）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `cann_lm::fused_infer_attention_score` | ⚠️ 需重构 | `fused_ops/fused_infer_attention_score.py` + `ge_converter.py:82` 映射 `ge.FusedInferAttentionScore`（含 block_table/num_key_value_heads/sparse_mode）。modeling_qwen3_5.py 的 full attention 可改用此融合算子（需重构 attention forward） |

### 5.6 激活 → `npu_swiglu`（未落地）

**当前实现**：`Qwen3_5MLP.forward`（行 569）= `F.silu(gate) * up` 纯 torch（或 `gate * silu(up)`）。

| 算子 | 状态 | 说明 |
|------|------|------|
| `npu_swiglu` | ❌ 未落地 | 可注册 `cann_lm::swiglu` + GE 映射 `ge.Swiglu`（查 CANN op_info）。优先级低，纯 torch 的 silu+mul 在 GE 导出时分解为 Swish/Sigmoid+Mul，已能导出 |

---

## 六、Qwen3.5 收益排序与建议落地顺序

按「解决导出 segfault / 实机推理提速」排序：

| 优先级 | 融合算子 | 替换目标 | 预期收益 |
|--------|----------|----------|----------|
| **P0** | `npu_chunk_gated_delta_rule`（→ `cann_lm::chunk_gated_delta_rule` + `ge.ChunkGatedDeltaRule`） | `torch_chunk_gated_delta_rule`（prefill） | **解决 export_to_epair segfault**：19418 节点 → 1 算子；实机 prefill 提速（单 kernel 替代 ~100 行循环） |
| **P0** | `npu_recurrent_gated_delta_rule`（→ `cann_lm::recurrent_gated_delta_rule` + GE 映射） | `torch_recurrent_gated_delta_rule`（decode） | 解决 decode 导出 + 实机 decode 提速（逐 token 循环 → 单算子 batch 递归） |
| P1 | `causal_conv1d_fn` / `causal_conv1d_update`（→ `cann_lm::causal_conv1d` + GE 映射） | `torch_causal_conv1d_update` | 线性注意力的 conv1d 融合（3 kernel → 1），导出 + 推理提速 |
| P1 | `FusedRMSNormGated`（→ `cann_lm::rms_norm_gated` + `ge.RmsNormGated`） | `Qwen3_5RMSNormGated`（norm + silu(gate) + mul） | 线性注意力 norm+gate 融合，或拆解用已有 `cann_lm::rms_norm` + `aten::silu` |
| P2 | `npu_rms_norm`（`cann_lm::rms_norm` 已落地，modeling 改调用） | `Qwen3_5RMSNorm` 纯 torch | 每层 4-6 个 RMSNorm，高频；GE 直接用融合 RmsNorm（而非分解 Rsqrt+Mul） |
| P2 | `npu_add_rms_norm`（`cann_lm::add_rms_norm` 已落地，modeling 改调用） | 残差 add + norm 两步 | 每层 2 处，需重构 forward 拓扑 |
| P3 | `npu_fused_infer_attention_score`（`cann_lm::fused_infer_attention_score` 已落地） | full attention eager（matmul+softmax+matmul） | 全注意力层融合，消除 repeat_kv + mask 构造 |
| P3 | `npu_swiglu`（未落地） | `F.silu(gate) * up` | MLP 激活融合，优先级低（纯 torch 已能导出） |

> **P0 是关键**：`chunk_gated_delta_rule` + `recurrent_gated_delta_rule` 是 Qwen3.5
> `export_to_epair` segfault 的唯二解（纯 torch for 循环无法被 torch.export 优化，图爆炸）。
> 实现路径：① `fused_ops/chunk_gated_delta_rule.py` 注册 schema + fake meta；② `ge_converter.py`
> 映射到 `ge.ChunkGatedDeltaRule`（查 CANN `aic-ascend910_93-ops-info-*.json` 的 op_info）；③
> `modeling_qwen3_5.py` 用 `torch.ops.cann_lm.chunk_gated_delta_rule` 替代 `torch_chunk_gated_delta_rule`。
> 参考 cann-recipes-infer 的 `npu_chunk_gated_delta_rule` 签名（行 1541）对齐参数。

---

## 七、平台差异速查（Qwen3.5）

| 平台 | `npu_chunk_gated_delta_rule` | fp8/mxfp8 量化 | `npu_mm_all_reduce_base` | MoE MC2 |
|------|------------------------------|----------------|---------------------------|---------|
| **950**（Atlas 950） | ❌ 走纯 torch prefill | ✅ 支持 | ✅ 支持 | group_type=0/3 |
| **910/A3**（非 950） | ✅ 融合算子 | ❌ 不支持 | ❌ A3 不支持 | `fullmesh_v2` |

> 本机 NPU 为 **Ascend910_9362**（非 950），支持 `npu_chunk_gated_delta_rule` 融合算子，
> 不支持 fp8/mxfp8 量化。融合算子落地后，910 实机推理可走 `npu_chunk_gated_delta_rule`，
> 950 走纯 torch fallback（但 950 纯 torch 在导出时仍会 segfault，需 950 专属解或保持
> `cann_lm::chunk_gated_delta_rule` 的 GE 映射让 950 也能用 GE 算子）。
