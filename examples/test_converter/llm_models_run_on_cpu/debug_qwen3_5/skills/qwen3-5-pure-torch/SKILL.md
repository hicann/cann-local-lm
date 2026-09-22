---
name: qwen3-5-pure-torch
description: |
  不依赖 transformers 库，纯 PyTorch 实现 Qwen3.5 模型定义，并用真实 checkpoint 调试与验证。当用户提到以下场景时使用：
  - 实现 / 编写 Qwen3.5 模型（纯 torch，无 transformers 依赖）
  - 从 HuggingFace 权重加载 Qwen3.5 到自定义实现
  - Qwen3.5 推理脚本（纯 torch）/ 出字
  - Qwen3.5 模型配置定义
  - 将 Qwen3.5 modeling 代码从 transformers 迁移到独立实现
  - 调试纯 torch Qwen3.5（加载真实权重、generate 退化、KV cache、causal mask）
  - 验证纯 torch Qwen3.5 与 HuggingFace transformers Qwen3.5ForCausalLM 数值对齐
version: 1.1.0
metadata:
  author: assistant
  category: model-implementation
  model: qwen3.5-0.8b
  reference: Qwen/Qwen3.5-0.8B
  allowedTools:
    - read
    - write
    - bash
    - task
    - question
    - webfetch
---

# Qwen3.5 纯 PyTorch 实现技能

## 前置条件

- Python >= 3.10
- PyTorch >= 2.0（仅依赖 `torch`，不依赖 `transformers`）
- safetensors（用于权重加载）

## Qwen3.5 架构概览

Qwen3.5 是一个混合架构模型，核心特点是 **线性注意力 + 全注意力交替层**，并支持多模态（文本 + 视觉）。

| 组件 | 说明 |
|------|------|
| **Text Model** | 文本语言模型（decoder-only），支持 GQA、QK Norm + Q Gate、混合线性/全注意力层、MRoPE |
| **Vision Model** | ViT 视觉编码器（继承 Qwen3VL），支持 3D RoPE、空间合并、bilinear 位置插值 |

### Text Model 关键特性

1. **混合层类型**：`layer_types` 配置交替使用 `full_attention`（标准多头注意力）和 `linear_attention`（GatedDeltaNet 线性注意力），默认每 4 层一个 full_attention
2. **Q Gate 机制**：`q_proj` 输出 2 倍维度，split 出 query 和 gate，gate 经 sigmoid 后调制 attention output
3. **QK Normalization**：Q 和 K 投影后各自经过 RMSNorm（centered: `1 + weight`）
4. **GatedDeltaNet 线性注意力**：基于 gated delta rule，支持 chunk 模式（prefill）和 recurrent 模式（decode），含 causal conv1d、recurrent state 缓存
5. **MRoPE（多模态旋转位置编码）**：3D 位置编码（temporal/height/width），interleaved 模式，partial_rotary_factor=0.25

### GatedDeltaNet 关键特性（Qwen3.5 vs Qwen3Next）

Qwen3.5 将 Qwen3Next 的合并投影拆分为 4 个独立投影：

| Qwen3Next | Qwen3.5 |
|-----------|---------|
| `in_proj_qkvz` (key_dim*2 + value_dim*2) | `in_proj_qkv` (key_dim*2 + value_dim) + `in_proj_z` (value_dim) |
| `in_proj_ba` (num_v_heads * 2) | `in_proj_b` (num_v_heads) + `in_proj_a` (num_v_heads) |

## 实施步骤

### 第 1 步：理解 Transformers 源码

查阅 HuggingFace transformers 仓库和 CANN-LM 中的源文件：

```
CANN-LM/cann_lm/models/qwen3_5/
├── configuration_qwen3_5.py   # Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
├── modeling_qwen3_5.py        # 完整模型实现（自动生成，约 2100 行）
├── modular_qwen3_5.py         # 模块化定义（手动维护）
├── tokenization_qwen3_5.py    # Tokenizer

CANN-LM/cann_lm/models/qwen3_next/
├── configuration_qwen3_next.py   # Qwen3NextConfig（Qwen3_5TextConfig 的基类）
├── modeling_qwen3_next.py        # GatedDeltaNet, MoE 等核心组件
```

继承链（transformers 参考）：`Qwen3NextRMSNorm → Qwen3_5RMSNorm`，`Qwen3NextAttention(+Q Gate) → Qwen3_5Attention`

### 第 2 步：创建目录结构

在 `CANN-LM/cann_lm/models/qwen3_5/` 下创建实现文件，仿照 gemma4 的标准 Python 包结构：

```
CANN-LM/cann_lm/models/qwen3_5/
├── __init__.py                    # 包导出
├── configuration_qwen3_5.py       # 模型配置（纯 dataclass，不继承 PreTrainedConfig）
├── modeling_qwen3_5.py            # 纯 PyTorch 模型实现（文本 CausalLM）
# 单元测试在 tests/llm_models/test_modeling_qwen3_5.py（按 tests/ 约定）

CANN-LM/cann_lm/models/skills/qwen3-5-pure-torch/
├── SKILL.md                       # 本文件（技能描述，不含 Python 代码）
```

### 第 3 步：实现 Configuration

不继承 `PreTrainedConfig`，使用纯 Python dataclass 定义配置。

**Qwen3_5TextConfig** 核心参数：

```python
@dataclass
class Qwen3_5TextConfig:
    model_type: str = "qwen3_5_text"
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    # 线性注意力参数
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # 混合层类型
    layer_types: list[str] | None = None  # 默认: 每4层一个 full_attention
    full_attention_interval: int = 4

    # RoPE 参数
    rope_parameters: dict = field(default_factory=lambda: {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    })

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
```

### 第 4 步：实现基础组件（纯 PyTorch）

所有组件只依赖 `torch` 和 `torch.nn`。

#### 4.1 RMSNorm（centered: 1 + weight）

```python
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # centered: 初始化为 0

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # centered RMSNorm
        return output.type_as(x)
```

#### 4.2 RMSNormGated（用于 GatedDeltaNet 输出）

```python
class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
```

#### 4.3 MRoPE（多模态旋转位置编码）

```python
class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        rope_params = config.rope_parameters
        self.rope_type = rope_params.get("rope_type", "default")
        base = rope_params.get("rope_theta", 1000000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
```

#### 4.4 旋转嵌入辅助函数

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

#### 4.5 Eager Attention Forward（含 Q Gate）

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

#### 4.6 MLP

```python
class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 第 5 步：实现 GatedDeltaNet（线性注意力）

这是 Qwen3.5 最核心的新组件。纯 PyTorch 实现需包含 torch fallback 版本的 chunk 和 recurrent 算子。

#### 5.1 辅助函数

```python
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)
```

#### 5.2 torch_chunk_gated_delta_rule

参见 `modeling_qwen3_5.py` 第 247-325 行的完整实现，核心逻辑：

1. 将 Q/K/V/beta/g 按 chunk_size 分块
2. 计算衰减掩码 `decay_mask = ((g.cumsum.unsqueeze - g.cumsum.unsqueeze).tril().exp()).tril()`
3. 计算 intra-chunk attention: `attn = -((k_beta @ k.T) * decay_mask).masked_fill(upper, 0)`
4. 递推修正: `attn[i, :i] = row + (row * sub).sum`
5. 对每个 chunk: `core_attn = attn_inter + attn @ v_new`, 更新 recurrent state

#### 5.3 torch_recurrent_gated_delta_rule

单步递推版本（用于 decode）：

```python
def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm=False):
    ...
    for i in range(sequence_length):
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    ...
```

#### 5.4 GatedDeltaNet Module

```python
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False,
                                 kernel_size=self.conv_kernel_size, groups=self.conv_dim,
                                 padding=self.conv_kernel_size - 1)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
```

forward 流程：
1. `in_proj_qkv` → causal conv1d → split 为 Q/K/V
2. `in_proj_z` → gate; `in_proj_b` → beta (sigmoid); `in_proj_a` → decay (softplus)
3. 计算 `g = -A_log.exp() * softplus(a + dt_bias)`
4. Q/K repeat_interleave 到 num_v_heads
5. 单 token: recurrent_gated_delta_rule; 多 token: chunk_gated_delta_rule
6. norm(core_attn_out, z) → out_proj

### 第 6 步：实现 Attention Layer（含 Q Gate）

```python
class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        bsz, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            scaling=self.scaling, dropout=0.0 if not self.training else self.config.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)  # Q Gate
        attn_output = self.o_proj(attn_output)
        return attn_output
```

### 第 7 步：实现 DecoderLayer

```python
class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states, attention_mask=attention_mask,
                past_key_values=past_key_values, position_embeddings=position_embeddings,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

### 第 8 步：实现 TextModel

```python
class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=False):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        if position_ids is None:
            past_len = 0
            if past_key_values is not None:
                past_len = past_key_values.get_seq_length()
            position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)

        # 简化版 causal mask
        causal_mask = None
        if attention_mask is not None:
            causal_mask = _create_causal_mask(attention_mask, inputs_embeds, past_key_values)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask if self.config.layer_types[i] == "full_attention" else attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

### 第 9 步：实现 ForCausalLM

```python
class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits, "past_key_values": past_key_values}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=1.0, do_sample=True, eos_token_id=None):
        past_kv = None
        for _ in range(max_new_tokens):
            out = self(input_ids=input_ids, past_key_values=past_kv)
            past_kv = out["past_key_values"]
            logits = out["logits"][:, -1, :]
            if do_sample and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids
```

### 第 10 步：权重加载

```python
from safetensors.torch import load_file

def load_qwen3_5_weights(model, model_path, device="cpu", dtype=torch.bfloat16):
    import json, os
    param_dict = {k: v for k, v in model.named_parameters()}
    loaded = set()

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    for shard in shards:
        shard_path = os.path.join(model_path, shard)
        state = load_file(shard_path, device="cpu")
        filtered = {k: v.to(device=device, dtype=dtype) for k, v in state.items() if k in param_dict}
        model.load_state_dict(filtered, strict=False)
        loaded.update(filtered.keys())

    missing = set(param_dict) - loaded
    if missing:
        print(f"[WARN] {len(missing)} keys missing: {list(missing)[:5]}...")
    return model
```

### 第 11 步：推理脚本

```python
"""Qwen3.5 inference using pure PyTorch model definition."""
import argparse, torch
from configuration_qwen3_5 import Qwen3_5TextConfig
from modeling_qwen3_5 import Qwen3_5ForCausalLM, load_qwen3_5_weights
from transformers import AutoTokenizer  # tokenizer 可复用

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, do_sample=True):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eos = tokenizer.eos_token_id
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, eos_token_id=eos)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--prompt", default="Hello! Tell me about yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 加载 config.json 并构造配置
    config = Qwen3_5TextConfig.from_json_file(os.path.join(args.model, "config.json"))
    model = Qwen3_5ForCausalLM(config)
    model = load_qwen3_5_weights(model, args.model, device, torch.bfloat16)
    model.eval()

    out = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"\n=== Output ===\n{out}")

if __name__ == "__main__":
    main()
```

## 与现有 transformers 实现的差异要点

| Transformers | 纯 PyTorch 实现 |
|---|---|
| 继承 `PreTrainedModel`, `GenerationMixin` | 继承 `nn.Module`，手动实现 generate |
| 使用 `DynamicCache` | 用简化版 KV cache（list 或 dict） |
| 使用 `ALL_ATTENTION_FUNCTIONS` 分派 sdpa/flash | 直接使用 eager attention |
| 配置继承 `PreTrainedConfig` / `Qwen3NextConfig` | 使用 Python dataclass |
| 依赖 `transformers.activations.ACT2FN` | 直接使用 `nn.SiLU()` |
| 依赖 `causal_conv1d`, `flash_linear_attention` | 使用 torch fallback 实现 |
| Vision 模型继承 `Qwen3VLVisionModel` | 文本模型优先实现，Vision 暂不实现 |

## KV Cache 简化方案

GatedDeltaNet 需要两种 cache：conv_state 和 recurrent_state。简化实现：

```python
class SimpleDeltaNetCache:
    def __init__(self, num_layers, conv_dim, conv_kernel_size, k_head_dim, v_head_dim, num_v_heads):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                "conv_states": torch.zeros(1, conv_dim, conv_kernel_size),
                "recurrent_states": torch.zeros(1, num_v_heads, k_head_dim, v_head_dim),
            })

    def has_previous_state(self, layer_idx):
        return self.layers[layer_idx]["conv_states"].sum() != 0

    def update_conv_state(self, new_state, layer_idx):
        self.layers[layer_idx]["conv_states"] = new_state

    def update_recurrent_state(self, new_state, layer_idx):
        self.layers[layer_idx]["recurrent_states"] = new_state
```

全注意力层的 KV cache 用简单的 dict/list 管理。

## 验证步骤（实况：Qwen3.5-0.8B VLM checkpoint）

用真实 VLM checkpoint（`Qwen/Qwen3.5-0.8B`，单分片 `model.safetensors-00001-of-00001.safetensors`，text 权重前缀 `model.language_model.*`，tied embeddings，无 `lm_head.weight`）调试时发现并修复的 4 个 bug：

### Bug 1：`load_qwen3_5_weights` 不做前缀重映射（致命）

- VLM checkpoint 的 text 权重 key 是 `model.language_model.layers.0.*` / `model.language_model.embed_tokens.weight` / `model.language_model.norm.weight`，而纯 torch 文本 CausalLM 的 param 名是 `model.layers.0.*` / `model.embed_tokens.weight` / `model.norm.weight`。
- 原 loader 只按 `k in param_dict` 过滤、不做映射 → **0 个 key 命中**，模型全是随机权重。
- 修复：加 `_remap_key`，`model.language_model.` → `model.`；visual/mtp 不在 param_dict 自然被丢弃；兜底 glob `model.safetensors-0000*-of-*.safetensors` 单分片命名；tied 时 `lm_head.weight` 从 missing 集合剔除。

```python
def _remap_key(k: str) -> str:
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k
```

### Bug 2：config 缺字段 → `from_pretrained` 静默丢弃

- 0.8B `text_config` 含 `attn_output_gate=true`、`mtp_num_hidden_layers`、`mtp_use_dedicated_embeddings`、`mamba_ssm_dtype`、`mlp_only_layers`、`rope_parameters.mrope_interleaved`，原 dataclass 没这些字段 → `from_dict` 的 `valid_keys` 过滤把它们扔了。
- 修复：dataclass 补全字段。`attn_output_gate=True` 与 full-attention 层 `q_proj` 输出 2×（query+gate）一致——Q Gate 必须保留。

### Bug 3：causal mask 缺失 → generate 预填充泄漏未来 token

- `TextModel.forward` 原来只在 `attention_mask is not None` 时建 mask；`generate` 默认传 `attention_mask=None` → **full-attention 层无因果屏蔽**，预填充每个 token 看到未来 token。
- 修复：`_create_causal_mask` 接受 `attention_mask=None`（纯因果，无 padding）；forward 预填充时**总建 mask**，decode（`seq_len==1 and past_len>0`）跳过（单 query 可看全部缓存）。

```python
past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
if seq_len == 1 and past_len > 0:
    causal_mask = None
else:
    causal_mask = _create_causal_mask(attention_mask, inputs_embeds, past_key_values)
```

### Bug 4：`SimpleKVCache.get_seq_length` KeyError

- 原取 `self._cache[0][0].shape[2]`，但 `layer_types[0]=='linear_attention'`（无 KV），第一层 KV 实际在 layer 3 → `KeyError: 0`。
- 修复：取任意已缓存 key：`first_key = next(iter(self._cache)); return self._cache[first_key][0].shape[2]`。

### 验证脚本

| 脚本 | 用途 |
|------|------|
| `cann_lm/models/run_qwen3_5.py` | 纯 torch 模型 + HF tokenizer 出字 |
| `cann_lm/models/qwen3_5/verify_qwen3_5_vs_hf.py` | 与 HF `Qwen3_5ForCausalLM` 逐 token 对拍（float32 + eager） |
| `cann_lm/hf_cpu.py` | 禁用 transformers 的 torch_npu probe（本环境必做，先于 `from transformers import`） |

```bash
# 出字
cd cann_lm/models && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 run_qwen3_5.py \
    --model /path/to/qwe3.5-0.8B --greedy --max-new-tokens 64
# 对拍
cd cann_lm/models/qwen3_5 && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 verify_qwen3_5_vs_hf.py \
    --model /path/to/qwe3.5-0.8B
```

### 验证结果（0.8B，float32 + eager）

| 指标 | 结果 |
|------|------|
| 参数量 | 752,393,024 == HF |
| embed_tokens 权重 | bit-identical（max\|Δ\|=0） |
| logits（chat prompt 19 token） | max\|Δ\| = **0.000e+00**，argmax 100%，top5 100% |
| logits（随机 32 token） | max\|Δ\| = **0.000e+00**，argmax 100% |
| prefill cache-vs-nocache | 8.5e-05（fp 噪声） |
| greedy 16 token | **identical=True** |
| 出字 | "你好！我是 Qwen3.5，由通义实验室自主研发的超…"（与 HF 一致） |

> HF 因没装 `flash-linear-attention`/`causal-conv1d` 回退到 torch 实现，与纯 torch 同算法，故对拍公平且 bit-identical。

### 调试技巧：先判 bug 在 forward 还是 generate

1. **退化输出（如 `at-at-at`）≠ forward 错**。先做 `forward(ids, use_cache=False)` 与 HF 对拍：若 bit-identical 则 forward 全对（含 GatedDeltaNet chunk prefill、interleaved MRoPE、Q Gate），bug 在 generate 的 cache 路径。
2. 再查 `forward(ids, use_cache=True)` vs `forward(ids, use_cache=False)`：若 prefill-with-cache ≠ no-cache，则 cache 预填充路径坏。
3. **单 token（纯 eos）prompt 易出退化输出**，调试生成务必用真实 chat template prompt（`tokenizer.apply_chat_template`）。

## 常见问题

### 权重 key 不匹配（VLM checkpoint 前缀）
- Qwen3.5 VLM checkpoint 的 text 权重前缀是 `model.language_model.*`（不是 `model.*`），纯 torch 文本 CausalLM 的 param 名是 `model.*`。
- 必须 `model.language_manager.` → `model.` 重映射，否则 **0 key 命中**（loader 不报错但全是随机权重，表现为输出乱码/退化）。
- `model.visual.*` / `mtp.*` 不在文本模型 param_dict，自然被过滤丢弃。
- tied（`tie_word_embeddings=true`）时 checkpoint 不存 `lm_head.weight`，加载后 `model.lm_head.weight = model.model.embed_tokens.weight` 绑定。

### config 字段被 `from_pretrained` 静默丢弃
- `from_dict` 只保留 dataclass 已声明字段，checkpoint 里多出的字段（`attn_output_gate`/`mtp_*`/`mamba_ssm_dtype`/`mlp_only_layers`/`mrope_interleaved`）会被无声扔掉。
- dataclass 必须补全这些字段，否则 `attn_output_gate` 丢了也不会报错，但后续若用它判断 Q Gate 会出错。

### generate 预填充泄漏未来 token（causal mask）
- `TextModel.forward` 若仅在 `attention_mask is not None` 时建 mask，而 `generate` 默认 `attention_mask=None` → full-attention 层预填充时无因果屏蔽。
- 表现：forward(no_cache) 与 HF 对拍可能 bit-identical（因为 HF 也可能传 None 走同一路径？），但 generate 的 prefill 与 no-cache forward 不一致，生成质量差。
- 修复：`_create_causal_mask(None,...)` 产出纯因果 mask；forward 预填充总建 mask。

### `SimpleKVCache.get_seq_length` KeyError: 0
- 混合层型模型里 layer 0 是 `linear_attention`（用 delta_cache，无 KV cache），第一层 KV 在第一个 `full_attention` 层（如 layer 3）。
- `get_seq_length` 不能假设 key 0 存在，要取任意已缓存 key。

### partial_rotary_factor = 0.25
- Qwen3.5 只对 head_dim 的 25% 应用 RoPE（256 的 1/4 = 64 维）
- `apply_rotary_pos_emb` 中 `rotary_dim = cos.shape[-1]`，剩余维度 pass-through

### GatedDeltaNet 的 conv1d 和 recurrent state
- Prefill: 使用 `torch_chunk_gated_delta_rule`
- Decode (seq_len=1): 使用 `torch_recurrent_gated_delta_rule`
- conv_state 用于 causal conv1d 的左上下文缓存

### 本环境用 transformers 必须先禁用 torch_npu probe
- `import cann_lm.hf_cpu`（在 `from transformers import ...` 之前），否则 transformers 的 `is_torch_npu_available()` 会因 `libhccl_compat.so` 缺失而 `import torch_npu` 致命报错。
- 跑脚本统一加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
