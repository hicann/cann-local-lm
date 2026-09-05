---
name: gemma4-pure-torch
description: |
  不依赖 transformers 库，纯 PyTorch 实现 Gemma-4 模型定义。当用户提到以下场景时使用：
  - 实现 / 编写 Gemma-4 模型（纯 torch，无 transformers 依赖）
  - 从 HuggingFace 权重加载 Gemma-4 到自定义实现
  - Gemma-4 推理脚本（纯 torch）
  - Gemma-4 模型配置定义
  - 将 Gemma-4 modeling 代码从 transformers 迁移到独立实现
version: 1.0.0
metadata:
  author: assistant
  category: model-implementation
allowedTools:
  - read
  - write
  - bash
  - task
  - question
  - webfetch
---

# Gemma-4 纯 PyTorch 实现技能

## 前置条件

- Python >= 3.10
- PyTorch >= 2.0（仅依赖 `torch`，不依赖 `transformers`）
- safetensors（用于权重加载）

## Gemma-4 架构概览

Gemma-4 是一个多模态模型，包含三个子模型：

| 组件 | 说明 |
|------|------|
| **Text Model** | 文本语言模型（decoder-only），支持 GQA、Soft-Capping、QK Norm、KV Cache 共享、多种 RoPE 类型 |
| **Vision Model** | ViT 视觉编码器，支持多维 (2D) RoPE、QK Norm、QKV Norm、空间池化 |
| **Audio Model** | 音频编码器（复杂），支持分块局部注意力+相对位置偏置、LightConv1d、双 FFN |

### Text Model 关键特性

1. **多种 RoPE 类型**：不同 layer 可使用不同 rope_type（`default`/`proportional`），不同层可有不同的 `head_dim`（`head_dim` vs `global_head_dim`）
2. **QK Normalization**：Q 和 K 投影后各自经过 RMSNorm
3. **Soft-Capping**：attention logits 除以 softcap → tanh → 乘回 softcap
4. **KV Cache 共享**：最后 n 层的 K/V 可以共享，支持 double-wide MLP
5. **混合注意力模式**：`full_attention` + `sliding_window_attention`

### Vision Model 关键特性

1. **多维 (2D) RoPE**：patch 有 (x, y) 坐标，RoPE 分别作用于两个空间维度
2. **QK Norm + V Norm**：Q、K、V 都经过 Norm
3. **无 scaling**：attention scaling = 1.0（不使用 head_dim^-0.5）
4. **PatchEmbedder**：input_proj + 2D 可学习位置嵌入表
5. **空间池化 (Pooler)**：将 patch tokens 池化到固定数量的 soft tokens

## 实施步骤

### 第 1 步：理解 Transformers 源码

查阅 HuggingFace transformers 仓库中的源文件，理解完整架构：

```
transformers/src/transformers/models/gemma4/
├── configuration_gemma4.py   # Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig, Gemma4AudioConfig
├── modeling_gemma4.py        # 完整模型实现
├── modular_gemma4.py         # 模块化定义（自动生成 modeling）
├── processing_gemma4.py      # 处理器
```

### 第 2 步：创建目录结构

在 `CANN-LM-py/models/` 下创建 gemma4 目录，参考现有 qwen3/qwen3_5 结构：

```
CANN-LM-py/models/gemma4/
├── __init__.py
├── configuration_gemma4.py    # 模型配置（纯 dataclass，不继承 PreTrainedConfig）
├── modeling_gemma4.py         # 纯 PyTorch 模型实现
└── tokenization_gemma4.py     # Tokenizer（可复用 transformers 的 Gemma tokenizer）
```

### 第 3 步：实现 Configuration

不继承 `PreTrainedConfig`，使用纯 Python dataclass 或普通类定义配置。

**Gemma4TextConfig** 核心参数：

```python
class Gemma4TextConfig:
    model_type = "gemma4_text"
    vocab_size: int = 256000
    hidden_size: int = 2560      # 或 4608 等
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 128   # 全局注意力层使用不同的 head_dim
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_logit_softcapping: float = 50.0
    attn_logit_softcapping: float | None = None  # per-layer override
    query_pre_attn_scalar: float = 224.0  # 或不同值
    attn_layer_types: list[str] = None   # ["full_attention", "sliding_window_attention", ...]
    sliding_window: int = 1024
    sliding_window_pattern: int = 6  # 每 N 层一个 sliding window
    num_kv_shared_layers: int = 0   # KV 共享层数
    use_double_wide_mlp: bool = False
    rope_parameters: dict = {}   # 包含每种 layer_type 的 rope 配置
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
```

**Gemma4VisionConfig** 核心参数：

```python
class Gemma4VisionConfig:
    model_type = "gemma4_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 72
    patch_size: int = 14
    position_embedding_size: int = 256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0
    use_clipped_linears: bool = False
    rope_parameters: dict = {"rope_type": "default", "rope_theta": 10000.0}
```

### 第 4 步：实现基础组件（纯 PyTorch）

所有组件只依赖 `torch` 和 `torch.nn`。

#### 4.1 RMSNorm

```python
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.float() * torch.pow(rms, -1.0)  # 与 JAX 保持一致用 pow
        if self.with_scale:
            x = x * self.weight.float()
        return x.type_as(x)
```

#### 4.2 RoPE（Text Model）

需要支持多种 rope_type，不同 layer_type 使用不同配置：

```python
class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.layer_types = set(config.attn_layer_types)
        # 为每种 layer_type 独立计算 inv_freq
        self._inv_freqs: dict[str, torch.Tensor] = {}
        
        for layer_type in self.layer_types:
            rope_cfg = config.rope_parameters.get(layer_type, {})
            rope_type = rope_cfg.get("rope_type", "default")
            base = rope_cfg.get("rope_theta", 10000.0)
            
            if layer_type == "full_attention" and rope_type == "proportional":
                dim = config.global_head_dim
            else:
                dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids, layer_type: str = "full_attention"):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

#### 4.3 旋转嵌入辅助函数

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

#### 4.4 Eager Attention Forward（含 Soft-Capping）

```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    softcap: float | None = None,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

#### 4.5 MLP

```python
class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Gemma4 使用 gelu_pytorch_tanh，近似等价于 GELU
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 第 5 步：实现 Attention Layer

关键区别于 Qwen3：
- 使用 `query_pre_attn_scalar` 而非 `head_dim^-0.5`
- Q/K 投影后各自 Norm
- 支持 soft-capping
- KV cache sharing（最后 n 层的 K/V 可复用）

```python
class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.attn_layer_types[layer_idx]
        self.is_global = self.layer_type == "full_attention"
        self.head_dim = config.global_head_dim if self.is_global else config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5 if self.is_global else self.head_dim ** -0.5
        self.softcap = config.attention_logit_softcapping if self.is_global else None
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,  # 简化版 KV cache
    ):
        bsz, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # 简化 KV cache 更新（生产环境应使用 DynamicCache）
            key_states = torch.cat([past_key_values[self.layer_idx][0], key_states], dim=2)
            value_states = torch.cat([past_key_values[self.layer_idx][1], value_states], dim=2)
        past_key_values[self.layer_idx] = (key_states, value_states)

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            softcap=self.softcap,
        )

        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
```

### 第 6 步：实现 DecoderLayer

混合注意力模式 + KV Cache 共享：

```python
class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.attn_layer_types[layer_idx]

        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
    ) -> torch.Tensor:
        # Pre-Attention Norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Pre-FFN Norm
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 第 7 步：实现 TextModel

```python
class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.Tensor = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device).unsqueeze(0)

        # 构建 causal mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        # 为每个 layer 计算 position_embeddings（不同 layer_type 可能不同）
        # 简化实现：所有层用 full_attention 的 rope
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids, "full_attention")

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

### 第 8 步：实现 ForCausalLM

```python
class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": past_key_values}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, **kwargs):
        """简化的贪心推理"""
        past_key_values = None
        for _ in range(max_new_tokens):
            outputs = self(input_ids=input_ids, past_key_values=past_key_values)
            past_key_values = outputs["past_key_values"]
            next_token = outputs["logits"][:, -1:].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == self.model.config.eos_token_id).any():
                break
        return input_ids
```

### 第 9 步：权重加载

从 HuggingFace safetensors 权重文件加载到自定义模型：

```python
from safetensors.torch import load_file

def load_gemma4_weights(model: Gemma4ForCausalLM, model_path: str, device: str = "cpu"):
    import json, os

    param_dict = dict(model.named_parameters())
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
        filtered = {k: v for k, v in state.items() if k in param_dict}
        model.load_state_dict(filtered, strict=False)
        loaded.update(filtered.keys())

    missing = set(param_dict) - loaded
    if missing:
        print(f"[WARN] {len(missing)} keys missing: {list(missing)[:5]}...")

    return model
```

### 第 10 步：推理脚本

创建 `CANN-LM-py/run_gemma4.py`，参考 `run_qwen3_5.py` 的模式：

```python
"""Gemma-4 inference using local CANN-LM-py/models/gemma4/ model definition."""

import json
import os
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")

from models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers import AutoTokenizer  # 仍可复用 transformers 的 tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512, 
             temperature: float = 1.0, do_sample: bool = False):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma-4 inference")
    parser.add_argument("--model", type=str, default="google/gemma-4-9b-it")
    parser.add_argument("--prompt", type=str, default="Hello! Tell me about yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model: {args.model}")
    config = Gemma4TextConfig.from_pretrained(args.model)
    model = Gemma4ForCausalLM(config)
    model = load_gemma4_weights(model, args.model, device)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print("Generating...")
    out = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
    print(f"\n=== Output ===\n{out}")


if __name__ == "__main__":
    main()
```

## 与现有 transformers 实现的差异要点

| Transformers | 纯 PyTorch 实现 |
|---|---|
| 继承 `PreTrainedModel`, `GenerationMixin` | 继承 `nn.Module`，手动实现 generate |
| 使用 `Cache`, `DynamicCache` | 用 `list[tuple[Tensor]]` 简化 KV cache |
| 使用 `ALL_ATTENTION_FUNCTIONS` 分派 sdpa/flash | 直接使用 eager attention |
| 配置继承 `PreTrainedConfig` | 使用 Python dataclass / 普通类 |
| 依赖 `transformers.activations.ACT2FN` | 直接使用 `nn.GELU(approximate="tanh")` |

## KV Cache 共享 (可选高级特性)

Gemma-4 的 KV 共享是通过在 forward 中将某层的 K/V 替换为之前层的 K/V 实现的。在自定义实现中可以在 `Gemma4TextModel.forward` 的层循环里处理：

```python
# 在 layer loop 中
if layer_idx >= config.num_hidden_layers - config.num_kv_shared_layers:
    shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers - 1
    # 复用 shared_layer_idx 的 K/V
    ...
```

## Sliding Window Attention (可选)

Gemma-4 中部分层使用 `sliding_window_attention`，其 causal mask 只允许当前 token 关注最近 `sliding_window` 个 token。实现方式：

```python
def create_sliding_window_mask(seq_len: int, window_size: int, device, dtype):
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[..., i, start:i+1] = 0
    return mask
```

## 验证步骤

1. **形状验证**：随机输入，验证各层输出形状正确
2. **权重对齐**：加载 HuggingFace 权重后，与 transformers 输出逐 token 比对
3. **推理测试**：用真实 prompt 测试生成质量

## 常见问题

### 权重 key 不匹配
- HuggingFace 权重 key 前缀通常为 `model.language_model.` 或 `language_model.`
- 自定义模型直接使用 `model.layers.*`，需要做 key 映射
- 解决方案：在 `load_state_dict` 前进行 key 重映射

### head_dim 混用
- `full_attention` 层使用 `global_head_dim`，`sliding_window_attention` 层使用 `head_dim`
- 确保每层 Attention 初始化时用正确的 `head_dim`

### RoPE 频率计算
- 实际实现中 `inv_freq` 计算方式：`1.0 / (base ** (torch.arange(0, dim//2, 2, dtype=torch.float) / dim))`
- 注意：遍历范围是 `dim//2` 而不是 `dim`（与 HuggingFace 一致）
