# 如何用纯 PyTorch 编写 LLM 模型 (以 qwen3 为例)

> 不依赖 transformers 的 PreTrainedModel/PreTrainedConfig，用纯 `nn.Module` + `@dataclass` 实现完整 LLM
>
> 参考实现：`cann_lm/models/qwen3/`

## 触发条件

当需要用纯 PyTorch 编写新的 LLM 模型定义（不使用 transformers 基类）时使用此 skill。

## 目录结构

```
cann_lm/models/qwen3/
├── __init__.py                  # 导出 config, model, load_weights
├── configuration_qwen3.py       # config dataclass
├── modeling_qwen3.py           # model class + load_qwen3_weights
└── export_wrapper.py            # ExportWrapper 子类 (见 export-wrapper-and-verify skill)
```

## 1. Config: 纯 dataclass

```python
@dataclass
class Qwen3TextConfig:
    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, path: str) -> "Qwen3TextConfig":
        """从 HF config.json 构造"""
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3TextConfig":
        """过滤未知字段后构造"""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
```

**关键点：**
- `from_pretrained` / `from_dict` / `to_dict` 三个方法必须实现（converter 依赖）
- `from_dict` 过滤未知字段 — HF config.json 可能有多余字段
- `model_type` 必须与 config.json 中的 `model_type` 一致（converter 用它自动探测）

## 2. Model: nn.Module 子类

### 2.1 ForCausalLM (顶层)

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.model = Qwen3TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                output_hidden_states=False) -> dict:
        model_output = self.model(input_ids, attention_mask, position_ids,
                                 past_key_values, inputs_embeds)
        hidden_states = model_output[0]
        past_key_values = model_output[1]
        logits = self.lm_head(hidden_states)
        return {"loss": None, "logits": logits,
                "past_key_values": past_key_values,
                "hidden_states": None}
```

**关键点：**
- 不继承 `PreTrainedModel`，直接 `nn.Module`
- forward 返回 **dict**（不是 ModelOutput dataclass）
- `past_key_values` 是 `list[(k,v),...]`（不是 Cache 对象）

### 2.2 TextModel (中层)

```python
class Qwen3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(config)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = self._prepare_position_ids(inputs_embeds, past_key_values)

        causal_mask = self._prepare_causal_mask(inputs_embeds, attention_mask, past_key_values)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask, past_key_values)
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

### 2.3 _prepare_causal_mask — 4D 透传

```python
def _prepare_causal_mask(self, inputs_embeds, attention_mask, past_key_values):
    bsz, seq_len = inputs_embeds.shape[:2]

    # decode 单步 (seq_len==1 + 有 cache) → 不需要 mask
    if past_key_values is not None and past_key_values[0] is not None and seq_len == 1:
        return None

    # None → 动态构造 causal mask
    if attention_mask is None:
        mask = torch.triu(torch.full((seq_len, seq_len), min_val), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # 2D → 构造完整 mask (含 padding)
    if attention_mask.dim() == 2:
        ...  # 略

    # 4D → 直接透传 (torch.export 友好)
    return attention_mask
```

**关键点：** 4D 透传 (`return attention_mask`) 是必须的 — ExportWrapper 传 4D mask 时走透传路径，避免动态构造。

### 2.4 RMSNorm

```python
class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 用 x*x 替代 x.pow(2) (数值等价, trace 更稳定)
        variance = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

### 2.5 Attention

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        q = self.q_proj(hidden_states).view(...).transpose(1, 2)
        k = self.k_proj(hidden_states).view(...).transpose(1, 2)
        v = self.v_proj(hidden_states).view(...).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache (list-based, torch.export 友好)
        if past_key_values is not None:
            cache = past_key_values[self.layer_idx]
            if cache is not None and cache[0] is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            past_key_values[self.layer_idx] = (k, v)

        # repeat_kv for GQA
        k = repeat_kv(k, num_heads // num_kv_heads)
        v = repeat_kv(v, num_heads // num_kv_heads)

        # attention
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.o_proj(out.transpose(1, 2).reshape(...))
        return out, None
```

**关键点：**
- `past_key_values` 是 `list[(k,v) or None]` — 纯 tensor，不用 Cache 对象
- `torch.cat([cache, new])` 拼接 cache — torch.export 可追踪
- GQA 用 `repeat_kv`（显式 Concat 或 Expand）

## 3. Weight Loading

```python
def load_qwen3_weights(model, model_path, device="cpu", dtype=None):
    """从 HF safetensors 加载权重"""
    sd = load_safetensors_file(model_path)  # 读 model.safetensors

    param_dict = dict(model.named_parameters())
    for k, v in sd.items():
        if k in param_dict and v.shape == param_dict[k].shape:
            param_dict[k].data.copy_(v.to(device=device, dtype=dtype))

    # Tied weights
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model
```

**关键点：**
- 不用 `load_state_dict(strict=True)` — 允许部分缺失
- key 直接匹配（qwen3 不需要重映射）
- Tied weights 手动设置

## 4. __init__.py 导出

```python
from .configuration_qwen3 import Qwen3TextConfig
from .modeling_qwen3 import Qwen3ForCausalLM, load_qwen3_weights
from .export_wrapper import Qwen3ExportWrapper
```

**关键点：** `load_qwen3_weights` 必须导出 — converter 的 `model_registry.py` 需要导入它。

## 5. 与 transformers 的差异

| 方面 | transformers | 纯 PyTorch (我们) |
|------|-------------|-------------------|
| Config 基类 | `PreTrainedConfig` | `@dataclass` |
| Model 基类 | `PreTrainedModel` | `nn.Module` |
| KV cache | `DynamicCache` 对象 | `list[(k,v),...]` |
| Causal mask | `masking_utils.create_causal_mask` | 自己的 `_prepare_causal_mask` (含 4D 透传) |
| Weight loading | `from_pretrained` (内置) | 自己的 `load_*_weights` |
| forward 返回 | `ModelOutput` dataclass | `dict` |
| RoPE | 动态计算 | 可预缓存表 |

## 6. 验证与 HF 的一致性

编写 `verify_qwen3_vs_hf.py`：
1. 加载 HF 模型 `Qwen3ForCausalLM.from_pretrained(path)`
2. 加载自定义模型 `load_qwen3_weights(model, path)`
3. 相同输入跑两遍
4. 对比 logits — float16 下差异应在 0.02~0.05 范围
5. 对比生成的 token 序列 — 应完全一致
