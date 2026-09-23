# 如何用纯 PyTorch 编写 LLM 模型 (以 qwen3.5 为例)

> qwen3.5 的特殊之处：混合架构 (full + linear attention)、SimpleKVCache (list-based)、delta_cache
>
> 参考实现：`cann_lm/models/qwen3_5/`

## 触发条件

当需要编写混合架构（full + linear attention）的纯 PyTorch LLM 模型时使用此 skill。

## 目录结构

```
cann_lm/models/qwen3_5/
├── __init__.py
├── configuration_qwen3_5.py    # config (含 layer_types)
├── modeling_qwen3_5.py         # model + SimpleKVCache + SimpleDeltaNetCache
└── export_wrapper.py           # Qwen3_5ExportWrapper
```

## qwen3.5 关键特性

| 方面 | qwen3.5 |
|------|---------|
| 架构 | 混合 (6 full + 18 linear) |
| `layer_types` | `["linear","linear","linear","full","linear",...]` |
| KV cache | `SimpleKVCache` (list-based, torch.export 友好) |
| Linear attention cache | `SimpleDeltaNetCache` (conv_state + recurrent_state) |
| ForCausalLM.forward | 不接受 `position_ids`（内部计算） |
| `_create_causal_mask` | 需手动添加 4D 透传 |
| `linear_attn_mask` | 有（`torch.all(mask==1)` 检查，需移除避免数据依赖） |

## 1. Config — 增加 layer_types

```python
@dataclass
class Qwen3_5TextConfig:
    # ... 基础字段（vocab/hidden/layers 等） ...
    full_attention_interval: int = 4  # 每 N 层一个 full attention
    layer_types: list[str] = None     # 计算得出

    def __post_init__(self):
        if self.layer_types is None:
            n = self.num_hidden_layers
            self.layer_types = [
                "full_attention" if i % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(n)
            ]
```

**关键点：** `layer_types` 决定哪些层用 full attention（有 KV cache），哪些用 linear attention（有 delta_cache）。

## 2. SimpleKVCache — list-based (torch.export 友好)

```python
class SimpleKVCache:
    """用 list + tensor 属性，不用 dict（torch.export 可追踪）"""
    def __init__(self):
        self.keys: list[torch.Tensor | None] = []
        self.values: list[torch.Tensor | None] = []

    def update(self, key_states, value_states, layer_idx):
        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]
```

**对比 transformers DynamicCache：**
```python
# transformers DynamicLayer (同样用 tensor 属性 + torch.cat):
class DynamicLayer:
    def __init__(self):
        self.keys = torch.tensor([])
        self.values = torch.tensor([])
    def update(self, key, value):
        self.keys = torch.cat([self.keys, key], dim=-2)
        self.values = torch.cat([self.values, value], dim=-2)
        return self.keys, self.values
```

设计一致：list 存储 + tensor 属性 + `torch.cat` 拼接。

## 3. SimpleDeltaNetCache — 线性注意力 cache

```python
class SimpleDeltaNetCache:
    """线性注意力的 recurrent state (delta rule 增量更新)"""
    def __init__(self, num_layers, conv_dim, conv_kernel_size,
                 k_head_dim, v_head_dim, num_v_heads, ...):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'conv_state': torch.zeros(num_v_heads, conv_dim, k_head_dim + v_head_dim),
                'recurrent_state': None,  # 延迟初始化
            })

    def update_conv_state(self, new_state, layer_idx):
        self.layers[layer_idx]['conv_state'] = new_state

    def update_recurrent_state(self, new_state, layer_idx):
        self.layers[layer_idx]['recurrent_state'] = new_state
```

**注意：** 当前 ExportWrapper 传 `delta_cache=None`（轻量版不支持线性注意力 cache）。后续需注册 pytree 才能让 torch.export 追踪。

## 4. ForCausalLM.forward — 不接受 position_ids

```python
class Qwen3_5ForCausalLM(nn.Module):
    def forward(self, input_ids, attention_mask=None,
                past_key_values=None, delta_cache=None,
                inputs_embeds=None, labels=None, use_cache=False):
        # 注意: 没有 position_ids 参数!
        # model 内部计算 position_ids
        hidden_states, past_key_values, delta_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            delta_cache=delta_cache,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return {"logits": logits, "past_key_values": past_key_values,
                "delta_cache": delta_cache}
```

## 5. TextModel.forward — 混合架构调度

```python
class Qwen3_5TextModel(nn.Module):
    def forward(self, input_ids, attention_mask, past_key_values, delta_cache, use_cache):
        # 内部计算 position_ids
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(past_len, past_len + seq_len, ...)
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)
        text_position_ids = position_ids[0]

        # causal mask (4D 透传)
        causal_mask = _create_causal_mask(attention_mask, inputs_embeds, past_key_values)

        # linear_attn_mask: 不能用 torch.all (数据依赖)
        linear_attn_mask = attention_mask
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            linear_attn_mask = None  # decode 不需要 linear mask

        # 遍历层 — 按 layer_types 分发
        for i, layer in enumerate(self.layers):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = layer(hidden_states, position_embeddings, layer_mask,
                                  past_key_values, delta_cache)
```

## 6. _create_causal_mask — 4D 透传

```python
def _create_causal_mask(attention_mask, inputs_embeds, past_key_values):
    # 4D 透传 (必须! torch.export 友好)
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    # None → 动态构造 (torch.export 不可追踪, 但 wrapper 不走这条路)
    if attention_mask is None:
        ...
    # 2D → 构造完整 mask
    if attention_mask.dim() == 2:
        ...
```

## 7. DecoderLayer — 混合分发

```python
class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if config.layer_types[layer_idx] == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.self_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        # ... mlp, norm 等

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values, delta_cache):
        if isinstance(self.self_attn, Qwen3_5Attention):
            # full attention: 用 past_key_values (SimpleKVCache)
            attn_output, _ = self.self_attn(hidden_states, position_embeddings,
                                             attention_mask, past_key_values)
        else:
            # linear attention: 用 delta_cache (SimpleDeltaNetCache)
            attn_output, _ = self.self_attn(hidden_states, attention_mask,
                                             cache_params=delta_cache)
        ...
```

## 8. 需要修复的 torch.export 问题

| 问题 | 原始代码 | 修复 |
|------|---------|------|
| SimpleKVCache dict | `self._cache = {}` + dict 赋值 | 改为 list + tensor 属性 |
| torch.all 数据依赖 | `torch.all(attention_mask == 1)` | 简化为仅检查 cache 状态 |
| 4D 透传缺失 | 无透传，总是动态构造 | 添加 `if mask.dim() == 4: return mask` |

## 9. 验证

```bash
# 单步 + 全流程 4 层验证
python verify_qwen3_5_all.py

# 预期:
# Layer 1 (单步): PASS (diff=0.0)
# Layer 2 (decode): PASS (diff=0.0, tokens match)
# Layer 3 (vs HF): PASS (diff=0.019, tokens match)
# Layer 4 (epair): PASS (diff=0.019, tokens match)
```
