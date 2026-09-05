# Qwen3.5 ExportWrapper 验证 Skill

> 验证 Qwen3.5 ExportWrapper 精度：单步 forward + 多轮 decode + HF transformers 对比 + epair 导出
>
> 模型：qwe3.5-0.8B (752M params, 248K vocab, 24 层混合架构: 6 full + 18 linear attention)
>
> 最后验证：2026-07-16

## 触发条件

当需要验证 qwen3.5 ExportWrapper、调试 SimpleKVCache/torch.export 问题、或对比 HF transformers 输出时使用此 skill。

## 验证工具

- `bash`：运行验证脚本
- `read`：读取验证结果

## 环境准备

```bash
# 用例环境前置 (① converter venv + torchair stubs; 详见 llm_models_run_on_cpu/check_env.sh)
# 一次性 (需三个 HF 路径都 export): source ../../../check_env.sh
# 或单独: source <proj_root>/scripts/set_env_export.sh (基底已设 PYTHONPATH/AUTOLOAD/LD_LIBRARY_PATH)
PYTHON="${CONVERTER_PYTHON}"   # 基底已设 (venv python)
```

## 验证层次

### 第 1 层：单步 forward 精度 (verify_wrapper_qwen3_5.py)

验证 wrapper 单步 forward 输出与原始模型一致：

```bash
cd examples/test_converter/llm_models_run_on_cpu/debug_qwen3_5
python verify_wrapper_qwen3_5.py
```

预期：`max_diff=0.000000e+00`，6 层 full-attention KV cache 全部匹配。

### 第 2 层：全流程验证 (verify_qwen3_5_all.py)

一站式验证 4 层精度（单步 + decode + HF 对比 + epair 导出）：

```bash
python verify_qwen3_5_all.py
```

预期结果：
```
Layer 1 (单步 vs 原始):    PASS (diff=0.0)
Layer 2 (多轮 decode):     PASS (diff=0.0, tokens match)
Layer 3 (wrapper vs HF):   PASS (diff=0.019, tokens match)
Layer 4 (epair vs HF):     PASS (diff=0.019, tokens match)
Overall: ALL PASS
```

### 第 3 层：单独验证脚本

| 脚本 | 用途 |
|------|------|
| `verify_wrapper_qwen3_5.py` | 单步 forward (wrapper vs 原始模型) |
| `verify_qwen3_5_all.py` | 全流程 4 层一站式验证 |

## 验证脚本索引

| 脚本 | 用途 | 模型 | 状态 |
|------|------|------|------|
| `verify_wrapper_qwen3_5.py` | 单步 forward 精度 | qwe3.5-0.8B | PASS (diff=0.0) |
| `verify_qwen3_5_all.py` | 全流程 4 层验证 | qwe3.5-0.8B | ALL PASS |

## 调试经验

### 1. SimpleKVCache dict 操作 → torch.export 不可追踪

**现象**: torch.export 追踪 SimpleKVCache 的 dict 操作失败

**原因**: `SimpleKVCache._cache` 是 Python dict，`dict.__setitem__` 和 dict 查找是 Python 控制流，torch.export 无法静态追踪。

**解决**: 改为 list + tensor 属性（对齐 transformers DynamicCache 设计）：

```python
# 修改前 (dict, 不可追踪):
class SimpleKVCache:
    def __init__(self):
        self._cache = {}
    def update(self, key, value, layer_idx):
        if layer_idx in self._cache:           # dict 查找
            cached_key, cached_value = self._cache[layer_idx]  # dict 访问
            key = torch.cat([cached_key, key], dim=2)
        self._cache[layer_idx] = (key, value)   # dict 赋值

# 修改后 (list, 可追踪):
class SimpleKVCache:
    def __init__(self):
        self.keys = []      # list[Tensor | None]
        self.values = []    # list[Tensor | None]
    def update(self, key, value, layer_idx):
        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]
```

**文件**: `cann_lm/models/qwen3_5/modeling_qwen3_5.py`

### 2. torch.all 数据依赖 → torch.export 不可追踪

**现象**: `GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)`

**原因**: `torch.all(attention_mask == 1)` 检查 mask 是否全 1，这是数据依赖控制流，torch.export 无法在追踪时求值。

**解决**: 简化为仅检查 cache 状态（保留 prefill 传 mask 逻辑）：

```python
# 修改前 (数据依赖):
if (past_key_values is not None and past_key_values.get_seq_length() > 0) or (
    attention_mask is not None and torch.all(attention_mask == 1)  # ← 不可追踪
):
    linear_attn_mask = None

# 修改后 (非数据依赖):
linear_attn_mask = attention_mask
if past_key_values is not None and past_key_values.get_seq_length() > 0:
    linear_attn_mask = None
```

**文件**: `cann_lm/models/qwen3_5/modeling_qwen3_5.py`

### 3. KV cache 层索引映射错误

**现象**: Layer 1 (单步) PASS，但 Layer 2 (多轮 decode) FAIL — decode 步 token 不一致

**原因**: qwen3.5 是混合架构，full_attention 层的索引是 [3,7,11,15,19,23]（不是连续的 [0,1,2,3,4,5]）。wrapper 展平 KV cache 时丢失了层索引信息，重建时用了连续索引 → cache 存错位置。

**解决**: 通过 `config.layer_types` 映射正确层索引：

```python
def _get_full_attn_layer_indices(self):
    """从 config.layer_types 获取 full_attention 层的索引"""
    layer_types = getattr(self.config, "layer_types", [])
    return [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

def _drive_layers(self, ...):
    cache = SimpleKVCache()
    if past_key_values is not None:
        full_attn_indices = self._get_full_attn_layer_indices()
        for idx, (k, v) in enumerate(past_key_values):
            layer_idx = full_attn_indices[idx]  # 映射到正确层索引
            cache.update(k, v, layer_idx)
```

**文件**: `cann_lm/models/qwen3_5/export_wrapper.py`

### 4. logits shape 不匹配（output_pos）

**现象**: wrapper 输出 `[1,1,vocab]`（有 output_pos 选位）vs 原始模型 `[1,4,vocab]`（全位置）

**解决**: 对比时取最后位置 `logits_wrap[:, -1:, :]` vs `logits_orig[:, -1:, :]`

### 5. forward 不接受 position_ids

**现象**: `TypeError: Qwen3_5ForCausalLM.forward() got an unexpected keyword argument 'position_ids'`

**原因**: qwen3.5 的 ForCausalLM.forward() 没有 `position_ids` 参数（内部计算）。与 qwen3 不同。

**解决**: 原始模型调用时不传 `position_ids`。wrapper 的 `_drive_layers` 也不传给模型。

### 6. HF DynamicCache API 不兼容

**现象**: `TypeError: 'DynamicCache' object is not subscriptable`

**解决**: transformers 5.13.1 的 DynamicCache 正确 API:
```python
k_hf = kv_hf.layers[i].keys
v_hf = kv_hf.layers[i].values
```

### 7. 4D mask 透传缺失

**现象**: qwen3.5 的 `_create_causal_mask` 总是动态构造 mask，torch.export 无法追踪

**解决**: 添加 4D 透传（与 qwen3/gemma4/transformers 对齐）:
```python
def _create_causal_mask(attention_mask, inputs_embeds, past_key_values):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask  # 4D 透传
    ...
```

**文件**: `cann_lm/models/qwen3_5/modeling_qwen3_5.py`

## qwen3.5 vs qwen3 关键差异

| 差异点 | qwen3 | qwen3.5 |
|--------|-------|---------|
| 架构 | 28 层全 full attention | 24 层混合 (6 full + 18 linear) |
| cache 类型 | `list[(k,v),...]` | `SimpleKVCache` (改 list 后可追踪) |
| KV cache 层索引 | 连续 [0..27] | 不连续 [3,7,11,15,19,23] |
| position_ids | ForCausalLM.forward 接受 | ForCausalLM.forward 不接受（内部计算） |
| delta_cache | 无 | 有（线性注意力 recurrent state，当前传 None） |
| 4D mask 透传 | 已有 | 需新增 |
| torch.all 检查 | 无 | 有（需移除） |
| torch.export | 直接可用 | 需修复 3 个问题后可用 |

## 注意事项

- qwen3.5 模型较大 (752M + 248K vocab)，CPU 上推理慢，seq_len 用 4 而非 8
- transformers 需安装: `pip install transformers`
- HF model 的 linear attention fallback warning 可忽略（缺少 flash-linear-attention 库）
- 所有验证脚本需要 `PYTHONUNBUFFERED=1` 避免输出缓冲
- `verify_qwen3_5_all.py` 是一站式脚本，运行一次即可完成 4 层验证
