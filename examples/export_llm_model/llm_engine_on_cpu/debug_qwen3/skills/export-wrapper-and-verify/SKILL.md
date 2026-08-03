# ExportWrapper 验证 Skill

> 验证 ExportWrapper 精度：wrapper 输出 vs 原始模型 vs HF transformers
>
> 适用模型：qwen3 (已验证), qwen3.5 (部分验证), gemma4 (待验证)

## 触发条件

当需要验证 ExportWrapper 的精度、对比 HF transformers 输出、或调试 wrapper 导出问题时使用此 skill。

## 验证工具

- `bash`：运行验证脚本
- `read`：读取验证结果

## 验证层次

### 第 1 层：单步 forward 精度 (verify_wrapper.py)

验证 wrapper 的单步 forward 输出与原始模型一致：

```bash
cd examples/export_llm_model
python verify_wrapper.py
```

预期：`max_diff=0.000000e+00`，wrapper 与原始模型完全一致。

### 第 2 层：多轮 decode 精度 (verify_wrapper_qwen3_5_decode.py)

验证 prefill + decode + 多轮生成的 KV cache 传递正确：

```bash
python verify_wrapper_qwen3_5_decode.py
```

预期：每步 `max_diff=0.000000e+00`，生成的 token 序列完全一致。

### 第 3 层：wrapper vs HF transformers (verify_wrapper_vs_hf.py)

验证 cann_lm 自定义模型与 transformers 标准实现的等价性：

```bash
pip install transformers  # 如果未安装
python verify_wrapper_vs_hf.py
```

预期：`max_diff` 在 0.02~0.05 范围内（float16 数值差异），**生成的 token 序列一致**。

### 第 4 层：epair 导出 vs HF (verify_epair_vs_hf.py)

验证 torch.export → epair 导出后的精度保持：

```bash
python verify_epair_vs_hf.py
```

预期：`max_diff` 在 0.02~0.04 范围内，**生成的 token 序列一致**。

## 环境准备

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
_PYTHON_LIB_DIR="/opt/buildtools/python-3.12.9/lib"
_TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
export LD_LIBRARY_PATH="${_PYTHON_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="/mnt/workspace/torchair_lazy_venv/bin/python3"
```

## 调试经验

### 1. `use_cache` 参数不存在

**现象**: `TypeError: forward() got an unexpected keyword argument 'use_cache'`

**原因**: `Qwen3ForCausalLM.forward()` 不接受 `use_cache`（这是 `Qwen3Model.forward()` 的参数）。

**解决**: wrapper 的 `_drive_layers` 中不传 `use_cache`。

### 2. KV cache 形状不匹配

**现象**: `RuntimeError: Sizes of tensors must match except in dimension 2`

**原因**: 模型用 `torch.cat([cache, new_keys], dim=2)` 拼接 cache 和新 keys。cache 的维度顺序必须与 key_states 一致：`[batch, num_kv_heads, seq_len, head_dim]`。

**解决**: `get_sample_inputs` 中 cache 形状用 `[batch, num_kv_heads, kv_cache_max_len, head_dim]`，不是 `[kv_cache_max_len, num_kv_heads, batch, head_dim]`。

### 3. mask 与 cache+keys 总长度不匹配

**现象**: `RuntimeError: The size of tensor a (2056) must match the size of tensor b (2048)`

**原因**: 预分配 cache (2048) + 新 keys (8) = 2056，但 mask 只有 2048。

**解决**: 轻量版用 `past_key_values=None` + causal mask `[1,1,seq_len,seq_len]`。不预分配 KV cache，让模型内部创建。

### 4. `past_key_values=None` 导致无法展平

**现象**: `TypeError: 'NoneType' object is not iterable`

**原因**: 传 `None` 时模型不创建 cache，返回 `past_key_values=None`。

**解决**: wrapper 的 `_drive_layers` 中 `None → [None]*num_layers` 触发模型内部 cache 创建。展平时检查 `if layer_kv is not None`。

### 5. qwen3.5 只有 6 层 full attention

**现象**: `IndexError: list index out of range`

**原因**: qwen3.5 是混合架构（6 full + 18 linear），SimpleKVCache 只有 6 层条目，不是 24 层。

**解决**: 遍历用 `for idx, layer_i in enumerate(orig_layers)`，`idx` 索引 wrapper 展平列表，`layer_i` 索引原始 cache。

### 6. qwen3.5 的 4D mask 透传缺失

**现象**: `_create_causal_mask` 总是动态构造 mask，torch.export 无法追踪。

**原因**: qwen3 和 gemma4 有 `return attention_mask`（4D 透传），qwen3.5 没有。

**解决**: 给 `cann_lm/models/qwen3_5/modeling_qwen3_5.py` 的 `_create_causal_mask` 添加：
```python
if attention_mask is not None and attention_mask.dim() == 4:
    return attention_mask
```

### 7. epair_load 后无法用 None 调用

**现象**: `AttributeError: 'NoneType' object has no attribute 'size'`

**原因**: `torch.export` 用 `past_key_values=None` 追踪时，None 的处理逻辑（`None → [None]*N`）是 Python 代码，epair 序列化/反序列化后丢失。

**解决**: prefill 用 `ep_direct.module()(*traced_inputs)`（直接用导出时的输入），decode 用 wrapper 直接调用。

### 8. 导出和推理用了不同的随机输入

**现象**: logits 差异 13+，token 序列完全不同。

**原因**: `get_sample_inputs` 内部 `torch.randint` 生成了一组 input_ids，`main()` 中又 `torch.randint` 生成了另一组。

**解决**: 推理时用 `traced_inputs[0]`（导出时的 input_ids），确保 HF 和 ep_direct 用相同输入。

### 9. output_pos 选位导致 logits shape 不匹配

**现象**: ep_direct 输出 `[1,1,vocab]`（有 output_pos 选位），HF 输出 `[1,4,vocab]`（全位置）。

**解决**: 对比改为 `logits_ep` vs `logits_hf[:, -1:, :]`。

### 10. qwen3.5 SimpleKVCache 无法 torch.export

**现象**: torch.export 追踪 SimpleKVCache 的 dict 操作失败。

**原因**: `SimpleKVCache._cache` 是 Python dict，`dict.__setitem__` 和 dict 查找是 Python 控制流，torch.export 无法静态追踪。

**解决方向**: 注册 pytree，让 torch.export 把 SimpleKVCache 当作 `k_0,v_0,k_1,v_1,...` 的 tensor 集合。`SimpleDeltaNetCache`（线性注意力）也需要类似处理。

### 11. HF DynamicCache API 不兼容

**现象**: `TypeError: 'DynamicCache' object is not subscriptable` 或 `AttributeError: 'DynamicCache' object has no attribute 'key_cache'`

**原因**: transformers 5.13.1 的 `DynamicCache` 不支持 `kv[i]` 索引，也没有 `key_cache`/`value_cache` 属性。正确 API 是 `kv.layers[i].keys` 和 `kv.layers[i].values`。

**解决**: 对比 KV cache 时用：
```python
k_hf = kv_hf.layers[i].keys
v_hf = kv_hf.layers[i].values
```

### 12. 图外查表 (use_embeddings) 与图内查表的精度对比

**现象**: 图外查表模式 logits max_diff=0.037，与图内查表 (0.023) 在同一量级。

**原因**: 两种模式都经过同一个模型 forward，差异来源相同（RMSNorm/RoPE/attention 的 float16 实现）。图外查表的 embedding 查表本身完全一致 (diff=0.0)。

**结论**: 图外查表功能等价，可用于 embedding 量化场景。

## 验证脚本索引

| 脚本 | 用途 | 模型 | 状态 |
|------|------|------|------|
| `verify_wrapper.py` | 单步 forward 精度 | qwen3-0.6B | PASS (diff=0.0) |
| `verify_wrapper_qwen3_5.py` | 单步 forward 精度 | qwe3.5-0.8B | PASS (diff=0.0) |
| `verify_wrapper_qwen3_5_decode.py` | 多轮 decode 精度 | qwen3-0.6B | PASS (diff=0.0, tokens match) |
| `verify_wrapper_vs_hf.py` | wrapper vs HF transformers | qwen3-0.6B | PASS (diff=0.047, tokens match) |
| `verify_epair_vs_hf.py` | epair 导出 vs HF | qwen3-0.6B | PASS (diff=0.023, tokens match) |
| `verify_wrapper_vs_hf_embeddings.py` | 图外查表 vs HF | qwen3-0.6B | PASS (diff=0.037, tokens match) |

## 注意事项

- qwen3.5 多轮 decode 在 CPU 上超时（752M + 248K vocab），用 qwen3-0.6B 替代
- transformers 需要额外安装：`pip install transformers`
- 所有验证脚本需要 `PYTHONUNBUFFERED=1` 避免输出缓冲
- epair_load 回环对 `past_key_values=None` 有序列化问题，prefill 用 ep_direct
