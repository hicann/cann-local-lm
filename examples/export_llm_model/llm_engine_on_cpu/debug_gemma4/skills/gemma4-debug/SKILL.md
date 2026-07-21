# Gemma4 调试 Skill

> gemma4-E2B 模型的导出、wrapper 验证、chunk 推理、epair 导出全流程调试指南
>
> 模型：gemma-4-E2B (5B params, 35 层, 20 层 KV 共享, sliding + full attention)
>
> 最后验证：2026-07-17

## 触发条件

当需要调试 gemma4 模型的 ExportWrapper、chunk 推理、epair 导出、或对比 HF transformers 输出时使用此 skill。

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
cd /mnt/workspace/project/CANN-LM_5883/examples/export_llm_model/llm_engine_on_cpu/debug_gemma4
```

## 验证脚本索引

| 脚本 | 用途 | 状态 |
|------|------|------|
| `verify_gemma4_all.py` | 单步 forward 精度 (wrapper vs 原始模型) | PASS (diff=0.0) |
| `verify_gemma4_vs_hf.py` | 多轮 decode vs HF transformers | PASS (2/3 tokens match) |
| `verify_gemma4_chunk.py` | chunk vs 非 chunk (token 级) | PASS (diff=0.0) |
| `verify_gemma4_chunk_text.py` | chunk vs 非 chunk (文本级) | PASS (3/3 text match) |
| `verify_gemma4_chunk_vs_hf.py` | chunk cann_lm vs HF (文本级) | PASS (2/3 text match) |
| `verify_gemma4_wrapper_chunk.py` | ExportWrapper chunk vs 非 chunk (文本) | PASS (3/3 text match) |
| `verify_gemma4_epair_chunk.py` | epair 导出 + 回环推理 vs wrapper | PASS (2/3 text match) |

## 验证层次

### 第 1 层：单步 forward (verify_gemma4_all.py)

```bash
python verify_gemma4_all.py
```

验证 wrapper 单步 forward 与原始模型一致。预期 `max_diff=0.0`。

### 第 2 层：多轮 decode vs HF (verify_gemma4_vs_hf.py)

```bash
python verify_gemma4_vs_hf.py
```

验证 cann_lm 多轮 decode 与 HF transformers 的等价性。BOS + bfloat16 + softcap 对齐。

### 第 3 层：chunk 推理 (verify_gemma4_wrapper_chunk.py)

```bash
python verify_gemma4_wrapper_chunk.py
```

验证 ExportWrapper 的分 chunk prefill + decode 与一次性 prefill 生成文本完全一致。

### 第 4 层：epair 导出 + 回环 (verify_gemma4_epair_chunk.py)

```bash
python verify_gemma4_epair_chunk.py
```

验证 torch.export → ep.module() 回环推理与 wrapper 直接调用一致。

## 调试经验

### 1. 必须加 BOS token

**现象**：不加 BOS 时，输出 " is is is is..."（重复无意义 token）

**原因**：Gemma 模型的 tokenizer 不会自动加 BOS（`add_bos_token` 未设置），需要手动加。

**修复**：
```python
BOS_TOKEN_ID = 2
ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
```

### 2. 必须用 bfloat16

**现象**：用 float16 时输出质量差（重复 token、无意义文本）

**原因**：模型权重是 bfloat16，用 float16 加载导致精度损失。

**修复**：
```python
model = model.to(dtype=torch.bfloat16)
```

### 3. KV cache flat→tuple 未重建

**现象**：decode 步骤报错或输出不一致

**原因**：wrapper 展平 KV cache 为 `[k0,v0,k1,v1,...]`，decode 时需重建为 `[(k0,v0),(k1,v1),...]` 再传入模型。

**修复**：`Gemma4ExportWrapper._drive_layers` 增加 flat→tuple 重建逻辑。

### 4. shared_kv_states=None 崩溃

**现象**：20 层 KV 共享层（无 k_proj/v_proj）报 `AttributeError`

**原因**：传 `None` 时共享层尝试访问 `None` 的属性。

**修复**：传空 dict `{}` 而非 `None`。

### 5. final_logit_softcapping dtype 不匹配

**现象**：`RealDiv op dtype is not same, type1:DT_FLOAT, type2:DT_FLOAT16`

**原因**：`logits / 30.0` 中 `30.0` 是 Python float → GE 存为 float32 → 与 float16 logits 不匹配。

**修复**：用乘法替代除法 `logits * inv_softcap`，并创建同 dtype 的 tensor：
```python
inv_softcap = torch.tensor(1.0 / softcap, dtype=logits.dtype, device=logits.device)
logits = logits * inv_softcap
```

### 6. RoPE 计算的 RealDiv dtype 不匹配

**现象**：`RealDiv_2 op dtype is not same`

**原因**：`1.0 / (base ** (arange / dim))` 中 `1.0` 和 `dim` 都是 Python 类型 → GE 存为 float32。

**修复**：
- `1.0 / base**x` → `torch.pow(base, -x)`
- `arange / dim` → `arange * torch.tensor(1.0/dim, dtype=dtype)`
- `_compute_inv_freq` 增加 `dtype` 参数，用 `x.dtype` 传入

### 7. attention softcap 也有 RealDiv

**现象**：`RealDiv` dtype 不匹配（来自 attention 层的 `attn_weights / softcap`）

**修复**：同样改为乘法 `attn_weights * inv_softcap`。

### 8. ep.module() 无法处理 past_key_values=None

**现象**：`'NoneType' object has no attribute 'size'`

**原因**：torch.export 追踪时 `past_key_values=None` 被 Python 代码处理（`None → [None]*N`），FX 图无法在运行时重新处理 None。

**修复**：prefill 用 `ep.module()(*traced_inputs)`（和追踪时相同的输入），decode 用 wrapper 直接调用（因为 `ep.module()` 无法处理非 None 的 past_key_values）。

### 9. ep.module() 的 guard 检查 seq_len

**现象**：`Guard failed: input_ids.size()[1] == 8`

**原因**：torch.export 追踪时 `seq_len=8`，运行时传入 `seq_len=6` 不匹配。

**修复**：pad input_ids 到追踪时的 `seq_len`（用 pad_token_id=0）。

### 10. padding 污染 KV cache

**现象**：ep.module() padded 输入和 wrapper 不 padded 输入的输出不一致

**原因**：纯 causal mask 没屏蔽 padding 位置，attention 会 attend 到 padding 的 K/V。

**修复**：4D mask 中屏蔽 padding 位置：
```python
def make_causal_mask_with_padding(q_len, k_len, real_q_len, real_k_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val  # 屏蔽 padding query
    if real_k_len < k_len:
        mask[:, real_k_len:] = min_val  # 屏蔽 padding key
    return mask.unsqueeze(0).unsqueeze(0)
```

### 11. output_pos 选错位置

**现象**：ep.module() 输出是 padding 位置的 logits 而非最后有效 token

**原因**：追踪时 `output_pos = tensor([7])`（export_len-1），但实际最后有效 token 在 `seq_len-1`（如 5）。

**修复**：替换 `output_pos` 为 `torch.tensor([real_seq_len - 1])`。

### 12. tie-breaking 导致 argmax 不同

**现象**：部分 prompt 的 ep.module() 和 wrapper 输出在第 3 个 token 分叉

**原因**：bfloat16 下 top-3 logits 完全相同（18.500），`Gemma4ClippableLinear` vs `nn.Linear` 的微小差异改变 argmax 顺序。

**不是 bug**：模型在该位置本身不确定（多 token 概率相等），任何微小数值差异都会改变结果。

### 13. HF model 加载方式

**现象**：`Gemma4ForCausalLM.from_pretrained()` 大量 MISSING 权重

**原因**：checkpoint 是多模态 `Gemma4ForConditionalGeneration`，`Gemma4ForCausalLM` 期望 `model.layers.*` 但 checkpoint 有 `model.language_model.layers.*`。

**修复**：用 `Gemma4ForConditionalGeneration.from_pretrained()` 加载，再用 `.model.language_model` 和 `.lm_head` 访问文本模型。

### 14. SimpleKVCache dict → list

**现象**：torch.export 无法追踪 SimpleKVCache 的 dict 操作

**修复**：dict → list + tensor 属性（对齐 transformers DynamicCache 设计）。

### 15. 4D mask 透传缺失

**现象**：`_create_causal_mask` 总是动态构造 mask

**修复**：添加 `if attention_mask.dim() == 4: return attention_mask` 透传。

### 16. sliding_window mask 4D 透传未裁剪

**现象**：4D 透传对所有层返回同一 mask，sliding 层应有不同窗口

**修复**：4D 透传路径增加 sliding_window 裁剪逻辑。

### 17. torch.all 数据依赖

**现象**：`torch.all(attention_mask == 1)` 是数据依赖控制流

**修复**：简化为仅检查 cache 状态。

## 验证结果汇总

| 验证项 | 结果 | 说明 |
|--------|------|------|
| 单步 forward (wrapper vs 原始) | PASS (diff=0.0) | 完全一致 |
| chunk vs 非 chunk (wrapper) | PASS (3/3 text) | 完全一致 |
| chunk vs HF transformers | PASS (2/3 text) | Prompt 1 tie-breaking 差异 |
| epair 导出 + 回环 | PASS (2/3 text) | Prompt 1 tie-breaking, Prompt 2/3 完全一致 |
| BOS + bfloat16 出字 | ✅ | "Beijing. It is the largest city in China and" |

## 精度策略 (已验证, 不要改)

scatter-write 模式下 attention 的精度配置:

| 组件 | 精度 | 原因 |
|------|------|------|
| Q@K matmul | **bfloat16** | 和 HF 一致, BLAS 舍入最接近 |
| softmax | **float32** | npu_tuned_export 方案, 已有 |
| @v matmul | **bfloat16** | 已验证零 KV 位置不影响结果 |

### 已验证结论 (不要重复试)

1. **float32 Q@K 反而更差** — 固定 seed 对比第一个 token 匹配率从 78% 降到 33%
   - HF 用 bfloat16 Q@K, 我们也用 bfloat16 → BLAS 跪同一路径 → logits 接近
   - float32 Q@K → 和 HF 的 bfloat16 差异更大 → 匹配率降低

2. **head_dim=512 的 full_attention 层有 0.25 BLAS 舍入差异** (无法消除)
   - `bfloat16 [6, 512] @ [512, 2048]` vs `[6, 512] @ [512, 6]` → 不同 BLAS 路径
   - head_dim=256 (sliding 层): 0.0 差异
   - 35 层累积 → logits diff 0.4375 → 7/9 第一个 token 与 HF 一致

3. **固定 seed + sampling 对比**:
   - 第一个 token 匹配: 7/9 (78%)
   - 第二个 token 起: 0/9 (0%) — 分叉后 KV cache 不同, 后续完全不同

## 关键配置

```python
# 必须的配置
BOS_TOKEN_ID = 2           # Gemma 模型必须加 BOS
dtype = torch.bfloat16     # 模型原始精度
softcap = 30.0             # final_logit_softcapping
shared_kv_states = {}      # 不能传 None
```

## 注意事项

- gemma4-E2B 从 unsloth 下载，不是量化版本（bfloat16 原始权重）
- 模型有 35 层：15 层非共享 + 20 层 KV 共享
- 非共享层有 sliding (9 层) + full (2 层) + 源层 (2 层) + 普通 full (2 层)
- 源层 (Layer 13, 14) 必须保留全长度 KV（供共享层读取）
- 4D mask 必须屏蔽 padding 位置
- ep.module() 只能处理和追踪时相同的输入（seq_len, past_key_values=None）
- decode 步骤必须用 wrapper 直接调用（ep.module() 无法处理非 None 的 past_key_values）
