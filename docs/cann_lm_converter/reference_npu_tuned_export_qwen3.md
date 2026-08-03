# npu_tuned_export Qwen3 分析：相对 transformers 的修改

> 分析对象：`cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/npu_tuned_model/qwen3/`
> 对照对象：transformers v5.13.1 的 `models/qwen3/`
> 最后更新: 2026-07-14

## 概述

npu_tuned_export 是标准 transformers qwen3 的一个** fork**，经过手术式修改使其成为 ONNX 导出就绪的静态图模型，可直接编译为 Ascend OM。

核心改造：将 transformers 的**动态推理模型**（DynamicCache 增长、动态 mask 构造、动态 RoPE 计算）变为**静态图模型**（预分配 buffer、外部 mask 输入、预缓存 RoPE 表），满足 Ascend OM 编译的静态形状要求。

## 1. Config 差异

| 方面 | npu_tuned | transformers 标准 |
|------|-----------|-------------------|
| 风格 | 经典 `__init__` + 位置参数 | `@strict` dataclass + `__post_init__` |
| RoPE 配置 | `rope_scaling=None`（旧式 dict） | `rope_parameters: RopeParameters`（结构化类型） |
| `layer_types` | 无此字段 | 有，`__post_init__` 中计算 |
| `sliding_window` | 始终 `=4096`（int） | `int | None`，根据 `use_sliding_window` 动态设置 |
| `kv_cache_max_len` | **不在 config 中**，导出时 monkey-patch | 无此字段 |

关键：npu_tuned config 是一个**旧版快照**，不新增 config 字段。`kv_cache_max_len` 在导出脚本中直接 `config.kv_cache_max_len = 2048` 注入。

## 2. 模型架构修改

### 2.1 RMSNorm

```python
# transformers:
variance = hidden_states.pow(2).mean(-1, keepdim=True)

# npu_tuned (功能等价，改写为了 trace 稳定性):
variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
```

### 2.2 MLP — 扁平化 batch+seq

```python
# npu_tuned:
x = x.reshape(-1, x.shape[-1])                    # [bsz*seq, hidden] → 2D
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
down_proj = down_proj.reshape(1, -1, x.shape[-1])  # [1, bsz*seq, hidden] → 前导 dim=1
```

原因：NPU/ONNX GEMM 核倾向于固定前导 batch dim=1 的 2D 输入，避免动态 batch/seq reshape。

### 2.3 RoPE — 预缓存表 + Gather

| | transformers | npu_tuned |
|---|---|---|
| cos/sin 计算 | **动态计算**：`inv_freq @ position_ids` 每次 forward 算 | **预缓存表**：`torch.outer(t, inv_freq)` 预计算 `[1,1,max_seq,dim]` |
| 位置 | 模型级共享，传 `position_embeddings` 给每层 | **每层独立** `self.rotary_emb` |
| 索引方式 | 直接使用算好的 `[batch, seq, dim]` | `cos[position_ids]` — **Gather over 常量表** |
| 原因 | 灵活但动态形状多 | Gather over 常量表对 ONNX 追踪友好 |

### 2.4 apply_rotary_pos_emb

```python
# npu_tuned:
cos = cos.squeeze(1).squeeze(0)         # [seq_len, dim]
cos = cos[position_ids].unsqueeze(1)   # [bs, 1, seq_len, dim]  ← Gather by position
```

### 2.5 repeat_kv — 显式 Concat 替代 Expand

```python
# transformers: broadcast (高效但动态形状)
hidden_states[:, :, None, :, :].expand(...).reshape(...)

# npu_tuned: 显式 Concat (低效但静态形状)
rep_list = [hidden_states] * n_rep
hidden_states = torch.cat(rep_list, dim=1)  # 物化重复
```

原因：ONNX `Expand` 在广播维度上可能不稳定，显式 `Concat` 的形状确定。

### 2.6 Attention — 核心改造

| 方面 | transformers | npu_tuned |
|------|-------------|-----------|
| KV cache 写入 | `past_key_values.update(k, v, layer_idx)` — 不透明 Cache 对象 | **scatter-write 到预分配 buffer**：`past_key_value[0][new_kv_cache_pos] = k.transpose(0,2)` |
| KV cache 读取 | Cache 返回更新后的 tensor | **读回整个 buffer**：`key_states = past_key_value[0].transpose(0,2)` |
| Attention 计算 | 调度到 `ALL_ATTENTION_FUNCTIONS` | 内联：`matmul(q, k.T) * scale → softmax → matmul(attn, v)` |
| 返回 | `(attn_output, attn_weights)` | `(attn_output, attn_weights, past_key_value)` ← cache 是输出 |

### 2.7 输出 reshape

```python
# npu_tuned: 扁平化到 2D 再 o_proj
attn_output = attn_output.reshape(*input_shape, -1).contiguous().reshape(-1, last_dim)
attn_output = self.o_proj(attn_output).view(bsz, -1, hidden)
```

## 3. KV Cache 处理 — 定义性改变

### transformers (DynamicCache)

```python
# 不透明 Cache 对象，每次 forward 增长
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
# Cache 内部是 list of [k, v]，逐步 append
```

### npu_tuned (静态 buffer + scatter write)

```python
# 预分配固定大小 buffer，scatter 写入
past_key_value[0][new_kv_cache_pos] = key_states.transpose(0, 2)   # 写
past_key_value[1][new_kv_cache_pos] = value_states.transpose(0, 2)
key_states   = past_key_value[0].transpose(0, 2)                     # 读整个 buffer
value_states = past_key_value[1].transpose(0, 2)
```

**特性：**
- **预分配**：shape `[kv_cache_max_len, kv_head_num, batch, head_dim]`，一次性分配
- **Per-layer 对**：每层一对 `(past_key_in{i}, past_value_in{i})`
- **Scatter 写入**：`new_kv_cache_pos`（1D int tensor）告诉写入位置
- **全量读回**：写入后对整个 buffer 做 attention，用 `attention_mask` 屏蔽无效位置
- **图 I/O**：每个 buffer 既是输入也是输出

**设计权衡**：用内存（attention over padded cache + mask）换图静态形状确定性。

## 4. forward 签名差异

### Qwen3Attention.forward

```python
# transformers:
def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
    → (attn_output, attn_weights)

# npu_tuned:
def forward(self, hidden_states, attention_mask, position_ids,
            past_key_value, new_kv_cache_pos, output_attentions, use_cache):
    → (attn_output, attn_weights, past_key_value)  # cache 是输出
```

新增输入：`position_ids`（替代 `position_embeddings`）、`past_key_value`（tensor tuple，替代 Cache）、`new_kv_cache_pos`（cache 写入索引）。
新增输出：`past_key_value`（更新后的 cache）。

### Export Wrapper forward

```python
def forward(self, input_ids, attention_mask, position_ids,
            past_key_values, new_kv_cache_pos=None,
            embed_scale=None, output_pos=None):
    → (lm_logits, *kv_caches_out)  # 展平所有层的 KV cache
```

新增输入：
- `past_key_values`：list of per-layer `(key_buf, value_buf)` tensor
- `new_kv_cache_pos`：cache 写入位置
- `embed_scale`：per-token 反量化 scale（量化 embedding 用）
- `output_pos`：选择输出哪个 token 的 logits（decode 只算最后一个）

新增输出：
- `*kv_caches_out`：展平的所有层 K/V buffer（2 × num_layers 个 tensor）

## 5. Export Wrapper 设计

`Qwen3ForCausalLMWrapper` **绕过** `Qwen3Model.forward` 和 `Qwen3ForCausalLM.forward`，直接驱动 decoder layers：

```
wrapper.forward():
  1. Embedding 查表 (或接收外部 embedding)
  2. for layer in self.model.model.layers:          ← 直接遍历层
       past_key_value = past_key_values[idx]
       layer(..., new_kv_cache_pos=..., use_cache=True)
       kv_caches_out.extend(layer 的 KV 输出)
  3. if output_pos: hidden_states = hidden_states[:, output_pos, :]  ← 选位
  4. norm + lm_head → lm_logits
  5. return (lm_logits, *kv_caches_out)
```

**为什么绕过 Qwen3Model.forward？** 因为 `Qwen3Model.forward` 内部有 `_update_causal_mask` / `create_causal_mask` 等动态控制流分支，这些分支依赖运行时 seq 长度，无法被 `torch.export` 静态追踪。wrapper 直接遍历层，mask 由外部 `attention_mask` tensor 提供。

### 两种 wrapper 变体

| Wrapper | embedding 处理 | 用途 |
|---------|---------------|------|
| `Qwen3ForCausalLMWrapper` | 图内做 embedding 查表 | 标准 |
| `Qwen3ForCausalLMWithoutEmbeddingWrapper` | 接收外部 embedding，图内不做查表 | embedding 量化到 int8 在 host 侧跑 |

## 6. 导出脚本流程

```python
export_llama(export_config):
  1. 构建 wrapper (model_cls, model_path, embedding_config)
  2. 可选: 加载量化权重 / LoRA / gemm→matmul 替换
  3. config.kv_cache_max_len = kv_cache_max_len  ← monkey-patch
  4. 构造 dummy inputs:
     - input_ids [1, seq_len] int64
     - attention_mask [1, 1, seq_len, kv_cache_max_len] float32
     - position_ids [1, seq_len] int64
     - per-layer past_key_in{i} / past_value_in{i} [kv_cache_max_len, kv_heads, batch, head_dim]
     - new_kv_cache_pos [seq_len] int64
     - (optional) embed_scale, output_pos
  5. torch.onnx.export(wrapper, dummy_inputs, opset=12, ...)
  6. do_opt.process_onnx(...)  ← ONNX 后处理
  7. onnx.save(..., external_data=True)  ← 图权分离
```

## 7. ONNX 后处理

| 步骤 | 说明 |
|------|------|
| **权重压缩** | 大 initializer (>1MB) 替换为 `ConstantOfShape` 占位，让 `onnxsim` 不需加载大权重 |
| **onnxsim 简化** | 跳过 `fuse_matmul_add_bias_into_gemm`/`fuse_qkv`/`eliminate_duplicate_initializer`（会破坏 NPU 图匹配） |
| **权重解压** | 恢复真实 initializer |
| **INT32 转换** | `input_ids`/`position_ids`/`new_kv_cache_pos` → INT32（Ascend 偏好 INT32 索引） |
| **节点重命名** | 按 state-dict key 命名 ONNX 节点 |
| **共享权重拆分** | tied weights (lm_head/embed_tokens) → 深拷贝为独立 tensor |

## 8. NPU 优化 (do_opt.py)

| 功能 | 说明 |
|------|------|
| **Embedding 量化** | `embed_tokens.weight` → int8 + per-row scale，导出 `.embedding_weights` + `.embedding_dequant_scale` |
| **Gemm→MatMul** | `nn.Linear` → `FLinearMatmul`（`MatMul`+`Add`），某些 NPU 图匹配器偏好 MatMul |
| **LoRA 注入** | `FLinearLora` 包装 `nn.Linear`，从 `lora_config.json` 读取策略 |
| **LoRA 配置生成** | 扫描导出的 ONNX，生成 `_lora_config.json`（映射层名→节点名） |

## 9. 改造总结

| 改造点 | transformers 原始 | npu_tuned 修改 | 原因 |
|--------|-----------------|---------------|------|
| KV cache | DynamicCache 动态增长 | 预分配 buffer + scatter write | 静态形状 |
| RoPE | 动态 `inv_freq @ position_ids` | 预缓存表 + `cos[position_ids]` Gather | 静态形状 |
| Mask | `create_causal_mask` 动态构造 | 外部 4D tensor 输入 | 消除控制流 |
| repeat_kv | `expand` 广播 | 显式 `Concat` | 形状确定 |
| MLP/o_proj | 3D `[bsz, seq, hidden]` | 2D `[1, bsz*seq, hidden]` | NPU GEMM 偏好 |
| Model.forward | 完整调用（含 mask 构造） | wrapper 绕过，直接遍历层 | 跳过动态分支 |
| 输出选位 | 返回全部 logits | `output_pos` 选位 | decode 只需 1 个 logit |
| 导出格式 | — | ONNX + 外部 .pb 权重 | 图权分离 |
| 索引类型 | INT64 | INT32 | Ascend 偏好 |

**核心原则**：消除所有动态形状和动态控制流，使图完全静态化，满足 Ascend OM 编译的静态形状要求。
