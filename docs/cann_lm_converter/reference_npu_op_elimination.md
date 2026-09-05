---
name: npu-op-elimination
description: |
  NPU 算子消除技能:把 HF 模型导出图里 NPU 不支持 / 低效的算子改造掉,使 epair→OM 编译流水线通过且产物干净。
  当用户提到以下场景时使用:
  - 导出图含 NPU 不支持的算子 (ge.MaskedFill / ge.Where / ge.SelectV2 等)
  - 导出图含低效算子想替换 (BroadcastTo → ConcatV2D,Pow/Neg/Cos/Sin → Gather)
  - OM 编译失败报 "operator not supported" 或某算子 kernel 缺失
  - 把 gitee npu_tuned 模式 (host 预构 mask / repeat_kv concat / RoPE 预缓存) 应用到 cann_lm 模型
  - 对 gemma4 / qwen3 / qwen3.5 做 export 前的图改造
version: 1.0.0
metadata:
  author: assistant
  category: graph-rewrite
  last_updated: 2026-08-10
  demonstrated_on: gemma4 (e2b 真实配置)
  applicable_to: [gemma4, qwen3, qwen3_5]
allowedTools:
  - read
  - grep
  - glob
  - edit
  - write
  - bash
---

# NPU 算子消除技能

把 HF 模型导出图里 NPU 不支持 / 低效的算子改造成 NPU 友好形式,使 epair→OM 编译通过且算子表干净。
来源: gitee `npu_tuned` 模式 (参考 `docs/cann_lm_converter/reference_npu_tuned_export_qwen3.md`)。
首次完整应用于 cann_lm gemma4 (e2b 真实配置: 35 层后 20 KV 共享, head_dim=256/global=512, PLE 开, sliding=512, kv_max=2048)。

## 总览: 三处改造

| # | 改造 | 消除的算子 (导出图) | 改造后算子 | 位置 |
|---|---|---|---|---|
| 1 | host 预构 mask + bypass forward | `ge.MaskedFill` (NPU 不支持) | mask 作为图输入 (无图内构造) | `export_wrapper.py` |
| 2 | repeat_kv reshape+cat | `BroadcastTo` | `ConcatV2D` | `modeling_*.py` |
| 3 | RoPE 预缓存表 + Gather | `Pow`/`Neg`/`Cos`/`Sin` | `Gather` | `modeling_*.py` |

验证基线: e2b gemma4 三次 OM 编译后 dump 算子表确认 MaskedFill=0、BroadcastTo=0 (ConcatV2D=70)、Pow/Neg/Cos/Sin=0;数值 maxdiff=0.00e+00。

## 何时应用

- OM 编译报某算子不支持 (查 CANN ops-info 无该算子定义)
- ATC dump 里出现 `BroadcastTo` / `Pow` / `Neg` / `Cos` / `Sin` / `MaskedFill` / `Where` / `SelectV2` 等想消除的算子
- 适配新模型到 cann_lm export pipeline 前的图清洗

## 改造 1: host 预构 mask (消除 ge.MaskedFill)

### 现象
gemma4 `_prepare_causal_mask` 在图内用 `torch.full` + masked_fill 构造 attention mask (causal + sliding window)。导出后图里出现 `ge.MaskedFill`。NPU 不支持该算子 (grep CANN ops-info 确认无此算子定义), OM 编译失败。

### 原因
mask 构造逻辑 (causal 三角 + sliding window) 含数据依赖控制流, torch.export 能追踪但图内物化成 MaskedFill。NPU 算子库无 MaskedFill kernel。

### 解决
**把 mask 构造搬到 host/pipeline 侧,作为图输入传入; wrapper 绕过 `Model.forward` 直接驱动层, 静态按 `layer_types` 给每层分 mask。** 图内不再构造 mask → 无 MaskedFill。

`forward` 签名: 单个 `attention_mask` 拆成 `mask_full` + `mask_sliding` 两个图输入 (host 每 pipeline 步重建):

```python
def forward(self, input_ids=None, mask_full=None, mask_sliding=None,
            position_ids=None, past_key_values=None, new_kv_cache_pos=None,
            embeddings=None, output_pos=None,
            per_layer_token_identity=None, embeddings_scale=None):
    ...
    logits, past_kv_flat = self._drive_layers(
        input_ids, mask_full, mask_sliding, position_ids,
        past_key_values, new_kv_cache_pos, embeddings, per_layer_token_identity)
    if output_pos is not None:
        logits = logits[:, output_pos, :]
    return logits, *past_kv_flat
```

`_drive_layers` 层循环里静态分配 mask (无控制流, torch.export 友好):

```python
for i, layer in enumerate(tm.layers):
    layer_type = self._layer_types[i] if i < len(self._layer_types) else "full_attention"
    mask_i = mask_sliding if layer_type == "sliding_attention" else mask_full
    hidden_states = layer(hidden_states, per_layer_input,
                         shared_kv_states=shared_kv_states,
                         position_embeddings=rope_embs[layer_type],
                         attention_mask=mask_i,
                         past_key_values=kv_list,
                         new_kv_cache_pos=new_kv_cache_pos)
```

host 侧 mask 构造 (`get_sample_inputs` 调, 不进图, 可用任意算子):

```python
def _build_mask_full(self, seq_len, kv_max, dtype, new_kv_cache_pos):
    """causal-over-buffer: k<=q → 0, else MIN。q 位 = new_kv_cache_pos。"""
    min_val = torch.finfo(dtype).min
    q = new_kv_cache_pos.to(dtype=dtype).unsqueeze(1)        # [seq, 1]
    k = torch.arange(kv_max, dtype=dtype).unsqueeze(0)       # [1, kv_max]
    mask = torch.where(k <= q, torch.tensor(0.0, dtype=dtype), min_val)
    return mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)

def _build_mask_sliding(self, seq_len, kv_max, dtype, new_kv_cache_pos):
    """causal + sliding window: k<=q AND q-k<=W-1 → 0, else MIN。
    kv_len <= sliding_window 时退化为 mask_full。"""
    w = self._sliding_window
    if not w or w <= 0 or kv_max <= w:
        return self._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
    min_val = torch.finfo(dtype).min
    q = new_kv_cache_pos.to(dtype=dtype).unsqueeze(1)
    k = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
    diff = q - k
    attend = (diff >= 0) & (diff <= w - 1)
    mask = torch.where(attend, torch.tensor(0.0, dtype=dtype), min_val)
    return mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)
```

### 关键点
- mask 不进图 → host 可用 `torch.where`/`torch.arange` (这些若进图也会变成要消除的算子)
- per-layer 分配是**静态**的 (按 `config.layer_types` 索引, 无 if/else 控制流) → torch.export 可追踪
- eager generate 路径不受影响 (仍走 `model.forward` 含 `_prepare_causal_mask` 的 None/2D 分支), 只改 export 路径
- PLE 开启时 `per_layer_token_identity` 同理 host 预算 (embed_tokens_per_layer 查表外置)

### 文件
- `cann_lm/models/gemma4/export_wrapper.py` (`Gemma4ExportWrapper.forward` / `_drive_layers` / `_build_mask_full` / `_build_mask_sliding`)

## 改造 2: repeat_kv reshape+cat (BroadcastTo → ConcatV2D)

### 现象
GQA (grouped query attention) 的 `repeat_kv` 用 `expand` 复制 KV heads。导出图里出现 `BroadcastTo`。该算子在 NPU 上效率低/可能 fusion 不到。

### 原因
`hidden_states.expand(batch, num_kv_heads*n_rep, slen, head_dim)` 在导出图里物化成 BroadcastTo。

### 解决
**用 reshape+cat+view 替代 expand (npu_tuned 方案), 使导出图里是 ConcatV2D 而非 BroadcastTo。**

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (b, kv_heads, s, d) -> (b, heads, s, d)。
    用 reshape+cat+view 替代 expand, 导出图里是 ConcatV2D 而非 BroadcastTo。"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch * num_kv_heads, 1, slen, head_dim)
    hidden_states = torch.cat([hidden_states] * n_rep, dim=1)
    return hidden_states.view(batch, num_kv_heads * n_rep, slen, head_dim)
```

### 关键点
- **头序必须是 kv_head-major**: 先 reshape 到 `[b*kv, 1, s, d]` (kv 并入 dim0, dim1 留 size-1 给 cat), 再 `cat([x]*n_rep, dim=1)` 物化 n_rep 份, 最后 view 回 `[b, kv*n_rep, s, d]`。kv 外层、n_rep 内层, 与 expand+reshape 逐元素一致
- **naive `cat([x]*n_rep, dim=1)` 是 rep-major 会与 q 错位, 不可用** — 必须先 reshape 把 kv 并到 dim0
- 数值零差 (maxdiff=0.00e+00), e2b OM dump 实测 ConcatV2D=70 (=35 层 × 2(k,v)), BroadcastTo=0

### 文件
- `cann_lm/models/gemma4/modeling_gemma4.py` `repeat_kv()` (qwen3/qwen3.5 同名函数同改)

## 改造 3: RoPE 预缓存表 + Gather (Pow/Neg/Cos/Sin → Gather)

### 现象
RoPE 的 `forward` 在图内动态算 cos/sin: `inv_freq = pow(base, -exponents)` → `freqs = outer(t, inv_freq)` → `emb = cat(freqs, freqs)` → `cos(emb)` / `sin(emb)`。导出图里出现 `Pow`/`Neg`/`Cos`/`Sin` 一堆动态算子。

### 原因
cos/sin 表是**常量** (只依赖 position 和 rope_theta, 不依赖输入), 但原实现每次 forward 重算, 图里物化成 Pow/Neg/Cos/Sin。

### 解决
**`__init__` 预缓存 cos/sin 表 (`register_buffer persistent=False`), `forward` 只 Gather (`cos_cached[position_ids]`)。** 全部 init 算 (inv_freq + outer + cat + cos/sin), 不进图。

`__init__` 预缓存 (per layer_type):

```python
for layer_type in self.layer_types:
    ...
    # 预缓存 cos/sin 表 (init 算: inv_freq + outer + cat + cos/sin, 全在 init, 不进图)
    # forward 只 Gather (cos_cached[position_ids]), 消除 Pow/Neg/Cos/Sin 动态算子
    inv_freq = self._compute_inv_freq(layer_type, dev, torch.float32)
    t = torch.arange(max_seq, device=dev, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer(f"cos_cached_{layer_type}", emb.cos(), persistent=False)
    self.register_buffer(f"sin_cached_{layer_type}", emb.sin(), persistent=False)
```

`forward` Gather:

```python
@torch.no_grad()
def forward(self, x, position_ids, layer_type="full_attention"):
    # Gather over 预缓存常量表。无 inv_freq/cos/sin/pow/neg 动态算子。
    cos_cached = getattr(self, f"cos_cached_{layer_type}")  # [max_seq, dim] 常量
    sin_cached = getattr(self, f"sin_cached_{layer_type}")
    cos = cos_cached[position_ids]   # -> [batch, seq, dim]
    sin = sin_cached[position_ids]
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### 关键点
- `persistent=False` → 不进 state_dict (导出权重不含, 避免冗余); 但 register_buffer 让它跟着模型走
- per layer_type 各缓存一份 (`cos_cached_full_attention` / `cos_cached_sliding_attention`), 因 gemma4 不同 layer_type 的 head_dim / rope_theta 不同
- 数值零差 (maxdiff=0.00e+00), e2b OM dump 实测 Pow/Neg/Cos/Sin=0
- `_compute_inv_freq` 仍保留在 init 调用 (proportional / default 两种 rope_type 分支), 不进图即可

### 文件
- `cann_lm/models/gemma4/modeling_gemma4.py` `Gemma4TextRotaryEmbedding`

## 验证方法 (改造后必做)

### 1. 数值精度验证 (eager)
wrapper 单步 forward vs 原始 model.forward, 对比 logits maxdiff:
- 预期 maxdiff=0.00e+00 (三处改造都是恒等数值变换)
- 脚本: `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_bypass.py`

### 2. 算子表验证 (OM dump)
epair → OM 编译后, dump 算子表确认目标算子消除:

```bash
# ATC dump (编译时加 --dump=...)
# 看 atc_dump_*/ge_*.json 的 op_list, 统计算子计数
python3 -c "
import json
ops = json.load(open('atc_dump.json'))['op_list']
from collections import Counter
c = Counter(o['type'] for o in ops)
for op in ['MaskedFill','BroadcastTo','ConcatV2D','Pow','Neg','Cos','Sin','Gather','Where','SelectV2']:
    print(f'{op:20s} {c.get(op,0)}')
"
```

预期:
| 算子 | 改造前 | 改造后 |
|---|---|---|
| MaskedFill | >0 | 0 |
| BroadcastTo | >0 | 0 |
| ConcatV2D | 少 | 70 (e2b, =35层×2) |
| Pow/Neg/Cos/Sin | 各>0 | 各0 |
| Gather | 少 | 增多 (rope) |

### 3. 关 AUB 验证 (可选)
若 `AutomaticBufferFusionOp` 干扰, 用 fusion_switch_file 关 AutomaticUbFusion:
```bash
# fusion_autoub_off.cfg 内容: {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}
export GE_FUSION_SWITCH_FILE=/path/to/fusion_autoub_off.cfg
```
编译后 dump 确认 AutomaticBufferFusionOp=0。

## 环境准备

### converter (CPU-only, 导出 epair)
```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
# venv: /mnt/workspace/torch-delegate-harmony/.venv (torch 2.9.0+cpu + torchair 源码编译 NO_ASCEND_SDK=1)
```

### NPU (OM 编译 + 推理验证)
```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
# 系统 Python 3.11.4 (torch 2.7.1+cpu, torch_npu 2.7.1.post4, Ascend910 ×2)
```

## 调试经验

### 1. host mask 的 sliding 退化
`_build_mask_sliding` 里 `kv_max <= sliding_window` 时必须退化为 `mask_full` (causal 已隐含在窗口内), 否则 sliding 层会多算不该算的 token。

### 2. repeat_kv 头序错误
naive `torch.cat([x]*n_rep, dim=1)` 不 reshape 会得到 rep-major (所有 n_rep 份连续), 与 query 的 kv_head-major 错位 → 数值爆炸。必须先 `reshape(b*kv, 1, s, d)` 把 kv 并到 dim0。

### 3. RoPE register_buffer persistent
`persistent=False` 必加, 否则 cos/sin 表进 state_dict, epair 权重会含冗余常量表 (max_seq × dim 的 float32 表, 几十 MB)。

### 4. bypass forward 不能破坏 eager generate
wrapper 的 `_drive_layers` 只走 export 路径; `model.forward` (含 `_prepare_causal_mask` 的 None/2D 分支) 必须保留给 eager generate 用。两路径独立, 互不影响。

### 5. PLE 查表外置
gemma4 PLE 的 `embed_tokens_per_layer` 查表若留在图内, 是大 embedding gather (vocab × num_layers × hidden)。host 预算 `per_layer_token_identity` 外置, `detach_externalized_lookups()` 删掉 embedding 模块, 图内只剩 project。qwen3/qwen3.5 无 PLE, 跳过。

## 跨模型适用性

| 改造 | gemma4 | qwen3 | qwen3.5 | 备注 |
|---|:---:|:---:|:---:|---|
| 1. host mask | ✓ 已应用 | 待应用 | 待应用 | 有 sliding window 的模型才需 mask_sliding; 纯 causal 用 mask_full 即可 |
| 2. repeat_kv concat | ✓ 已应用 | 待应用 | 待应用 | 所有 GQA 模型通用 (n_rep>1) |
| 3. RoPE 预缓存 | ✓ 已应用 | 待应用 | 待应用 | 所有动态算 cos/sin 的实现通用 |

> "待应用" = 该改造逻辑通用, 适配新模型时按本文代码片段对齐其 `repeat_kv` / RoPE / export_wrapper 即可。

## 相关文件

| 文件 | 作用 |
|---|---|
| `cann_lm/models/gemma4/export_wrapper.py` | 改造 1 (bypass forward + host mask + PLE 外置) |
| `cann_lm/models/gemma4/modeling_gemma4.py` `repeat_kv()` | 改造 2 (reshape+cat) |
| `cann_lm/models/gemma4/modeling_gemma4.py` `Gemma4TextRotaryEmbedding` | 改造 3 (预缓存 + Gather) |
| `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_bypass.py` | 数值验证 (wrapper vs 原始) |
| `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_export.py` | 导出验证 |
| `examples/test_converter/export_llm_model/om_npu_verify/` | OM 编译 + 推理工具 |
| `docs/cann_lm_converter/reference_npu_tuned_export_qwen3.md` | npu_tuned 模式参考来源 |
| `docs/cann_lm_converter/converter_design.md` | converter 全流程设计 |

## 参考提交
- `a429d4a` feat(gemma4): bypass-forward wrapper + host mask + repeat_kv concat + rope precompute
