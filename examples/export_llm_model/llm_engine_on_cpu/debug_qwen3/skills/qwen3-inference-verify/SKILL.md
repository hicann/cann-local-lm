---
name: qwen3-inference-verify
description: |
  用纯 PyTorch 实现的标准 Qwen3 跑生成（出字）并验证实现正确性。当用户提到以下场景时使用：
  - 用纯 torch Qwen3（cann_lm/models/qwen3）跑推理 / 生成 / 出字
  - 验证纯 torch Qwen3 实现是否正确（数学不变量 / 与 HuggingFace transformers 对拍）
  - 在本环境使用 transformers 但需要禁用 torch_npu probe
  - 纯 torch Qwen3 与 HF Qwen3ForCausalLM 逐 token 数值对齐
  - Qwen3 KV cache 一致性 / 因果性测试
version: 1.0.0
metadata:
  author: assistant
  category: model-verification
  model: qwen3-0.6b
  reference: Qwen/Qwen3-0.6B
allowedTools:
  - read
  - write
  - bash
  - task
  - question
  - webfetch
---

# Qwen3 生成与验证技能

## 前置条件

- Python >= 3.10
- PyTorch >= 2.0（仅依赖 `torch`，不依赖 `transformers` 做模型）
- safetensors（权重加载）
- transformers >= 5.x（**仅用于 tokenizer 和参考实现对拍**，不用于跑自定义模型）
- 一份本地 Qwen3 checkpoint（如 `Qwen/Qwen3-0.6B`），含 `config.json` + `model.safetensors` + tokenizer 文件

## 涉及的文件

```
cann_lm/models/qwen3/
├── __init__.py
├── configuration_qwen3.py     # Qwen3TextConfig（纯 dataclass）
├── modeling_qwen3.py           # 纯 torch Qwen3ForCausalLM + load_qwen3_weights
├── test_qwen3.py               # 数学不变量测试（tiny + 真实权重）
└── verify_qwen3_vs_hf.py       # 与 HF transformers 逐 token 数值对拍

cann_lm/models/run_qwen3.py     # 出字脚本：纯 torch 模型 + HF tokenizer
cann_lm/hf_cpu.py               # 禁用 transformers 的 torch_npu probe（环境 shim）
```

## Qwen3 架构要点（标准版，区别于 Qwen3.5）

标准 Qwen3 是 Qwen3.5 的**基类**，去掉 Qwen3.5 的所有花活：

| 特性 | 标准 Qwen3 | Qwen3.5 |
|------|-----------|---------|
| RMSNorm | 标准 `weight * norm`（weight init 1） | centered `1 + weight`（init 0） |
| RoPE | 默认全旋转，`rope_theta` 单值 | MRoPE，`partial_rotary_factor=0.25`，`mrope_section` |
| Attention | GQA + QK-norm，**无 Q Gate** | GQA + QK-norm + Q Gate（q_proj 2×，sigmoid 调制） |
| 层类型 | 全 `full_attention` | `full_attention` / `linear_attention` 交替 |
| 线性注意力 | 无 | GatedDeltaNet |
| MLP 激活 | SiLU | SiLU |

Qwen3-0.6B config：`hidden_size=1024`、`num_hidden_layers=28`、`num_attention_heads=16`、`num_key_value_heads=8`(GQA)、`head_dim=128`、`tie_word_embeddings=True`、`vocab_size=151936`、`rope_theta=1e6`。

## 关键环境要点

### 1. 禁用 torch_npu（本环境必做）

本环境装了 `torch_npu` 但缺 `libhccl_compat.so`。transformers 的 `is_torch_npu_available()` 用 `importlib.util.find_spec("torch_npu")` 判定（包存在→True），随后 `import torch_npu` 会**致命报错**。解决：在任何 `from transformers import ...` 之前先 `import cann_lm.hf_cpu`，它把 `is_torch_npu_available` 打成 `False`（lru_cache 会缓存），从此不再 `import torch_npu`：

```python
import cann_lm.hf_cpu  # noqa: F401  必须在 from transformers import ... 之前
from transformers import AutoTokenizer, Qwen3ForCausalLM
```

### 2. 环境变量

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 阻止 torch autoload npu backend
export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
```

所有命令都加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，否则 `import torch` 就会因 npu autoload 失败。

## 生成（出字）流程

### 脚本：run_qwen3.py

`cann_lm/models/run_qwen3.py`：纯 torch `Qwen3ForCausalLM`+`load_qwen3_weights` 跑模型，transformers 只当 tokenizer 用（`apply_chat_template` + `decode`）。

```bash
cd cann_lm/models
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 run_qwen3.py \
    --model /mnt/workspace/data/models/qwen3-0.6B \
    --greedy --max-new-tokens 256 \
    --prompt "你好，请用一句话介绍你自己。"
```

### generate() 的关键点：首步必须初始化 KV cache list

纯 torch `Qwen3TextModel.forward` 在 `past_key_values is None` 时**不创建 cache list**、直接把 None 透传给各层（这是为了 `torch.export` 友好：无 list mutation）。所以 `generate()` 首步必须**显式初始化** `[None]*num_layers`，否则 cache 分支永不触发、每步全量重算（结果碰巧对，但 O(n²) 且 cache 代码没被验证）：

```python
past_key_values = None
num_layers = self.config.num_hidden_layers
for _ in range(max_new_tokens):
    if past_key_values is None:
        step_ids = input_ids
        past_key_values = [None] * num_layers   # ← 关键：触发 cache 创建
    else:
        step_ids = input_ids[:, -1:]            # 后续步只喂末位 token
    outputs = self(input_ids=step_ids, past_key_values=past_key_values)
    past_key_values = outputs["past_key_values"]
    logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
    ...
```

### decode 单 token 用 minimal BPE（无 transformers 时的回退）

若无 transformers/tokenizers 库，可用 `vocab.json` + GPT2 `bytes_to_unicode` 做最小解码（id→text），用于检查 greedy 是否产出连贯文本：

```python
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}
# vocab.json: token(str)->id; 反转得 id->token
def decode(ids, rev_vocab):
    text = "".join(rev_vocab.get(int(i), f"<{i}>") for i in ids)
    return bytes(_BYTE_DECODER[c] for c in text if c in _BYTE_DECODER).decode("utf-8", errors="replace")
```

注意：`tokenizer_config.json` 的 `added_tokens_decoder` 是 `{id_str: {"content": ...}}`，id 在 key 上而非 value 里。

## 验证流程（两层）

### 第一层：数学不变量（test_qwen3.py，无需参考实现）

实现错了必然挂，与权重无关。tiny 随机模型即可测：

```bash
cd cann_lm/models/qwen3
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 test_qwen3.py
```

核心不变量：

| 不变量 | 含义 |
|--------|------|
| **因果性** | `forward(prefix)[:, k-1] == forward(full)[:, k-1]` 对所有 k（position i 不看未来 token） |
| **KV cache 一致** | 逐 token 增量解码（带 cache）逐位置 logits == 全量 prefill（无 cache） |
| **greedy == argmax** | `generate` 首个 token == `argmax(forward(prompt)[:, -1])` |
| **tied embedding** | `tie_word_embeddings=True` 时 `lm_head.weight is embed_tokens.weight`（同一 tensor） |
| **RoPE 保范** | `\|\|RoPE(q)\|\|_2 == \|\|q\|\|_2`（旋转是等距变换） |
| **RMSNorm 公式** | weight=1 时输出 == 手算 `x * rsqrt(mean(x²)+eps)` |
| **确定性 / 无 NaN** | eval 下两次 forward 完全相等；logits 全 finite |

加真实权重 `--model PATH` 再跑：0 missing key + 真实权重下 KV cache 增量 vs prefill 的 max diff + greedy 解码出连贯文本。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 test_qwen3.py --model /mnt/workspace/data/models/qwen3-0.6B
```

### 第二层：HF transformers 对拍（verify_qwen3_vs_hf.py，终极证明）

同一份 checkpoint 分别装进纯 torch 实现和 HF `Qwen3ForCausalLM`，同输入比 logits + greedy 序列。**float32 + 两侧 eager attention** 消除 SDPA 融合差异，差值就是实现差异本身。

```bash
cd cann_lm/models/qwen3
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 verify_qwen3_vs_hf.py \
    --model /mnt/workspace/data/models/qwen3-0.6B
```

关键设计：

```python
import cann_lm.hf_cpu  # noqa  禁用 torch_npu probe
from cann_lm.models.qwen3 import Qwen3ForCausalLM, load_qwen3_weights
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM, AutoTokenizer

# 纯 torch 侧
mine = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
mine = load_qwen3_weights(mine, path, device="cpu", dtype=torch.float32)
# HF 侧：eager attention 对齐
hf = HFQwen3ForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager")

with torch.no_grad():
    hf_logits = hf(ids).logits
    my_logits = mine(input_ids=ids)["logits"]
diff = (my_logits - hf_logits).abs()
```

比对指标：`max|Δ|`、`mean|Δ|`、`argmax_match%`、`top5_overlap%`、greedy 生成 token 序列是否完全相等。

## 验证结果基线（Qwen3-0.6B）

| 指标 | 结果 |
|------|------|
| test_qwen3.py（tiny 不变量） | 33/33 PASS |
| test_qwen3.py --model（真实权重） | 40/40 PASS |
| 真实权重 KV cache 增量 vs prefill（bf16） | max\|Δ\| = **0.00e+00**（精确） |
| 真实权重 greedy 解码 | `Question: What is the value of the expression 2 +`（连贯英文） |
| 算术 `25*4` | 正确推理到 100 |
| HF 对拍（float32, eager） | **logits max\|Δ\| = 0.000e+00**，argmax 100% 一致 |
| HF 对拍 greedy 16 token | **identical=True**（逐 token 相同） |
| 参数量 | 596,049,920 == HF 596,049,920 |
| embed_tokens 权重 | bit-identical（max\|Δ\|=0） |

## 关键陷阱

### 1. generate() 不初始化 cache → 每步全量重算
`TextModel.forward` 在 `past=None` 时不建 cache list（为 export 友好）。generate 必须首步 `[None]*num_layers`，否则 cache 分支永不触发。表现为"输出对但慢"，且 cache 代码完全没被验证。由 KV cache 一致性测试暴露并修复。

### 2. bfloat16 不能过 converter 的 pack_weights
`cann_lm/converter/export/epair_utils.py` 的 `pack_weights` 走 `tensor.numpy().tobytes()`，numpy 不支持 bf16 → `unsupported ScalarType BFloat16`。converter 跑 Qwen3 时用 `--dtype float16`（不是 bf16）。这是既有 epair_utils 的限制，不是模型问题。

### 3. torch_npu probe 致命
见"关键环境要点"。不 source CANN set_env.sh（那会让 torch_npu importable → transformers 误判 NPU 可用）。用 `cann_lm/hf_cpu.py` shim 直接关 probe。

### 4. converter export 路径不能碰 cache
`torch.export.export(model, (input_ids,))` 时 `past_key_values=None`，forward 走无 cache 分支。所以 `TextModel.forward` 里 `past_key_values is None` 时**不要**创建 list，直接透传 None 给层，层里 `if past_key_values is not None:` 跳过 cache。任何 list mutation 都会让 export 炸。

### 5. 权重 key 映射
`load_qwen3_weights` **不做任何 key 重映射**——HF safetensors 的 key 名（`model.layers.N.self_attn.q_proj.weight` 等）与纯 torch 实现的 param 名完全一致。"0 missing key"即等价于权重映射正确。

## 与现有 transformers 实现的差异要点

| Transformers | 纯 PyTorch 实现 |
|---|---|
| 继承 `PreTrainedModel`, `GenerationMixin` | 继承 `nn.Module`，手动实现 generate |
| `DynamicCache` | `list[(k, v)]`，首步 `[None]*num_layers` |
| SDPA / flash attention 分派 | eager attention（手写 softmax） |
| 配置继承 `PreTrainedConfig` | 纯 dataclass `Qwen3TextConfig` |
| `ACT2FN` | 直接 `nn.SiLU()` |
| QK-norm 在 `Qwen3Attention` | 同，`Qwen3RMSNorm(head_dim)` 作用于 q/k |

## 常见问题

### Q: 为什么要 float32 + eager 对拍？
bf16 下 HF 用 SDPA（融合 softmax+matmul）与纯 torch eager 的数值路径不同，会有 ~1e-2 量级噪声，无法区分"实现差异"和"精度噪声"。float32 + 两侧 eager 把噪声降到 0，差值纯粹反映实现差异。

### Q: greedy 为什么用 eos_set 而非单 eos？
Qwen3 的 `eos_token_id` 可能是 list（如 `[151645, 151643]`）。`generate` 用 `eos_set` 兼容 int 和 list。

### Q: export 和 generate 的 past_key_values 为何不同？
export 调 `model(input_ids)`（past=None，无 cache 路径，无 list mutation，trace 友好）；generate 调 `model(input_ids, past=[None]*N)`（有 cache 路径，list 被各层填充）。同一 forward，靠 `past_key_values is not None` 分支切换。

### Q: 没装 transformers 怎么出字？
用 `vocab.json` + GPT2 `bytes_to_unicode` 做最小 BPE **解码**（id→text）即可看 greedy 是否连贯；编码（text→id）需要完整 BPE（`merges.txt` + 预切分正则），无 `regex` 库时用 `re` 写 GPT2 模式 `[^\W\d_]+` 即可。
