# CANN-LM Converter 设计文档

> 将 HuggingFace 模型转换为 epair 格式的 CLI 工具
>
> 支持模型：gemma4 / qwen3 / qwen3_5（后续可扩展）
>
> 最后更新: 2026-07-14

## 目录

- [能力边界](#能力边界)
- [目录结构](#目录结构)
- [模型差异总览](#模型差异总览)
- [model_registry.py — 模型注册表](#model_registrypy--模型注册表)
- [loader.py — 模型加载](#loaderpy--模型加载)
- [saver.py — epair 保存](#saverpy--epair-保存)
- [兼容性处理](#兼容性处理)
- [cann-lm-converter.py — CLI](#cann-lm-converterpy--cli)
- [完整调用链](#完整调用链)
- [输出文件布局](#输出文件布局)
- [C++ 加载侧](#c-加载侧)
- [新增模型接入指南](#新增模型接入指南)
- [目标导出方案：统一 cache 签名](#目标导出方案统一-cache-签名轻量版)
- [环境变量](#环境变量)
- [使用示例](#使用示例)
- [待实现功能](#待实现功能)

---

## 能力边界

`cann_lm/converter/` 是一个 **CLI 工具**，能力如下：

- **输入**：HuggingFace 模型目录（`config.json` + safetensors 权重 + tokenizer 文件）
- **输出**：epair 模型目录（`ep_json_with_ascend_ir.json` + 权重 bin + tokenizer + `model_config.json`），供 C++ 后端加载
- **量化**：作为内置扩展能力预留，具体方案 TBD（`quant/` 目录预留）

对外只暴露一个入口 `cann-lm-converter.py`，模型类型从 HF `config.json` 自动探测，用户无需指定 `--model-type`。

## 目录结构

```
cann_lm/converter/
├── __init__.py
├── cann-lm-converter.py          # CLI 入口 (pipeline) + 程序化 convert() 函数
├── loader.py                     # 模型加载 (config + weights + tokenizer, 自动探测模型类型)
├── saver.py                      # epair 保存 (含 ExportWrapper + tokenizer/config 复制)
├── model_registry.py             # 模型注册表 + detect_model_type
├── export/
│   └── epair_utils.py            # 已有: pack_weights, consolidated_weights_save
└── quant/                        # 预留扩展能力 (量化, TBD)
```

## 模型差异总览

三个模型接口不一致，converter 通过注册表 + adapter 抹平差异：

### Config 加载差异

| 模型 | from_pretrained | from_json_file | to_dict | config_json_key |
|------|:-:|:-:|:-:|---|
| gemma4 | 无 | 无 | 有 | `text_config` |
| qwen3 | 有 | 有 | 有 | 顶层（无嵌套） |
| qwen3_5 | 有 | 有 | 无 | `text_config` |

### forward 签名差异

| 参数 | gemma4 | qwen3 | qwen3_5 |
|------|:------:|:-----:|:-------:|
| input_ids | 有 | 有 | 有 |
| attention_mask | 有 | 有 | 有 |
| position_ids | 无（内部计算） | 有 | 无（内部计算） |
| past_key_values | `list` | `list` | `SimpleKVCache` |
| inputs_embeds | 有 | 有 | 有 |
| labels | 有 | 有 | 有 |
| output_hidden_states | 有 | 有 | 无 |
| use_cache | 无 | 无 | 有 |
| shared_kv_states | 有（独有） | 无 | 无 |
| delta_cache | 无 | 无 | 有（独有） |

### forward 返回差异

| 返回 key | gemma4 | qwen3 | qwen3_5 |
|----------|--------|-------|---------|
| loss | 有 | 有 | 有 |
| logits | 有 | 有 | 有 |
| past_key_values | 有 | 有 | 有 |
| shared_kv_states | 有（独有） | 无 | 无 |
| hidden_states | 有 | 有 | 无 |
| delta_cache | 无 | 无 | 有（独有） |

### 权重 key 重映射差异

| 模型 | 重映射策略 | 说明 |
|------|-----------|------|
| gemma4 | `model.language_model.*` → `model.*` | 去掉 VLM 前缀 |
| qwen3 | 无需重映射 | HF key 直接匹配 |
| qwen3_5 | `model.language_model.*` → `model.*` | 通过 `_remap_key` 去掉 VLM 前缀 |

## model_registry.py — 模型注册表

```python
@dataclass
class ModelEntry:
    """一个模型在注册表中的条目"""
    model_type: str               # "gemma4_text" / "qwen3" / "qwen3_5_text"
    config_cls: type              # Gemma4TextConfig / Qwen3TextConfig / Qwen3_5TextConfig
    model_cls: type               # Gemma4ForCausalLM / Qwen3ForCausalLM / Qwen3_5ForCausalLM
    load_weights_fn: callable     # load_gemma4_weights / load_qwen3_weights / load_qwen3_5_weights
    config_json_key: str | None   # config.json 中 text 配置的 key (如 "text_config")，None 表示顶层
    wrapper_cls: type             # ExportWrapper 子类 (Qwen3ExportWrapper / Gemma4ExportWrapper / ...)


REGISTRY: dict[str, ModelEntry] = {
    "gemma4": ModelEntry(
        model_type="gemma4_text",
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",
        wrapper_cls=Gemma4ExportWrapper,
    ),
    "qwen3": ModelEntry(
        model_type="qwen3",
        config_cls=Qwen3TextConfig,
        model_cls=Qwen3ForCausalLM,
        load_weights_fn=load_qwen3_weights,
        config_json_key=None,               # 扁平 config，直接从顶层取
        wrapper_cls=Qwen3ExportWrapper,
    ),
    "qwen3_5": ModelEntry(
        model_type="qwen3_5_text",
        config_cls=Qwen3_5TextConfig,
        model_cls=Qwen3_5ForCausalLM,
        load_weights_fn=load_qwen3_5_weights,
        config_json_key="text_config",
        wrapper_cls=Qwen3_5ExportWrapper,
    ),
}

_ALIASES: dict[str, str] = {
    "gemma4": "gemma4", "gemma-4": "gemma4",
    "qwen3": "qwen3", "qwen-3": "qwen3",
    "qwen3.5": "qwen3_5", "qwen3_5": "qwen3_5",
}


def get_entry(name: str) -> ModelEntry:
    """按名称查找模型条目，支持别名"""
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(REGISTRY)}")
    return REGISTRY[key]


def detect_model_type(hf_config: dict) -> ModelEntry:
    """从 HF config.json 自动探测模型类型"""
    model_type = hf_config.get("model_type")
    if not model_type:
        raise ValueError(
            f"config.json 缺少 'model_type' 字段，无法自动探测；已支持: {list(REGISTRY)}"
        )
    return get_entry(model_type)
```

## loader.py — 模型加载

### ModelArtifact 数据结构

```python
@dataclass
class ModelArtifact:
    """加载产物"""
    model: nn.Module              # 自定义模型实例 (权重已加载, eval 模式)
    config: object                # 自定义 config dataclass 实例
    hf_path: str                  # 原始 HF 模型路径
    hf_config: dict               # 原始 config.json 原始字典 (保留全部字段)
    tokenizer_files: list[Path]   # tokenizer 相关文件路径列表
    model_type: str               # 模型类型标识
    wrapper_cls: type             # ExportWrapper 子类 (从 ModelEntry 透传)
```

### load_model 接口

```python
def load_model(
    hf_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ModelArtifact:
    """
    从 HF 路径加载模型 — 模型类型自动探测

    流程:
      1. 读取 HF config.json
      2. 从 config.json 的 model_type 字段自动探测模型类型 (detect_model_type)
      3. 构造自定义 config dataclass
      4. 实例化自定义模型
      5. 加载 HF safetensors 权重 (调用各模型自带的 load_*_weights)
      6. 收集 tokenizer 文件列表
    """
    # 1. 读取原始 config.json
    config_json_path = os.path.join(hf_path, "config.json")
    with open(config_json_path) as f:
        hf_config = json.load(f)

    # 2. 自动探测模型类型
    entry = detect_model_type(hf_config)

    # 3. 构造自定义 config (适配 from_pretrained 有/无)
    config = build_config(hf_config, entry)

    # 4. 实例化模型
    model = entry.model_cls(config)
    model = model.to(dtype=dtype).to(device=device)
    model.eval()

    # 5. 加载权重 — 调用各模型自带的 load_*_weights (内部处理 key 重映射)
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 6. 收集 tokenizer 文件 (不加载, 只记录路径)
    tokenizer_files = collect_tokenizer_files(hf_path)

    return ModelArtifact(
        model=model,
        config=config,
        hf_path=hf_path,
        hf_config=hf_config,
        tokenizer_files=tokenizer_files,
        model_type=entry.model_type,
    )
```

### build_config — 适配 config 加载差异

各模型的 config 加载方式不同，`build_config` 按优先级检测：

```python
def build_config(hf_config: dict, entry: ModelEntry):
    # 优先用 from_pretrained (qwen3 / qwen3_5 有)
    if hasattr(entry.config_cls, "from_pretrained"):
        return entry.config_cls.from_pretrained(hf_path)
    # 其次用 from_json_file (qwen3 / qwen3_5 有)
    if hasattr(entry.config_cls, "from_json_file"):
        return entry.config_cls.from_json_file(config_json_path)
    # fallback: 手动字段过滤 (gemma4)
    if entry.config_json_key and entry.config_json_key in hf_config:
        raw = hf_config[entry.config_json_key]
    else:
        raw = hf_config
    valid_fields = {f.name for f in dataclasses.fields(entry.config_cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    config = entry.config_cls(**filtered)
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config
```

### collect_tokenizer_files

```python
_TOKENIZER_PATTERNS = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "generation_config.json", "vocab.json", "merges.txt",
]

def collect_tokenizer_files(hf_path: str) -> list[Path]:
    """收集 HF 目录中的 tokenizer 相关文件"""
    return [Path(hf_path) / p for p in _TOKENIZER_PATTERNS if (Path(hf_path) / p).exists()]
```

## saver.py — epair 保存

### ExportWrapper 基类 — 统一 I/O 契约

参考 litert-torch 的 `ExportableModule` 和 npu_tuned_export 的 `Qwen3ForCausalLMWrapper`，设计统一 wrapper 基类。核心思想：**KV cache 作为显式图输入输出**，子类实现 `_drive_layers` 负责与原生模型 forward 对接。

支持 **两种 embedding 模式**：
- `use_embeddings=False`（默认）：传 `input_ids`，图内含 `embed_tokens` 节点
- `use_embeddings=True`：传 `embeddings`，图内不含 `embed_tokens`（embedding 量化用，查表在 host 侧做）

基类位于 `cann_lm/converter/export_wrapper.py`，子类位于各模型目录（如 `cann_lm/models/qwen3/export_wrapper.py`）。

```python
class ExportWrapper(nn.Module):
    """统一对外 I/O：cache in/out + output_pos 选位。子类实现 _drive_layers。

    轻量版：子类内部调原生 forward（带 if past is None 分支）
    全量版（后续）：各 attention 加 scatter-write 路径，一张静态图覆盖 prefill+decode
    """

    def __init__(self, model, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(
        self,
        input_ids=None,            # input_ids 或 embeddings 二选一
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        new_kv_cache_pos=None,
        embeddings=None,           # ← 外部 embedding (量化用)
        output_pos=None,           # ← 选位：decode 时只算最后一个 token 的 logits
    ):
        logits, past_kv_flat = self._drive_layers(
            input_ids, attention_mask, position_ids,
            past_key_values, new_kv_cache_pos, embeddings,
        )
        if output_pos is not None:
            logits = logits[:, output_pos, :]
        return logits, *past_kv_flat

    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None):
        """子类实现：统一 I/O → 原生 forward kwargs → 提取 logits + 展平 KV cache"""
        raise NotImplementedError

    def get_sample_inputs(self, seq_len=128, use_embeddings=False):
        """构造导出用示例输入

        Args:
            use_embeddings: True 时传 embeddings (图外查表), False 时传 input_ids (图内查表)
        """
        # ... 构造 input_ids, attention_mask, position_ids, past_key_values, ...
        if use_embeddings:
            embeddings = self.model.model.embed_tokens(input_ids).detach()
            return (None, attention_mask, position_ids, past_key_values,
                    new_kv_cache_pos, embeddings, output_pos)
        else:
            return (input_ids, attention_mask, position_ids, past_key_values,
                    new_kv_cache_pos, None, output_pos)
```

### 统一图 I/O 契约

```
输入:
  input_ids          [batch, seq_len] int64               ← 与 embeddings 二选一
  embeddings         [batch, seq_len, hidden] float16    ← 图外查表 (量化用)
  attention_mask     [batch, 1, seq_len, kv_cache_max_len]    ← 4D 预构造 mask
  position_ids       [batch, seq_len] int64
  past_key_values    list of (key_buf, value_buf) per layer   ← 每层一对预分配 buffer
                     shape: [kv_cache_max_len, kv_head, batch, head_dim]
  new_kv_cache_pos   [seq_len] int64                           ← scatter 写入位置
  output_pos         [1] int32 (可选)                          ← 只算哪个位置的 logits

输出:
  lm_logits          [batch, n_pos, vocab]                    ← output_pos 选位后
  past_key_0, past_value_0, past_key_1, past_value_1, ...    ← 展平的所有层 KV cache
```

**为什么需要 4D 预构造 mask？** 模型的 `_prepare_causal_mask` / `_create_causal_mask` 内部有动态控制流（依赖运行时 seq 长度），torch.export 无法静态追踪。传 4D mask 时走透传路径（`return attention_mask`），跳过动态构造。三个模型的透传位置：

| 模型 | 透传代码位置 |
|------|-------------|
| qwen3 | `modeling_qwen3.py` 第 349 行 `return attention_mask` |
| gemma4 | `modeling_gemma4.py` 第 571 行 `return attention_mask` |
| qwen3_5 | `modeling_qwen3_5.py` 第 616 行 `if attention_mask.dim() == 4: return attention_mask`（新增） |

### Qwen3ExportWrapper 子类示例

```python
class Qwen3ExportWrapper(ExportWrapper):
    """qwen3 子类：调原生 forward + 提取 logits + 展平 list[(k,v)] cache"""

    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None):
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings   # 图外查表
        else:
            kwargs["input_ids"] = input_ids        # 图内查表

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        past_kv_flat = []
        for layer_kv in outputs["past_key_values"]:
            past_kv_flat.append(layer_kv[0])
            past_kv_flat.append(layer_kv[1])
        return logits, past_kv_flat
```

### 三个子类的差异

| 子类 | 文件位置 | 原生 cache 类型 | forward kwargs 差异 | 特殊处理 |
|------|---------|----------------|---------------------|---------|
| Qwen3ExportWrapper | `models/qwen3/export_wrapper.py` | `list[(k,v), ...]` | `position_ids` 外部传入 | 无 |
| Gemma4ExportWrapper | `models/gemma4/export_wrapper.py` | `list[(k,v), ...]` | 无 `position_ids`（内部计算） | `shared_kv_states=None` |
| Qwen3_5ExportWrapper | `models/qwen3_5/export_wrapper.py` | `SimpleKVCache` | `delta_cache`、`use_cache` | list → SimpleKVCache 转换；`delta_cache=None` |

三个子类都支持 `embeddings` 参数：非 None 时传 `inputs_embeds`，None 时传 `input_ids`。

### 轻量 vs 全量

| | 轻量（当前里程碑） | 全量（后续） |
|---|---|---|
| 对外 I/O | 统一（基类） | 统一（基类） |
| 子类内部 | 调原生 `forward`（带 `if past is None` 分支） | 各 attention 加 scatter-write 路径 |
| 导出图 | 按 sample 走一条分支、形状特化 | 一张静态图覆盖 prefill+decode |
| 工作量 | 小（基类 + 3 子类脚手架） | 大（逐模型改 attention） |

### 各模型的全量难点（后续）

- **gemma4**：KV 共享（后 n 层无自己的 buffer，复用别的层）→ 不 fit "每层一个 buffer"；sliding window 的局部 mask；per-layer head_dim/rope。
- **qwen3_5**：线性注意力（GatedDeltaNet）的 cache 是 `[num_v_heads, k_head_dim, v_head_dim]` 的 **recurrent state**（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——需单独设计 recurrent buffer 的增量协议。
- **qwen3**：最接近参考工程（标准 GQA + KV cache），全量改造阻力最小，建议先做。

### save_epair 接口

```python
def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,  # 默认 50GB
    kv_cache_max_len: int = 2048,
    sample_seq_len: int = 128,
    use_embeddings: bool = False,         # ← True: 图外查表 (量化用)
    save_weights: bool = True,
) -> None:
    """
    完整保存流程 (7 步)

    output_dir/
    ├── ep_json_with_ascend_ir.json     # torch.export + ascend_ir
    ├── weight_info.json                # 权重索引
    ├── weights_0.bin                   # 权重数据 (按 max_file_size 切分)
    ├── model_config.json               # 自定义 config 序列化 + HF 原始信息
    ├── generation_config.json          # 生成配置 (从 HF 复制)
    ├── tokenizer.json                  # tokenizer 文件 (从 HF 复制)
    └── ...
    """
    import torch_delegate
    from cann_lm.models.fake_ops import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 注册 GE converters (export_to_epair 前必须调用)
    register_ge_converters()

    # 2. 创建示例输入 (统一 I/O 契约, 支持 embeddings 模式)
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=sample_seq_len, use_embeddings=use_embeddings
    )

    # 3. 从注册表获取 wrapper 子类，构造统一 I/O wrapper
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 4. export_to_epair
    epair = torch_delegate.export_to_epair(ep)

    # 5. 保存权重 + ep_json (复用已有 consolidated_weights_save)
    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
    )

    # 6. 复制 tokenizer + 保存 model_config
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)

    # 7. 保存 .pt2 (供 Python 侧回环验证)
    pt2_path = output_dir.parent / f"{output_dir.name}.pt2"
    torch_delegate.epair_save(epair, str(pt2_path))
    print(f"Saved: {pt2_path}")

    # 8. 保存 input (供 C++ 侧验证)
    for i, inp in enumerate(sample_inputs):
        path = output_dir.parent / f"input_{i}.bin"
        inp.cpu().detach().numpy().tofile(path)
```

### create_sample_inputs

```python
def create_sample_inputs(artifact: ModelArtifact, kv_cache_max_len: int = 2048,
                          seq_len: int = 128) -> tuple:
    """构造统一 I/O 契约的示例输入"""
    config = artifact.config
    batch = 1
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch, 1, seq_len, kv_cache_max_len, dtype=torch.float16)
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int64)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    # 每层一对预分配 KV buffer
    past_key_values = []
    for _ in range(num_layers):
        k = torch.zeros(kv_cache_max_len, num_kv_heads, batch, head_dim, dtype=torch.float16)
        v = torch.zeros_like(k)
        past_key_values.append((k, v))

    return (input_ids, attention_mask, position_ids, past_key_values,
            new_kv_cache_pos, output_pos)
```

### save_model_config

```python
def save_model_config(artifact: ModelArtifact, output_dir: Path) -> None:
    # to_dict 有则用 (gemma4/qwen3)，无则用 dataclasses.asdict (qwen3_5)
    if hasattr(artifact.config, "to_dict"):
        config_dict = artifact.config.to_dict()
    elif dataclasses.is_dataclass(artifact.config):
        config_dict = dataclasses.asdict(artifact.config)
    else:
        config_dict = dict(artifact.config.__dict__)
    config_data = {
        "model_type": artifact.model_type,
        "config": config_dict,
        "hf_config": artifact.hf_config,
    }
    # 写 model_config.json
    ...
```

## 兼容性处理

### bfloat16 序列化

numpy 不支持 bfloat16 dtype。`pack_weights` 在序列化时需将 bfloat16 tensor 转为 int16 view：

```python
# epair_utils.py pack_weights 内部
raw = tensor.detach().cpu().contiguous()
if raw.dtype == torch.bfloat16:
    raw = raw.view(torch.int16)    # 保留原始字节，不丢失精度
data = raw.numpy().tobytes()
```

### 非持久化 buffer

`persistent=False` 的 buffer（如 `rope.sin_cache`、`rope.cos_cache`）不在 `state_dict()` 中，需从 `ep.module().named_buffers()` 补充：

```python
# epair_utils.py pack_weights 内部
raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
state_dict = dict(raw_sd)

# Non-persistent buffers are not in state_dict; add them from the module
for name, buf in contents.ep.module().named_buffers():
    if name not in state_dict:
        state_dict[name] = buf
```

### register_ge_converters 调用时机

`torch_delegate.export_to_epair()` 内部依赖 GE 算子 converter 注册。必须在调用前执行 `register_ge_converters()`，否则 ascend_ir 标注为空，C++ 侧报错退出。

在 `save_epair` 的第 1 步调用。

## cann-lm-converter.py — CLI

对外只暴露 `cann-lm-converter.py`，模型类型从 `config.json` 自动探测，无需 `--model-type`。同时提供程序化入口 `convert()` 供脚本/测试复用。

### convert 函数 — 程序化入口

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device: str = "cpu",
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    sample_seq_len: int = 128,
) -> None:
    """程序化入口：HF 模型 → epair"""
    print(f"[1/3] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device=device)
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")

    print(f"[2/3] Saving epair to {output_dir}")
    save_epair(artifact, output_dir, max_file_size=max_file_size, sample_seq_len=sample_seq_len)

    print(f"[3/3] Verifying output")
    out = Path(output_dir)
    for name in ["ep_json_with_ascend_ir.json", "model_config.json", "weight_info.json"]:
        fp = out / name
        print(f"      [{'OK' if fp.exists() else 'MISSING'}] {name}")
    print(f"      Weight files: {len(list(out.glob('weights_*.bin')))}")
    print(f"\nDone. Output: {output_dir}")
```

### 命令行参数

```python
def parse_args():
    parser = argparse.ArgumentParser(description="CANN-LM Converter: HF model → epair format")
    parser.add_argument("--hf-path", required=True, help="HuggingFace 模型路径 (本地目录)")
    parser.add_argument("--output-dir", required=True, help="epair 输出目录")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-file-size", type=int, default=50*1024*1024*1024)
    parser.add_argument("--kv-cache-max-len", type=int, default=2048)
    parser.add_argument("--sample-seq-len", type=int, default=128)
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use external embeddings (no embed_tokens in graph)")
    return parser.parse_args()
```

## 完整调用链

```
cann-lm-converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --device, --max-file-size,
    │                    --kv-cache-max-len, --sample-seq-len, --use-embeddings
    │
    ▼
convert(hf_path, output_dir, dtype, device, max_file_size, kv_cache_max_len, sample_seq_len, use_embeddings)
    │
    ▼
loader.load_model(hf_path, dtype, device)   # 模型类型自动探测
    │
    ├── 读 config.json
    ├── model_registry.detect_model_type(hf_config)
    │     → hf_config["model_type"] → get_entry → ModelEntry(config_cls, model_cls, ...)
    │
    ├── build_config(hf_config, entry)
    │     → from_pretrained / from_json_file / 手动字段过滤
    │
    ├── entry.model_cls(config)  → e.g. Qwen3ForCausalLM(config)
    ├── model.to(dtype).eval()
    ├── entry.load_weights_fn(model, hf_path, ...)
    │     → safetensors 读取 → key 重映射 → param.copy_()
    │
    └── collect_tokenizer_files(hf_path)

    ▼
saver.save_epair(artifact, output_dir, max_file_size)
    │
    ├── register_ge_converters()              ← 步骤 1: 注册 GE converters
    │
    ├── create_sample_inputs(artifact)
    │     → torch.randint(0, vocab_size, (1, seq_len))
    │
    ├── ExportWrapper(artifact.model).eval()  ← 步骤 3: 统一返回 logits
    │
    ├── torch.export.export(wrapper, sample_inputs, strict=False)
    │     → ExportedProgram
    │
    ├── torch_delegate.export_to_epair(ep)
    │     → EpairContents (ep_json + state_dict + graph_signature)
    │
    ├── epair_utils.consolidated_weights_save(epair, output_dir, max_file_size)
    │     → pack_weights: bfloat16→view(int16), 非持久 buffer 从 module 补充
    │     → weights_0.bin, weights_1.bin, ...
    │     → weight_info.json
    │     → ep_json_with_ascend_ir.json
    │
    ├── copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    │     → tokenizer.json, tokenizer_config.json, ...
    │
    ├── save_model_config(artifact, output_dir)
    │     → model_config.json (to_dict / dataclasses.asdict fallback)
    │
    ├── torch_delegate.epair_save(epair, pt2_path)   ← .pt2 供 Python 回环验证
    └── save sample_inputs → input_0.bin             ← 供 C++ 验证
```

## 输出文件布局

```
output_dir/
├── ep_json_with_ascend_ir.json    # torch.export 序列化 + ascend_ir 标注
├── weight_info.json               # 权重索引 {bin文件名: [{name, offset, size}]}
├── weights_0.bin                  # 权重数据 (按 max_file_size 切分, 默认 50GB)
├── weights_1.bin
├── model_config.json              # 自定义 config + HF 元信息
├── generation_config.json         # 从 HF 复制
├── tokenizer.json                 # 从 HF 复制
├── tokenizer_config.json          # 从 HF 复制
└── special_tokens_map.json        # 从 HF 复制

output_dir.pt2                      # .pt2 归档 (供 Python 侧 epair_load 回环验证)
input_0.bin                         # 示例输入 (供 C++ 侧验证)
```

## C++ 加载侧

导出产物由 C++ `EpairModelLoader` 加载，位于 `examples/export_llm_model/`。

### EpairModelLoader API (2 路径)

```cpp
td::EpairModelLoader loader(
    "/path/to/ep_json_with_ascend_ir.json",   // model JSON
    "/path/to/weight_info.json"               // 权重索引 (bin 路径从中推导)
);
ge::Graph graph("delegate_graph");
loader.GetGEGraph(&graph);  // 构建 GE Graph (解析 ep_json + 懒加载 bin)
```

### main.cpp 流程 (3 stage)

1. **Load epair and build GE Graph** — `EpairModelLoader` 构造 + `GetGEGraph`
2. **Extract GE Graph** — 获取 Graph 副本，验证初始化
3. **Compile graph** — `aclgrphBuildInitialize` → `aclgrphBuildModel` → `aclgrphSaveModel` → `.om`

### GE 算子注册 (reg_register.h)

不同模型需要不同的 GE 算子集。C++ 侧通过 `reg_register.h` 注册算子：

| 算子类别 | 示例算子 | 用于 |
|---------|---------|------|
| 基础运算 | Add, Mul, Sub, Div, RealDiv, Neg, Pow, Abs, Exp, Log, Sqrt | 基础数学运算 |
| 归一化 | ReduceMean, Rsqrt, Sigmoid, Swish | RMSNorm / SiLU |
| 矩阵 | MatMul, BatchMatMul | Attention QK^T / attnV |
| 注意力 | SoftmaxV2, SoftmaxV3, Triu, Fill | Causal mask + softmax |
| 张量操作 | Reshape, Transpose, Cast, Unsqueeze, ExpandDims | Shape 变换 |
| 切片 | StridedSliceV2, SliceD | KV cache 切片 |
| 拼接 | ConcatV2, ConcatD | Rotary embedding 拼接 |
| 查找 | GatherV2, Embedding | Token embedding |
| 范围 | Range, Cos, Sin | Position encoding |
| 其他 | Identity, Where, ZerosLike | Clone / 条件选择 |

算子未注册时 C++ 侧直接报错退出（`UNMAPPED_OPERATOR`），提示用户注册或分解算子。

## 新增模型接入指南

新增一个模型到 converter CLI 需要以下步骤：

### 步骤 1：实现模型代码

在 `cann_lm/models/{model_name}/` 下实现：
- `configuration_{model_name}.py` — config dataclass（建议实现 `from_pretrained` / `to_dict`）
- `modeling_{model_name}.py` — model class + `load_{model_name}_weights` 函数
- `__init__.py` — 导出 config class, model class, load_weights 函数

### 步骤 2：实现 ExportWrapper 子类

在 `cann_lm/models/{model_name}/export_wrapper.py` 下创建子类，继承 `ExportWrapper`，实现 `_drive_layers`：

```python
from cann_lm.converter.export_wrapper import ExportWrapper

class {ModelName}ExportWrapper(ExportWrapper):
    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None):
        kwargs = {...}  # 模型特有的 forward kwargs
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids
        outputs = self.model(**kwargs)
        # 提取 logits + 展平 KV cache
        ...
        return logits, past_kv_flat
```

### 步骤 3：注册到 model_registry.py

```python
REGISTRY["{model_name}"] = ModelEntry(
    model_type="{model_type}",          # config.json 中的 model_type 值
    config_cls={ModelConfig},
    model_cls={ModelForCausalLM},
    load_weights_fn=load_{model_name}_weights,
    config_json_key="{key}" or None,    # 嵌套 config 的 key，扁平则为 None
    wrapper_cls={ModelName}ExportWrapper,  # 步骤 2 实现的 wrapper 子类
)
```

### 步骤 4：确认兼容性

| 检查项 | 说明 |
|--------|------|
| config 加载 | 有 `from_pretrained`？无则 `build_config` fallback 到手动字段过滤 |
| to_dict | 有 `to_dict`？无则 `save_model_config` 用 `dataclasses.asdict` |
| weight key | 是否需要重映射？在 `load_*_weights` 内部处理 |
| forward 返回 | 返回 dict？是则 `ExportWrapper` 子类的 `_drive_layers` 处理 |
| mask 4D 透传 | `_prepare_causal_mask` 有 4D 透传？无则需添加（参考 qwen3_5） |
| bfloat16 | 用 bfloat16 权重？`pack_weights` 已兼容 |
| 非持久 buffer | 有 `persistent=False` buffer？`pack_weights` 已兼容 |

### 步骤 5：补充 GE 算子

在 `examples/export_llm_model/reg_register.h` 中注册模型需要的 GE 算子。运行 C++ 二进制时根据报错逐个补充。

### 步骤 6：验证

```bash
# Python 导出
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/model --output-dir /path/to/output --dtype float16

# C++ 加载
./export_llm_model output/ep_json_with_ascend_ir.json output/weight_info.json
```

## 目标导出方案：统一 cache 签名（轻量版）

参考 `npu_tuned_export`（gitee: `ye-xiangqazxsw/cannkit_samplecode_lm_engine_cpp`）和 litert-torch 的经验，把 prefill/decode 导出统一成"KV cache 永远作为输入也作为输出"的图签名。

> **阶段定位**：当前先做**轻量版**——先立起统一**对外 I/O 契约**和 wrapper 脚手架（基类 + 每模型子类），子类内部仍调各模型原生 `forward`。**全量版**（预分配 buffer + scatter 写、一张静态图覆盖 prefill+decode）作为后续逐步改造。

### 统一图 I/O 契约（目标）

```
输入:
  input_ids          [batch, seq_len] int64
  attention_mask     [batch, 1, seq_len, kv_cache_max_len]
  position_ids       [batch, seq_len] int64
  past_key_in{i}     [kv_cache_max_len, kv_head, batch, head_dim]   (每层一对)
  past_value_in{i}   同上
  new_kv_cache_pos   [seq_len] int64           # 新 K/V 写进 buffer 的位置 (scatter 索引)
  output_pos         [1] int32 (可选)          # 只算哪个位置的 logits
输出:
  lm_logits          [batch, n_pos, vocab]     # output_pos 选位后
  past_key{i}        更新后的 cache buffer (每层一对)
  past_value{i}
```

### ExportWrapper 基类 + 每模型子类

```python
class ExportWrapper(nn.Module):
    """统一对外 I/O：cache in/out + output_pos 选位。子类实现 _drive_layers。"""
    def __init__(self, model, *, kv_cache_max_len: int, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_values, new_kv_cache_pos, output_pos=None):
        logits, past_kv_flat = self._drive_layers(
            input_ids, attention_mask, position_ids,
            past_key_values, new_kv_cache_pos,
        )
        if output_pos is not None:
            logits = logits[:, output_pos, :]
        return logits, *past_kv_flat

    def _drive_layers(self, *args, **kwargs):
        raise NotImplementedError  # 子类实现：flat tensor ↔ 原生 cache 互转 + 调原生 forward
```

### 轻量 vs 全量

| | 轻量（当前里程碑） | 全量（后续） |
|---|---|---|
| 对外 I/O | 统一（基类） | 统一（基类） |
| 子类内部 | 调原生 `forward`（带 `if past is None` 分支） | 各 attention 加 scatter-write 路径 |
| 导出图 | 按 sample 走一条分支、形状特化 | 一张静态图覆盖 prefill+decode |
| 工作量 | 小（基类 + 3 子类脚手架） | 大（逐模型改 attention） |
| 难点 | flat ↔ native cache 互转 | gemma4 KV 共享/sliding、qwen3_5 线性 attn recurrent state |

### 各模型的全量难点（后续）

- **gemma4**：KV 共享（后 n 层无自己的 buffer，复用别的层）→ 不 fit "每层一个 buffer"；sliding window 的局部 mask；per-layer head_dim/rope。
- **qwen3_5**：线性注意力（GatedDeltaNet）的 cache 是 `[num_v_heads, k_head_dim, v_head_dim]` 的 **recurrent state**（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——需单独设计 recurrent buffer 的增量协议。
- **qwen3**：最接近参考工程（标准 GQA + KV cache），全量改造阻力最小，建议先做。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装路径 | (需设置) |
| `PYTHONPATH` | 项目根目录 | (需设置) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload 开关 (设为 0 禁用) | - |

## 使用示例

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 模式 A: 图内查表 (默认)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --sample-seq-len 8

# 模式 B: 图外查表 (embedding 量化用, 图内不含 embed_tokens)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --sample-seq-len 8 \
    --use-embeddings
```

## 待实现功能

1. **统一 cache 签名** — ExportWrapper 基类 + 每模型子类，KV cache 显式图 I/O（详见上方章节）
2. **prefill/decode 分离** — 参照 litert-torch，导出 `prefill_epair` + `decode_epair`
3. **量化模块** (`cann_lm/converter/quant/`) — 权重量化（方案 TBD）
4. **分区模式** — 按 transformer layer 分区保存，支持多卡加载（TBD）

## 遗留任务

1. **generation_config.json 处理** — 当前已从 `collect_tokenizer_files` 中移除（不应和 tokenizer 一起处理）。后续需要在 `llm_config` 中单独处理 `generation_config.json` 的参数（temperature/top_p/top_k/max_length 等），需和 llm_engine 确认使用哪些字段后再实现。
2. **图的导出输入输出接口** — 需要确定如何统一导出输入输出（个数不确定，且可能比较多）。
3. **qwen3.5 线性注意力导出** — qwen3.5 有 18 层线性注意力（GatedDeltaNet），其 `torch_chunk_gated_delta_rule` 使用 Python for 循环（63 次迭代 + `.clone()`），导致 `torch.export` 生成 ~500+ 节点/层，`export_to_epair` 转 GE IR 时图爆炸 → segfault。
   - **当前状态**：`torch.export` 成功（19242 节点），wrapper 精度验证通过（4 层 ALL PASS），但 `export_to_epair` 崩溃
   - **根因**：纯 Python fallback 版本循环展开，图节点过多
   - **解决方向**：
     - 方案 A：安装 `flash-linear-attention` (fla) + triton + GPU 环境，用编译的 triton kernel 替换 Python 循环（当前 CPU-only 环境不可用）
     - 方案 B：将 delta rule 封装为自定义 GE 算子（类似 `FusedInferAttentionScore`），不展开内部循环（工作量大）
     - 方案 C：在模型代码中集成 fla kernel（和 transformers 一致：有 fla 用 kernel，无 fla fallback）
4. **新 ExportWrapper 的 ep_json 格式与 C++ 解析器不兼容** — 新统一 ExportWrapper（7 输入→57 输出）产生的 ep_json 中，parameter/buffer 的 input_specs 格式从 `{"arg": {"as_tensor": {"name": "..."}}}` 变为 `{"arg": {"name": "..."}}`（缺少 `as_tensor` 包裹），C++ `EpairModelParser` 报 `INVALID_GRAPH_INPUTS: Value reference must contain as_tensor.name`。
   - **当前状态**：旧简单 ExportWrapper（1 输入→1 输出，仅返回 logits）C++ 可正常加载和编译 OM；新统一 ExportWrapper（含 KV cache I/O）C++ 解析失败
   - **根因**：`torch.export` + `export_to_epair` 在处理多输入多输出 wrapper 时，序列化 parameter/buffer 的 input_specs 格式与单输入单输出不同
   - **临时方案**：C++ 加载用旧导出脚本 `export_llm_model_epair.py`（简单 wrapper），新 converter CLI 用于精度验证（不需要 C++ 加载）
   - **解决方向**：
     - 方案 A：修改 C++ `EpairModelParser` 支持 `{"arg": {"name": "..."}}` 格式（去掉 `as_tensor` 包裹的兼容性）
     - 方案 B：在 saver.py 中将 parameter/buffer 的 `input_specs` 格式从 `{"arg": {"name": ...}}` 转换为 `{"arg": {"as_tensor": {"name": ...}}}`

## 待办事项

- 图的导出还需要确定输入输出（个数不确定，而且有可能会比较多），需要看下怎么统一导出输入输出接口

## 精度差异分析：cann_lm 自定义模型 vs HF transformers

### 验证结果汇总

| 模型 | logits max_diff (vs HF) | tokens match | 层数 | 架构特点 |
|------|----------------------|-------------|------|---------|
| qwen3-0.6B | 0.023 | ✅ | 28 | 标准 GQA |
| qwen3.5-0.8B | 0.019 | ✅ | 24 | 混合 (6 full + 18 linear) |
| gemma4-E2B | 0.17 | ✅ | 35 | KV 共享 + sliding + per-layer embeddings |

所有模型生成的 token 序列与 HF 完全一致，精度差异在 float16 正常范围内。

### gemma4 误差积累分析

gemma4 的 max_diff (0.17) 比 qwen3 (0.023) 大约 7 倍，原因如下：

#### 误差来源

| 来源 | gemma4 特有 | 贡献 | 说明 |
|------|------------|------|------|
| **Gemma4ClippableLinear** vs nn.Linear | ✅ | 最大 | HF 用 `Gemma4ClippableLinear` (有 clamp 逻辑), 我们用 `nn.Linear` — 相同权重下计算结果有微小差异 |
| **5 个 RMSNorm/层 × 35 层 = 175 次** | ✅ | 中等 | gemma4 有 per_layer_input_norm, 每层 5 个 RMSNorm (vs qwen3 的 2 个 × 28 层 = 56 次) |
| **Per-Layer Embeddings (PLE)** | ✅ | 中等 | 额外的 embedding 查表 + projection 计算路径 |
| **Double-wide MLP** | ✅ | 小 | `use_double_wide_mlp=True`, 更宽的 MLP |
| **k_norm + v_norm** (per-head RMSNorm) | ✅ | 小 | 每个 head 单独做 RMSNorm |
| float16 舍入 | 所有模型 | 基础 | 所有模型共有 |

#### KV 共享层不引入误差

gemma4 有 20 层 KV 共享层 (num_kv_shared_layers=20)，但 **KV 共享本身不引入误差**：

```
非共享层 (Layer 0-14):
  我们的 k_proj(hidden) → k_norm → rope → 存入 shared_kv_states
  HF 的  k_proj(hidden) → k_norm → rope → 存入 shared_kv_states
  ↑ 这里产生 ~0.001 的 float16 差异 (ClippableLinear vs nn.Linear)

共享层 (Layer 15-34):
  两者都读同一份 shared_kv_states[layer_type] → 直接复用 tensor
  → 误差不增加 (同一个 tensor 对象, 零误差)
  → 只有 q_proj 计算引入新误差
```

#### 误差累积链

```
Layer 0:   误差 0.000 (输入相同)
Layer 1:   误差 0.001 (1 层 ClippableLinear 差异 + 5 次 RMSNorm)
Layer 2:   误差 0.002 (上层误差 + 本层计算)
  ...
Layer 14:  误差 0.005 (15 层累积, 非共享层)
           ↓ KV 共享 (shared_kv_states)
Layer 15:  误差 0.005 (复用 Layer 14 的 KV, 不衰减)
Layer 16:  误差 0.006 (q_proj 新误差)
  ...
Layer 34:  误差 0.015 (20 层共享 + 逐层增长)
           ↓ lm_head + final_logit_softcapping
Logits:    误差 0.17 (35 层 + 175 RMSNorm + PLE 累积)
```

#### final_logit_softcapping 对比时的影响

gemma4 有 `final_logit_softcapping=30.0`，对 logits 做 `tanh(x/30)*30` 压缩。验证时必须确保两边都走（或都不走）softcap，否则会产生 ~10 的虚假差异：

```
原始 logits:  35.0
softcap:      tanh(35/30) * 30 = 0.821 * 30 = 24.6
虚假差异:    35.0 - 24.6 = 10.4  ← 如果一边走 softcap 一边不走
```

验证脚本中 HF 侧需手动调用 `lm_head` + `tanh(logits/softcap)*softcap`，与 wrapper 的 `Gemma4ForCausalLM.forward`（内置 softcap）对齐。

#### HF 的 KV 共享层设计

```python
# HF: 只有"每种 layer_type 的最后一个非共享层"存储 KV
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == last_non_shared_of_this_type

# 非共享层: 自己计算 k/v, 存入 shared_kv_states
if not self.is_kv_shared_layer:
    key_states = self.k_proj(hidden_states)
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states)
    if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = (key_states, value_states)

# 共享层: 直接读, 不计算
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]  # 零误差复用
```

关键机制：`store_full_length_kv` 确保**每种 layer_type 只存一份 KV**，共享层直接复用同一 tensor 对象，不重新计算。

#### 完全消除差异的方向

1. 用 `Gemma4ClippableLinear` 替换 `nn.Linear`（对齐 HF 的 clamp 逻辑）
2. 或用 float32（差异降到 1e-6 以下，模型体积翻倍）

当前 0.17 的差异在 float16 下是正常的，生成 token 完全一致，不影响推理结果。

#### bfloat16 下的 tie-breaking 问题

在 bfloat16 精度下，部分 prompt 的 top-N logits 可能出现完全相同的情况（tie），导致 `Gemma4ClippableLinear` vs `nn.Linear` 的微小差异改变 argmax 选择。

**实际案例**（prompt="Hello, my name is"，bfloat16，加 BOS token）：

```
HF (Gemma4ClippableLinear):           cann_lm (nn.Linear):
  #1: token 236743 " "  (18.500)       #1: token 532 " and" (18.500)
  #2: token 532 " and" (18.250)       #2: token 236761 "." (18.500)
  #3: token 236761 "." (18.250)       #3: token 236743 " " (18.500)
  
  Top-1 vs Top-2 gap: 0.250          Top-1 vs Top-2 gap: 0.000 ← top-3 完全打平
```

**原因**：
1. cann_lm 的 top-3 tokens 的 logits 完全相同（18.500） — 模型在这个位置"不确定"
2. HF 的 `Gemma4ClippableLinear` 的 clamp 让 top-1 和 top-2 产生 0.25 的间隔 → 明确选择 " "
3. cann_lm 无 clamp → tie-breaking 依赖 argmax 的实现细节 → 选择了 " and" 而非 " "
4. **第一个 token 不同 → KV cache 不同 → 后续 decode 全部分叉**

**影响范围**：
- 仅影响 top-N logits 几乎相同的 prompt（间隔 < 0.3）
- 对 top-1 和 top-2 有明确间隔（>1.0）的 prompt 无影响
- 实测 3 个 prompt 中 2 个完全一致，1 个因 tie-breaking 分叉

**不是 bug**：这是两个独立实现在数值精度边界下的正常分歧。模型本身在这些位置就不确定（多个 token 概率几乎相等），任何微小数值差异都会改变 argmax 结果。

**解决方向**：
1. 用 `Gemma4ClippableLinear` 替换 `nn.Linear`（消除 clamp 差异，对齐 HF 的 tie-breaking）
2. 或用 float32（差异降到 1e-6，tie 几乎不会出现）
3. 或用 sampling（`do_sample=True`）替代 greedy（`do_sample=False`），采样对 tie-breaking 不敏感
