# 模型导出模块设计文档

> 把 HuggingFace 模型导出为 epair 格式的完整设计：注册表 → 加载 → ExportWrapper → torch.export → epair 落盘 → C++ 加载。
>
> 涵盖 `cann_lm/converter/` 下的 `model_registry.py` / `loader.py` / `saver.py` / `export_wrapper.py` / `export/`（含 C++ `csrc/`）。
>
> CLI 入口见 [cli_design.md](../cli/cli_design.md)，量化接入见 [quant_design.md](../quant/quant_design.md)。
>
> 状态: 待核对 | 最后核对: 2026-08-22

## 目录

- [概述](#概述)
- [模型注册表 model_registrypy](#模型注册表-model_registrypy)
- [loaderpy — 模型加载](#loaderpy--模型加载)
- [saverpy — epair 保存](#saverpy--epair-保存)
- [兼容性处理](#兼容性处理)
- [Export 模块 epair_utilspy](#export-模块-epair_utilspy)
- [C 加载侧](#c-加载侧)
- [GE 算子注册两层](#ge-算子注册两层)
- [输出文件布局](#输出文件布局)
- [prefilldecode 分离导出](#prefilldecode-分离导出)
- [新增模型接入指南](#新增模型接入指南)
- [精度差异分析](#精度差异分析)
- [待实现与遗留导出相关](#待实现与遗留导出相关)

---

## 概述

`cann_lm/converter/` 是把 HF 模型导出为 epair 格式的工具链。能力边界：

- **输入**：HuggingFace 模型目录（`config.json` + safetensors 权重 + tokenizer 文件）
- **输出**：epair 模型目录（`ep_json_with_ascend_ir.json` + 权重 bin + tokenizer + `model_config.json`），供 C++ 后端加载
- **量化**：进行中（详见 [quant_design.md](../quant/quant_design.md)），方案 W4A8 s8s4

### 端到端流程

```
PyTorch Model
    │
    ▼  torch.export.export
ExportedProgram
    │
    ▼  torch_delegate.export_to_epair
EpairContents (ep_json + state_dict)
    │
    ▼  consolidated_weights_save
output_dir/
├── ep_json_with_ascend_ir.json
├── weight_info.json
├── weights_0.bin
├── weights_1.bin
└── ...
    │
    ▼  C++ EpairModelLoader(model_json_path, weights_index_path)
ge::Graph
```

### 目录结构

```
cann_lm/converter/
├── __init__.py
├── cann_lm_converter.py          # CLI 入口 (pipeline) + 程序化 convert() 函数  ← 见 cli_design.md
├── loader.py                     # 模型加载 (config + weights + tokenizer, 自动探测模型类型)
├── saver.py                      # epair 保存 (save_epair / save_epair_prefill_decode / save_model_config / copy_tokenizer_files)
├── model_registry.py             # 模型注册表 (ModelEntry + REGISTRY + get_entry + detect_model_type + available_models)
├── export_wrapper.py             # ExportWrapper 基类 (forward + get_sample_inputs + make_sample_inputs)
└── export/
    ├── epair_utils.py            # pack_weights + consolidated_weights_save
    ├── ge_converter.py           # register_ge_converters (cann_lm 自定义算子 + 部分 aten 算子的 GE 映射)
    ├── _quant_matmul_s8s4_converter.py # cann_lm::quant_matmul_s8s4 GE converter (详见 quant_design.md)
    └── csrc/                     # C++ 后端 (epair_model_loader / ge_graph_builder / ge_weight_loader, 详见下文 C++ 加载侧)
```

自定义算子库位于 `cann_lm/models/fused_ops/`（`torch.ops.cann_lm.*`），每个算子 `define` schema + `register_fake` meta，再在 `ge_converter.py` 注册对应 GE converter：
`add_rms_norm` / `rms_norm` / `rotary_mul` / `rope_in_place` / `quant_matmul_s8s4` /
`fused_infer_attention_score` / `beam_search`。其中：
- **有 CPU eager kernel**（可 eager 跑 + 导出）：`add_rms_norm` / `rms_norm` / `rotary_mul`
- **仅 fake/meta（export-only，无 eager 计算）**：`rope_in_place` / `quant_matmul_s8s4` / `fused_infer_attention_score` / `beam_search` — 这类算子只占 schema/shape，真实计算靠 GE converter 映射到 NPU 算子。

## 模型注册表 model_registry.py

三个模型接口不一致，converter 通过注册表 + adapter 抹平差异。

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
    """Auto-detect the model entry from HF config.json's top-level `model_type`."""
    model_type = hf_config.get("model_type")
    if not model_type:
        raise ValueError(
            f"config.json missing 'model_type' field; supported: {available_models()}"
        )
    return get_entry(model_type)


def available_models() -> list[str]:
    """已注册模型名列表（CLI 报错提示用）"""
    return list(REGISTRY)
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
        wrapper_cls=entry.wrapper_cls,
    )
```

### build_config — 统一字段过滤

各模型 config 加载方式不同（有的 config_json_key 嵌套、有的顶层），但 `build_config` **统一走字段过滤**：先按 `config_json_key` 取出嵌套或顶层子字典，再用 dataclass 字段名过滤未知字段，最后触发 `__post_init__`（计算 head_dim 等派生量）。

```python
def build_config(hf_config: dict, entry: ModelEntry):
    # 嵌套 config (gemma4/qwen3_5): 从 text_config 子字典取；扁平 config (qwen3): 顶层
    if entry.config_json_key in hf_config:
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

> 注：`config_cls` 上的 `from_pretrained` / `from_json_file` 方法（qwen3/qwen3_5 有）不再被 `build_config` 调用——统一走字段过滤路径，避免不同 class 的加载方式分叉。

### collect_tokenizer_files

```python
_TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

_OPTIONAL_TOKENIZER_PATTERNS = [
    "tokenizer.model",      # SentencePiece 旧格式
    "vocab.json",           # BPE 旧格式
    "merges.txt",           # BPE 旧格式
]


def collect_tokenizer_files(hf_path: str) -> list[Path]:
    """收集 HF 目录中的 tokenizer 相关文件。

    必选 3 个 (tokenizer.json 系)；无 tokenizer.json 时补充旧格式
    (tokenizer.model / vocab.json / merges.txt)。
    generation_config.json 不在此处理（见遗留任务）。
    """
    files = []
    patterns = _TOKENIZER_PATTERNS
    if not (Path(hf_path) / "tokenizer.json").exists():
        patterns = patterns + _OPTIONAL_TOKENIZER_PATTERNS
    for p in patterns:
        fp = Path(hf_path) / p
        if fp.exists():
            files.append(fp)
    return files
```

## saver.py — epair 保存

### ExportWrapper 基类 — 统一 I/O 契约

参考 litert-torch 的 `ExportableModule` 和 npu_tuned_export 的 `Qwen3ForCausalLMWrapper`，设计统一 wrapper 基类。核心思想：**KV cache 作为显式图输入输出**，子类实现 `_drive_layers` 负责与原生模型 forward 对接。

支持 **两种 embedding 模式**：
- `use_embeddings=False`（默认）：传 `input_ids`，图内含 `embed_tokens` 节点
- `use_embeddings=True`（纯 embeddings）：`input_ids` **不入图**（None placeholder），host 算好 `embeddings` 传入；model forward 收到 `inputs_embeds` 跳过 `embed_tokens` 查表。gemma4 因 PLE（Per-Layer Embedding）的 `embed_tokens_per_layer` 查表也外置，**额外传 `per_layer_token_identity`**（host 算好的 token identity）；qwen3/qwen3_5 无 PLE，仅传 `embeddings`。

> 注：`use_embeddings=True` 时 `saver` 在 `get_sample_inputs`（用它算 sample）+ host 权重抽取**之后**、`torch.export` **之前** 调用 `wrapper.detach_externalized_lookups()` 把外置查表子模块（`embed_tokens`，gemma4 加 `embed_tokens_per_layer`）从模型 `del` 掉 → torch.export 不 lift 它们 → `weights_*.bin` **不含**冗余查表权重（≈ OM，GE 编译本就剪掉它们），查表权重单独打包到 `host_weights_info.json` + `host_weights_*.bin` 给 host（带 dtype/shape）。依赖纯 embeddings forward 不访问这些子模块（`embed_tokens` 被 `inputs_embeds` 跳过、gemma4 `embed_tokens_per_layer` 被实例方法替换跳过）；`saver` 导出后断言它们不在 `inputs_to_parameters` 兜底。tied qwen3：del `embed_tokens` 后 `lm_head.weight`（同数据）仍在 → 被 OM bake；host 从 `host_weights` 拿 `embed_tokens.weight` 查表（lm_head/embed_tokens 双角色，非冗余）。

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
        input_ids=None,            # use_embeddings=False 时传; =True 时 None (不入图)
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        new_kv_cache_pos=None,
        embeddings=None,           # ← use_embeddings=True: host 算好的 embeddings
        output_pos=None,           # ← 选位：decode 时只算最后一个 token 的 logits
        per_layer_token_identity=None,  # ← 仅 gemma4 (PLE): host 算好的 token identity
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
  input_ids          [batch, seq_len] int64               ← use_embeddings=False 时传;
                                                            =True 时 None (不入图)
  embeddings         [batch, seq_len, hidden] float16    ← use_embeddings=True: host 算好的
                                                            embed_tokens 查表结果
  per_layer_token_identity  [batch, seq_len, L, h_pl] float   ← 仅 gemma4 (PLE):
                                                            host 算好的 embed_tokens_per_layer
                                                            查表结果 (reshape 后)
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
| gemma4 | `modeling_gemma4.py` 第 619 行 `return attention_mask` |
| qwen3_5 | `modeling_qwen3_5.py` 第 616 行 `if attention_mask.dim() == 4: return attention_mask`（新增） |

### Qwen3ExportWrapper 子类示例

```python
class Qwen3ExportWrapper(ExportWrapper):
    """qwen3 子类：调原生 forward + 提取 logits + 展平 list[(k,v)] cache"""

    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None,
                      per_layer_token_identity=None):  # qwen3 无 PLE, 忽略
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "new_kv_cache_pos": new_kv_cache_pos,
        }
        # 纯 embeddings: embeddings 提供时只传 inputs_embeds (不传 input_ids),
        # model forward `if inputs_embeds is None` 跳过 embed_tokens 查表 → input_ids 不入图。
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

> gemma4 子类（`Gemma4ExportWrapper`）在纯 embeddings 分支额外处理 PLE：用**实例方法替换**把 `text_model._compute_per_layer_inputs` 临时替换为返回 host 算好的 `per_layer_token_identity`，使 model.forward 跳过 `embed_tokens_per_layer(input_ids)` 查表 + bf16 反查，而层循环/mask/rope/lm_head 仍复用 model.forward（不改 modeling）。`_build_external_extras` 算 `per_layer_token_identity = embed_tokens_per_layer(input_ids).reshape(b,s,L,h_pl)`。详见 `models/gemma4/export_wrapper.py`。

### 三个子类的差异

| 子类 | 文件位置 | 原生 cache 类型 | forward kwargs 差异 | 特殊处理 |
|------|---------|----------------|---------------------|---------|
| Qwen3ExportWrapper | `models/qwen3/export_wrapper.py` | `list[(k,v), ...]` | `position_ids` 外部传入 | 无 PLE；`per_layer_token_identity` 忽略 |
| Gemma4ExportWrapper | `models/gemma4/export_wrapper.py` | `list[(k,v), ...]` | `position_ids` 外部传入 | `shared_kv_states=None`；**纯 embeddings 时实例方法替换 `_compute_per_layer_inputs` 外置 PLE 查表**，收 `per_layer_token_identity` |
| Qwen3_5ExportWrapper | `models/qwen3_5/export_wrapper.py` | `SimpleKVCache` | `delta_cache`、`use_cache` | list → SimpleKVCache 转换；`delta_cache=None`；无 PLE |

三个子类都支持纯 embeddings：`embeddings` 非 None 时只传 `inputs_embeds`（不传 `input_ids`，input_ids 不入图），None 时传 `input_ids`（图内查表）。gemma4 额外用 `per_layer_token_identity` 外置 PLE。

### 轻量 vs 全量

| | 轻量（当前里程碑） | 全量（后续） |
|---|---|---|
| 对外 I/O | 统一（基类） | 统一（基类） |
| 子类内部 | 调原生 `forward`（带 `if past is None` 分支） | 各 attention 加 scatter-write 路径 |
| 导出图 | 按 sample 走一条分支、形状特化 | 一张静态图覆盖 prefill+decode |
| 工作量 | 小（基类 + 3 子类脚手架） | 大（逐模型改 attention） |

### 各模型的全量难点（后续）

- **gemma4**：KV 共享（后 n 层无自己的 buffer，复用别的层）→ 不 fit "每层一个 buffer"；sliding window 的局部 mask；per-layer head_dim/rope。
- **qwen3_5**：线性注意力（GatedDeltaNet）的 cache 是 `[num_v_heads, k_head_dim, v_head_dim]` 的 **recurrent state**（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——需单独设计 recurrent buffer 的增量协议。详见 [reference_qwen3_5_export.md](../reference_qwen3_5_export.md)。
- **qwen3**：最接近参考工程（标准 GQA + KV cache），全量改造阻力最小，建议先做。

### save_epair 接口

```python
def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,  # 默认 50GB
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = False,         # ← True: 图外查表 (量化用)
    save_weights: bool = True,
) -> None:
    """
    保存流程（不输出 .pt2 / input_N.bin，二者由 verify 脚本另行生成）

    output_dir/
    ├── ep_json_with_ascend_ir.json     # torch.export + ascend_ir
    ├── weight_info.json                # 权重索引
    ├── weights_0.bin                   # 权重数据 (按 max_file_size 切分)
    ├── model_config.json               # 自定义 config 序列化 + HF 原始信息
    └── tokenizer 文件
    """
    import torch_delegate
    from cann_lm.converter.export.ge_converter import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 注册 GE converters (export_to_epair 前必须调用)
    register_ge_converters()

    # 2. 从 ModelEntry 透传的 wrapper 子类构造统一 I/O wrapper
    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/converter/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # 3. 创建示例输入 (统一 I/O 契约, 支持 embeddings 模式)
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=use_embeddings
    )

    # 4. torch.export (strict=False 允许 None placeholder / 动态分支)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 5. export_to_epair
    epair = torch_delegate.export_to_epair(ep)

    # C++ 侧已兜底处理 (无需 Python 侧 strip ep_json):
    #   - _assert_tensor_metadata 节点: PlanNodes 顶部 IsAssertTensorMetadataNode 跳过
    #   - as_none 的 constant_input: PreparePlaceholders 里 NodeInputKind::None → continue

    # 6. 释放原始模型内存 (大模型 OOM 防护)
    #    epair.ep 持有 ExportedProgram 的独立引用, 删除 model/wrapper/ep 不影响 epair
    del ep, wrapper
    del artifact.model
    import gc; gc.collect()

    # 7. 保存权重 + ep_json (复用已有 consolidated_weights_save)
    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
    )

    # 8. 复制 tokenizer + 保存 model_config
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)
```

### get_sample_inputs — 基类实现

示例输入构造已从独立函数迁移到 `ExportWrapper.get_sample_inputs` 基类方法，saver.py 直接调用 `wrapper.get_sample_inputs(seq_len, use_embeddings)`。

基类 mask 策略：**causal-over-buffer**（scatter-write 语义），shape `[batch, 1, seq_len, kv_cache_max_len]`：

```python
# ExportWrapper.get_sample_inputs 内 mask 构造
min_val = torch.finfo(dtype).min
q_pos = torch.arange(seq_len, dtype=dtype).unsqueeze(1)    # [seq_len, 1]
k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)    # [1, kv_max]
mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, kv_max)
```

要求模型的 attention 用 `index_copy` scatter-write（新 KV 写入 buffer，attn 读取的 K = buffer 本身，维度 = kv_max）。
非 scatter 模型（仍用 `torch.cat` 拼接 past+new，K 维度 = kv_max + seq_len）导出会因维度不匹配报 broadcast 错误 — 见遗留任务。

其余输入：input_ids（random）、position_ids（arange）、new_kv_cache_pos（arange）、output_pos（`[seq_len-1]`）、past_key_values（每层一对零 buffer，per-layer head_dim 由 `_get_layer_head_dim` 决定）。

纯 embeddings 模式（`use_embeddings=True`）：`input_ids` 不入位置参数（None placeholder），改由 `embeddings = embed_tokens(input_ids).detach()`（host 算）；gemma4 经 `_build_external_extras` 额外追加 `per_layer_token_identity`。返回 7-tuple（qwen3/qwen3_5）或 8-tuple（gemma4，末位 `per_layer_token_identity`），对齐 `forward` 签名（`per_layer_token_identity` 默认 None，7-tuple 时取默认）。

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

## Export 模块 epair_utils.py

### 核心数据结构

#### EpairContents

`EpairContents` 来自 `torch_delegate`，是 epair 加载后的内存表示，包含：

- `ep`: `ExportedProgram` 对象，可调用 `state_dict()` 获取权重
- `ep_json`: 带 ascend IR 标注的模型 JSON
- `graph_signature`: 图签名，描述参数/缓冲区/常量的占位符映射

#### ConsolidatedWeights

```python
@dataclasses.dataclass
class ConsolidatedWeights:
    bin_files: list[bytes]                    # 多个 bin 文件的原始字节列表
    weights_index: dict[str, list[dict]]      # {bin文件名: [{"name", "offset", "size"}, ...]}
```

`ConsolidatedWeights` 只包含权重数据，不含 EP JSON。由 `pack_weights()` 生成，`consolidated_weights_save()` 写入磁盘。

### 文件布局

```
output_dir/
├── ep_json_with_ascend_ir.json    # 带 ascend_ir 的完整模型描述
├── weight_info.json               # 张量索引（含 bin 文件名引用）
├── weights_0.bin                  # 拼接的原始张量数据（文件 0）
├── weights_1.bin                  # 拼接的原始张量数据（文件 1，切分时产生）
├── ...                            # 更多 bin 文件（按需切分）
├── host_weights_info.json         # (仅 use_embeddings=True) host 查表权重索引
└── host_weights_0.bin             # (仅 use_embeddings=True) host 查表权重数据
                                    # embed_tokens / gemma4 embed_tokens_per_layer
```

所有文件直接写入 `output_dir/`，不再创建 `data/` 子目录。

> `host_weights_*` 仅在 `use_embeddings=True`（纯 embeddings 导出）时产生。`saver` 在 `torch.export` 前调用 `wrapper.detach_externalized_lookups()` 把外置查表子模块（`embed_tokens` / gemma4 `embed_tokens_per_layer`）从模型 `del` 掉 → torch.export 不 lift 它们 → `weights_*.bin` / `weight_info.json` **不含**查表权重（≈ OM，无冗余；GE 编译本就剪掉它们）。`consolidated_weights_save` 经 `host_side_weights` 把查表权重（`embed_tokens.weight`，gemma4 加 `embed_tokens_per_layer.weight`）单独打包到 `host_weights_info.json` + `host_weights_*.bin`（带 dtype/shape）供 host 查表。C++ `GeWeightLoader` 只读 `weight_info.json`，对 host 文件零感知。tied 模型：del `embed_tokens` 后 `lm_head.weight`（同数据）仍在 weights（被 OM bake），host 从 host_weights 拿 `embed_tokens.weight` 查表。

### 文件切分机制

`pack_weights` 支持按 `max_file_size` 参数（默认 5KB）对输出 bin 文件进行切分：

**切分规则**：

1. **逐张量检查**：遍历张量，检查将当前张量加入当前 bin 文件后是否超过 `max_file_size`
2. **不跨文件切分**：单个张量永远不会被拆分到两个文件中。若当前 bin 文件非空且加入新张量会超限，则先将当前 bin 文件封存，新张量从下一个文件开始写入
3. **超大张量**：若单个张量本身超过 `max_file_size`，它将独占一个 bin 文件（不拆分）
4. **命名规则**：bin 文件命名为 `weights_0.bin`、`weights_1.bin`、...，序号从 0 开始递增

**切分示例**

假设有 3 个张量，大小分别为 48B、36B、6000B，`max_file_size=5120`（5KB）：

```
weights_0.bin: [tensor_0: 48B] [tensor_1: 36B]    → 84B (< 5120)
weights_1.bin: [tensor_2: 6000B]                   → 6000B (单独文件，超限但不拆分)
```

对应 `weight_info.json`：
```json
{
  "weights_0.bin": [
    {"name": "tensor_0", "offset": 0,  "size": 48},
    {"name": "tensor_1", "offset": 48, "size": 36}
  ],
  "weights_1.bin": [
    {"name": "tensor_2", "offset": 0,  "size": 6000}
  ]
}
```

### 接口说明

#### pack_weights(contents, *, max_file_size=5120) -> ConsolidatedWeights

从 `EpairContents` 提取张量数据并打包为多个 bin 文件（内存中）。

参数：
- `contents`: `EpairContents` 对象
- `max_file_size`: 单个 bin 文件最大字节数，超过则切分到下一个文件。默认 5KB（测试用）

返回：
- `ConsolidatedWeights` 对象（`bin_files` + `weights_index`）

实现逻辑：
1. 从 `contents.ep.state_dict()` 收集所有参数和缓冲区
2. 根据 `graph_signature` 的 `inputs_to_parameters`、`inputs_to_buffers`、`inputs_to_lifted_tensor_constants` 筛选出图中引用的张量
3. 张量数据按 `numpy.tobytes()` 序列化
4. 按切分规则依次写入 bin 文件，构建索引（name/file/offset/size）

#### consolidated_weights_save(contents, output_dir, *, save_weights=True, max_file_size=5120, ep_json_filename="ep_json_with_ascend_ir.json", host_side_weights=None)

将 `EpairContents` 写入磁盘。

参数：
- `contents`: `EpairContents` 对象（由 `torch_delegate.export_to_epair()` 返回）
- `output_dir`: 输出目录（文件直接写入此目录，不创建子目录）
- `save_weights`: 为 `True` 时调用 `pack_weights` 生成 `weights_*.bin` 和 `weight_info.json`；为 `False` 时只写 EP JSON
- `max_file_size`: 单个 bin 文件最大字节数，传递给 `pack_weights`
- `ep_json_filename`: EP JSON 文件名（默认 `ep_json_with_ascend_ir.json`；prefill/decode 分离时分别传 `prefill_...` / `decode_...`）
- `host_side_weights`: 可选 `list[(name, tensor)]`，纯 embeddings 导出（`use_embeddings=True`）时的 host 查表权重（`embed_tokens.weight`，gemma4 加 `embed_tokens_per_layer.weight`）。非空时由 `_pack_host_side_weights_to_dir` 写入独立的 `host_weights_*.bin` + `host_weights_info.json`（条目含 `name/offset/size/dtype/shape`）。与 `weights_*.bin` 独立，C++ `GeWeightLoader` 零感知。

内部流程：
1. 若 `save_weights=True`，调用 `pack_weights(contents, max_file_size=...)` 得到 `ConsolidatedWeights`
2. 依次写入 `weights_0.bin`、`weights_1.bin`、...（受 `save_weights` 控制）
3. 写入 `weight_info.json`（受 `save_weights` 控制）
4. 若 `host_side_weights` 非空，写入 `host_weights_0.bin`、... + `host_weights_info.json`
5. 从 `contents.ep_json` 写入 `<ep_json_filename>`（始终写入）

## C++ 加载侧

导出产物由 C++ `EpairModelLoader` 加载。**加载器库源码**位于 `cann_lm/converter/export/csrc/`（`backend/epair_model_loader.cc` / `ge_graph_builder.cc` / `ge_weight_loader.cc` / `epair_model_parser.cc`，构建产物 `libtorch_delegate_backend.so`）；**示例 main + 算子注册**位于 `examples/test_converter/export_llm_model/`（`main.cpp` + `reg_register.h`）。

### EpairModelLoader API (2 路径)

```cpp
td::EpairModelLoader loader(
    "/path/to/ep_json_with_ascend_ir.json",   // model JSON
    "/path/to/weight_info.json"               // 权重索引 (bin 路径从中推导)
);
ge::Graph graph("delegate_graph");
loader.GetGEGraph(&graph);  // 构建 GE Graph (解析 ep_json + 懒加载 bin)
```

- 构造函数接收 2 个文件路径：`model_json_path` 和 `weights_index_path`
- 权重 bin 文件路径从 `weight_info.json` 的 `file` 字段自动推导（相对于索引文件目录）
- 使用 PIMPL 模式，对外 ABI 稳定
- `GetGEGraph` 返回内部 Graph 的拷贝副本，可单独编译和执行

### C++ 内部调用链

```
EpairModelLoader(model_json_path, weights_index_path)
    │
    ▼  Impl::Initialize
    ├── 读取 model JSON 文件内容
    └── GeGraphBuilder(model_json_string, weights_index_path, result)
            │
            ├── EpairModelParser::Parse(model_json) → ParsedModel
            ├── GeWeightLoader::Create(weights_index_path, model, result)
            │       │
            │       ├── 解析 weight_info.json
            │       ├── 推导 bin 文件绝对路径 (index 目录 + file 字段)
            │       └── 构建 logical_name → TensorRecordInfo 映射
            │
            └── Build(result) → ge::Graph
                    ├── PrepareContext
                    ├── PreparePlaceholders (创建 Data/Const 节点)
                    ├── PlanNodes (创建 Ascend 算子节点)
                    │   ├── 跳过 _assert_tensor_metadata 节点（见下方）
                    │   └── 遇到无法映射到 Ascend IR 的算子时直接报错退出
                    ├── BindDeferredOutputs
                    ├── BindDeferredEdges
                    └── FinalizeGraph
```

### GeWeightLoader 懒加载机制

`GeWeightLoader` 不在构造时一次性读取所有 bin 文件，而是采用 **懒加载** 策略：

1. `LoadConsolidated` 只解析 `weight_info.json`，为每个张量记录其所属 bin 文件的绝对路径（存入 `TensorRecordInfo::record_path`）
2. `ReadTensorData` 首次访问某个 bin 文件时，才将其读入 `bin_files_cache_`（`unordered_map<path, vector<uint8_t>>`）
3. 后续访问同一 bin 文件的张量直接从缓存读取

### 无法映射算子的处理

`PlanNodes` 遍历所有图节点时，若遇到 `ascend_ir.source` 为空的节点（即无法映射到 Ascend IR 的算子），**直接报错退出**，不再执行 fallback codegen。

错误信息示例：
```
PlanNodes failed: node [node_1_aten_arange_start] has no ascend_ir mapping (source is empty),
target=torch.ops.aten.arange.start, effective_target=torch.ops.aten.arange.start.
This operator cannot be converted to Ascend IR.
Please register the GE operator or decompose the operation in the model definition.
```

用户需要：
1. 在 `reg_register.h` 中注册缺失的 GE 算子，或
2. 在模型定义中用已支持的算子组合替换不支持的算子

> CANN-LM 已移除 `fallback_codegen`，未映射算子直接报错；故只需 `ge_graph_builder.cc` 的 PlanNodes 跳过即可。converter 主路径靠 C++ 侧 `PlanNodes` 顶部 `IsAssertTensorMetadataNode` 跳过 `_assert_tensor_metadata` 节点，不再依赖 Python 侧 strip（原 `export_llm_model_epair.py` 的 `_strip_assert_nodes_ep_json` 已随该脚本废弃移除）。

### _assert_tensor_metadata 节点处理

`torch.export`（以及 `torch_delegate.export_to_epair` 内部由 torchair `_optimize_and_convert` 跑的 decomposition）会自动插入 `torch.ops.aten._assert_tensor_metadata.default` 节点，用于运行时校验中间张量的 shape/dtype/stride。这类节点的特点：

- schema 为 `() -> ()`，**无输出**、无 GE 算子映射（`ascend_ir.source` 为空）；
- 纯 PyTorch 运行时验证，不是计算节点，GE 编译阶段不需要；
- 每层 transformer 约产生 9 个，qwen3-0.6B（28 层）共 ~262 个。

**问题**：若不处理，`PlanNodes` 会把它们当作 `ascend_ir.source` 为空的未映射算子，走到「无法映射算子的处理」分支报 `UNMAPPED_OPERATOR` 错误导致构图失败。

**修复**（参考 torch-delegate PR #93）：在 `GeGraphBuilder::PlanNodes` 节点循环**顶部**用 `IsAssertTensorMetadataNode(node)` 识别并 `continue` 跳过，不进入 converter/fallback/未映射算子分支。

```cpp
// ge_graph_builder.cc
bool IsAssertTensorMetadataNode(const ParsedNode &node) {
  return node.target == "torch.ops.aten._assert_tensor_metadata.default";
}

// PlanNodes 循环顶部
for (const auto &node : parser.Nodes()) {
  if (IsAssertTensorMetadataNode(node)) {
    TDLOGI("PlanNodes skipping _assert_tensor_metadata node: %s", node.name.c_str());
    continue;
  }
  // ...后续 converter / 未映射算子处理
}
```

### model_constants_config.json 分析

`data/constants/model_constants_config.json` 是 PT2 归档格式的内部元信息文件，由 `torch_delegate.epair_save()` 生成。以 qwen3-0.6B 导出为例，记录 `model.lifted_tensor_0/1`、`model.rotary_emb.inv_freq` 等常量的 `path_name`/`is_param`/`tensor_meta`。

`input_specs` 是 ep_json 中 `graph_module.signature` 下的数组字段，列出模型的所有输入。`model_constants_config.json` 中的 3 个常量**全部出现在 `input_specs` 中**（作为 `tensor_constant` 或 `buffer`）。

**信息覆盖情况**：

| 信息 | model_constants_config.json 来源 | CANN-LM_5883 来源 | 是否已覆盖 |
|------|---|---|---|
| 常量数据 | `data/constants/tensor_N` 文件 | `weights_*.bin`（`pack_weights` 已打包） | 已覆盖 |
| 常量名/类型 | `config` 对象的 key 和 `is_param` | `input_specs` 的 `kind` 和 `name` | 已覆盖 |
| dtype/shape | `tensor_meta.dtype` / `tensor_meta.sizes` | C++ 侧从 `ep_json` 图节点推导 | 已覆盖 |
| 数据存储位置 | `path_name` → `data/constants/` 目录 | `weight_info.json` 的 `file` + `offset` | 已覆盖 |

**结论**：`model_constants_config.json` 是 PT2 归档格式的冗余元信息文件。CANN-LM_5883 通过以下路径已覆盖其全部信息，**不需要单独解析 `model_constants_config.json`**：
1. **数据**：`pack_weights` 通过 `gs.inputs_to_lifted_tensor_constants` + `ep.module().named_buffers()` 将常量数据打包到 `weights_*.bin`
2. **元信息**：C++ 侧 `EpairModelParser` 解析 `input_specs` 获取 kind/name，`GeWeightLoader` 从 `weight_info.json` 获取数据位置
3. **dtype/shape**：C++ 侧从 `ep_json` 的图节点推导张量元信息

## GE 算子注册两层

GE 算子注册分两层，互补：

1. **Python 侧 `converter/export/ge_converter.py`** — `register_ge_converters()` 把 `torch.ops` 节点映射到 GE 算子，在 `export_to_epair` 前调用（save_epair 步骤 1）。覆盖：
   - `cann_lm.*` 自定义算子：`add_rms_norm`/`rms_norm`/`rotary_mul`/`rope_in_place`/`quant_matmul_s8s4`/`fused_infer_attention_score`/`beam_search`（算子定义在 `cann_lm/models/fused_ops/`，GE 映射实现见 [`fused_ops.md`](../../cann_lm_models/fused_ops/fused_ops.md) §3）
   - 部分 `aten.*` 算子：`softplus`/`eye`/`type_as`/`index_copy`/`index_put`（torch.export 残留的 aten 节点，torchair 无内置映射时补）

2. **C++ 侧 `examples/test_converter/export_llm_model/reg_register.h`** — 注册 GE 图编译期需要的算子实现（aclgrphBuildModel 前），不同模型需要的算子集不同：

| 算子类别 | 示例算子 | 用于 |
|---------|---------|------|
| 基础运算 | Add, Mul, Sub, Div, RealDiv, Neg, Pow, Abs, Exp, Log, Log1p, Sqrt | 基础数学运算 |
| 归一化 | ReduceMean, Rsqrt, Sigmoid, Swish | RMSNorm / SiLU |
| 矩阵 | MatMul, BatchMatMul | Attention QK^T / attnV |
| 注意力 | SoftmaxV2, SoftmaxV3, Triu, Fill | Causal mask + softmax |
| 张量操作 | Reshape, Transpose, Cast, Unsqueeze, ExpandDims | Shape 变换 |
| 切片 | StridedSliceV2, SliceD | KV cache 切片 |
| 拼接 | ConcatV2, ConcatD | Rotary embedding 拼接 |
| 查找 | GatherV2, Embedding | Token embedding |
| 范围 | Range, Cos, Sin | Position encoding |
| 其他 | Identity, Where, ZerosLike, Greater, SelectV2 | Clone / 条件选择 / 比较 |

算子未注册时 C++ 侧直接报错退出（`UNMAPPED_OPERATOR`），提示用户注册或分解算子。

> 量化算子 `QuantMatmulS8S4` + `QuantBatchMatmulV4` + `DynamicQuant` 的注册见 [quant_design.md](../quant/quant_design.md) "C++ 注册与产物"。

## 输出文件布局

```
output_dir/
├── ep_json_with_ascend_ir.json    # torch.export 序列化 + ascend_ir 标注
├── weight_info.json               # 权重索引 {bin文件名: [{name, offset, size}]}
├── weights_0.bin                  # 权重数据 (按 max_file_size 切分, 默认 50GB)
├── weights_1.bin
├── model_config.json              # 自定义 config + HF 元信息 + use_embeddings 标记
├── tokenizer.json                 # 从 HF 复制
├── tokenizer_config.json          # 从 HF 复制
├── special_tokens_map.json        # 从 HF 复制
├── host_weights_info.json         # (仅 use_embeddings=True) host 查表权重索引 {bin: [{name,offset,size,dtype,shape}]}
└── host_weights_0.bin             # (仅 use_embeddings=True) embed_tokens / gemma4 embed_tokens_per_layer
```

> `save_epair` **不输出** `.pt2`（Python 回环验证）和 `input_N.bin`（C++ 验证输入），二者由 verify 脚本另行生成，不纳入正式产物（与 prefill-decode 模式一致）。
>
> `host_weights_*` 仅 `use_embeddings=True` 时产生：`saver.detach_externalized_lookups()` 在导出前把外置查表子模块（`embed_tokens` / gemma4 `embed_tokens_per_layer`）从模型 del 掉 → torch.export 不 lift → `weights_*.bin` **不含**它们（≈ OM，无冗余）；查表权重单独打包到 `host_weights_*`（带 dtype/shape）供 host 查表。`model_config.json` 的 `use_embeddings` 字段记录导出模式，供下游判断图第一输入是 `input_ids` 还是 `embeddings`。C++ `GeWeightLoader` 只读 `weight_info.json`，对 host 文件零感知。

## prefill/decode 分离导出

实际 LLM 推理需要 prefill（处理 prompt，seq_len 可变）+ decode（逐 token 生成，seq_len=1）两个阶段。`torch.export` 形状特化后单张图无法覆盖两个阶段（导出时 seq_len=128 的图既不能 prefill 也不能 decode），故 converter 支持 `--mode prefill-decode` 导出两张图。

### 设计

- **prefill 图**：seq_len=`seq_len`（默认 128，可配）特化。实际推理时 input_ids pad 到 `seq_len`，attention_mask 屏蔽 pad 位置，output_pos=[prompt_len-1] 取最后实际 token 的 logits
- **decode 图**：seq_len=1 特化，所有 decode 步复用
- **权重共享**：两张图的 state_dict 相同（同一模型），`pack_weights` 只执行一次，`weight_info.json` + `weights_*.bin` 只写一份

### 接口

```python
from cann_lm.converter.saver import save_epair_prefill_decode

save_epair_prefill_decode(
    artifact, output_dir,
    *,
    max_file_size=50*1024**3,      # 单 bin 最大字节数
    kv_cache_max_len=2048,         # KV buffer 第一维
    seq_len=128,           # prefill 图 seq_len (可配)
    use_embeddings=False,          # True: 图外查表
    save_weights=True,             # 是否写 weights_*.bin (False 只写 ep_json)
)
```

> CLI 用法见 [cli_design.md](../cli/cli_design.md) "prefill/decode 分离模式 CLI"。

### 产物布局

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # prefill 图 (seq_len 由参数指定)
├── decode_ep_json_with_ascend_ir.json    # decode 图 (seq_len=1)
├── weight_info.json                       # 共享权重索引 (一份)
├── weights_0.bin                          # 共享权重 (一份)
├── model_config.json
└── tokenizer files
```

C++ 加载两图共享一份权重：
```cpp
td::EpairModelLoader prefill_loader("prefill_ep_json_with_ascend_ir.json",
                                     "weight_info.json");
td::EpairModelLoader decode_loader("decode_ep_json_with_ascend_ir.json",
                                   "weight_info.json");
```

### 不保存 .pt2 / input_N.bin

正式产物不含 `.pt2`（Python 回环验证）和 `input_N.bin`（C++ 验证输入），由测试脚本（verify）自行生成，避免正式流程承担测试职责。

### 全链路已通（Python 导出 + C++ OM 编译）

- `aten.index_copy.default` GE converter 缺失已解决：在 `converter/export/ge_converter.py` 用 `ge.InplaceUpdate` + `ge.TransposeD` 实现了 converter。converter CLI 导出 gemma4 prefill/decode + C++ `main.cpp` 加载编译 OM 全链路跑通（prefill 6079 节点 / decode 6044 节点 / `prefill.om` + `decode.om` 各 9.4GB）。

- **纯 embeddings（`--use-embeddings`）全链路已通**：converter CLI 导出 + C++ `main.cpp` OM 编译跑通——qwen3-0.6B（`qwen3_pure_emb.om` 374MB，weights ≈ OM）+ gemma4-E2B（5B 参数 35 层含 PLE，`gemma4_e2b_pure.om` 4.57GB，ep_json 19MB / weights 4.56GB ≈ OM / host_weights 5.5GB 查表权重给 host）。`saver.detach_externalized_lookups()` 在导出前 del 外置查表子模块 → torch.export 不 lift → `weights_*.bin` 不含冗余查表权重（≈ OM，无 5.5GB 冗余）。数值验证由 OM runner（`examples/test_converter/export_llm_model/om_npu_verify/run_*_om.py`，OM vs eager/HF，ep 回环的超集）承接——原 `verify_qwen3_epair.py`/`verify_gemma4_epair.py`（`ep.module()` vs wrapper 的 ep 回环）已重构进 OM runner。gemma4 经 `run_gemma4_om.py` 验证（OM vs eager maxdiff<1.0 MATCH）+ `run_gemma4_gen.py` 文本生成通过；qwen3.5 经 `run_qwen3_5_om.py`（OM vs EP/HF MATCH）。**注**：qwen3（base）`run_qwen3_om.py` 暴露 OM 数值异常（OM vs eager maxdiff≈20，cann_lm eager≈HF 正确、OM 输出错误），疑在其动态 RoPE（Pow/Cos/Sin，gemma4 用预缓存表不触发），OM 计算侧待排查。

### 逐图常量内联：actual_seq_qlen/kvlen → `ge.Const`（消除共享冲突 + PartitionedCall）

prefill-decode 共享 `weights_0.bin` 带来一个隐患：modeling 里 `torch.tensor([q_len])` 创建的 `actual_seq_qlen`/`actual_seq_kvlen`（`FusedInferAttentionScore` 的逐图小常量，prefill=128、decode=1）被 `torch.export` lift 成 GE 图里的 **`Data` 引用节点**（指向共享权重里的同名常量）。prefill `save_weights=True` 把自己的 128 写进 `weights_0.bin`，decode `save_weights=False` 不写却按同名复用 → decode 的融合注意力算子拿到 prefill 的 128 而非自己的 1；且 `Data` 引用输入被 GE 标记 tilingDepend → `FusedInferAttentionScore` 进 `PartitionedCall` 动态子图（950 编译还触发 `actualSeqLengthsQ(128) > Q_S(1)` 校验失败）。

**修复**（`cann_lm/converter/export/epair_utils.py: inline_fused_attn_seq_constants()` + `saver.py:save_epair_prefill_decode` 调用）：在 prefill/decode ep_json 写出后做后处理——把每个 `FusedInferAttentionScore` 算子的 `actual_seq_qlen`(input[5]) / `actual_seq_kvlen`(input[6]) 输入引用替换为内联的 `ge.Const`。函数签名 `inline_fused_attn_seq_constants(ep_json_path, qlen_value, kvlen_value=None)`：`qlen_value` = query 长度（prefill=seq_len、decode=1）；`kvlen_value` 默认 None→用 qlen_value（**prefill 兼容**，无 past 时 kvlen=qlen=seq_len 正确），**decode 显式传 `kv_cache_max_len`**（见下）。`Const` op 严格对齐 ascend_ir 既有格式（`attr.value.t.data` base64 原始字节 + `_readable_value.s`）。

**效果**：
- 逐图取值正确（decode qlen 拿到 1 不再串 prefill 的 128）→ 950 tiling 校验 `1 ≤ Q_S=1` 通过，decode OM 编译成功。
- 真 const 不再 tilingDepend → 不进 `PartitionedCall`（OM 回到 4.86GB，而非动态派生的 8.1GB 膨胀）。
- 不动 modeling（`fused_attention_forward` 保持原 `torch.tensor([q_len])`，CPU eager 不受影响；export decode 被 saver fix 覆盖为 kv_max）。

> **decode `kvlen_value=kv_cache_max_len` 的原因**（qwen3.5 FIA decode 回归根因，详见 [`ops_replacement.md`](../ops_replacement.md)「FIA 调查与现状」节）：`actual_seq_kvlen` 语义是"kv 的有效长度"（torchair 文档）。若 decode 烘成静态 `ge.Const(q_len=1)`，FIA kernel **静态截断 kv 读取到 1 个 token**（kv[0:1]）丢全部 past → `1+1=` → `2019-01-01`。正确做法是 decode kvlen=`kv_cache_max_len`（让 kernel 读全 buffer），实际 attend 范围由 **显式 `atten_mask`** 决定（mask + kvlen=kv_max 两者都要；单独 mask 不够——静态 Const=1 仍截断，mask 被忽略）。modeling 须保证 decode full-attn 层拿到非 None 的 4D `attention_mask`（qwen3.5 去掉 `if seq_len==1 and past_len>0: causal_mask=None` 特例）。
>
> **prefill `kvlen_value` 限制**（不可用 kv_max，多 chunk 未支持）：prefill 当前用默认 `kvlen_value=None`→qlen=seq_len（**仅单 chunk 正确**，无 past 时 kvlen=seq_len 对）。**不能用 kv_max**——prefill q_len>1 + kvlen=kv_max 会让 FIA kernel 读越界零位产出退化（gemma4 q_len=128 全 pad、qwen3.5 q_len=8 输出 `2222...`，已排除 mask 构造原因：两 kvlen 用同一 padded mask，仅 kvlen 值不同）。多 chunk prefill（prompt > seq_len）正确值应是动态 `new_kv_cache_pos.max()+1`（chunk N = start+seq），静态 ge.Const 无法表达 → FIA prefill 不支持多 chunk。详见 [`ops_replacement.md`](../ops_replacement.md)「prefill FIA 限制」节。
>
> **qwen3.5 当前全 eager**（2026-08-22）：FIA decode 修复已验证（committed `2f4755d`）但因 prefill 多 chunk 限制 + 方案 B（eager prefill + FIA decode via modeling `if q_len==1` 分支）撞 torch.export 双分支常量 lift 致 C++ 建图崩溃，**暂回退全 eager**（prefill+decode 都 `eager_attention_forward`）。saver fix 的 `kvlen_value` 逻辑保留（全 eager 下无 FIA 节点→no-op），`fused_attention_forward`/`_pad_mask_to_kv_square` 留 dead code 便于重启用。全 eager 下单/多 chunk prefill + decode 都正确。
>
> q_len = `query.shape[2]`：prefill = `seq_len`（参数），decode = 1（`save_epair_prefill_decode` 固定 `seq_len=1`）。内联后旧的 `lifted_tensor` / `Reshape` 引用变 dead，C++ 加载侧忽略。
>
> 相关：[`ops_replacement.md`](../ops_replacement.md) §7（关 InplaceAddRmsNorm 融合）同属"让 OM 终态对齐 [`fused_ops.md`](../cann_lm_models/fused_ops/fused_ops.md) §1 算子总表"的编译期调整，但本节是 saver/epair 格式层的逐图常量内联（导出期），二者互补。

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

在 `examples/test_converter/export_llm_model/reg_register.h` 中注册模型需要的 GE 算子。运行 C++ 二进制时根据报错逐个补充。

### 步骤 6：验证

```bash
# Python 导出
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model --output-dir /path/to/output --dtype float16

# C++ 加载
./export_llm_model output/ep_json_with_ascend_ir.json output/weight_info.json
```

## 精度差异分析

### 验证结果汇总

| 模型 | logits max_diff (vs HF) | tokens match | 层数 | 架构特点 |
|------|----------------------|-------------|------|---------|
| qwen3-0.6B | 0.023 | ✅ | 28 | 标准 GQA |
| qwen3.5-0.8B | 0.019 | ✅ | 24 | 混合 (6 full + 18 linear) |
| gemma4-E2B | 0.17 | ✅ | 35 | KV 共享 + sliding + per-layer embeddings |

所有模型生成的 token 序列与 HF 完全一致，精度差异在 float16 正常范围内。

### gemma4 export decode bug 修复 (2026-07-25)

`examples/test_converter/export_llm_model/llm_engine_on_cpu/debug_gemma4/compare_hf_vs_ep.py` 之前 EP（wrapper scatter 模式）decode 退化重复（"my name is my name is" / "人工智能人工智能"），而 CLM（cat 模式 `model.generate`）== HF。根因 + 修复：

1. **`_prepare_causal_mask` 4D sliding window decode bug**（根因，`modeling_gemma4.py`）：4D mask 路径对 sliding_attention 层套 window 时用 `q_range=arange(q_len)` 算 q 位置。decode 时 `q_len=1` → `arange(1)=[0]`，但 decode token 在序列中真实位置是 6（非 0），window 算出"只 k_pos=0 可见"，屏蔽所有其他 K → sliding 层只看第一个 prefill token → 退化。修复：`_prepare_causal_mask` 加 `new_kv_cache_pos` 参数，4D sliding 用 `new_kv_cache_pos`（q 在 buffer 的真实写入位置，与 k_pos 同基准）算 window，decode 时正确。两处调用（text_model forward 内）传 `new_kv_cache_pos`。
2. **`Gemma4ExportWrapper._drive_layers` `shared_kv_states` 持久化**（`models/gemma4/export_wrapper.py`）：wrapper 丢弃 model 返回的 `shared_kv_states`（每次 forward 新建空 dict），而 `model.generate`（`modeling_gemma4.py` line 800）正确跨步持久。修复：wrapper 存 `self._shared_kv_states`，跨 prefill/decode 调用持久（对齐 generate）。非退化根因（store 层 decode 覆盖 shared dict），但更稳健。

验证：greedy 模式 EP == HF == CLM 完全一致（"Hello, my name is" → " Alex. How are you today?"；"人工智能的未来发展方向是" → "多维度的，涵盖了从基础理论的突破到实际应用场景的深度融合。"）。sampling 模式 EP 不退化，第一 token 多数对，分叉后输出连贯（符合 bfloat16 tie-breaking 预期）。

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

#### scatter-write 下 bfloat16 attention 精度策略 (已验证, 不要改)

scatter-write 模式下，KV cache 是固定 `[kv_cache_max_len]` 的预分配 buffer，attention matmul 规模从 `[seq_len, seq_len]` 变为 `[seq_len, kv_cache_max_len]`。经逐层分析确认：

| 组件 | 精度 | 原因 |
|------|------|------|
| Q@K matmul | **bfloat16** (保持原始) | 和 HF 一致, BLAS 舍入最接近 HF 的 bfloat16 |
| softmax | **float32** (npu_tuned_export 方案, 已有) | bfloat16 下 [6, 2048] 求和有精度损失 |
| @v matmul | **bfloat16** (保持原始) | 已验证零 KV 位置不影响结果 |

**已验证的实验结论 (不要重复试)**:

1. **float32 Q@K 反而更差** — 和 HF 的 bfloat16 差异更大, 固定 seed 对比第一个 token 匹配率从 78% (7/9) 降到 33% (3/9)
   - 原因: HF 用 bfloat16 Q@K (小矩阵), 我们也用 bfloat16 (大矩阵) → BLAS 走同一路径, 舍入相似 → logits 接近
   - float32 Q@K → 和 HF 的 bfloat16 差异更大 → 匹配率降低

2. **scatter-write 下 head_dim=512 的 full_attention 层有 0.25 BLAS 舍入差异** — `bfloat16 [6, 512] @ [512, 2048]` vs `bfloat16 [6, 512] @ [512, 6]` 用不同 BLAS 代码路径
   - head_dim=256 (sliding 层): 0.0 差异 (BLAS 走同一路径)
   - head_dim=512 (full 层): 0.25 差异 → 35 层累积 → logits diff 0.4375
   - 无法消除 (float32 会引入更大差异)

3. **固定 seed + sampling 对比结果**:
   - 第一个 token 匹配率: 7/9 (78%)
   - 第二个 token 匹配率: 0/9 (0%) — 一旦分叉, KV cache 不同, 后续完全不同
   - HF 输出始终更连贯 (2B 小模型 + 0.4375 diff 足够改变采样路径)

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

## 待实现与遗留（导出相关）

1. **统一 cache 签名（全量版）** — ExportWrapper 基类 + 每模型子类（**轻量版已实现**：基类 `forward`/`get_sample_inputs` + 3 个子类 `_drive_layers`，KV cache 显式图 I/O）；**全量版待做**（各 attention 加 scatter-write 路径，一张静态图覆盖 prefill+decode）
2. **prefill/decode 分离** — **已实现**（`saver.save_epair_prefill_decode` + CLI `--mode prefill-decode`）。gemma4 Python 导出 + C++ OM 编译全链路跑通
3. **generation_config.json 处理** — 当前已从 `collect_tokenizer_files` 中移除。后续需要在 `llm_config` 中单独处理 `generation_config.json` 的参数（temperature/top_p/top_k/max_length 等），需和 llm_engine 确认使用哪些字段后再实现
4. **图的导出输入输出接口** — 需要确定如何统一导出输入输出（个数不确定，且可能比较多）
5. **qwen3.5 线性注意力导出** — 详见 [reference_qwen3_5_export.md](../reference_qwen3_5_export.md)
6. **新 ExportWrapper 的 ep_json 格式与 C++ 解析器不兼容** — **[已解决]** C++ `epair_model_parser.cc` 的 `parse_named_arg` 已支持 `arg.name` 和 `arg.as_tensor.name` 两种格式
7. **qwen3 / qwen3.5 scatter-write 改造** — 当前方案 B（预分配 buffer + `index_copy` scatter-write）仅实现 gemma4。qwen3 和 qwen3.5 的 attention 层同样需要从 `torch.cat` 改为 `index_copy` + 预分配 buffer。详见 [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) "scatter-write 改造"
8. **aten.index_copy.default GE converter 缺失** — **[已解决]** 在 `converter/export/ge_converter.py` 注册 `aten.index_copy.default` converter，用 `ge.InplaceUpdate` + `ge.TransposeD` 实现

> 量化相关待实现/遗留见 [quant_design.md](../quant/quant_design.md) "已知问题与遗留"。

## 源文件索引

| 文件 | 职责 |
|------|------|
| `cann_lm/converter/model_registry.py` | 模型注册表（ModelEntry + REGISTRY + detect_model_type） |
| `cann_lm/converter/loader.py` | `load_model` + `build_config` + `collect_tokenizer_files` |
| `cann_lm/converter/saver.py` | `save_epair` / `save_epair_prefill_decode` / `save_model_config` / `copy_tokenizer_files` |
| `cann_lm/converter/export_wrapper.py` | `ExportWrapper` 基类 |
| `cann_lm/models/{model}/export_wrapper.py` | 各模型 `ExportWrapper` 子类 |
| `cann_lm/converter/export/epair_utils.py` | 权重打包（`pack_weights`）、磁盘写入（`consolidated_weights_save`） |
| `cann_lm/converter/export/ge_converter.py` | `register_ge_converters`（cann_lm.* + aten.* → GE 算子映射） |
| `cann_lm/converter/export/csrc/inc/external/torch_delegate/epair_model_loader.h` | C++ 公共 API：`EpairModelLoader` |
| `cann_lm/converter/export/csrc/backend/epair_model_loader.cc` | `EpairModelLoader::Impl` 实现 |
| `cann_lm/converter/export/csrc/backend/ge_graph_builder.h/.cc` | `GeGraphBuilder`：解析 model JSON + 构建 GE Graph |
| `cann_lm/converter/export/csrc/backend/ge_weight_loader.h/.cc` | `GeWeightLoader`：解析 `weight_info.json`、懒加载 bin |
| `cann_lm/converter/export/csrc/backend/epair_model_parser.h/.cc` | `EpairModelParser`：解析 `ep_json_with_ascend_ir.json` |

## 文档关联

| 文档 | 关联内容 |
|------|---------|
| [cli_design.md](../cli/cli_design.md) | CLI 入口 `cann_lm_converter.py` / `convert()` / 命令行参数 / 使用示例 |
| [quant_design.md](../quant/quant_design.md) | 量化接入 `--quant-path` / `swap_to_quant_linear_from_dopt` / `quant_matmul_s8s4` |
| [converter_design.md](../converter_design.md) | 跨模块调用链总览 / 模型差异总览 / 精度汇总 |
| [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) | qwen3.5 线性注意力导出 / `ChunkGatedDeltaRule` / delta_states tensor 化（参考） |
| [reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md) | gemma4 端到端 6 阶段流程示例（参考） |
| [reference_python_to_om_self_test_flow.md](../reference_python_to_om_self_test_flow.md) | 改源码到 OM 出字的自验证流程（参考） |
