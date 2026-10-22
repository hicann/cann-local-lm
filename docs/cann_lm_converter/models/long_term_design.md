# 长期设计：统一基础设施方案

> 参考 transformers 库，逐步引入 PreTrainedConfig / ModelOutput / AutoModel 等基类
>
> 目标：使 cann_lm.models 像 transformers 一样可通过通用接口加载和推理
>
> 状态: 待核对 | 最后核对: 2026-07-14

## 背景

短期方案用 adapter 绕过了模型代码差异，但存在局限：
- 每新增模型需手动注册
- forward kwargs 手工填写
- 无统一 Cache / generate / tokenizer 接口
- dict 返回导致 torch.export 兼容性问题

长期方案通过引入基类，从模型代码层面统一接口。

## 设计原则

- **基类继承** — 所有模型继承统一的 PreTrainedConfig / PreTrainedModel
- **组合优先** — GenerationMixin / Cache 作为 Mixin，不侵入模型代码
- **渐进迁移** — 基类可向下兼容，旧模型逐步迁移
- **torch.export 原生支持** — ModelOutput 替代 dict，不再需要 ExportWrapper

## 目录结构

```
cann_lm/
├── models/
│   ├── __init__.py                  # 导出 AutoModel / AutoConfig
│   ├── base/                        # 新增: 基类定义
│   │   ├── __init__.py
│   │   ├── configuration.py         # PreTrainedConfig
│   │   ├── modeling.py              # PreTrainedModel + ModelOutput
│   │   ├── cache.py                 # Cache / DynamicCache / StaticCache
│   │   ├── generation.py            # GenerationConfig + GenerationMixin
│   │   └── registry.py              # AutoModel / AutoConfig
│   ├── gemma4/                      # 迁移: 继承基类
│   └── qwen3_5/                     # 迁移: 继承基类
```

## P0: PreTrainedConfig 基类

```python
# cann_lm/models/base/configuration.py

@dataclass
class PreTrainedConfig:
    """所有模型 config 的基类"""
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    dtype: str | None = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "PreTrainedConfig":
        """统一加载入口 — 读 config.json → 构造 config"""
        with open(f"{path}/config.json") as f:
            config_dict = json.load(f)
        # 处理嵌套 config (如 text_config)
        if hasattr(cls, "_config_key") and cls._config_key in config_dict:
            config_dict = config_dict[cls._config_key]
        config_dict.update(kwargs)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PreTrainedConfig":
        """从字典构造 — 过滤未知字段"""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        config = cls(**filtered)
        if hasattr(config, "__post_init__"):
            config.__post_init__()
        return config

    def to_dict(self) -> dict:
        """序列化为纯字典"""
        return dataclasses.asdict(self)

    def save_pretrained(self, path: str) -> None:
        """保存 config.json"""
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, path: str) -> "PreTrainedConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
```

**迁移要求：**

| 模型 | 需要做的 | 解决的问题 |
|------|---------|-----------|
| gemma4 | 补充 from_pretrained / from_json_file | 当前无统一加载入口 |
| qwen3_5 | 补充 to_dict | 当前缺失 |

## P0: ModelOutput 统一输出

```python
# cann_lm/models/base/modeling.py

@dataclass
class ModelOutput:
    """标准化模型输出 — torch.export 友好"""
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple | None = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)


@dataclass
class CausalLMOutput(ModelOutput):
    """CausalLM 专用输出"""
    pass
```

**解决的问题：**
- 不再需要 ExportWrapper（forward 直接返回 CausalLMOutput）
- torch.export 的 output_specs 只有一个 as_tensor（logits），其余为 as_none
- 但 as_none 需要在 ep_json 层过滤（当前已实现）

## P0: AutoModel / AutoConfig

```python
# cann_lm/models/base/registry.py

_REGISTRY: dict[str, type] = {}

def register_model(model_type: str):
    """装饰器 — 注册模型到 AutoModel 体系"""
    def decorator(cls):
        _REGISTRY[model_type] = cls
        return cls
    return decorator


class AutoConfig:
    @staticmethod
    def from_pretrained(path: str, **kwargs) -> PreTrainedConfig:
        with open(f"{path}/config.json") as f:
            config_dict = json.load(f)
        model_type = config_dict.get("model_type", config_dict.get("text_config", {}).get("model_type", ""))
        if model_type not in _REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}', available: {list(_REGISTRY)}")
        return _REGISTRY[model_type].from_pretrained(path, **kwargs)


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(path: str, *, dtype=torch.float16, device="cpu") -> "PreTrainedModel":
        config = AutoConfig.from_pretrained(path)
        model_cls = _REGISTRY[config.model_type]
        model = model_cls(config).to(dtype=dtype).to(device).eval()
        model.load_weights(path, device=device, dtype=dtype)
        return model
```

**使用方式：**
```python
# 注册 (在模型 __init__.py 中)
@register_model("qwen3_5")
class Qwen3_5ForCausalLM(PreTrainedModel):
    ...

# 加载 (一行代码)
model = AutoModelForCausalLM.from_pretrained("/path/to/qwen3.5-0.8B")
config = AutoConfig.from_pretrained("/path/to/qwen3.5-0.8B")
```

## P1: Cache 基类

```python
# cann_lm/models/base/cache.py

class Cache:
    """KV Cache 基类"""
    def update(self, key: Tensor, value: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_seq_len(self, layer_idx: int) -> int:
        raise NotImplementedError

    def to_legacy(self) -> list:
        """转换为旧式 list[(k,v)] — 向下兼容"""
        raise NotImplementedError


class DynamicCache(Cache):
    """动态增长缓存 — 推理默认"""
    def __init__(self):
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []

    def update(self, key, value, layer_idx):
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_len(self, layer_idx):
        return self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0


class StaticCache(Cache):
    """预分配固定大小 — 编译优化"""
    def __init__(self, max_batch_size, max_seq_len, num_layers, num_kv_heads, head_dim, dtype, device):
        self.key_cache = [torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, ...) for _ in range(num_layers)]
        self.value_cache = [...]
        self.seq_lens = [0] * num_layers
```

**迁移要求：**
- gemma4: list → DynamicCache (包装层转换)
- qwen3_5: SimpleKVCache → DynamicCache, SimpleDeltaNetCache 保留为子类

## P1: GenerationConfig + GenerationMixin

```python
# cann_lm/models/base/generation.py

@dataclass
class GenerationConfig:
    """生成参数 — 独立于模型"""
    max_length: int = 20
    max_new_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    @classmethod
    def from_pretrained(cls, path: str) -> "GenerationConfig":
        gen_path = f"{path}/generation_config.json"
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                return cls(**json.load(f))
        return cls()

    def save_pretrained(self, path: str) -> None:
        with open(f"{path}/generation_config.json", "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


class GenerationMixin:
    """generate() 通用实现 — 不写死在模型里"""

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """子类实现 — 为每步生成准备输入"""
        raise NotImplementedError

    def generate(self, input_ids: Tensor, generation_config: GenerationConfig | None = None,
                 stopping_criteria=None, **kwargs) -> Tensor:
        if generation_config is None:
            generation_config = GenerationConfig()
        # 通用 generate 实现: greedy / sample / beam
        ...
```

**迁移要求：**
- 每个模型的 `generate()` 移到 `GenerationMixin`
- 模型只需实现 `prepare_inputs_for_generation`
- `generation_config.json` 从 HF 目录自动加载

## P2: Tokenizer 集成

```python
# cann_lm/models/base/tokenization.py

class PreTrainedTokenizer:
    """统一 tokenizer 接口"""
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def batch_encode(self, texts: list[str], padding=True) -> dict: ...
    def save_pretrained(self, path: str) -> None: ...
    @classmethod
    def from_pretrained(cls, path: str) -> "PreTrainedTokenizer": ...
```

当前方案只复制 tokenizer 文件，后续可集成 `tokenizers` 库实现 encode/decode。

## P3: 多模态 Processor

```python
class ProcessorMixin:
    """组合 tokenizer + image_processor + audio_processor"""
    def __call__(self, text=None, images=None, audio=None) -> dict: ...
    @classmethod
    def from_pretrained(cls, path: str) -> "ProcessorMixin": ...
```

gemma4 / qwen3_5 有 vision_config 但当前不需要 processor，留作后续。

## 迁移路线图

### 阶段 1: P0 基类 (最小可用)
- [ ] 实现 `PreTrainedConfig` (from_pretrained / to_dict / save_pretrained)
- [ ] 实现 `ModelOutput` / `CausalLMOutput`
- [ ] 实现 `AutoConfig` / `AutoModelForCausalLM` + `@register_model` 装饰器
- [ ] gemma4 迁移到基类 (补充 from_pretrained)
- [ ] qwen3_5 迁移到基类 (补充 to_dict)
- [ ] converter 层简化: 去掉 loader.py 的适配逻辑，直接用 AutoModel

### 阶段 2: P1 推理接口
- [ ] 实现 `Cache` / `DynamicCache` / `StaticCache`
- [ ] 实现 `GenerationConfig` / `GenerationMixin`
- [ ] 各模型迁移 generate() 到 Mixin
- [ ] 各模型迁移 cache 到 DynamicCache

### 阶段 3: P2+ 生态
- [ ] Tokenizer 集成 (encode/decode/chat_template)
- [ ] Processor (多模态)
- [ ] 注意力实现切换 (eager / sdpa / flash)
- [ ] 权重管理优化 (分片加载 / tied weights 自动处理)

## 与短期方案的关系

| 短期方案 | 长期方案替代 |
|---------|-------------|
| model_registry.py 手动注册 | @register_model 装饰器自动注册 |
| loader.py 适配 config 加载差异 | PreTrainedConfig.from_pretrained 统一 |
| saver.py ExportWrapper | ModelOutput 返回 logits，不再需要 wrapper |
| saver.py 过滤 assert 节点 | 仍需保留 (torch.export 内部行为) |
| converter cann_lm_converter.py | AutoModelForCausalLM.from_pretrained 一行代码 |
