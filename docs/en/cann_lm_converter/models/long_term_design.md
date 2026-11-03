# Long-Term Design: Unified Infrastructure Scheme

English | [简体中文](../../../zh/cann_lm_converter/models/long_term_design.md)

> Reference the transformers library, gradually introduce base classes such as PreTrainedConfig / ModelOutput / AutoModel
>
> Goal: make cann_local_lm.models loadable and inferable through a unified interface like transformers
>
> Status: Pending review | Last reviewed: 2026-07-14

## Background

The short-term scheme uses adapters to bypass model code differences, but has limitations:
- Each new model requires manual registration
- forward kwargs are filled in manually
- No unified Cache / generate / tokenizer interface
- dict return causes torch.export compatibility issues

The long-term scheme unifies the interface at the model code level by introducing base classes.

## Design Principles

- **Base class inheritance** — All models inherit a unified PreTrainedConfig / PreTrainedModel
- **Composition first** — GenerationMixin / Cache as Mixins, non-invasive to model code
- **Gradual migration** — Base classes are downward compatible, old models migrate gradually
- **Native torch.export support** — ModelOutput replaces dict, ExportWrapper no longer needed

## Directory Structure

```
cann_local_lm/
├── models/
│   ├── __init__.py                  # Export AutoModel / AutoConfig
│   ├── base/                        # New: base class definitions
│   │   ├── __init__.py
│   │   ├── configuration.py         # PreTrainedConfig
│   │   ├── modeling.py              # PreTrainedModel + ModelOutput
│   │   ├── cache.py                 # Cache / DynamicCache / StaticCache
│   │   ├── generation.py            # GenerationConfig + GenerationMixin
│   │   └── registry.py              # AutoModel / AutoConfig
│   ├── gemma4/                      # Migrated: inherit base classes
│   └── qwen3_5/                     # Migrated: inherit base classes
```

## P0: PreTrainedConfig Base Class

```python
# cann_local_lm/models/base/configuration.py

@dataclass
class PreTrainedConfig:
    """Base class for all model configs"""
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
        """Unified loading entry — read config.json → construct config"""
        with open(f"{path}/config.json") as f:
            config_dict = json.load(f)
        # Handle nested config (e.g. text_config)
        if hasattr(cls, "_config_key") and cls._config_key in config_dict:
            config_dict = config_dict[cls._config_key]
        config_dict.update(kwargs)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PreTrainedConfig":
        """Construct from dict — filter unknown fields"""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        config = cls(**filtered)
        if hasattr(config, "__post_init__"):
            config.__post_init__()
        return config

    def to_dict(self) -> dict:
        """Serialize to a plain dict"""
        return dataclasses.asdict(self)

    def save_pretrained(self, path: str) -> None:
        """Save config.json"""
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, path: str) -> "PreTrainedConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
```

**Migration requirements:**

| Model | What to do | Problem solved |
|------|---------|-----------|
| gemma4 | Add from_pretrained / from_json_file | Currently no unified loading entry |
| qwen3_5 | Add to_dict | Currently missing |

## P0: ModelOutput Unified Output

```python
# cann_local_lm/models/base/modeling.py

@dataclass
class ModelOutput:
    """Standardized model output — torch.export friendly"""
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
    """CausalLM-specific output"""
    pass
```

**Problems solved:**
- ExportWrapper no longer needed (forward directly returns CausalLMOutput)
- torch.export's output_specs has only one as_tensor (logits), the rest are as_none
- But as_none needs to be filtered at the ep_json layer (already implemented)

## P0: AutoModel / AutoConfig

```python
# cann_local_lm/models/base/registry.py

_REGISTRY: dict[str, type] = {}

def register_model(model_type: str):
    """Decorator — register model into the AutoModel system"""
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

**Usage:**
```python
# Register (in model __init__.py)
@register_model("qwen3_5")
class Qwen3_5ForCausalLM(PreTrainedModel):
    ...

# Load (one line)
model = AutoModelForCausalLM.from_pretrained("/path/to/qwen3.5-0.8B")
config = AutoConfig.from_pretrained("/path/to/qwen3.5-0.8B")
```

## P1: Cache Base Class

```python
# cann_local_lm/models/base/cache.py

class Cache:
    """KV Cache base class"""
    def update(self, key: Tensor, value: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_seq_len(self, layer_idx: int) -> int:
        raise NotImplementedError

    def to_legacy(self) -> list:
        """Convert to legacy list[(k,v)] — downward compatible"""
        raise NotImplementedError


class DynamicCache(Cache):
    """Dynamically growing cache — inference default"""
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
    """Pre-allocated fixed size — compilation optimization"""
    def __init__(self, max_batch_size, max_seq_len, num_layers, num_kv_heads, head_dim, dtype, device):
        self.key_cache = [torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, ...) for _ in range(num_layers)]
        self.value_cache = [...]
        self.seq_lens = [0] * num_layers
```

**Migration requirements:**
- gemma4: list → DynamicCache (wrapper layer conversion)
- qwen3_5: SimpleKVCache → DynamicCache, SimpleDeltaNetCache retained as subclass

## P1: GenerationConfig + GenerationMixin

```python
# cann_local_lm/models/base/generation.py

@dataclass
class GenerationConfig:
    """Generation params — independent of the model"""
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
    """generate() generic implementation — not hardcoded in the model"""

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Implemented by subclass — prepare input for each generation step"""
        raise NotImplementedError

    def generate(self, input_ids: Tensor, generation_config: GenerationConfig | None = None,
                 stopping_criteria=None, **kwargs) -> Tensor:
        if generation_config is None:
            generation_config = GenerationConfig()
        # Generic generate implementation: greedy / sample / beam
        ...
```

**Migration requirements:**
- Each model's `generate()` moves to `GenerationMixin`
- Models only need to implement `prepare_inputs_for_generation`
- `generation_config.json` is auto-loaded from the HF directory

## P2: Tokenizer Integration

```python
# cann_local_lm/models/base/tokenization.py

class PreTrainedTokenizer:
    """Unified tokenizer interface"""
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def batch_encode(self, texts: list[str], padding=True) -> dict: ...
    def save_pretrained(self, path: str) -> None: ...
    @classmethod
    def from_pretrained(cls, path: str) -> "PreTrainedTokenizer": ...
```

The current scheme only copies tokenizer files; later we can integrate the `tokenizers` library to implement encode/decode.

## P3: Multimodal Processor

```python
class ProcessorMixin:
    """Compose tokenizer + image_processor + audio_processor"""
    def __call__(self, text=None, images=None, audio=None) -> dict: ...
    @classmethod
    def from_pretrained(cls, path: str) -> "ProcessorMixin": ...
```

gemma4 / qwen3_5 have vision_config but currently don't need a processor; reserved for later.

## Migration Roadmap

### Phase 1: P0 Base Classes (minimal viable)
- [ ] Implement `PreTrainedConfig` (from_pretrained / to_dict / save_pretrained)
- [ ] Implement `ModelOutput` / `CausalLMOutput`
- [ ] Implement `AutoConfig` / `AutoModelForCausalLM` + `@register_model` decorator
- [ ] Migrate gemma4 to base classes (add from_pretrained)
- [ ] Migrate qwen3_5 to base classes (add to_dict)
- [ ] Simplify converter layer: remove loader.py adaptation logic, use AutoModel directly

### Phase 2: P1 Inference Interface
- [ ] Implement `Cache` / `DynamicCache` / `StaticCache`
- [ ] Implement `GenerationConfig` / `GenerationMixin`
- [ ] Migrate each model's generate() to the Mixin
- [ ] Migrate each model's cache to DynamicCache

### Phase 3: P2+ Ecosystem
- [ ] Tokenizer integration (encode/decode/chat_template)
- [ ] Processor (multimodal)
- [ ] Attention implementation switching (eager / sdpa / flash)
- [ ] Weight management optimization (sharded loading / tied weights auto-handling)

## Relationship with the Short-Term Scheme

| Short-term scheme | Long-term scheme replacement |
|---------|-------------|
| model_registry.py manual registration | @register_model decorator auto-registration |
| loader.py adapts config loading differences | PreTrainedConfig.from_pretrained unified |
| saver.py ExportWrapper | ModelOutput returns logits, wrapper no longer needed |
| saver.py filters assert nodes | Still needed (torch.export internal behavior) |
| converter cann_lm_converter.py | AutoModelForCausalLM.from_pretrained one-liner |
