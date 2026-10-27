# Short-Term Design: Converter Layer Adapter Scheme

English | [简体中文](../../../zh/cann_lm_converter/models/short_term_design.md)

> Do not modify model code; smooth out gemma4/qwen3_5 interface differences via registry + adapter
>
> Status: Pending review | Last reviewed: 2026-07-14

## Background

The two models (gemma4, qwen3_5) share the same skeleton but have inconsistent interfaces:

- Config loading differs (gemma4 has no from_pretrained, qwen3_5 has no to_dict)
- forward signatures differ (position_ids present/absent, delta_cache present/absent)
- forward returns a dict but with different keys
- load_weights signatures differ (different default dtype, different export location)

## Design Principles

- **Don't modify model code** — All adaptation done in the converter layer
- **Registry-driven** — Each model registers once and auto-connects to the pipeline
- **torch.export first** — Wrapper uniformly returns logits, filters assert nodes

## Directory Structure

```
cann_lm/
├── models/                          # Not modified
│   ├── gemma4/
│   └── qwen3_5/
├── converter/
│   ├── model_registry.py            # Model registry (difference records)
│   ├── loader.py                    # Load adaptation (config + weights + tokenizer)
│   ├── saver.py                     # Save adaptation (export + epair + filter)
│   ├── cann_lm_converter.py         # CLI entry
│   └── export/
│       └── epair_utils.py           # Weight packing (already exists)
```

## model_registry.py — Difference Records

Each model's interface differences are registered via `ModelEntry`:

```python
@dataclass
class ModelEntry:
    config_cls: type                 # Gemma4TextConfig / Qwen3_5TextConfig / ...
    model_cls: type                  # Gemma4ForCausalLM / Qwen3_5ForCausalLM / ...
    load_weights_fn: callable        # load_gemma4_weights / load_qwen3_5_weights / ...
    config_json_key: str             # "text_config" or None (flat config)
    forward_kwargs: dict             # None kwargs during torch.export
    needs_wrapper: bool = True        # Whether ExportWrapper is needed (dict → logits)

REGISTRY = {
    "gemma4": ModelEntry(
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",     # Nested config
        forward_kwargs={
            "attention_mask": None,
            "past_key_values": None,
            "inputs_embeds": None,
            "labels": None,
            "shared_kv_states": None,      # gemma4-specific
            "output_hidden_states": False,
        },
    ),
    "qwen3_5": ModelEntry(
        config_cls=Qwen3_5TextConfig,
        model_cls=Qwen3_5ForCausalLM,
        load_weights_fn=load_qwen3_5_weights,
        config_json_key="text_config",
        forward_kwargs={
            "attention_mask": None,
            "past_key_values": None,
            "delta_cache": None,           # qwen3_5-specific
            "inputs_embeds": None,
            "labels": None,
            "use_cache": False,            # qwen3_5-specific
        },
    ),
}
```

## loader.py — Load Adaptation

```python
def load_model(hf_path, model_name, *, dtype=torch.float16, device="cpu"):
    entry = get_entry(model_name)

    # 1. Config loading — adapt to from_pretrained presence/absence
    if hasattr(entry.config_cls, "from_pretrained"):
        config = entry.config_cls.from_pretrained(hf_path)
    elif hasattr(entry.config_cls, "from_json_file"):
        config = entry.config_cls.from_json_file(f"{hf_path}/config.json")
    else:
        # gemma4: manual read + field filtering
        with open(f"{hf_path}/config.json") as f:
            hf_cfg = json.load(f)
        raw = hf_cfg.get(entry.config_json_key, hf_cfg) if entry.config_json_key else hf_cfg
        valid_fields = {f.name for f in dataclasses.fields(entry.config_cls)}
        config = entry.config_cls(**{k: v for k, v in raw.items() if k in valid_fields})

    # 2. Instantiate + load weights
    model = entry.model_cls(config).to(dtype=dtype).to(device).eval()
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 3. Collect tokenizer files
    tokenizer_files = collect_tokenizer_files(hf_path)

    return ModelArtifact(model, config, hf_path, ..., model_type=entry.model_type)
```

## saver.py — Save Adaptation

```python
class ExportWrapper(nn.Module):
    """Unify forward return — only take logits"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        out = self.model(input_ids=input_ids)
        return out["logits"] if isinstance(out, dict) else out


def save_epair(artifact, output_dir, *, max_file_size=512*1024*1024):
    # 1. torch.export (wrapper uniformly returns logits)
    wrapper = ExportWrapper(artifact.model).eval()
    sample_inputs = (torch.randint(0, artifact.config.vocab_size, (1, 8), dtype=torch.long),)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 2. export_to_epair
    epair = torch_delegate.export_to_epair(ep)

    # 3. Filter _assert_tensor_metadata nodes (run_decompositions re-inserts them)
    nodes = epair.ep_json["graph_module"]["graph"]["nodes"]
    epair.ep_json["graph_module"]["graph"]["nodes"] = [
        n for n in nodes if "assert" not in n.get("target", "").lower()
    ]

    # 4. Save weights + ep_json
    consolidated_weights_save(epair, output_dir, max_file_size=max_file_size)

    # 5. Copy tokenizer + save model_config
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)
```

## Handled Differences Checklist

| Difference | Adapter scheme | Affected files |
|------|-------------|---------|
| gemma4 has no from_pretrained | loader checks hasattr, falls back to manual construction | loader.py |
| qwen3_5 has no to_dict | saver uses dataclasses.asdict fallback | saver.py |
| forward returns dict | ExportWrapper returns only logits | saver.py |
| forward kwargs differ | ModelEntry.forward_kwargs records | model_registry.py |
| config_json_key differs | ModelEntry records, loader adapts | model_registry.py |
| load_weights default dtype differs | Caller uniformly passes dtype=torch.float16 | loader.py |
| _assert_tensor_metadata | Filter target containing "assert" at ep_json layer | saver.py |
| load_gemma4_weights not exported | model_registry directly imports the modeling module | model_registry.py |

## Limitations

- Each new model requires a manual registry entry
- forward kwargs must be filled in manually (cannot be auto-derived)
- ExportWrapper assumes the returned dict contains a "logits" key
- No unified Cache interface (list / SimpleKVCache / SimpleDeltaNetCache are incompatible)
- No unified generate (each model implements independently, inference behavior inconsistent)
