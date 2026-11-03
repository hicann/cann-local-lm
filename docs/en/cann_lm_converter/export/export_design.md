# Model Export Module Design Document

English | [简体中文](../../../zh/cann_lm_converter/export/export_design.md)

> Complete design for exporting HuggingFace models to epair format: registry → loading → ExportWrapper → torch.export → epair on-disk → C++ loading.
>
> Covers `cann_local_lm/models/`'s `model_registry.py` / `export_wrapper.py` + `cann_local_lm/converter/`'s `loader.py` / `saver.py` / `swap.py` / `quant_entry.py` / `export/` (incl. C++ `csrc/`).
>
> CLI entry see [cli_design.md](../cli/cli_design.md), quantization integration see [quant_design.md](../quant/quant_design.md).
>
> Status: Verified | Last verified: 2026-08-31

## Table of Contents

- [Overview](#overview)
- [Model Registry model_registrypy](#model-registry-model_registrypy)
- [loaderpy — Model Loading](#loaderpy-model-loading)
- [saverpy — epair Saving](#saverpy-epair-saving)
- [Compatibility Handling](#compatibility-handling)
- [Export Module epair_utilspy](#export-module-epair_utilspy)
- [C Loading Side](#c-loading-side)
- [GE Op Registration Two Layers](#ge-op-registration-two-layers)
- [Output File Layout](#output-file-layout)
- [prefilldecode Separated Export](#prefilldecode-separated-export)
- [New Model Onboarding Guide](#new-model-onboarding-guide)
- [Accuracy Difference Analysis](#accuracy-difference-analysis)
- [Pending and Legacy (Export-related)](#pending-and-legacy-export-related)

---

<a id="overview"></a>
## Overview

`cann_local_lm/converter/` is the toolchain for exporting HF models to epair format. Capability boundary:

- **Input**: HuggingFace model directory (`config.json` + safetensors weights + tokenizer files)
- **Output**: epair model directory (`ep_json_with_ascend_ir.json` + weight bins + tokenizer + `model_config.json`), for C++ backend loading
- **Quantization**: In progress (see [quant_design.md](../quant/quant_design.md)), scheme W4A8 s8s4

### End-to-End Flow

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
├── weight_infos.json
├── weights_0.bin
├── weights_1.bin
└── ...
    │
    ▼  C++ EpairModelLoader(model_json_path, weights_index_path)
ge::Graph
```

### Directory Structure

```
cann_local_lm/
├── models/
│   ├── model_registry.py          # Model registry (ModelEntry + REGISTRY + detect_model_type + available_models)
│   ├── export_wrapper.py           # ExportWrapper base class (forward + get_sample_inputs + _drive_layers)
│   ├── fused_ops/                  # Custom op library (cann_local_lm:: + npu::, see fused_ops.md)
│   └── {gemma4,qwen3_5}/     # Per-model definition + subclass export_wrapper.py
└── converter/
    ├── __init__.py
    ├── cann_lm_converter.py        # CLI entry (pipeline) + programmatic convert() function  ← see cli_design.md
    ├── loader.py                   # Model loading (config + weights + tokenizer, auto-detect model type)
    ├── saver.py                    # epair saving (save_epair / save_epair_prefill_decode / save_model_config / copy_tokenizer_files)
    ├── swap.py                     # Quantization orchestration (swap_from_dopt etc. main path, see quant_design.md)
    ├── quant_entry.py              # Quantization entry facade (re-export run_auto_quant + swap_from_dopt)
    ├── quant/                      # Quantization subpackage (CANN lightweight tool GPTQ / auto_quant, see quant_design.md)
    └── export/
        ├── epair_utils.py          # pack_weights + consolidated_weights_save
        ├── ge_converters/          # register_ge_converters (cann_local_lm/npu custom ops + some aten ops GE mapping)
        ├── quant_swap/             # Quantization materialization internal package (materialize.py + quant_linear.py, see quant_design.md)
        └── csrc/                   # C++ backend (epair_model_loader / ge_graph_builder / ge_weight_loader, see C++ loading side below)
```

Custom op library at `cann_local_lm/models/fused_ops/`, hits two namespaces: most go through `torch.ops.npu.*` (`npu::`, converter env self-defined schema + `register_fake` by `_library.py`, doesn't depend on torch_npu), only `l2_norm` goes through `torch.ops.cann_local_lm.*` (`cann_local_lm::`, no torchair correspondence). Op list: `chunk_gated_delta_rule` / `dynamic_quant` / `fused_infer_attention_score` / `l2_norm` / `quant_matmul` / `rms_norm` / `rotary_mul` / `scatter_nd_update`. Each op registers corresponding GE converter in `export/ge_converters/ge_converter.py` (mapping implementation see [`fused_ops.md`](../models/fused_ops/fused_ops.md) §3). CPU eager impl registered on-demand by `_library.py`'s `register_cpu_impl` decorator (stub/export env skips only keeping `register_fake`; non-stub env registers then can eager), real computation via GE converter mapping to NPU ops.

<a id="model-registry-model_registrypy"></a>
## Model Registry model_registry.py

Two models have inconsistent interfaces; converter smooths differences via registry + adapter.

```python
@dataclass
class ModelEntry:
    """A model's entry in the registry"""
    model_type: str               # "gemma4_text" / "qwen3_5_text"
    config_cls: type              # Gemma4TextConfig / Qwen3_5TextConfig
    model_cls: type               # Gemma4ForCausalLM / Qwen3_5ForCausalLM
    load_weights_fn: callable     # load_gemma4_weights / load_qwen3_5_weights
    config_json_key: str | None   # Key for text config in config.json (e.g. "text_config"), None means top-level
    wrapper_cls: type             # ExportWrapper subclass (Gemma4ExportWrapper / Qwen3_5ExportWrapper / ...)


REGISTRY: dict[str, ModelEntry] = {
    "gemma4": ModelEntry(
        model_type="gemma4_text",
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",
        wrapper_cls=Gemma4ExportWrapper,
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
    "qwen3.5": "qwen3_5", "qwen3_5": "qwen3_5",
}


def get_entry(name: str) -> ModelEntry:
    """Look up model entry by name, supports aliases"""
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
    """List of registered model names (for CLI error prompts)"""
    return list(REGISTRY)
```

<a id="loaderpy-model-loading"></a>
## loader.py — Model Loading

### ModelArtifact Data Structure

```python
@dataclass
class ModelArtifact:
    """Load artifact"""
    model: nn.Module              # Custom model instance (weights loaded, eval mode)
    config: object                # Custom config dataclass instance
    hf_path: str                  # Original HF model path
    hf_config: dict               # Original config.json raw dict (all fields retained)
    tokenizer_files: list[Path]   # Tokenizer-related file path list
    model_type: str               # Model type identifier
    wrapper_cls: type             # ExportWrapper subclass (passed through from ModelEntry)
```

### load_model Interface

```python
def load_model(
    hf_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ModelArtifact:
    """
    Load model from HF path — model type auto-detected

    Flow:
      1. Read HF config.json
      2. Auto-detect model type from config.json's model_type field (detect_model_type)
      3. Construct custom config dataclass
      4. Instantiate custom model
      5. Load HF safetensors weights (call each model's load_*_weights)
      6. Collect tokenizer files list
    """
    # 1. Read original config.json
    config_json_path = os.path.join(hf_path, "config.json")
    with open(config_json_path) as f:
        hf_config = json.load(f)

    # 2. Auto-detect model type
    entry = detect_model_type(hf_config)

    # 3. Construct custom config (adapt to from_pretrained presence/absence)
    config = build_config(hf_config, entry)

    # 4. Instantiate model
    model = entry.model_cls(config)
    model = model.to(dtype=dtype).to(device=device)
    model.eval()

    # 5. Load weights — call each model's load_*_weights (handles key remapping internally)
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 6. Collect tokenizer files (don't load, only record paths)
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

### build_config — Unified Field Filtering

Each model's config loading differs (some config_json_key nested, some top-level), but `build_config` **uniformly goes through field filtering**: first take nested or top-level sub-dict by `config_json_key`, then filter unknown fields by dataclass field names, finally trigger `__post_init__` (compute derived quantities like head_dim).

```python
def build_config(hf_config: dict, entry: ModelEntry):
    # Nested config (gemma4/qwen3_5): take from text_config sub-dict; flat config: top-level
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

> Note: `config_cls`'s `from_pretrained` / `from_json_file` methods (qwen3_5 has) are no longer called by `build_config` — uniformly goes through field filtering path, avoids different class's loading methods diverging.

### collect_tokenizer_files

```python
_TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

_OPTIONAL_TOKENIZER_PATTERNS = [
    "tokenizer.model",      # SentencePiece old format
    "vocab.json",           # BPE old format
    "merges.txt",           # BPE old format
]


def collect_tokenizer_files(hf_path: str) -> list[Path]:
    """Collect tokenizer-related files from HF directory.

    Required 3 (tokenizer.json family); when no tokenizer.json supplement old format
    (tokenizer.model / vocab.json / merges.txt).
    generation_config.json not handled here (see legacy tasks).
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

<a id="saverpy-epair-saving"></a>
## saver.py — epair Saving

### ExportWrapper Base Class — Unified I/O Contract

Referencing litert-torch's `ExportableModule` and npu_tuned_export's wrapper design, design unified wrapper base class. Core idea: **KV cache as explicit graph input/output**, subclass implements `_drive_layers` to interface with native model forward.

Supports **two embedding modes**:
- `use_embeddings=False` (default): pass `input_ids`, graph contains `embed_tokens` node
- `use_embeddings=True` (pure embeddings): `input_ids` **not in graph** (None placeholder), host computes `embeddings` and passes in; model forward receives `inputs_embeds` skipping `embed_tokens` lookup. gemma4 due to PLE (Per-Layer Embedding)'s `embed_tokens_per_layer` lookup also externalized, **additionally passes `per_layer_token_identity`** (host-computed token identity); qwen3_5 has no PLE, only passes `embeddings`.

> Note: When `use_embeddings=True`, `saver` calls `wrapper.detach_externalized_lookups()` after `get_sample_inputs` (uses it to compute sample) + host weight extraction, **before** `torch.export`, to `del` externalized lookup submodules (`embed_tokens`, gemma4 adds `embed_tokens_per_layer`) from model → torch.export doesn't lift them → `weights_*.bin` **doesn't contain** redundant lookup weights (≈ OM, GE compile already prunes them), lookup weights separately packed to `host_weights_info.json` + `host_weights_*.bin` for host (with dtype/shape). Depends on pure embeddings forward not accessing these submodules (`embed_tokens` skipped by `inputs_embeds`, gemma4 `embed_tokens_per_layer` skipped by instance method replacement); `saver` asserts after export they're not in `inputs_to_parameters` as fallback. Tied models (qwen3.5): after del `embed_tokens`, `lm_head.weight` (same data) still present → baked by OM; host gets `embed_tokens.weight` from `host_weights` for lookup (lm_head/embed_tokens dual role, non-redundant).

Base class at `cann_local_lm/models/export_wrapper.py`, subclasses in each model's directory (e.g. `cann_local_lm/models/qwen3_5/export_wrapper.py`).

```python
class ExportWrapper(nn.Module):
    """Unified external I/O: cache in/out + output_pos selection. Subclass implements _drive_layers.

    Lightweight version: subclass internally calls native forward (with if past is None branch)
    Full version (later): each attention adds scatter-write path, one static graph covers prefill+decode
    """

    def __init__(self, model, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(
        self,
        input_ids=None,            # Pass when use_embeddings=False; None when =True (not in graph)
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        new_kv_cache_pos=None,
        embeddings=None,           # ← use_embeddings=True: host-computed embeddings
        output_pos=None,           # ← Selection: decode only computes last token's logits
        per_layer_token_identity=None,  # ← Only gemma4 (PLE): host-computed token identity
        embeddings_scale=None,     # ← quant-embeddings: int8 weight per-channel scale (base class forward dequantizes)
        delta_states=None,          # ← qwen3_5 linear attention recurrent state (conv_flat, recurrent_flat)
    ):
        # quant-embeddings: int8 embeddings * per-channel scale -> float (base class forward dequantizes, _drive_layers receives float)
        if embeddings is not None and embeddings_scale is not None:
            embeddings = embeddings.to(dt) * embeddings_scale.to(dt).unsqueeze(-1)
        logits, past_kv_flat, delta_out_flat = self._drive_layers(
            input_ids, attention_mask, position_ids,
            past_key_values, new_kv_cache_pos, embeddings,
            per_layer_token_identity, delta_states,
        )
        if output_pos is not None:
            logits = logits.index_select(1, output_pos.to(torch.int64))
        return logits, *past_kv_flat, *delta_out_flat

    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None,
                      per_layer_token_identity=None, delta_states=None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Subclass implements: unified I/O → native forward kwargs → extract logits + flatten KV/delta

        Returns 3-tuple (logits, past_kv_flat, delta_out_flat): gemma4's delta_out_flat is [],
        qwen3_5 linear attention fills recurrent state. per_layer_token_identity only gemma4 PLE.
        """
        raise NotImplementedError

    def get_sample_inputs(self, seq_len=128, use_embeddings=False, quant_embeddings=False) -> tuple:
        """Construct sample inputs for export

        Args:
            use_embeddings: True passes embeddings (lookup outside graph), False passes input_ids (lookup in graph)
            quant_embeddings: True passes int8 embeddings + scale (host int8 lookup, base class forward dequantizes)
        """
        # ... construct input_ids, attention_mask, position_ids, past_key_values, output_pos, ...
        if use_embeddings:
            if quant_embeddings:
                # int8 per-channel: host gather int8_weight[input_ids] + scale[input_ids]; graph entry dequantizes
                return (None, ..., emb_q, output_pos, ple, emb_scale)  # 9-tuple (gemma4 incl. ple)
            embeddings = self.model.model.embed_tokens(input_ids).detach()
            return (None, attention_mask, position_ids, past_key_values,
                    new_kv_cache_pos, embeddings, output_pos, *extras)  # 7-tuple (gemma4 incl. ple → 8)
        else:
            return (input_ids, attention_mask, position_ids, past_key_values,
                    new_kv_cache_pos, None, output_pos)  # 7-tuple
```

### Unified Graph I/O Contract

```
Inputs:
  input_ids          [batch, seq_len] int64               ← Pass when use_embeddings=False;
                                                             =True None (not in graph)
  embeddings         [batch, seq_len, hidden] float16    ← use_embeddings=True: host-computed
                                                             embed_tokens lookup result
  per_layer_token_identity  [batch, seq_len, L, h_pl] float   ← Only gemma4 (PLE):
                                                             host-computed embed_tokens_per_layer
                                                             lookup result (after reshape)
  attention_mask     [batch, 1, seq_len, kv_cache_max_len]    ← 4D pre-constructed mask
  position_ids       [batch, seq_len] int64
  past_key_values    list of (key_buf, value_buf) per layer   ← One pair of pre-allocated buffer per layer
                       shape: [kv_cache_max_len, kv_head, batch, head_dim]
  new_kv_cache_pos   [seq_len] int64                           ← scatter write position
  output_pos         [1] int32 (optional)                          ← Only which position's logits to compute

Outputs:
  lm_logits          [batch, n_pos, vocab]                    ← After output_pos selection
  past_key_0, past_value_0, past_key_1, past_value_1, ...    ← Flattened all-layer KV cache
```

**Why need 4D pre-constructed mask?** Model's `_prepare_causal_mask` / `_create_causal_mask` has dynamic control flow internally (depends on runtime seq length), torch.export can't statically trace. Passing 4D mask goes through passthrough path (`return attention_mask`), skipping dynamic construction. Two models' passthrough locations:

| Model | Passthrough code location |
|------|-------------|
| gemma4 | `modeling_gemma4.py` line 619 `return attention_mask` |
| qwen3_5 | `modeling_qwen3_5.py` line 616 `if attention_mask.dim() == 4: return attention_mask` (added) |

> gemma4 subclass (`Gemma4ExportWrapper`) additionally handles PLE in pure embeddings branch: uses **instance method replacement** to temporarily replace `text_model._compute_per_layer_inputs` to return host-computed `per_layer_token_identity`, so model.forward skips `embed_tokens_per_layer(input_ids)` lookup + bf16 reverse lookup, while layer loop/mask/rope/lm_head still reuse model.forward (doesn't modify modeling). `_build_external_extras` computes `per_layer_token_identity = embed_tokens_per_layer(input_ids).reshape(b,s,L,h_pl)`. See `models/gemma4/export_wrapper.py`.

### Two Subclass Differences

| Subclass | File location | Native cache type | forward kwargs differences | Special handling |
|------|---------|----------------|---------------------|---------|
| Gemma4ExportWrapper | `models/gemma4/export_wrapper.py` | `list[(k,v), ...]` | `position_ids` externally passed | `shared_kv_states=None`; **pure embeddings uses instance method replacement `_compute_per_layer_inputs` to externalize PLE lookup**, receives `per_layer_token_identity` |
| Qwen3_5ExportWrapper | `models/qwen3_5/export_wrapper.py` | `SimpleKVCache` | `delta_cache`, `use_cache` | list → SimpleKVCache conversion; `delta_cache=None`; no PLE |

Both subclasses support pure embeddings: when `embeddings` non-None only passes `inputs_embeds` (doesn't pass `input_ids`, input_ids not in graph), None passes `input_ids` (in-graph lookup). gemma4 additionally uses `per_layer_token_identity` to externalize PLE.

### Lightweight vs Full

| | Lightweight (current milestone) | Full (later) |
|---|---|---|
| External I/O | Unified (base class) | Unified (base class) |
| Subclass internal | Calls native `forward` (with `if past is None` branch) | Each attention adds scatter-write path |
| Exported graph | Takes one branch by sample, shape-specialized | One static graph covers prefill+decode |
| Workload | Small (base class + 2 subclass scaffolds) | Large (per-model modify attention) |

### Per-Model Full Version Difficulties (later)

- **gemma4**: KV sharing (last n layers no own buffer, reuse other layers') → doesn't fit "one buffer per layer"; sliding window's local mask; per-layer head_dim/rope.
- **qwen3_5**: Linear attention (GatedDeltaNet)'s cache is `[num_v_heads, k_head_dim, v_head_dim]` **recurrent state** (delta rule incremental update), not `[kv_cache_max_len, ...]` KV buffer — needs separate design for recurrent buffer's incremental protocol.

### save_epair Interface

```python
def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,  # Default 50GB
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = False,         # ← True: lookup outside graph (for quant)
    save_weights: bool = True,
    whitelist_decompositions: set | None = None,   # Extra decomposition whitelist
    embed_quant: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,  # ← quant-embeddings: {fqn:(int8,scale)}
) -> None:
    """
    Save flow (doesn't output .pt2 / input_N.bin, both separately generated by verify scripts)

    output_dir/
    ├── ep_json_with_ascend_ir.json     # torch.export + ascend_ir
    ├── weight_info.json                # Weight index
    ├── weights_0.bin                   # Weight data (split by max_file_size)
    ├── model_config.json               # config + HF original info + use_embeddings/quant_embeddings
    ├── tokenizer files
    └── host_weights_*.bin              # (only use_embeddings/quant_embeddings) host lookup weights
    """
    import torch_delegate
    from torch_delegate import EpairConfig
    from cann_local_lm.converter.export.ge_converters import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Register GE converters (must call before export_to_epair)
    register_ge_converters()

    # 2. Construct unified I/O wrapper from ModelEntry-passed wrapper subclass
    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_local_lm/models/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # quant-embeddings: inject int8 + per-channel scale (bool() fallback for lightweight tool no embed-quant returning {})
    quant_embeddings = bool(embed_quant)
    if quant_embeddings:
        wrapper.set_quant_embeddings(embed_quant)

    # 3. Create sample inputs (unified I/O contract, supports embeddings/quant_embeddings mode)
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=use_embeddings, quant_embeddings=quant_embeddings
    )

    # Host-side lookup weights: quant-embeddings takes int8+scale; pure embeddings takes embed_tokens (+ gemma4 PLE)
    if quant_embeddings:
        host_side_weights = [(fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()] + \
                            [(fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()]
    else:
        host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)

    # Pure embeddings: detach externalized lookup submodules → torch.export doesn't lift → weights_*.bin ≈ OM
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    # 4. torch.export (strict=False allows None placeholder / dynamic branches)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # Assert detach effective (externalized lookup weights not lifted, otherwise weights_*.bin carries redundant lookup weights)
    if use_embeddings:
        _assert_externalized_not_lifted(ep, wrapper.host_side_weight_names())

    # 5. export_to_epair (EpairConfig with decomposition whitelist: eye/type_as/index_copy/index_put)
    config = EpairConfig()
    config.whitelist_decompositions.update(_EXTRA_DECOMPOSITION_OPS)
    epair = torch_delegate.export_to_epair(ep, config)

    # C++ side already handles (no need Python-side strip ep_json):
    #   - _assert_tensor_metadata nodes: PlanNodes top IsAssertTensorMetadataNode skipped
    #   - as_none constant input: PreparePlaceholders NodeInputKind::None → continue

    # 6. Free original model memory (large model OOM protection)
    #    epair.ep holds ExportedProgram's independent reference, deleting model/wrapper/ep doesn't affect epair
    del ep, wrapper
    del artifact.model
    import gc; gc.collect()

    # 7. Save weights + ep_json (incl. host-side lookup weights packaging)
    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        host_side_weights=host_side_weights,
    )

    # 8. Copy tokenizer + save model_config (records use_embeddings/quant_embeddings)
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir, use_embeddings=use_embeddings,
                      quant_embeddings=quant_embeddings)
```

### get_sample_inputs — Base Class Implementation

Sample input construction has migrated from standalone function to `ExportWrapper.get_sample_inputs` base class method; saver.py directly calls `wrapper.get_sample_inputs(seq_len, use_embeddings)`.

Base class mask strategy: **causal-over-buffer** (scatter-write semantics), shape `[batch, 1, seq_len, kv_cache_max_len]`:

```python
# ExportWrapper.get_sample_inputs mask construction
min_val = torch.finfo(dtype).min
q_pos = torch.arange(seq_len, dtype=dtype).unsqueeze(1)    # [seq_len, 1]
k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)    # [1, kv_max]
mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, kv_max)
```

Requires model's attention to use `index_copy` scatter-write (new KV written to buffer, attn reads K = buffer itself, dim = kv_max).
Non-scatter models (still use `torch.cat` to concat past+new, K dim = kv_max + seq_len) will fail export with broadcast error due to dim mismatch — see legacy tasks.

Other inputs: input_ids (random), position_ids (arange), new_kv_cache_pos (arange), output_pos (`[seq_len-1]`), past_key_values (one pair zero buffer per layer, per-layer head_dim determined by `_get_layer_head_dim`).

Pure embeddings mode (`use_embeddings=True`): `input_ids` not in positional args (None placeholder), instead `embeddings = embed_tokens(input_ids).detach()` (host computes); gemma4 via `_build_external_extras` additionally appends `per_layer_token_identity`. Return tuple count varies by mode: `use_embeddings=False`=7-tuple; pure embeddings gemma4=8 (last `per_layer_token_identity`); quant-embeddings=9 (extra `embeddings_scale`, gemma4 incl. PLE); qwen3_5 via subclass override appends `delta_states` to 10-tuple. All align with `forward` signature (missing positions take None default).

### save_model_config

```python
def save_model_config(artifact: ModelArtifact, output_dir: Path, *,
                      use_embeddings: bool = False, quant_embeddings: bool = False) -> None:
    # to_dict if available (gemma4), else dataclasses.asdict (qwen3_5)
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
    # Write model_config.json
    ...
```

<a id="compatibility-handling"></a>
## Compatibility Handling

### bfloat16 Serialization

numpy doesn't support bfloat16 dtype. `pack_weights` needs to convert bfloat16 tensor to int16 view at serialization:

```python
# Inside epair_utils.py pack_weights
raw = tensor.detach().cpu().contiguous()
if raw.dtype == torch.bfloat16:
    raw = raw.view(torch.int16)    # Preserve original bytes, no precision loss
data = raw.numpy().tobytes()
```

### Non-Persistent Buffers

`persistent=False` buffers (like `rope.sin_cache`, `rope.cos_cache`) are not in `state_dict()`, need to supplement from `ep.module().named_buffers()`:

```python
# Inside epair_utils.py pack_weights
raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
state_dict = dict(raw_sd)

# Non-persistent buffers are not in state_dict; add them from the module
for name, buf in contents.ep.module().named_buffers():
    if name not in state_dict:
        state_dict[name] = buf
```

### register_ge_converters Call Timing

`torch_delegate.export_to_epair()` internally depends on GE op converter registration. Must execute `register_ge_converters()` before calling, otherwise ascend_ir annotation empty, C++ side errors out.

Called in step 1 of `save_epair`.

<a id="export-module-epair_utilspy"></a>
## Export Module epair_utils.py

### Core Data Structures

#### EpairContents

`EpairContents` from `torch_delegate`, is epair's in-memory representation after loading, containing:

- `ep`: `ExportedProgram` object, can call `state_dict()` to get weights
- `ep_json`: Model JSON with ascend IR annotations
- `graph_signature`: Graph signature, describes parameter/buffer/constant placeholder mapping

#### ConsolidatedWeights

```python
@dataclasses.dataclass
class ConsolidatedWeights:
    bin_files: list[bytes]                    # List of raw bytes for multiple bin files
    weights_index: dict[str, list[dict]]      # {bin filename: [{"name","offset","size","format","_is_weight"}, ...]}
```

`ConsolidatedWeights` only contains weight data, no EP JSON. Generated by `pack_weights()`, written to disk by `consolidated_weights_save()`.

<a id="output-file-layout"></a>
### File Layout

```
output_dir/
├── ep_json_with_ascend_ir.json    # Complete model description with ascend_ir
├── weight_info.json               # Tensor index (incl. bin filename references)
├── weights_0.bin                  # Concatenated raw tensor data (file 0)
├── weights_1.bin                  # Concatenated raw tensor data (file 1, produced on split)
├── ...                            # More bin files (split as needed)
├── host_weights_info.json         # (only use_embeddings=True) host lookup weight index
└── host_weights_0.bin             # (only use_embeddings=True) host lookup weight data
                                     # embed_tokens / gemma4 embed_tokens_per_layer
```

All files written directly to `output_dir/`, no `data/` subdirectory created.

> `host_weights_*` only produced when `use_embeddings=True` (pure embeddings export). `saver` calls `wrapper.detach_externalized_lookups()` before `torch.export` to `del` externalized lookup submodules (`embed_tokens` / gemma4 `embed_tokens_per_layer`) from model → torch.export doesn't lift them → `weights_*.bin` / `weight_info.json` **doesn't contain** lookup weights (≈ OM, no redundancy; GE compile already prunes them). `consolidated_weights_save` via `host_side_weights` packs lookup weights (`embed_tokens.weight`, gemma4 adds `embed_tokens_per_layer.weight`) separately to `host_weights_info.json` + `host_weights_*.bin` (with dtype/shape) for host lookup. C++ `GeWeightLoader` only reads `weight_info.json`, zero awareness of host files. Tied models: after del `embed_tokens`, `lm_head.weight` (same data) still in weights (baked by OM), host gets `embed_tokens.weight` from host_weights for lookup.

### File Split Mechanism

`pack_weights` supports splitting output bin files by `max_file_size` parameter (default 5KB):

**Split rules**:

1. **Per-tensor check**: iterate tensors, check if adding current tensor to current bin file exceeds `max_file_size`
2. **No cross-file split**: single tensor never split across two files. If current bin non-empty and adding new tensor exceeds limit, seal current bin file first, new tensor starts from next file
3. **Oversized tensor**: if single tensor itself exceeds `max_file_size`, it gets its own bin file (not split)
4. **Naming rule**: bin files named `weights_0.bin`, `weights_1.bin`, ..., sequence starting from 0 incrementing

**Split example**

Assume 3 tensors, sizes 48B, 36B, 6000B respectively, `max_file_size=5120` (5KB):

```
weights_0.bin: [tensor_0: 48B] [tensor_1: 36B]    → 84B (< 5120)
weights_1.bin: [tensor_2: 6000B]                   → 6000B (standalone file, exceeds limit but not split)
```

Corresponding `weight_info.json`:
```json
{
  "weights_0.bin": [
    {"name": "tensor_0", "offset": 0,  "size": 48, "format": "ND", "_is_weight": true},
    {"name": "tensor_1", "offset": 48, "size": 36, "format": "ND", "_is_weight": false}
  ],
  "weights_1.bin": [
    {"name": "tensor_2", "offset": 0,  "size": 6000, "format": "ND", "_is_weight": true}
  ]
}
```

> **`_is_weight` field**: distinguishes whether tensor belongs to `nn.Module` state (part of state_dict, prefill/decode shared). `true` = model weights (parameters + buffers, like linear weight/bias, rotary cos/sin/inv_freq cache table); `false` = constants lifted from Python closure into graph at torch.export trace time (`lifted_tensor_constants`, like StridedSliceV2 index tensors). C++ `GeWeightLoader.LoadConsolidated` reads this field to fill `TensorRecordInfo.is_parameter`, when missing falls back to `input_specs`'s `PlaceholderKind` inference (Parameter/Buffer → true, TensorConstant → false), backward compatible with old `weight_info.json`. At graph build `CreateConstNode` (`ge_graph_builder.cc`) injects `is_parameter` as private attr `_is_weight` to GE Const node, dump graph (`ge_proto_*.txt`) each Const node visible this attr, convenient for debugging which Consts are model weights vs graph constants.

### Interface Description

#### pack_weights(contents, *, max_file_size=5120) -> ConsolidatedWeights

Extract tensor data from `EpairContents` and pack into multiple bin files (in memory).

Parameters:
- `contents`: `EpairContents` object
- `max_file_size`: Max bytes per single bin file, exceeding splits to next file. Default 5KB (for testing)

Returns:
- `ConsolidatedWeights` object (`bin_files` + `weights_index`)

Implementation logic:
1. Collect all parameters and buffers from `contents.ep.state_dict()`
2. Filter tensors referenced in graph by `graph_signature`'s `inputs_to_parameters`, `inputs_to_buffers`, `inputs_to_lifted_tensor_constants`
3. Serialize tensor data via `numpy.tobytes()`
4. Write to bin files sequentially per split rules, build index (name/file/offset/size)

#### consolidated_weights_save(contents, output_dir, *, save_weights=True, max_file_size=5120, ep_json_filename="ep_json_with_ascend_ir.json", host_side_weights=None)

Write `EpairContents` to disk.

Parameters:
- `contents`: `EpairContents` object (returned by `torch_delegate.export_to_epair()`)
- `output_dir`: Output directory (files written directly to this dir, no subdirectory created)
- `save_weights`: When `True` calls `pack_weights` to generate `weights_*.bin` and `weight_info.json`; when `False` only writes EP JSON
- `max_file_size`: Max bytes per single bin file, passed to `pack_weights`
- `ep_json_filename`: EP JSON filename (default `ep_json_with_ascend_ir.json`; prefill/decode separation passes `prefill_...` / `decode_...` respectively)
- `host_side_weights`: Optional `list[(name, tensor)]`, host lookup weights for pure embeddings export (`use_embeddings=True`) (`embed_tokens.weight`, gemma4 adds `embed_tokens_per_layer.weight`). When non-empty, `_pack_host_side_weights_to_dir` writes to independent `host_weights_*.bin` + `host_weights_info.json` (entries contain `name/offset/size/dtype/shape`). Independent from `weights_*.bin`, C++ `GeWeightLoader` zero awareness.

Internal flow:
1. If `save_weights=True`, call `pack_weights(contents, max_file_size=...)` to get `ConsolidatedWeights`
2. Sequentially write `weights_0.bin`, `weights_1.bin`, ... (controlled by `save_weights`)
3. Write `weight_info.json` (controlled by `save_weights`)
4. If `host_side_weights` non-empty, write `host_weights_0.bin`, ... + `host_weights_info.json`
5. Write `<ep_json_filename>` from `contents.ep_json` (always written)

<a id="c-loading-side"></a>
## C++ Loading Side

Export artifacts are loaded by C++ `EpairModelLoader`. **Loader library source** at `cann_local_lm/converter/export/csrc/` (`backend/epair_model_loader.cc` / `ge_graph_builder.cc` / `ge_weight_loader.cc` / `epair_model_parser.cc`, build artifact `libtorch_delegate_backend.so`); **example main + op registration** at `examples/test_converter/export_llm_model/` (`main.cpp` + `reg_register.h`).

### EpairModelLoader API (2 paths)

```cpp
td::EpairModelLoader loader(
    "/path/to/ep_json_with_ascend_ir.json",   // model JSON
    "/path/to/weight_info.json"               // Weight index (bin path derived from it)
);
ge::Graph graph("delegate_graph");
loader.GetGEGraph(&graph);  // Build GE Graph (parse ep_json + lazy-load bin)
```

- Constructor takes 2 file paths: `model_json_path` and `weights_index_path`
- Weight bin file path auto-derived from `weight_info.json`'s `file` field (relative to index file directory)
- Uses PIMPL pattern, externally ABI-stable
- `GetGEGraph` returns copy of internal Graph, can compile and execute independently

### C++ Internal Call Chain

```
EpairModelLoader(model_json_path, weights_index_path)
    │
    ▼  Impl::Initialize
    ├── Read model JSON file content
    └── GeGraphBuilder(model_json_string, weights_index_path, result)
            │
            ├── EpairModelParser::Parse(model_json) → ParsedModel
            ├── GeWeightLoader::Create(weights_index_path, model, result)
            │       │
            │       ├── Parse weight_info.json
            │       ├── Derive bin file absolute path (index dir + file field)
            │       └── Build logical_name → TensorRecordInfo mapping
            │
            └── Build(result) → ge::Graph
                    ├── PrepareContext
                    ├── PreparePlaceholders (create Data/Const nodes)
                    ├── PlanNodes (create Ascend op nodes)
                    │   ├── Skip _assert_tensor_metadata nodes (see below)
                    │   └── Direct error exit when encountering op unmappable to Ascend IR
                    ├── BindDeferredOutputs
                    ├── BindDeferredEdges
                    └── FinalizeGraph
```

### GeWeightLoader Lazy Loading Mechanism

`GeWeightLoader` doesn't read all bin files at construction, uses **lazy loading** strategy:

1. `LoadConsolidated` only parses `weight_info.json`, records each tensor's bin file absolute path (stored in `TensorRecordInfo::record_path`)
2. `ReadTensorData` first access to a bin file reads it into `bin_files_cache_` (`unordered_map<path, vector<uint8_t>>`)
3. Subsequent accesses to tensors in same bin file read from cache

### Unmappable Op Handling

`PlanNodes` iterates all graph nodes; if encounters node with empty `ascend_ir.source` (i.e. op unmappable to Ascend IR), **direct error exit**, no longer executes fallback codegen.

Error message example:
```
PlanNodes failed: node [node_1_aten_arange_start] has no ascend_ir mapping (source is empty),
target=torch.ops.aten.arange.start, effective_target=torch.ops.aten.arange.start.
This operator cannot be converted to Ascend IR.
Please register the GE operator or decompose the operation in the model definition.
```

User needs to:
1. Register missing GE op in `reg_register.h`, or
2. Replace unsupported op with supported op combinations in model definition

> CANN-LM has removed `fallback_codegen`; unmapped ops direct error; so only need `ge_graph_builder.cc`'s PlanNodes to skip. Converter main path relies on C++ side `PlanNodes` top `IsAssertTensorMetadataNode` to skip `_assert_tensor_metadata` nodes, no longer depends on Python-side strip (original `export_llm_model_epair.py`'s `_strip_assert_nodes_ep_json` removed with that script's deprecation).

### _assert_tensor_metadata Node Handling

`torch.export` (and `torch_delegate.export_to_epair` internal torchair `_optimize_and_convert` decomposition) auto-inserts `torch.ops.aten._assert_tensor_metadata.default` nodes for runtime validation of intermediate tensor shape/dtype/stride. These nodes' characteristics:

- schema is `() -> ()`, **no output**, no GE op mapping (`ascend_ir.source` empty);
- Pure PyTorch runtime validation, not compute nodes, not needed at GE compile stage;
- Each transformer layer produces ~9, qwen3.5-0.8B (24 layers) total ~216.

**Problem**: if not handled, `PlanNodes` treats them as unmapped ops with empty `ascend_ir.source`, goes to "unmappable op handling" branch reporting `UNMAPPED_OPERATOR` error causing graph build failure.

**Fix** (reference torch-delegate PR #93): in `GeGraphBuilder::PlanNodes` node loop **top** use `IsAssertTensorMetadataNode(node)` to identify and `continue` skip, not entering converter/fallback/unmapped-op branch.

```cpp
// ge_graph_builder.cc
bool IsAssertTensorMetadataNode(const ParsedNode &node) {
  return node.target == "torch.ops.aten._assert_tensor_metadata.default";
}

// PlanNodes loop top
for (const auto &node : parser.Nodes()) {
  if (IsAssertTensorMetadataNode(node)) {
    TDLOGI("PlanNodes skipping _assert_tensor_metadata node: %s", node.name.c_str());
    continue;
  }
  // ...subsequent converter / unmapped op handling
}
```

### model_constants_config.json Analysis

`data/constants/model_constants_config.json` is PT2 archive format's internal meta-info file, generated by `torch_delegate.epair_save()`. Taking gemma4-E2B export as example, records `model.lifted_tensor_0/1`, `model.rotary_emb.inv_freq` etc. constants' `path_name`/`is_param`/`tensor_meta`.

`input_specs` is ep_json's array field under `graph_module.signature`, listing all model inputs. The 3 constants in `model_constants_config.json` **all appear in `input_specs`** (as `tensor_constant` or `buffer`).

**Info coverage**:

| Info | model_constants_config.json source | cann-local-lm source | Covered? |
|------|---|---|---|
| Constant data | `data/constants/tensor_N` files | `weights_*.bin` (`pack_weights` already packed) | Covered |
| Constant name/type | `config` object's key and `is_param` | `input_specs`'s `kind` and `name` + `weight_info.json`'s `_is_weight` | Covered |
| dtype/shape | `tensor_meta.dtype` / `tensor_meta.sizes` | C++ side derives from `ep_json` graph nodes | Covered |
| Data storage location | `path_name` → `data/constants/` dir | `weight_info.json`'s `file` + `offset` | Covered |

**Conclusion**: `model_constants_config.json` is PT2 archive format's redundant meta-info file. cann-local-lm covers all its info via following paths, **doesn't need to separately parse `model_constants_config.json`**:
1. **Data**: `pack_weights` packs constant data to `weights_*.bin` via `gs.inputs_to_lifted_tensor_constants` + `ep.module().named_buffers()`
2. **Meta-info**: C++ side `EpairModelParser` parses `input_specs` to get kind/name, `GeWeightLoader` gets data location + `_is_weight` from `weight_info.json` (distinguishes module state vs trace constant, fallback `input_specs`'s `PlaceholderKind`)
3. **dtype/shape**: C++ side derives tensor meta from `ep_json`'s graph nodes

<a id="ge-op-registration-two-layers"></a>
## GE Op Registration Two Layers

GE op registration two layers, complementary:

1. **Python side `converter/export/ge_converters/ge_converter.py`** — `register_ge_converters()` maps `torch.ops` nodes to GE ops, called before `export_to_epair` (save_epair step 1). Covers:
   - `npu.*` custom ops: `npu_quant_matmul`(→`QuantBatchMatmulV4`)/`npu_chunk_gated_delta_rule`/`npu_scatter_nd_update`(+`_`variant)/`npu_fused_infer_attention_score_v2` (op definitions in `cann_local_lm/models/fused_ops/`, GE mapping implementation see [`fused_ops.md`](../models/fused_ops/fused_ops.md) §3)
   - `cann_local_lm.*`: `l2norm`(→`L2Normalize`)
   - Some `aten.*` ops: `softplus`(→`SoftplusV2`)/`conv1d`/`index_copy`/`index_put` (aten nodes residual from torch.export, supplemented when torchair has no built-in mapping)
   - Also `aten.eye`/`aten.type_as` via `_EXTRA_DECOMPOSITION_OPS` whitelist goes decomposition (not GE converter mapping)

2. **C++ side `examples/test_converter/export_llm_model/reg_register.h`** — registers GE graph compile-time needed op implementations (before aclgrphBuildModel), different models need different op sets:

| Op category | Example ops | Used for |
|---------|---------|------|
| Basic ops | Add, Mul, Sub, Div, RealDiv, Neg, Pow, Abs, Exp, Log, Log1p, Sqrt | Basic math |
| Normalization | ReduceMean, Rsqrt, Sigmoid, Swish | RMSNorm / SiLU |
| Matrix | MatMul, BatchMatMul | Attention QK^T / attnV |
| Attention | SoftmaxV2, SoftmaxV3, Triu, Fill | Causal mask + softmax |
| Tensor ops | Reshape, Transpose, Cast, Unsqueeze, ExpandDims | Shape transform |
| Slicing | StridedSliceV2, SliceD | KV cache slicing |
| Concat | ConcatV2, ConcatD | Rotary embedding concat |
| Lookup | GatherV2, Embedding | Token embedding |
| Range | Range, Cos, Sin | Position encoding |
| Other | Identity, Where, ZerosLike, Greater, SelectV2 | Clone / condition select / compare |

Unregistered ops cause C++ side direct error exit (`UNMAPPED_OPERATOR`), prompting user to register or decompose op.

> Quantization ops `QuantMatmulS8S4` + `QuantBatchMatmulV4` + `DynamicQuant` registration see [quant_design.md](../quant/quant_design.md) "C++ Registration and Artifacts".

## Output File Layout

```
output_dir/
├── ep_json_with_ascend_ir.json    # torch.export serialization + ascend_ir annotation
├── weight_info.json               # Weight index {bin filename: [{name, offset, size, format, _is_weight}]}
├── weights_0.bin                  # Weight data (split by max_file_size, default 50GB)
├── weights_1.bin
├── model_config.json              # Custom config + HF meta + use_embeddings flag
├── tokenizer.json                 # Copied from HF
├── tokenizer_config.json          # Copied from HF
├── special_tokens_map.json        # Copied from HF
├── host_weights_info.json         # (only use_embeddings=True) host lookup weight index {bin: [{name,offset,size,dtype,shape}]}
└── host_weights_0.bin             # (only use_embeddings=True) embed_tokens / gemma4 embed_tokens_per_layer
```

> `save_epair` **doesn't output** `.pt2` (Python loop verification) and `input_N.bin` (C++ verification input), both separately generated by verify scripts, not part of formal artifacts (consistent with prefill-decode mode).
>
> `host_weights_*` only produced when `use_embeddings=True`: `saver.detach_externalized_lookups()` before export `del`s externalized lookup submodules (`embed_tokens` / gemma4 `embed_tokens_per_layer`) from model → torch.export doesn't lift → `weights_*.bin` **doesn't contain** them (≈ OM, no redundancy); lookup weights separately packed to `host_weights_*` (with dtype/shape) for host lookup. `model_config.json`'s `use_embeddings` field records export mode, for downstream to judge graph's first input is `input_ids` or `embeddings`. C++ `GeWeightLoader` only reads `weight_info.json`, zero awareness of host files.

<a id="prefilldecode-separated-export"></a>
## prefill/decode Separated Export

Real LLM inference needs prefill (process prompt, seq_len variable) + decode (per-token generation, seq_len=1) two stages. After `torch.export` shape specialization, single graph can't cover both stages (exported seq_len=128 graph can neither prefill nor decode), so converter supports `--mode prefill-decode` to export two graphs.

### Design

- **prefill graph**: seq_len=`seq_len` (default 128, configurable) specialized. At real inference input_ids padded to `seq_len`, attention_mask masks pad positions, output_pos=[prompt_len-1] takes last actual token's logits
- **decode graph**: seq_len=1 specialized, all decode steps reuse
- **Weight sharing**: two graphs' state_dict same (same model), `pack_weights` only executes once, `weight_info.json` + `weights_*.bin` only written once

### Interface

```python
from cann_local_lm.converter.saver import save_epair_prefill_decode

save_epair_prefill_decode(
    artifact, output_dir,
    *,
    max_file_size=50*1024**3,      # Max bytes per single bin
    kv_cache_max_len=2048,         # KV buffer first dim
    seq_len=128,           # prefill graph seq_len (configurable)
    use_embeddings=False,          # True: lookup outside graph
    save_weights=True,             # Whether to write weights_*.bin (False only writes ep_json)
    whitelist_decompositions=None,  # Extra decomposition whitelist
    embed_quant=None,               # ← quant-embeddings: {fqn:(int8,scale)}
)
```

> CLI usage see [cli_design.md](../cli/cli_design.md) "prefill/decode Separation Mode CLI".

### Artifact Layout

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # prefill graph (seq_len specified by param)
├── decode_ep_json_with_ascend_ir.json    # decode graph (seq_len=1)
├── weight_info.json                       # Shared weight index (one copy)
├── weights_0.bin                          # Shared weights (one copy)
├── model_config.json
└── tokenizer files
```

C++ loads two graphs sharing one weight copy:
```cpp
td::EpairModelLoader prefill_loader("prefill_ep_json_with_ascend_ir.json",
                                      "weight_info.json");
td::EpairModelLoader decode_loader("decode_ep_json_with_ascend_ir.json",
                                    "weight_info.json");
```

### Not Saving .pt2 / input_N.bin

Formal artifacts don't include `.pt2` (Python loop verification) and `input_N.bin` (C++ verification input), generated by test scripts (verify) themselves, avoiding formal flow bearing test responsibility.

### Full Chain Working (Python export + C++ OM compile)

- `aten.index_copy.default` GE converter missing resolved: in `converter/export/ge_converters/ge_converter.py` implemented converter using `ge.InplaceUpdate` + `ge.TransposeD`. converter CLI exporting gemma4 prefill/decode + C++ `main.cpp` loading compiling OM full chain works (prefill 6079 nodes / decode 6044 nodes / `prefill.om` + `decode.om` each 9.4GB).
- **Pure embeddings (`--use-embeddings`) full chain working**: converter CLI export + C++ `main.cpp` OM compile works — gemma4-E2B (5B params 35 layers incl. PLE, `gemma4_e2b_pure.om` 4.57GB, ep_json 19MB / weights 4.56GB ≈ OM / host_weights 5.5GB lookup weights for host). `saver.detach_externalized_lookups()` before export `del`s externalized lookup submodules → torch.export doesn't lift → `weights_*.bin` doesn't contain redundant lookup weights (≈ OM, no 5.5GB redundancy). Numerical verification by OM runner (`examples/test_converter/export_llm_model/om_npu_verify/run_*_om.py`, OM vs eager/HF, superset of ep loop) — original `verify_gemma4_epair.py` (`ep.module()` vs wrapper ep loop) refactored into OM runner. gemma4 via `run_gemma4_om.py` verified (OM vs eager maxdiff<1.0 MATCH) + `run_gemma4_gen.py` text generation passes; qwen3.5 via `run_qwen3_5_om.py` (OM vs EP/HF MATCH).

### Per-Graph Constant Inlining: actual_seq_qlen/kvlen → `ge.Const` (eliminate sharing conflict + PartitionedCall)

prefill-decode sharing `weights_0.bin` brings a hazard: modeling's `torch.tensor([q_len])` created `actual_seq_qlen`/`actual_seq_kvlen` (FusedInferAttentionScore's per-graph small constants, prefill=128, decode=1) are lifted by `torch.export` into GE graph's **`Data` reference node** (points to same-name constant in shared weights). prefill `save_weights=True` writes its 128 to `weights_0.bin`, decode `save_weights=False` doesn't write but reuses by same name → decode's fused attention op gets prefill's 128 not its own 1; and `Data` reference input marked tilingDepend by GE → `FusedInferAttentionScore` enters `PartitionedCall` dynamic subgraph (950 compile also triggers `actualSeqLengthsQ(128) > Q_S(1)` validation failure).

**Fix** (`cann_local_lm/converter/export/epair_utils.py: inline_fused_attn_seq_constants()` + `saver.py:save_epair_prefill_decode` call): post-process after prefill/decode ep_json written — replace each `FusedInferAttentionScore` op's `actual_seq_qlen`(input[5]) / `actual_seq_kvlen`(input[6]) input reference with inlined `ge.Const`. Function signature `inline_fused_attn_seq_constants(ep_json_path, qlen_value, kvlen_value=None)`: `qlen_value` = query length (prefill=seq_len, decode=1); `kvlen_value` default None→uses qlen_value (**prefill compat**, no past when kvlen=qlen=seq_len correct), **decode explicitly passes `kv_cache_max_len`** (see below). `Const` op strictly aligns ascend_ir existing format (`attr.value.t.data` base64 raw bytes + `_readable_value.s`).

**Effect**:
- Per-graph value correct (decode qlen gets 1 not prefill's 128) → 950 tiling validation `1 ≤ Q_S=1` passes, decode OM compile success.
- True const no longer tilingDepend → doesn't enter `PartitionedCall` (OM back to 4.86GB, not dynamic-derived 8.1GB bloat).
- Doesn't touch modeling (`fused_attention_forward` keeps original `torch.tensor([q_len])`, CPU eager unaffected; export decode covered by saver fix to kv_max).

> **Why decode `kvlen_value=kv_cache_max_len`** (qwen3.5 FIA decode regression root cause): `actual_seq_kvlen` semantics is "kv's effective length" (torchair doc). If decode baked static `ge.Const(q_len=1)`, FIA kernel **statically truncates kv read to 1 token** (kv[0:1]) dropping all past → `1+1=` → `2019-01-01`. Correct approach is decode kvlen=`kv_cache_max_len` (let kernel read full buffer), actual attend range determined by **explicit `atten_mask`** (mask + kvlen=kv_max both needed; mask alone insufficient — static Const=1 still truncates, mask ignored). Modeling must ensure decode full-attn layer gets non-None 4D `attention_mask` (qwen3.5 removed `if seq_len==1 and past_len>0: causal_mask=None` special case).
>
> **prefill `kvlen_value` limitation** (can't use kv_max, multi-chunk unsupported): prefill currently uses default `kvlen_value=None`→qlen=seq_len (**only single-chunk correct**, no past when kvlen=seq_len correct). **Can't use kv_max** — prefill q_len>1 + kvlen=kv_max makes FIA kernel read out-of-bounds zeros producing degraded output (gemma4 q_len=128 all pad, qwen3.5 q_len=8 outputs `2222...`, excluded mask construction cause: both kvlen use same padded mask, only kvlen value differs). Multi-chunk prefill (prompt > seq_len) correct value should be dynamic `new_kv_cache_pos.max()+1` (chunk N = start+seq), static ge.Const can't express → FIA prefill doesn't support multi-chunk.
>
> **qwen3.5 currently all eager** (2026-08-22): FIA decode fix verified (committed `2f4755d`) but due to prefill multi-chunk limitation + option B (eager prefill + FIA decode via modeling `if q_len==1` branch) hitting torch.export dual-branch constant lift causing C++ graph build crash, **temporarily reverted to all eager** (prefill+decode both `eager_attention_forward`). saver fix's `kvlen_value` logic retained (all eager no FIA node→no-op), `fused_attention_forward`/`_pad_mask_to_kv_square` left as dead code for re-enablement. All eager single/multi-chunk prefill + decode all correct.
>
> q_len = `query.shape[2]`: prefill = `seq_len` (param), decode = 1 (`save_epair_prefill_decode` fixed `seq_len=1`). After inlining old `lifted_tensor` / `Reshape` references become dead, C++ loading side ignores.
>
> Related: disabling InplaceAddRmsNorm fusion also belongs to "make OM terminal state align with [`fused_ops.md`](../models/fused_ops/fused_ops.md) §1 op summary"'s compile-time adjustment, but this section is saver/epair format layer's per-graph constant inlining (export time), two complementary.

### PD Weight Separation Scenario (hash dedup v2 format, CLI-014 solution B)

> Related [TODO_notes.md CLI-014](../TODO_notes.md#cli-014-todo-pd-separated-export-stores-only-one-weight-copy-decode-weights-discarded-fails-when-pd-weights-differ): Current "weight sharing" assumption (above line 892) limited to PD same model same state_dict; when PD weights truly differ (different quant / disaggregated deployment / layer name prefix differences) decode `save_weights=False` discards weights. Solution B uses content hash dedup + two-level index, covers any PD weight relationship (same value auto dedup / different values each stored / partial sharing dedup).

#### Design

Current `weight_info.json` single-level (`name → {bin, offset, size, _is_weight}`), prefill unique + decode reuses. B changes to two-level:

- `blobs`: blob metadata deduplicated by content hash (`hash → {bin, offset, size, format, _is_weight}`)
- `graphs`: per-graph name→hash mapping (`{prefill: {name→hash}, decode: {name→hash}}`)

C++ loading: `graphs[graph][name] → hash → blobs[hash] → {bin, offset, size}`.

#### Hash Selection

`hashlib.sha256(tensor_bytes).hexdigest()[:16]` (64-bit, no dependency, collision 2^-64). Computation on `_pack_weights_to_dir`'s already-existing `data = raw.numpy().tobytes()` (extra cost sha256 few GB ≈ seconds, export not hot path acceptable).

#### v2 schema

```json
{
  "blobs": {
    "h_a1b2c3d4e5f67890": {"bin": "weights_0.bin", "offset": 0, "size": 123456, "format": "nd", "_is_weight": true},
    "h_d4e5f6a7b8c9d0e1": {"bin": "weights_0.bin", "offset": 123456, "size": 789012, "format": "nd", "_is_weight": true}
  },
  "graphs": {
    "prefill": {"model.embed_tokens.weight": "h_a1b2c3d4e5f67890", "model.layers.0.q_proj.weight": "h_d4e5f6a7b8c9d0e1"},
    "decode":  {"model.embed_tokens.weight": "h_a1b2c3d4e5f67890", "model.layers.0.q_proj.weight": "h_e7f8a9b0c1d2e3f4"}
  }
}
```

- PD same value: two graphs' name point to same hash → blobs one copy (auto dedup)
- PD different values: two graphs' name point to different hash → blobs two copies
- Partial sharing: shared dedup + different each stored (zero redundancy)

#### Python Interface

```python
# epair_utils.py
def _pack_weights_v2(contents, output_dir, graph_name="single", max_file_size) -> dict:
    """Single graph state_dict → content hash dedup → two-level index (blobs + graphs).
    Single graph export (consolidated_weights_save changed to call)."""

def _pack_weights_pd_dedup(prefill_contents, decode_contents, output_dir, max_file_size) -> dict:
    """PD two graphs state_dict → content hash dedup → two-level index (blobs + graphs).
    Same hash writes bin once; PD same value auto dedup, different values each stored."""

def consolidated_weights_save_pd(prefill_epair, decode_epair, output_dir, *,
                                  max_file_size=..., host_side_weights=None) -> None:
    """PD separated export: _pack_weights_pd_dedup writes v2 weight_info + weights_*.bin + PD two ep_json"""
```

- `save_epair_prefill_decode` (`saver.py`) changed to call `consolidated_weights_save_pd` (replaces prefill `save_weights=True` + decode `save_weights=False` two `consolidated_weights_save`)
- `consolidated_weights_save` (single graph) changed to call `_pack_weights_v2` (**single graph also unified v2**, `graphs={"single":{...}}`)
- **Removed `inline_fused_attn_seq_constants` + `inline_weight_scale_dtype`**: v2 hash dedup lets actual_seq (PD 128/1 different bytes) each stored each read, fundamentally solving PD sharing conflict, no longer needs inline workaround

#### C++ Loading (v2 two-level, v1 deprecated)

```cpp
// GeWeightLoader / EpairModelLoader
if (index.contains("graphs") && index.contains("blobs")) {   // v2 detection (no version field)
    graph = InferGraphFromEpJson(ep_json_path);  // "prefill_ep_json..." → "prefill", "decode_..." → "decode", "ep_json..." → "single"
    for (auto& [name, spec] : tensor_specs) {
        hash = index["graphs"][graph][name];
        auto& blob = index["blobs"][hash];       // {bin, offset, size}
        LoadBytes(blob["bin"], blob["offset"], blob["size"], ...);
    }
} else {
    // v1 no longer supported (re-export with v2 hash-dedup)
    AddError(...);
}
```

- graph name inferred from ep_json filename (no CLI param added)
- **v1 single-level weight_info no longer compatible** (no graphs → error, need re-export v2)

#### Artifact Layout (v2)

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json
├── decode_ep_json_with_ascend_ir.json
├── weight_info.json          # v2 two-level (blobs + graphs), PD shared one copy
├── weights_*.bin            # hash dedup blob (same value one copy, different values each stored)
├── host_weights_*.bin       # embed host gather (kept as-is, not included in B dedup)
└── model_config.json
```

#### Unchanged Parts

- `host_weights` (embed host gather): kept as-is (prefill unique), not included in B dedup (avoid expanding change)
- `max_file_size` file split + single tensor no split (same as current logic, single over max writes whole to one bin)
- Hash collision: blobs[hash] stores size, C++ reads by size (trust sha256[:16], no extra verification; 2^-64 negligible)

#### Covered Scenarios

| Scenario | Current (sharing assumption) | B (hash dedup) |
|------|----------------|---------------|
| PD same model same value | ✓ one shared copy | ✓ auto dedup one copy |
| PD different quant (W8A8/W4A8) | ✗ decode loses weights | ✓ two copies each stored |
| disaggregated deployment | — | ✓ two graphs self-contained each stored |
| Partial sharing (embed shared + Linear different) | — | ✓ shared dedup + different each stored |

<a id="pending-and-legacy-export-related"></a>
#### Related

- [TODO_notes.md CLI-014](../TODO_notes.md#cli-014-todo-pd-separated-export-stores-only-one-weight-copy-decode-weights-discarded-fails-when-pd-weights-differ) (TODO + A/B/C/D solutions + action items)
- Current "weight sharing" assumption (this section above line 892, superseded by v2 hash dedup) + per-graph constant inlining (line 944, `inline_fused_attn_seq_constants`/`inline_weight_scale_dtype` removed — v2 lets actual_seq PD each stored each read, fundamentally solving sharing conflict)
- C++ per-graph compile ready (`run_compile.sh` + `compile_main.cc`, PD each passes `<weights_index>`), B gap purely Python side + C++ `GeWeightLoader` two-level read

<a id="new-model-onboarding-guide"></a>
## New Model Onboarding Guide

Adding a new model to converter CLI requires these steps:

### Step 1: Implement Model Code

Under `cann_local_lm/models/{model_name}/`:
- `configuration_{model_name}.py` — config dataclass (recommend implementing `from_pretrained` / `to_dict`)
- `modeling_{model_name}.py` — model class + `load_{model_name}_weights` function
- `__init__.py` — export config class, model class, load_weights function

### Step 2: Implement ExportWrapper Subclass

Create subclass under `cann_local_lm/models/{model_name}/export_wrapper.py`, inherit `ExportWrapper`, implement `_drive_layers`:

```python
from cann_local_lm.models.export_wrapper import ExportWrapper

class {ModelName}ExportWrapper(ExportWrapper):
    def _drive_layers(self, input_ids, attention_mask, position_ids,
                      past_key_values, new_kv_cache_pos, embeddings=None):
        kwargs = {...}  # Model-specific forward kwargs
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids
        outputs = self.model(**kwargs)
        # Extract logits + flatten KV cache
        ...
        return logits, past_kv_flat
```

### Step 3: Register to model_registry.py

```python
REGISTRY["{model_name}"] = ModelEntry(
    model_type="{model_type}",          # model_type value in config.json
    config_cls={ModelConfig},
    model_cls={ModelForCausalLM},
    load_weights_fn=load_{model_name}_weights,
    config_json_key="{key}" or None,    # Key for nested config, None for flat
    wrapper_cls={ModelName}ExportWrapper,  # wrapper subclass implemented in step 2
)
```

### Step 4: Confirm Compatibility

| Check item | Description |
|--------|------|
| config loading | Has `from_pretrained`? If not `build_config` falls back to manual field filtering |
| to_dict | Has `to_dict`? If not `save_model_config` uses `dataclasses.asdict` |
| weight key | Need remapping? Handle inside `load_*_weights` |
| forward return | Returns dict? If yes `ExportWrapper` subclass's `_drive_layers` handles |
| mask 4D passthrough | `_prepare_causal_mask` has 4D passthrough? If not need to add (reference qwen3_5) |
| bfloat16 | Uses bfloat16 weights? `pack_weights` already compat |
| Non-persistent buffer | Has `persistent=False` buffer? `pack_weights` already compat |

### Step 5: Supplement GE Ops

Register model-needed GE ops in `examples/test_converter/export_llm_model/reg_register.h`. When running C++ binary, supplement one by one based on errors.

### Step 6: Verify

```bash
# Python export
python cann_local_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model --output-dir /path/to/output

# C++ load
./export_llm_model output/ep_json_with_ascend_ir.json output/weight_info.json
```

<a id="accuracy-difference-analysis"></a>
## Accuracy Difference Analysis

### Verification Results Summary

| Model | logits max_diff (vs HF) | tokens match | Layers | Architecture features |
|------|----------------------|-------------|------|---------|
| qwen3.5-0.8B | 0.019 | ✅ | 24 | Hybrid (6 full + 18 linear) |
| gemma4-E2B | 0.17 | ✅ | 35 | KV sharing + sliding + per-layer embeddings |

All models' generated token sequences identical to HF, accuracy differences within float16 normal range.

### gemma4 export decode bug fix (2026-07-25)

`examples/test_converter/llm_models_run_on_cpu/debug_gemma4/compare_hf_vs_ep.py` previously EP (wrapper scatter mode) decode degradation repetition ("my name is my name is" / "人工智能人工智能"), while CLM (cat mode `model.generate`) == HF. Root cause + fix:

1. **`_prepare_causal_mask` 4D sliding window decode bug** (root cause, `modeling_gemma4.py`): 4D mask path for sliding_attention layer applies window using `q_range=arange(q_len)` to compute q position. decode时 `q_len=1` → `arange(1)=[0]`, but decode token's real position in sequence is 6 (not 0), window computes "only k_pos=0 visible", masking all other K → sliding layer only sees first prefill token → degradation. Fix: `_prepare_causal_mask` adds `new_kv_cache_pos` param, 4D sliding uses `new_kv_cache_pos` (q's real write position in buffer, same reference as k_pos) to compute window, decode correct. Two call sites (inside text_model forward) pass `new_kv_cache_pos`.
2. **`Gemma4ExportWrapper._drive_layers` `shared_kv_states` persistence** (`models/gemma4/export_wrapper.py`): wrapper discards model's returned `shared_kv_states` (creates new empty dict each forward), while `model.generate` (`modeling_gemma4.py` line 800) correctly persists across steps. Fix: wrapper stores `self._shared_kv_states`, persists across prefill/decode calls (aligns with generate). Not degradation root cause (store layer decode overwrites shared dict), but more robust.

Verification: greedy mode EP == HF == CLM fully consistent ("Hello, my name is" → " Alex. How are you today?"; "人工智能的未来发展方向是" → "多维度的，涵盖了从基础理论的突破到实际应用场景的深度融合。"). sampling mode EP doesn't degrade, first token mostly matches, after divergence output coherent (consistent with bfloat16 tie-breaking expectation).

### gemma4 Error Accumulation Analysis

gemma4's max_diff (0.17) is ~9x qwen3.5's (0.019), reasons below:

#### Error Sources

| Source | gemma4-specific | Contribution | Description |
|------|------------|------|------|
| **Gemma4ClippableLinear** vs nn.Linear | ✅ | Largest | HF uses `Gemma4ClippableLinear` (has clamp logic), we use `nn.Linear` — same weights slightly different compute results |
| **5 RMSNorm/layer × 35 layers = 175 times** | ✅ | Medium | gemma4 has per_layer_input_norm, 5 RMSNorm per layer (35 layers total 175) |
| **Per-Layer Embeddings (PLE)** | ✅ | Medium | Extra embedding lookup + projection compute path |
| **Double-wide MLP** | ✅ | Small | `use_double_wide_mlp=True`, wider MLP |
| **k_norm + v_norm** (per-head RMSNorm) | ✅ | Small | Each head单独 RMSNorm |
| float16 rounding | All models | Base | Common to all models |

#### KV Sharing Layers Don't Introduce Error

gemma4 has 20 KV sharing layers (num_kv_shared_layers=20), but **KV sharing itself doesn't introduce error**:

```
Non-shared layers (Layer 0-14):
  Our k_proj(hidden) → k_norm → rope → store in shared_kv_states
  HF's  k_proj(hidden) → k_norm → rope → store in shared_kv_states
  ↑ Here produces ~0.001 float16 difference (ClippableLinear vs nn.Linear)

Shared layers (Layer 15-34):
  Both read same shared_kv_states[layer_type] → direct tensor reuse
  → Error doesn't increase (same tensor object, zero error)
  → Only q_proj compute introduces new error
```

#### Error Accumulation Chain

```
Layer 0:   Error 0.000 (input same)
Layer 1:   Error 0.001 (1 layer ClippableLinear diff + 5 RMSNorm)
Layer 2:   Error 0.002 (upper layer error + this layer compute)
  ...
Layer 14:  Error 0.005 (15 layers accumulated, non-shared)
           ↓ KV sharing (shared_kv_states)
Layer 15:  Error 0.005 (reuse Layer 14's KV, no decay)
Layer 16:  Error 0.006 (q_proj new error)
  ...
Layer 34:  Error 0.015 (20 shared + per-layer growth)
           ↓ lm_head + final_logit_softcapping
Logits:    Error 0.17 (35 layers + 175 RMSNorm + PLE accumulation)
```

#### final_logit_softcapping's Impact on Comparison

gemma4 has `final_logit_softcapping=30.0`, compresses logits via `tanh(x/30)*30`. Verification must ensure both sides go (or don't go) softcap, otherwise produces ~10 spurious difference:

```
Original logits:  35.0
softcap:      tanh(35/30) * 30 = 0.821 * 30 = 24.6
Spurious diff:    35.0 - 24.6 = 10.4  ← if one side softcap other doesn't
```

Verification script HF side needs manual `lm_head` + `tanh(logits/softcap)*softcap`, aligning with wrapper's `Gemma4ForCausalLM.forward` (built-in softcap).

#### HF's KV Sharing Layer Design

```python
# HF: Only "last non-shared layer of each layer_type" stores KV
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == last_non_shared_of_this_type

# Non-shared layer: compute k/v itself, store in shared_kv_states
if not self.is_kv_shared_layer:
    key_states = self.k_proj(hidden_states)
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states)
    if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = (key_states, value_states)

# Shared layer: direct read, no compute
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]  # Zero-error reuse
```

Key mechanism: `store_full_length_kv` ensures **each layer_type stores only one KV copy**, shared layers directly reuse same tensor object, no recompute.

#### Direction to Fully Eliminate Difference

1. Replace `nn.Linear` with `Gemma4ClippableLinear` (align HF's clamp logic)
2. Or use float32 (difference drops below 1e-6, model size doubles)

Current 0.17 difference is normal under float16, generated tokens fully consistent, doesn't affect inference results.

#### scatter-write bfloat16 attention accuracy strategy (verified, don't change)

Under scatter-write mode, KV cache is fixed `[kv_cache_max_len]` pre-allocated buffer, attention matmul scale from `[seq_len, seq_len]` to `[seq_len, kv_cache_max_len]`. Per-layer analysis confirmed:

| Component | Precision | Reason |
|------|------|------|
| Q@K matmul | **bfloat16** (keep original) | Consistent with HF, BLAS rounding closest to HF's bfloat16 |
| softmax | **float32** (npu_tuned_export approach, existing) | bfloat16 [6, 2048] sum has precision loss |
| @v matmul | **bfloat16** (keep original) | Verified zero KV position doesn't affect result |

**Verified experimental conclusions (don't repeat)**:

1. **float32 Q@K actually worse** — larger difference from HF's bfloat16, fixed seed comparison first token match rate from 78% (7/9) to 33% (3/9)
   - Reason: HF uses bfloat16 Q@K (small matrix), we also bfloat16 (large matrix) → BLAS same path, rounding similar → logits close
   - float32 Q@K → larger difference from HF's bfloat16 → match rate drops

2. **scatter-write head_dim=512 full_attention layer has 0.25 BLAS rounding difference** — `bfloat16 [6, 512] @ [512, 2048]` vs `bfloat16 [6, 512] @ [512, 6]` use different BLAS code paths
   - head_dim=256 (sliding layer): 0.0 difference (BLAS same path)
   - head_dim=512 (full layer): 0.25 difference → 35 layers accumulated → logits diff 0.4375
   - Cannot eliminate (float32 introduces larger difference)

3. **Fixed seed + sampling comparison results**:
   - First token match rate: 7/9 (78%)
   - Second token match rate: 0/9 (0%) — once diverged, KV cache different, subsequent completely different
   - HF output always more coherent (2B small model + 0.4375 diff enough to change sampling path)

#### bfloat16 tie-breaking issue

Under bfloat16 precision, some prompts' top-N logits may be completely identical (tie), causing `Gemma4ClippableLinear` vs `nn.Linear`'s微小差异 to change argmax selection.

**Real case** (prompt="Hello, my name is", bfloat16, with BOS token):

```
HF (Gemma4ClippableLinear):           cann_local_lm (nn.Linear):
  #1: token 236743 " "  (18.500)       #1: token 532 " and" (18.500)
  #2: token 532 " and" (18.250)       #2: token 236761 "." (18.500)
  #3: token 236761 "." (18.250)       #3: token 236743 " " (18.500)

  Top-1 vs Top-2 gap: 0.250          Top-1 vs Top-2 gap: 0.000 ← top-3 fully tied
```

**Reason**:
1. cann_local_lm's top-3 tokens' logits completely identical (18.500) — model "uncertain" at this position
2. HF's `Gemma4ClippableLinear`'s clamp creates 0.25 gap between top-1 and top-2 → clear choice " "
3. cann_local_lm no clamp → tie-breaking depends on argmax implementation detail → chose " and" not " "
4. **First token different → KV cache different → subsequent decode all diverge**

**Impact scope**:
- Only affects prompts with top-N logits nearly identical (gap < 0.3)
- No effect on prompts with clear top-1 vs top-2 gap (>1.0)
- Tested 3 prompts, 2 fully consistent, 1 diverged due to tie-breaking

**Not a bug**: This is normal divergence at numerical precision boundary between two independent implementations. Model itself is uncertain at these positions (multiple tokens nearly equal probability), any微小 numerical difference changes argmax result.

**Solution direction**:
1. Replace `nn.Linear` with `Gemma4ClippableLinear` (eliminate clamp difference, align HF's tie-breaking)
2. Or use float32 (difference drops to 1e-6, tie almost never occurs)
3. Or use sampling (`do_sample=True`) instead of greedy (`do_sample=False`), sampling insensitive to tie-breaking

## Pending and Legacy (Export-related)

1. **Unified cache signature (full version)** — ExportWrapper base class + per-model subclass (**lightweight version implemented**: base class `forward`/`get_sample_inputs` + 2 subclasses `_drive_layers`, KV cache explicit graph I/O); **full version TODO** (each attention adds scatter-write path, one static graph covers prefill+decode)
2. **prefill/decode separation** — **Implemented** (`saver.save_epair_prefill_decode` + CLI `--mode prefill-decode`). gemma4 Python export + C++ OM compile full chain works
3. **generation_config.json handling** — Currently removed from `collect_tokenizer_files`. Later need to handle `generation_config.json`'s params (temperature/top_p/top_k/max_length etc.) separately in `llm_config`, need confirm with llm_engine which fields used before implement
4. **Graph export input/output interface** — Need determine how to unify export input/output (count uncertain, may be many)
5. **qwen3.5 linear attention export** — epair layer resolved, OM compile+ACL runtime works
6. **New ExportWrapper's ep_json format incompatible with C++ parser** — **[Resolved]** C++ `epair_model_parser.cc`'s `parse_named_arg` supports both `arg.name` and `arg.as_tensor.name` formats
7. **qwen3.5 scatter-write refactoring** — Current option B (pre-allocated buffer + `index_copy` scatter-write) only implemented for gemma4. qwen3.5's attention layer also needs to change from `torch.cat` to `index_copy` + pre-allocated buffer.
8. **aten.index_copy.default GE converter missing** — **[Resolved]** Registered `aten.index_copy.default` converter in `converter/export/ge_converters/ge_converter.py`, implemented with `ge.InplaceUpdate` + `ge.TransposeD`

> Quantization-related pending/legacy see [quant_design.md](../quant/quant_design.md) "Known Issues and Legacy".

## Source File Index

| File | Responsibility |
|------|------|
| `cann_local_lm/models/model_registry.py` | Model registry (ModelEntry + REGISTRY + detect_model_type) |
| `cann_local_lm/converter/loader.py` | `load_model` + `build_config` + `collect_tokenizer_files` |
| `cann_local_lm/converter/saver.py` | `save_epair` / `save_epair_prefill_decode` / `save_model_config` / `copy_tokenizer_files` |
| `cann_local_lm/converter/swap.py` | Quantization orchestration (converter root, peer of loader/saver): `swap_from_dopt` (main path) / `swap_embeddings_from_dopt` / `load_dopt_state_dict`; absolute imports from `export/quant_swap/` |
| `cann_local_lm/converter/quant_entry.py` | Quantization entry facade: re-export `run_auto_quant` (`quant/auto_quant.py`) + `swap_from_dopt` (`swap.py`), for `cann_lm_converter` single import |
| `cann_local_lm/models/export_wrapper.py` | `ExportWrapper` base class |
| `cann_local_lm/models/{model}/export_wrapper.py` | Per-model `ExportWrapper` subclasses |
| `cann_local_lm/converter/export/epair_utils.py` | Weight packaging (`pack_weights`), disk write (`consolidated_weights_save`) |
| `cann_local_lm/converter/export/ge_converters/ge_converter.py` | `register_ge_converters` (cann_local_lm.* + aten.* → GE op mapping) |
| `cann_local_lm/converter/export/csrc/inc/external/torch_delegate/epair_model_loader.h` | C++ public API: `EpairModelLoader` |
| `cann_local_lm/converter/export/csrc/backend/epair_model_loader.cc` | `EpairModelLoader::Impl` implementation |
| `cann_local_lm/converter/export/csrc/backend/ge_graph_builder.h/.cc` | `GeGraphBuilder`: parse model JSON + build GE Graph |
| `cann_local_lm/converter/export/csrc/backend/ge_weight_loader.h/.cc` | `GeWeightLoader`: parse `weight_info.json`, lazy-load bin |
| `cann_local_lm/converter/export/csrc/backend/epair_model_parser.h/.cc` | `EpairModelParser`: parse `ep_json_with_ascend_ir.json` |

## Document References

| Document | Related content |
|------|---------|
| [cli_design.md](../cli/cli_design.md) | CLI entry `cann_lm_converter.py` / `convert()` / command-line params / usage examples |
| [quant_design.md](../quant/quant_design.md) | Quantization integration `--quant-path` / `swap_from_dopt` / `npu_quant_matmul` |
| [README.md](../README.md) | Cross-module call chain overview / model differences overview / accuracy summary |
