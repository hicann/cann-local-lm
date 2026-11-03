# Embedding Host Externalization Design (Pure Embeddings Mode)

English | [简体中文](../../zh/cann_lm_converter/feature_embedding_design.md)

> Under `use_embeddings=True` (CLI `--use-embeddings` default, CLI-002), the `embed_tokens` lookup is moved from the graph to the host: the host looks up to get `embeddings` as a graph input (replacing `input_ids`). The graph contains no `embed_tokens` submodule + lookup; `weights_*.bin` does not contain lookup weights (≈ OM); lookup weights are packed separately into `host_weights_*.bin`.
>
> Supported: float embed (non-quantized) + int8 embed (quant, CANN lightweight tool `Quant_Embed_MinMax`). gemma4 has PLE (`per_layer_token_identity`, float lookup) specific to it.

## Table of Contents

- [Pure Embeddings Mode Flow](#pure-embeddings-mode-flow-use_embeddingstrue-default)
- [float embed (non-quantized)](#float-embed-non-quantized)
- [int8 embed (quant) processing flow](#int8-embed-quant-processing-flow)
- [Key Components](#key-components)
- [lift Concept (torch.export)](#lift-concept-torchexport)
- [gemma4 PLE Special (per_layer_token_identity)](#gemma4-ple-special-per_layer_token_identity)
- [Lightweight tool's implementation for gemma4](#lightweight-tools-implementation-for-gemma4)
- [Data Flow](#data-flow)
- [Artifact Layout](#artifact-layout)

---

<a id="pure-embeddings-mode-flow-use_embeddingstrue-default"></a>
## Pure Embeddings Mode Flow (use_embeddings=True, default)

### 10 steps

1. **`use_embeddings=True`** (convert `--use-embeddings` → saver)
2. **`get_sample_inputs(seq_len, use_embeddings, quant_embeddings)`** (export_wrapper.py:93-168): host lookup generates `embeddings`, `input_ids=None` (placeholder)
3. **`host_side_weight_names()`** (base class L182-184 / gemma4 L285-287): returns the FQNs of lookup weights to be externalized
4. **`_extract_host_side_weights(wrapper, use_embeddings)`** (saver.py:25-31 + L97): packs lookup weights into `host_side_weights`
5. **`detach_externalized_lookups()`** (saver.py:100-101 + base class L186-194 / gemma4 L289-295): `del` lookup submodules → torch.export does not lift
6. **`torch.export.export(wrapper, sample_inputs, strict=False)`** (saver.py:103): `embed_tokens` weight not lifted (not a graph input), graph input is `embeddings` (+ scale)
7. **`_assert_externalized_not_lifted(ep, names)`** (saver.py:105-106): checks `embed_tokens` weight is not in lifted
8. **`export_to_epair` + `consolidated_weights_save`** (saver.py:110, L119-124): graph weights → `weights_*.bin`; lookup weights → `host_weights_*.bin`
9. **`save_model_config(use_embeddings=true)`** (saver.py:127-128): records mode
10. **Host inference** (downstream Python): reads `model_config use_embeddings` + `host_weights` → gather → `embeddings` → pass to graph

---

<a id="float-embed-non-quantized"></a>
## float embed (non-quantized)

`get_sample_inputs` (base class L146-157, quant_embeddings=False):
```python
embeddings = self.model.model.embed_tokens(input_ids).detach()   # host lookup float
return (None, attention_mask, position_ids, past_key_values, new_kv_cache_pos,
        embeddings, output_pos, *extras)
```
- Graph input: **`embeddings` (float)** — one, no scale
- forward (export_wrapper.py:57-60): `embeddings_scale is None` → no dequant, uses float directly
- Host inference: reads `host_weights_*.bin` (`embed_tokens.weight` float) → gather `input_ids` → float embeddings → pass to graph

---

<a id="int8-embed-quant-processing-flow"></a>
## int8 embed (quant) processing flow

### 1. Lightweight tool quantizes embed (Quant_Embed_MinMax)

`dopt_wrapper.build_layer_strategy` (dopt_wrapper.py:41-43): `nn.Embedding → Quant_Embed_MinMax` (W8 per-channel int8 + scale).

Lightweight tool GPTQ output `model.safetensors` contains:
- `embed_tokens.weight` (fakequant fp16)
- `embed_tokens.quant_op.weight_quantizer.s` (per-channel scale `[vocab]`)

`dopt_config.json`: `embed_tokens: {quant_strategy: "Quant_Embed_MinMax"}`

### 2. swap extracts int8+scale (`swap_embeddings_from_dopt`, swap.py:35-56)

```python
def swap_embeddings_from_dopt(sd, model) -> dict[str, tuple[Tensor, Tensor]]:
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Embedding): continue
        wkey = f"{name}.weight"; skey = f"{name}.quant_op.weight_quantizer.s"
        if wkey not in sd or skey not in sd: continue   # Skip those without scale (e.g. gemma4 PLE not quantized)
        int8, scale = materialize_int8_per_channel_symmetric(sd[wkey], sd[skey])
        out[f"model.{name}.weight"] = (int8, scale)
    return out
```
- Iterates `nn.Embedding`, reads fakequant weight + per-channel scale from sd
- `materialize_int8_per_channel_symmetric` → **(int8 `[vocab,hidden]`, scale `[vocab]`)**
- Returns `embed_quant = {fqn: (int8, scale)}`, skips those without scale

### 3. saver injects into wrapper (`set_quant_embeddings`, saver.py:82-84)

```python
quant_embeddings = bool(embed_quant)   # Non-empty (lightweight tool quantized embed) → True
if quant_embeddings:
    wrapper.set_quant_embeddings(embed_quant)   # store self._quant_embeddings
```
`wrapper._quant_embeddings = {fqn: (int8, scale)}` (export_wrapper.py:36-38).

### 4. get_sample_inputs (quant): host gather int8+scale (base class L130-145)

```python
q_w, q_s = self._quant_embeddings["model.model.embed_tokens.weight"]
emb_q = q_w[input_ids]      # host gather int8 weight[input_ids] → int8 embeddings [seq,hidden]
emb_scale = q_s[input_ids]  # host gather scale[input_ids] → per-token scale [seq,1]
return (None, attention_mask, position_ids, past_key_values, new_kv_cache_pos,
        emb_q, output_pos, ple, emb_scale)
```
- **host gathers first**: `q_w[input_ids]` (int8 indexed by input id) + `q_s[input_ids]` (scale indexed by input id)
- Graph input: `emb_q` (int8, per-token) + `emb_scale` (per-token scale) — **values after host gather**, not complete weight/scale
- `input_ids=None` (placeholder)

### 5. saver pack host_weights (complete int8+scale, saver.py:90-95)

```python
if quant_embeddings:
    host_side_weights = [
        (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
    ] + [
        (fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()
    ]
```
- Packs **complete int8 `[vocab,hidden]` + scale `[vocab]`** directly from `embed_quant` (with `_q`/`_scale` suffixes)
- → `host_weights_*.bin` (for host to gather at inference)

### 6. detach + torch.export (saver.py:100-103)

```python
if use_embeddings:
    wrapper.detach_externalized_lookups()   # del embed_tokens (stripped, not lifted)
ep = torch.export.export(wrapper, sample_inputs, strict=False)
```
- `del text_model.embed_tokens` → weight not lifted (not in `weights_*.bin`)
- Graph input: `emb_q` (int8) + `emb_scale` (scale) + others; graph contains forward dequant

### 7. forward dequant (in graph, export_wrapper.py:57-60)

```python
if embeddings is not None and embeddings_scale is not None:   # Both non-None when quantized
    dt = self._model_dtype
    embeddings = embeddings.to(dt) * embeddings_scale.to(dt).unsqueeze(-1)   # int8*scale → float
```
`int8 * per-token scale → float` passed to `_drive_layers`.

### 8. host_weights packaging + model_config

- `consolidated_weights_save`: graph weights → `weights_*.bin` + `weight_info.json`; `host_side_weights` (int8 + scale, with `_q`/`_scale`) → `host_weights_*.bin` + `host_weights_info.json`
- `save_model_config`: `quant_embeddings: true` (downstream judges graph's first input is int8 embeddings + scale)

### 9. Host inference (downstream Python)

- Read `model_config quant_embeddings=true` → graph's first input is int8 embeddings + scale
- Read `host_weights_*.bin` (`embed_tokens.weight_q` int8 `[vocab,hidden]` + `embed_tokens.weight_scale` `[vocab]`)
- Host lookup: `input_ids → embed_q gather → int8 embeddings`; `input_ids → embed_scale gather → per-token scale`
- Pass int8 embeddings + scale to graph → graph forward `int8*scale→float` dequant

---

<a id="key-components"></a>
## Key Components

### `set_quant_embeddings` (export_wrapper.py:36-38)

Injects `embed_quant` (`{fqn: (int8_weight, per_channel_scale)}`) into `wrapper._quant_embeddings`, enabling quant-embeddings mode (host int8 lookup + forward dequant). Trigger condition: `bool(embed_quant)` True (when lightweight tool quantized embed).

### `detach_externalized_lookups` (export_wrapper.py:186-194 / gemma4:289-295)

```python
def detach_externalized_lookups(self):
    text_model = self.model.model
    if hasattr(text_model, "embed_tokens"):
        del text_model.embed_tokens   # gemma4 also del embed_tokens_per_layer
```
Strips lookup submodules → torch.export does not lift their weights → `weights_*.bin` does not contain lookup weights (≈ OM).

### `_assert_externalized_not_lifted` (saver.py:34-48 + L105-106)

```python
def _assert_externalized_not_lifted(ep, names):
    lifted = set()
    gs = ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for _placeholder, logical_name in mapping.items():
            lifted.add(logical_name)
    leaked = [n for n in names if n in lifted]
    if leaked:
        raise RuntimeError(f"detach_externalized_lookups ineffective: {leaked} still lifted. ...")
```
Checks `embed_tokens` weight is not in `inputs_to_parameters`/`buffers`/`lifted_tensor_constants` (if present, detach was ineffective, raise: forward touched the lookup submodule).

### `_extract_host_side_weights` (saver.py:25-31)

```python
def _extract_host_side_weights(wrapper, use_embeddings):
    if not use_embeddings:
        return []
    candidate_names = wrapper.host_side_weight_names()
    sd = wrapper.state_dict()
    return [(name, sd[name].detach().cpu()) for name in candidate_names if name in sd]
```
When use_embeddings, takes lookup weights from `host_side_weight_names()` + `state_dict()`, packs `[(name, tensor)]` (float embed).

### `host_side_weight_names` (base class L182-184 / gemma4 L285-287)

- Base class: `["model.model.embed_tokens.weight"]`
- gemma4: `["model.model.embed_tokens.weight", "model.model.embed_tokens_per_layer.weight"]` (PLE extra)

---

<a id="lift-concept-torchexport"></a>
## lift Concept (torch.export)

**lift**: torch.export "promotes" a model's parameter/buffer to a graph input placeholder (weight as graph input, not an internal constant), decoupling the graph from the weights.

- **Plain nn.Module**: weight is a module attribute (`module.weight`), used directly in forward (internal constant)
- **After torch.export trace**: weight is lifted to a graph input placeholder (a Tensor input parameter), referenced by placeholder in the graph
- **Purpose**: decouple graph from weights — same graph can use different weights (swap weights without re-tracing)

### Manifestation (ExportedProgram.graph_signature)

- `inputs_to_parameters`: weight placeholder → parameter logical name mapping
- `inputs_to_buffers`: buffer similarly
- `inputs_to_lifted_tensor_constants`: tensor constants

### epair/saver context

| | Weight state | Goes into weights_*.bin? |
|---|---|---|
| **lifted** (submodule in model) | Graph input placeholder | ✅ (redundant, pure embeddings should not have) |
| **Not lifted after detach** (submodule stripped) | Not in graph | ❌ (≈ OM, lookup weights go via host_weights) |

`detach_externalized_lookups` makes `embed_tokens` weight **not lifted** (not in `weights_*.bin`, avoids redundancy); lookup weights go via `host_weights_*.bin`. `_assert_externalized_not_lifted` is the fallback check.

---

<a id="gemma4-ple-special-per_layer_token_identity"></a>
## gemma4 PLE Special (per_layer_token_identity)

### per_layer_token_identity has no scale

gemma4 `_build_external_extras` (gemma4/export_wrapper.py:273-283):
```python
def _build_external_extras(self, input_ids):
    """gemma4 PLE: host computes per_layer_token_identity (embed_tokens_per_layer lookup result)"""
    m = self.model.model
    token_identity = m.embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape, m.config.num_hidden_layers, m.hidden_size_per_layer_input
    ).detach()
    return (token_identity,)   # Single tensor, no scale
```
- `token_identity = embed_tokens_per_layer(input_ids)` (direct float lookup, reshape + detach)
- shape `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`
- **Returns single tensor, no scale pair**

forward signature (export_wrapper.py:44-55): `per_layer_token_identity` single tensor (no `_scale` parameter), unlike `embeddings` which has `embeddings_scale`.

### Comparison with embed_tokens quantization

| | embed_tokens (quant) | PLE (embed_tokens_per_layer) |
|---|---|---|
| Graph input | `emb_q` (int8) + `emb_scale` (scale) — **two** | `per_layer_token_identity` (float) — **one** |
| forward | `int8*scale→float` dequant | Use float directly |
| host gather | `q_w[input_ids]` + `q_s[input_ids]` | `embed_tokens_per_layer(input_ids)` (float lookup) |
| host_weights | int8 + scale (with `_q`/`_scale`) | float weight (`_extract_host_side_weights`) |

### Key Point

`_build_external_extras` uses `m.embed_tokens_per_layer(input_ids)` (**direct float lookup**), even when `embed_tokens` is quantized (`quant_embeddings=True` goes `emb_q`/`emb_scale` int8 path), **PLE still goes float lookup** — `get_sample_inputs` quant branch only takes `embed_tokens`'s int8+scale from `_quant_embeddings`, PLE still uses `_build_external_extras`'s float result.

That is, **embed_tokens is quantized (W8 int8+scale), PLE is not quantized (float)**.

---

<a id="lightweight-tools-implementation-for-gemma4"></a>
## Lightweight tool's implementation for gemma4

### 1. embed_tokens (Gemma4TextScaledWordEmbedding) → QGemma4ScaledEmbedding

**do_opt.py:22-26** (visible_type_dict):
```python
visible_type_dict = {
    torch.nn.Embedding: QEmbedding,
    torch.nn.Linear: QLinear,
    "Gemma4TextScaledWordEmbedding": QGemma4ScaledEmbedding,
}
```
`get_target_op_cls` (L29-43): prefers `type(m)`, then `m.__class__.__name__` (matches "Gemma4TextScaledWordEmbedding").

**qembed.py:67-106** (QGemma4ScaledEmbedding):
- weight per-channel int8 quantization (forces `per_channel=True`, L80-82)
- **`embed_scale`** (gemma4 specific): `m.embed_scale` or `scalar_embed_scale` (L94-98) — gemma4 scaled embed = `embed_tokens * embed_scale`
- `tie_lmhead` (optional, L88-92)
- That is, gemma4 embed_tokens goes through **QGemma4ScaledEmbedding** (int8 + embed_scale), not plain QEmbedding

### 2. PLE `per_layer_model_projection` (Linear) → GPTQ int4

**ptq_loop.py:267-297**:
```python
ple_proj_key = 'model.per_layer_model_projection'
ple_weight_config = _get_weight_config(args.quant_config, ple_proj_key)
if ple_weight_config is not None:
    ple_proj_layer = model.model.per_layer_model_projection.to(dev)
    ple_proj_quant = WeightQuantHessian(ple_proj_layer)   # GPTQ Hessian
    ple_proj_quant.quantizer.configure(wbits, perchannel=True, sym=args.sym, ...)
    for ids_cpu in input_ids_list:
        inputs_embeds = model.model.embed_tokens(ids_cpu.to(dev))
        out = ple_proj_layer(inputs_embeds)   # projection forward
        ple_proj_quant.add_batch(inputs_embeds.data, out.data)   # Hessian accumulation
    ple_proj_quant.fasterquant(blocksize, percdamp, groupsize, ...)   # GPTQ int4
```
- **`per_layer_model_projection`** (PLE projection Linear layer) goes through **GPTQ Hessian int4** (bit/groupsize/blocksize, perchannel, sym)
- Not `embed_tokens_per_layer` lookup, but the projection layer

### 3. PLE per_layer_input (lookup + projection, ptq_loop.py:149-164)

```python
ple_enabled = bool(getattr(model.model, "hidden_size_per_layer_input", None))   # gemma4 PLE flag
if ple_enabled:
    ids_cpu = input_ids_list[j]
    orig_embed = model.model.embed_tokens(ids_cpu.to(dev))   # embed_tokens lookup
    pli = model.model.get_per_layer_inputs(ids_cpu.to(dev), None)   # PLE per_layer_inputs
    pli = model.model.project_per_layer_inputs(orig_embed, pli)   # projection
    per_layer_input = pli[:, :, i, :]   # per-layer input
```
- When PLE enabled (`hidden_size_per_layer_input`), `embed_tokens(input_ids)` + `get_per_layer_inputs` + `project_per_layer_inputs` → per_layer_input (per-layer input)
- Passed to layer forward (L182: `per_layer_input=per_layer_input`)

### embed_tokens_per_layer lookup (→ per_layer_token_identity)

The lightweight tool does not specifically quantize `embed_tokens_per_layer` lookup (only used as `get_per_layer_inputs` input, float) → cann_local_lm's `per_layer_token_identity` is also float (no scale).

### Lightweight tool gemma4 quantization summary

| Component | Lightweight tool handling | Has scale? |
|------|---------|----------|
| `embed_tokens` (Gemma4TextScaledWordEmbedding) | QGemma4ScaledEmbedding (int8 + embed_scale) | ✅ (weight scale + embed_scale) |
| `per_layer_model_projection` (PLE projection Linear) | GPTQ int4 | ✅ (group_size/scale) |
| `embed_tokens_per_layer` (PLE lookup) | Not specifically quantized (only as `get_per_layer_inputs` input, float) | ❌ (float lookup) |
| **per_layer_token_identity** (cann_local_lm) | `embed_tokens_per_layer(input_ids)` float lookup | ❌ (no scale) |

**The lightweight tool quantizes `per_layer_model_projection` (projection Linear, int4), not `embed_tokens_per_layer` (lookup Embedding)**. `embed_tokens_per_layer` in the lightweight tool is only used as `get_per_layer_inputs` lookup input (float), not quantized → cann_local_lm's `per_layer_token_identity` (= embed_tokens_per_layer lookup) also has no scale (float).

---

<a id="data-flow"></a>
## Data Flow

### float embed (non-quant)

```
input_ids (host)
  ↓ host lookup (host_weights_*.bin: embed_tokens.weight gather)
  ↓ embeddings (float)
  ↓ pass to graph (graph first input embeddings, replacing input_ids)
graph (forward uses float directly, no dequant)
```

### int8 embed (quant)

```
Lightweight tool output (embed_tokens fakequant fp16 + scale [vocab])
  ↓ swap materialize_int8_per_channel_symmetric
int8 [vocab,hidden] + scale [vocab] (embed_quant)
  ↓ saver pack host_weights (complete)
host_weights_*.bin (int8 + scale, with _q/_scale suffixes)
  ↓ get_sample_inputs: host gather input_ids
Graph input: emb_q (int8 per-token) + emb_scale (per-token scale)
  ↓ torch.export (detach embed_tokens, not lifted)
graph (forward dequant: int8*scale → float)
  ↓ host inference
host reads host_weights → gather input_ids → int8+scale → pass to graph → dequant
```

### int8 embed position reference

| Position | int8/scale state | Shape |
|------|----------------|------|
| Lightweight tool output | fakequant fp16 weight + scale buffer | `[vocab,hidden]` + `[vocab]` |
| swap `embed_quant` | materialized int8 + scale | `[vocab,hidden]` + `[vocab]` |
| `host_weights_*.bin` | **complete** int8 + scale (with `_q`/`_scale`) | `[vocab,hidden]` + `[vocab]` |
| Graph input (sample_inputs) | **after host gather** per-token | `[seq,hidden]` + `[seq,1]` |
| Graph forward | dequant `int8*scale→float` | `[seq,hidden]` |

---

<a id="artifact-layout"></a>
## Artifact Layout

### use_embeddings=True

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # Graph IR (no embed_tokens lookup)
├── decode_ep_json_with_ascend_ir.json
├── weight_info.json                       # Graph weight index (no embed_tokens)
├── weights_0.bin                          # Graph weights (Linear int4+scale + LayerNorm, no lookup weights ≈ OM)
├── host_weights_info.json                 # host weight index
├── host_weights_0.bin                     # Lookup weights (float embed_tokens.weight or int8+scale with _q/_scale)
├── model_config.json                     # use_embeddings=true (quant_embeddings=true if int8)
└── tokenizer files
```

### C++ Loading Side

- `GeWeightLoader` reads `weight_info.json` + `weights_*.bin` (graph weights)
- **Does not read `host_weights`** — host lookup is on the Python inference script side (`llm_models_run_on_cpu/debug_*` etc.)
- `model_config use_embeddings` lets downstream judge whether graph's first input is `input_ids` or `embeddings`

---

## Key Points Summary

1. **embed_tokens not in graph** (`detach_externalized_lookups` → not lifted), lookup weights in `host_weights_*.bin`
2. **Graph input is host-gathered per-token** (`embeddings`/`emb_q`+`emb_scale`), not complete weight/scale
3. **forward dequant `int8*scale→float`** (when quant, in graph)
4. **gemma4 PLE (`per_layer_token_identity`) float lookup, no scale**; lightweight tool quantizes `per_layer_model_projection` (projection Linear int4), not `embed_tokens_per_layer` lookup
5. **embed_tokens quantization** (W8 int8+scale) and **PLE not quantized** (float) coexist — embed_tokens goes int8, PLE goes float

## Related

- Design document: [export_design.md](./export/export_design.md) "Pure Embeddings Mode" / "saver" / "host_weights mechanism"
- Code: `cann_local_lm/converter/saver.py`, `cann_local_lm/models/export_wrapper.py`, `cann_local_lm/converter/swap.py` (`swap_embeddings_from_dopt`), `cann_local_lm/models/gemma4/export_wrapper.py`
- Lightweight tool: `cann_local_lm/converter/quant/dopt_lm/do_opt.py`, `modules/qembed.py`, `opt_funcs/weight_quant_ptq/ptq_algo_hessian/ptq_loop.py`
