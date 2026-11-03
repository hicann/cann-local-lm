English | [简体中文](../../../zh/cann_lm_engine/embedding/embedding_config.md)

This file specifies the configuration parameter input structure for the embedding module:

example:

```json
{
    "embedding_config": {
        "embedding_weights": "gemma4.embedding_weights",
        "embedding_quant_mode": "per_token",
        "embedding_load_mode": "in_memory",
        "embedding_type": "fp16",
        "embedding_scale_type": "fp16",
        "token_identity_type": "int8",
        "token_identity_scale_type": "fp32"
    }
}
```

## Keys

| Key | Meaning | Default |
|---|---|---|
| `embedding_weights` | Single-file path (four-segment layout, see below); empty = not loaded (tokenId deployment form) | required (when embedding is enabled) |
| `embedding_quant_mode` | Scale granularity: `per_token` / `per_group` | `per_token` |
| `embedding_load_mode` | Load mode: `in_memory` / `in_flash` | `in_memory` |
| `embedding_type` | Main table storage type: `fp32` / `fp16` / `int8` | `fp32` |
| `embedding_scale_type` | Main scale storage type: `fp32` / `fp16` | `none` |
| `token_identity_type` | Identity table (per_layer weight) read type: `fp32` / `fp16` / `int8` | `fp16` |
| `token_identity_scale_type` | Identity scale (per_layer scale) read type: `fp32` / `fp16` | `fp16` |

- The `embedding_dequant_scale` key is deprecated: the `embedding_weights`
  single file carries all scale segments; a leftover non-empty value only
  produces a WARN and does not affect parsing.
- Whether identity segments are read is decided by the deployment graph slots
  (the graph is authoritative); the two `token_identity_*` keys are still parsed
  independently when the main table is empty (tokenId form).

## Single-file four-segment layout

The file referenced by `embedding_weights` packs four segments contiguously
(row-major within each segment, no file header, no inter-segment padding);
zero-length segments are physically absent and later offsets shift forward:

```
┌──────────────────────────────┬ offset 0
│ seg1 embedding weight        │ L1 = vocab × hidden × elem(embedding_type)
├──────────────────────────────┬ L1
│ seg2 per_layer weight        │ L2 = vocab × W × elem(token_identity_type)
│   identity table [vocab, W]  │   W = product of graph slot dims (0 if disabled)
├──────────────────────────────┬ L1 + L2
│ seg3 weight scale            │ L3 = vocab × groups × elem(embedding_scale_type)
├──────────────────────────────┬ L1 + L2 + L3
│ seg4 per_layer scale         │ L4 = vocab × elem(token_identity_scale_type)
└──────────────────────────────┴ total = L1 + L2 + L3 + L4
```

When identity is enabled (graph carries the slots), the total file size must
equal the four-segment sum exactly (validated by the engine at load time).
