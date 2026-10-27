English | [简体中文](../../../zh/cann_lm_engine/embedding/embedding_config.md)

This file specifies the configuration parameter input structure for the embedding module:

example:

```json
{
    "embedding_config": {
        "embedding_weights": "gemma4.embedding_weights",
        "embedding_dequant_scale": "gemma4.embedding_dequant_scale",
        "embedding_quant_mode": "per_token",
        "embedding_load_mode": "in_memory",
        "embedding_type": "fp16",
        "embedding_scale_type": "fp16"
    }
}
```
