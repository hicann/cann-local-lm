该文件指定embedding模块的配置参数输入结构：

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
