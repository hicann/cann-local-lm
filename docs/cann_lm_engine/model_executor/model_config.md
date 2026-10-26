该文件指定model_executor模块的配置参数输入结构：

example:

```json
{
    "model_config": {
        "hidden_size": 2048,
        "max_io_tokens": 16384,
        "model_type": "gemma4",
        "num_hidden_layers": "1",
        "prefill_len": [
            64
        ],
        "decode_len": [
            1
        ],
        "pad_token_id": 0,
        "vocab_size": 153376,
        "vocab_align_size": 153376,
        "kv_cache_max_len": 1024,
        "kv_cache_format": "nd",
        "num_attention_kv_heads": 8,
        "num_attention_head_dims": 128,
        "batch_num": 1,
        "kv_cache_type": "fp16"
    }
}
```
