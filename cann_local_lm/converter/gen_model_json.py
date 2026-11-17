"""从 HuggingFace config.json 生成 llm_engine 的 model.json。

字段来源说明：
  [config] = 从 HF config.json 的 text_config 提取
  [param]  = 从函数参数取（seq_len / kv_cache_max_len 等）
  [fixed]  = 写死（gemma4-E4B 已验证部署配置）
  [path]   = 部署设备路径，函数参数传入
"""

import json
import sys

# ── 写死的部署参数（gemma4-E4B 已验证配置）──────────────────────────
_FIXED_MAX_IO_TOKENS = 8192  # 输入输出最大长度（部署上限）
_FIXED_KV_CACHE_FORMAT = "nd"  # KV cache 布局
_FIXED_KV_CACHE_TYPE = (
    "fp16"  # KV cache 数据类型（部署用 fp16，不是 config 的 bfloat16）
)
_FIXED_EMBEDDING_QUANT_MODE = "per_token"
_FIXED_EMBEDDING_LOAD_MODE = "in_memory"
_FIXED_EMBEDDING_TYPE = "int8"
_FIXED_EMBEDDING_SCALE_TYPE = "fp32"
_FIXED_TOKEN_IDENTITY_TYPE = "int8"
_FIXED_TOKEN_IDENTITY_SCALE_TYPE = "fp32"

_FIXED_BOS_TOKEN = "<bos>"
_FIXED_EOS_TOKEN = "<eos>"
_FIXED_EOS_MULTI_TURN_TOKEN = "<turn|>"
_FIXED_ADD_BOS = True


def _extract_from_hf_config(hf_config: dict) -> dict:
    """从 HF config.json 提取能提取的字段。"""
    text_cfg = hf_config.get("text_config", {})

    hidden_size = text_cfg.get("hidden_size")
    model_type = hf_config.get("model_type")
    num_hidden_layers = text_cfg.get("num_hidden_layers")
    pad_token_id = text_cfg.get("pad_token_id", 0)
    vocab_size = text_cfg.get("vocab_size")
    num_kv_heads = text_cfg.get("num_key_value_heads")
    num_head_dims = text_cfg.get("global_head_dim")
    layer_types = text_cfg.get("layer_types", [])
    sliding_window = text_cfg.get("sliding_window")
    sliding_window_head_dim = text_cfg.get("head_dim")

    bos_token_id = text_cfg.get("bos_token_id")
    eos_token_id = text_cfg.get("eos_token_id")
    top_eos = hf_config.get("eos_token_id")
    eos_multi_turn = None
    if isinstance(top_eos, list) and len(top_eos) > 1:
        eos_multi_turn = top_eos[1]

    # 校验必填字段
    missing = []
    if hidden_size is None:
        missing.append("text_config.hidden_size")
    if model_type is None:
        missing.append("model_type")
    if num_hidden_layers is None:
        missing.append("text_config.num_hidden_layers")
    if vocab_size is None:
        missing.append("text_config.vocab_size")
    if num_kv_heads is None:
        missing.append("text_config.num_key_value_heads")
    if num_head_dims is None:
        missing.append("text_config.global_head_dim")
    if not layer_types:
        missing.append("text_config.layer_types")
    if bos_token_id is None:
        missing.append("text_config.bos_token_id")
    if eos_token_id is None:
        missing.append("text_config.eos_token_id")
    if missing:
        raise ValueError(f"config.json 缺少必填字段: {missing}")

    if len(layer_types) != num_hidden_layers:
        print(
            f"[WARN] layer_types 长度({len(layer_types)}) "
            f"!= num_hidden_layers({num_hidden_layers})",
            file=sys.stderr,
        )

    return {
        "hidden_size": hidden_size,
        "model_type": model_type,
        "num_hidden_layers": num_hidden_layers,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
        "num_attention_kv_heads": num_kv_heads,
        "num_attention_head_dims": num_head_dims,
        "layer_types": layer_types,
        "sliding_window": sliding_window,
        "sliding_window_head_dim": sliding_window_head_dim,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "eos_multi_turn": eos_multi_turn,
    }


def gen_model_json(
    hf_config_path: str,
    *,
    seq_len: int = 64,
    decode_len: int = 1,
    kv_cache_max_len: int = 2048,
    batch_num: int = 1,
    embedding_weights_path: str = "",
    tokenizer_file_path: str = "",
    max_io_tokens: int | None = None,
    kv_cache_format: str | None = None,
    kv_cache_type: str | None = None,
    sliding_window: int | None = None,
) -> dict:
    """从 HF config.json 生成 llm_engine model.json 字典。

    Args:
        hf_config_path: HuggingFace config.json 文件路径。
        seq_len: prefill 序列长度。
        decode_len: decode 序列长度，默认 1。
        kv_cache_max_len: KV cache 最大长度。
        batch_num: batch 数。
        embedding_weights_path: embedding 权重设备路径。
        tokenizer_file_path: tokenizer.json 设备路径。
        max_io_tokens / kv_cache_format / kv_cache_type: 可选覆盖写死值。
        sliding_window: 可选覆盖 config.json 的 text_config.sliding_window。

    Returns:
        dict with keys: model_config, embedding_config, tokenizer_config
    """
    with open(hf_config_path) as f:
        hf_config = json.load(f)

    cfg = _extract_from_hf_config(hf_config)

    model_config = {
        "hidden_size": cfg["hidden_size"],
        "max_io_tokens": max_io_tokens or _FIXED_MAX_IO_TOKENS,
        "model_type": cfg["model_type"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "prefill_len": [seq_len],
        "decode_len": [decode_len],
        "pad_token_id": cfg["pad_token_id"],
        "vocab_size": cfg["vocab_size"],
        "vocab_align_size": cfg["vocab_size"],
        "kv_cache_max_len": kv_cache_max_len,
        "kv_cache_format": kv_cache_format or _FIXED_KV_CACHE_FORMAT,
        "num_attention_kv_heads": cfg["num_attention_kv_heads"],
        "num_attention_head_dims": cfg["num_attention_head_dims"],
        "batch_num": batch_num,
        "kv_cache_type": kv_cache_type or _FIXED_KV_CACHE_TYPE,
        "sliding_window": sliding_window or cfg["sliding_window"],
        "sliding_window_head_dim": cfg["sliding_window_head_dim"],
        "layer_types": cfg["layer_types"],
    }

    embedding_config = {
        "embedding_weights": embedding_weights_path,
        "embedding_quant_mode": _FIXED_EMBEDDING_QUANT_MODE,
        "embedding_load_mode": _FIXED_EMBEDDING_LOAD_MODE,
        "embedding_type": _FIXED_EMBEDDING_TYPE,
        "embedding_scale_type": _FIXED_EMBEDDING_SCALE_TYPE,
        "token_identity_type": _FIXED_TOKEN_IDENTITY_TYPE,
        "token_identity_scale_type": _FIXED_TOKEN_IDENTITY_SCALE_TYPE,
    }

    tokenizer_config = {
        "tokenizer_file": tokenizer_file_path,
        "bos_token_id": cfg["bos_token_id"],
        "eos_token_id": cfg["eos_token_id"],
        "bos_token": _FIXED_BOS_TOKEN,
        "eos_token": _FIXED_EOS_TOKEN,
        "add_bos": _FIXED_ADD_BOS,
        "special_tokens": {
            _FIXED_EOS_MULTI_TURN_TOKEN: cfg["eos_multi_turn"],
        },
    }

    return {
        "model_config": model_config,
        "embedding_config": embedding_config,
        "tokenizer_config": tokenizer_config,
    }
