"""gen_model_json.py 单元测试 — _extract_from_hf_config / gen_model_json (纯 dict ops).

测: _extract_from_hf_config 从 hf_config 提取字段 (正常/缺必填字段报错/
sliding_window+head_dim 提取/缺 sliding_window 或 head_dim 时静默返回 None/layer_types 长度警告/
eos_multi_turn 从顶层 eos_token_id 列表第二元素提取, 缺失或非列表或长度 1 时为 None),
gen_model_json 生成 model.json 字典 (结构/默认值/参数覆盖/路径参数/写死值覆盖/
special_tokens 为 {"<turn|>": eos_multi_turn_id} 映射/embedding_dequant_scale 键废弃/
sliding_window 与 sliding_window_head_dim 缺失时输出 null).
跳过: saver.save_epair* 内的调用 (需 torch.export + torchair GE, 归 e2e).
"""

import json

import pytest

from cann_local_lm.converter.gen_model_json import (
    _extract_from_hf_config,
    gen_model_json,
)


# ── 测试用 hf_config 构造 ──


def _make_hf_config(
    *,
    hidden_size=2560,
    model_type="gemma4",
    num_hidden_layers=42,
    vocab_size=262144,
    num_key_value_heads=2,
    global_head_dim=512,
    head_dim=256,
    sliding_window=512,
    layer_types=None,
    bos_token_id=1,
    eos_token_id=106,
    pad_token_id=0,
    top_eos_token_id=None,
):
    """构造一个最小可用的 hf_config dict (匹配 gemma4 真实 config 字段)。

    top_eos_token_id: 顶层 eos_token_id 列表; None 时使用默认 [1, 106] (第二元素
    即 eos_multi_turn id 106)。传入其他值则覆盖顶层 eos_token_id。
    """
    if layer_types is None:
        # 5 sliding + 1 full 循环, 共 42 层
        layer_types = []
        for i in range(num_hidden_layers):
            layer_types.append(
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            )
    if top_eos_token_id is None:
        top_eos_token_id = [1, 106]
    return {
        "model_type": model_type,
        "eos_token_id": top_eos_token_id,
        "text_config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "vocab_size": vocab_size,
            "num_key_value_heads": num_key_value_heads,
            "global_head_dim": global_head_dim,
            "head_dim": head_dim,
            "sliding_window": sliding_window,
            "layer_types": layer_types,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        },
    }


def _write_config(tmp_path, hf_config):
    """把 hf_config 写到 tmp_path/config.json, 返回路径字符串。"""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(hf_config))
    return str(p)


# ── _extract_from_hf_config: 正常提取 ──


def test_extract_returns_all_expected_fields():
    hf_config = _make_hf_config()
    cfg = _extract_from_hf_config(hf_config)
    assert set(cfg.keys()) == {
        "hidden_size",
        "model_type",
        "num_hidden_layers",
        "pad_token_id",
        "vocab_size",
        "num_attention_kv_heads",
        "num_attention_head_dims",
        "layer_types",
        "sliding_window",
        "sliding_window_head_dim",
        "bos_token_id",
        "eos_token_id",
        "eos_multi_turn",
    }


def test_extract_field_values_match_config():
    hf_config = _make_hf_config(
        hidden_size=512,
        num_hidden_layers=4,
        vocab_size=32000,
        num_key_value_heads=8,
        global_head_dim=64,
        head_dim=128,
        sliding_window=1024,
    )
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["hidden_size"] == 512
    assert cfg["num_hidden_layers"] == 4
    assert cfg["vocab_size"] == 32000
    assert cfg["num_attention_kv_heads"] == 8
    assert cfg["num_attention_head_dims"] == 64
    assert cfg["model_type"] == "gemma4"
    assert cfg["pad_token_id"] == 0


def test_extract_sliding_window_from_text_config():
    """sliding_window 从 text_config.sliding_window 提取 (原写死 256 已删除)。"""
    hf_config = _make_hf_config(sliding_window=768)
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["sliding_window"] == 768


def test_extract_sliding_window_head_dim_from_head_dim():
    """sliding_window_head_dim 从 text_config.head_dim 提取 (区别于 global_head_dim)。"""
    hf_config = _make_hf_config(head_dim=256, global_head_dim=512)
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["sliding_window_head_dim"] == 256
    assert cfg["num_attention_head_dims"] == 512


def test_extract_layer_types_correct():
    layer_types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    hf_config = _make_hf_config(num_hidden_layers=4, layer_types=layer_types)
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["layer_types"] == layer_types


def test_extract_pad_token_id_default_zero():
    hf_config = _make_hf_config()
    hf_config["text_config"].pop("pad_token_id")
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["pad_token_id"] == 0


# ── _extract_from_hf_config: eos_multi_turn 提取 ──


def test_extract_eos_multi_turn_from_top_level_list():
    """顶层 eos_token_id=[1, 106] → eos_multi_turn 取第二元素 106。"""
    hf_config = _make_hf_config(top_eos_token_id=[1, 106])
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] == 106


def test_extract_eos_multi_turn_single_element_list_is_none():
    """顶层 eos_token_id 单元素列表 [1] → eos_multi_turn 为 None。"""
    hf_config = _make_hf_config(top_eos_token_id=[1])
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


def test_extract_eos_multi_turn_non_list_is_none():
    """顶层 eos_token_id 非列表 (int) → eos_multi_turn 为 None。"""
    hf_config = _make_hf_config(top_eos_token_id=106)
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


def test_extract_eos_multi_turn_missing_is_none():
    """顶层 eos_token_id 缺失 → eos_multi_turn 为 None。"""
    hf_config = _make_hf_config()
    hf_config.pop("eos_token_id")
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


# ── _extract_from_hf_config: 缺字段报错 ──


@pytest.mark.parametrize(
    "missing_key",
    [
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_key_value_heads",
        "global_head_dim",
        "layer_types",
        "bos_token_id",
        "eos_token_id",
    ],
)
def test_extract_missing_text_config_field_raises(missing_key):
    hf_config = _make_hf_config()
    hf_config["text_config"].pop(missing_key)
    with pytest.raises(ValueError, match="config.json 缺少必填字段"):
        _extract_from_hf_config(hf_config)


def test_extract_missing_model_type_raises():
    hf_config = _make_hf_config()
    hf_config.pop("model_type")
    with pytest.raises(ValueError, match="model_type"):
        _extract_from_hf_config(hf_config)


def test_extract_missing_text_config_section_raises():
    """text_config 整个 key 缺失 → 所有字段 None → 报错。"""
    hf_config = {"model_type": "gemma4"}
    with pytest.raises(ValueError, match="config.json 缺少必填字段"):
        _extract_from_hf_config(hf_config)


# ── _extract_from_hf_config: layer_types 长度警告 ──


def test_extract_layer_types_length_mismatch_warns(capsys):
    hf_config = _make_hf_config(
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention"],  # 长度 2 != 4
    )
    _extract_from_hf_config(hf_config)
    err = capsys.readouterr().err
    assert "[WARN] layer_types 长度(2) != num_hidden_layers(4)" in err


# ── gen_model_json: 结构与默认值 ──


def test_gen_model_json_returns_three_sections(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    result = gen_model_json(path)
    assert set(result.keys()) == {
        "model_config",
        "embedding_config",
        "tokenizer_config",
    }


def test_gen_model_json_model_config_defaults(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path)["model_config"]
    # [config] 提取的字段
    assert mc["hidden_size"] == 2560
    assert mc["model_type"] == "gemma4"
    assert mc["num_hidden_layers"] == 42
    assert mc["vocab_size"] == 262144
    assert mc["vocab_align_size"] == 262144  # = vocab_size
    assert mc["num_attention_kv_heads"] == 2
    assert mc["num_attention_head_dims"] == 512
    assert mc["pad_token_id"] == 0
    # [param] 默认值
    assert mc["prefill_len"] == [64]
    assert mc["decode_len"] == [1]
    assert mc["kv_cache_max_len"] == 2048
    assert mc["batch_num"] == 1
    # [fixed] 写死值
    assert mc["max_io_tokens"] == 8192
    assert mc["kv_cache_format"] == "nd"
    assert mc["kv_cache_type"] == "fp16"
    # [config] sliding_window 从 text_config.sliding_window 提取 (不再写死 256)
    assert mc["sliding_window"] == 512
    assert mc["sliding_window_head_dim"] == 256
    # layer_types
    assert len(mc["layer_types"]) == 42


def test_gen_model_json_embedding_config_defaults(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    ec = gen_model_json(path)["embedding_config"]
    assert ec["embedding_weights"] == ""
    assert "embedding_dequant_scale" not in ec  # 键已废弃
    assert ec["embedding_quant_mode"] == "per_token"
    assert ec["embedding_load_mode"] == "in_memory"
    assert ec["embedding_type"] == "int8"
    assert ec["embedding_scale_type"] == "fp32"
    assert ec["token_identity_type"] == "int8"
    assert ec["token_identity_scale_type"] == "fp32"


def test_gen_model_json_tokenizer_config_defaults(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    tc = gen_model_json(path)["tokenizer_config"]
    assert tc["tokenizer_file"] == ""
    assert tc["bos_token_id"] == 1
    assert tc["eos_token_id"] == 106
    assert tc["bos_token"] == "<bos>"
    assert tc["eos_token"] == "<eos>"
    assert tc["add_bos"] is True
    # special_tokens 的 key 是固定 token 字符串 "<turn|>", value 是 eos_multi_turn id
    assert tc["special_tokens"] == {"<turn|>": 106}


def test_gen_model_json_eos_multi_turn_token_key_maps_to_id(tmp_path):
    """special_tokens 为 {"<turn|>": eos_multi_turn_id}, id 取自顶层 eos_token_id 列表第二元素。"""
    path = _write_config(tmp_path, _make_hf_config(top_eos_token_id=[1, 106]))
    tc = gen_model_json(path)["tokenizer_config"]
    assert tc["special_tokens"] == {"<turn|>": 106}


def test_gen_model_json_eos_multi_turn_missing_maps_to_none(tmp_path):
    """顶层 eos_token_id 缺失 → special_tokens 值为 None, key 仍为 "<turn|>"。"""
    hf_config = _make_hf_config()
    hf_config.pop("eos_token_id")
    path = _write_config(tmp_path, hf_config)
    tc = gen_model_json(path)["tokenizer_config"]
    assert tc["special_tokens"] == {"<turn|>": None}


# ── gen_model_json: 缺字段容错 ──


def test_gen_model_json_sliding_window_none_when_missing(tmp_path, capsys):
    """text_config.sliding_window 缺失 → model_config 输出 None, 且不打印 WARN。"""
    hf_config = _make_hf_config()
    hf_config["text_config"].pop("sliding_window")
    path = _write_config(tmp_path, hf_config)
    mc = gen_model_json(path)["model_config"]
    assert "sliding_window" in mc
    assert mc["sliding_window"] is None
    assert "WARN" not in capsys.readouterr().err


def test_gen_model_json_sliding_window_head_dim_none_when_missing(tmp_path, capsys):
    """text_config.head_dim 缺失 → 键恒存在且值为 None, 且不打印 WARN。"""
    hf_config = _make_hf_config()
    hf_config["text_config"].pop("head_dim")
    path = _write_config(tmp_path, hf_config)
    mc = gen_model_json(path)["model_config"]
    assert "sliding_window_head_dim" in mc
    assert mc["sliding_window_head_dim"] is None
    assert "WARN" not in capsys.readouterr().err


# ── gen_model_json: 参数覆盖 ──


def test_gen_model_json_seq_len_overrides_prefill(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, seq_len=128)["model_config"]
    assert mc["prefill_len"] == [128]


def test_gen_model_json_decode_len_overrides(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, decode_len=4)["model_config"]
    assert mc["decode_len"] == [4]


def test_gen_model_json_kv_cache_max_len_overrides(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, kv_cache_max_len=4096)["model_config"]
    assert mc["kv_cache_max_len"] == 4096


def test_gen_model_json_batch_num_overrides(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, batch_num=3)["model_config"]
    assert mc["batch_num"] == 3


# ── gen_model_json: 路径参数 ──


def test_gen_model_json_path_params_reflected(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    result = gen_model_json(
        path,
        embedding_weights_path="/device/embed.bin",
        tokenizer_file_path="/device/tokenizer.json",
    )
    assert result["embedding_config"]["embedding_weights"] == "/device/embed.bin"
    assert result["tokenizer_config"]["tokenizer_file"] == "/device/tokenizer.json"


# ── gen_model_json: 写死值覆盖 ──


def test_gen_model_json_max_io_tokens_override(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, max_io_tokens=4096)["model_config"]
    assert mc["max_io_tokens"] == 4096


def test_gen_model_json_kv_cache_format_override(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, kv_cache_format="nz")["model_config"]
    assert mc["kv_cache_format"] == "nz"


def test_gen_model_json_kv_cache_type_override(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, kv_cache_type="bf16")["model_config"]
    assert mc["kv_cache_type"] == "bf16"


def test_gen_model_json_sliding_window_override(tmp_path):
    """参数 sliding_window 优先于 config 的 text_config.sliding_window=512。"""
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, sliding_window=1024)["model_config"]
    assert mc["sliding_window"] == 1024


# ── gen_model_json: 文件不存在 ──


def test_gen_model_json_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gen_model_json(str(tmp_path / "nonexistent.json"))
