"""gen_model_json.py 单元测试 — _extract_from_hf_config / gen_model_json (纯 dict ops).

测: _extract_from_hf_config 从 hf_config 提取字段 (正常/缺字段报错/eos_multi_turn 列表/layer_types 长度警告),
gen_model_json 生成 model.json 字典 (结构/默认值/参数覆盖/路径参数/写死值覆盖/eos_multi_turn).
跳过: saver.save_epair* 内的调用 (需 torch.export + torchair GE, 归 e2e).
"""

import json

import pytest

from cann_lm.converter.gen_model_json import _extract_from_hf_config, gen_model_json


# ── 测试用 hf_config 构造 ──


def _make_hf_config(
    *,
    hidden_size=2560,
    model_type="gemma4",
    num_hidden_layers=42,
    vocab_size=262144,
    num_key_value_heads=2,
    global_head_dim=512,
    layer_types=None,
    bos_token_id=1,
    eos_token_id=106,
    top_eos_token_id=None,
    pad_token_id=0,
):
    """构造一个最小可用的 hf_config dict。"""
    if layer_types is None:
        # 5 sliding + 1 full 循环, 共 42 层
        layer_types = []
        for i in range(num_hidden_layers):
            layer_types.append(
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            )
    cfg = {
        "model_type": model_type,
        "text_config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "vocab_size": vocab_size,
            "num_key_value_heads": num_key_value_heads,
            "global_head_dim": global_head_dim,
            "layer_types": layer_types,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        },
    }
    if top_eos_token_id is not None:
        cfg["eos_token_id"] = top_eos_token_id
    return cfg


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
    )
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["hidden_size"] == 512
    assert cfg["num_hidden_layers"] == 4
    assert cfg["vocab_size"] == 32000
    assert cfg["num_attention_kv_heads"] == 8
    assert cfg["num_attention_head_dims"] == 64
    assert cfg["model_type"] == "gemma4"
    assert cfg["pad_token_id"] == 0


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


def test_extract_eos_multi_turn_from_list():
    hf_config = _make_hf_config(top_eos_token_id=[1, 106])
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] == 106


def test_extract_eos_multi_turn_none_when_single_element_list():
    """top_eos_token_id=[1] 长度 1, 不满足 len > 1 → eos_multi_turn=None。"""
    hf_config = _make_hf_config(top_eos_token_id=[1])
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


def test_extract_eos_multi_turn_none_when_not_list():
    hf_config = _make_hf_config(top_eos_token_id=106)
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


def test_extract_eos_multi_turn_none_when_absent():
    hf_config = _make_hf_config()
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["eos_multi_turn"] is None


def test_extract_pad_token_id_default_zero():
    hf_config = _make_hf_config()
    hf_config["text_config"].pop("pad_token_id")
    cfg = _extract_from_hf_config(hf_config)
    assert cfg["pad_token_id"] == 0


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
    assert "WARN" in err
    assert "2" in err and "4" in err


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
    assert mc["sliding_window"] == 256
    # layer_types
    assert len(mc["layer_types"]) == 42


def test_gen_model_json_embedding_config_defaults(tmp_path):
    path = _write_config(tmp_path, _make_hf_config())
    ec = gen_model_json(path)["embedding_config"]
    assert ec["embedding_weights"] == ""
    assert ec["embedding_dequant_scale"] == ""
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
    assert tc["special_tokens"] == {}


def test_gen_model_json_eos_multi_turn_in_special_tokens(tmp_path):
    hf_config = _make_hf_config(top_eos_token_id=[1, 106])
    path = _write_config(tmp_path, hf_config)
    tc = gen_model_json(path)["tokenizer_config"]
    assert tc["special_tokens"]["eos_multi_turn"] == 106


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
        embedding_dequant_scale_path="/device/scale.bin",
        tokenizer_file_path="/device/tokenizer.json",
    )
    assert result["embedding_config"]["embedding_weights"] == "/device/embed.bin"
    assert result["embedding_config"]["embedding_dequant_scale"] == "/device/scale.bin"
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
    path = _write_config(tmp_path, _make_hf_config())
    mc = gen_model_json(path, sliding_window=512)["model_config"]
    assert mc["sliding_window"] == 512


# ── gen_model_json: 文件不存在 ──


def test_gen_model_json_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gen_model_json(str(tmp_path / "nonexistent.json"))
