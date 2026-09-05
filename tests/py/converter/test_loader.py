"""loader.py 单元测试 — ModelArtifact/build_config/collect_tokenizer_files (纯 dict/文件 ops).

测: ModelArtifact dataclass 字段, build_config 从 hf_config + ModelEntry 构造 config
(字段过滤 / config_json_key 嵌套 / fallback 顶层 / __post_init__), collect_tokenizer_files
用假 HF 目录验返回 Path 列表 (有/无 tokenizer.json 两种分支).
跳过: load_model 全流程 (需真实 HF 权重/safetensors + model_cls 实例化, 太重).
"""
import dataclasses
import json
from pathlib import Path

import pytest
import torch.nn as nn

from cann_lm.converter.loader import (
    ModelArtifact,
    build_config,
    collect_tokenizer_files,
)
from cann_lm.models.model_registry import REGISTRY, ModelEntry


# ── ModelArtifact dataclass ──

def test_model_artifact_fields():
    fields = {f.name for f in dataclasses.fields(ModelArtifact)}
    assert fields == {
        "model", "config", "hf_path", "hf_config", "tokenizer_files",
        "model_type", "wrapper_cls", "quant_info",
    }


def test_model_artifact_quant_info_defaults_none():
    art = ModelArtifact(
        model=nn.Linear(1, 1), config=object(), hf_path="/x", hf_config={},
        tokenizer_files=[], model_type="qwen3", wrapper_cls=None,
    )
    assert art.quant_info is None


# ── build_config: 字段过滤 (qwen3, config_json_key=None → 用顶层 hf_config) ──

def test_build_config_qwen3_filters_unknown_keys():
    entry = REGISTRY["qwen3"]
    hf_config = {
        "model_type": "qwen3", "hidden_size": 512, "num_hidden_layers": 2,
        "vocab_size": 100, "junk_field": "should_be_dropped",
    }
    config = build_config(hf_config, entry)

    assert config.model_type == "qwen3"
    assert config.hidden_size == 512
    assert config.num_hidden_layers == 2
    assert config.vocab_size == 100
    assert not hasattr(config, "junk_field")


def test_build_config_qwen3_uses_defaults_for_missing_fields():
    entry = REGISTRY["qwen3"]
    # 只给 model_type, 其余字段用 config_cls 默认值
    config = build_config({"model_type": "qwen3"}, entry)
    assert config.hidden_size == 1024  # Qwen3TextConfig 默认
    assert config.num_hidden_layers == 28


# ── build_config: config_json_key 嵌套 (gemma4, key="text_config") ──

def test_build_config_gemma4_reads_nested_text_config():
    entry = REGISTRY["gemma4"]
    hf_config = {
        "model_type": "gemma4",
        "text_config": {"hidden_size": 256, "num_hidden_layers": 2},
    }
    config = build_config(hf_config, entry)

    assert config.hidden_size == 256
    assert config.num_hidden_layers == 2
    assert config.model_type == "gemma4_text"


def test_build_config_gemma4_post_init_ran():
    entry = REGISTRY["gemma4"]
    config = build_config({"text_config": {"num_hidden_layers": 2}}, entry)
    assert config.layer_types is not None
    assert len(config.layer_types) == 2
    assert config.layer_types[-1] == "full_attention"  # post_init 保证最后一层 full
    assert config.rope_parameters is not None


def test_build_config_gemma4_falls_back_to_top_level_when_key_absent():
    entry = REGISTRY["gemma4"]
    # config_json_key="text_config" 但 hf_config 无该键 → fallback 用顶层 hf_config
    config = build_config({"hidden_size": 100, "num_hidden_layers": 2}, entry)
    assert config.hidden_size == 100


def test_build_config_qwen3_5_reads_nested_text_config():
    entry = REGISTRY["qwen3_5"]
    config = build_config(
        {"model_type": "qwen3_5", "text_config": {"hidden_size": 64, "num_hidden_layers": 1}},
        entry,
    )
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 1


# ── collect_tokenizer_files ──

def _touch(base: Path, names):
    for n in names:
        (base / n).write_text("{}")


def test_collect_tokenizer_files_with_tokenizer_json(tmp_path):
    _touch(tmp_path, ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json"])
    files = collect_tokenizer_files(str(tmp_path))
    names = {f.name for f in files}
    # 有 tokenizer.json → 只收集 3 个标准 pattern (不含 optional)
    assert names == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    assert all(isinstance(f, Path) for f in files)


def test_collect_tokenizer_files_missing_some_standard(tmp_path):
    _touch(tmp_path, ["tokenizer.json", "special_tokens_map.json"])  # 缺 tokenizer_config
    files = collect_tokenizer_files(str(tmp_path))
    names = {f.name for f in files}
    assert names == {"tokenizer.json", "special_tokens_map.json"}


def test_collect_tokenizer_files_fallback_to_legacy_when_no_tokenizer_json(tmp_path):
    # 无 tokenizer.json → 追加 optional patterns (tokenizer.model / vocab.json / merges.txt)
    _touch(tmp_path, ["tokenizer_config.json", "special_tokens_map.json",
                      "tokenizer.model", "vocab.json"])
    files = collect_tokenizer_files(str(tmp_path))
    names = {f.name for f in files}
    assert names == {"tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.model", "vocab.json"}


def test_collect_tokenizer_files_empty_dir(tmp_path):
    assert collect_tokenizer_files(str(tmp_path)) == []
