"""saver.py 单元测试 — save_model_config / copy_tokenizer_files (纯文件 ops).

测: save_model_config 用 stub ModelArtifact 写 model_config.json
(to_dict / dataclass / __dict__ 三条 config 路径 + use_embeddings / quant_embeddings / quant_info),
copy_tokenizer_files 用 tmp_path 造源文件列表验复制.
跳过: save_epair / save_epair_prefill_decode (需 torch.export 的 wrapper/ep_module +
      torchair GE 转换, 太重).
"""

import dataclasses
import json

import torch.nn as nn

from cann_local_lm.converter.loader import ModelArtifact
from cann_local_lm.converter.saver import copy_tokenizer_files, save_model_config


@dataclasses.dataclass
class _ToDictConfig:
    hidden_size: int = 64

    def to_dict(self) -> dict:
        return {"hidden_size": self.hidden_size}


@dataclasses.dataclass
class _PlainDataclassConfig:
    hidden_size: int = 32
    num_layers: int = 2


class _DictConfig:
    def __init__(self):
        self.hidden_size = 16


def _artifact(config, *, model_type="qwen3_5", quant_info=None, hf_config=None):
    return ModelArtifact(
        model=nn.Linear(1, 1),
        config=config,
        hf_path="/x",
        hf_config=hf_config if hf_config is not None else {"model_type": "qwen3_5"},
        tokenizer_files=[],
        model_type=model_type,
        wrapper_cls=None,
        quant_info=quant_info,
    )


# ── save_model_config: config 路径 ──


def test_save_model_config_uses_to_dict(tmp_path):
    save_model_config(_artifact(_ToDictConfig()), tmp_path)
    data = json.loads((tmp_path / "model_config.json").read_text())
    assert data["model_type"] == "qwen3_5"
    assert data["config"] == {"hidden_size": 64}
    assert data["hf_config"] == {"model_type": "qwen3_5"}
    assert data["use_embeddings"] is False
    assert data["quant_embeddings"] is False
    assert "quant_info" not in data  # quant_info=None → 不写


def test_save_model_config_uses_dataclass_asdict(tmp_path):
    save_model_config(_artifact(_PlainDataclassConfig()), tmp_path)
    data = json.loads((tmp_path / "model_config.json").read_text())
    assert data["config"] == {"hidden_size": 32, "num_layers": 2}


def test_save_model_config_uses_dict_fallback(tmp_path):
    save_model_config(_artifact(_DictConfig()), tmp_path)
    data = json.loads((tmp_path / "model_config.json").read_text())
    assert data["config"] == {"hidden_size": 16}


# ── save_model_config: flags ──


def test_save_model_config_use_embeddings_flag(tmp_path):
    save_model_config(
        _artifact(_ToDictConfig()), tmp_path, use_embeddings=True, quant_embeddings=True
    )
    data = json.loads((tmp_path / "model_config.json").read_text())
    assert data["use_embeddings"] is True
    assert data["quant_embeddings"] is True


def test_save_model_config_writes_quant_info_when_present(tmp_path):
    qi = {"scheme": "W4A8-s8s4", "group_size": 128, "bit": 4}
    save_model_config(_artifact(_ToDictConfig(), quant_info=qi), tmp_path)
    data = json.loads((tmp_path / "model_config.json").read_text())
    assert data["quant_info"] == qi


# ── copy_tokenizer_files ──


def test_copy_tokenizer_files_copies_all(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_files = []
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        f = src_dir / name
        f.write_text("payload-" + name)
        src_files.append(f)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    copy_tokenizer_files(src_files, out_dir)

    for f in src_files:
        dest = out_dir / f.name
        assert dest.exists()
        assert dest.read_text() == f.read_text()


def test_copy_tokenizer_files_empty_list_no_error(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copy_tokenizer_files([], out_dir)  # 空列表不报错, 不写任何文件
    assert not any(out_dir.iterdir())
