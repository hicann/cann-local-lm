"""Saver — export model to epair format with tokenizer and config."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import torch

from cann_lm.converter.loader import ModelArtifact
from cann_lm.converter.export.epair_utils import consolidated_weights_save


def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 512 * 1024 * 1024,
    sample_seq_len: int = 128,
    save_weights: bool = True,
) -> None:
    import torch_delegate

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_inputs = create_sample_inputs(artifact, sample_seq_len)

    ep = torch.export.export(artifact.model, sample_inputs, strict=False)

    epair = torch_delegate.export_to_epair(ep)

    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)

    save_model_config(artifact, output_dir)


def create_sample_inputs(artifact: ModelArtifact, seq_len: int = 128) -> tuple:
    config = artifact.config
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else 128000
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64)
    return (input_ids,)


def save_model_config(artifact: ModelArtifact, output_dir: Path) -> None:
    config_obj = artifact.config
    if hasattr(config_obj, "to_dict"):
        config_dict = config_obj.to_dict()
    elif dataclasses.is_dataclass(config_obj):
        config_dict = dataclasses.asdict(config_obj)
    else:
        config_dict = dict(config_obj.__dict__)

    config_data = {
        "model_type": artifact.model_type,
        "config": config_dict,
        "hf_config": artifact.hf_config,
    }
    path = output_dir / "model_config.json"
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2, default=str)
    print(f"Saved: {path}")


def copy_tokenizer_files(files: list[Path], output_dir: Path) -> None:
    for fp in files:
        dest = output_dir / fp.name
        shutil.copy2(fp, dest)
        print(f"Saved: {dest}")
