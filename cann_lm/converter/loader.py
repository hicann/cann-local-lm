"""Model loader — load HF weights into custom model definitions."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cann_lm.converter.model_registry import ModelEntry, detect_model_type


@dataclasses.dataclass
class ModelArtifact:
    model: nn.Module
    config: Any
    hf_path: str
    hf_config: dict
    tokenizer_files: list[Path]
    model_type: str
    wrapper_cls: type  # ExportWrapper subclass (passed through from ModelEntry)
    quant_info: dict | None = None  # quant metadata (scheme/group_size/scope/...), None=unquantized


def load_model(
    hf_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ModelArtifact:
    config_json_path = os.path.join(hf_path, "config.json")
    with open(config_json_path) as f:
        hf_config = json.load(f)

    entry = detect_model_type(hf_config)
    config = build_config(hf_config, entry)

    model = entry.model_cls(config)
    model = model.to(dtype=dtype).to(device=device)
    model.eval()
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    tokenizer_files = collect_tokenizer_files(hf_path)

    return ModelArtifact(
        model=model,
        config=config,
        hf_path=hf_path,
        hf_config=hf_config,
        tokenizer_files=tokenizer_files,
        model_type=entry.model_type,
        wrapper_cls=entry.wrapper_cls,
    )


def build_config(hf_config: dict, entry: ModelEntry):
    if entry.config_json_key in hf_config:
        raw = hf_config[entry.config_json_key]
    else:
        raw = hf_config

    valid_fields = {f.name for f in dataclasses.fields(entry.config_cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    config = entry.config_cls(**filtered)
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


_TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

_OPTIONAL_TOKENIZER_PATTERNS = [
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
]


def collect_tokenizer_files(hf_path: str) -> list[Path]:
    files = []
    patterns = _TOKENIZER_PATTERNS
    # fall back to legacy tokenizer files when tokenizer.json is absent
    if not (Path(hf_path) / "tokenizer.json").exists():
        patterns = patterns + _OPTIONAL_TOKENIZER_PATTERNS
    for p in patterns:
        fp = Path(hf_path) / p
        if fp.exists():
            files.append(fp)
    return files
