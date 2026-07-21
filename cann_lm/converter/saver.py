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
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    sample_seq_len: int = 128,
    use_embeddings: bool = False,
    save_weights: bool = True,
) -> None:
    import torch_delegate
    from cann_lm.models.fake_ops import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    register_ge_converters()

    # 从注册表获取 wrapper 子类，构造统一 I/O wrapper
    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/converter/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # 构造示例输入 (统一 I/O 契约)
    sample_inputs = wrapper.get_sample_inputs(seq_len=sample_seq_len, use_embeddings=use_embeddings)

    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    epair = torch_delegate.export_to_epair(ep)

    # C++ 侧已兜底处理：
    # - _assert_tensor_metadata 节点：PlanNodes 顶部 IsAssertTensorMetadataNode 跳过
    # - as_none 的 constant_input：PreparePlaceholders 里 NodeInputKind::None → continue 跳过
    # 无需 Python 侧 strip ep_json。

    # 释放原始模型内存 (大模型 OOM 防护)
    # epair.ep 持有 ExportedProgram 的独立引用, 删除 model/wrapper/ep 不影响 epair
    del ep, wrapper
    del artifact.model
    import gc; gc.collect()

    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)

    pt2_path = output_dir.parent / f"{output_dir.name}.pt2"
    torch_delegate.epair_save(epair, str(pt2_path))
    print(f"Saved: {pt2_path}")

    for i, inp in enumerate(sample_inputs):
        path = output_dir.parent / f"input_{i}.bin"
        if isinstance(inp, torch.Tensor):
            raw = inp.cpu().detach()
            if raw.dtype == torch.bfloat16:
                raw = raw.view(torch.int16)
            raw.numpy().tofile(path)
            print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")


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
