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
    from cann_lm.converter.export.ge_converter import register_ge_converters

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

def save_epair_prefill_decode(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    prefill_seq_len: int = 128,
    use_embeddings: bool = False,
    save_weights: bool = True,
) -> None:
    """Prefill/decode split export: emit two graphs (prefill + decode) + shared weights.

    Output layout (under output_dir):
    - ``prefill_ep_json_with_ascend_ir.json`` — prefill graph (seq_len=prefill_seq_len)
    - ``decode_ep_json_with_ascend_ir.json`` — decode graph (seq_len=1, reusable)
    - ``weight_info.json`` + ``weights_*.bin`` — shared weights (packed once)
    - ``model_config.json`` + tokenizer files

    At inference time:
    - prefill: pad input_ids to prefill_seq_len, mask pad positions,
      output_pos=[prompt_len-1] to select the last real token's logits
    - decode: seq_len=1, reuse decode graph, feed previous-step KV cache

    Does NOT save .pt2 or input_N.bin (those belong in a verify script).
    """
    import torch_delegate
    from cann_lm.converter.export.ge_converter import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    register_ge_converters()

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

    # Export prefill_ep (seq_len=prefill_seq_len, shape-specialized)
    prefill_sample = wrapper.get_sample_inputs(
        seq_len=prefill_seq_len, use_embeddings=use_embeddings
    )
    prefill_ep = torch.export.export(wrapper, prefill_sample, strict=False)
    prefill_epair = torch_delegate.export_to_epair(prefill_ep)
    del prefill_ep

    # Export decode_ep (seq_len=1, reused for all decode steps)
    decode_sample = wrapper.get_sample_inputs(seq_len=1, use_embeddings=use_embeddings)
    decode_ep = torch.export.export(wrapper, decode_sample, strict=False)
    decode_epair = torch_delegate.export_to_epair(decode_ep)

    # Release model memory (OOM guard); epair.ep holds independent ExportedProgram ref
    del decode_ep, wrapper
    del artifact.model
    import gc; gc.collect()

    # Write weights + prefill ep_json (save_weights=True, only once)
    consolidated_weights_save(
        prefill_epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        ep_json_filename="prefill_ep_json_with_ascend_ir.json",
    )

    # Write decode ep_json only (save_weights=False, weights already packed above)
    consolidated_weights_save(
        decode_epair, output_dir,
        save_weights=False,
        ep_json_filename="decode_ep_json_with_ascend_ir.json",
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)


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
