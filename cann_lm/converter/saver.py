"""Saver — export model to epair format with tokenizer and config."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import torch

from cann_lm.converter.loader import ModelArtifact
from cann_lm.converter.export.epair_utils import consolidated_weights_save


def _remove_assert_nodes(ep_json: dict) -> int:
    """从 ep_json 中移除 _assert_tensor_metadata 断言节点。"""
    nodes = ep_json["graph_module"]["graph"]["nodes"]
    before = len(nodes)
    ep_json["graph_module"]["graph"]["nodes"] = [
        n for n in nodes if "assert" not in n.get("target", "").lower()
    ]
    removed = before - len(ep_json["graph_module"]["graph"]["nodes"])
    if removed > 0:
        print(f"[Strip] 移除了 {removed} 个 _assert_tensor_metadata 节点")
    return removed


def _remove_none_inputs(ep_json: dict) -> int:
    """从 ep_json 中移除 as_none 的输入引用。

    当 wrapper 传 past_key_values=None 或 embeddings=None 时，
    ep_json 的 graph.inputs 和 input_specs 中会记录 as_none。
    C++ EpairModelParser 不支持 as_none 输入类型。
    """
    removed = 0
    # 1. 从 graph.inputs 移除 as_none 条目
    graph = ep_json.get("graph_module", {}).get("graph", {})
    inputs = graph.get("inputs", [])
    filtered_inputs = [inp for inp in inputs if not (len(inp) == 1 and "as_none" in inp)]
    removed += len(inputs) - len(filtered_inputs)
    graph["inputs"] = filtered_inputs

    # 2. 从 input_specs 移除 as_none 的 constant_input
    specs = ep_json.get("graph_module", {}).get("signature", {}).get("input_specs", [])
    filtered_specs = []
    for spec in specs:
        kind = list(spec.keys())[0]
        detail = spec[kind]
        if kind == "constant_input":
            value = detail.get("value", {}) if isinstance(detail, dict) else {}
            if isinstance(value, dict) and value.get("as_none") is True:
                removed += 1
                continue
        filtered_specs.append(spec)
    ep_json["graph_module"]["signature"]["input_specs"] = filtered_specs

    if removed > 0:
        print(f"[Strip] 移除了 {removed} 个 as_none 输入引用")
    return removed


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

    _remove_assert_nodes(epair.ep_json)
    _remove_none_inputs(epair.ep_json)

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
            inp.cpu().detach().numpy().tofile(path)
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
