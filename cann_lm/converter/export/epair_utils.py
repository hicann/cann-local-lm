"""Utilities for packing consolidated weights into binary files + index."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

DEFAULT_SPLIT_SIZE = 1024 * 1024 * 1024 * 50 # 50GB

@dataclasses.dataclass
class ConsolidatedWeights:
    """Packed consolidated weights in memory.

    All data is held in memory — use ``consolidated_weights_save()`` to write
    to disk when needed.

    Attributes:
        bin_files: List of packed weight binary data. Multiple files are
            created when tensor data exceeds ``max_file_size``.
        weights_index: Mapping from bin file name (e.g. ``"weights_0.bin"``)
            to a list of ``{"name": str, "offset": int, "size": int}`` entries
            describing each tensor's location within that file.
    """
    bin_files: list[bytes]
    weights_index: dict[str, list[dict]]


def pack_weights(contents, *, max_file_size: int = DEFAULT_SPLIT_SIZE) -> ConsolidatedWeights:
    """Pack weights from EpairContents into ConsolidatedWeights (in memory).

    Reads tensor data from ``contents.ep.state_dict()``, concatenates raw
    tensor blobs into one or more bin files, and builds a ``weights_index``
    mapping each bin file name to its tensor entries::

        {"weights_0.bin": [
            {"name": "<param_name>", "offset": <byte_offset>, "size": <bytes>}
        ]}

    Only tensors referenced in the ``ExportedProgram`` graph signature
    (parameters, buffers, and lifted tensor constants) are included.

    When the accumulated data in the current bin file would exceed
    ``max_file_size``, a new bin file is started. Individual tensors are
    never split across files — if a single tensor is larger than
    ``max_file_size``, it occupies its own file.

    Args:
        contents: EpairContents object containing model and weights.
        max_file_size: Maximum size (in bytes) of each bin file before
            splitting to the next file. Defaults to 5 KB for testing.

    Returns:
        ConsolidatedWeights with ``bin_files`` and ``weights_index``.
    """
    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)

    # Non-persistent buffers: add from ep.module().named_buffers() (uses module path names)
    for name, buf in contents.ep.module().named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    # Lifted tensor constants: add from ep.constants (uses ep_json tensor_constant_name)
    if hasattr(contents.ep, 'constants') and isinstance(contents.ep.constants, dict):
        for name, tensor in contents.ep.constants.items():
            if name not in state_dict and isinstance(tensor, torch.Tensor):
                state_dict[name] = tensor

    spec_names = set()
    gs = contents.ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for placeholder, logical_name in mapping.items():
            if logical_name in state_dict:
                spec_names.add(logical_name)

    names = [n for n in state_dict if n in spec_names]

    # Direct-to-disk writing (avoid accumulating all weights in memory)
    index: dict[str, list[dict]] = {}

    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    total_written = 0

    # Open first file for writing
    bin_path = None

    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()
        del raw  # Free tensor reference

        # If adding this tensor would exceed the limit, start a new file
        if current_offset > 0 and current_offset + len(data) > max_file_size:
            if bin_path is not None:
                bin_path.close()
            current_file_idx += 1
            current_file_name = f"weights_{current_file_idx}.bin"
            current_offset = 0

        # Open file if not yet open
        if current_offset == 0:
            import os
            bin_path = open(os.path.join(".", current_file_name), "wb")
            index[current_file_name] = []

        index[current_file_name].append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
        })

        bin_path.write(data)
        current_offset += len(data)
        total_written += len(data)
        del data  # Free bytes immediately

    # Close last file
    if bin_path is not None:
        bin_path.close()

    num_files = current_file_idx + 1
    print(f"Packed {len(index)} tensors into {num_files} file(s) ({total_written / 1024:.0f} KB)")


def consolidated_weights_save(contents, output_dir: str | Path, *,
                              save_weights: bool = True,
                              max_file_size: int = 5 * 1024,
                              ep_json_filename: str = "ep_json_with_ascend_ir.json") -> None:
    """Write epair contents to disk in consolidated weights format.

    Writes files directly under ``output_dir`` (no ``data/`` subdirectory):

    - ``weights_0.bin``, ``weights_1.bin``, ... — raw concatenated tensor bytes
    - ``weight_info.json`` — index describing each tensor and its bin file
    - ``<ep_json_filename>`` — annotated EP JSON (default ``ep_json_with_ascend_ir.json``)

    Weight files are written directly to disk (streaming, not in-memory)
    to support large models (>8B params) without OOM.

    Args:
        contents: EpairContents object containing model and weights.
        output_dir: Directory to write files into.
        save_weights: If False, skip writing ``weights_*.bin`` and
            ``weight_info.json``.
        max_file_size: Maximum size (in bytes) of each bin file before
            splitting. Defaults to 5 KB for testing.
        ep_json_filename: Filename for the annotated EP JSON. Defaults to
            ``ep_json_with_ascend_ir.json``. Use a custom name for
            prefill/decode split export (e.g.
            ``prefill_ep_json_with_ascend_ir.json``).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_weights:
        weights_index = _pack_weights_to_dir(contents, output_dir, max_file_size)

        idx_path = output_dir / "weight_info.json"
        with open(idx_path, "w") as f:
            json.dump(weights_index, f, indent=2)
        print(f"Saved: {idx_path}")
    else:
        print("  [skip] weights_*.bin + weight_info.json (save_weights=False)")

    ep_json_path = output_dir / ep_json_filename
    _strip_none_input_descs(contents.ep_json)
    with open(ep_json_path, "w") as f:
        json.dump(contents.ep_json, f, indent=2)
    print(f"Saved: {ep_json_path}")


def _strip_none_input_descs(ep_json: dict) -> None:
    """Remove inputDesc entries with dtype=None from QuantBatchMatmulV4 ascend_ir ops.

    QuantBatchMatmulV4's optional inputs (bias, y_scale, x1_offset, ...) are
    emitted as inputDesc with dtype=None when not provided. The C++ GE graph
    builder creates GE inputs for ALL inputDesc, including None-dtype ones.
    The InferShape pass then fails on the None bias (shape=[] → dimension mismatch).

    Only QuantBatchMatmulV4 is affected — other ops (Cast, Mul, etc.) must keep
    their inputDesc intact (stripping those breaks the C++ parser).
    """
    graph = ep_json.get("graph_module", {}).get("graph", {})
    for node in graph.get("nodes", []):
        ai = node.get("ascend_ir")
        if not isinstance(ai, dict):
            continue
        for op in ai.get("ops", []) or []:
            if not isinstance(op, dict):
                continue
            if op.get("name", "") != "QuantBatchMatmulV4":
                continue
            desc = op.get("inputDesc")
            inp = op.get("input")
            if isinstance(desc, list) and isinstance(inp, list) and len(desc) == len(inp):
                # Only strip entries that are BOTH dtype=None AND input ref is empty
                keep = [(d, r) for d, r in zip(desc, inp)
                        if not (d.get("dtype") is None and (r is None or r == ""))]
                op["inputDesc"] = [d for d, _ in keep]
                op["input"] = [r for _, r in keep]


def _pack_weights_to_dir(contents, output_dir: Path, max_file_size: int) -> dict[str, list[dict]]:
    """Pack weights from EpairContents directly to disk in output_dir.

    Streaming write — never accumulates more than one tensor in memory.
    """
    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)

    # Non-persistent buffers: add from ep.module().named_buffers()
    for name, buf in contents.ep.module().named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    # Lifted tensor constants: add from ep.constants
    if hasattr(contents.ep, 'constants') and isinstance(contents.ep.constants, dict):
        for name, tensor in contents.ep.constants.items():
            if name not in state_dict and isinstance(tensor, torch.Tensor):
                state_dict[name] = tensor

    # Filter to only graph-signature-referenced tensors
    spec_names = set()
    gs = contents.ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for placeholder, logical_name in mapping.items():
            if logical_name in state_dict:
                spec_names.add(logical_name)

    names = [n for n in state_dict if n in spec_names]

    index: dict[str, list[dict]] = {}
    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    total_written = 0
    bin_fp = None

    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()
        del raw

        if current_offset > 0 and current_offset + len(data) > max_file_size:
            if bin_fp is not None:
                bin_fp.close()
            current_file_idx += 1
            current_file_name = f"weights_{current_file_idx}.bin"
            current_offset = 0

        if current_offset == 0:
            bin_fp = open(output_dir / current_file_name, "wb")
            index[current_file_name] = []

        index[current_file_name].append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
        })

        bin_fp.write(data)
        current_offset += len(data)
        total_written += len(data)
        del data

    if bin_fp is not None:
        bin_fp.close()

    print(f"Packed {len(index)} tensors into {current_file_idx + 1} file(s) "
          f"({total_written / 1024:.0f} KB)")
    return index
