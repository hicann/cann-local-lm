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

    bin_files: list[bytes] = []
    index: dict[str, list[dict]] = {}

    current_bin = bytearray()
    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0

    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()

        # If current bin is non-empty and adding this tensor would exceed the
        # limit, start a new file. Individual tensors are never split.
        if current_offset > 0 and current_offset + len(data) > max_file_size:
            bin_files.append(bytes(current_bin))
            current_bin = bytearray()
            current_file_idx += 1
            current_file_name = f"weights_{current_file_idx}.bin"
            current_offset = 0

        index.setdefault(current_file_name, []).append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
        })

        current_bin.extend(data)
        current_offset += len(data)

    # Flush the last file
    if current_bin:
        bin_files.append(bytes(current_bin))

    total = sum(len(b) for b in bin_files)
    print(f"Packed {len(index)} tensors into {len(bin_files)} file(s) ({total / 1024:.0f} KB)")
    return ConsolidatedWeights(
        bin_files=bin_files,
        weights_index=index,
    )


def consolidated_weights_save(contents, output_dir: str | Path, *,
                              save_weights: bool = True,
                              max_file_size: int = 5 * 1024) -> None:
    """Write epair contents to disk in consolidated weights format.

    Writes files directly under ``output_dir`` (no ``data/`` subdirectory):

    - ``weights_0.bin``, ``weights_1.bin``, ... — raw concatenated tensor bytes
    - ``weight_info.json`` — index describing each tensor and its bin file
    - ``ep_json_with_ascend_ir.json`` — annotated EP JSON

    ``ep_json_with_ascend_ir.json`` is always written. Weight files are
    written only when ``save_weights=True``. When the total weight data
    exceeds ``max_file_size``, it is split across multiple ``weights_N.bin``
    files. Individual tensors are never split.

    Args:
        contents: EpairContents object containing model and weights.
        output_dir: Directory to write files into.
        save_weights: If False, skip writing ``weights_*.bin`` and
            ``weight_info.json``.
        max_file_size: Maximum size (in bytes) of each bin file before
            splitting. Defaults to 5 KB for testing.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_weights:
        cw = pack_weights(contents, max_file_size=max_file_size)

        for idx, bin_data in enumerate(cw.bin_files):
            bin_path = output_dir / f"weights_{idx}.bin"
            bin_path.write_bytes(bin_data)
            print(f"Saved: {bin_path} ({len(bin_data) / 1024:.0f} KB)")

        idx_path = output_dir / "weight_info.json"
        with open(idx_path, "w") as f:
            json.dump(cw.weights_index, f, indent=2)
        print(f"Saved: {idx_path}")
    else:
        print("  [skip] weights_*.bin + weight_info.json (save_weights=False)")

    ep_json_path = output_dir / "ep_json_with_ascend_ir.json"
    with open(ep_json_path, "w") as f:
        json.dump(contents.ep_json, f, indent=2)
    print(f"Saved: {ep_json_path}")
