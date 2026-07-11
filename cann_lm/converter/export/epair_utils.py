"""Utilities for packing consolidated weights into a single binary + index."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ConsolidatedWeights:
    """Packed consolidated weights in memory.

    All data is held in memory — use ``consolidated_weights_save()`` to write
    to disk when needed.

    Attributes:
        weights_bin: Concatenated raw tensor bytes (``weights.bin`` content).
        weights_index: List of ``{"name": str, "offset": int, "size": int}``
            entries describing each tensor's location in ``weights_bin``.
    """
    weights_bin: bytes
    weights_index: list[dict]


def pack_weights(contents) -> ConsolidatedWeights:
    """Pack weights from EpairContents into ConsolidatedWeights (in memory).

    Reads tensor data from ``contents.ep.state_dict()``, concatenates all raw
    tensor blobs into ``weights_bin``, and builds a ``weights_index`` with
    entries::

        {"name": "<param_name>", "size": <bytes>, "offset": <byte_offset>}

    Only tensors referenced in the ``ExportedProgram`` graph signature
    (parameters, buffers, and lifted tensor constants) are included.

    Args:
        contents: EpairContents object containing model and weights.

    Returns:
        ConsolidatedWeights with ``weights_bin`` and ``weights_index``.
    """
    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)

    spec_names = set()
    gs = contents.ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for placeholder, logical_name in mapping.items():
            if logical_name in state_dict:
                spec_names.add(logical_name)

    names = sorted(spec_names)

    weights_bin = bytearray()
    index = []
    offset = 0
    for name in names:
        tensor = state_dict[name]
        data = tensor.detach().cpu().numpy().tobytes()
        weights_bin.extend(data)
        index.append({
            "name": name,
            "size": len(data),
            "offset": offset,
        })
        offset += len(data)

    print(f"Packed {len(index)} tensors ({offset / 1024:.0f} KB)")
    return ConsolidatedWeights(
        weights_bin=bytes(weights_bin),
        weights_index=index,
    )


def consolidated_weights_save(contents, output_dir: str | Path, *, save_weights: bool = True) -> None:
    """Write epair contents to disk in consolidated weights format.

    Writes three files under ``output_dir/data/``:

    - ``weights.bin`` — raw concatenated tensor bytes
    - ``weights_index.json`` — index describing each tensor
    - ``ep_json_with_ascend_ir.json`` — annotated EP JSON

    ``ep_json_with_ascend_ir.json`` is always written. ``weights.bin`` and
    ``weights_index.json`` are written only when ``save_weights=True``.

    Args:
        contents: EpairContents object containing model and weights.
        output_dir: Directory to write ``data/`` subdirectory into.
        save_weights: If False, skip writing ``weights.bin`` and
            ``weights_index.json``.
    """
    output_dir = Path(output_dir).resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if save_weights:
        cw = pack_weights(contents)
        bin_path = data_dir / "weights.bin"
        bin_path.write_bytes(cw.weights_bin)

        idx_path = data_dir / "weights_index.json"
        with open(idx_path, "w") as f:
            json.dump(cw.weights_index, f, indent=2)

        print(f"Saved: {bin_path} ({len(cw.weights_bin) / 1024:.0f} KB)")
        print(f"Saved: {idx_path}")
    else:
        print("  [skip] weights.bin + weights_index.json (save_weights=False)")

    ep_json_path = data_dir / "ep_json_with_ascend_ir.json"
    with open(ep_json_path, "w") as f:
        json.dump(contents.ep_json, f, indent=2)

    print(f"Saved: {ep_json_path}")
