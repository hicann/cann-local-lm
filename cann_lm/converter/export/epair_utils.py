"""Utilities for packing consolidated weights into a single binary + index."""

from __future__ import annotations

import json
from pathlib import Path


def pack_weights(
    model_dir: str | Path,
    output_bin: str | Path | None = None,
    output_json: str | Path | None = None,
) -> tuple[Path, Path]:
    """Pack individual weight/constant tensor files into a single binary.

    Reads ``data/weights/model_weights_config.json`` and ``data/constants/model_constants_config.json``
    from *model_dir*, concatenates all raw tensor blobs into *output_bin*, and writes a
    ``data/weights_index.json`` with entries::

        {"name": "<param_name>", "size": <bytes>, "offset": <byte_offset>}

    After packing, the old ``data/weights/`` and ``data/constants/`` directories are removed.

    Args:
        model_dir: The top-level model directory (contains ``data/``).
        output_bin: Output binary path (default: ``<model_dir>/data/weights.bin``).
        output_json: Output JSON index path (default: ``<model_dir>/data/weights_index.json``).

    Returns:
        (output_bin, output_json) paths.
    """
    model_dir = Path(model_dir).resolve()
    data_dir = model_dir / "data"

    if output_bin is None:
        output_bin = data_dir / "weights.bin"
    if output_json is None:
        output_json = data_dir / "weights_index.json"
    output_bin = Path(output_bin).resolve()
    output_json = Path(output_json).resolve()
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    import shutil

    index = []
    offset = 0

    with open(output_bin, "wb") as bin_f:
        for sub_dir, config_name in [("weights", "model_weights_config.json"),
                                     ("constants", "model_constants_config.json")]:
            config_path = data_dir / sub_dir / config_name
            if not config_path.is_file():
                continue

            with open(config_path) as f:
                config = json.load(f)

            for param_name, meta in config["config"].items():
                file_path = data_dir / sub_dir / meta["path_name"]
                if not file_path.is_file():
                    raise FileNotFoundError(f"Tensor file not found: {file_path} (for {param_name})")

                data = file_path.read_bytes()
                bin_f.write(data)
                index.append({
                    "name": param_name,
                    "size": len(data),
                    "offset": offset,
                })
                offset += len(data)

    with open(output_json, "w") as f:
        json.dump(index, f, indent=2)

    # Remove old weight/constant directories
    for sub_dir in ["weights", "constants"]:
        shutil.rmtree(data_dir / sub_dir, ignore_errors=True)

    print(f"Packed {len(index)} tensors → {output_bin} ({offset / 1024:.0f} KB)")
    print(f"Index → {output_json}")
    return output_bin, output_json
