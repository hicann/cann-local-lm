"""Utilities for extracting .epair archives and packing weights."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def extract_epair(epair_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Extract an .epair ZIP archive to a directory.

    Args:
        epair_path: Path to the .epair file.
        output_dir: Output directory. Defaults to the .epair filename without suffix.

    Returns:
        Path to the extracted directory.
    """
    epair_path = Path(epair_path).resolve()
    if not epair_path.is_file():
        raise FileNotFoundError(f"epair file not found: {epair_path}")

    if output_dir is None:
        output_dir = epair_path.with_suffix("")
    output_dir = Path(output_dir).resolve()

    with zipfile.ZipFile(epair_path) as z:
        z.extractall(output_dir)

    # The ZIP root is a subdirectory named after the model
    model_root = output_dir / "simple_transformer"
    print(f"Extracted: {epair_path} → {output_dir}")
    print(f"Model root: {model_root}")
    return output_dir


def pack_weights(
    extract_dir: str | Path,
    output_bin: str | Path = "weights.bin",
    output_json: str | Path = "weights_index.json",
) -> tuple[Path, Path]:
    """Pack all weight files from an extracted epair into a single binary file.

    Reads ``model_weights_config.json`` inside the epair extract, concatenates
    raw weight blobs into *output_bin*, and writes *output_json* with entries::

        {"name": "<param_name>", "size": <bytes>, "offset": <byte_offset>}

    Args:
        extract_dir: Directory where the epair was extracted.
        output_bin: Output binary file (weights concatenated).
        output_json: Output JSON index file.

    Returns:
        (output_bin, output_json) paths.
    """
    extract_dir = Path(extract_dir).resolve()
    model_root = extract_dir / "simple_transformer"
    weights_dir = model_root / "data" / "weights"

    config_path = weights_dir / "model_weights_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    output_bin = Path(output_bin).resolve()
    output_json = Path(output_json).resolve()
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    index = []
    offset = 0

    # Config entries are in insertion order (Python 3.7+)
    with open(output_bin, "wb") as bin_f:
        for param_name, meta in config["config"].items():
            weight_name = meta["path_name"]
            weight_path = weights_dir / weight_name
            if not weight_path.is_file():
                raise FileNotFoundError(f"Weight file not found: {weight_path} (for {param_name})")

            data = weight_path.read_bytes()
            bin_f.write(data)

            index.append({
                "name": param_name,
                "size": len(data),
                "offset": offset,
            })
            offset += len(data)

    with open(output_json, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Packed {len(index)} weights → {output_bin} ({offset / 1024:.0f} KB)")
    print(f"Index → {output_json}")
    return output_bin, output_json
