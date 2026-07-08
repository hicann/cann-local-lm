"""Utilities for extracting .epair archives and packing weights."""

from __future__ import annotations

import json
import shutil
import tempfile
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


def _resolve_model_root(extract_dir: Path) -> Path:
    """Return the model root directory inside an extracted epair."""
    candidate = extract_dir / "simple_transformer"
    if candidate.is_dir():
        return candidate
    # Fall back to the extract dir itself
    return extract_dir


def pack_weights(
    extract_dir: str | Path,
    output_bin: str | Path = "weights.bin",
    output_json: str | Path = "weights_index.json",
) -> tuple[Path, Path]:
    """Pack all weight files from an extracted epair into a single binary file.

    Reads both ``model_weights_config.json`` and ``model_constants_config.json``,
    concatenates all raw tensor blobs into *output_bin*, and writes a single
    JSON index with entries::

        {"name": "<param_name>", "size": <bytes>, "offset": <byte_offset>}

    Args:
        extract_dir: Directory where the epair was extracted.
        output_bin: Output binary file (weights concatenated).
        output_json: Output JSON index file.

    Returns:
        (output_bin, output_json) paths.
    """
    extract_dir = Path(extract_dir).resolve()
    model_root = _resolve_model_root(extract_dir)

    output_bin = Path(output_bin).resolve()
    output_json = Path(output_json).resolve()
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    index = []
    offset = 0

    with open(output_bin, "wb") as bin_f:
        for sub_dir, config_name in [("weights", "model_weights_config.json"),
                                     ("constants", "model_constants_config.json")]:
            config_path = model_root / "data" / sub_dir / config_name
            if not config_path.is_file():
                continue

            with open(config_path) as f:
                config = json.load(f)

            for param_name, meta in config["config"].items():
                file_path = model_root / "data" / sub_dir / meta["path_name"]
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

    print(f"Packed {len(index)} tensors → {output_bin} ({offset / 1024:.0f} KB)")
    print(f"Index → {output_json}")
    return output_bin, output_json


def consolidate_epair(
    epair_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Consolidate an .epair by replacing individual weight/constant files
    with a single ``weights.bin`` + ``weights_index.json``.

    The output archive keeps all other records (model JSON, extra files,
    metadata) intact.

    Args:
        epair_path: Path to the original .epair file.
        output_path: Output path for the consolidated .epair.
            Defaults to overwriting the original.

    Returns:
        Path to the consolidated .epair.
    """
    epair_path = Path(epair_path).resolve()
    if not epair_path.is_file():
        raise FileNotFoundError(f"epair file not found: {epair_path}")

    if output_path is None:
        output_path = epair_path
    output_path = Path(output_path).resolve()

    with tempfile.TemporaryDirectory(prefix="epair_consolidate_") as tmp:
        # 1. Extract
        extract_epair(epair_path, tmp)
        model_root = _resolve_model_root(Path(tmp))

        # 2. Pack weights + constants into single bin + json
        data_dir = model_root / "data"
        bin_path = data_dir / "weights.bin"
        idx_path = data_dir / "weights_index.json"
        pack_weights(tmp, bin_path, idx_path)

        # 3. Remove old weight/constant directories
        for sub_dir in ["weights", "constants"]:
            shutil.rmtree(data_dir / sub_dir, ignore_errors=True)

        # 4. Rewrite .epair
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_dir(model_root, output_path)
        print(f"Consolidated: {output_path}")

    return output_path


def _zip_dir(source_dir: Path, output_zip: Path) -> None:
    """Create a ZIP archive from a directory, preserving the subdirectory
    as the archive root."""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_STORED) as z:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = str(file_path.relative_to(source_dir.parent))
                z.write(file_path, arcname)
