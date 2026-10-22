#!/usr/bin/env python3
"""Validate dopt GPTQ ST output without loading model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


class VerificationError(RuntimeError):
    """Raised when an ST output contract is not satisfied."""


def _load_mapping(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"missing {kind}: {path}")
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise VerificationError(f"invalid {kind}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{kind} must contain a mapping: {path}")
    return value


def _require_nonempty(path: Path, kind: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise VerificationError(f"missing or empty {kind}: {path}")


def _verify_runtime_config(output_dir: Path) -> Path:
    config_path = output_dir / "config.yaml"
    config = _load_mapping(config_path, "archived optimize config")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise VerificationError("config.yaml: dataset must be a mapping")
    train_files = dataset.get("train_files")
    if (
        not isinstance(train_files, list)
        or not train_files
        or not all(isinstance(item, str) for item in train_files)
    ):
        raise VerificationError(
            "config.yaml: dataset.train_files must be a non-empty string list"
        )
    ptq_samples = dataset.get("ptq_samples")
    if (
        not isinstance(ptq_samples, int)
        or isinstance(ptq_samples, bool)
        or ptq_samples < 1
    ):
        raise VerificationError(
            "config.yaml: dataset.ptq_samples must be a positive integer"
        )
    cutoff_len = config.get("cutoff_len")
    if (
        not isinstance(cutoff_len, int)
        or isinstance(cutoff_len, bool)
        or cutoff_len < 1
    ):
        raise VerificationError("config.yaml: cutoff_len must be a positive integer")
    return config_path


def _verify_dopt_config(output_dir: Path) -> tuple[Path, int, dict[str, int]]:
    config_path = output_dir / "dopt_config.json"
    config = _load_mapping(config_path, "dopt layer strategy")
    strategies = config.get("layer_strategy")
    if not isinstance(strategies, dict) or not strategies:
        raise VerificationError(
            "dopt_config.json: layer_strategy must be a non-empty mapping"
        )
    strategy_counts: Counter[str] = Counter()
    for layer_name, strategy in strategies.items():
        if not isinstance(layer_name, str) or not isinstance(strategy, dict):
            raise VerificationError(
                "dopt_config.json: every layer strategy must be a mapping"
            )
        if (
            not isinstance(strategy.get("quant_strategy"), str)
            or not strategy["quant_strategy"]
        ):
            raise VerificationError(
                f"dopt_config.json: layer {layer_name!r} has no quant_strategy"
            )
        strategy_name = strategy["quant_strategy"].split(";", maxsplit=1)[0]
        if strategy_name == "Quant_act_weight_eco":
            raise VerificationError(
                f"dopt_config.json: layer {layer_name!r} still uses legacy strategy "
                "Quant_act_weight_eco"
            )
        strategy_counts[strategy_name] += 1
    if strategy_counts["Quant_INT8-4_ECO"] == 0:
        raise VerificationError(
            "dopt_config.json: no Linear layer uses Quant_INT8-4_ECO"
        )
    return config_path, len(strategies), dict(sorted(strategy_counts.items()))


def _hf_weight_files(output_dir: Path) -> list[Path]:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return sorted(
        {
            path
            for pattern in patterns
            for path in output_dir.glob(pattern)
            if path.is_file()
        }
    )


def _verify_quant_weights(output_dir: Path, save_format: str) -> list[Path]:
    state_dict = output_dir / "trained_quant_weight.pth"
    hf_weights = _hf_weight_files(output_dir)
    if save_format == "state-dict":
        _require_nonempty(state_dict, "quantized state dict")
        return [state_dict]
    if save_format == "hf":
        if not hf_weights:
            raise VerificationError(
                f"missing HuggingFace quantized weights under {output_dir}"
            )
        for path in hf_weights:
            _require_nonempty(path, "HuggingFace quantized weight")
        _require_nonempty(output_dir / "config.json", "HuggingFace model config")
        return hf_weights
    if state_dict.is_file():
        _require_nonempty(state_dict, "quantized state dict")
        return [state_dict]
    if hf_weights:
        for path in hf_weights:
            _require_nonempty(path, "HuggingFace quantized weight")
        _require_nonempty(output_dir / "config.json", "HuggingFace model config")
        return hf_weights
    raise VerificationError(f"no quantized weight artifact found under {output_dir}")


def verify_output(output_dir: Path, save_format: str = "auto") -> dict[str, Any]:
    """Validate configs and quantized artifacts and return a compact report."""
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise VerificationError(f"output directory does not exist: {output_dir}")
    config_path = _verify_runtime_config(output_dir)
    dopt_config_path, strategy_count, strategy_counts = _verify_dopt_config(output_dir)
    weight_files = _verify_quant_weights(output_dir, save_format)
    return {
        "output_dir": str(output_dir),
        "config": config_path.name,
        "dopt_config": dopt_config_path.name,
        "strategy_count": strategy_count,
        "strategy_counts": strategy_counts,
        "weight_files": [path.name for path in weight_files],
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-format", choices=("auto", "state-dict", "hf"), default="auto"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_output(args.output_dir, args.save_format)
    except VerificationError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("[PASS] dopt quantization output contract satisfied")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
