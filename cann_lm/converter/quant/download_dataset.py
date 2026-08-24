"""Download and format calibration datasets as JSONL files.

Examples:
    python cann_lm/converter/quant/download_dataset.py ultrachat_200k \
        --output-dir ./calibration_data
    python cann_lm/converter/quant/download_dataset.py c4 \
        --output-file /data/calibrate_c4.jsonl
    python cann_lm/converter/quant/download_dataset.py bfclv4 \
        --bfcl-dir /data/bfcl --output-dir /data/calibration

The output directory defaults to ``./calibration_data``. Hugging Face's default
cache is used unless ``--cache-dir`` is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


# These variables must be set before importing datasets. setdefault preserves
# values explicitly configured by the caller.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


DEFAULT_OUTPUT_DIR = Path("calibration_data")
DEFAULT_BFCL_DIR = Path("bfcl")
DEFAULT_COIG_FILE = Path("COIG-CQIA") / "COIG-CQIA-full.jsonl"


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Import datasets lazily so that ``--help`` works without the package."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it before downloading datasets."
        ) from exc
    return load_dataset(*args, **kwargs)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _validate_num_samples(num_samples: int) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")


def _prepare_output_path(output_path: str | os.PathLike[str]) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_cache_dir(cache_dir: str | os.PathLike[str] | None) -> str | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _sample(dataset: Any, num_samples: int, seed: int) -> Any:
    _validate_num_samples(num_samples)
    return dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))


def _write_dataset(
    dataset: Any, output_path: str | os.PathLike[str], **kwargs: Any
) -> Path:
    path = _prepare_output_path(output_path)
    dataset.to_json(str(path), **kwargs)
    return path


def UltraChat_200k(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR
    / "calibrate_ultrachat.jsonl",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 8192,
    seed: int = 42,
) -> Path:
    dataset = _load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        cache_dir=_prepare_cache_dir(cache_dir),
    )

    def format_text(example: dict[str, Any]) -> dict[str, str]:
        prompt = example.get("prompt", "")
        messages = example.get("messages", [])
        parts = [prompt, ""]
        parts.extend(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages
        )
        return {"text": "\n".join(parts).strip()}

    train_dataset = _sample(dataset["train_sft"], num_samples, seed)
    train_dataset = train_dataset.map(
        format_text,
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(train_dataset, output_path)


def FineWeb_Edu(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR
    / "calibrate_fineweb_edu.jsonl",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 512,
    seed: int = 42,
) -> Path:
    dataset = _load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=f"train[:{num_samples}]",
        cache_dir=_prepare_cache_dir(cache_dir),
    )
    train_dataset = _sample(dataset, num_samples, seed)
    train_dataset = train_dataset.map(
        lambda example: {"text": example["text"]},
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(train_dataset, output_path)


def ToolBench(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR
    / "calibrate_toolbench.jsonl",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 8192,
    seed: int = 42,
) -> Path:
    dataset = _load_dataset(
        "tuandunghcmut/toolbench-v1",
        split="train",
        cache_dir=_prepare_cache_dir(cache_dir),
    )

    def format_toolbench(example: dict[str, Any]) -> dict[str, str]:
        conversations = example.get("conversations")
        if isinstance(conversations, dict):
            roles = conversations.get("from")
            values = conversations.get("value")
            if isinstance(roles, list) and isinstance(values, list):
                text = "\n".join(
                    f"{role}: {content}" for role, content in zip(roles, values)
                )
                return {"text": text.strip()}
        return {"text": str(example)}

    train_dataset = _sample(dataset, num_samples, seed)
    train_dataset = train_dataset.map(
        format_toolbench,
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(train_dataset, output_path)


def BFCLv4(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR / "calibrate_bfclv4.jsonl",
    *,
    bfcl_dir: str | os.PathLike[str] = DEFAULT_BFCL_DIR,
    num_samples: int = 8192,
    seed: int = 42,
) -> Path:
    _validate_num_samples(num_samples)
    source_dir = Path(bfcl_dir).expanduser()
    json_files = (
        "BFCL_v3_simple.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json",
    )

    all_data: list[dict[str, Any]] = []
    loaded_files: list[Path] = []
    for filename in json_files:
        source_file = source_dir / filename
        if not source_file.is_file():
            continue
        loaded_files.append(source_file)
        with source_file.open(encoding="utf-8") as file:
            all_data.extend(json.loads(line) for line in file if line.strip())

    if not loaded_files:
        raise FileNotFoundError(
            f"No BFCL source files were found in '{source_dir}'. "
            "Pass the correct directory with --bfcl-dir."
        )

    def format_bfcl(example: dict[str, Any]) -> dict[str, str]:
        question = example.get("question", [])
        functions = example.get("function", [])
        if not question:
            return {"text": str(example)}

        parts: list[str] = []
        for turn in question:
            if not isinstance(turn, list):
                continue
            for message in turn:
                if isinstance(message, dict) and "content" in message:
                    parts.append(f"{message.get('role', 'user')}: {message['content']}")

        question_text = "\n".join(parts)
        text = f"Question: {question_text}"
        if functions:
            text += f"\nFunctions: {json.dumps(functions, ensure_ascii=False)}"
        return {"text": text.strip()}

    random.Random(seed).shuffle(all_data)
    formatted = (format_bfcl(example) for example in all_data[:num_samples])
    path = _prepare_output_path(output_path)
    with path.open("w", encoding="utf-8") as file:
        for item in formatted:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def C4(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR / "calibrate_c4.jsonl",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 8192,
    seed: int = 42,
) -> Path:
    _validate_num_samples(num_samples)
    dataset = _load_dataset(
        "allenai/c4",
        name="en",
        split="train",
        streaming=True,
        cache_dir=_prepare_cache_dir(cache_dir),
    )
    train_dataset = dataset.take(num_samples).shuffle(seed=seed)
    train_dataset = train_dataset.map(
        lambda example: {"text": example["text"]},
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(train_dataset, output_path)


def COIG_CQIA(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR
    / "calibrate_coig_cqia.jsonl",
    *,
    input_file: str | os.PathLike[str] = DEFAULT_COIG_FILE,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 2048,
    seed: int = 42,
) -> Path:
    source_file = Path(input_file).expanduser()
    if not source_file.is_file():
        raise FileNotFoundError(
            f"COIG-CQIA source file '{source_file}' does not exist. "
            "Pass the correct file with --coig-file."
        )

    dataset = _load_dataset(
        "json",
        data_files=str(source_file),
        split="train",
        cache_dir=_prepare_cache_dir(cache_dir),
    )

    def format_text(example: dict[str, Any]) -> dict[str, str]:
        text = f"instruction: {example.get('instruction', '')}\n"
        if example.get("input"):
            text += f"input: {example['input']}\n"
        text += f"output: {example.get('output', '')}"
        return {"text": text.strip()}

    train_dataset = _sample(dataset, num_samples, seed)
    train_dataset = train_dataset.map(
        format_text,
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(
        train_dataset,
        output_path,
        force_ascii=False,
        lines=True,
        orient="records",
    )


def mC4(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR / "calibrate_c4_zh.jsonl",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int = 8192,
    seed: int = 42,
) -> Path:
    _validate_num_samples(num_samples)
    dataset = _load_dataset(
        "allenai/c4",
        name="zh",
        split="train",
        streaming=True,
        cache_dir=_prepare_cache_dir(cache_dir),
    )
    train_dataset = dataset.take(num_samples).shuffle(seed=seed)
    train_dataset = train_dataset.map(
        lambda example: {"text": example["text"]},
        remove_columns=train_dataset.column_names,
    )
    return _write_dataset(
        train_dataset,
        output_path,
        force_ascii=False,
        lines=True,
        orient="records",
    )


@dataclass(frozen=True)
class DatasetDefinition:
    handler: Callable[..., Path]
    output_filename: str
    default_num_samples: int


DATASETS = {
    "ultrachat_200k": DatasetDefinition(
        UltraChat_200k, "calibrate_ultrachat.jsonl", 8192
    ),
    "fineweb_edu": DatasetDefinition(FineWeb_Edu, "calibrate_fineweb_edu.jsonl", 512),
    "toolbench": DatasetDefinition(ToolBench, "calibrate_toolbench.jsonl", 8192),
    "bfclv4": DatasetDefinition(BFCLv4, "calibrate_bfclv4.jsonl", 8192),
    "c4": DatasetDefinition(C4, "calibrate_c4.jsonl", 8192),
    "coig_cqia": DatasetDefinition(COIG_CQIA, "calibrate_coig_cqia.jsonl", 2048),
    "mc4": DatasetDefinition(mC4, "calibrate_c4_zh.jsonl", 8192),
}


def download_dataset(
    dataset_name: str,
    *,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    output_file: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    num_samples: int | None = None,
    seed: int = 42,
    bfcl_dir: str | os.PathLike[str] = DEFAULT_BFCL_DIR,
    coig_file: str | os.PathLike[str] = DEFAULT_COIG_FILE,
) -> Path:
    """Download one dataset and return the generated JSONL path."""
    try:
        definition = DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose one of: {supported}"
        ) from exc

    sample_count = (
        definition.default_num_samples if num_samples is None else num_samples
    )
    _validate_num_samples(sample_count)
    target = (
        Path(output_file).expanduser()
        if output_file is not None
        else Path(output_dir).expanduser() / definition.output_filename
    )
    kwargs: dict[str, Any] = {
        "output_path": target,
        "num_samples": sample_count,
        "seed": seed,
    }
    if dataset_name == "bfclv4":
        kwargs["bfcl_dir"] = bfcl_dir
    else:
        kwargs["cache_dir"] = cache_dir
    if dataset_name == "coig_cqia":
        kwargs["input_file"] = coig_file

    return definition.handler(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and format a calibration dataset as JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=sorted(DATASETS), help="dataset to prepare")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the generated JSONL file",
    )
    output_group.add_argument(
        "--output-file",
        type=Path,
        help="complete path of the generated JSONL file",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory; uses the Hugging Face default if omitted",
    )
    parser.add_argument(
        "--num-samples",
        type=_positive_int,
        help=(
            "number of output samples; defaults to 512 for FineWeb-Edu, "
            "2048 for COIG-CQIA, and 8192 otherwise"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    parser.add_argument(
        "--bfcl-dir",
        type=Path,
        default=DEFAULT_BFCL_DIR,
        help="directory containing the BFCL JSON source files",
    )
    parser.add_argument(
        "--coig-file",
        type=Path,
        default=DEFAULT_COIG_FILE,
        help="path to COIG-CQIA-full.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output_path = download_dataset(
        args.dataset,
        output_dir=args.output_dir,
        output_file=args.output_file,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        bfcl_dir=args.bfcl_dir,
        coig_file=args.coig_file,
    )
    print(f"Dataset written to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
