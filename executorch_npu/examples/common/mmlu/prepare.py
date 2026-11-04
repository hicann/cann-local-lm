# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Build shared batch-runner inputs from MMLU or multilingual MMMLU prompts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable

from .datasets import mmlu, mmmlu
from .protocol import choice_token_ids


def write_inputs(
    *, model: Path, output_dir: Path, max_length: int, prompts: Iterable[dict]
) -> None:
    from transformers import AutoTokenizer

    if max_length < 1:
        raise ValueError("max_length must be positive")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = choice_token_ids(tokenizer)
    (output_dir.parent / "choice_token_ids.txt").write_text(
        ",".join(map(str, ids)) + "\n", encoding="utf-8"
    )

    count = 0
    truncated_total = 0
    max_tokens = 0
    with (
        (output_dir / "manifest.tsv").open("w", encoding="utf-8") as manifest,
        (output_dir / "samples.jsonl").open("w", encoding="utf-8") as samples,
    ):
        for prompt_data in prompts:
            tokens = tokenizer.encode(prompt_data["prompt"], add_special_tokens=False)
            truncated = len(tokens) > max_length
            if truncated:
                tokens = tokens[-max_length:]
                truncated_total += 1
            max_tokens = max(max_tokens, len(tokens))
            sample_dir = output_dir / str(count)
            sample_dir.mkdir(parents=True, exist_ok=True)
            token_path = sample_dir / "tokens.bin"
            position_path = sample_dir / "positions.bin"
            token_path.write_bytes(struct.pack(f"<{len(tokens)}q", *tokens))
            position_path.write_bytes(
                struct.pack(f"<{len(tokens)}q", *range(len(tokens)))
            )
            sample = {"id": count}
            if "language" in prompt_data:
                sample["language"] = prompt_data["language"]
            sample.update(
                {
                    "subject": prompt_data["subject"],
                    "answer": prompt_data["answer"],
                    "input_tokens": len(tokens),
                    "left_truncated": truncated,
                    "prompt": prompt_data["prompt"],
                }
            )
            samples.write(json.dumps(sample) + "\n")
            manifest.write(f"{count} {token_path} {position_path} {len(tokens)}\n")
            count += 1
    print(f"samples={count} truncated={truncated_total} max_tokens={max_tokens}")


def main(default_model: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", "--model", dest="model_path", type=Path, default=default_model
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("mmlu", "mmmlu"), default="mmlu")
    parser.add_argument("--max-length", type=int, default=2047)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--source", choices=("parquet", "hf", "csv"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--subjects", type=Path, help="newline-separated MMLU subjects")
    parser.add_argument("--limit-per-subject", type=int, default=0)
    parser.add_argument(
        "--data-dir", type=Path, help="directory of MMMLU language CSV files"
    )
    parser.add_argument("--languages", help="comma-separated MMMLU languages")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="strided sample count per MMMLU language; 0 keeps all",
    )
    args = parser.parse_args()
    if args.num_fewshot < 0:
        parser.error("--num-fewshot must be nonnegative")

    if args.benchmark == "mmlu":
        source = args.source or "parquet"
        if source == "csv":
            parser.error("MMLU uses --source parquet or hf")
        if args.data_dir is not None or args.languages is not None or args.limit:
            parser.error("--data-dir, --languages, and --limit are MMMLU options")
        subjects = None
        if args.subjects is not None:
            subjects = [
                line.strip()
                for line in args.subjects.read_text().splitlines()
                if line.strip()
            ]
        prompts = mmlu.iter_prompts(
            source=source,
            snapshot=args.snapshot,
            task_dir=args.task_dir,
            subjects=subjects,
            limit_per_subject=args.limit_per_subject,
            num_fewshot=args.num_fewshot,
        )
    else:
        if args.source not in (None, "csv"):
            parser.error("MMMLU uses CSV inputs: --source csv")
        if args.data_dir is None or not args.languages:
            parser.error("MMMLU requires --data-dir and --languages")
        if (
            args.snapshot is not None
            or args.subjects is not None
            or args.limit_per_subject
        ):
            parser.error(
                "--snapshot, --subjects, and --limit-per-subject are MMLU options"
            )
        languages = [item.strip() for item in args.languages.split(",") if item.strip()]
        if not languages:
            parser.error("--languages must contain at least one language")
        prompts = mmmlu.iter_prompts(
            data_dir=args.data_dir,
            languages=languages,
            task_dir=args.task_dir,
            limit_per_language=args.limit,
            num_fewshot=args.num_fewshot,
        )
    write_inputs(
        model=args.model_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        prompts=prompts,
    )
    protocol = {
        "benchmark": args.benchmark,
        "source": source if args.benchmark == "mmlu" else "csv",
        "num_fewshot": args.num_fewshot,
        "max_length": args.max_length,
    }
    if args.benchmark == "mmmlu":
        protocol["languages"] = languages
        protocol["limit_per_language"] = args.limit
    (args.output_dir.resolve().parent / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
