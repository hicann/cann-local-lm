# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Multilingual openai/MMMLU prompts matching lm-eval's first_n policy."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from ..protocol import format_question_parts, LABELS


def default_task_dir() -> Path:
    override = os.environ.get("LM_EVAL_MMMLU_TASK_DIR")
    if override:
        path = Path(override)
    else:
        import lm_eval

        path = Path(lm_eval.__file__).resolve().parent / "tasks/openai-mmmlu/default"
    if not path.is_dir():
        raise FileNotFoundError(f"MMMLU task directory is missing: {path}")
    return path


def description_for(task_dir: Path, language: str, subject: str) -> str:
    import yaml

    task_file = task_dir / f"mmmlu_{language.lower()}_{subject}.yaml"
    if not task_file.is_file():
        raise FileNotFoundError(f"MMMLU task description is missing: {task_file}")
    # compose reads scalars without trying to execute lm-eval's !function tags.
    document = yaml.compose(task_file.read_text(), Loader=yaml.BaseLoader)
    if document is None:
        raise ValueError(f"empty MMMLU task description: {task_file}")
    for key, value in document.value:
        if key.value == "description":
            return value.value
    raise ValueError(f"MMMLU task has no description: {task_file}")


def load_language(data_dir: Path, language: str) -> list[dict]:
    path = data_dir / f"mmlu_{language.replace('_', '-')}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stride_indices(total: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= total:
        return list(range(total))
    step = total / limit
    return sorted({min(total - 1, int(index * step)) for index in range(limit)})


def format_row(row: dict) -> str:
    return format_question_parts(row["Question"], [row[label] for label in LABELS])


def first_n_excluding(rows: list[dict], current: dict, count: int) -> list[dict]:
    # lm-eval FirstNSampler removes eval_doc when fewshot_split == test_split.
    pool = [row for row in rows if row != current]
    if count > len(pool):
        raise ValueError("MMMLU subject has too few other samples for few-shot prompts")
    return pool[:count]


def iter_prompts(
    *,
    data_dir: Path,
    languages: list[str],
    task_dir: Path | None,
    limit_per_language: int,
    num_fewshot: int,
) -> Iterator[dict]:
    task_dir = task_dir or default_task_dir()
    for requested_language in languages:
        language = requested_language.strip().upper().replace("-", "_")
        rows = load_language(data_dir, language)
        by_subject: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_subject[row["Subject"]].append(row)
        descriptions = {
            subject: description_for(task_dir, language, subject)
            for subject in by_subject
        }
        for index in stride_indices(len(rows), limit_per_language):
            row = rows[index]
            subject = row["Subject"]
            demos = "\n\n".join(
                format_row(sample) + sample["Answer"]
                for sample in first_n_excluding(by_subject[subject], row, num_fewshot)
            )
            if demos:
                demos += "\n\n"
            yield {
                "language": language,
                "subject": subject,
                "answer": row["Answer"],
                "prompt": descriptions[subject] + demos + format_row(row),
            }
