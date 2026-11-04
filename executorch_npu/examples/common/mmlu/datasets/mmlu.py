# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""English cais/mmlu prompts using the lm-eval MMLU task descriptions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ..protocol import format_question, LABELS


SKIP_CONFIGS = {"all", "auxiliary_train"}


def default_snapshot() -> Path:
    root = Path.home() / ".cache/huggingface/hub/datasets--cais--mmlu/snapshots"
    snapshots = sorted(path for path in root.glob("*") if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no cached cais/mmlu snapshot under {root}")
    return snapshots[-1]


def default_task_dir() -> Path:
    if os.environ.get("LM_EVAL_TASK_DIR"):
        return Path(os.environ["LM_EVAL_TASK_DIR"])
    import lm_eval

    return Path(lm_eval.__file__).resolve().parent / "tasks/mmlu/default"


def subjects_for(source: str, snapshot: Path | None) -> list[str]:
    if source == "parquet":
        assert snapshot is not None
        return sorted(
            path.name
            for path in snapshot.iterdir()
            if path.is_dir() and path.name not in SKIP_CONFIGS
        )
    from datasets import get_dataset_config_names

    return [
        name
        for name in get_dataset_config_names("cais/mmlu")
        if name not in SKIP_CONFIGS
    ]


def records_for(source: str, subject: str, snapshot: Path | None):
    if source == "parquet":
        import pandas as pd

        assert snapshot is not None
        folder = snapshot / subject
        dev = pd.read_parquet(folder / "dev-00000-of-00001.parquet").to_dict("records")
        test = pd.read_parquet(folder / "test-00000-of-00001.parquet").to_dict(
            "records"
        )
        return dev, test
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", subject)
    return dataset["dev"], dataset["test"]


def iter_prompts(
    *,
    source: str,
    snapshot: Path | None,
    task_dir: Path | None,
    subjects: list[str] | None,
    limit_per_subject: int,
    num_fewshot: int,
) -> Iterator[dict]:
    import yaml

    if source not in {"parquet", "hf"}:
        raise ValueError("MMLU source must be parquet or hf")
    if source == "parquet":
        snapshot = snapshot or default_snapshot()
    task_dir = task_dir or default_task_dir()
    selected = subjects if subjects is not None else subjects_for(source, snapshot)
    for subject in selected:
        if subject in SKIP_CONFIGS:
            continue
        description = yaml.safe_load((task_dir / f"mmlu_{subject}.yaml").read_text())[
            "description"
        ]
        dev, test = records_for(source, subject, snapshot)
        demos = "\n\n".join(
            format_question(doc) + LABELS[doc["answer"]]
            for doc in list(dev)[:num_fewshot]
        )
        if demos:
            demos += "\n\n"
        for index, doc in enumerate(test):
            if 0 < limit_per_subject <= index:
                break
            yield {
                "subject": subject,
                "answer": LABELS[doc["answer"]],
                "prompt": description + demos + format_question(doc),
            }
