# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Summarize choice-score logs and optionally compare a reference run."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


SCORE_RE = re.compile(
    r"ASCEND_BATCH_SCORES id=(\d+) A=([^ ]+) B=([^ ]+) C=([^ ]+) D=([^ ]+)"
)


def ratio(correct: int, count: int) -> float:
    return correct / count if count else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="reference records.jsonl")
    parser.add_argument("--benchmark", choices=("mmlu", "mmmlu"))
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
    if not samples:
        raise ValueError("sample file is empty")
    has_language = {"language" in sample for sample in samples}
    if len(has_language) != 1:
        raise ValueError("sample file mixes MMLU and MMMLU records")
    benchmark = "mmmlu" if True in has_language else "mmlu"
    if args.benchmark is not None and args.benchmark != benchmark:
        raise ValueError(f"sample file contains {benchmark}, not {args.benchmark}")
    protocol_path = args.samples.parent.parent / "protocol.json"
    protocol_metadata = (
        json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
    )
    if protocol_metadata and protocol_metadata.get("benchmark") != benchmark:
        raise ValueError("protocol metadata does not match sample benchmark")
    num_fewshot = protocol_metadata.get("num_fewshot", 5)
    scores = {}
    for log_path in args.logs:
        for line in log_path.read_text().splitlines():
            match = SCORE_RE.fullmatch(line.strip())
            if match:
                sample_id = int(match.group(1))
                if sample_id in scores:
                    raise ValueError(f"duplicate choice scores for sample {sample_id}")
                scores[sample_id] = [float(match.group(index)) for index in range(2, 6)]
    if set(scores) != set(range(len(samples))):
        raise RuntimeError(f"score coverage mismatch: {len(scores)}/{len(samples)}")
    by_subject = defaultdict(lambda: [0, 0])
    by_language = defaultdict(lambda: [0, 0])
    by_language_subject = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    records = []
    for sample in samples:
        values = scores[sample["id"]]
        prediction = "ABCD"[max(range(4), key=lambda index: values[index])]
        correct = prediction == sample["answer"]
        by_subject[sample["subject"]][0] += correct
        by_subject[sample["subject"]][1] += 1
        if benchmark == "mmmlu":
            by_language[sample["language"]][0] += correct
            by_language[sample["language"]][1] += 1
            bucket = by_language_subject[sample["language"]][sample["subject"]]
            bucket[0] += correct
            bucket[1] += 1
        records.append(
            {
                **sample,
                "prediction": prediction,
                "scores": dict(zip("ABCD", values)),
                "correct": correct,
            }
        )
    correct = sum(record["correct"] for record in records)
    summary = {
        "samples": len(records),
        "correct": correct,
        "accuracy": ratio(correct, len(records)),
        "accuracy_percent": 100 * ratio(correct, len(records)),
    }
    if benchmark == "mmlu":
        summary["protocol"] = (
            f"lm_eval MMLU default {num_fewshot}-shot, left truncation to max length"
        )
        summary["subjects"] = {
            subject: {
                "correct": values[0],
                "samples": values[1],
                "accuracy": ratio(*values),
            }
            for subject, values in sorted(by_subject.items())
        }
    else:
        summary["protocol"] = (
            f"lm_eval openai-mmmlu {num_fewshot}-shot, "
            "fewshot_split=test sampler=first_n "
            "excluding eval_doc, bare A/B/C/D token argmax, left truncation"
        )
        summary["languages"] = {
            language: {
                "correct": values[0],
                "samples": values[1],
                "accuracy": ratio(*values),
                "accuracy_percent": 100 * ratio(*values),
            }
            for language, values in sorted(by_language.items())
        }
        summary["language_subjects"] = {
            language: {
                subject: {
                    "correct": values[0],
                    "samples": values[1],
                    "accuracy": ratio(*values),
                }
                for subject, values in sorted(subjects.items())
            }
            for language, subjects in sorted(by_language_subject.items())
        }
        summary["truncated"] = sum(bool(record["left_truncated"]) for record in records)
    if args.reference is not None:
        reference = {
            int(record["id"]): record
            for record in (
                json.loads(line) for line in args.reference.read_text().splitlines()
            )
        }
        if set(reference) != set(scores):
            raise ValueError("reference and current run have different sample IDs")
        for record in records:
            other = reference[record["id"]]
            fields = ("prompt", "answer", "subject", "language")
            if tuple(record.get(field) for field in fields) != tuple(
                other.get(field) for field in fields
            ):
                raise ValueError(
                    f"reference sample {record['id']} has different prompt or answer"
                )
        agreement = sum(
            record["prediction"] == reference[record["id"]]["prediction"]
            for record in records
        )
        reference_correct = sum(
            bool(reference[record["id"]]["correct"]) for record in records
        )
        summary["reference"] = {
            "path": str(args.reference),
            "correct": reference_correct,
            "accuracy_percent": 100 * reference_correct / len(records),
            "agreement_percent": 100 * agreement / len(records),
            "accuracy_delta_pp": 100 * (correct - reference_correct) / len(records),
        }
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: summary[key] for key in ("samples", "correct", "accuracy_percent")}
        )
    )


if __name__ == "__main__":
    main()
