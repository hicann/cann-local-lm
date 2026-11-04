# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Pure MMLU prompt and option conventions."""

LABELS = "ABCD"


def format_question_parts(question: str, choices: list[str]) -> str:
    return "\n".join(
        [
            str(question).strip(),
            *(
                f"{label}. {choice}"
                for label, choice in zip(LABELS, choices, strict=True)
            ),
            "Answer:",
        ]
    )


def format_question(doc: dict) -> str:
    return format_question_parts(doc["question"], doc["choices"])


def choice_token_ids(tokenizer) -> list[int]:
    values = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(ids) != 1 for ids in values):
        raise ValueError("A/B/C/D must each encode to one token")
    return [ids[0] for ids in values]
