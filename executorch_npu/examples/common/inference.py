# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared tokenizer and stateful runner session driver."""

from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path


def run_session(
    *,
    model_path: Path,
    pte_dir: Path,
    prompt: str,
    runner: Path,
    max_new_tokens: int,
    max_seq_len: int,
    device_id: int,
    state_namespace: str,
    min_prefill_len: int,
    mask_window: int | None = None,
    causal_mask: bool = False,
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    length = len(tokens)
    if not min_prefill_len <= length < max_seq_len:
        raise ValueError(
            f"prompt has {length} tokens; prefill requires [{min_prefill_len}, {max_seq_len - 1}]"
        )
    with tempfile.TemporaryDirectory(prefix="cann_session_") as folder:
        root = Path(folder)
        token_file, position_file = root / "tokens.bin", root / "positions.bin"
        token_file.write_bytes(struct.pack(f"<{length}q", *tokens))
        position_file.write_bytes(struct.pack(f"<{length}q", *range(length)))
        command = [
            str(runner),
            "session",
            str(pte_dir / "model.pte"),
            str(pte_dir / "ascend.weights.package.ge_aot.ptd"),
            "prefill",
            str(token_file),
            str(position_file),
            "decode",
            str(length),
            "1,1",
            str(max_new_tokens),
            str(device_id),
            state_namespace,
            "--max-seq-len",
            str(max_seq_len),
        ]
        if mask_window is not None:
            command += ["--mask-window", str(mask_window)]
        elif causal_mask:
            command += ["--causal-mask"]
        result = subprocess.run(command, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    print(stdout, end="")
    if stderr:
        import sys

        print(stderr, end="", file=sys.stderr)
    result.check_returncode()
    generated = [
        int(part.split("=", 1)[1])
        for line in stdout.splitlines()
        if line.startswith("ASCEND_SESSION_TOKEN")
        for part in line.split()
        if part.startswith("token_id=")
    ]
    if generated:
        print(
            f"Generated text: {tokenizer.decode(generated, skip_special_tokens=True)}"
        )


def main(
    default_model: Path,
    default_pte_dir: Path,
    *,
    min_prefill_len: int,
    mask_window: int | None = None,
    causal_mask: bool = False,
    default_namespace: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--pte-dir", type=Path, default=default_pte_dir)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("build/release/executorch_npu_stateful_runner"),
    )
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--state-namespace", default=default_namespace)
    args = parser.parse_args()
    run_session(
        model_path=args.model_path,
        pte_dir=args.pte_dir,
        prompt=args.prompt,
        runner=args.runner,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        device_id=args.device_id,
        state_namespace=args.state_namespace,
        min_prefill_len=min_prefill_len,
        mask_window=mask_window,
        causal_mask=causal_mask,
    )
