# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run PTE, CPU eager, or NPU eager MMLU choice-logit scoring."""

from __future__ import annotations

import argparse
import multiprocessing
import struct
import subprocess
import time
from pathlib import Path


def read_manifest(path: Path) -> list[tuple[int, Path, int]]:
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            sample_id, tokens, _positions, length = line.split()
            entries.append((int(sample_id), Path(tokens), int(length)))
    return entries


def read_tokens(path: Path, expected: int):
    import torch

    data = path.read_bytes()
    if len(data) != expected * 8:
        raise ValueError(f"{path}: manifest length differs from token file")
    return torch.tensor(struct.unpack(f"<{expected}q", data), dtype=torch.long)


def score_shard(
    model, entries, choice_tensor, device, output: Path, log_every: int, label: str
) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for index, (sample_id, token_path, length) in enumerate(entries):
            tokens = read_tokens(token_path, length).unsqueeze(0).to(device)
            positions = torch.arange(length, dtype=torch.long, device=device).unsqueeze(
                0
            )
            logits = model(
                input_ids=tokens,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            ).logits[0, -1, :]
            scores = logits.index_select(0, choice_tensor).float().cpu().tolist()
            if not all(
                value == value and abs(value) != float("inf") for value in scores
            ):
                raise ValueError(f"non-finite choice logits for sample {sample_id}")
            stream.write(
                f"ASCEND_BATCH_SCORES id={sample_id} "
                f"A={scores[0]} B={scores[1]} C={scores[2]} D={scores[3]}\n"
            )
            stream.flush()
            done = index + 1
            if done % log_every == 0 or done == len(entries):
                elapsed = time.time() - started
                print(
                    f"EAGER_PROGRESS {label} done={done}/{len(entries)} elapsed_s={elapsed:.1f}",
                    flush=True,
                )


def run_eager(args, entries, choice_ids: list[int]) -> None:
    import torch

    if args.backend == "npu-eager":
        import torch_npu  # noqa: F401

        if args.workers != 1:
            raise ValueError(
                "NPU eager uses one worker per device; use independent shards"
            )
    device = torch.device(
        args.device or ("cpu" if args.backend == "cpu-eager" else "npu:0")
    )
    if args.backend == "npu-eager":
        if device.type != "npu":
            raise ValueError("npu-eager requires an NPU device")
        torch.npu.set_device(device)
    elif device.type != "cpu":
        raise ValueError("cpu-eager requires a CPU device")
    if args.workers > 1 and args.backend == "cpu-eager":
        torch.set_num_threads(1)

    from examples.auto import AutoModelForCausalLM

    dtype_name = args.dtype or (
        "float32" if args.backend == "cpu-eager" else "bfloat16"
    )
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            strict=True,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    choice_tensor = torch.tensor(choice_ids, dtype=torch.long, device=device)
    if args.workers == 1:
        score_shard(
            model,
            entries,
            choice_tensor,
            device,
            args.output,
            args.log_every,
            "worker0",
        )
        return

    context = multiprocessing.get_context("fork")
    processes = []
    for worker in range(args.workers):
        output = args.output.with_name(
            f"{args.output.stem}.w{worker}{args.output.suffix}"
        )
        process = context.Process(
            target=score_shard,
            args=(
                model,
                entries[worker :: args.workers],
                choice_tensor,
                device,
                output,
                args.log_every,
                f"worker{worker}",
            ),
            name=f"cpu-eager-{worker}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"{process.name} exited with {process.exitcode}")


def run_pte(
    args, choice_ids: list[int], *, mask_window: int | None, causal_mask: bool
) -> None:
    runner = args.runner or Path("build/release/executorch_npu_stateful_runner")
    if args.pte is None or args.ptd is None:
        raise ValueError("pte backend requires --pte and --ptd")
    if args.max_seq_len is None:
        raise ValueError("pte backend requires --max-seq-len")
    command = [
        str(runner),
        "batch",
        str(args.pte),
        str(args.ptd),
        args.method,
        str(args.manifest),
        str(args.device_id),
        args.state_namespace,
        "--max-seq-len",
        str(args.max_seq_len),
        "--choice-token-ids",
        ",".join(map(str, choice_ids)),
    ]
    if mask_window is not None:
        command += ["--mask-window", str(mask_window)]
    elif causal_mask:
        command += ["--causal-mask"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(
    default_model: Path,
    *,
    mask_window: int | None = None,
    causal_mask: bool = False,
    default_namespace: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("pte", "cpu-eager", "npu-eager"), required=True
    )
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--choice-token-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pte", type=Path)
    parser.add_argument("--ptd", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--method", default="prefill")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--device")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--state-namespace", default=default_namespace)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    choice_ids = [int(x) for x in args.choice_token_ids.split(",")]
    if len(choice_ids) != 4:
        raise ValueError("--choice-token-ids requires A,B,C,D token IDs")
    if (
        args.workers < 1
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        raise ValueError("invalid worker or shard count")
    if args.backend == "pte":
        if args.shard_count != 1 or args.limit:
            raise ValueError(
                "PTE batch uses the full manifest; prepare a subset manifest if needed"
            )
        run_pte(args, choice_ids, mask_window=mask_window, causal_mask=causal_mask)
    else:
        entries = read_manifest(args.manifest)
        entries = [x for x in entries if x[0] % args.shard_count == args.shard_index]
        if args.limit > 0:
            entries = entries[: args.limit]
        run_eager(args, entries, choice_ids)
