# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Fixed-trajectory, batch=1 end-to-end regression for Gemma4 / Qwen3.5.

python -m examples.common.light_eval baseline --model MODEL --model-path WEIGHTS --pte-dir PTE --output BASE
python -m examples.common.light_eval run --model MODEL --model-path WEIGHTS --pte-dir PTE --baseline BASE --output RUN
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("ttft_ms", "tpot_ms", "total_ms", "throughput_tps")
TEXTS = [
    (
        "short",
        128,
        "A library has books on science, history, and literature. Readers borrow books and return them for other people to enjoy. ",
        "Explain why public libraries are useful.",
    ),
    (
        "medium",
        512,
        "An engineer measures a program using identical inputs. Model loading is separated from request latency. Repeated measurements help identify noise. Accuracy must remain stable after an optimization. ",
        "Summarize the principles of a fair performance comparison.",
    ),
    (
        "long",
        2048,
        "A coastal town collects rainwater in reservoirs. In spring the water level rises, while summer demand increases. Engineers monitor storage, repair leaks, and encourage conservation. The town balances household needs with agriculture and ecosystem protection. ",
        "Describe three practical ways the town can manage its water supply.",
    ),
]


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def command_text(args):
    return subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout.strip()


def prepare(cfg, directory):
    from transformers import AutoTokenizer

    directory.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    cases = []
    for name, length, passage, question in TEXTS:
        # Keep both the chat prefix and final question, removing only excess context.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": passage * 150 + "\n" + question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < length:
            raise ValueError("fixture text is too short")
        ids = ids[: length - 64] + ids[-64:]
        tokens = directory / f"{name}.i64"
        positions = directory / f"{name}.positions.i64"
        np.asarray(ids, dtype="<i8").tofile(tokens)
        np.arange(length, dtype="<i8").tofile(positions)
        cases.append(
            {
                "name": name,
                "length": length,
                "tokens": tokens.name,
                "positions": positions.name,
                "sha256": digest(tokens),
                "decoded_text": tokenizer.decode(ids),
            }
        )
    write_json(directory / "cases.json", cases)
    return cases


def pte_case(cfg, case, inputs, out, trajectory=None, all_cases=None, reference=None):
    args = [
        cfg["runner"],
        "session",
        str(Path(cfg["pte_dir"]) / "model.pte"),
        str(Path(cfg["pte_dir"]) / "ascend.weights.package.ge_aot.ptd"),
        "prefill",
        str(inputs / case["tokens"]),
        str(inputs / case["positions"]),
        "decode",
        str(case["length"]),
        "1,1",
        str(cfg["output_tokens"]),
        str(cfg.get("device_id", 0)),
        "light_eval",
        "--max-seq-len",
        str(cfg["capacity"]),
        "--eval-output",
        str(out),
        "--warmup",
        str(cfg["warmup"]),
        "--repeats",
        str(cfg["repeats"]),
    ]
    args += ["--mask-window", "512"] if cfg["model"] == "gemma4" else ["--causal-mask"]
    if trajectory:
        args += ["--forced-tokens", str(trajectory)]
    out.mkdir(parents=True)
    label = case["name"]
    if all_cases is not None:
        manifest = out / "cases.tsv"
        lines = []
        for item in all_cases:
            forced = (
                str(reference / item["name"] / "trajectory.i64") if reference else "-"
            )
            lines.append(
                "\t".join(
                    [
                        item["name"],
                        str(inputs / item["tokens"]),
                        str(inputs / item["positions"]),
                        str(item["length"]),
                        forced,
                    ]
                )
            )
        manifest.write_text("\n".join(lines) + "\n")
        args += ["--eval-manifest", str(manifest)]
        label = "all cases (one model load)"
    write_json(out / "command.json", args)
    print(f"PTE {cfg['model']} {label} starting", flush=True)
    with (out / "runner.log").open("wb") as log:
        subprocess.run(
            args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600
        )
    print(f"PTE {cfg['model']} {label} complete", flush=True)


def eager_worker(cfg, inputs, reference, output):
    import torch
    import torch_npu  # noqa: F401

    from examples.auto import AutoModelForCausalLM

    adapter = importlib.import_module(f"examples.{cfg['model']}.export.adapter")
    device = torch.device(f"npu:{cfg.get('device_id', 0)}")
    torch.npu.set_device(device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            cfg["model_path"],
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            strict=True,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    for case in read_json(inputs / "cases.json"):
        ids = np.fromfile(inputs / case["tokens"], dtype="<i8").tolist()
        forced = np.fromfile(
            reference / case["name"] / "trajectory.i64", dtype="<i8"
        ).tolist()
        if len(forced) != cfg["output_tokens"]:
            raise ValueError("trajectory length mismatch")
        out = output / case["name"]
        out.mkdir(parents=True)
        np.asarray(forced, dtype="<i8").tofile(out / "trajectory.i64")
        for trial in range(cfg["warmup"] + cfg["repeats"]):
            print(f"EAGER {cfg['model']} {case['name']} trial={trial}", flush=True)
            torch.npu.synchronize()
            start = phase = time.perf_counter()
            steps, snapshots, predictions = [], [], []
            cache = adapter.create_cache(
                model.config,
                max_seq_len=cfg["capacity"],
                device=device,
                dtype=torch.bfloat16,
            )
            wrapper = adapter.ExportWrapper(model, cache)
            with torch.inference_mode():
                tokens = torch.tensor([ids], dtype=torch.long, device=device)
                positions = torch.arange(len(ids), dtype=torch.long, device=device)
                # Same externally supplied prefill masks as the PTE ABI.
                if cfg["model"] == "gemma4":
                    masks = (
                        adapter.build_mask(
                            len(ids),
                            cfg["capacity"],
                            512,
                            device=torch.device("cpu"),
                            dtype=torch.bfloat16,
                        ).to(device),
                        adapter.build_mask(
                            len(ids),
                            cfg["capacity"],
                            None,
                            device=torch.device("cpu"),
                            dtype=torch.bfloat16,
                        ).to(device),
                    )
                else:
                    masks = (
                        adapter.build_causal_mask(
                            len(ids),
                            cfg["capacity"],
                            device=torch.device("cpu"),
                            dtype=torch.bfloat16,
                        ).to(device),
                    )
                for step in range(cfg["output_tokens"]):
                    if step:
                        tokens = torch.tensor(
                            [[forced[step - 1]]], dtype=torch.long, device=device
                        )
                        positions = torch.tensor(
                            [len(ids) + step - 1], dtype=torch.long, device=device
                        )
                        logits = wrapper(tokens, positions)
                    else:
                        logits = wrapper(tokens, positions, *masks)
                    # Host-visible output + greedy selection are in the timed path,
                    # just as the PTE runner's synchronous D2H and GreedyToken.
                    host = logits.float().cpu()
                    token = int(host.argmax().item())
                    end = time.perf_counter()
                    steps.append((end - phase) * 1000)
                    arr = host.numpy().copy()
                    if not np.isfinite(arr).all():
                        raise ValueError("non-finite eager logits")
                    snapshots.append(arr)
                    predictions.append(token)
                    phase = time.perf_counter()
            wall = (time.perf_counter() - start) * 1000
            if trial >= cfg["warmup"]:
                index = trial - cfg["warmup"]
                logits = np.stack(snapshots)
                logits.astype("<f4").tofile(out / f"trial_{index}.f32")
                total = sum(steps)
                write_json(
                    out / f"trial_{index}.json",
                    {
                        "trial": index,
                        "output_tokens": len(steps),
                        "vocab_size": logits.shape[1],
                        "ttft_ms": steps[0],
                        "tpot_ms": sum(steps[1:]) / (len(steps) - 1),
                        "total_ms": total,
                        "throughput_tps": len(steps) * 1000 / total,
                        "wall_with_capture_ms": wall,
                        "predictions": predictions,
                        "step_ms": steps,
                    },
                )
            del wrapper, cache, masks, logits, host, snapshots


def load_logits(folder, trial=0):
    meta = read_json(folder / f"trial_{trial}.json")
    a = np.fromfile(folder / f"trial_{trial}.f32", dtype="<f4")
    a = a.reshape(meta["output_tokens"], meta["vocab_size"])
    if not np.isfinite(a).all():
        raise ValueError(f"non-finite logits: {folder}")
    return a


def compare(a, b):
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"incompatible logits shapes: {a.shape}, {b.shape}")
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite logits")
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denominator == 0):
        raise ValueError("zero-norm logits")
    cosine = np.clip(np.sum(a * b, axis=1) / denominator, -1, 1)
    return {
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "worst_step": int(cosine.argmin()),
        "cosine_by_step": cosine.tolist(),
        "top1_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
        "max_abs_diff": float(np.abs(a - b).max()),
    }


def summarize(cfg, root, baseline, establishing):
    rows = []
    for case in read_json(root / "inputs/cases.json"):
        name = case["name"]
        for backend in ["pte", "eager"] if establishing else ["pte"]:
            folder = root / backend / name
            trials = [
                read_json(folder / f"trial_{i}.json") for i in range(cfg["repeats"])
            ]
            arrays = [load_logits(folder, i) for i in range(cfg["repeats"])]
            for item in trials:
                if item["output_tokens"] != cfg["output_tokens"] or any(
                    not np.isfinite(item[k]) or item[k] <= 0 for k in METRICS
                ):
                    raise ValueError("invalid timing or incomplete output")
            base = baseline
            if (folder / "trajectory.i64").read_bytes() != (
                base / "pte" / name / "trajectory.i64"
            ).read_bytes():
                raise ValueError("cannot compare different decode trajectories")
            eager = load_logits(base / "eager" / name)
            pte = load_logits(base / "pte" / name)
            # Compare every measured repeat, retain the worst repeat in the report.
            ec = [compare(a, eager) for a in arrays]
            pc = [compare(a, pte) for a in arrays]
            repeat = [compare(a, arrays[0]) for a in arrays]
            row = {
                "case": name,
                "length": case["length"],
                "backend": backend,
                **{k: float(np.median([t[k] for t in trials])) for k in METRICS},
                "ranges": {
                    k: [min(t[k] for t in trials), max(t[k] for t in trials)]
                    for k in METRICS
                },
                "vs_eager": min(ec, key=lambda x: x["cosine_min"]),
                "vs_pte": min(pc, key=lambda x: x["cosine_min"]),
                "repeat_cosine_min": min(x["cosine_min"] for x in repeat),
                "repeat_top1_min": min(x["top1_agreement"] for x in repeat),
            }
            if establishing:
                row["status"] = "BASELINE"
            else:
                old = next(
                    x
                    for x in read_json(base / "summary.json")["rows"]
                    if x["case"] == name and x["backend"] == "pte"
                )
                tol = max(1e-5, 5 * (1 - old["repeat_cosine_min"]))
                floor = old["vs_eager"]["cosine_min"] - tol
                changed = (
                    row["vs_pte"]["cosine_min"] < 1 - tol
                    or row["vs_eager"]["cosine_min"] < floor
                    or row["vs_pte"]["top1_agreement"] < old["repeat_top1_min"]
                )
                row["status"] = "WARN" if changed else "PASS"
                row["cosine_tolerance"] = tol
                row["speedup_vs_initial_pte"] = old["total_ms"] / row["total_ms"]
            rows.append(row)
    result = {
        "model": cfg["model"],
        "capacity": cfg["capacity"],
        "output_tokens": cfg["output_tokens"],
        "baseline": str(baseline),
        "rows": rows,
        "status": "WARN"
        if any(x["status"] == "WARN" for x in rows)
        else ("BASELINE" if establishing else "PASS"),
    }
    write_json(root / "summary.json", result)
    return result


def append_report(path, result, root, stage):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Gemma4 / Qwen3.5 轻量性能与数值回归记录\n\n"
            "固定容量 5120，batch=1，BF16；输入 128/512/2048 tokens，每次输出 32 tokens。\n"
            "轨迹取初始 PTE 首次贪心结果，后续 eager/PTE 均喂入相同前缀；输出不因 EOS 提前停止。\n"
            "各用例预热 1 次、测量 3 次，报告中位数；模型加载、文件读取、日志及 logits 快照不计时。\n"
            "TTFT 包含请求状态初始化、mask、搬运、prefill 和选词；总延迟为移除快照开销后的各步耗时之和。\n"
            "TPOT=(总延迟−TTFT)/31；吞吐=32/总延迟秒。原始 wall_with_capture_ms 保留供审计。\n"
            "这些是固定轨迹的推理端到端指标，不包含 tokenization，也不是服务并发吞吐。\n"
            "余弦比较完整词表最终 logits；step=0 为 prefill。逐层检查和 MMLU 不属于本快速集。\n"
            "初始 eager 使用与 PTE 相同的模型代码、静态容量和有状态 decode；PTE 标准 export。\n"
            "每次命令默认顺序执行两个模型；请保持 NPU 独占。数值 WARN 用于人工检查，不自动覆盖基线。\n\n"
        )
    with path.open("a") as f:
        f.write(
            f"## {datetime.now(timezone.utc).isoformat()} — {stage} — {result['model']}\n\n"
        )
        f.write(f"结果目录：`{root}`；配置/提交/产物指纹见 `metadata.json`。\n\n")
        f.write(
            "| 用例 | 后端 | 对 eager 最低余弦 | 对初始 PTE 最低余弦 | 对 PTE top-1 一致率 | TTFT ms | TPOT ms | 总延迟 ms | tokens/s | 状态 |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in result["rows"]:
            f.write(
                f"| {r['case']} ({r['length']}) | {r['backend']} | {r['vs_eager']['cosine_min']:.8f} | {r['vs_pte']['cosine_min']:.8f} | {r['vs_pte']['top1_agreement']:.2%} | {r['ttft_ms']:.2f} | {r['tpot_ms']:.2f} | {r['total_ms']:.2f} | {r['throughput_tps']:.3f} | {r['status']} |\n"
            )
        f.write("\n")


def metadata(cfg):
    packages = {}
    for name in ["torch", "torch-npu", "executorch", "transformers", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    files = [
        Path(cfg["runner"]),
        Path(cfg["pte_dir"]) / "model.pte",
        Path(cfg["pte_dir"]) / "ascend.weights.package.ge_aot.ptd",
    ]
    files += sorted(Path(cfg["model_path"]).glob("*.safetensors"))
    files += [
        p
        for p in Path(cfg["model_path"]).glob("*.json")
        if p.name
        in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ]
    ]
    print(f"Fingerprinting {cfg['model']} artifacts", flush=True)
    return {
        "config": cfg,
        "utc": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command_text(["git", "rev-parse", "HEAD"]),
        "git_status": command_text(["git", "status", "--short"]),
        "git_diff": command_text(["git", "diff"]),
        "eval_source_sha256": digest(__file__),
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "ASCEND_RT_VISIBLE_DEVICES",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
            ]
        },
        "npu_smi": subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True
        ).stdout,
        "files": [fingerprint(p) for p in files],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "run", "eager-worker", "report"])
    parser.add_argument(
        "--config", type=Path, help="optional external evaluation configuration"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--pte-dir", type=Path)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("build/release/executorch_npu_stateful_runner"),
    )
    parser.add_argument("--capacity", type=int, default=5120)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--model", choices=["gemma4", "qwen3_5"])
    parser.add_argument("--stage", default="initial")
    parser.add_argument("--log", type=Path, default=Path("docs/performance_log.md"))
    args = parser.parse_args()
    if args.config is not None:
        configs = read_json(args.config)
    else:
        if args.model is None or args.model_path is None or args.pte_dir is None:
            parser.error(
                "provide --model, --model-path and --pte-dir, or an external --config"
            )
        configs = [
            dict(
                model=args.model,
                model_path=str(args.model_path.resolve()),
                pte_dir=str(args.pte_dir.resolve()),
                runner=str(args.runner.resolve()),
                capacity=args.capacity,
                output_tokens=args.output_tokens,
                warmup=args.warmup,
                repeats=args.repeats,
                device_id=args.device_id,
            )
        ]
    if args.mode == "eager-worker":
        if args.config is None:
            parser.error("eager-worker requires its generated result config")
        eager_worker(
            configs, args.baseline / "inputs", args.baseline / "pte", args.output
        )
        return
    if args.mode == "run" and args.baseline is None:
        parser.error("run requires --baseline")
    for cfg in configs:
        if args.model and cfg["model"] != args.model:
            continue
        root = args.output.resolve() / cfg["model"]
        base = (args.baseline.resolve() / cfg["model"]) if args.baseline else root
        if args.mode == "report":
            result = summarize(cfg, root, base, args.baseline is None)
            append_report(args.log, result, root, args.stage)
            continue
        if root.exists():
            raise FileExistsError(
                f"output already exists; use a new stage directory: {root}"
            )
        if (
            cfg["capacity"] < 2048 + cfg["output_tokens"]
            or cfg["output_tokens"] < 2
            or cfg["repeats"] < 1
            or cfg["warmup"] < 1
        ):
            raise ValueError("invalid lengths/repeats/warmup")
        root.mkdir(parents=True)
        meta = metadata(cfg)
        write_json(root / "metadata.json", meta)
        write_json(root / "config.json", cfg)
        if args.mode == "baseline":
            cases = prepare(cfg, root / "inputs")
        else:
            old = read_json(base / "config.json")
            for key in [
                "model",
                "model_path",
                "capacity",
                "output_tokens",
                "device_id",
            ]:
                if cfg.get(key) != old.get(key):
                    raise ValueError(f"baseline-incompatible config: {key}")
            # Checkpoint/tokenizer changes are not numerical optimization comparisons.
            oldfiles = read_json(base / "metadata.json")["files"]
            modeldir = str(Path(cfg["model_path"]).resolve()) + "/"
            for item in oldfiles:
                if item["path"].startswith(modeldir):
                    current = next(
                        x for x in meta["files"] if x["path"] == item["path"]
                    )
                    if current["sha256"] != item["sha256"]:
                        raise ValueError("model/tokenizer differs from baseline")
            shutil.copytree(base / "inputs", root / "inputs")
            cases = read_json(root / "inputs/cases.json")
        for case in cases:
            if digest(root / "inputs" / case["tokens"]) != case["sha256"]:
                raise ValueError("input fingerprint mismatch")
        reference = None if args.mode == "baseline" else base / "pte"
        pte_case(
            cfg,
            cases[0],
            root / "inputs",
            root / "pte",
            all_cases=cases,
            reference=reference,
        )
        if args.mode == "baseline":
            with (root / "eager.log").open("wb") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "examples.common.light_eval",
                        "eager-worker",
                        "--config",
                        str(root / "config.json"),
                        "--baseline",
                        str(root),
                        "--output",
                        str(root / "eager"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=7200,
                )
        result = summarize(cfg, root, base, args.mode == "baseline")
        append_report(args.log, result, root, args.stage)
        print(
            json.dumps(
                {"model": cfg["model"], "status": result["status"], "output": str(root)}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
