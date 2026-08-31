#!/usr/bin/env python3
"""Run the gemma4 and qwen3.5 dopt demo workflows as system tests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from verify_output import VerificationError, verify_output


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
QUANT_DIR = PROJECT_ROOT / "cann_lm" / "converter" / "quant"
TEMPLATE_PATH = SCRIPT_DIR / "config.yaml.template"
# Keep the ST self-contained by using the tracked calibration set that the
# converter's default config also references.
CALIBRATION_DATA = QUANT_DIR / "calibrate_sets" / "calibrate_c4.jsonl"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"
PLACEHOLDERS = {
    "__CALIBRATION_DATA__",
    "__PTQ_SAMPLES__",
    "__FP16__",
    "__CUTOFF_LEN__",
    "__NUM_SAMPLES__",
}


@dataclass(frozen=True)
class ModelCase:
    key: str
    st_id: str
    display_name: str
    model_types: tuple[str, ...]
    ptq_samples: int
    cutoff_len: int
    num_samples: int
    default_device_id: str
    model_arg: str


MODEL_CASES = {
    "gemma4": ModelCase(
        key="gemma4",
        st_id="DOPT-ST-GEMMA4",
        display_name="gemma4 E4B",
        model_types=("gemma4", "gemma4_text"),
        ptq_samples=8,
        cutoff_len=32,
        num_samples=256,
        default_device_id="1",
        model_arg="gemma4_model",
    ),
    "qwen3_5": ModelCase(
        key="qwen3_5",
        st_id="DOPT-ST-QWEN3_5",
        display_name="Qwen3.5 4B",
        model_types=("qwen3_5", "qwen3_5_text"),
        ptq_samples=1024,
        cutoff_len=2048,
        num_samples=256,
        default_device_id="0",
        model_arg="qwen3_5_model",
    ),
}


class STError(RuntimeError):
    """Raised when an ST precondition or execution step fails."""


def _load_json_mapping(path: Path, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise STError(f"invalid {kind} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise STError(f"{kind} must contain a JSON object: {path}")
    return value


def _validate_model_path(
    model_path_value: str | Path | None, model_case: ModelCase
) -> Path:
    if model_path_value is None:
        env_name = (
            "GEMMA4_MODEL_PATH" if model_case.key == "gemma4" else "QWEN3_5_MODEL_PATH"
        )
        raise STError(
            f"{model_case.display_name} model path is required; set {env_name} or "
            f"--{model_case.model_arg.replace('_', '-')}"
        )
    model_path = Path(model_path_value).expanduser().resolve()
    if not model_path.is_dir():
        raise STError(f"model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    config = _load_json_mapping(config_path, "model config")
    model_type = config.get("model_type")
    if model_type not in model_case.model_types:
        expected = ", ".join(model_case.model_types)
        raise STError(
            f"{model_case.display_name} expected model_type in ({expected}), "
            f"got {model_type!r}: {config_path}"
        )
    return model_path


def _validate_calibration_data(calibration_data: Path = CALIBRATION_DATA) -> Path:
    calibration_data = calibration_data.resolve()
    if not calibration_data.is_file():
        raise STError(f"built-in calibration JSONL does not exist: {calibration_data}")
    try:
        with calibration_data.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("text"), str)
                    or not row["text"]
                ):
                    raise STError(
                        f"calibration JSONL line {line_number} must contain a non-empty string field 'text'"
                    )
                return calibration_data
    except json.JSONDecodeError as exc:
        raise STError(f"invalid calibration JSONL {calibration_data}: {exc}") from exc
    raise STError(f"calibration JSONL has no data rows: {calibration_data}")


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    _validate_output_target(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _validate_output_target(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise STError(
            f"output directory must be absent or empty to prevent stale-artifact PASS: {output_dir}"
        )


def _profile_values(model_case: ModelCase, profile: str) -> tuple[int, int, int]:
    if profile == "smoke":
        return 1, 32, 1
    return model_case.ptq_samples, model_case.cutoff_len, model_case.num_samples


def _render_optimize_config(
    model_case: ModelCase,
    profile: str,
    calibration_data: Path,
    destination: Path,
    *,
    fp16: bool,
) -> Path:
    ptq_samples, cutoff_len, num_samples = _profile_values(model_case, profile)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__CALIBRATION_DATA__": json.dumps(str(calibration_data), ensure_ascii=False),
        "__PTQ_SAMPLES__": str(ptq_samples),
        "__FP16__": "true" if fp16 else "false",
        "__CUTOFF_LEN__": str(cutoff_len),
        "__NUM_SAMPLES__": str(num_samples),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    unresolved = sorted(
        placeholder for placeholder in PLACEHOLDERS if placeholder in template
    )
    if unresolved:
        raise STError(f"unresolved config placeholders: {', '.join(unresolved)}")
    destination.write_text(template, encoding="utf-8")
    try:
        config = yaml.safe_load(template)
    except yaml.YAMLError as exc:
        raise STError(f"generated optimize config is invalid: {exc}") from exc
    if config["dataset"]["train_files"] != [str(calibration_data)]:
        raise STError(
            "generated optimize config did not preserve built-in calibration_data"
        )
    if (
        config["dataset"]["ptq_samples"] != ptq_samples
        or config["cutoff_len"] != cutoff_len
    ):
        raise STError(
            f"generated optimize config does not match {model_case.key}/{profile}"
        )
    return destination


def _build_quant_command(
    python: str,
    model_path: Path,
    optimize_config: Path,
    output_dir: Path,
) -> list[str]:
    return [
        python,
        "-u",
        str(QUANT_DIR / "dopt_wrapper.py"),
        "--model-path",
        str(model_path),
        "--optimize-config",
        str(optimize_config),
        "--output-dir",
        str(output_dir),
    ]


def _runtime_env(device: str, device_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(QUANT_DIR), env.get("PYTHONPATH", "")) if item
    )
    env["DEVICE"] = device
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    if device == "npu":
        env["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    elif device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = device_id
    return env


def _run_and_tee(command: list[str], env: dict[str, str], log_path: Path) -> None:
    print(f"[RUN] {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=QUANT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise STError(
            f"dopt quantization failed with exit {return_code}; see {log_path}"
        )


def _selected_cases(selection: str) -> list[ModelCase]:
    if selection == "all":
        return list(MODEL_CASES.values())
    return [MODEL_CASES[selection]]


def _device_id(args: argparse.Namespace, model_case: ModelCase) -> str:
    value = getattr(args, f"{model_case.key}_device_id")
    return value if value is not None else model_case.default_device_id


def _run_model_case(
    args: argparse.Namespace,
    model_case: ModelCase,
    calibration_data: Path,
) -> dict[str, Any]:
    model_path = _validate_model_path(getattr(args, model_case.model_arg), model_case)
    output_dir = _prepare_output_dir(args.output_root / model_case.key)
    optimize_config = _render_optimize_config(
        model_case,
        args.profile,
        calibration_data,
        output_dir / "runtime_config.yaml",
        fp16=args.device != "cpu",
    )
    command = _build_quant_command(args.python, model_path, optimize_config, output_dir)
    ptq_samples, cutoff_len, _ = _profile_values(model_case, args.profile)
    print(
        f"[CASE] {model_case.st_id} model={model_case.display_name} profile={args.profile} "
        f"ptq_samples={ptq_samples} cutoff_len={cutoff_len} device={args.device}",
        flush=True,
    )
    _run_and_tee(
        command,
        _runtime_env(args.device, _device_id(args, model_case)),
        output_dir / "logs.log",
    )
    report = verify_output(output_dir, "state-dict")
    report["case"] = model_case.st_id
    report["profile"] = args.profile
    print(
        f"[PASS] {model_case.st_id} real quantization and output verification succeeded"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_real_cases(args: argparse.Namespace) -> int:
    calibration_data = _validate_calibration_data()
    model_cases = _selected_cases(args.models)
    for model_case in model_cases:
        _validate_model_path(getattr(args, model_case.model_arg), model_case)
        _validate_output_target(
            (args.output_root / model_case.key).expanduser().resolve()
        )
    reports = [
        _run_model_case(args, model_case, calibration_data)
        for model_case in model_cases
    ]
    print(
        f"[PASS] dopt real-model ST suite completed: {', '.join(item['case'] for item in reports)}"
    )
    return 0


def run_verify(args: argparse.Namespace) -> int:
    reports = []
    for model_case in _selected_cases(args.models):
        report = verify_output(args.output_root / model_case.key, "state-dict")
        report["case"] = model_case.st_id
        reports.append(report)
    print("[PASS] DOPT-ST-VERIFY output verification succeeded")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def _expect_failure(action: Any, expected_text: str) -> None:
    try:
        action()
    except (STError, VerificationError) as exc:
        if expected_text not in str(exc):
            raise STError(f"unexpected negative-check error: {exc}") from exc
        return
    raise STError(f"negative check unexpectedly passed; expected {expected_text!r}")


def run_check(_: argparse.Namespace) -> int:
    source_files = (
        Path(__file__),
        SCRIPT_DIR / "run.sh",
        TEMPLATE_PATH,
        SCRIPT_DIR / "verify_output.py",
    )
    machine_path_prefix = "/" + "mnt/workspace/"
    for source_file in source_files:
        if machine_path_prefix in source_file.read_text(encoding="utf-8"):
            raise STError(
                f"machine-specific path found in tracked ST source: {source_file}"
            )
    entrypoint = QUANT_DIR / "dopt_wrapper.py"
    if not entrypoint.is_file():
        raise STError(f"dopt entrypoint does not exist: {entrypoint}")
    calibration_data = _validate_calibration_data()

    with tempfile.TemporaryDirectory(prefix="dopt_st_check_") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        for model_case in MODEL_CASES.values():
            model_dir = temp_dir / f"{model_case.key}_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": model_case.model_types[0]}), encoding="utf-8"
            )
            output_dir = temp_dir / model_case.key
            output_dir.mkdir()
            rendered_config = _render_optimize_config(
                model_case,
                "demo",
                calibration_data,
                output_dir / "config.yaml",
                fp16=True,
            )
            assert _validate_model_path(model_dir, model_case) == model_dir.resolve()
            command = _build_quant_command(
                "python3", model_dir.resolve(), rendered_config, output_dir
            )
            if str(entrypoint) not in command:
                raise STError(f"{model_case.key} command does not use dopt_wrapper.py")
            dopt_config_path = output_dir / "dopt_config.json"
            strategy_fixture = {
                "layer_strategy": {
                    "model.layers.0.self_attn.q_proj": {
                        "type": "fixture.Linear",
                        "quant_strategy": "Quant_INT8-4_ECO",
                    }
                }
            }
            dopt_config_path.write_text(json.dumps(strategy_fixture), encoding="utf-8")
            fake_weight = output_dir / "trained_quant_weight.pth"
            fake_weight.write_bytes(b"contract-only fixture; not a model weight")
            report = verify_output(output_dir, "state-dict")
            if report["strategy_count"] != 1:
                raise STError("output verifier returned an unexpected strategy count")
            strategy_fixture["layer_strategy"]["model.layers.0.self_attn.q_proj"][
                "quant_strategy"
            ] = "Quant_act_weight_eco"
            dopt_config_path.write_text(json.dumps(strategy_fixture), encoding="utf-8")
            _expect_failure(
                lambda: verify_output(output_dir, "state-dict"),
                "legacy strategy Quant_act_weight_eco",
            )
            strategy_fixture["layer_strategy"]["model.layers.0.self_attn.q_proj"][
                "quant_strategy"
            ] = "Quant_INT8-4_ECO"
            dopt_config_path.write_text(json.dumps(strategy_fixture), encoding="utf-8")
            fake_weight.unlink()
            _expect_failure(
                lambda: verify_output(output_dir, "state-dict"),
                "missing or empty quantized state dict",
            )
        _expect_failure(
            lambda: _validate_model_path(None, MODEL_CASES["gemma4"]),
            "GEMMA4_MODEL_PATH",
        )
    print(
        "[PASS] DOPT-ST-CHECK gemma4/qwen3.5 config, command and fail-fast checks succeeded "
        "(no model weights or device used)"
    )
    return 0


def _add_suite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", choices=("all", "gemma4", "qwen3_5"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    check_parser = subparsers.add_parser(
        "check", help="check both real-case definitions without loading weights"
    )
    check_parser.set_defaults(handler=run_check)

    run_parser = subparsers.add_parser(
        "run", help="run the gemma4 and qwen3.5 real-model quantization cases"
    )
    _add_suite_arguments(run_parser)
    run_parser.add_argument(
        "--gemma4-model",
        default=os.environ.get("GEMMA4_MODEL_PATH"),
        help="runtime gemma4 model directory (or GEMMA4_MODEL_PATH)",
    )
    run_parser.add_argument(
        "--qwen3-5-model",
        dest="qwen3_5_model",
        default=os.environ.get("QWEN3_5_MODEL_PATH"),
        help="runtime Qwen3.5 model directory (or QWEN3_5_MODEL_PATH)",
    )
    run_parser.add_argument("--profile", choices=("demo", "smoke"), default="demo")
    run_parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    run_parser.add_argument("--gemma4-device-id", default=None)
    run_parser.add_argument(
        "--qwen3-5-device-id", dest="qwen3_5_device_id", default=None
    )
    run_parser.add_argument(
        "--python", default=sys.executable, help="Python with torch and device packages"
    )
    run_parser.set_defaults(handler=run_real_cases)

    verify_parser = subparsers.add_parser(
        "verify", help="verify outputs for one or both real-model cases"
    )
    _add_suite_arguments(verify_parser)
    verify_parser.set_defaults(handler=run_verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (STError, VerificationError, OSError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
