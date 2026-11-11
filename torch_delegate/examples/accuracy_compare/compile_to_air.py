# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Demo entrypoint for platform-side ExportedProgram to AIR compilation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_delegate
from torch.export.passes import move_to_device_pass

_ROOT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(_ROOT_DIR))
from ops import _LIBRARY as _placeholder_ops  # noqa: E402, F401
from ge_converters import register_beam_search_converter  # noqa: E402, F401
from placeholder_to_custom_pass import rewrite_placeholder_to_custom  # noqa: E402, F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile demo .pt2 to .air.")
    parser.add_argument(
        "--ep-path",
        default=str(_ROOT_DIR / "generate/sample.pt2"),
    )
    parser.add_argument(
        "--air-path",
        default=str(_ROOT_DIR / "generate/sample.air"),
    )
    parser.add_argument("--rewrite-only", action="store_true")
    parser.add_argument(
        "--compile-device",
        default="npu:0",
        help="Device used to run TorchAir export.",
    )
    return parser.parse_args()


def _is_npu_device(device: str | torch.device) -> bool:
    device_type = str(device).split(":", maxsplit=1)[0]
    return device_type == "npu"


def _resolve_tensor_device(compile_device: str | torch.device) -> torch.device:
    if not _is_npu_device(compile_device):
        return torch.device(compile_device)

    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
        raise RuntimeError(
            "compile_device=npu requires torch_npu because TorchAir export "
            "needs real NPU example_inputs, not meta tensors."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "NPU compile requires torch_npu and CANN runtime. "
            "Please source the CANN environment first (set_env.sh), or set the ASCEND_HOME_PATH environment variable."
        ) from exc
    return torch.device(compile_device)


def _new_tensor_like(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = tuple(tensor.shape)
    if tensor.layout != torch.strided:
        return torch.empty_like(tensor, device=device)
    try:
        return torch.empty_strided(
            shape,
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=device,
        )
    except RuntimeError:
        return torch.empty(shape, dtype=tensor.dtype, device=device)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(
            _to_device(value.detach(), device),
            requires_grad=value.requires_grad,
        )
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            return _new_tensor_like(value, device)
        if value.device == device:
            return value
        if value.device.type == "cuda" and device.type == "npu":
            return value.cpu().to(device)
        return value.to(device)
    if isinstance(value, torch.device):
        if value.type in ("cuda", "cpu"):
            return device
        return value
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return type(value)(
            (key, _to_device(item, device)) for key, item in value.items()
        )
    return value


def _retarget_graph_devices(
    exported_program: torch.export.ExportedProgram,
    device: torch.device,
) -> None:
    for node in exported_program.graph.nodes:
        node.args = _to_device(node.args, device)
        node.kwargs = _to_device(node.kwargs, device)
    exported_program.graph.lint()
    exported_program.graph_module.recompile()


def _move_exported_program_to_device(
    exported_program: torch.export.ExportedProgram,
    device: torch.device,
) -> torch.export.ExportedProgram:
    _retarget_graph_devices(exported_program, device)

    return torch.export.ExportedProgram(
        exported_program.graph_module,
        exported_program.graph,
        exported_program.graph_signature,
        _to_device(exported_program.state_dict, device),
        exported_program.range_constraints,
        exported_program.module_call_graph,
        example_inputs=_to_device(exported_program.example_inputs, device),
        constants=_to_device(exported_program.constants, device),
        verifiers=exported_program.verifiers,
    )


def _example_inputs(
    exported_program: torch.export.ExportedProgram,
    device: torch.device,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if exported_program.example_inputs is None:
        raise ValueError("ExportedProgram does not carry example_inputs.")
    args, kwargs = exported_program.example_inputs
    return _to_device(tuple(args), device), _to_device(dict(kwargs), device)


def compile_ep_to_air(
    ep_path: str | Path,
    air_path: str | Path,
    rewrite_only: bool = False,
    compile_device: str | torch.device = "npu:0",
) -> Path:
    air_path = Path(air_path).resolve()
    output_dir = air_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(Path(ep_path).resolve())
    rewritten_program = rewrite_placeholder_to_custom(exported_program)
    device = _resolve_tensor_device(compile_device)

    if not rewrite_only:
        rewritten_program = _move_exported_program_to_device(
            rewritten_program,
            device,
        )

    rewritten_path = output_dir / f"{air_path.stem}.rewritten.pt2"
    torch.export.save(rewritten_program, rewritten_path)
    rewritten_path.with_suffix(".graph.py").write_text(
        rewritten_program.graph_module.code,
        encoding="utf-8",
    )

    if rewrite_only:
        return rewritten_path

    register_beam_search_converter()
    epair_path = output_dir / "rewritten.epair"
    rewritten_program = move_to_device_pass(rewritten_program, str(device))
    epair = torch_delegate.export_to_epair(rewritten_program)
    torch_delegate.epair_save(epair, epair_path)
    return epair_path
    # args, kwargs = _example_inputs(rewritten_program, device)
    # model = rewritten_program.module().to(device)

    # torchair.dynamo_export(
    #     *args,
    #     model=model,
    #     export_path=str(output_dir),
    #     export_name=air_path.stem,
    #     **kwargs,
    # )

    # return air_path


def main() -> None:
    args = _parse_args()
    output_path = compile_ep_to_air(
        ep_path=args.ep_path,
        air_path=args.air_path,
        rewrite_only=args.rewrite_only,
        compile_device=args.compile_device,
    )
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
