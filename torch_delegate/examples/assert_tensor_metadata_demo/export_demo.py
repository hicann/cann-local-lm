# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Minimal demo: _assert_tensor_metadata in torch.export and how to strip it.

This example uses ONLY standard aten ops (no custom operators, no fallback
codegen). The model's two ``tensor.to()`` calls become ``aten.to.dtype``
nodes in the exported graph. ``torch.export`` then automatically inserts
``aten._assert_tensor_metadata`` guard nodes before each ``aten.to.*`` node
(via ``_insert_aten_to_metadata_assert_pass``) to verify the input tensor's
dtype / device / layout at runtime.

These guards have no computation semantics (schema ``-> ()``) but are
rejected by downstream GE / torchair converters, so we strip them right
after export — the same technique used in
``accuracy_compare/export_program.py::_remove_assert_tensor_metadata``.

Flow:
    1. torch.export.export  -> graph contains _assert_tensor_metadata
    2. remove_assert_tensor_metadata() -> graph cleaned
    3. torch.export.save    -> .pt2
    4. torch_delegate.export_to_epair -> .epair
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from pathlib import Path

import torch
from torch import nn

try:
    import torch_npu  # noqa: F401  — torch_delegate depends on torch_npu/torchair
    import torch_delegate as td
except ImportError:
    td = None

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generate"

_ASSERT_TARGET = torch.ops.aten._assert_tensor_metadata.default


class SimpleModel(nn.Module):
    """A minimal model that triggers ``_assert_tensor_metadata`` on export.

    The two ``.to()`` calls produce ``aten.to.dtype`` nodes. torch.export
    inserts an ``_assert_tensor_metadata`` guard before each of them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.mm(x, self.weight)
        y = y.to(torch.float32)  # aten.to.dtype  -> _assert_tensor_metadata
        y = torch.relu(y)
        return y.to(torch.bfloat16)  # aten.to.dtype  -> _assert_tensor_metadata


def build_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (torch.randn(2, 4),)


def remove_assert_tensor_metadata(ep: torch.export.ExportedProgram) -> int:
    """Strip ``aten._assert_tensor_metadata`` nodes from the exported graph.

    These nodes are torch.export's runtime metadata guards (schema ``-> ()``).
    They carry no computation semantics but are rejected by downstream
    GE / torchair converters. Removing them at export time keeps both the
    epair and ``torch.compile`` paths clean.
    """
    removed = 0
    for node in list(ep.graph.nodes):
        if node.op == "call_function" and node.target is _ASSERT_TARGET:
            ep.graph.erase_node(node)
            removed += 1
    if removed:
        ep.graph.lint()
        ep.graph_module.recompile()
    return removed


def _count_asserts(ep: torch.export.ExportedProgram) -> int:
    return sum(
        1
        for n in ep.graph.nodes
        if n.op == "call_function" and n.target is _ASSERT_TARGET
    )


def _print_summary(ep: torch.export.ExportedProgram, label: str) -> None:
    n = _count_asserts(ep)
    print(f"\n{'=' * 64}")
    print(f"[{label}]  _assert_tensor_metadata nodes: {n}")
    print(f"{'=' * 64}")
    for node in ep.graph.nodes:
        if node.op == "call_function" and node.target is _ASSERT_TARGET:
            print(f"  {node.name}  kwargs={dict(node.kwargs)}")
    print()
    print(ep.graph)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = SimpleModel().eval()
    example_inputs = build_inputs()

    # ---- Step 1: torch.export ----
    print("Step 1  torch.export.export(strict=False)")
    ep = torch.export.export(model, example_inputs, strict=False)
    _print_summary(ep, "BEFORE removal")
    (OUTPUT_DIR / "graph_before.py").write_text(str(ep.graph), encoding="utf-8")

    # ---- Step 2: strip _assert_tensor_metadata ----
    print("\nStep 2  remove_assert_tensor_metadata()")
    removed = remove_assert_tensor_metadata(ep)
    print(f"  removed {removed} node(s)")
    _print_summary(ep, "AFTER removal")
    (OUTPUT_DIR / "graph_after.py").write_text(str(ep.graph), encoding="utf-8")

    # ---- Step 3: save .pt2 ----
    pt2_path = OUTPUT_DIR / "simple_model.pt2"
    torch.export.save(ep, pt2_path)
    print(f"\nStep 3  ExportedProgram -> {pt2_path}")

    # ---- Step 4: export to epair ----
    if td is not None:
        print("\nStep 4  torch_delegate.export_to_epair()")
        epair = td.export_to_epair(ep)
        epair_path = OUTPUT_DIR / "simple_model.epair"
        td.epair_save(epair, epair_path)
        print(f"  epair -> {epair_path}")
    else:
        print("\nStep 4  (skipped - torch_delegate not installed)")

    # ---- Step 5: sanity check + save reference ----
    with torch.no_grad():
        ref = model(*example_inputs)
        out = ep.module()(*example_inputs)
    diff = (ref - out).abs().max().item()
    print(f"\nStep 5  verify  shape={out.shape} dtype={out.dtype}  max|diff|={diff}")

    # save inputs as binary (float32 for tensor, + input_meta.txt)
    meta_lines = []
    for i, inp in enumerate(example_inputs):
        p = OUTPUT_DIR / f"input_{i}.bin"
        inp.cpu().float().detach().numpy().tofile(p)
        meta_lines.append(f"{i}: x  dtype=float32  shape={list(inp.shape)}")
        print(f"  input[{i}] shape={list(inp.shape)} -> {p}")
    (OUTPUT_DIR / "input_meta.txt").write_text(
        "\n".join(meta_lines) + "\n", encoding="utf-8"
    )

    # save reference output (cast to float32 for easy C++ comparison)
    ref_path = OUTPUT_DIR / "ref_output.bin"
    ref.cpu().float().detach().numpy().tofile(ref_path)
    print(f"  ref_output shape={list(ref.shape)} dtype={ref.dtype} -> {ref_path}")

    print(f"\nDone  files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
