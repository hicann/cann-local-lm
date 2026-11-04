# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PT2 serialization and Ascend PTE compilation steps."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Mapping


def save_pt2(
    wrapper,
    inputs: dict,
    output: Path,
    dynamic_shapes=None,
    *,
    timings: dict | None = None,
) -> Path:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(
            wrapper, args=(), kwargs=inputs, dynamic_shapes=dynamic_shapes, strict=False
        )
    captured = time.perf_counter()
    torch.export.save(program, output)
    if timings is not None:
        counts = Counter(
            str(n.target) for n in program.graph.nodes if n.op == "call_function"
        )
        timings.update(
            capture_s=captured - started,
            serialize_s=time.perf_counter() - captured,
            graph_nodes=len(list(program.graph.nodes)),
            target_counts=dict(counts),
        )
    print(output)
    return output


def compile_pte(
    methods: Mapping[str, Path],
    output_dir: Path,
    *,
    acl_soc_name: str,
    roles: Mapping[str, str],
    precision_mode: str | None = "origin",
    register_npu_ops: bool = True,
    timings: dict | None = None,
) -> Path:
    import torch

    if register_npu_ops:
        import torch_npu  # noqa: F401

        # Saved PT2 references the AIR op before any model forward runs.
        import torchair.ops._npu_fused_infer_attention_score_v2  # noqa: F401
    from executorch.exir import to_edge_transform_and_lower

    from executorch_npu import (
        AscendCompileOptions,
        AscendPartitioner,
        aten_edge_compile_config,
    )
    from executorch_npu.passes import lower_softmax_int

    started = time.perf_counter()
    exported = {name: torch.export.load(path) for name, path in methods.items()}
    for program in exported.values():
        lower_softmax_int(program)
    partitioners = {}
    for name in exported:
        compile_kwargs = {
            "method_name": name,
            "delegate_index": 0,
            "acl_soc_name": acl_soc_name,
            "role": roles[name],
        }
        if precision_mode is not None:
            compile_kwargs["offline_build_options"] = {
                "ge.exec.precision_mode_v2": precision_mode
            }
        options = AscendCompileOptions(**compile_kwargs)
        partitioners[name] = [AscendPartitioner(options.to_compile_specs())]

    loaded = time.perf_counter()
    program = to_edge_transform_and_lower(
        exported,
        partitioner=partitioners,
        compile_config=aten_edge_compile_config(),
    ).to_executorch()
    compiled = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    pte = output_dir / "model.pte"
    pte.write_bytes(program.buffer)
    program.write_tensor_data_to_file(output_dir)
    if timings is not None:
        timings.update(
            load_and_prepare_s=loaded - started,
            lower_and_compile_s=compiled - loaded,
            write_s=time.perf_counter() - compiled,
            total_s=time.perf_counter() - started,
        )
    print(pte)
    return pte
