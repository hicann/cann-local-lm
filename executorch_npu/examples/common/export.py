# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PT2 serialization and Ascend PTE compilation steps."""

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Mapping


# Serializes the install -> to_edge -> restore window of the EP-release shim
# below: concurrent compile_pte calls would otherwise nest monkey-patches and
# leak the first shim (its closure pinning the exported dict) onto the module.
# Consistent with the backend's compile_lock, which already serializes GE
# compilation across threads.
_EDGE_SHIM_LOCK = threading.Lock()


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


def _malloc_trim() -> None:
    """Return freed heap pages to the OS (glibc arenas keep them otherwise)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _release_eps_after_edge(exported: dict) -> object:
    """Install a scoped shim releasing the original ExportedPrograms early.

    ``to_edge_transform_and_lower`` never touches its ``programs`` dict after
    ``_gen_edge_manager_for_partitioners`` returns (the GE lowering inside
    ``to_backend`` works off the edge manager only). Clearing the dict at that
    point frees the EP-resident weights (~archive size x method count) before
    the memory-heavy GE phase, which is what makes 3-method joint compile fit
    a 70GB host. Returns a restore() callable; no-op if the executorch
    internals moved (joint compile then behaves as before).
    """
    import gc

    try:
        from executorch.exir.program import _program as _exir
    except Exception:
        return lambda: None
    original = getattr(_exir, "_gen_edge_manager_for_partitioners", None)
    if original is None:
        return lambda: None

    def shim(partitioner_arg, aten_programs, *args, **kwargs):
        result = original(partitioner_arg, aten_programs, *args, **kwargs)
        if aten_programs is exported:
            aten_programs.clear()
            gc.collect()
            _malloc_trim()
        return result

    _exir._gen_edge_manager_for_partitioners = shim

    def restore() -> None:
        _exir._gen_edge_manager_for_partitioners = original

    return restore


def _dedup_cross_method_weights(exported: dict) -> int:
    """Share byte-identical weight storages across methods' ExportedPrograms.

    prefill/decode PT2s embed the same LLM weights; torch.export.load
    materializes an independent copy per EP. For each tensor pair (same key,
    shape, dtype, contiguous) verified byte-equal, re-point the later EP's
    tensor onto the reference storage in place (tensor.set_), so every
    reference holder sees shared storage and the duplicate pages are freed.
    Returns bytes freed. Verified: qwen mm pair 3.92GB -> 2.61GB cur RSS,
    state_dict/range_constraints intact.
    """
    import gc

    import torch

    def sd_bytes(ep) -> int:
        return sum(t.numel() * t.element_size() for t in ep.state_dict.values())

    names = sorted(exported, key=lambda n: -sd_bytes(exported[n]))
    if len(names) < 2:
        return 0
    ref_sd = exported[names[0]].state_dict
    freed = 0
    for name in names[1:]:
        sd = exported[name].state_dict
        for key, tensor in sd.items():
            ref = ref_sd.get(key)
            if (
                ref is not None
                and ref is not tensor
                and ref.shape == tensor.shape
                and ref.dtype == tensor.dtype
                and ref.is_contiguous()
                and tensor.is_contiguous()
                # A contiguous view with a non-zero storage offset (or one
                # sitting inside an oversized storage) would pass the guards
                # above, while set_(storage, 0, ...) reads from the storage
                # start - silent weight corruption. Require the reference to
                # own its storage exactly (also avoids pinning a larger
                # underlying storage alive).
                and ref.storage_offset() == 0
                and ref.untyped_storage().nbytes() == ref.numel() * ref.element_size()
                and torch.equal(ref, tensor)
            ):
                # set_ is an in-place op; autograd rejects it on leaf
                # tensors that require grad (EPs exported from modules with
                # trainable params). Toggle the flag off/on around the swap.
                wanted = tensor.requires_grad
                if wanted:
                    tensor.requires_grad_(False)
                with torch.no_grad():
                    tensor.set_(ref.untyped_storage(), 0, ref.size(), ref.stride())
                if wanted:
                    tensor.requires_grad_(True)
                freed += ref.numel() * ref.element_size()
    if freed:
        gc.collect()
        _malloc_trim()
    print(f"[DEDUP] cross-method weight storages shared: {freed / 2**30:.2f}GB")
    return freed


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
    _dedup_cross_method_weights(exported)
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
    with _EDGE_SHIM_LOCK:
        restore = _release_eps_after_edge(exported)
        try:
            program = to_edge_transform_and_lower(
                exported,
                partitioner=partitioners,
                compile_config=aten_edge_compile_config(),
            ).to_executorch()
        finally:
            restore()
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
