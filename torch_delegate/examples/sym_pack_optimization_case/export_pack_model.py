# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import base64
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.export import Dim

import torch_delegate as td

try:
    from torchair.core.utils import logger

    logger.setLevel(logging.WARN)
except ImportError:
    pass

OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PackReproduceModel(nn.Module):
    """int 参数做 view, 产生 _inputs_all_sym=True 的冗余 Pack.

    当 int 参数 (m, n) 不等于任何 tensor 维度时, _optimize_sym_input
    无法将 m/n 匹配到 tensor.size(), SymInt 保留为独立 placeholder,
    在 GE 图中变为 CPU Data 节点, Pack 输入为 Data, _inputs_all_sym=True.

    Group A: view([m, n]) × 4 → 4 Pack[Data_m, Data_n] (3 冗余)
    Group B: view([n, m]) × 2 → 2 Pack[Data_n, Data_m] (1 冗余)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, m, n):
        a1 = x.view([m, n])
        a2 = x.view([m, n])
        a3 = x.view([m, n])
        a4 = x.view([m, n])
        sum_a = (a1 + a2 + a3 + a4).sum()

        b1 = x.view([n, m])
        b2 = x.view([n, m])
        sum_b = (b1 + b2).sum()

        return sum_a + sum_b


def _analyze_packs_in_epair(epair_path):
    with zipfile.ZipFile(str(epair_path), "r") as zf:
        ir_json_name = None
        for name in zf.namelist():
            if "ascend_ir" in name or "annotated" in name:
                ir_json_name = name
                break
        if ir_json_name is None:
            print("  [WARN] ascend_ir JSON not found")
            return
        data = json.loads(zf.read(ir_json_name).decode("utf-8"))

    const_values = {}
    nodes = data.get("graph_module", {}).get("graph", {}).get("nodes", [])
    for node in nodes:
        for op in node.get("ascend_ir", {}).get("ops", []):
            if op.get("type") == "Const":
                readable = op.get("attr", {}).get("_readable_value", {}).get("s", "")
                if readable:
                    try:
                        decoded = base64.b64decode(readable).decode("utf-8")
                    except Exception:
                        decoded = "?"
                else:
                    decoded = "?"
                const_values[op["name"]] = decoded

    pack_ops = []
    data_count = gather_count = 0
    for node in nodes:
        for op in node.get("ascend_ir", {}).get("ops", []):
            if op["type"] == "Pack":
                inputs = op.get("input", [])
                sem = [
                    f"Const({const_values[inp.split(':')[0]]})"
                    if inp.split(":")[0] in const_values
                    else inp
                    for inp in inputs
                ]
                has_sym = "_inputs_all_sym" in op.get("attr", {})
                pack_ops.append((op["name"], sem, has_sym))
            elif op["type"] == "Data":
                data_count += 1
            elif op["type"] == "Gather":
                gather_count += 1

    print(
        f"  GE graph ops: Data={data_count} Gather={gather_count} Pack={len(pack_ops)}"
    )
    for name, sem, has_flag in pack_ops:
        print(f"    {name}: inputs={sem}  _inputs_all_sym={has_flag}")

    sem_groups = Counter(tuple(sem) for _, sem, _ in pack_ops)
    total = len(pack_ops)
    unique = len(sem_groups)
    redundant = total - unique
    print(f"  Pack total={total} unique={unique} redundant={redundant}")
    all_sym = all(has_flag for _, _, has_flag in pack_ops) if pack_ops else False
    print(f"  _inputs_all_sym={'True' if all_sym else 'False'}")


def main():
    model = PackReproduceModel().eval()

    x = torch.randn(4, 3)
    m_val, n_val = 2, 6
    dynamic_shapes = {
        "x": {0: Dim.AUTO, 1: Dim.AUTO},
        "m": Dim.AUTO,
        "n": Dim.AUTO,
    }

    print(f"Input: x={list(x.shape)}, m={m_val}, n={n_val}, m*n={m_val * n_val}")

    ep = torch.export.export(model, (x, m_val, n_val), dynamic_shapes=dynamic_shapes)

    pt2_path = OUTPUT_DIR / "model.pt2"
    torch.export.save(ep, pt2_path)
    print(f"Saved {pt2_path}")

    contents = td.export_to_epair(ep)
    epair_path = OUTPUT_DIR / "model.epair"
    td.epair_save(contents, epair_path)
    print(f"Saved {epair_path}")
    print(
        f"Converted: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}"
    )

    print("\n=== Pack analysis ===")
    _analyze_packs_in_epair(epair_path)

    # Save inputs
    x.float().cpu().numpy().tofile(str(OUTPUT_DIR / "input_0.bin"))
    np.array([m_val], dtype=np.int64).tofile(str(OUTPUT_DIR / "input_1.bin"))
    np.array([n_val], dtype=np.int64).tofile(str(OUTPUT_DIR / "input_2.bin"))
    meta = (
        f"0: x  dtype=float32  shape={list(x.shape)}\n"
        f"1: m  dtype=int64  shape=[1]\n"
        f"2: n  dtype=int64  shape=[1]\n"
    )
    (OUTPUT_DIR / "input_meta.txt").write_text(meta)
    print("\nSaved input binaries")

    # Load and verify
    epair = td.epair_load(str(epair_path))
    loaded_model = epair.ep.module()

    class _NoopGuards(nn.Module):
        def forward(self, *args):
            return None

    if hasattr(loaded_model, "_guards_fn"):
        loaded_model._guards_fn = _NoopGuards()

    output1 = loaded_model(x, m_val, n_val)
    print(f"Output (x=[4,3], m=2, n=6): {output1.item():.6f}")

    new_x = torch.randn(3, 4)
    output2 = loaded_model(new_x, 3, 4)
    print(f"Output (x=[3,4], m=3, n=4): {output2.item():.6f}")

    output1.float().cpu().numpy().tofile(str(OUTPUT_DIR / "ref_output.bin"))
    print(f"Saved ref_output.bin (value={output1.item():.6f})")


if __name__ == "__main__":
    main()
