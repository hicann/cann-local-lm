# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""torchair torch.compile 路径 Pack 优化示例.

对比 torch.export 路径 (export_pack_model.py), 验证 torchair
torch.compile(dynamic=True) 的 optimize_sym_pack Pass 能否消除冗余 Pack.

模型与 export_pack_model.py 完全一致:
  forward(self, x, m, n)
    x: [a, b]  tensor, numel = a*b
    m: int     SymInt (不等于 a 或 b)
    n: int     SymInt (不等于 a 或 b), m*n == a*b

torchair torch.compile(dynamic=True) 路径:
  int 参数 m, n → ge.Data(placement='CPU', shape=[])
  view([m, n]) → parse_symlist → ge.Pack([Data_m, Data_n])
  → _inputs_all_sym = True (输入全是 CPU Data)
  → optimize_sym_pack 合并冗余 Pack

预期: 6 Pack → 0 Pack (100% 消除)
"""

import re
from pathlib import Path

import torch
import torch.nn as nn
import torchair


class PackReproduceModel(nn.Module):
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


def _get_pack_data_count(concrete_graph):
    pack_count = 0
    data_count = 0
    for node in concrete_graph.graph.op:
        if node.type == "Pack":
            pack_count += 1
        if node.type == "Data":
            data_count += 1
    return pack_count, data_count


_original_optimize = None


def _install_pack_counter():
    from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph

    global _original_optimize
    _original_optimize = GeConcreteGraph.optimize_graph_without_runtime

    def counting_wrapper(self, *args, **kwargs):
        before_pack, before_data = _get_pack_data_count(self)
        print(f"\n{'=' * 60}")
        print(f"  [before] Pack: {before_pack}, Data: {before_data}")
        print(f"{'=' * 60}")

        result = _original_optimize(self, *args, **kwargs)

        after_pack, after_data = _get_pack_data_count(self)
        print(f"  [after]  Pack: {after_pack}, Data: {after_data}")
        if before_pack > 0:
            eliminated = before_pack - after_pack
            print(
                f"  [result] eliminated Pack: {eliminated} "
                f"({100 * eliminated // before_pack}%)"
            )
        print(f"{'=' * 60}\n")
        return result

    GeConcreteGraph.optimize_graph_without_runtime = counting_wrapper


def _restore_pack_counter():
    from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph

    GeConcreteGraph.optimize_graph_without_runtime = _original_optimize


def _count_in_pbtxt(pbtxt_path):
    if not pbtxt_path.exists():
        return None, None
    content = pbtxt_path.read_text()
    pack_count = len(re.findall(r'op:\s*"Pack"', content))
    data_count = len(re.findall(r'op:\s*"Data"', content))
    return pack_count, data_count


def main():
    _install_pack_counter()

    config = torchair.CompilerConfig()
    dump_dir = Path(__file__).resolve().parent / "graph_dump_output"
    dump_dir.mkdir(exist_ok=True)
    config.debug.graph_dump.type = "pbtxt"
    config.debug.graph_dump.path = str(dump_dir)

    model = PackReproduceModel().eval()
    compiled = torch.compile(
        model,
        backend=torchair.get_npu_backend(compiler_config=config),
        dynamic=True,
    )

    x = torch.randn(4, 3)
    m_val, n_val = 2, 6

    print(f"Input: x={list(x.shape)}, m={m_val}, n={n_val}, m*n={m_val * n_val}")
    print("Expected: 6 Pack (int params -> CPU Data -> _inputs_all_sym=True)")
    print("          optimize_sym_pack eliminates all redundant Pack")
    print(f"pbtxt dump dir: {dump_dir}")

    output = compiled(x, m_val, n_val)
    print(f"\nOutput: {output.item():.6f}")

    _restore_pack_counter()

    print(f"\n{'=' * 60}")
    print("  pbtxt dump analysis")
    print(f"{'=' * 60}")

    pbtxt_files = sorted(dump_dir.glob("*.pbtxt"))
    if not pbtxt_files:
        print("  No pbtxt files found")
    else:
        originals = sorted([f for f in pbtxt_files if "original" in f.name])
        optimized = sorted([f for f in pbtxt_files if "optimized" in f.name])
        latest_orig = originals[-1] if originals else None
        latest_opt = optimized[-1] if optimized else None

        for label, f in [
            ("original (before pass)", latest_orig),
            ("optimized (after pass)", latest_opt),
        ]:
            if f is None:
                print(f"  [{label}] not found")
                continue
            pack_cnt, data_cnt = _count_in_pbtxt(f)
            print(f"  [{label}] Pack={pack_cnt}, Data={data_cnt}")


if __name__ == "__main__":
    main()
