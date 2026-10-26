# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""导出“深而宽”的 matmul 模型（y = f^depth(x)）为 epair 及输入数据。

结构：每层由 `--branches` 条**独立**的 MatMul+ReLU 分支相加，再与层 bias 相加并 ReLU，
堆叠 `--depth` 层。权重按分支复用（不同层共享同一组权重常量）。

设计意图（配合并发验证）：
- 深层数 + 多算子：单次 RunModel 的 Device 侧执行时间达到毫秒级以上，
  便于稳定测量多实例/多流并发效果；
- 图内存在大量互相独立的分支：图本身是长序列，用于观测跨进程多流并发
  （GE 图内自动多流为实验特性，本示例暂不开启）；
- 权重跨层复用：常量规模 = branches 个 dim x dim，避免深层堆叠导致显存膨胀。

用法：
    python3 export_matmul_epair.py [--dim 1024] [--depth 24] [--branches 8] [--out-dir .]

生成：
    matmul_model.epair   # 含常量权重/偏置，用户输入为 x
    input_0.bin          # x 的 float32 原始数据（dim x dim）
"""

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (注册 NPU 后端)
import torch_delegate as td


class DeepMatMulModel(torch.nn.Module):
    def __init__(self, dim: int, depth: int, branches: int):
        super().__init__()
        self.depth = depth
        self.branches = branches
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dim, dim)) for _ in range(branches)]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dim)) for _ in range(depth)]
        )

    def forward(self, x):
        out = x
        for layer in range(self.depth):
            h = torch.relu(torch.mm(out, self.weights[0]))
            for b in range(1, self.branches):
                h = h + torch.relu(torch.mm(out, self.weights[b]))
            out = torch.relu(h + self.biases[layer])
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24, help="堆叠层数")
    parser.add_argument("--branches", type=int, default=8, help="每层独立并行分支数")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    model = DeepMatMulModel(args.dim, args.depth, args.branches).eval().npu()
    example_input = (torch.randn(args.dim, args.dim).npu(),)

    print(
        f"Exporting deep model: dim={args.dim} depth={args.depth} branches={args.branches} "
        f"(~{args.depth * (3 * args.branches + 1)} ops: per layer {args.branches} MatMul + "
        f"{args.branches + 1} ReLU + {args.branches} Add)"
    )
    ep = torch.export.export(model, example_input)
    epair_path = root / "matmul_model.epair"
    td.epair_save(td.export_to_epair(ep), epair_path)
    print(f"Saved {epair_path}")

    input_path = root / "input_0.bin"
    example_input[0].cpu().float().detach().numpy().tofile(input_path)
    print(f"Saved input[0] shape={list(example_input[0].shape)} -> {input_path}")


if __name__ == "__main__":
    main()
