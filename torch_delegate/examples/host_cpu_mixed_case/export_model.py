#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
HostCpu 混合调度示例 - 导出与归属校验

导出三种形态的模型并校验 Device 归属结果（manifest v2 + ascend_ir.device）：
- model_mixed:       NPU → CPU → NPU（CPU 段夹在 NPU 段中间）
- model_pure_cpu:    纯 CPU 段
- model_non_adjacent: CPU → NPU → CPU（非相邻 CPU 段，共享输入）
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401,E402

import torch_delegate as td  # noqa: E402


class ModelMixed(torch.nn.Module):
    """x → mm(NPU) → feature_scale(CPU) → relu(NPU)"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8) * 0.1)

    def forward(self, x):
        z = torch.mm(x, self.weight)
        z = torch.ops.hcpu_ops.feature_scale(z, 2.0)
        return torch.relu(z)


class ModelPureCpu(torch.nn.Module):
    """x → feature_scale(CPU) → feature_merge(CPU, x)"""

    def forward(self, x):
        s = torch.ops.hcpu_ops.feature_scale(x, 0.5)
        return torch.ops.hcpu_ops.feature_merge(s, x)


class ModelNonAdjacent(torch.nn.Module):
    """x → feature_scale(CPU) → mm(NPU) → relu(NPU) → feature_merge(CPU, relu_out, x)"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8) * 0.1)

    def forward(self, x):
        s = torch.ops.hcpu_ops.feature_scale(x, 1.5)
        z = torch.mm(s, self.weight)
        z = torch.relu(z)
        return torch.ops.hcpu_ops.feature_merge(z, x)


class ModelRecommender(torch.nn.Module):
    """推荐网络三段式：

    CPU 段1 embedding 查表 → NPU 段 3 层 MLP（Task Tower）→ CPU 段2 排序打分
    """

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(16, 8) * 0.1)
        self.w1 = torch.nn.Parameter(torch.randn(8, 32) * 0.1)
        self.w2 = torch.nn.Parameter(torch.randn(32, 32) * 0.1)
        self.w3 = torch.nn.Parameter(torch.randn(32, 8) * 0.1)
        self.rank_bias = torch.nn.Parameter(torch.randn(4, 8) * 0.01)

    def forward(self, ids_f):
        # ids 以 float32 传入（规避 GE HostCpu staging 对 DT_INT64 Data 输入的截断，
        # 见 custom_ops.py feature_gather 注释），kernel 内转回 int64 查表。
        v = torch.ops.hcpu_ops.feature_gather(self.embedding, ids_f)
        h = torch.relu(torch.mm(v, self.w1))
        h = torch.relu(torch.mm(h, self.w2))
        scores = torch.mm(h, self.w3)
        return torch.ops.hcpu_ops.score_rank(scores, self.rank_bias)


MODELS = {
    "model_mixed": ModelMixed(),
    "model_pure_cpu": ModelPureCpu(),
    "model_non_adjacent": ModelNonAdjacent(),
    "model_recommender": ModelRecommender(),
}

EXPECTED_CPU_OPS = {
    "model_mixed": {"hcpu_ops.feature_scale.default"},
    "model_pure_cpu": {
        "hcpu_ops.feature_scale.default",
        "hcpu_ops.feature_merge.default",
    },
    "model_non_adjacent": {
        "hcpu_ops.feature_scale.default",
        "hcpu_ops.feature_merge.default",
    },
    "model_recommender": {
        "hcpu_ops.feature_gather.default",
        "hcpu_ops.score_rank.default",
    },
}


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    data = tensor.detach().cpu().contiguous().float().numpy()
    path.write_bytes(data.tobytes())
    shape_path = path.with_suffix(path.suffix + "_shape.txt") if path.suffix else path
    shape_path.write_text(",".join(str(d) for d in data.shape))


def export_one(name: str, module, x: torch.Tensor, output_dir: Path) -> None:
    print(f"\n{'=' * 60}\n导出 {name}\n{'=' * 60}")
    module.eval()
    with torch.no_grad():
        ref = module(x)
    assert ref.dtype == torch.float32, f"{name}: 输出应为 float32，实际 {ref.dtype}"

    ep = torch.export.export(module, (x,))
    contents = td.export_to_epair(ep)

    manifest = contents.manifest
    print(
        f"  manifest: version={manifest.version}, cpu_nodes={manifest.cpu_nodes}, "
        f"converted={manifest.converted_nodes}/{manifest.total_nodes}"
    )
    assert manifest.version == 2, f"manifest version 应为 2，实际 {manifest.version}"
    expected_cpu = len(EXPECTED_CPU_OPS[name])
    assert manifest.cpu_nodes == expected_cpu, (
        f"{name}: cpu_nodes 应为 {expected_cpu}，实际 {manifest.cpu_nodes}"
    )

    cpu_ops = set()
    for node in contents.ep_json["graph_module"]["graph"]["nodes"]:
        ascend_ir = node.get("ascend_ir")
        if ascend_ir and ascend_ir.get("device") == "cpu":
            cpu_ops.add(node["target"].removeprefix("torch.ops."))
            assert ascend_ir["source"] == "", "CPU 归属仅出现在 fallback 节点"
    assert cpu_ops == EXPECTED_CPU_OPS[name], (
        f"{name}: CPU 算子集合不符，实际 {cpu_ops}，预期 {EXPECTED_CPU_OPS[name]}"
    )
    print(f"  ascend_ir.device=cpu 节点: {sorted(cpu_ops)}")

    epair_path = output_dir / f"{name}.epair"
    td.epair_save(contents, epair_path)
    save_tensor(output_dir / f"input_{name.removeprefix('model_')}_x.bin", x)
    save_tensor(output_dir / f"ref_{name.removeprefix('model_')}.bin", ref)
    print(f"  保存 epair: {epair_path}")


def main() -> None:
    torch.manual_seed(2026)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(4, 8)
    # ids 用 float32 传入（GE HostCpu staging 对 DT_INT64 Data 输入截断的规避，
    # 见 custom_ops.py feature_gather 注释）
    ids_f = torch.tensor([0.0, 3.0, 1.0, 7.0], dtype=torch.float32)
    for name, module in MODELS.items():
        export_one(
            name, module, ids_f if name == "model_recommender" else x, output_dir
        )

    print(f"\n{'=' * 60}\n所有模型导出并校验通过\n{'=' * 60}")


if __name__ == "__main__":
    main()
