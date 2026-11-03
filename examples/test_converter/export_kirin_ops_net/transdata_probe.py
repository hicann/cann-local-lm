# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransData 探针: 测几个最小候选网, 找出 ND 路径能产生 TransData 的最小网 (CLI-015 L2).

候选 (从小到大):
  single     : x[1,K] @ q_proj[K,N]            (单 MatMul, 2D 权重, 基线, 预期 0 TransData)
  linear3d   : x[2,8,K] @ q_proj[K,N]           (Linear 作用 3D, 广播 batched matmul, 2D 权重)
  bmm        : x[2,8,K] bmm q_proj^T[K,N]      (真 bmm 3D@3D, 2D 权重转置 broadcast)
  two_linear : x[1,K] -> q_proj -> k_proj       (两条链式 Linear)

用法 (两阶段 env):
  # ① converter env: 导出+编译各候选
  python transdata_probe.py export --net single --output out/td_single.pt2
  python transdata_probe.py compile --ep-path out/td_single.pt2 --output-dir out/td_single_epair
  # ② CANN env: cann_lm_compile + atc --mode=1, 数 TransData
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import Tensor, nn

K = int(os.environ.get("PROBE_K", "128"))
N = int(os.environ.get("PROBE_N", "128"))


class SingleLinear(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.q_proj(x)


class Linear3D(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.q_proj(x)


class Bmm3D(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        w = self.q_proj.weight.t()
        return torch.bmm(x, w.unsqueeze(0).expand(x.shape[0], K, N))


class TwoLinear(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)
        self.k_proj = nn.Linear(N, K, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.k_proj(self.q_proj(x))


class AddMatMul(nn.Module):
    """Add (compute op) before MatMul — triggers TransData on Add output -> MatMul input."""

    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)
        self.bias = nn.Parameter(torch.zeros(1, K, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return self.q_proj(x + self.bias)


class TransposeMatMul(nn.Module):
    """Transpose before MatMul — triggers TransData on Transpose output -> MatMul input."""

    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = nn.Linear(K, N, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:  # x [K, M=8] -> t() [M, K] -> q_proj [M, N]
        return self.q_proj(x.t())


class _BareWeight(nn.Module):
    """A bare parameter submodule so the state_dict key is `<name>.weight` (matches *_proj.weight)."""

    def __init__(self, weight: Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)


class BareMatMul(nn.Module):
    """x[M,K] @ q_proj.weight[K,N] via bare matmul (no nn.Linear / F.linear / weight.T -> no Transpose_1).

    The weight is stored pre-transposed [K,N] (not nn.Linear's [N,K]), so torch.export traces
    `aten.matmul(x, weight)` directly — no F.linear decomposition, no Transpose on the weight.
    """

    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.q_proj = _BareWeight(torch.randn(K, N, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:  # x [1, K] @ weight [K, N] -> [1, N]
        return x @ self.q_proj.weight


NETS = {
    "single": SingleLinear,
    "linear3d": Linear3D,
    "bmm": Bmm3D,
    "two_linear": TwoLinear,
    "add_mm": AddMatMul,
    "transpose_mm": TransposeMatMul,
    "bare_matmul": BareMatMul,
}


def build_model(net: str, dtype: torch.dtype, device: torch.device) -> nn.Module:
    return NETS[net](dtype).to(device).eval()


def build_sample_inputs(
    net: str, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, ...]:
    if net in ("single", "two_linear", "add_mm", "bare_matmul"):
        return (torch.randn(1, K, dtype=dtype, device=device),)
    if net == "transpose_mm":
        return (torch.randn(K, 8, dtype=dtype, device=device),)
    return (torch.randn(2, 8, K, dtype=dtype, device=device),)


def export_program(
    net: str, output_path: Path, dtype: torch.dtype, device_name: str
) -> Path:
    device = torch.device(device_name)
    model = build_model(net, dtype, device)
    sample_inputs = build_sample_inputs(net, dtype, device)
    ep = torch.export.export(model, sample_inputs, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, output_path)
    return output_path


def compile_to_epair(ep_path: Path, output_dir: Path, max_file_size: int) -> Path:
    import torch_delegate
    from cann_local_lm.converter.export.epair_utils import consolidated_weights_save
    from cann_local_lm.converter.export.ge_converters import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ep = torch.export.load(Path(ep_path).resolve())
    register_ge_converters()
    epair = torch_delegate.export_to_epair(ep)
    consolidated_weights_save(
        epair, output_dir, save_weights=True, max_file_size=max_file_size
    )
    return output_dir


def _dtype_arg(s: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[s]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TransData probe: find smallest net with TransData."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_e = sub.add_parser("export")
    p_e.add_argument("--net", choices=list(NETS), required=True)
    p_e.add_argument("--output", required=True)
    p_e.add_argument("--dtype", type=_dtype_arg, default=torch.float16)
    p_e.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p_c = sub.add_parser("compile")
    p_c.add_argument("--ep-path", required=True)
    p_c.add_argument("--output-dir", required=True)
    p_c.add_argument("--max-file-size", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    if args.command == "export":
        print(
            f"Exported: {export_program(args.net, Path(args.output), args.dtype, args.device)}"
        )
    elif args.command == "compile":
        print(
            f"Compiled: {compile_to_epair(Path(args.ep_path), Path(args.output_dir), args.max_file_size)}"
        )


if __name__ == "__main__":
    main()
