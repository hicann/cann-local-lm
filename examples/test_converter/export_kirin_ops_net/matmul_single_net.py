# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MatmulSingleNet — 单个 nn.Linear (单 MatMul + 单权重 Const), 用于 NZ 权重格式验证 (CLI-015 L2).

结构:
  x[B, M] -> q_proj(x)[B, N]   (q_proj.weight [N, M], 无 bias)
算子: MatMul (仅 1 个), 权重 Const 1 个 (q_proj.weight, 匹配 weight_format_rules 的 *_proj.weight -> NZ).

用法:
  python matmul_single_net.py export --output out/matmul_single.pt2
  CANN_LM_ENABLE_WEIGHT_TRANS=1 python matmul_single_net.py compile \\
      --ep-path out/matmul_single.pt2 --output-dir out/matmul_single_nz_epair
  cann_lm_compile out/matmul_single_nz_epair/ep_json_with_ascend_ir.json \\
      out/matmul_single_nz_epair/weight_info.json out/matmul_single_nz

对照 (ND, env off):
  python matmul_single_net.py compile --ep-path out/matmul_single.pt2 --output-dir out/matmul_single_nd_epair
  cann_lm_compile out/matmul_single_nd_epair/ep_json_with_ascend_ir.json \\
      out/matmul_single_nd_epair/weight_info.json out/matmul_single_nd
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import Tensor, nn


M = 128
N = 128


class MatmulSingleNet(nn.Module):
    """Single Linear -> one MatMul + one weight Const (q_proj.weight)."""

    def __init__(self, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(M, N, bias=False).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.q_proj(x)


def build_sample_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tensor, ...]:
    return (torch.randn(1, M, dtype=dtype, device=device),)


def build_model(dtype: torch.dtype, device: torch.device) -> MatmulSingleNet:
    return MatmulSingleNet(dtype).to(device).eval()


def export_program(output_path: Path, dtype: torch.dtype, device_name: str) -> Path:
    device = torch.device(device_name)
    model = build_model(dtype, device)
    sample_inputs = build_sample_inputs(dtype, device)
    ep = torch.export.export(model, sample_inputs, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, output_path)
    return output_path


def compile_to_epair(ep_path: Path, output_dir: Path, max_file_size: int) -> Path:
    """Compile a .pt2 to consolidated epair. Set CANN_LM_ENABLE_WEIGHT_TRANS=1 to NZ-convert q_proj.weight."""
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
        description="Single MatMul net (one Linear) for NZ weight format verification (CLI-015 L2)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export", help="torch.export -> .pt2")
    p_exp.add_argument("--output", default="out/matmul_single.pt2")
    p_exp.add_argument("--dtype", type=_dtype_arg, default=torch.float16)
    p_exp.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    p_cmp = sub.add_parser(
        "compile",
        help=".pt2 -> consolidated .epair (env CANN_LM_ENABLE_WEIGHT_TRANS=1 for NZ)",
    )
    p_cmp.add_argument("--ep-path", default="out/matmul_single.pt2")
    p_cmp.add_argument("--output-dir", default="out/matmul_single_epair")
    p_cmp.add_argument("--max-file-size", type=int, default=512 * 1024 * 1024)

    args = parser.parse_args()
    if args.command == "export":
        out_path = export_program(Path(args.output), args.dtype, args.device)
        print(f"Exported program saved to: {out_path}")
    elif args.command == "compile":
        d = compile_to_epair(
            Path(args.ep_path), Path(args.output_dir), args.max_file_size
        )
        print(f"Consolidated epair written to: {d}")


if __name__ == "__main__":
    main()
