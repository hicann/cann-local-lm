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

"""Demo: a small network built out of the CANN fused AddRmsNorm op.

AddRmsNorm is the CANN `es_nn` fused operator that does residual-add + RMSNorm
in one kernel:  x = x1 + x2;  y = x * rsqrt(mean(x^2)+eps) * gamma;
outputs (y, rstd, x). See `es_nn/es_AddRmsNorm_c.h` and torchair
`ge.AddRmsNorm`.

This script defines an export-only
torch op `cann_lm::add_rms_norm`, a torchair GE converter mapping it to
`ge.AddRmsNorm`, a tiny 2-layer residual+RMSNorm network, and an
export(.pt2)/compile(.epair) CLI.

Usage:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python export_addrmsnorm_model.py export \
      --output ./generate/addrmsnorm.pt2 --device cpu
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python export_addrmsnorm_model.py compile \
      --ep-path ./generate/addrmsnorm.pt2 --output-dir ./generate/addrmsnorm_epair
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn

from cann_lm.converter.export.epair_utils import consolidated_weights_save

# ─────────────────────────────────────────────────────────────────────
# op schema + fake (export-only, no real kernel needed)
# ─────────────────────────────────────────────────────────────────────
_LIBRARY = torch.library.Library("cann_lm", "FRAGMENT")
_LIBRARY.define(
    "add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, *, float epsilon=1e-6) "
    "-> (Tensor y, Tensor rstd, Tensor x)"
)


@torch.library.register_fake("cann_lm::add_rms_norm")
def _add_rms_norm_meta(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del gamma, epsilon
    y = torch.empty_like(x1)
    rstd = x1.new_empty(x1.shape[:-1] + (1,), dtype=torch.float32)
    x_sum = torch.empty_like(x1)
    return y, rstd, x_sum


@torch.library.impl("cann_lm::add_rms_norm", "cpu")
def _add_rms_norm_cpu(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 与 CANN es_nn AddRmsNorm 对齐: x = x1 + x2; y = x * rsqrt(mean(x^2)+eps) * gamma
    x = x1 + x2
    ms = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + epsilon)
    y = (x.float() * rstd).to(x1.dtype) * gamma
    return y, rstd, x


def register_ge_converters() -> None:
    from torchair import register_fx_node_ge_converter
    from torchair._ge_concrete_graph import ge_apis as ge

    @register_fx_node_ge_converter(torch.ops.cann_lm.add_rms_norm.default)
    def _convert_add_rms_norm(
        x1,
        x2,
        gamma,
        *,
        epsilon: float = 1e-6,
        meta_outputs=None,
    ):
        del meta_outputs
        # ge.AddRmsNorm(x1, x2, gamma, *, epsilon) -> (y, rstd, x)
        return ge.AddRmsNorm(x1, x2, gamma, epsilon=epsilon)


# ─────────────────────────────────────────────────────────────────────
# model: 2-layer residual + AddRmsNorm (transformer pre-norm style)
# ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
SEQ_LEN = 8
HIDDEN_DIM = 64
EPS = 1e-6


class AddRmsNormModel(nn.Module):
    """Two residual+RMSNorm blocks; each fuses add+norm via AddRmsNorm."""

    def __init__(self) -> None:
        super().__init__()
        # bias=False: nn.Linear 默认 bias=True 会导出成 aten.addmm -> MatMulV2，
        # 而 stub GE 的 OperatorFactory 不注册 MatMulV2；bias=False 走 aten.mm ->
        # MatMul（已注册），与 qwen3 等已验证模型一致。
        self.fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False, dtype=torch.float32)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False, dtype=torch.float32)
        self.gamma1 = nn.Parameter(torch.ones(HIDDEN_DIM, dtype=torch.float32))
        self.gamma2 = nn.Parameter(torch.ones(HIDDEN_DIM, dtype=torch.float32))
        self.out_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # block 1: y = rms_norm(x + fc1(x))
        t1 = self.fc1(x)
        y1, _, _ = torch.ops.cann_lm.add_rms_norm.default(x, t1, self.gamma1, epsilon=EPS)
        # block 2: y = rms_norm(y1 + fc2(y1))
        t2 = self.fc2(y1)
        y2, _, _ = torch.ops.cann_lm.add_rms_norm.default(y1, t2, self.gamma2, epsilon=EPS)
        return self.out_proj(y2)


def build_sample_inputs(device: torch.device | str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    device = torch.device(device)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=torch.float32)
    return (x,)


# ─────────────────────────────────────────────────────────────────────
# export / compile
# ─────────────────────────────────────────────────────────────────────
def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is not available")
    return torch.device(device_name)


def export_sample_program(
    output_path: Path,
    device_name: str = "auto",
    print_graph: bool = False,
) -> tuple[Path, torch.export.ExportedProgram]:
    device = select_device(device_name)
    model = AddRmsNormModel().eval().to(device)
    sample_inputs = build_sample_inputs(device)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, output_path)

    graph_path = output_path.with_suffix(".graph.py")
    if print_graph:
        print(exported_program)
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")
    return output_path, exported_program


def compile_ep_to_epair(
    ep_path: str | Path,
    output_dir: str | Path,
    rewrite_only: bool = False,
    max_file_size: int = 512 * 1024 * 1024,
) -> Path:
    """Compile a .pt2 to the consolidated epair format (a directory).

    Writes (via consolidated_weights_save):
      output_dir/ep_json_with_ascend_ir.json
      output_dir/weight_info.json
      output_dir/weights_0.bin [, weights_1.bin, ...]
    """
    import torch_delegate
    from torch_delegate import LiteConfig, lite_export_context

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(Path(ep_path).resolve())

    rewritten_path = output_dir / f"{Path(ep_path).stem}.rewritten.pt2"
    torch.export.save(exported_program, rewritten_path)
    rewritten_path.with_suffix(".graph.py").write_text(
        exported_program.graph_module.code, encoding="utf-8"
    )

    if rewrite_only:
        return rewritten_path

    lite_config = LiteConfig(target_soc=os.environ.get("SOC_VERSION", "Ascend910B"), allow_hf32=False)
    with lite_export_context(lite_config):
        register_ge_converters()
        epair = torch_delegate.export_to_epair(exported_program)
    # _assert_tensor_metadata 节点（torchair decomposition 副作用）由 C++ PlanNodes
    # 侧 IsAssertTensorMetadataNode 跳过，无需 Python strip。
    consolidated_weights_save(
        epair, output_dir, save_weights=True, max_file_size=max_file_size
    )
    return output_dir


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def run_eager(
    device_name: str = "cpu",
) -> torch.Tensor:
    """Run AddRmsNormModel eagerly to verify the real add_rms_norm CPU kernel."""
    device = select_device(device_name)
    model = AddRmsNormModel().eval().to(device)
    (x,) = build_sample_inputs(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input  shape: {list(x.shape)}, dtype={x.dtype}")
    print(f"Output shape: {list(y.shape)}, dtype={y.dtype}")
    print(f"Output (first 8): {y.flatten()[:8].tolist()}")
    return y


def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="generate/addrmsnorm.pt2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-graph", action="store_true")


def _add_compile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ep-path", default="generate/addrmsnorm.pt2")
    parser.add_argument("--output-dir", default="generate/addrmsnorm_epair")
    parser.add_argument("--rewrite-only", action="store_true")
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=512 * 1024 * 1024,
        help="Max bytes per weights_N.bin (default 512MB).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a small AddRmsNorm network to .pt2 and compile to .epair.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export ExportedProgram to .pt2")
    _add_export_args(export_parser)

    compile_parser = subparsers.add_parser("compile", help="Compile .pt2 to .epair")
    _add_compile_args(compile_parser)

    eager_parser = subparsers.add_parser(
        "eager", help="Run AddRmsNormModel eagerly to verify the CPU add_rms_norm kernel"
    )
    eager_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )

    args = parser.parse_args()

    if args.command == "export":
        output_path = Path(args.output).resolve()
        exported_path, _ = export_sample_program(
            output_path=output_path, device_name=args.device, print_graph=args.print_graph
        )
        print(f"Exported program saved to: {exported_path}")
        print(f"Graph text saved to: {exported_path.with_suffix('.graph.py')}")
    elif args.command == "compile":
        output_dir = compile_ep_to_epair(
            ep_path=args.ep_path,
            output_dir=args.output_dir,
            rewrite_only=args.rewrite_only,
            max_file_size=args.max_file_size,
        )
        print(f"Consolidated epair written to: {output_dir}")
    elif args.command == "eager":
        run_eager(device_name=args.device)


if __name__ == "__main__":
    main()
