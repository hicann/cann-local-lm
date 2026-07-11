"""Export SimpleNetwork → .pt2 → consolidated weights (with dynamic shapes).

Usage:
  python examples/compile_model/export_network.py export [--dim 64] [--batch-size 2] [--dynamic-batch]
  python examples/compile_model/export_network.py pack <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import pack_weights, consolidated_weights_save
from simple_network import SimpleNetwork


def export(output_dir: str | Path, dim: int = 64, batch_size: int = 2,
           dynamic_batch: bool = False) -> Path:
    device = torch.device("cpu")
    model = SimpleNetwork(dim).eval().to(device)
    sample_inputs = (torch.randn(batch_size, dim, device=device),)

    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch")},)

    exported_program = torch.export.export(
        model, sample_inputs, strict=False, dynamic_shapes=dynamic_shapes,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt2_path = output_dir / "simple_network.pt2"
    torch.export.save(exported_program, pt2_path)

    graph_path = pt2_path.with_suffix(".graph.py")
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")

    print(f"Exported: {pt2_path}")
    print(f"Graph:    {graph_path}")
    if dynamic_batch:
        print(f"Batch dim: dynamic (Dim batch)")
    else:
        print(f"Batch dim: static ({batch_size})")
    return pt2_path


def compile_and_pack(pt2_path: str | Path, output_dir: str | Path) -> None:
    import torch_delegate

    pt2_path = Path(pt2_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(pt2_path)
    contents = torch_delegate.export_to_epair(exported_program)
    cw = pack_weights(contents)
    consolidated_weights_save(cw, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SimpleNetwork → .pt2 → consolidated weights."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export .pt2, compile, pack weights")
    export_parser.add_argument("--output-dir", default="generate", help="Output directory")
    export_parser.add_argument("--export-only", action="store_true")
    export_parser.add_argument("--compile-only", action="store_true")
    export_parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    export_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    export_parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")

    pack_parser = subparsers.add_parser("pack", help="Load .epair / .pt2 file and pack consolidated weights")
    pack_parser.add_argument("epair_path", help="Path to .epair or .pt2 file")
    pack_parser.add_argument("--output-dir", default=".", help="Output directory (default: current dir)")

    args = parser.parse_args()

    if args.command == "export":
        output_dir = Path(args.output_dir).resolve()
        pt2_path = output_dir / "simple_network.pt2"

        if not args.compile_only:
            export(output_dir, dim=args.dim, batch_size=args.batch_size,
                   dynamic_batch=args.dynamic_batch)
            print()

        if not args.export_only:
            compile_and_pack(pt2_path, output_dir)
            print()

        for p in (pt2_path, output_dir / "data"):
            if p.is_file() or p.is_dir():
                size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                print(f"  {p} ({size / 1024:.0f} KB)")

    elif args.command == "pack":
        import torch_delegate

        epair_path = Path(args.epair_path).resolve()
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if epair_path.suffix == ".epair":
            contents = torch_delegate.epair_load(epair_path)
        else:
            exported_program = torch.export.load(epair_path)
            contents = torch_delegate.export_to_epair(exported_program)

        cw = pack_weights(contents)
        consolidated_weights_save(cw, output_dir)


if __name__ == "__main__":
    main()
