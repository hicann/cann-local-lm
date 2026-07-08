"""Export SimpleTransformer to .pt2/.epair, extract epair, and pack weights.

Subcommands:
  export   Export SimpleTransformer → .pt2 → .epair
  extract  Extract .epair archive to a directory
  pack     Pack extracted weights into a single binary + JSON index

Environment setup (PYTHONPATH, LD_PRELOAD, which Python) must be done
by the caller — use scripts/export_epair.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.converter.export.epair_utils import extract_epair, pack_weights
from cann_lm.converter.export.simple_transformer import SimpleTransformer

VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 8
D_FF = 256
N_LAYERS = 2
MAX_SEQ_LEN = 128
BATCH_SIZE = 1
SEQ_LEN = 8


def export(output_dir: str | Path) -> Path:
    device = torch.device("cpu")
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN).eval().to(device)
    sample_inputs = (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device),)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt2_path = output_dir / "simple_transformer.pt2"
    torch.export.save(exported_program, pt2_path)

    graph_path = pt2_path.with_suffix(".graph.py")
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")

    print(f"Exported: {pt2_path}")
    print(f"Graph:    {graph_path}")
    return pt2_path


def compile_ep_to_epair(pt2_path: str | Path, epair_path: str | Path) -> Path:
    import torch_delegate

    pt2_path = Path(pt2_path).resolve()
    epair_path = Path(epair_path).resolve()
    epair_path.parent.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(pt2_path)
    epair = torch_delegate.export_to_epair(exported_program)
    torch_delegate.epair_save(epair, epair_path)
    print(f"Compiled: {epair_path}")
    return epair_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SimpleTransformer → .pt2 → .epair, "
        "extract epair archive, and pack weights."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- export ---
    export_parser = subparsers.add_parser("export", help="Export .pt2 and compile to .epair")
    export_parser.add_argument("--output-dir", default="generate", help="Output directory")
    export_parser.add_argument("--export-only", action="store_true")
    export_parser.add_argument("--compile-only", action="store_true")

    # --- extract ---
    extract_parser = subparsers.add_parser("extract", help="Extract .epair archive")
    extract_parser.add_argument("epair_path", help="Path to .epair file")
    extract_parser.add_argument("--output-dir", default=None, help="Output directory (default: epair name without suffix)")

    # --- pack ---
    pack_parser = subparsers.add_parser("pack", help="Pack extracted weights into a binary + index")
    pack_parser.add_argument("extract_dir", help="Extracted epair directory")
    pack_parser.add_argument("--output-bin", default="weights.bin", help="Output binary file")
    pack_parser.add_argument("--output-json", default="weights_index.json", help="Output JSON index")

    args = parser.parse_args()

    if args.command == "export":
        output_dir = Path(args.output_dir).resolve()
        pt2_path = output_dir / "simple_transformer.pt2"
        epair_path = output_dir / "simple_transformer.epair"

        if not args.compile_only:
            export(output_dir)
            print()

        if not args.export_only:
            compile_ep_to_epair(pt2_path, epair_path)
            print()

        for p in (pt2_path, epair_path):
            if p.is_file():
                print(f"  {p} ({p.stat().st_size / 1024:.0f} KB)")

    elif args.command == "extract":
        extract_epair(args.epair_path, args.output_dir)

    elif args.command == "pack":
        pack_weights(args.extract_dir, args.output_bin, args.output_json)


if __name__ == "__main__":
    main()
