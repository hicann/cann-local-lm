"""Export SimpleTransformer → .pt2 → model directory with consolidated weights.

Subcommands:
  export   Export and compile to .epair, then pack weights into ``data/``
  pack     Pack an already-extracted model directory into consolidated format

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

from cann_lm.converter.export.epair_utils import pack_weights
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


def compile_model(pt2_path: str | Path, epair_path: str | Path) -> Path:
    import torch_delegate

    pt2_path = Path(pt2_path).resolve()
    epair_path = Path(epair_path).resolve()
    epair_path.parent.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(pt2_path)
    epair = torch_delegate.export_to_epair(exported_program)
    torch_delegate.epair_save(epair, epair_path)
    return epair_path


def extract_and_pack(epair_path: str | Path, output_dir: str | Path) -> Path:
    import zipfile

    epair_path = Path(epair_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(epair_path) as z:
        z.extractall(output_dir)

    # The ZIP root is a subdirectory named after the model
    model_dir = output_dir / "simple_transformer" if (output_dir / "simple_transformer").is_dir() else output_dir
    pack_weights(model_dir)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SimpleTransformer → .pt2 → .epair, then pack weights."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- export ---
    export_parser = subparsers.add_parser("export", help="Export .pt2, compile to .epair, pack weights")
    export_parser.add_argument("--output-dir", default="generate", help="Output directory")
    export_parser.add_argument("--export-only", action="store_true")
    export_parser.add_argument("--compile-only", action="store_true")

    # --- pack ---
    pack_parser = subparsers.add_parser("pack", help="Pack an extracted model directory into consolidated format")
    pack_parser.add_argument("model_dir", help="Model directory containing data/")
    pack_parser.add_argument("--output-bin", default=None, help="Output binary file (default: data/weights.bin)")
    pack_parser.add_argument("--output-json", default=None, help="Output JSON index (default: data/weights_index.json)")

    args = parser.parse_args()

    if args.command == "export":
        output_dir = Path(args.output_dir).resolve()
        pt2_path = output_dir / "simple_transformer.pt2"
        epair_path = output_dir / "simple_transformer.epair"

        if not args.compile_only:
            export(output_dir)
            print()

        if not args.export_only:
            compile_model(pt2_path, epair_path)
            print()
            extract_and_pack(epair_path, output_dir)

        for p in (pt2_path, epair_path, output_dir / "simple_transformer"):
            if p.is_file() or p.is_dir():
                size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                print(f"  {p} ({size / 1024:.0f} KB)")

    elif args.command == "pack":
        pack_weights(args.model_dir, args.output_bin, args.output_json)


if __name__ == "__main__":
    main()
