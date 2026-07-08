"""Export SimpleTransformer to .pt2 and optionally compile to .epair.

Usage:
  python -m cann_lm.converter.export.export_simple_transformer export
  python -m cann_lm.converter.export.export_simple_transformer compile
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from cann_lm.converter.export.simple_transformer import SimpleTransformer

VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 8
D_FF = 256
N_LAYERS = 2
MAX_SEQ_LEN = 128
BATCH_SIZE = 1
SEQ_LEN = 8


def load_model(weights_path: str | Path, device: torch.device) -> SimpleTransformer:
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN).eval().to(device)
    path = Path(weights_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=str(device))
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def build_sample_inputs(device: torch.device) -> tuple[torch.Tensor]:
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    return (tokens,)


def export(
    output_dir: str | Path,
    weights_path: str | Path | None = None,
    device_name: str = "cpu",
) -> Path:
    device = torch.device(device_name)
    if weights_path is not None:
        model = load_model(weights_path, device)
    else:
        model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN).eval().to(device)

    sample_inputs = build_sample_inputs(device)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simple_transformer.pt2"
    torch.export.save(exported_program, output_path)

    graph_path = output_path.with_suffix(".graph.py")
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")

    print(f"Exported program saved to: {output_path}")
    print(f"Graph saved to: {graph_path}")

    return output_path


def compile_ep_to_epair(ep_path: str | Path, epair_path: str | Path) -> Path:
    import torch_delegate

    ep_path = Path(ep_path).resolve()
    epair_path = Path(epair_path).resolve()
    epair_path.parent.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(ep_path)
    epair = torch_delegate.export_to_epair(exported_program)
    torch_delegate.epair_save(epair, epair_path)
    print(f"Compiled epair saved to: {epair_path}")
    return epair_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleTransformer export and epair compilation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export to .pt2")
    export_parser.add_argument("--output-dir", default="generate", help="Output directory for .pt2")
    export_parser.add_argument("--weights", default=None, help="Path to saved weights (optional)")
    export_parser.add_argument("--device", default="cpu", help="Device for export")

    compile_parser = subparsers.add_parser("compile", help="Compile .pt2 to .epair")
    compile_parser.add_argument("--ep-path", default="generate/simple_transformer.pt2")
    compile_parser.add_argument("--epair-path", default="generate/simple_transformer.epair")

    args = parser.parse_args()

    if args.command == "export":
        export(output_dir=args.output_dir, weights_path=args.weights, device_name=args.device)
    elif args.command == "compile":
        compile_ep_to_epair(ep_path=args.ep_path, epair_path=args.epair_path)


if __name__ == "__main__":
    main()
