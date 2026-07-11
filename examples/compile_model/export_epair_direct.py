"""Export SimpleNetwork → epair directory (consolidated weights + EP JSON).

Bypasses torch_delegate/torchair incompatibility by reading EP JSON directly
from the .pt2 archive.

Usage
-----
Static batch (default batch=2)::

    python examples/compile_model/export_epair_direct.py \\
        --output-dir /tmp/epair_out

Dynamic batch::

    python examples/compile_model/export_epair_direct.py \\
        --output-dir /tmp/epair_out --dynamic-batch

Custom dim / batch-size::

    python examples/compile_model/export_epair_direct.py \\
        --output-dir /tmp/epair_out --dim 128 --batch-size 4

C++ loader (directory mode auto-detected)::

    EpairModelLoader loader("/tmp/epair_out");
    loader.CompileAndLoad(options, stream);

Output layout::

    <output-dir>/
    ├── simple_network.pt2          # torch.export 存档
    └── data/
        ├── weights.bin              # 拼接后的张量二进制
        ├── weights_index.json       # [{"name", "offset", "size"}, ...]
        └── ep_json_with_ascend_ir.json  # 完整 EP JSON（含 ascend_ir）
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import pack_weights, consolidated_weights_save
from simple_network import SimpleNetwork


class EpairContents:
    """Minimal EpairContents compatible with pack_weights."""
    def __init__(self, ep, ep_json: bytes):
        self.ep = ep
        self.ep_json = ep_json


def export_epair(output_dir: str | Path, dim: int = 64, batch_size: int = 2,
                 dynamic_batch: bool = False) -> None:
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

    with zipfile.ZipFile(pt2_path) as z:
        ep_json = json.loads(z.read("serialized_exported_program.json"))

    contents = EpairContents(ep=exported_program, ep_json=ep_json)

    cw = pack_weights(contents)
    consolidated_weights_save(cw, output_dir)

    print(f"EP JSON + weights: {output_dir / 'data'}/")
    print(f"  weights.bin         ({sum(f.stat().st_size for f in (output_dir / 'data').rglob('*') if f.is_file()) / 1024:.0f} KB)")
    print(f"  weights_index.json")
    print(f"  ep_json_with_ascend_ir.json")
    print(f".pt2:              {pt2_path}")
    print(f"Dynamic batch:     {dynamic_batch}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SimpleNetwork -> epair directory"
    )
    parser.add_argument("--output-dir", default="generate_epair", help="Output directory")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")
    args = parser.parse_args()

    export_epair(
        args.output_dir,
        dim=args.dim,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
    )


if __name__ == "__main__":
    main()
