from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.models.fake_ops import register_ge_converters
import torch_delegate

from simple_network import SimpleNetwork


ROOT = Path(__file__).resolve().parent

register_ge_converters()

torch.manual_seed(43)
device = torch.device("cpu")
model = SimpleNetwork(dim=64).eval().to(device)
sample_inputs = (torch.randn(2, 64, device=device),)
exported_program = torch.export.export(model, sample_inputs, strict=False)

epair = torch_delegate.export_to_epair(exported_program)
output_dir = ROOT / "epair_model"
consolidated_weights_save(epair, output_dir, save_weights=True)

# Also save a .pt2 file for Python-side epair_load validation
pt2_path = ROOT / "epair_model.pt2"
torch_delegate.epair_save(epair, str(pt2_path))
print(f"Saved: {pt2_path}")

for i, inp in enumerate(sample_inputs):
    path = ROOT / f"input_{i}.bin"
    inp.cpu().detach().numpy().tofile(path)
    print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")
