from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import consolidated_weights_save
import torch_delegate


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3))
        self.register_buffer("bias", torch.ones(3, 3))

    def forward(self, x):
        y = torch.mm(x, self.weight) + self.bias
        return torch.transpose(y, 0, 1)


ROOT = Path(__file__).resolve().parent

torch.manual_seed(43)
device = torch.device("cpu")
model = MyModel().eval().to(device)
sample_inputs = (torch.randn(3, 4, device=device),)
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
    inp.cpu().float().detach().numpy().tofile(path)
    print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")
