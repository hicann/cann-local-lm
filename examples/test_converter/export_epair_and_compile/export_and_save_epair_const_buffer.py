import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.converter.export.ge_converters import register_ge_converters
import torch_delegate
from torch_delegate import LiteConfig, lite_export_context

from simple_network_const_buffer import SimpleNetworkWithConstBuffer


ROOT = Path(__file__).resolve().parent

target_soc = os.environ.get("SOC_VERSION", "Ascend910B")
lite_config = LiteConfig(target_soc=target_soc, allow_hf32=False)
with lite_export_context(lite_config):
    register_ge_converters()

torch.manual_seed(43)
device = torch.device("cpu")
model = SimpleNetworkWithConstBuffer(dim=64).eval().to(device)
sample_inputs = (torch.randn(2, 64, device=device),)
exported_program = torch.export.export(model, sample_inputs, strict=False)

# Sanity: confirm all three input kinds are present in the graph signature
gs = exported_program.graph_signature
print(f"graph_signature: parameters={len(gs.inputs_to_parameters)} "
      f"buffers={len(gs.inputs_to_buffers)} "
      f"lifted_tensor_constants={len(gs.inputs_to_lifted_tensor_constants)}")

with lite_export_context(lite_config):
    epair = torch_delegate.export_to_epair(exported_program)
output_dir = ROOT / "epair_model_const_buffer"
consolidated_weights_save(epair, output_dir, save_weights=True)

# Also save a .pt2 file for Python-side epair_load validation
pt2_path = ROOT / "epair_model_const_buffer.pt2"
torch_delegate.epair_save(epair, str(pt2_path))
print(f"Saved: {pt2_path}")

for i, inp in enumerate(sample_inputs):
    path = ROOT / f"input_const_buffer_{i}.bin"
    inp.cpu().detach().numpy().tofile(path)
    print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")

# Verify _is_weight classification in weight_info.json
info_path = output_dir / "weight_info.json"
with open(info_path) as f:
    weight_info = json.load(f)
from collections import Counter
kind_count = Counter()
kind_size = Counter()
for entries in weight_info.values():
    for e in entries:
        is_w = e.get("_is_weight", "<missing>")
        kind_count[is_w] += 1
        kind_size[is_w] += e["size"]
print(f"\nweight_info.json _is_weight distribution:")
for k, v in sorted(kind_count.items(), key=lambda kv: str(kv[0])):
    print(f"  _is_weight={k!s:<7}: count={v:>3}  size={kind_size[k]:>8} B")
