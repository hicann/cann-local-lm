from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from simple_network import SimpleNetwork
import torch_delegate
import warnings

warnings.filterwarnings("ignore")

output_dir1 = Path("/mnt/workspace/project/CANN-LM_5883/examples/compile_model/epair_with_weights")
output_dir1.mkdir(parents=True, exist_ok=True)

output_dir2 = Path("/mnt/workspace/project/CANN-LM_5883/examples/compile_model/epair_json_only")
output_dir2.mkdir(parents=True, exist_ok=True)

DIM = 64
BATCH_SIZE = 2

# ========== 第一个模型：导出完整目录（含权重） ==========
device = torch.device("cpu")
model = SimpleNetwork(DIM).eval().to(device)
sample_inputs = (torch.randn(BATCH_SIZE, DIM, device=device),)
exported_program = torch.export.export(model, sample_inputs, strict=False)

epair = torch_delegate.export_to_epair(exported_program)
consolidated_weights_save(epair, output_dir1, save_weights=True)

# ========== 第二个模型：仅导出 EP JSON（不含权重） ==========
model2 = SimpleNetwork(DIM + 1).eval().to(device)
sample_inputs2 = (torch.randn(BATCH_SIZE + 1, DIM + 1, device=device),)
exported_program2 = torch.export.export(model2, sample_inputs2, strict=False)
epair2 = torch_delegate.export_to_epair(exported_program2)
consolidated_weights_save(epair2, output_dir2, save_weights=False)


