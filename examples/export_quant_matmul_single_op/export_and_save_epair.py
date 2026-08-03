"""Export the single-op quant_matmul module to epair (for OM compile test).

Produces ep_json_with_ascend_ir.json + weight_info.json + weights_0.bin.
The ep_json contains one QuantBatchMatmulV4 node (W4A8 baseline config).
Reuses the converter export pipeline (register_ge_converters + torch.export
+ export_to_epair + consolidated_weights_save).

Run on CPU (stub env); then OM-compile on NPU via the C++ main.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.converter.export.ge_converter import register_ge_converters
import torch_delegate

# registers npu_quant_matmul (schema + fake + cpu impl) on stub env
from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401

from quant_matmul_net import QuantMatmulSingleOp


ROOT = Path(__file__).resolve().parent

register_ge_converters()

torch.manual_seed(43)
m, k, n, gs = 4, 256, 8, 256
net = QuantMatmulSingleOp(m, k, n, gs).eval()
# runtime inputs: x1 (int8 act), pertoken_scale (fp32 per-token)
x1 = torch.randint(-127, 127, (m, k), dtype=torch.int8)
pertoken = torch.ones(m, dtype=torch.float32) * 0.01
sample_inputs = (x1, pertoken)

exported_program = torch.export.export(net, sample_inputs, strict=False)
epair = torch_delegate.export_to_epair(exported_program)

output_dir = ROOT / "epair_model"
consolidated_weights_save(epair, output_dir, save_weights=True)

# also save runtime inputs as .bin (for potential execution later; OM compile itself doesn't need them)
for i, inp in enumerate(sample_inputs):
    path = ROOT / f"input_{i}.bin"
    inp.cpu().detach().numpy().tobytes()
    inp.cpu().contiguous().numpy().tobytes()
    path.write_bytes(inp.cpu().contiguous().numpy().tobytes())
    print(f"Saved input[{i}] dtype={inp.dtype} shape={list(inp.shape)} -> {path}")

print(f"\nepair saved to: {output_dir}")
print(f"  ep_json_with_ascend_ir.json  (contains 1 QuantBatchMatmulV4 node)")
print(f"  weight_info.json + weights_0.bin  (weight_q int8 + weight_scale fp32 as Const)")
print(f"\nNext: OM-compile on NPU via examples/export_and_load_epair/build/export_and_load_example")
