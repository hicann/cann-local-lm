# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AddToAddNet — 从 Add 到 Add 的网络 (参考 aicore_only_net, 去 L2Normalize/ScatterNdUpdate).

结构 (首尾都是 Add, 无 L2Normalize/ScatterNdUpdate):
  x -> GatherV2(pos) -> Add(x+pos) -> RmsNorm
     -> Sigmoid(gate) / Gelu(ffn) -> Mul(gate*ffn) -> Add(h + gate*ffn) -> out
算子: GatherV2, Add, RmsNorm, Sigmoid, Gelu, Mul, Add (全 aicore).
(即 aicore_only_net 去掉末尾 L2Normalize + ScatterNdUpdate + kv_cache)

用法:
  python add_to_add_net.py air --output out/add_to_add
  atc --framework=1 --model out/add_to_add.air --output out/add_to_add --soc_version=KirinX90 --external_weight=0
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import cann_lm.models.fused_ops  # noqa: F401  (注册 npu_rms_norm)


BATCH = 2
SEQ = 4
HIDDEN = 64
MAX_POS = 16
EPS = 1e-6


class AddToAddNet(nn.Module):
    """从 Add 到 Add: GatherV2->Add->RmsNorm->Sigmoid/Gelu->Mul->Add."""

    def __init__(self, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.eps = EPS
        self.pos_table = nn.Parameter(torch.randn(MAX_POS, HIDDEN, dtype=dtype))
        self.register_buffer("pos_ids", torch.arange(SEQ, dtype=torch.int64))
        self.norm_weight = nn.Parameter(torch.ones(HIDDEN, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.index_select(self.pos_table, 0, self.pos_ids).unsqueeze(0)  # GatherV2
        h, _ = torch.ops.npu.npu_rms_norm.default(x + pos, self.norm_weight, epsilon=self.eps)  # Add + RmsNorm
        gate = torch.sigmoid(h)                       # Sigmoid
        ffn = F.gelu(h, approximate="tanh")           # Gelu
        return h + gate * ffn                         # Mul + Add (残差)


def build_sample_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tensor, ...]:
    torch.manual_seed(0)
    return (torch.randn(BATCH, SEQ, HIDDEN, dtype=dtype, device=device),)


def build_model(dtype: torch.dtype, device: torch.device) -> AddToAddNet:
    return AddToAddNet(dtype=dtype).eval().to(device)


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def run_eager(dtype: torch.dtype, device_name: str) -> Tensor:
    device = select_device(device_name)
    model = build_model(dtype, device)
    (x,) = build_sample_inputs(dtype, device)
    with torch.no_grad():
        out = model(x)
    print(f"Input  shape: {list(x.shape)}, dtype={x.dtype}")
    print(f"Output shape: {list(out.shape)}, dtype={out.dtype}")
    return out


def export_air(output_path: Path, dtype: torch.dtype, device_name: str) -> Path:
    """torchair.dynamo_export -> .air (npu_rms_norm 由 torchair npu:: converter 处理)."""
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    device = select_device(device_name)
    torch.manual_seed(0)  # 固定 seed: 模型参数 + 输入都可复现 (须与 gen_golden 一致)
    model = build_model(dtype, device).eval()
    sample_inputs = build_sample_inputs(dtype, device)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchair.dynamo_export(*sample_inputs, model=model, export_path=str(out.parent),
                           export_name=out.stem, config=CompilerConfig())
    air_file = Path(str(out) + ".air")
    print(f"AIR saved: {air_file}")
    return air_file


def gen_golden(output_dir: Path, dtype: torch.dtype) -> Path:
    """生成 golden 输入/输出 (固定 seed eager CPU 参考输出, 存 .bin + meta.json).

    输入: x[B,S,D] fp16
    输出: out[B,S,D] fp16
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)  # 固定 seed: 模型参数 + 输入都可复现
    model = build_model(dtype, torch.device("cpu"))
    (x,) = build_sample_inputs(dtype, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        out = model(x)
    tensors = {
        "input_x": x,
        "output_out": out,
    }
    meta = []
    for name, t in tensors.items():
        path = output_dir / f"{name}.bin"
        t.cpu().detach().numpy().tofile(path)
        meta.append({"name": name, "file": path.name,
                     "shape": list(t.shape), "dtype": str(t.dtype)})
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} -> {path.name}")
    meta_path = output_dir / "golden_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"golden meta -> {meta_path}")
    return output_dir


def _dtype_arg(s: str) -> torch.dtype:
    return {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[s]


def main() -> None:
    parser = argparse.ArgumentParser(description="AddToAddNet — 从 Add 到 Add (无 L2Normalize/ScatterNdUpdate).")
    sub = parser.add_subparsers(dest="command", required=True)
    pe = sub.add_parser("eager", help="Run eagerly.")
    pe.add_argument("--dtype", type=_dtype_arg, default=torch.float16)
    pe.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    pa = sub.add_parser("air", help="torchair.dynamo_export -> .air")
    pa.add_argument("--output", default="out/add_to_add")
    pa.add_argument("--dtype", type=_dtype_arg, default=torch.float16)
    pa.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    pg = sub.add_parser("golden", help="生成 golden 输入/输出 (.bin + meta.json)")
    pg.add_argument("--output-dir", default="out/golden_add_to_add")
    pg.add_argument("--dtype", type=_dtype_arg, default=torch.float16)
    args = parser.parse_args()
    if args.command == "eager":
        run_eager(args.dtype, args.device)
    elif args.command == "air":
        f = export_air(Path(args.output), args.dtype, args.device)
        print(f"AIR model: {f}")
    elif args.command == "golden":
        d = gen_golden(Path(args.output_dir), args.dtype)
        print(f"Golden data written to: {d}")


if __name__ == "__main__":
    main()
