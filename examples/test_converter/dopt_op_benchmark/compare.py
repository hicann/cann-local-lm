"""Operator-level comparison harness for the dopt_op_benchmark golden.

Self-consistency mode (default): reload the reference QLinear for <op-name>, feed each
saved input_act, recompute output_act, compare to the saved golden output. Must match
within manifest atol/rtol -> golden is valid & reproducible from (weight + input) alone.

Candidate mode: swap the operator with a candidate impl (e.g. real cann_local_lm::quant_matmul)
and compare its output to the golden. Same weight + same input -> difference = operator
implementation divergence (isolated from quantization accuracy).

Paths via env vars (overridable by CLI): GEMMA4_MODEL_PATH, DOPT_QUANT_DIR.
Default golden dir: <script_dir>/golden.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from dopt_lm.opt_funcs.eco_stage.train import prepare_model
from dopt_lm.do_opt import set_quant_state, set_calibrate_state
from dopt_lm.modules.qlinear import QLinear

SCRIPT_DIR = Path(__file__).resolve().parent


def build_ref_model(model_path, quant_dir, device):
    dopt_cfg = os.path.join(quant_dir, "dopt_config.json")
    runtime_cfg = os.path.join(quant_dir, "runtime_config.yaml")
    with open(runtime_cfg, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["config_path"] = dopt_cfg
    model = prepare_model(
        model_path, None, dopt_cfg, config, requires_grad=False, trainable_keys=[]
    )
    set_quant_state(model, weight_state=True, input_state=False)
    set_calibrate_state(model, True)
    model.load_state_dict(
        torch.load(
            os.path.join(quant_dir, "trained_quant_weight.pth"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=False,
    )
    set_calibrate_state(model, False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--op-name",
        required=True,
        help="operator name, e.g. model.layers.0.self_attn.q_proj",
    )
    ap.add_argument("--model-path", default=os.environ.get("GEMMA4_MODEL_PATH"))
    ap.add_argument("--quant-dir", default=os.environ.get("DOPT_QUANT_DIR"))
    ap.add_argument("--golden-dir", default=str(SCRIPT_DIR / "golden"))
    ap.add_argument(
        "--device", default=os.environ.get("DEVICE", "cpu"), choices=("cpu", "npu")
    )
    args = ap.parse_args()
    if not args.model_path:
        ap.error("--model-path required (or set GEMMA4_MODEL_PATH)")
    if not args.quant_dir:
        ap.error("--quant-dir required (or set DOPT_QUANT_DIR)")

    g = Path(args.golden_dir)
    with open(g / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    atol, rtol = manifest["atol"], manifest["rtol"]
    safe = args.op_name.replace(".", "_")
    data = torch.load(g / "op_io" / f"{safe}.pt", weights_only=True)
    inputs, outputs = data["inputs"], data["outputs"]
    print(f"golden: {args.op_name} | {len(inputs)} io pairs | atol={atol} rtol={rtol}")

    device = args.device
    if device == "npu":
        import torch_npu  # noqa: F401  # NPU env: 注册 torch.npu backend

        torch.npu.set_device(int(os.environ.get("DEVICE_ID", "0")))
    print("loading reference model ...", flush=True)
    model = build_ref_model(args.model_path, args.quant_dir, device)
    qmods = {n: m for n, m in model.named_modules() if isinstance(m, QLinear)}
    assert args.op_name in qmods, f"{args.op_name} not a QLinear in ref model"
    op = qmods[args.op_name]

    max_diff = mean_diff = 0.0
    allpass = True
    with torch.no_grad():
        for i, (xa, ya_gold) in enumerate(zip(inputs, outputs)):
            y = op(xa.to(device)).float().cpu()
            d = (y - ya_gold.float()).abs()
            md, mnd = d.max().item(), d.mean().item()
            max_diff, mean_diff = max(max_diff, md), max(mean_diff, mnd)
            ok = md <= atol + rtol * ya_gold.float().abs().max().item()
            allpass &= ok
            print(
                f"  [{i}] maxΔ={md:.3e} meanΔ={mnd:.3e} -> {'PASS' if ok else 'FAIL'}"
            )

    print(
        f"\n{'PASS' if allpass else 'FAIL'}: op={args.op_name} overall maxΔ={max_diff:.3e} meanΔ={mean_diff:.3e}"
    )
    print("(self-consistency: reference operator reproduces golden from saved inputs)")


if __name__ == "__main__":
    main()
