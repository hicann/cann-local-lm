"""Minimal operator-level benchmark: capture reference-operator (QLinear fakequant)
I/O as golden data. Candidate operators (cann_local_lm::quant_matmul / OM int4) align
against this golden on the SAME quantized weights, isolating operator correctness
from quantization-accuracy (no fp16 baseline involved).

Contract (see golden/manifest.json): for each captured operator, golden =
(input_act, output_act) under the fixed quantized weights (trained_quant_weight.pth).
A candidate operator, given the same input_act + the same weight/quant-params from
the .pth, must reproduce output_act within manifest atol/rtol.

Paths via env vars (overridable by CLI): GEMMA4_MODEL_PATH, DOPT_QUANT_DIR.
Default golden output: <script_dir>/golden (gitignored, regenerated on demand).
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from dopt_lm.opt_funcs.eco_stage.train import prepare_model
from dopt_lm.do_opt import set_quant_state, set_calibrate_state
from dopt_lm.modules.qlinear import QLinear

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = [
    "中国的首都是",
    "The capital of France is",
    "1 + 1 =",
    "Python is a",
]
ATOL = 1e-2
RTOL = 1e-2


def default_op_sample(n_layers: int) -> list[str]:
    mid = n_layers // 2
    types = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    names = [f"model.layers.{li}.{t}" for li in (0, mid, n_layers - 1) for t in types]
    names += ["lm_head"]
    return names


def build_ref_model(model_path, dopt_cfg, runtime_cfg, device):
    config = yaml.safe_load(open(runtime_cfg))
    config["config_path"] = dopt_cfg
    model = prepare_model(
        model_path, None, dopt_cfg, config, requires_grad=False, trainable_keys=[]
    )
    set_quant_state(model, weight_state=True, input_state=False)
    set_calibrate_state(model, True)
    weight_pth = os.path.join(os.path.dirname(dopt_cfg), "trained_quant_weight.pth")
    model.load_state_dict(
        torch.load(weight_pth, map_location="cpu", weights_only=True), strict=False
    )
    set_calibrate_state(model, False)
    return model.to(device).eval(), config


def capture(model, tok, prompts, device, op_names):
    qmods = {n: m for n, m in model.named_modules() if isinstance(m, QLinear)}
    targets = {n: qmods[n] for n in op_names if n in qmods}
    missing = [n for n in op_names if n not in qmods]
    if missing:
        print(
            f"[warn] {len(missing)} requested ops not QLinear (skipped): {missing[:5]}..."
        )

    captured = {n: {"inputs": [], "outputs": []} for n in targets}
    e2e_logits = []

    handles = []

    def make_hook(name):
        def hook(_m, inp, out):
            captured[name]["inputs"].append(inp[0].detach().to("cpu"))
            captured[name]["outputs"].append(out.detach().to("cpu"))

        return hook

    for n, m in targets.items():
        handles.append(m.register_forward_hook(make_hook(n)))

    with torch.no_grad():
        for p in prompts:
            enc = tok(p, return_tensors="pt").to(device)
            out = model(**enc)
            e2e_logits.append(out.logits[0, -1].float().cpu())
    for h in handles:
        h.remove()
    return captured, torch.stack(e2e_logits), list(targets.keys())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model-path",
        default=os.environ.get("GEMMA4_MODEL_PATH"),
        help="HF model dir (or GEMMA4_MODEL_PATH)",
    )
    ap.add_argument(
        "--quant-dir",
        default=os.environ.get("DOPT_QUANT_DIR"),
        help="dopt output dir with trained_quant_weight.pth + dopt_config.json + runtime_config.yaml (or DOPT_QUANT_DIR)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "golden"),
        help="golden output dir (default: <script>/golden)",
    )
    ap.add_argument(
        "--device", default=os.environ.get("DEVICE", "cpu"), choices=("cpu", "npu")
    )
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument(
        "--operators",
        default="sample",
        help="'sample' (representative) | 'all' | comma-list",
    )
    args = ap.parse_args()
    if not args.model_path:
        ap.error("--model-path required (or set GEMMA4_MODEL_PATH)")
    if not args.quant_dir:
        ap.error("--quant-dir required (or set DOPT_QUANT_DIR)")

    device = args.device
    if device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ.get("DEVICE_ID", "0")))
    dopt_cfg = os.path.join(args.quant_dir, "dopt_config.json")
    runtime_cfg = os.path.join(args.quant_dir, "runtime_config.yaml")
    weight_pth = os.path.join(args.quant_dir, "trained_quant_weight.pth")

    out = Path(args.output_dir)
    (out / "op_io").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(
        f"[1/2] building reference QLinear model (fakequant) on {device} ...",
        flush=True,
    )
    model, config = build_ref_model(args.model_path, dopt_cfg, runtime_cfg, device)
    n_layers = len(model.model.layers)
    if args.operators == "all":
        op_names = [n for n, m in model.named_modules() if isinstance(m, QLinear)]
    elif args.operators == "sample":
        op_names = default_op_sample(n_layers)
    else:
        op_names = args.operators.split(",")
    print(
        f"      capturing {len(op_names)} operators, {len(args.prompts)} prompts",
        flush=True,
    )

    print("[2/2] running forward + capturing operator I/O ...", flush=True)
    captured, e2e_logits, present = capture(model, tok, args.prompts, device, op_names)

    for name, data in captured.items():
        safe = name.replace(".", "_")
        torch.save(
            {"inputs": data["inputs"], "outputs": data["outputs"]},
            out / "op_io" / f"{safe}.pt",
        )
    torch.save(e2e_logits, out / "e2e_logits.pt")

    inputs_meta = [
        {"prompt": p, "input_ids": tok(p)["input_ids"]} for p in args.prompts
    ]
    manifest = {
        "name": "dopt_op_benchmark",
        "operator_impl": "dopt.QLinear.fakequant (quant->dequant->fp matmul, weight-only)",
        "model_path": args.model_path,
        "quant_weight": weight_pth,
        "quant_dir": args.quant_dir,
        "dopt_config": dopt_cfg,
        "runtime_config": runtime_cfg,
        "device": device,
        "n_layers": n_layers,
        "prompts": args.prompts,
        "inputs": inputs_meta,
        "captured_operators": present,
        "e2e_logits": "e2e_logits.pt  # [n_prompts, vocab] last-token logits",
        "op_io_dir": "op_io/<op_name>.pt  # {inputs:[act...], outputs:[act...]} per prompt",
        "comparison_contract": (
            "Candidate operator (e.g. cann_local_lm::quant_matmul / OM int4): load the SAME weight "
            "from quant_weight for <op_name>, feed saved op_io/<op>.inputs[i], compute output, "
            f"compare to op_io/<op>.outputs[i] within atol={ATOL} rtol={RTOL}. "
            "No fp16 baseline -> isolates operator correctness from quantization accuracy."
        ),
        "atol": ATOL,
        "rtol": RTOL,
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2, ensure_ascii=False)

    print(f"\n=== golden written to {out} ===")
    print(f"  operators captured: {len(present)} / {len(op_names)} requested")
    print(f"  prompts: {len(args.prompts)}")
    print(f"  e2e_logits: {tuple(e2e_logits.shape)}")
    for n in present:
        d = captured[n]
        s0 = d["inputs"][0]
        print(
            f"    {n:42s} in={tuple(s0.shape)} {str(s0.dtype):12s} n_io={len(d['inputs'])}"
        )
    print(f"\n  manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
