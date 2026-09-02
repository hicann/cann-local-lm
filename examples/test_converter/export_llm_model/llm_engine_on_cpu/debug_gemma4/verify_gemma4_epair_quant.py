#!/usr/bin/env python3
"""verify_gemma4_epair_quant.py — gemma4 quant PLE (int8+scale) epair 回环验证。

验证 CLI-012 修复: PLE quant 适配后, ep.module(traced, forward dequant ple_q*ple_scale)
vs wrapper(eager, forward dequant) logits 一致 + PLE dequant 数值 sanity。
需 dopt GPTQ 产物 (--quant-path, 含 model.safetensors)。

用法:
  source scripts/set_env_export.sh
  python verify_gemma4_epair_quant.py --hf-path <gemma4> --quant-path <dopt_out>
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from cann_lm.converter.loader import load_model
from cann_lm.converter.swap import swap_embeddings_from_dopt
from cann_lm.converter.export.ge_converters import register_ge_converters
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def parse_args():
    p = argparse.ArgumentParser(description="gemma4 quant PLE epair 回环验证")
    p.add_argument("--hf-path", required=True, help="gemma4 HF 模型路径")
    p.add_argument("--quant-path", required=True, help="dopt GPTQ 产物目录 (含 model.safetensors)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--kv-cache-max-len", type=int, default=2048)
    return p.parse_args()


def load_dopt_sd(quant_path):
    st = Path(quant_path) / "model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(f"dopt 产物 model.safetensors 不存在: {st}")
    return load_file(str(st))


def main():
    args = parse_args()
    dtype = torch.bfloat16
    register_ge_converters()

    print("[1/4] load gemma4 HF model + wrapper")
    art = load_model(args.hf_path, dtype=dtype, device="cpu")
    wrapper = Gemma4ExportWrapper(
        art.model, kv_cache_max_len=args.kv_cache_max_len, batch=1
    ).eval()
    print(f"  ple_enabled={wrapper._ple_enabled}")

    print("[2/4] load dopt 产物 + swap_embeddings_from_dopt + set_quant_embeddings")
    sd = load_dopt_sd(args.quant_path)
    embed_quant = swap_embeddings_from_dopt(sd, art.model)
    ple_key = "model.model.embed_tokens_per_layer.weight"
    print(f"  embed_quant keys: {list(embed_quant)}")
    print(f"  PLE quantized: {ple_key in embed_quant}")
    if ple_key not in embed_quant:
        print("  [WARN] PLE 未量化 (无 scale), 退出 (CLI-012 针对 PLE quant)")
        return
    wrapper.set_quant_embeddings(embed_quant)

    print("[3/4] export (quant_embeddings=True, use_embeddings=True)")
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=args.seq_len, use_embeddings=True, quant_embeddings=True
    )
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    print(f"  ep nodes: {len(ep.graph.nodes)}")

    ple_q, ple_scale = sample_inputs[8], sample_inputs[10]
    emb_q, emb_scale = sample_inputs[6], sample_inputs[9]
    print(f"  ple_q: {tuple(ple_q.shape)} {ple_q.dtype}, ple_scale: {tuple(ple_scale.shape)} {ple_scale.dtype}")
    print(f"  emb_q: {tuple(emb_q.shape)} {emb_q.dtype}, emb_scale: {tuple(emb_scale.shape)} {emb_scale.dtype}")

    print("[4/4] ep.module(traced) vs wrapper(eager) logits 一致 + PLE dequant sanity")
    with torch.no_grad():
        out_ep = ep.module()(*sample_inputs)
        out_w = wrapper(*sample_inputs)
    logits_ep, logits_w = out_ep[0], out_w[0]
    diff = (logits_ep.to(torch.float32) - logits_w.to(torch.float32)).abs()
    print(f"  logits ep vs wrapper: max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
    match = diff.max().item() < 1e-3
    print(f"  ep==wrapper (quant PLE dequant): {match}")

    ple_dequant = ple_q.to(torch.float32) * ple_scale.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
    print(f"  PLE dequant: min={ple_dequant.min().item():.4f} max={ple_dequant.max().item():.4f} "
          f"mean={ple_dequant.mean().item():.4f} nan={ple_dequant.isnan().any().item()} inf={ple_dequant.isinf().any().item()}")

    print(f"\n{'='*60}")
    if match:
        print("[PASS] gemma4 quant PLE epair 回环一致 (CLI-012 forward dequant 在 trace 中)")
    else:
        print("[FAIL] ep vs wrapper 不一致 (PLE quant dequant trace 问题)")


if __name__ == "__main__":
    main()
