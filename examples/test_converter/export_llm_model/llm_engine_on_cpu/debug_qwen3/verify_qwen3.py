#!/usr/bin/env python3
"""End-to-end Qwen3 verification: unit tests + text generation + wrapper self-test.

合并自 test_qwen3.py (单元测试) + run_qwen3.py (端到端推理) + verify_wrapper.py (wrapper 自测).

Usage:
    # unit tests only (random weights, quick shape/pipeline check)
    python verify_qwen3.py --test-only

    # unit tests + inference with real safetensors weights
    python verify_qwen3.py --model /mnt/workspace/data/models/qwen3-0.6B --prompt "你好"

    # skip unit tests, run inference only
    python verify_qwen3.py --no-tests --model /mnt/workspace/data/models/qwen3-0.6B

环境变量 (详见 verify_epair_vs_hf.py docstring):
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LLM_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(_LLM_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_ENGINE_DIR))
import hf_cpu  # noqa: F401  disables torch_npu probe (must precede transformers import)

from cann_lm.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_qwen3_weights,
    repeat_kv,
    rotate_half,
)
from cann_lm.models.qwen3.configuration_qwen3 import Qwen3TextConfig
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

PASS = 0
FAIL = 0
SKIP = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def info(msg):
    print(f"  [INFO] {msg}")


def small_config(**over) -> Qwen3TextConfig:
    base = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
    )
    base.update(over)
    return Qwen3TextConfig(**base)


# ════════════════════════════════════════════
# Unit tests (from test_qwen3.py)
# ════════════════════════════════════════════

def test_config():
    print("\n=== Configuration ===")
    cfg = small_config()
    check("model_type", cfg.model_type == "qwen3")
    check("head_dim field", cfg.head_dim == 32)
    check("tie_word_embeddings", cfg.tie_word_embeddings is True)
    cfg2 = Qwen3TextConfig.from_dict({"vocab_size": 99, "architectures": ["X"], "hidden_size": 16})
    check("from_dict filters unknown", cfg2.vocab_size == 99 and cfg2.hidden_size == 16)
    info(f"  config: {cfg.hidden_size}d x {cfg.num_hidden_layers}L, GQA {cfg.num_attention_heads}/{cfg.num_key_value_heads}")
    return cfg


def test_rmsnorm():
    print("\n=== RMSNorm ===")
    cfg = small_config()
    norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(2, 5, cfg.hidden_size)
    out = norm(x)
    check("RMSNorm shape preserved", out.shape == x.shape)
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (x_f * torch.rsqrt(var + cfg.rms_norm_eps)).to(x.dtype)
    check("RMSNorm matches formula (weight=1)", torch.allclose(out, ref, atol=1e-5))
    check("RMSNorm keeps unit RMS (weight=1)", torch.allclose(
        out.float().pow(2).mean(-1).sqrt(), torch.ones(2, 5), atol=1e-4))
    check("RMSNorm finite", torch.isfinite(out).all().item())


def test_rope():
    print("\n=== RoPE ===")
    cfg = small_config()
    rope = Qwen3RotaryEmbedding(cfg)
    b, s, h, d = 2, 6, 3, cfg.head_dim
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    pos = torch.arange(s).unsqueeze(0).expand(b, -1)
    cos, sin = rope(q, pos)
    check("cos/sin shape", cos.shape == (b, s, d) and sin.shape == (b, s, d))
    q2, k2 = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
    check("rotated q shape", q2.shape == q.transpose(1, 2).shape)
    q_t = q.transpose(1, 2)
    nq = q_t.float().pow(2).sum(-1).sqrt()
    nq2 = q2.float().pow(2).sum(-1).sqrt()
    check("RoPE preserves q norm", torch.allclose(nq, nq2, atol=1e-4))
    nk = k.transpose(1, 2).float().pow(2).sum(-1).sqrt()
    nk2 = k2.float().pow(2).sum(-1).sqrt()
    check("RoPE preserves k norm", torch.allclose(nk, nk2, atol=1e-4))
    rh = rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    check("rotate_half", torch.equal(rh, torch.tensor([-3.0, -4.0, 1.0, 2.0])))


def test_mlp():
    print("\n=== MLP ===")
    cfg = small_config()
    mlp = Qwen3MLP(cfg)
    x = torch.randn(2, 5, cfg.hidden_size)
    out = mlp(x)
    check("MLP shape", out.shape == x.shape)
    check("MLP finite", torch.isfinite(out).all().item())
    check("MLP has gate/up/down", hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"))


def test_attention_and_layer():
    print("\n=== Attention / DecoderLayer ===")
    cfg = small_config()
    b, s = 2, 6
    rope = Qwen3RotaryEmbedding(cfg)
    pos = torch.arange(s).unsqueeze(0)
    cos, sin = rope(torch.randn(b, s, cfg.hidden_size), pos)
    for i in range(cfg.num_hidden_layers):
        attn = Qwen3Attention(cfg, i)
        x = torch.randn(b, s, cfg.hidden_size)
        o = attn(x, position_embeddings=(cos, sin))
        check(f"attn layer {i} shape", o.shape == x.shape)
        check(f"attn layer {i} finite", torch.isfinite(o).all().item())
        check(f"attn layer {i} has q_norm/k_norm", hasattr(attn, "q_norm") and hasattr(attn, "k_norm"))
        layer = Qwen3DecoderLayer(cfg, i)
        ol = layer(x, position_embeddings=(cos, sin))
        check(f"decoder layer {i} shape", ol.shape == x.shape)
        break


def test_forward_shape_and_tie(cfg):
    print("\n=== Forward shape / tied embeddings ===")
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    out = lm(input_ids=ids)
    check("logits shape", out["logits"].shape == (2, 8, cfg.vocab_size))
    check("loss None w/o labels", out["loss"] is None)
    check("logits finite", torch.isfinite(out["logits"]).all().item())
    if cfg.tie_word_embeddings:
        same = lm.lm_head.weight is lm.model.embed_tokens.weight
        check("lm_head.weight IS embed_tokens.weight (tied)", same)
    info(f"  params: {sum(p.numel() for p in lm.parameters()):,}")


def test_determinism(cfg):
    print("\n=== Determinism (eval) ===")
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    with torch.no_grad():
        o1 = lm(input_ids=ids)["logits"]
        o2 = lm(input_ids=ids)["logits"]
    check("two forwards identical", torch.equal(o1, o2))


def test_causal_invariance(cfg):
    print("\n=== Causal invariance: forward(prefix) == forward(full)[:, :k] ===")
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    with torch.no_grad():
        full = lm(input_ids=ids)["logits"]
        ok = True
        for k in range(1, ids.shape[1] + 1):
            pref = lm(input_ids=ids[:, :k])["logits"]
            if not torch.allclose(pref[:, -1, :], full[:, k - 1, :], atol=1e-4):
                ok = False
                info(f"  mismatch at prefix length {k}")
                break
    check("prefix last-position == full[:, k-1] for all k", ok)


def test_kv_cache_consistency(cfg):
    print("\n=== KV cache: incremental decode == full prefill ===")
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    n = ids.shape[1]
    with torch.no_grad():
        full = lm(input_ids=ids)["logits"]
        past = [None] * cfg.num_hidden_layers
        per_pos = []
        for i in range(n):
            step = ids[:, :1] if i == 0 else ids[:, i:i + 1]
            out = lm(input_ids=step, past_key_values=past)
            past = out["past_key_values"]
            per_pos.append(out["logits"][:, -1, :])
        inc = torch.stack(per_pos, dim=1)
        max_abs = (inc - full).abs().max().item()
        ok = torch.allclose(inc, full, atol=1e-3)
    info(f"  max|inc - full| = {max_abs:.2e}")
    check("incremental-with-cache == full prefill (per position)", ok)


def test_greedy_equals_argmax(cfg):
    print("\n=== greedy first token == argmax(forward(prompt)[:, -1]) ===")
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        argmax_next = lm(input_ids=ids)["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        gen = lm.generate(ids, max_new_tokens=1, do_sample=False)
    check("greedy first token == argmax", torch.equal(gen[:, -1:], argmax_next))


def test_loss(cfg):
    print("\n=== Loss / labels ===")
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    labels = ids.clone()
    with torch.no_grad():
        out = lm(input_ids=ids, labels=labels)
    check("loss scalar", out["loss"] is not None and out["loss"].ndim == 0)
    check("loss finite", torch.isfinite(out["loss"]).all().item())


def test_truncation():
    print("\n=== Layer truncation ===")
    for n in [2, 4, 8]:
        cfg = small_config(num_hidden_layers=n)
        lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
        out = lm(input_ids=ids)["logits"]
        check(f"{n} layers logits", out.shape == (1, 4, cfg.vocab_size))


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def test_wrapper_vs_model(cfg):
    print("\n=== Wrapper vs Model (单步, max_diff=0.0) ===")
    b, s = 1, 8
    model = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    dtype = next(model.parameters()).dtype

    input_ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
    attention_mask = make_causal_mask(s, dtype)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([s - 1], dtype=torch.int32)

    with torch.no_grad():
        out_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=[None] * cfg.num_hidden_layers,
        )
        logits_orig = out_orig["logits"]
        kv_orig = out_orig["past_key_values"]

        logits_wrap, *kv_flat = wrapper(
            input_ids, attention_mask, position_ids,
            None,
            torch.arange(s, dtype=torch.long),
            None,
            output_pos,
        )

    expected = logits_orig[:, output_pos.item():output_pos.item() + 1, :]
    max_diff = (logits_wrap - expected).abs().max().item()
    check(f"wrapper logits match (max_diff={max_diff:.6e})", max_diff == 0.0)

    kv_pairs = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, len(kv_flat), 2)]
    kv_match = True
    max_kv_diff = 0.0
    for i in range(min(len(kv_orig), len(kv_pairs))):
        k_orig, v_orig = kv_orig[i]
        k_wrap, v_wrap = kv_pairs[i]
        k_diff = (k_wrap - k_orig).abs().max().item()
        v_diff = (v_wrap - v_orig).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        if k_diff != 0.0 or v_diff != 0.0:
            kv_match = False
    check(f"wrapper KV cache match (max_diff={max_kv_diff:.6e})", kv_match and max_kv_diff == 0.0)
    info(f"  wrapper vs model: logits diff={max_diff:.6e}, KV diff={max_kv_diff:.6e}")


def run_unit_tests() -> bool:
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    t0 = time.time()
    cfg = test_config()
    test_rmsnorm()
    test_rope()
    test_mlp()
    test_attention_and_layer()
    test_forward_shape_and_tie(cfg)
    test_determinism(cfg)
    test_causal_invariance(cfg)
    test_kv_cache_consistency(cfg)
    test_greedy_equals_argmax(cfg)
    test_loss(cfg)
    test_truncation()
    test_wrapper_vs_model(cfg)
    elapsed = time.time() - t0
    print(f"\n  Unit tests: {PASS} passed, {FAIL} failed, {SKIP} skipped ({elapsed:.2f}s)")
    return FAIL == 0


# ════════════════════════════════════════════
# Model loading + inference (from run_qwen3.py)
# ════════════════════════════════════════════

def load_cann_model(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    cfg = Qwen3TextConfig.from_pretrained(model_path)
    model = Qwen3ForCausalLM(cfg).to(dtype=dtype).to(device=device).eval()
    model = load_qwen3_weights(model, model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, cfg


def build_model(model_path: str = None, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    if model_path:
        return load_cann_model(model_path, dtype=dtype, device=device)
    cfg = small_config()
    model = Qwen3ForCausalLM(cfg).to(dtype=dtype).to(device=device).eval()
    return model, None, cfg


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> str:
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    out_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def main():
    global PASS, FAIL, SKIP
    parser = argparse.ArgumentParser(description="Qwen3 verification: unit tests + inference")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run unit tests, skip inference")
    parser.add_argument("--model", type=str, default=HF_PATH,
                        help="Path to Qwen3 safetensors checkpoint")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--no-tests", action="store_true",
                        help="Skip unit tests, run inference only")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu)")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")

    if not args.no_tests:
        ok = run_unit_tests()
        if args.test_only:
            print(f"\n{'=' * 50}")
            total = PASS + FAIL + SKIP
            print(f"Results: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP ({total} total)")
            print(f"{'=' * 50}")
            return 0 if ok else 1

    model, tokenizer, cfg = load_cann_model(args.model, dtype=torch.bfloat16, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {type(model).__name__}, {n_params / 1e9:.3f}B params, bfloat16 on {args.device}")

    print(f"\n--- prompt ---\n{args.prompt}")
    out = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(f"\n=== output ===\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
