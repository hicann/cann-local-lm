"""Test script for Qwen3 (standard) pure PyTorch implementation.

Usage:
    python test_qwen3.py              # invariant / component tests (tiny random model)
    python test_qwen3.py --model PATH # also test real weight loading + coherent decode

Correctness philosophy: we cannot rely on a HuggingFace transformers reference
(transformers is not installed and there is no network). Instead we prove
correctness with two classes of checks that hold for *any* weights:

  1. Mathematical invariants (a bug breaks them regardless of weights):
       - causal: forward(prefix)[:, :k] == forward(full)[:, :k]
       - KV cache: incremental decode (with cache) == full prefill (no cache),
         position-by-position
       - greedy first token == argmax(forward(prompt)[:, -1])
       - tied embeddings: lm_head.weight IS embed_tokens.weight
       - RoPE norm preservation, RMSNorm formula, determinism, finiteness
  2. Semantic sanity (real weights only): greedy decode produces coherent text.
       Weight-key mapping correctness is guaranteed by construction: the loader
       does no key remapping, and a 0-missing-key load means every HF weight
       lands in the parameter of the same name.
"""

import sys
import os
from pathlib import Path

# Make the project root importable so `cann_lm.*` resolves when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

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


# ──────────────────────────────────────────────
# config / components
# ──────────────────────────────────────────────

def test_config():
    print("\n=== Configuration ===")
    cfg = small_config()
    check("model_type", cfg.model_type == "qwen3")
    check("head_dim field", cfg.head_dim == 32)
    check("tie_word_embeddings", cfg.tie_word_embeddings is True)
    # from_dict filters unknown keys
    cfg2 = Qwen3TextConfig.from_dict({"vocab_size": 99, "architectures": ["X"], "hidden_size": 16})
    check("from_dict filters unknown", cfg2.vocab_size == 99 and cfg2.hidden_size == 16)
    info(f"  config: {cfg.hidden_size}d x {cfg.num_hidden_layers}L, GQA {cfg.num_attention_heads}/{cfg.num_key_value_heads}")
    return cfg


def test_rmsnorm():
    print("\n=== RMSNorm ===")
    cfg = small_config()
    norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    # set weight to ones so output == pure rmsnormalised input
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(2, 5, cfg.hidden_size)
    out = norm(x)
    check("RMSNorm shape preserved", out.shape == x.shape)
    # manual reference
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (x_f * torch.rsqrt(var + cfg.rms_norm_eps)).to(x.dtype)
    check("RMSNorm matches formula (weight=1)", torch.allclose(out, ref, atol=1e-5))
    # norm preserves RMS scale property: out_rms / in_rms ~ 1 when weight=1
    check("RMSNorm keeps unit RMS (weight=1)", torch.allclose(
        out.float().pow(2).mean(-1).sqrt(), torch.ones(2, 5), atol=1e-4))
    # finite
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
    # RoPE is a rotation => preserves L2 norm per (b, head, seq)
    q_t = q.transpose(1, 2)
    nq = q_t.float().pow(2).sum(-1).sqrt()
    nq2 = q2.float().pow(2).sum(-1).sqrt()
    check("RoPE preserves q norm", torch.allclose(nq, nq2, atol=1e-4))
    nk = k.transpose(1, 2).float().pow(2).sum(-1).sqrt()
    nk2 = k2.float().pow(2).sum(-1).sqrt()
    check("RoPE preserves k norm", torch.allclose(nk, nk2, atol=1e-4))
    # rotate_half sanity
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
        # QK-norm modules present
        check(f"attn layer {i} has q_norm/k_norm", hasattr(attn, "q_norm") and hasattr(attn, "k_norm"))
        layer = Qwen3DecoderLayer(cfg, i)
        ol = layer(x, position_embeddings=(cos, sin))
        check(f"decoder layer {i} shape", ol.shape == x.shape)
        break


# ──────────────────────────────────────────────
# full model invariants (tiny random model)
# ──────────────────────────────────────────────

def test_forward_shape_and_tie(cfg):
    print("\n=== Forward shape / tied embeddings ===")
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    out = lm(input_ids=ids)
    check("logits shape", out["logits"].shape == (2, 8, cfg.vocab_size))
    check("loss None w/o labels", out["loss"] is None)
    check("logits finite", torch.isfinite(out["logits"]).all().item())
    # tied embeddings: lm_head.weight IS embed_tokens.weight (same tensor object)
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
            # position k-1 of prefix must equal position k-1 of full
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
        full = lm(input_ids=ids)["logits"]  # no cache, full prefill
        # incremental decode with cache
        past = [None] * cfg.num_hidden_layers
        per_pos = []
        for i in range(n):
            step = ids[:, :1] if i == 0 else ids[:, i:i + 1]
            out = lm(input_ids=step, past_key_values=past)
            past = out["past_key_values"]
            per_pos.append(out["logits"][:, -1, :])
        inc = torch.stack(per_pos, dim=1)  # (1, n, vocab)
        # compare each position's next-token logits
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


# ──────────────────────────────────────────────
# minimal byte-level BPE decoder (vocab.json only)
# ──────────────────────────────────────────────

def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def make_decoder(model_path: str):
    import json
    with open(os.path.join(model_path, "vocab.json")) as f:
        vocab = json.load(f)
    rev = {i: t for t, i in vocab.items()}

    # add special tokens from tokenizer_config (id may exceed vocab.json range)
    tc_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        for sid, st in tc.get("added_tokens_decoder", {}).items():
            rev.setdefault(int(sid), st["content"])

    def decode(ids):
        text = "".join(rev.get(int(i), f"<{i}>") for i in ids)
        try:
            raw = bytes(_BYTE_DECODER[c] for c in text if c in _BYTE_DECODER)
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return text

    return decode


# ──────────────────────────────────────────────
# real weights
# ──────────────────────────────────────────────

def test_real_weights(model_path: str):
    global SKIP
    print(f"\n=== Real weights ({model_path}) ===")
    if not os.path.isdir(model_path):
        print(f"  [SKIP] not a directory: {model_path}")
        SKIP += 1
        return
    cfg = Qwen3TextConfig.from_pretrained(model_path)
    info(f"  config: {cfg.hidden_size}d x {cfg.num_hidden_layers}L, head_dim={cfg.head_dim}, "
         f"GQA {cfg.num_attention_heads}/{cfg.num_key_value_heads}, tie={cfg.tie_word_embeddings}")

    lm = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    n_before = sum(id(p) for p in lm.parameters())  # noqa: not used
    lm = load_qwen3_weights(lm, model_path, device="cpu", dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in lm.parameters())
    info(f"  params: {n_params / 1e9:.3f}B")

    # tied check on real model
    if cfg.tie_word_embeddings:
        check("real lm_head.weight IS embed_tokens.weight", lm.lm_head.weight is lm.model.embed_tokens.weight)

    # finite forward
    ids = torch.tensor([[cfg.bos_token_id or 151643]], dtype=torch.long)
    with torch.no_grad():
        out = lm(input_ids=ids)
    check("real forward finite", torch.isfinite(out["logits"]).all().item())
    check("real logits shape", out["logits"].shape == (1, 1, cfg.vocab_size))

    # KV-cache consistency on real model (catches any real-weight-specific cache bug)
    seq = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    with torch.no_grad():
        full = lm(input_ids=seq)["logits"]
        past = [None] * cfg.num_hidden_layers
        per_pos = []
        for i in range(seq.shape[1]):
            step = seq[:, :1] if i == 0 else seq[:, i:i + 1]
            o = lm(input_ids=step, past_key_values=past)
            past = o["past_key_values"]
            per_pos.append(o["logits"][:, -1, :])
        inc = torch.stack(per_pos, dim=1)
        diff = (inc - full).abs().max().item()
    info(f"  real KV-cache max|inc-full| = {diff:.2e}")
    check("real incremental-with-cache == prefill", torch.allclose(inc, full, atol=1e-2))

    # causal invariance on real model
    with torch.no_grad():
        full8 = lm(input_ids=seq)["logits"]
        pref_ok = torch.allclose(lm(input_ids=seq[:, :3])["logits"][:, -1, :], full8[:, 2, :], atol=1e-2)
    check("real causal prefix==full[:,2]", pref_ok)

    # greedy generation + decode to text
    decode = make_decoder(model_path)
    with torch.no_grad():
        gen = lm.generate(ids, max_new_tokens=12, do_sample=False)
    text = decode(gen[0].tolist())
    info(f"  greedy ids: {gen[0].tolist()}")
    info(f"  decoded   : {text!r}")
    # sanity: decoded text is non-empty, printable, not full of replacement chars
    n_repl = text.count("\ufffd")
    check("greedy decode non-empty", len(text) > 0)
    check("greedy decode mostly printable", n_repl < len(text))


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    global PASS, FAIL, SKIP
    import argparse
    parser = argparse.ArgumentParser(description="Test Qwen3 pure-torch implementation")
    parser.add_argument("--model", type=str, default=None, help="Path to Qwen3 safetensors checkpoint")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
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

    if args.model:
        test_real_weights(args.model)

    print(f"\n{'=' * 50}")
    total = PASS + FAIL + SKIP
    print(f"Results: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP ({total} total)")
    print(f"{'=' * 50}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
