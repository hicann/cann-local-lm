#!/usr/bin/env python3
"""qwen3 模型定义单元测试 (从 llm_models_run_on_cpu/debug_qwen3/verify_qwen3.py 抽出).

pytest 风格: 顶层 test_* 函数, check() 失败即 raise (pytest 捕获).
用 converter venv (无 torch_npu) 跑: pytest tests/llm_models/test_qwen3.py
"""
import torch
import pytest

from cann_lm.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from cann_lm.models.qwen3.configuration_qwen3 import Qwen3TextConfig
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper


def check(name, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    assert condition, f"{name} failed"


def info(msg):
    print(f"  [INFO] {msg}")


def make_small_config(**over) -> Qwen3TextConfig:
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


@pytest.fixture
def config():
    return make_small_config()


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """4D causal mask [1, 1, seq_len, seq_len] (单元测试用)."""
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def test_config(config: Qwen3TextConfig):
    print("\n=== Configuration ===")
    check("model_type", config.model_type == "qwen3")
    check("head_dim field", config.head_dim == 32)
    check("tie_word_embeddings", config.tie_word_embeddings is True)
    cfg2 = Qwen3TextConfig.from_dict({"vocab_size": 99, "architectures": ["X"], "hidden_size": 16})
    check("from_dict filters unknown", cfg2.vocab_size == 99 and cfg2.hidden_size == 16)
    info(f"  config: {config.hidden_size}d x {config.num_hidden_layers}L, GQA {config.num_attention_heads}/{config.num_key_value_heads}")


def test_rmsnorm(config: Qwen3TextConfig):
    print("\n=== RMSNorm ===")
    norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(2, 5, config.hidden_size)
    out = norm(x)
    check("RMSNorm shape preserved", out.shape == x.shape)
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (x_f * torch.rsqrt(var + config.rms_norm_eps)).to(x.dtype)
    check("RMSNorm matches formula (weight=1)", torch.allclose(out, ref, atol=1e-5))
    check("RMSNorm keeps unit RMS (weight=1)", torch.allclose(
        out.float().pow(2).mean(-1).sqrt(), torch.ones(2, 5), atol=1e-4))
    check("RMSNorm finite", torch.isfinite(out).all().item())


def test_rope(config: Qwen3TextConfig):
    print("\n=== RoPE ===")
    rope = Qwen3RotaryEmbedding(config)
    b, s, h, d = 2, 6, 3, config.head_dim
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


def test_mlp(config: Qwen3TextConfig):
    print("\n=== MLP ===")
    mlp = Qwen3MLP(config)
    x = torch.randn(2, 5, config.hidden_size)
    out = mlp(x)
    check("MLP shape", out.shape == x.shape)
    check("MLP finite", torch.isfinite(out).all().item())
    check("MLP has gate/up/down", hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"))


def test_attention_and_layer(config: Qwen3TextConfig):
    print("\n=== Attention / DecoderLayer ===")
    b, s = 2, 6
    rope = Qwen3RotaryEmbedding(config)
    pos = torch.arange(s).unsqueeze(0)
    cos, sin = rope(torch.randn(b, s, config.hidden_size), pos)
    for i in range(config.num_hidden_layers):
        attn = Qwen3Attention(config, i)
        x = torch.randn(b, s, config.hidden_size)
        o = attn(x, position_embeddings=(cos, sin))
        check(f"attn layer {i} shape", o.shape == x.shape)
        check(f"attn layer {i} finite", torch.isfinite(o).all().item())
        check(f"attn layer {i} has q_norm/k_norm", hasattr(attn, "q_norm") and hasattr(attn, "k_norm"))
        layer = Qwen3DecoderLayer(config, i)
        ol = layer(x, position_embeddings=(cos, sin))
        check(f"decoder layer {i} shape", ol.shape == x.shape)
        break


def test_forward_shape_and_tie(config: Qwen3TextConfig):
    print("\n=== Forward shape / tied embeddings ===")
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    out = lm(input_ids=ids)
    check("logits shape", out["logits"].shape == (2, 8, config.vocab_size))
    check("loss None w/o labels", out["loss"] is None)
    check("logits finite", torch.isfinite(out["logits"]).all().item())
    if config.tie_word_embeddings:
        same = lm.lm_head.weight is lm.model.embed_tokens.weight
        check("lm_head.weight IS embed_tokens.weight (tied)", same)
    info(f"  params: {sum(p.numel() for p in lm.parameters()):,}")


def test_determinism(config: Qwen3TextConfig):
    print("\n=== Determinism (eval) ===")
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 6), dtype=torch.long)
    with torch.no_grad():
        o1 = lm(input_ids=ids)["logits"]
        o2 = lm(input_ids=ids)["logits"]
    check("two forwards identical", torch.equal(o1, o2))


def test_causal_invariance(config: Qwen3TextConfig):
    print("\n=== Causal invariance: forward(prefix) == forward(full)[:, :k] ===")
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
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


def test_kv_cache_consistency(config: Qwen3TextConfig):
    print("\n=== KV cache: incremental decode == full prefill ===")
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    n = ids.shape[1]
    with torch.no_grad():
        full = lm(input_ids=ids)["logits"]
        past = [None] * config.num_hidden_layers
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


def test_greedy_equals_argmax(config: Qwen3TextConfig):
    print("\n=== greedy first token == argmax(forward(prompt)[:, -1]) ===")
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        argmax_next = lm(input_ids=ids)["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        gen = lm.generate(ids, max_new_tokens=1, do_sample=False)
    check("greedy first token == argmax", torch.equal(gen[:, -1:], argmax_next))


def test_loss(config: Qwen3TextConfig):
    print("\n=== Loss / labels ===")
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    labels = ids.clone()
    with torch.no_grad():
        out = lm(input_ids=ids, labels=labels)
    check("loss scalar", out["loss"] is not None and out["loss"].ndim == 0)
    check("loss finite", torch.isfinite(out["loss"]).all().item())


def test_truncation():
    print("\n=== Layer truncation ===")
    for n in [2, 4, 8]:
        cfg = make_small_config(num_hidden_layers=n)
        lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
        out = lm(input_ids=ids)["logits"]
        check(f"{n} layers logits", out.shape == (1, 4, cfg.vocab_size))


def test_wrapper_vs_model(config: Qwen3TextConfig):
    """验证 Qwen3ExportWrapper 单步 forward 与原始 model 一致 (max_diff=0.0)."""
    print("\n=== Wrapper vs Model (单步, max_diff=0.0) ===")
    b, s = 1, 8
    model = Qwen3ForCausalLM(config).to(torch.float32).eval()
    wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    dtype = next(model.parameters()).dtype

    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.long)
    attention_mask = make_causal_mask(s, dtype)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([s - 1], dtype=torch.int32)

    with torch.no_grad():
        out_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=[None] * config.num_hidden_layers,
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
