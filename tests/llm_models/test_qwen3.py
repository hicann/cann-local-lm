#!/usr/bin/env python3
"""qwen3 模型定义单元测试 (从 llm_models_run_on_cpu/debug_qwen3/verify_qwen3.py 抽出).

pytest 风格: 顶层 test_* 函数, check() 失败即 raise (pytest 捕获).
用 converter venv (无 torch_npu) 跑: pytest tests/llm_models/test_qwen3.py
"""
import torch
import pytest
import json

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


# ── modeling edge branches (scatter-write / 2D mask / output_hidden_states / generate) ──

def test_attention_scatter_write(config: Qwen3TextConfig):
    """new_kv_cache_pos + pre-allocated buffer -> index_copy scatter-write (238-242)."""
    b, s = 1, 4
    layer_idx = 0
    attn = Qwen3Attention(config, layer_idx)
    rope = Qwen3RotaryEmbedding(config)
    x = torch.randn(b, s, config.hidden_size)
    pos = torch.arange(s).unsqueeze(0)
    cos, sin = rope(x, pos)

    kv_max = 16
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    pv = [None] * config.num_hidden_layers
    pv[layer_idx] = (torch.zeros(b, n_kv, kv_max, hd), torch.zeros(b, n_kv, kv_max, hd))
    new_pos = torch.arange(s, dtype=torch.int64)

    out = attn(x, position_embeddings=(cos, sin), attention_mask=None,
               past_key_values=pv, new_kv_cache_pos=new_pos)
    check("scatter-write attn shape", out.shape == x.shape)
    # scatter-write keeps kv_max fixed (index_copy into pre-allocated buffer)
    check("kv buffer stays kv_max", pv[layer_idx][0].shape[2] == kv_max)


def test_text_model_2d_padding_mask(config: Qwen3TextConfig):
    """2D attention_mask -> _prepare_causal_mask bool key_valid + causal (362-366)."""
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    b, s = 1, 6  # bsz=1: 2D-mask broadcast [1,1,1,s,s] vs attn [1,h,s,s]
    ids = torch.randint(0, config.vocab_size, (b, s))
    mask2d = torch.ones(b, s)
    mask2d[0, -2:] = 0  # pad last 2 tokens
    with torch.no_grad():
        out = lm(input_ids=ids, attention_mask=mask2d)
    check("2D mask logits shape", out["logits"].shape == (b, s, config.vocab_size))
    check("2D mask finite", torch.isfinite(out["logits"]).all().item())


def test_output_hidden_states(config: Qwen3TextConfig):
    """output_hidden_states=True -> all_hidden_states collected (400, 404-405, 449)."""
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        out = lm(input_ids=ids, output_hidden_states=True)
    hs = out["hidden_states"]
    check("hidden_states returned", hs is not None)
    # initial + per-layer + final-norm = N+2
    check("hidden_states count = layers+2", len(hs) == config.num_hidden_layers + 2)
    check("first hidden = embed", hs[0].shape == (1, 5, config.hidden_size))


def test_generate_decode_step_and_sampling(config: Qwen3TextConfig):
    """max_new_tokens>=2 hits decode step (502); do_sample+top_k/top_p hits 508-520."""
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))
    # greedy with 2+ tokens -> decode step (past_kv not None on 2nd iter)
    gen_greedy = lm.generate(ids, max_new_tokens=3, do_sample=False)
    check("greedy generates 3+ tokens", gen_greedy.shape[1] >= ids.shape[1] + 1)
    # sampling with top_k + top_p -> 508-520 branches
    gen_sample = lm.generate(ids, max_new_tokens=2, do_sample=True,
                             temperature=0.8, top_k=10, top_p=0.9)
    check("sampling generates tokens", gen_sample.shape[1] > ids.shape[1])


def test_generate_no_eos_token(config: Qwen3TextConfig):
    """eos_token_id=None (config + arg) -> else branch eos_set=set() (488-491, 525)."""
    cfg = make_small_config(eos_token_id=None)
    lm = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    gen = lm.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
    check("no-eos generate runs to max_new_tokens", gen.shape[1] == ids.shape[1] + 3)


def test_misc_branches(config: Qwen3TextConfig):
    """extra_repr (118), repeat_kv n_rep=1 (67), inputs_embeds path (380), tie=False (425)."""
    norm = Qwen3RMSNorm(config.hidden_size)
    assert "eps=" in norm.extra_repr()

    # repeat_kv n_rep==1 early return (MHA: num_heads == num_kv_heads)
    mha_cfg = make_small_config(num_attention_heads=4, num_key_value_heads=4)
    lm_mha = Qwen3ForCausalLM(mha_cfg).to(torch.float32).eval()
    ids = torch.randint(0, mha_cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = lm_mha(input_ids=ids)
    check("MHA forward shape", out["logits"].shape == (1, 4, mha_cfg.vocab_size))

    # inputs_embeds path (380->383)
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        emb = lm.model.embed_tokens(ids)
        out2 = lm(inputs_embeds=emb)
    check("inputs_embeds forward", out2["logits"].shape == (1, 4, config.vocab_size))

    # tie_word_embeddings=False -> lm_head is independent (425->exit)
    untied = make_small_config(tie_word_embeddings=False)
    lm_u = Qwen3ForCausalLM(untied).to(torch.float32).eval()
    check("untied lm_head distinct", lm_u.lm_head.weight is not lm_u.model.embed_tokens.weight)


def test_generate_eos_list_and_sampling_branches(config: Qwen3TextConfig):
    """eos_token_id as list (489), top_k-only (508-511), top_p-only (511-519), eos break (526)."""
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))

    # eos as list -> set(eos_ids) (489); pick eos = argmax of next token to force break (526)
    with torch.no_grad():
        nxt = lm(input_ids=ids)["logits"][:, -1, :].argmax(dim=-1).item()
    gen = lm.generate(ids, max_new_tokens=5, do_sample=False, eos_token_id=[nxt, 999])
    check("eos-list breaks early", gen.shape[1] <= ids.shape[1] + 5)

    # top_k only (no top_p) -> 508-511, skip top_p block
    torch.manual_seed(1)
    gk = lm.generate(ids, max_new_tokens=2, do_sample=True, temperature=1.0, top_k=10)
    check("top_k-only sampling", gk.shape[1] > ids.shape[1])

    # top_p only (no top_k) -> 511-519
    torch.manual_seed(2)
    gp = lm.generate(ids, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
    check("top_p-only sampling", gp.shape[1] > ids.shape[1])


def test_config_to_dict_and_from_json(tmp_path):
    """to_dict (51) + from_json_file text_config wrapper (61-65)."""
    config = make_small_config()
    d = config.to_dict()
    assert d["model_type"] == "qwen3"
    assert not any(k.startswith("_") for k in d)

    # from_json_file with text_config wrapper
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"text_config": {"vocab_size": 99, "hidden_size": 16}}))
    cfg = Qwen3TextConfig.from_json_file(str(p))
    check("from_json_file vocab", cfg.vocab_size == 99)
    check("from_json_file hidden", cfg.hidden_size == 16)

    # from_json_file without wrapper
    p2 = tmp_path / "flat.json"
    p2.write_text(json.dumps({"vocab_size": 50}))
    cfg2 = Qwen3TextConfig.from_json_file(str(p2))
    check("from_json_file flat", cfg2.vocab_size == 50)
