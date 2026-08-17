#!/usr/bin/env python3
"""验证 gemma4 Gemma4ExportWrapper 重写: host mask 与 model 4D 一致 + bypass vs model.forward

重写后 wrapper 绕过 Gemma4TextModel.forward 直接驱动层, sliding_attention 的 mask 由
host 预构造 (mask_full=causal-over-buffer, mask_sliding=causal+window) 作为图输入;
图内不再有 masked_fill (NPU 不支持 ge.MaskedFill)。

本脚本验证两点 (sliding+full+PLE 配置, prefill+decode):
  1. host _build_mask_full / _build_mask_sliding == model _prepare_causal_mask 4D 分支输出
  2. wrapper.forward (bypass, host mask) == model.forward (OLD, 4D cob, 含 masked_fill)
     的 logits + KV (逐位一致, maxdiff=0)

环境 (converter venv, CPU-only):
  source /home/developer/Ascend/cann/set_env.sh
  export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  python verify_gemma4_wrapper_bypass.py
"""

import torch

from cann_lm.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def build_model():
    cfg = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=64,  # PLE 开启
        num_kv_shared_layers=0, sliding_window=3,
        rms_norm_eps=1e-6, max_position_embeddings=64, tie_word_embeddings=False,
        attention_bias=False, pad_token_id=0,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    torch.manual_seed(42)
    return Gemma4ForCausalLM(cfg).eval(), cfg


def cob_mask(q_positions, kv_len, dtype):
    """caller 传入的 causal-over-buffer [1,1,q,kv] (k<=q → 0, else MIN)。"""
    MIN = torch.finfo(dtype).min
    q = torch.tensor(q_positions, dtype=dtype).unsqueeze(1)
    k = torch.arange(kv_len, dtype=dtype).unsqueeze(0)
    return torch.where(k <= q, torch.tensor(0.0, dtype=dtype), MIN).unsqueeze(0).unsqueeze(0)


def main():
    model, cfg = build_model()
    tm = model.model
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=8, batch=1).eval()
    S, KV_MAX = 4, 8
    dtype = torch.float32
    input_ids = torch.randint(0, cfg.vocab_size, (1, S), dtype=torch.int64)
    next_tok = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.int64)

    ok = True
    with torch.no_grad():
        # ---------- prefill ----------
        nkp_pf = torch.arange(S)
        inputs_embeds = tm.embed_tokens(input_ids)
        cob_pf = cob_mask(list(range(S)), KV_MAX, dtype)
        m_full = tm._prepare_causal_mask(inputs_embeds, cob_pf, "full_attention", new_kv_cache_pos=nkp_pf)
        m_slide = tm._prepare_causal_mask(inputs_embeds, cob_pf, "sliding_attention", new_kv_cache_pos=nkp_pf)
        h_full = wrapper._build_mask_full(S, KV_MAX, dtype, nkp_pf)
        h_slide = wrapper._build_mask_sliding(S, KV_MAX, dtype, nkp_pf)
        mf = torch.equal(h_full, m_full)
        ms = torch.equal(h_slide, m_slide)
        print(f"[prefill] host mask_full   == model 4D full   : {mf}")
        print(f"[prefill] host mask_sliding== model 4D sliding: {ms}")
        ok &= mf and ms

        # OLD: model.forward (4D cob, 内部 _prepare_causal_mask 含 masked_fill)
        bufs = [(torch.zeros(1, 2, KV_MAX, 8, dtype=dtype),
                 torch.zeros(1, 2, KV_MAX, 8, dtype=dtype)) for _ in range(2)]
        pos_pf = torch.arange(S).unsqueeze(0)
        out_old = model(input_ids=input_ids, attention_mask=cob_pf, position_ids=pos_pf,
                        past_key_values=bufs, new_kv_cache_pos=nkp_pf, shared_kv_states=None)
        old_l0, old_kv = out_old["logits"], out_old["past_key_values"]
        old_kv_snap = [(k.clone(), v.clone()) for k, v in old_kv]

        # NEW: wrapper.forward (bypass, host mask)
        bufs2 = [(torch.zeros(1, 2, KV_MAX, 8, dtype=dtype),
                  torch.zeros(1, 2, KV_MAX, 8, dtype=dtype)) for _ in range(2)]
        new_l0, *new_flat0 = wrapper(
            input_ids=input_ids, mask_full=h_full, mask_sliding=h_slide,
            position_ids=pos_pf, past_key_values=bufs2, new_kv_cache_pos=nkp_pf,
            embeddings=None, per_layer_token_identity=None)
        l0_close = torch.allclose(old_l0, new_l0, atol=1e-5)
        print(f"[prefill] logits allclose (OLD vs NEW): {l0_close}  maxdiff={(old_l0-new_l0).abs().max().item():.2e}")
        ok &= l0_close
        for i in range(2):
            ke = torch.equal(old_kv_snap[i][0], new_flat0[i*2]) and \
                 torch.equal(old_kv_snap[i][1], new_flat0[i*2+1])
            print(f"[prefill] layer{i} KV equal: {ke}")
            ok &= ke

        # ---------- decode (q at position 4, sliding layer0 window=[2,3,4]) ----------
        nkp_dec = torch.tensor([S])
        cob_dec = cob_mask([S], KV_MAX, dtype)
        m_full_d = tm._prepare_causal_mask(inputs_embeds[:, -1:, :], cob_dec, "full_attention", new_kv_cache_pos=nkp_dec)
        m_slide_d = tm._prepare_causal_mask(inputs_embeds[:, -1:, :], cob_dec, "sliding_attention", new_kv_cache_pos=nkp_dec)
        h_full_d = wrapper._build_mask_full(1, KV_MAX, dtype, nkp_dec)
        h_slide_d = wrapper._build_mask_sliding(1, KV_MAX, dtype, nkp_dec)
        mf_d = torch.equal(h_full_d, m_full_d)
        ms_d = torch.equal(h_slide_d, m_slide_d)
        print(f"[decode] host mask_full   == model 4D full   : {mf_d}")
        print(f"[decode] host mask_sliding== model 4D sliding: {ms_d}")
        ok &= mf_d and ms_d

        pos_dec = torch.tensor([S]).unsqueeze(0)
        out_old_d = model(input_ids=next_tok, attention_mask=cob_dec, position_ids=pos_dec,
                          past_key_values=old_kv, new_kv_cache_pos=nkp_dec, shared_kv_states=None)
        old_l1, old_kv_d = out_old_d["logits"], out_old_d["past_key_values"]
        bufs_new = [(new_flat0[i*2], new_flat0[i*2+1]) for i in range(2)]
        new_l1, *new_flat1 = wrapper(
            input_ids=next_tok, mask_full=h_full_d, mask_sliding=h_slide_d,
            position_ids=pos_dec, past_key_values=bufs_new, new_kv_cache_pos=nkp_dec,
            embeddings=None, per_layer_token_identity=None)
        l1_close = torch.allclose(old_l1, new_l1, atol=1e-5)
        print(f"[decode] logits allclose (OLD vs NEW): {l1_close}  maxdiff={(old_l1-new_l1).abs().max().item():.2e}")
        ok &= l1_close
        for i in range(2):
            ke = torch.equal(old_kv_d[i][0], new_flat1[i*2]) and \
                 torch.equal(old_kv_d[i][1], new_flat1[i*2+1])
            print(f"[decode] layer{i} KV equal: {ke}")
            ok &= ke

    print()
    print("=" * 40)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 40)
    return ok


if __name__ == "__main__":
    main()
