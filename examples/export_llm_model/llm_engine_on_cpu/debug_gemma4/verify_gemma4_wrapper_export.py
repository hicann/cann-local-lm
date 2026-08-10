#!/usr/bin/env python3
"""验证 gemma4 Gemma4ExportWrapper 可导出 + 导出图无 masked_fill 节点

重写后 sliding mask 由 host 预构造作为图输入, 图内不再构造 mask。本脚本:
  1. torch.export 能 trace wrapper.forward (图内查表 + use_embeddings 两模式)
  2. FX 图中 masked_fill 节点数 == 0 (NPU 不支持 ge.MaskedFill, 必须为 0)
  3. mask 应用走 aten.add (→ ge.Add, 支持)

环境 (converter venv, CPU-only):
  source /home/developer/Ascend/cann/set_env.sh
  export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  python verify_gemma4_wrapper_export.py
"""

import torch

from cann_lm.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def build():
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
    return Gemma4ForCausalLM(cfg).eval()


def count_nodes(ep):
    n_mf = n_add = 0
    for n in ep.graph.nodes:
        if n.op == "call_function":
            t = str(n.target)
            if "masked_fill" in t:
                n_mf += 1
            if t.endswith("aten.add.default"):
                n_add += 1
    return n_mf, n_add


def main():
    model = build()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=8, batch=1).eval()
    ok = True

    # 1. 图内查表模式 (input_ids, PLE 内查)
    si = wrapper.get_sample_inputs(seq_len=4, use_embeddings=False)
    try:
        ep = torch.export.export(wrapper, si, strict=False)
        n_mf, n_add = count_nodes(ep)
        print(f"[use_embeddings=False] torch.export OK; masked_fill={n_mf} (期望 0), aten.add={n_add}")
        ok &= (n_mf == 0)
    except Exception as e:
        print(f"[use_embeddings=False] EXPORT FAIL: {repr(e)[:300]}")
        ok = False

    # 2. use_embeddings=True (host 查表 + per_layer_token_identity), detach 后导出 (模拟 saver)
    si2 = wrapper.get_sample_inputs(seq_len=4, use_embeddings=True)
    wrapper.detach_externalized_lookups()
    try:
        ep2 = torch.export.export(wrapper, si2, strict=False)
        n_mf2, _ = count_nodes(ep2)
        print(f"[use_embeddings=True]  torch.export OK (detached); masked_fill={n_mf2} (期望 0)")
        ok &= (n_mf2 == 0)
    except Exception as e:
        print(f"[use_embeddings=True]  EXPORT FAIL: {repr(e)[:300]}")
        ok = False

    print()
    print("=" * 40)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 40)
    return ok


if __name__ == "__main__":
    main()
