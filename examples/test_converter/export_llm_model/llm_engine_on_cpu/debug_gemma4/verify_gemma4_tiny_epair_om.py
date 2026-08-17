#!/usr/bin/env python3
"""导出 tiny gemma4 (sliding+full+PLE) 到 epair, 供 main.cpp 编译为 OM 验证新 wrapper。

新 wrapper 绕过 forward + host 预构 mask, 图内无 masked_fill。本脚本验证:
  torch.export -> torchair export_to_epair -> consolidated_weights_save 产出 epair 目录
(含 *_ep_json_with_ascend_ir.json + weight_info.json + weights_*.bin), 供
om_npu_verify/build/export_llm_model 编译为 OM。

环境 (converter venv, CPU + torchair stubs):
  source /home/developer/Ascend/cann/set_env.sh
  export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  python verify_gemma4_tiny_epair_om.py [output_dir]
"""

import sys
from pathlib import Path

import torch

from cann_lm.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.converter.export.ge_converter import register_ge_converters
from cann_lm.converter.export.epair_utils import consolidated_weights_save


def build_tiny():
    import os
    ple = int(os.environ.get("GEMMA4_PLE", "1"))
    sliding = int(os.environ.get("GEMMA4_SLIDING", "1"))
    real = int(os.environ.get("GEMMA4_REALDIMS", "0"))  # 1=用真实 e2b 算子形状
    if real:
        # 真实 e2b 算子形状 (head_dim=256/global=512/hidden=2304/sliding=512) 但 2 层 + 小 vocab, 快
        hd, ghd, hid, inter, kv, sw, vocab, layers = 256, 512, 2304, 9216, 2048, 512, 64, 2
        heads, kvh = 8, 4
    else:
        hd, ghd, hid, inter, kv, sw, vocab, layers = 8, 8, 32, 64, 8, 3, 64, 2
        heads, kvh = 4, 2
    layer_types = (["sliding_attention", "full_attention"] if sliding
                   else ["full_attention", "full_attention"])
    cfg = Gemma4TextConfig(
        vocab_size=vocab, hidden_size=hid, intermediate_size=inter, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kvh, head_dim=hd, global_head_dim=ghd,
        hidden_size_per_layer_input=(256 if ple and real else (8 if ple else 0)),
        vocab_size_per_layer_input=64,
        num_kv_shared_layers=0, sliding_window=sw,
        rms_norm_eps=1e-6, max_position_embeddings=kv, tie_word_embeddings=False,
        attention_bias=False, pad_token_id=0,
        layer_types=layer_types,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    torch.manual_seed(42)
    return Gemma4ForCausalLM(cfg).eval(), cfg, ple, sliding, kv


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_epair_gemma4").resolve()
    register_ge_converters()
    model, cfg, ple, sliding, kv = build_tiny()
    print(f"config: PLE={ple} sliding={sliding} layer_types={cfg.layer_types} "
          f"head_dim={cfg.head_dim} hidden={cfg.hidden_size} kv_max={kv}")
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=kv, batch=1).eval()

    sample_inputs = wrapper.get_sample_inputs(seq_len=4, use_embeddings=False)
    print(f"[1/3] torch.export (sample args={len(sample_inputs)})")
    exported = torch.export.export(wrapper, sample_inputs, strict=False)

    import torch_delegate
    print("[2/3] torchair export_to_epair")
    epair = torch_delegate.export_to_epair(exported)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] consolidated_weights_save -> {out_dir}")
    consolidated_weights_save(epair, out_dir, save_weights=True,
                              max_file_size=512 * 1024 * 1024)

    jsons = sorted(p.name for p in out_dir.glob("*ep_json*ascend_ir*.json"))
    wi = (out_dir / "weight_info.json").name if (out_dir / "weight_info.json").exists() else None
    print()
    print("=== epair 产物 ===")
    print(f"  dir: {out_dir}")
    print(f"  ep_json: {jsons}")
    print(f"  weight_info: {wi}")
    bins = sorted(out_dir.glob("weights_*.bin"))
    print(f"  weights_*.bin: {[b.name for b in bins]}")
    if jsons and wi:
        print()
        print("编译命令:")
        print(f"  om_npu_verify/build/export_llm_model {out_dir}/{jsons[0]} {out_dir}/{wi} tiny_gemma4")


if __name__ == "__main__":
    main()
