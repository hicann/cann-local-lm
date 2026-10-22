#!/usr/bin/env python3
"""导出 qwen3.5 decode 图 (seq_len=1, 走 conv1d_update + chunk decode 分支).

decode 分支由 delta_cache._has_state 触发 (非 past_key_values). 构造:
- delta_cache (SimpleDeltaNetCache, _has_state=True) 设到 wrapper._delta_cache_box
- past=6 对 full_attn KV buffer (非 None)
- seq_len=1 sample (input_ids[1,1], decode mask, pos=[cached_len], new_kv=[cached_len])

验证 torch.export 能否 trace decode 分支 (conv1d_update 不走 nn.Conv1d prefill 路径),
后续 OM 编译能否过 infershape/tiling.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT) not in sys.path: sys.path.insert(0, str(_PROJECT))
from cann_lm.converter.export.ge_converters import register_ge_converters
from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from cann_lm.models.qwen3_5.modeling_qwen3_5 import SimpleDeltaNetCache
import torch_delegate
from cann_lm.converter.export.epair_utils import consolidated_weights_save

HF = os.environ["HF"]
KV_MAX = 2048
CACHED_LEN = 8
OUT_DIR = str(Path(__file__).resolve().parent / "epair_qwen3_5_decode")


def main():
    register_ge_converters()
    cfg = Qwen3_5TextConfig.from_pretrained(HF)
    m = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    m = load_qwen3_5_weights(m, HF, device="cpu", dtype=torch.bfloat16)
    w = Qwen3_5ExportWrapper(m, kv_cache_max_len=KV_MAX, batch=1).eval()

    # delta_cache: 触发 decode 分支 (has_previous_state -> use_precomputed_states=True)
    lin = next(l.linear_attn for l in m.model.layers if l.layer_type == "linear_attention")
    dc = SimpleDeltaNetCache(
        num_layers=len(m.model.layers), conv_dim=lin.conv_dim,
        conv_kernel_size=lin.conv_kernel_size, k_head_dim=lin.head_k_dim,
        v_head_dim=lin.head_v_dim, num_v_heads=lin.num_v_heads,
        batch_size=1, dtype=torch.bfloat16)
    dc._has_state = True
    w._delta_cache_box[0] = dc

    # decode sample: past=6 对 full_attn KV (非 None), seq_len=1, delta_states 图输入
    dtype = torch.bfloat16
    full_idx = w._get_full_attn_layer_indices()
    past = []
    for i in full_idx:
        hd = w._get_layer_head_dim(i)
        nk = cfg.num_key_value_heads
        past.append((torch.zeros(1, nk, KV_MAX, hd, dtype=dtype),
                     torch.zeros(1, nk, KV_MAX, hd, dtype=dtype)))
    # delta_states: (conv_flat, recurrent_flat) 每层一对, 作为图输入避常量 mutate
    lin = next(l.linear_attn for l in m.model.layers if l.layer_type == "linear_attention")
    n_linear = sum(1 for i, lt in enumerate(cfg.layer_types)
                   if lt == "linear_attention" and i < len(m.model.layers))
    conv_flat = [torch.zeros(1, lin.conv_dim, lin.conv_kernel_size, dtype=dtype)
                 for _ in range(n_linear)]
    rec_flat = [torch.zeros(1, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim, dtype=dtype)
                for _ in range(n_linear)]
    delta_states = (conv_flat, rec_flat)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.int64)
    minv = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, 1, KV_MAX, dtype=dtype)
    mask[:, :, :, CACHED_LEN + 1:] = minv
    pos = torch.tensor([[CACHED_LEN]], dtype=torch.int64)
    new_kv = torch.tensor([CACHED_LEN], dtype=torch.int64)
    out_pos = torch.tensor([0], dtype=torch.int32)
    sample = (input_ids, mask, pos, past, new_kv, None, out_pos, None, None, delta_states)

    print("[1/3] torch.export (decode, seq_len=1)...", flush=True)
    ep = torch.export.export(w, sample, strict=False)
    print(f"      OK, nodes={len(ep.graph.nodes)}", flush=True)

    print("[2/3] export_to_epair...", flush=True)
    epair = torch_delegate.export_to_epair(ep)

    out = Path(OUT_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    consolidated_weights_save(epair, out, save_weights=True, max_file_size=512 * 1024 * 1024)
    print(f"[3/3] saved: {out}", flush=True)


if __name__ == "__main__":
    main()
