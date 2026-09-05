#!/usr/bin/env python3
"""ACL python 运行 qwen3_5.om + 对比 eager wrapper (验证 OM 运行机制)。

仿照 run_gemma4_om.py, 适配 qwen3.5 的 delta_states 图输入 (prefill 含零 delta).
流程: wrapper.get_sample_inputs (CPU, 含 delta_states) → eager CPU 跑 (chunk CPU impl)
      → OM (acl NPU) 用同一份 sample 跑 → 对比 OM logits vs eager logits.
"""
import os, sys, json, dataclasses
import torch

# input_ids 入图 OM (已清除, 需用 CLI `cann_lm_converter.py --no-use-embeddings` 重新导出+编译).
# embeddings 入图验证用 run_qwen3_5_om_emb.py (推荐, OM=qwen3_5_emb_noaub.om).
OM = os.environ["QWEN35_OM"]
HF = os.environ["HF"]
SEQ = 8; KV_MAX = 2048; DEV = 0
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper

import acl
ACL_OK = 0; H2D = 1; D2H = 2
def check(n, r):
    if r != ACL_OK: raise RuntimeError(f"{n} ret={r}")
def acl_init_tolerant():
    r = acl.init()
    if r not in (0, 100002): raise RuntimeError(f"acl.init ret={r}")
def t_bytes(t):
    t = t.cpu().contiguous()
    if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()
def bytes_to_tensor(b, shape, dtype):
    if dtype == torch.bfloat16: return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
    return torch.frombuffer(b, dtype=dtype).reshape(shape)


def flatten_to_om_inputs(sample, in_names, in_dtypes):
    """展平 qwen3.5 sample → OM 输入 (按 OM input name 顺序).

    sample: (input_ids, mask, pos, past[24对], new_kv, None, output_pos, None, None, delta_states)
    delta_states = (conv_flat[18], rec_flat[18])
    """
    input_ids, mask, pos, past_kv, new_kv, emb, output_pos, ple, emb_s, delta_states = sample
    conv_flat, rec_flat = delta_states
    kv_flat = []
    for k, v in past_kv:
        kv_flat.append(k); kv_flat.append(v)
    # delta input 展平: torch.export pytree 展平 (conv_flat 18 + rec_flat 18)
    delta_flat = list(conv_flat) + list(rec_flat)
    by_name = {"input_ids": input_ids, "attention_mask": mask, "position_ids": pos,
               "new_kv_cache_pos": new_kv, "output_pos": output_pos}
    inputs = []
    for name in in_names:
        if name in by_name:
            inputs.append(by_name[name])
        elif name.startswith("past_key_values_"):
            parts = name.split("_"); L = int(parts[-2]); K = int(parts[-1])
            inputs.append(kv_flat[L * 2 + K])
        elif name.startswith("delta_states_"):
            # delta_states_A_B: A=0 conv_flat, A=1 recurrent_flat; B=layer index
            parts = name.split("_"); A = int(parts[-2]); B = int(parts[-1])
            n_lin = len(delta_flat) // 2
            inputs.append(delta_flat[A * n_lin + B])
        else:
            raise ValueError(f"unknown OM input {name}")
    return inputs


def run_om(sample):
    """load OM → 查 in/out names → flatten sample → execute → collect."""
    acl_init_tolerant(); check("set_device", acl.rt.set_device(DEV))
    mid, ret = acl.mdl.load_from_file(OM); check("load", ret)
    desc = acl.mdl.create_desc(); check("get_desc", acl.mdl.get_desc(desc, mid))
    ni = acl.mdl.get_num_inputs(desc)
    in_names = [acl.mdl.get_input_name_by_index(desc, i) for i in range(ni)]
    print(f"  OM inputs ({ni}): {in_names[:8]}{'...' if ni>8 else ''}")
    inputs = flatten_to_om_inputs(sample, in_names, None)
    no = acl.mdl.get_num_outputs(desc)
    out_names = [acl.mdl.get_output_name_by_index(desc, i) for i in range(no)]
    print(f"  OM outputs ({no}): {out_names[:4]}{'...' if no>4 else ''}")
    in_ds = acl.mdl.create_dataset(); in_bufs = []
    for t in inputs:
        b = t_bytes(t); sz = len(b); buf, ret = acl.rt.malloc(sz, 2); check("malloc", ret)
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(buf, sz)); in_bufs.append((buf, sz, b))
    out_ds = acl.mdl.create_dataset(); out_meta = []
    for i in range(no):
        sz = acl.mdl.get_output_size_by_index(desc, i); buf, ret = acl.rt.malloc(sz, 2); check("malloc out", ret)
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(buf, sz))
        dims, _ = acl.mdl.get_output_dims(desc, i)
        out_meta.append((buf, sz, tuple(dims["dims"][:dims["dimCount"]])))
    for buf, sz, b in in_bufs:
        check("memcpy in", acl.rt.memcpy(buf, sz, acl.util.bytes_to_ptr(b), len(b), H2D))
    print("  executing OM..."); check("execute", acl.mdl.execute(mid, in_ds, out_ds))
    outs = []
    for buf, sz, shape in out_meta:
        hb, _ = acl.rt.malloc_host(sz); check("memcpy out", acl.rt.memcpy(hb, sz, buf, sz, D2H))
        ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb)
        outs.append(bytes_to_tensor(ob, shape, torch.bfloat16))
    for buf, _, _ in in_bufs: acl.rt.free(buf)
    for buf, _, _ in out_meta: acl.rt.free(buf)
    acl.mdl.unload(mid); acl.rt.reset_device(DEV); acl.finalize()
    return outs, in_names, out_names


def main():
    print("[1/3] Load qwen3.5-0.8B (CPU) + wrapper + get_sample_inputs")
    cfg = Qwen3_5TextConfig.from_pretrained(HF)
    model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_qwen3_5_weights(model, HF, device="cpu", dtype=torch.bfloat16)
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    sample = wrapper.get_sample_inputs(seq_len=SEQ)
    print(f"  layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}, sample len={len(sample)}")
    # 用真实 input_ids (非 random) 测试 nan 是否和输入有关
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    real_ids = tok("Hello, my name is", return_tensors="pt")["input_ids"]
    if real_ids.shape[1] < SEQ:
        real_ids = torch.cat([real_ids, torch.zeros(1, SEQ - real_ids.shape[1], dtype=torch.int64)], dim=1)
    else:
        real_ids = real_ids[:, :SEQ]
    sample = list(sample); sample[0] = real_ids; sample = tuple(sample)
    print(f"  input_ids={real_ids.tolist()}")
    input_ids = sample[0]

    print("[2/3] OM (acl NPU) run → logits_om")
    outs, in_names, out_names = run_om(sample)
    logits_om = outs[0]
    om_top5 = torch.topk(logits_om[0,0].float(),5).indices.tolist()
    print(f"  OM logits shape={list(logits_om.shape)}, top5={om_top5}")
    print(f"  OM logits[:5]={logits_om[0,0,:5].float().tolist()}, has_nan={torch.isnan(logits_om).any().item()}, has_inf={torch.isinf(logits_om).any().item()}")

    # [3a] OM vs EP (相同输入, OM NPU vs EP CPU) — 验证 OM pipeline 正确性
    ep_npy = os.path.join(os.path.dirname(__file__), "ep_logits_dump.npz")
    if os.path.exists(ep_npy):
        print("[3a] compare OM vs EP (same input, OM NPU vs EP CPU)")
        import numpy as _np
        d = _np.load(ep_npy)
        logits_ep = torch.from_numpy(d["logits"]).bfloat16()
        ids_ep = torch.from_numpy(d["input_ids"])
        ids_match = torch.equal(ids_ep, input_ids)
        diff_ep = (logits_om.float() - logits_ep.float()).abs().max().item()
        ep_top5 = torch.topk(logits_ep[0, 0].float(), 5).indices.tolist()
        print(f"  EP logits shape={list(logits_ep.shape)}, top5={ep_top5}, has_nan={torch.isnan(logits_ep).any().item()}")
        print(f"  input_ids match: {ids_match}")
        print(f"  OM vs EP logits maxdiff = {diff_ep:.4e}  top5_overlap={len(set(om_top5)&set(ep_top5))}/5  -> {'MATCH' if diff_ep < 1.0 else 'MISMATCH'}")
    else:
        print("[3a] skip OM vs EP (ep_logits_dump.npz not found; run dump_ep_logits.py in venv first)")

    # [3b] OM vs HF transformers (cann_lm vs HF 实现差异, 仅参考)
    print("[3b] compare OM vs HF transformers")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llm_models_run_on_cpu"))
    import hf_cpu  # 禁 torch_npu probe (shim)
    from transformers import Qwen3_5ForCausalLM as HFModel
    hf_model = HFModel.from_pretrained(HF, torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
    with torch.no_grad():
        out_hf = hf_model(input_ids=input_ids)
    logits_hf = out_hf.logits[:, -1:, :]  # 取最后 token
    diff = (logits_om.float() - logits_hf.float()).abs().max().item()
    hf_top5 = torch.topk(logits_hf[0, 0].float(), 5).indices.tolist()
    print(f"  HF logits shape={list(logits_hf.shape)}, top5={hf_top5}")
    print(f"  OM vs HF logits maxdiff = {diff:.4e}  top5_overlap={len(set(om_top5)&set(hf_top5))}/5  -> {'MATCH' if diff < 5.0 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
