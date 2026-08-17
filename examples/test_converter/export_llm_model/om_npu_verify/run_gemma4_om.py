#!/usr/bin/env python3
"""ACL python 运行 gemma4_noaub.om + 对比 eager wrapper (验证 OM 运行机制)。

流程: wrapper.get_sample_inputs (CPU) → eager 在 NPU 跑 (npu fused ops 仅 NPU) →
模型回 CPU → OM (acl) 用同一份 CPU sample 跑 → 对比 OM logits vs eager logits。
"""
import os, sys, json, dataclasses
import torch

OM = os.environ.get("GEMMA4_OM", "/mnt/workspace/project/CANN-LM_5883/examples/export_llm_model/om_npu_verify/gemma4_noaub.om")
E2B = "/mnt/workspace/data/models/gemma-4-E2B-it"
SEQ = 8; KV_MAX = 2048; DEV = 0
sys.path.insert(0, "/mnt/workspace/project/CANN-LM_5883")

import torch_npu  # noqa
from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

# --- ACL helpers ---
import acl
ACL_OK = 0; H2D = 1; D2H = 2
def check(n, r):
    if r != ACL_OK: raise RuntimeError(f"{n} ret={r}")
def acl_init_tolerant():
    """acl.init; torch_npu 可能已 init acl (ret=100002), 容错。"""
    r = acl.init()
    if r not in (0, 100002): raise RuntimeError(f"acl.init ret={r}")
def t_bytes(t):
    t = t.cpu().contiguous()
    if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()
def bytes_to_tensor(b, shape):
    return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)

def move_sample(sample, dev):
    """sample 10-tuple (含 past_kv list[(k,v)]); 整体移到 dev。"""
    res = []
    for t in sample:
        if isinstance(t, torch.Tensor): res.append(t.to(dev))
        elif isinstance(t, list): res.append([(k.to(dev), v.to(dev)) for k, v in t])
        else: res.append(t)
    return tuple(res)

def flatten_to_om_inputs(sample, in_names):
    """展平 wrapper sample → OM 77 输入 (按 OM input name 顺序)。"""
    _, mask_full, mask_sliding, position_ids, past_kv, new_pos, embeddings, output_pos, ple, _ = sample
    kv_flat = []
    for k, v in past_kv: kv_flat.append(k); kv_flat.append(v)
    by_name = {"mask_full": mask_full, "mask_sliding": mask_sliding, "position_ids": position_ids,
               "new_kv_cache_pos": new_pos, "embeddings": embeddings, "output_pos": output_pos,
               "per_layer_token_identity": ple}
    inputs = []
    for name in in_names:
        if name in by_name: inputs.append(by_name[name])
        elif name.startswith("past_key_values_"):
            parts = name.split("_"); L = int(parts[-2]); K = int(parts[-1])
            inputs.append(kv_flat[L * 2 + K])
        else: raise ValueError(f"unknown OM input {name}")
    return inputs

def run_om(sample):
    """单次 load OM → 查 in_names → flatten sample → execute → collect。"""
    acl_init_tolerant(); check("set_device", acl.rt.set_device(DEV))
    mid, ret = acl.mdl.load_from_file(OM); check("load", ret)
    desc = acl.mdl.create_desc(); check("get_desc", acl.mdl.get_desc(desc, mid))
    ni = acl.mdl.get_num_inputs(desc)
    in_names = [acl.mdl.get_input_name_by_index(desc, i) for i in range(ni)]
    inputs = flatten_to_om_inputs(sample, in_names)
    print(f"  OM: {ni} inputs, {acl.mdl.get_num_outputs(desc)} outputs; flattened {len(inputs)}")
    in_ds = acl.mdl.create_dataset(); in_bufs = []
    for t in inputs:
        b = t_bytes(t); sz = len(b); buf, ret = acl.rt.malloc(sz, 2); check("malloc", ret)
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(buf, sz)); in_bufs.append((buf, sz, b))
    out_ds = acl.mdl.create_dataset(); out_meta = []
    no = acl.mdl.get_num_outputs(desc)
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
        ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb); outs.append(bytes_to_tensor(ob, shape))
    for buf, _, _ in in_bufs: acl.rt.free(buf)
    for buf, _, _ in out_meta: acl.rt.free(buf)
    acl.mdl.unload(mid); acl.rt.reset_device(DEV); acl.finalize()
    return outs

def main():
    print("[1/4] Load gemma4-E2B (CPU) + wrapper + get_sample_inputs")
    with open(f"{E2B}/config.json") as f: hf_cfg = json.load(f)
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    cfg = Gemma4TextConfig(**{k: v for k, v in hf_cfg["text_config"].items() if k in valid}); cfg.__post_init__()
    model = Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_gemma4_weights(model, E2B, device="cpu", dtype=torch.bfloat16)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    sample = wrapper.get_sample_inputs(seq_len=SEQ, use_embeddings=True)  # CPU
    print(f"  layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}, sample len={len(sample)}")

    print("[2/4] eager (NPU) → out_eager")
    model.npu()
    sample_npu = move_sample(sample, "npu")
    with torch.no_grad():
        out_eager, *_ = wrapper(*sample_npu)
    out_eager = out_eager.cpu()
    print(f"  eager logits shape={list(out_eager.shape)}, top5={torch.topk(out_eager[0,0].float(),5).indices.tolist()}")
    model.cpu()  # free NPU for acl

    print("[3/4] OM (acl) run → logits_om")
    outs = run_om(sample)
    logits_om = outs[0]
    print(f"  OM logits shape={list(logits_om.shape)}, top5={torch.topk(logits_om[0,0].float(),5).indices.tolist()}")

    print("[4/4] compare OM vs eager")
    diff = (logits_om.float() - out_eager.float()).abs().max().item()
    print(f"  OM vs eager logits maxdiff = {diff:.4e}  -> {'MATCH (OM 运行机制正确)' if diff < 1.0 else 'MISMATCH'}")

if __name__ == "__main__":
    main()
