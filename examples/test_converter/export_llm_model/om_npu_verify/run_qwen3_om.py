#!/usr/bin/env python3
"""ACL python 运行 qwen3 OM + 对比 eager wrapper (验证 OM 运行机制)。

仿 run_gemma4_om.py, 适配 qwen3 (标准 GQA, 无 PLE, in-graph embed_tokens):
  - sample 7-tuple (input_ids, mask, pos, past_kv, new_kv, None, output_pos)
  - in-graph: OM 内含 embed_tokens, runtime 输入是 input_ids (非 embeddings)

流程: wrapper.get_sample_inputs(use_embeddings=False, CPU) → eager NPU 跑 →
      OM (acl) 用同一份 CPU sample 跑 → 对比 OM logits vs eager logits。

OM 产物路径用 QWEN3_OM 环境变量覆盖 (默认 /tmp/opencode/qwen3_calib/qwen3_noaub.om):
  QWEN3_OM=/path/to/qwen3.om python3 run_qwen3_om.py
"""
import os, sys, dataclasses
import torch

OM = os.environ.get("QWEN3_OM", "/tmp/opencode/qwen3_calib/qwen3_full.om")
HF = os.environ["HF"]
SEQ = 8; KV_MAX = 2048; DEV = 0

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper

# --- ACL helpers ---
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
def bytes_to_tensor(b, shape):
    return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)

def move_sample(sample, dev):
    res = []
    for t in sample:
        if isinstance(t, torch.Tensor): res.append(t.to(dev))
        elif isinstance(t, list): res.append([(k.to(dev), v.to(dev)) for k, v in t])
        else: res.append(t)
    return tuple(res)

def flatten_to_om_inputs(sample, in_names):
    """展平 qwen3 in-graph sample (7-tuple) → OM 输入 (按 OM input name 顺序)。

    sample: (input_ids, attention_mask, position_ids, past_kv[(k,v),...],
            new_kv_cache_pos, None, output_pos)
    """
    input_ids, mask, pos, past_kv, new_kv, _emb, output_pos = sample
    kv_flat = []
    for k, v in past_kv:
        kv_flat.append(k); kv_flat.append(v)
    by_name = {"input_ids": input_ids, "attention_mask": mask, "position_ids": pos,
               "new_kv_cache_pos": new_kv, "output_pos": output_pos}
    inputs = []
    for name in in_names:
        if name in by_name:
            inputs.append(by_name[name])
        elif name.startswith("past_key_values_"):
            parts = name.split("_"); L = int(parts[-2]); K = int(parts[-1])
            inputs.append(kv_flat[L * 2 + K])
        else:
            raise ValueError(f"unknown OM input {name}")
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
    print("[1/4] Load qwen3-0.6B (CPU) + wrapper + get_sample_inputs")
    cfg = Qwen3TextConfig.from_pretrained(HF)
    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_qwen3_weights(model, HF, device="cpu", dtype=torch.bfloat16)
    nl_env = os.environ.get("NUM_LAYERS")
    if nl_env:
        nl = int(nl_env)
        total = len(model.model.layers)
        if nl < total:
            model.model.layers = model.model.layers[:nl]
            print(f"  截断 model.model.layers {total} → {nl} (NUM_LAYERS env, 与 OM 对齐)")
    wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    sample = wrapper.get_sample_inputs(seq_len=SEQ, use_embeddings=False)  # in-graph: input_ids 入图
    print(f"  layers={len(model.model.layers)}, vocab={cfg.vocab_size}, sample len={len(sample)}")

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
