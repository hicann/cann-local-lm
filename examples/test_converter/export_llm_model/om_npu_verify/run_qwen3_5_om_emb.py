#!/usr/bin/env python3
"""ACL python 运行 qwen3_5_emb.om (embeddings 入图) + 对比 HF (验证 OM 运行机制).

与 run_qwen3_5_om.py (input_ids 入图) 的区别:
- OM 图输入含 embeddings [1,8,1024], 无 input_ids (embed_tokens 查表外置到 host)
- host 用 CPU 模型 embed_tokens 查表算 embeddings 喂 OM (tie_word_embeddings,
  embed_tokens weight = lm_head weight; OM 图内 lm_head 保留, 查表权重 host 侧复用)
- 对比 HF (HF 用 input_ids, 数值等价: cann_lm embed_tokens weight 来自 HF)

流程: CPU 模型查表 embeddings → OM (acl NPU) run → 对比 HF logits + top5.

用法:
  source scripts/set_env_cann.sh --npu   # ③ NPU verify-runtime
  cd examples/test_converter/export_llm_model/om_npu_verify
  HF=<hf_model_dir> QWEN35_OM=<om_path> $DOPT_PYTHON run_qwen3_5_om_emb.py
"""
import os, sys
import torch

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
    if t.dtype == torch.int64: return t.numpy().tobytes()
    if t.dtype == torch.int32: return t.numpy().tobytes()
    return t.numpy().tobytes()
def bytes_to_tensor(b, shape, dtype):
    if dtype == torch.bfloat16:
        return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
    return torch.frombuffer(b, dtype=dtype).reshape(shape)


def flatten_to_om_inputs(sample, in_names):
    """展平 embeddings 入图 sample → OM 输入 (按 OM input name 顺序).

    sample: (None, mask, pos, past[6对], new_kv, embeddings, output_pos, None, None, delta_states)
    OM 图输入无 input_ids, 有 embeddings (pos 5).
    delta_states = (conv_flat[18], recurrent_flat[18]).
    """
    _input_ids, mask, pos, past_kv, new_kv, embeddings, output_pos, _ple, _emb_s, delta_states = sample
    conv_flat, rec_flat = delta_states
    kv_flat = []
    for k, v in past_kv:
        kv_flat.append(k); kv_flat.append(v)
    delta_flat = list(conv_flat) + list(rec_flat)
    by_name = {"attention_mask": mask, "position_ids": pos,
               "new_kv_cache_pos": new_kv, "output_pos": output_pos,
               "embeddings": embeddings}
    inputs = []
    for name in in_names:
        if name in by_name:
            inputs.append(by_name[name])
        elif name.startswith("past_key_values_"):
            parts = name.split("_"); L = int(parts[-2]); K = int(parts[-1])
            inputs.append(kv_flat[L * 2 + K])
        elif name.startswith("delta_states_"):
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
    has_ids = "input_ids" in in_names
    has_emb = "embeddings" in in_names
    print(f"  OM inputs ({ni}): input_ids={'Y' if has_ids else 'N'} embeddings={'Y' if has_emb else 'N'}")
    inputs = flatten_to_om_inputs(sample, in_names)
    no = acl.mdl.get_num_outputs(desc)
    out_names = [acl.mdl.get_output_name_by_index(desc, i) for i in range(no)]
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
    print("[1/4] Load qwen3.5-0.8B (CPU) + wrapper + 真实 input_ids")
    cfg = Qwen3_5TextConfig.from_pretrained(HF)
    model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_qwen3_5_weights(model, HF, device="cpu", dtype=torch.bfloat16)
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    print(f"  layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}, tie={cfg.tie_word_embeddings}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    real_ids = tok("Hello, my name is", return_tensors="pt")["input_ids"]
    if real_ids.shape[1] < SEQ:
        real_ids = torch.cat([real_ids, torch.zeros(1, SEQ - real_ids.shape[1], dtype=torch.int64)], dim=1)
    else:
        real_ids = real_ids[:, :SEQ]
    print(f"  input_ids={real_ids.tolist()}")

    print("[2/4] host 查表 embeddings = embed_tokens(input_ids) + 构造 use_embeddings sample")
    sample = wrapper.get_sample_inputs(seq_len=SEQ, use_embeddings=True)
    sample = list(sample)
    # 用真实 input_ids 查表 (wrapper.get_sample_inputs 用 random input_ids 算 embeddings, 此处替换)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(real_ids).detach()
    sample[0] = None          # input_ids 不入图 (None placeholder)
    sample[5] = embeddings     # embeddings 图输入 (host 查表结果)
    sample = tuple(sample)
    print(f"  embeddings shape={list(embeddings.shape)} dtype={embeddings.dtype}")

    print("[3/4] OM (acl NPU) run → logits_om")
    outs, in_names, out_names = run_om(sample)
    logits_om = outs[0]
    om_top5 = torch.topk(logits_om[0, 0].float(), 5).indices.tolist()
    print(f"  OM logits shape={list(logits_om.shape)}, top5={om_top5}")
    print(f"  OM logits[:5]={logits_om[0, 0, :5].float().tolist()}")
    print(f"  has_nan={torch.isnan(logits_om).any().item()}, has_inf={torch.isinf(logits_om).any().item()}")

    print("[4/4] compare OM vs HF transformers")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llm_models_run_on_cpu"))
    import hf_cpu  # 禁 torch_npu probe (shim)
    from transformers import Qwen3_5ForCausalLM as HFModel
    hf_model = HFModel.from_pretrained(HF, torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
    with torch.no_grad():
        out_hf = hf_model(input_ids=real_ids)
    logits_hf = out_hf.logits[:, -1:, :]
    diff = (logits_om.float() - logits_hf.float()).abs().max().item()
    hf_top5 = torch.topk(logits_hf[0, 0].float(), 5).indices.tolist()
    overlap = len(set(om_top5) & set(hf_top5))
    print(f"  HF logits shape={list(logits_hf.shape)}, top5={hf_top5}")
    print(f"  OM vs HF logits maxdiff = {diff:.4e}  top5_overlap={overlap}/5  "
          f"-> {'MATCH' if diff < 5.0 and overlap == 5 else 'MISMATCH'}")
    # delta 输出 nan 检查 (18 conv + 18 recurrent 交替)
    delta_out = outs[1:]
    n_delta = len(delta_out)
    delta_nan = any(torch.isnan(t).any().item() for t in delta_out)
    print(f"  delta outputs ({n_delta}): has_nan={delta_nan}")

    if overlap == 5 and not torch.isnan(logits_om).any().item():
        print(f"\n[PASS] embeddings 入图 OM 验证通过: OM top5 == HF top5 == {om_top5}")
    else:
        print(f"\n[FAIL] embeddings 入图 OM 验证失败")


if __name__ == "__main__":
    main()
