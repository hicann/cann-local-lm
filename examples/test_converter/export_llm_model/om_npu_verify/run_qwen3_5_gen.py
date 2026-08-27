#!/usr/bin/env python3
"""ACL python 运行 prefill.om + decode.om 生成文本 (qwen3.5-0.8B).

prefill OM (seq=8) 处理 prompt → logits + 12 KV(full_attn) + 36 delta;
decode OM (seq=1) 逐 token 生成, KV + delta 累积传递.
用法:
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  cd examples/test_converter/export_llm_model/om_npu_verify
  /opt/buildtools/Python-3.11.4/bin/python3 run_qwen3_5_gen.py "Hello, my name is"
"""
import os, sys, dataclasses, json
import torch

# input_ids 入图 OM (已清除, 需用 CLI `cann_lm_converter.py --no-use-embeddings --mode prefill-decode`
# 重新导出+编译 prefill/decode 两图). embeddings 入图单 OM 验证用 run_qwen3_5_om_emb.py.
PREFILL_OM = os.environ.get("PREFILL_OM", "/tmp/opencode/qwen3_5_pd/prefill.om")
DECODE_OM = os.environ.get("DECODE_OM", "/tmp/opencode/qwen3_5_pd/decode.om")
HF = "/mnt/workspace/data/models/Qwen3.5-0.8B"
KV_MAX = 2048; DEV = 0; MAX_NEW = 20; PREFILL_SEQ = 8
sys.path.insert(0, "/mnt/workspace/project/CANN-LM_5883")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from transformers import AutoTokenizer

import acl
OK = 0; H2D = 1; D2H = 2
def ck(n, r):
    if r != OK: raise RuntimeError(f"{n} ret={r}")
def tb(t):
    t = t.cpu().contiguous()
    if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()
def b2t(b, shape, dtype=torch.bfloat16):
    if dtype == torch.bfloat16: return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
    return torch.frombuffer(b, dtype=dtype).reshape(shape)


class OMRunner:
    def __init__(self, om_path):
        self.mid, ret = acl.mdl.load_from_file(om_path); ck("load", ret)
        self.desc = acl.mdl.create_desc(); ck("desc", acl.mdl.get_desc(self.desc, self.mid))
        self.ni = acl.mdl.get_num_inputs(self.desc); self.no = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(self.ni)]
        self.out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(self.no)]
        self.in_shapes = []
        for i in range(self.ni):
            d, _ = acl.mdl.get_input_dims(self.desc, i)
            self.in_shapes.append(tuple(d["dims"][:d["dimCount"]]))
        self.out_shapes = []
        for i in range(self.no):
            d, _ = acl.mdl.get_output_dims(self.desc, i)
            self.out_shapes.append(tuple(d["dims"][:d["dimCount"]]))
        self.in_ds = acl.mdl.create_dataset(); self.in_bufs = []
        for i in range(self.ni):
            sz = acl.mdl.get_input_size_by_index(self.desc, i)
            buf, r = acl.rt.malloc(sz, 2); ck("malloc in", r)
            acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(buf, sz))
            self.in_bufs.append((buf, sz))
        self.out_ds = acl.mdl.create_dataset(); self.out_bufs = []
        for i in range(self.no):
            sz = acl.mdl.get_output_size_by_index(self.desc, i)
            buf, r = acl.rt.malloc(sz, 2); ck("malloc out", r)
            acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(buf, sz))
            self.out_bufs.append((buf, sz))
    def run(self, inputs):
        for i, t in enumerate(inputs):
            b = tb(t)
            if len(b) > self.in_bufs[i][1]:
                print(f"  [ERR] input {i} ({self.in_names[i]}): src={len(b)} > dst={self.in_bufs[i][1]}, shape={list(t.shape)} dtype={t.dtype}")
                ck("memcpy", 107000)
            ck("memcpy", acl.rt.memcpy(self.in_bufs[i][0], self.in_bufs[i][1], acl.util.bytes_to_ptr(b), len(b), H2D))
        ck("execute", acl.mdl.execute(self.mid, self.in_ds, self.out_ds))
        outs = []
        for i, (buf, sz) in enumerate(self.out_bufs):
            hb, _ = acl.rt.malloc_host(sz)
            ck("memcpy out", acl.rt.memcpy(hb, sz, buf, sz, D2H))
            ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb)
            outs.append(b2t(ob, self.out_shapes[i]))
        return outs


def flatten_prefill(sample, in_names):
    """sample: (input_ids, mask, pos, past[6对], new_kv, None, output_pos, None, None, delta_states)."""
    input_ids, mask, pos, past_kv, new_kv, _, output_pos, _, _, delta_states = sample
    conv_flat, rec_flat = delta_states
    kv_flat = []
    for k, v in past_kv:
        kv_flat.append(k); kv_flat.append(v)
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
            parts = name.split("_"); A = int(parts[-2]); B = int(parts[-1])
            n_lin = len(delta_flat) // 2
            inputs.append(delta_flat[A * n_lin + B])
        else:
            raise ValueError(f"unknown OM input {name}")
    return inputs


def build_prefill(chunk_ids, real_len, start_pos, kv_prev, delta_prev, wrapper, in_names):
    """chunked prefill: 构造 chunk 的 mask/pos/new_kv + KV/delta 累积传递.

    Args:
        chunk_ids: [1, PREFILL_SEQ] padded chunk (pad=0).
        real_len: 本块真实 token 数 (<= PREFILL_SEQ).
        start_pos: 本块在 buffer 的起始位置.
        kv_prev: 上一块 6 对 KV (None → 零 buffer).
        delta_prev: 上一块 36 delta flat (conv+rec 分组, None → 零).
    """
    dtype = torch.bfloat16
    # mask minv: 用 -1e4 而非 finfo.min (NPU SoftmaxV2 对 bf16 min -3.4e38 + 多 attend 位置溢出 nan)
    minv = torch.tensor(-1e4, dtype=dtype)
    # mask: causal over buffer, q_pos = start_pos..start_pos+seq-1; pad 位不 attend
    q_pos = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(KV_MAX, dtype=dtype).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), minv)
    if real_len < PREFILL_SEQ:
        mask[real_len:, :] = minv
    mask = mask.unsqueeze(0).unsqueeze(0).contiguous()
    pos = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int64).unsqueeze(0)
    new_kv = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int64)
    out_pos = torch.tensor([real_len - 1], dtype=torch.int32)
    # past KV: kv_prev (累积) 或零 buffer (首次)
    if kv_prev is not None:
        past = kv_prev
    else:
        full_idx = wrapper._get_full_attn_layer_indices()
        nk = wrapper.config.num_key_value_heads
        past = [(torch.zeros(1, nk, KV_MAX, wrapper._get_layer_head_dim(i), dtype=dtype),
                 torch.zeros(1, nk, KV_MAX, wrapper._get_layer_head_dim(i), dtype=dtype))
                for i in full_idx]
    # delta: delta_prev (累积) 或零 (首次)
    if delta_prev is not None:
        n_lin = len(delta_prev) // 2
        conv_flat = list(delta_prev[:n_lin])
        rec_flat = list(delta_prev[n_lin:])
    else:
        delta = wrapper._build_zero_delta_states()
        conv_flat, rec_flat = list(delta[0]), list(delta[1])
    sample = (chunk_ids, mask, pos, past, new_kv, None, out_pos, None, None, (conv_flat, rec_flat))
    return flatten_prefill(sample, in_names)


def flatten_decode(input_ids, mask, pos, kv_pairs, new_kv, output_pos, delta_flat, in_names):
    """decode: past 6对(full_attn), delta 36 flat."""
    kv_flat = []
    for k, v in kv_pairs:
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
        elif name.startswith("delta_states_"):
            parts = name.split("_"); A = int(parts[-2]); B = int(parts[-1])
            n_lin = len(delta_flat) // 2
            inputs.append(delta_flat[A * n_lin + B])
        else:
            raise ValueError(f"unknown OM input {name}")
    return inputs


def reconstruct_kv(outs):
    """prefill/decode OM 输出: [logits, 12 KV, 36 delta(交替 c0,r0,c1,r1,...)]. → 6 对 KV + 36 delta flat(conv+rec)."""
    logits = outs[0]
    kv_flat = outs[1:13]  # 12 = 6 对
    kv_pairs = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, 12, 2)]
    delta_out = list(outs[13:49])  # 36, 交替 [c0,r0,c1,r1,...]
    conv_flat = delta_out[0::2]   # 18 conv
    rec_flat = delta_out[1::2]     # 18 rec
    delta_flat = conv_flat + rec_flat  # conv+rec 顺序 (匹配 OM input delta_states_A_B: A=0 conv, A=1 rec)
    return logits, kv_pairs, delta_flat


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, my name is"
    tok = AutoTokenizer.from_pretrained(HF)
    all_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    real_len = len(all_ids)
    if real_len > KV_MAX:
        print(f"[WARN] prompt {real_len} > KV_MAX {KV_MAX}, truncating")
        all_ids = all_ids[:KV_MAX]; real_len = KV_MAX
    print(f"prompt: {prompt!r}  real_tokens={real_len}  ids={all_ids.tolist()}")

    print("[0] Load model + wrapper (CPU, for sample_inputs helpers)")
    cfg = Qwen3_5TextConfig.from_pretrained(HF)
    model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_qwen3_5_weights(model, HF, device="cpu", dtype=torch.bfloat16)
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    wrapper._delta_cache_box[0] = None

    print("\n[1/3] init ACL + load prefill.om + decode.om")
    r = acl.init()
    if r not in (0, 100002): ck("acl.init", r)
    ck("set_device", acl.rt.set_device(DEV))
    runner_pf = OMRunner(PREFILL_OM); runner_dc = OMRunner(DECODE_OM)
    print(f"  prefill: {runner_pf.ni} in / {runner_pf.no} out")
    print(f"  decode:  {runner_dc.ni} in / {runner_dc.no} out")

    # chunked prefill: 按 PREFILL_SEQ 切块, 逐块跑 prefill OM, KV + delta 累积传递
    num_chunks = (real_len + PREFILL_SEQ - 1) // PREFILL_SEQ
    print(f"\n[2/3] prefill ({num_chunks} chunk(s), seq={PREFILL_SEQ})")
    kv_prev = None; delta_prev = None; logits = None
    for ci in range(num_chunks):
        start = ci * PREFILL_SEQ
        end = min(start + PREFILL_SEQ, real_len)
        clen = end - start
        chunk_ids = all_ids[start:end]
        if clen < PREFILL_SEQ:
            chunk_ids = torch.cat([chunk_ids, torch.zeros(PREFILL_SEQ - clen, dtype=torch.long)])
        chunk_ids = chunk_ids.unsqueeze(0)
        pf_inputs = build_prefill(chunk_ids, clen, start, kv_prev, delta_prev, wrapper, runner_pf.in_names)
        pf_outs = runner_pf.run(pf_inputs)
        logits, kv_pairs, delta_flat = reconstruct_kv(pf_outs)
        kv_prev = kv_pairs; delta_prev = delta_flat
        print(f"  chunk {ci}: tokens [{start}:{end}] (real={clen}) → ok")
    next_tok = logits[0, 0].argmax().item()
    tokens = [next_tok]
    print(f"  prefill → tok={next_tok} ({tok.decode([next_tok])!r})")
    print(f"  logits top5: {torch.topk(logits[0,0].float(), 5).indices.tolist()}")
    print(f"  has_nan: {torch.isnan(logits).any().item()}")

    print(f"\n[3/3] decode loop ({MAX_NEW} steps)")
    cached_len = real_len
    dtype = torch.bfloat16
    minv = torch.finfo(dtype).min
    for step in range(MAX_NEW):
        nt = torch.tensor([[next_tok]], dtype=torch.int64)
        # decode mask: cached_len 位 attend, 之后 mask
        mask = torch.zeros(1, 1, 1, KV_MAX, dtype=dtype)
        if cached_len + 1 < KV_MAX:
            mask[:, :, :, cached_len + 1:] = minv
        pos = torch.tensor([[cached_len]], dtype=torch.int64)
        new_kv = torch.tensor([cached_len], dtype=torch.int64)
        out_pos = torch.tensor([0], dtype=torch.int32)
        dc_inputs = flatten_decode(nt, mask, pos, kv_pairs, new_kv, out_pos, delta_flat, runner_dc.in_names)
        dc_outs = runner_dc.run(dc_inputs)
        logits, kv_pairs, delta_flat = reconstruct_kv(dc_outs)
        cached_len += 1
        next_tok = logits[0, 0].argmax().item()
        tokens.append(next_tok)
        text = tok.decode(tokens, skip_special_tokens=True)
        print(f"  step {step+1}: tok={next_tok} ({tok.decode([next_tok])!r})  | text: {text!r}")
        if next_tok in (tok.eos_token_id or 0, 0): print("  (EOS/pad)"); break

    full = tok.decode(tokens, skip_special_tokens=True)
    print(f"\n{'='*60}\nPrompt: {prompt!r}\nGenerated: {full!r}\n{'='*60}")


if __name__ == "__main__":
    main()
