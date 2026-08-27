#!/usr/bin/env python3
"""ACL python 运行 prefill.om + decode.om 生成文本 (gemma4-E2B)。

prefill OM (seq=128) 处理 prompt → logits + 70 kv;
decode OM (seq=1) 逐 token 生成。
关键: 用 wrapper.get_sample_inputs 的 mask/kv 约定 (gemma4 shared KV 要求 pad 位
不被 all-mask,否则污染 store 层 KV → 共享层输出错误) + 覆盖 embeddings/PLE/output_pos。

用法:
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  cd examples/test_converter/export_llm_model/om_npu_verify
  /opt/buildtools/Python-3.11.4/bin/python3 run_gemma4_gen.py "The capital of China is"

OM 产物路径 (prefill.om + decode.om) 在本脚本 PREFILL_OM / DECODE_OM 常量里配置。
当前指向 /tmp/opencode/gemma4_pd/ (converter --mode prefill-decode 导出 + C++ 编译产物)。
"""
import os, sys, json, dataclasses
import torch
import torch.nn.functional as F

PREFILL_OM = os.environ.get("PREFILL_OM", "/tmp/opencode/gemma4_pd/prefill.om")
DECODE_OM = os.environ.get("DECODE_OM", "/tmp/opencode/gemma4_pd/decode.om")
E2B = "/mnt/workspace/data/models/gemma-4-E2B-it"
KV_MAX = 2048; DEV = 0; MAX_NEW = 30; PREFILL_SEQ = 128
sys.path.insert(0, "/mnt/workspace/project/CANN-LM_5883")

import torch_npu  # noqa (for wrapper eager if needed; mainly for mask helpers)
from safetensors import safe_open
from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from transformers import AutoTokenizer

DT = torch.bfloat16

# --- load model + wrapper (CPU, for mask helpers + kv structure) ---
print("[0] Load E2B model + wrapper (CPU)")
with open(f"{E2B}/config.json") as f: hf_cfg = json.load(f)
valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
cfg = Gemma4TextConfig(**{k:v for k,v in hf_cfg["text_config"].items() if k in valid}); cfg.__post_init__()
model = Gemma4ForCausalLM(cfg).to(dtype=DT).eval()
model = load_gemma4_weights(model, E2B, device="cpu", dtype=DT)
wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
HID = cfg.hidden_size; N_LAY = cfg.num_hidden_layers; PLE_H = cfg.hidden_size_per_layer_input
EMB_SCALE = HID ** 0.5; PLE_SCALE = PLE_H ** 0.5
print(f"  layers={N_LAY}, hidden={HID}")

# --- embed weights (for override) ---
with safe_open(f"{E2B}/model.safetensors", "pt") as sf:
    emb_w = sf.get_tensor("model.language_model.embed_tokens.weight")
    ple_w = sf.get_tensor("model.language_model.embed_tokens_per_layer.weight")
tok = AutoTokenizer.from_pretrained(E2B)

# --- ACL helpers ---
import acl
OK = 0; H2D = 1; D2H = 2
def ck(n, r):
    if r != OK: raise RuntimeError(f"{n} ret={r}")
def tb(t):
    t = t.cpu().contiguous()
    if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()
def b2t(b, shape):
    return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)

class OMRunner:
    def __init__(self, om_path):
        self.mid, ret = acl.mdl.load_from_file(om_path); ck("load", ret)
        self.desc = acl.mdl.create_desc(); ck("desc", acl.mdl.get_desc(self.desc, self.mid))
        self.ni = acl.mdl.get_num_inputs(self.desc); self.no = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(self.ni)]
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
            ck("memcpy", acl.rt.memcpy(self.in_bufs[i][0], self.in_bufs[i][1], acl.util.bytes_to_ptr(b), len(b), H2D))
        ck("execute", acl.mdl.execute(self.mid, self.in_ds, self.out_ds))
        outs = []
        for i, (buf, sz) in enumerate(self.out_bufs):
            hb, _ = acl.rt.malloc_host(sz)
            ck("memcpy out", acl.rt.memcpy(hb, sz, buf, sz, D2H))
            ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb)
            outs.append(b2t(ob, self.out_shapes[i]))
        return outs

def real_emb(input_ids):
    return (F.embedding(input_ids, emb_w) * EMB_SCALE).to(DT)
def real_ple(input_ids):
    return (F.embedding(input_ids, ple_w) * PLE_SCALE).reshape(*input_ids.shape, N_LAY, PLE_H).to(DT)

def build_mask_full(seq_len, kv_max, start_pos, batch=1):
    """bool 方阵 [batch,1,kv_max,kv_max], 绝对位 causal。False=attend, True=mask out。

    前 seq_len 行: query@绝对位 (start_pos+q) attend kv[0:start_pos+q+1]
    后 (kv_max-seq_len) 行: padding 全 True (mask out, 无 query, kernel 用
    actual_seq_qlen=seq_len 只取前 seq_len 行)。
    本地构造 (取代 wrapper._build_mask_full), 完全自控 mask 内容。
    """
    mask = torch.ones(batch, 1, kv_max, kv_max, dtype=torch.bool)
    q_abs = (start_pos + torch.arange(seq_len)).unsqueeze(1)   # [seq, 1] 绝对位
    k = torch.arange(kv_max).unsqueeze(0)                      # [1, kv_max]
    attend = k <= q_abs                                        # [seq, kv_max] True=attend
    mask[:, :, :seq_len, :] = ~attend                          # False=attend, True=mask out
    return mask

def build_mask_sliding(seq_len, kv_max, start_pos, sliding_window, batch=1):
    """bool 方阵 [batch,1,kv_max,kv_max], 绝对位 causal + sliding window。

    前 seq_len 行: query@绝对位 attend kv[max(0, abs-w+1):abs+1] (窗口内+causal)
    后 (kv_max-seq_len) 行: padding 全 True。
    kv_len <= sliding_window 时 causal 已隐含在窗口内, 退化为 mask_full。
    """
    w = sliding_window
    if not w or w <= 0 or kv_max <= w:
        return build_mask_full(seq_len, kv_max, start_pos, batch)
    mask = torch.ones(batch, 1, kv_max, kv_max, dtype=torch.bool)
    q_abs = (start_pos + torch.arange(seq_len)).unsqueeze(1)
    k = torch.arange(kv_max).unsqueeze(0)
    diff = q_abs - k
    attend = (diff >= 0) & (diff <= w - 1)
    mask[:, :, :seq_len, :] = ~attend
    return mask

def build_prefill(chunk_ids, chunk_real_len, start_pos, kv_prev):
    """用 wrapper mask + 覆盖。支持 chunked prefill (超长 prompt 分块)。

    Args:
        chunk_ids: [1, PREFILL_SEQ] padded chunk。
        chunk_real_len: 本块真实 token 数 (<= PREFILL_SEQ)。
        start_pos: 本块在 buffer 中的起始位置 (new_kv_cache_pos + position_ids 偏移)。
        kv_prev: 上一块的 70 kv 输出 (None → 用 wrapper 的零 buffer)。
    """
    s = list(wrapper.get_sample_inputs(seq_len=PREFILL_SEQ, use_embeddings=True))
    new_pos = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int64)
    s[1] = build_mask_full(PREFILL_SEQ, KV_MAX, start_pos)
    s[2] = build_mask_sliding(PREFILL_SEQ, KV_MAX, start_pos, wrapper._sliding_window)
    s[3] = new_pos.unsqueeze(0)                                              # position_ids [1,128]
    s[5] = new_pos                                                           # new_kv_cache_pos [128]
    s[6] = real_emb(chunk_ids)                                               # embeddings
    s[8] = real_ple(chunk_ids)                                               # PLE
    s[7] = torch.tensor([chunk_real_len - 1], dtype=torch.int32)             # output_pos (块内最后真实位)
    _, mf, ms, pos, _, npos, emb, opos, ple, _ = s
    if kv_prev is not None:
        kv = kv_prev
    else:
        _, _, _, _, pkv, _, _, _, _, _ = wrapper.get_sample_inputs(seq_len=PREFILL_SEQ, use_embeddings=True)
        kv = [t for pair in pkv for t in pair]
    by_name = {"mask_full": mf, "mask_sliding": ms, "position_ids": pos, "new_kv_cache_pos": npos,
               "embeddings": emb, "output_pos": opos, "per_layer_token_identity": ple}
    inputs = []
    for i, name in enumerate(runner_pf.in_names):
        if name in by_name: inputs.append(by_name[name])
        elif name.startswith("past_key_values"):
            parts = name.split("_"); L = int(parts[-2]); K = int(parts[-1])
            inputs.append(kv[L*2+K])
    return inputs

def build_decode(next_token, cached_len, kv_prev):
    """用 wrapper mask helpers (seq=1, new_kv_cache_pos=[cached_len]) + 覆盖。"""
    new_pos = torch.tensor([cached_len], dtype=torch.int64)
    mf = build_mask_full(1, KV_MAX, cached_len)
    ms = build_mask_sliding(1, KV_MAX, cached_len, wrapper._sliding_window)
    pos = torch.tensor([[cached_len]], dtype=torch.int64)
    opos = torch.tensor([0], dtype=torch.int32)
    emb = real_emb(next_token)       # [1,1,HID]
    ple = real_ple(next_token)       # [1,1,N_LAY,PLE_H]
    by_name = {"mask_full": mf, "mask_sliding": ms, "position_ids": pos, "new_kv_cache_pos": new_pos,
               "embeddings": emb, "output_pos": opos, "per_layer_token_identity": ple}
    inputs = []
    kv_idx = 0
    for name in runner_dc.in_names:
        if name in by_name: inputs.append(by_name[name])
        elif name.startswith("past_key_values"):
            inputs.append(kv_prev[kv_idx]); kv_idx += 1
    return inputs

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, my name is"
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_ids = tok(text, return_tensors="pt")["input_ids"][0]  # [real_len]
    real_len = len(all_ids)
    if real_len > KV_MAX:
        print(f"[WARN] prompt {real_len} > KV_MAX {KV_MAX}, truncating")
        all_ids = all_ids[:KV_MAX]; real_len = KV_MAX
    print(f"prompt: {prompt!r}  real_tokens={real_len}")

    print("\n[1/3] init ACL + load prefill.om + decode.om")
    r = acl.init()
    if r not in (0, 100002): ck("acl.init", r)
    ck("set_device", acl.rt.set_device(DEV))
    global runner_pf, runner_dc
    runner_pf = OMRunner(PREFILL_OM); runner_dc = OMRunner(DECODE_OM)
    print(f"  prefill: {runner_pf.ni} in / {runner_pf.no} out")
    print(f"  decode:  {runner_dc.ni} in / {runner_dc.no} out")

    # chunked prefill: 按 PREFILL_SEQ 切块, 逐块跑 prefill OM, KV 累积传递
    num_chunks = (real_len + PREFILL_SEQ - 1) // PREFILL_SEQ
    print(f"\n[2/3] prefill ({num_chunks} chunk(s), seq={PREFILL_SEQ})")
    kv_prev = None; logits = None
    for ci in range(num_chunks):
        start = ci * PREFILL_SEQ
        end = min(start + PREFILL_SEQ, real_len)
        clen = end - start
        chunk_ids = all_ids[start:end]
        if clen < PREFILL_SEQ:
            chunk_ids = torch.cat([chunk_ids, torch.full((PREFILL_SEQ - clen,), tok.pad_token_id, dtype=torch.long)])
        chunk_ids = chunk_ids.unsqueeze(0)  # [1, PREFILL_SEQ]
        pf_inputs = build_prefill(chunk_ids, clen, start, kv_prev)
        pf_outs = runner_pf.run(pf_inputs)
        logits = pf_outs[0]; kv_prev = pf_outs[1:]
        print(f"  chunk {ci}: tokens [{start}:{end}] (real={clen}) → ok")
    next_tok = logits[0, 0].argmax().item()
    tokens = [next_tok]
    print(f"  prefill → tok={next_tok} ({tok.decode([next_tok])!r})")
    print(f"  logits top5: {torch.topk(logits[0,0].float(), 5).indices.tolist()}")

    print(f"\n[3/3] decode loop ({MAX_NEW} steps)")
    cached_len = real_len
    for step in range(MAX_NEW):
        nt = torch.tensor([[next_tok]], dtype=torch.int64)
        dc_inputs = build_decode(nt, cached_len, kv_prev)
        dc_outs = runner_dc.run(dc_inputs)
        logits = dc_outs[0]; kv_prev = dc_outs[1:]
        cached_len += 1
        next_tok = logits[0, 0].argmax().item()
        tokens.append(next_tok)
        text = tok.decode(tokens, skip_special_tokens=True)
        print(f"  step {step+1}: tok={next_tok} ({tok.decode([next_tok])!r})  | text: {text!r}")
        if next_tok in (tok.eos_token_id, 1): print("  (EOS)"); break

    full = tok.decode(tokens, skip_special_tokens=True)
    print(f"\n{'='*60}\nPrompt: {prompt!r}\nGenerated: {full!r}\n{'='*60}")

if __name__ == "__main__":
    main()
