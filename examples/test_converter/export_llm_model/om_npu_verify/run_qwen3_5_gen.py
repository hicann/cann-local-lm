#!/usr/bin/env python3
"""ACL python 运行 prefill.om + decode.om 生成文本 (qwen3.5-0.8B, use_embeddings 入图).

OM 图输入含 embeddings [1,seq,hidden], 无 input_ids (embed_tokens 查表外置到 host);
host 用 cann_lm 模型 embed_tokens 查表算 embeddings 喂 OM (每 prefill chunk + 每 decode token);
KV cache 6 对 + 36 delta_states 累积传递 + mask/pos 由 runner 构造.

支持单轮/多轮 (命令行 prompt 数决定): 1 个 prompt = 单轮; N 个 prompt = 多轮会话
(每轮 tok.apply_chat_template 包累积 messages, 完整 prefill chat 历史, KV 每轮重置).

用法:
  source scripts/set_env_cann.sh --npu   # ③ NPU verify-runtime
  cd examples/test_converter/export_llm_model/om_npu_verify
  HF=<hf_model_dir> $DOPT_PYTHON run_qwen3_5_gen.py "Hello, my name is"           # 单轮
  HF=<hf_model_dir> $DOPT_PYTHON run_qwen3_5_gen.py "Hello, my name is Alice." "What is my name?"  # 多轮

OM 产物 (prefill.om + decode.om, use-embeddings, converter --mode prefill-decode 默认 use-emb)
路径由 PREFILL_OM / DECODE_OM env 指定.
"""
import os, sys
import torch
import torch_npu  # NPU env: 让 torch_npu 的 npu:: 算子先注册, 避免 cann_lm fused_ops stub 先注册后 torch_npu 重复定义崩溃

PREFILL_OM = os.environ.get("PREFILL_OM", "/tmp/opencode/qwen3_5_pd/prefill.om")
DECODE_OM = os.environ.get("DECODE_OM", "/tmp/opencode/qwen3_5_pd/decode.om")
HF = os.environ["HF"]
KV_MAX = 2048; DEV = 0; MAX_NEW = 20; PREFILL_SEQ = int(os.environ.get("PREFILL_SEQ", 8))
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
def b2t(b, shape):
    return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(shape)


class OMRunner:
    def __init__(self, om_path):
        self.mid, ret = acl.mdl.load_from_file(om_path); ck("load", ret)
        self.desc = acl.mdl.create_desc(); ck("desc", acl.mdl.get_desc(self.desc, self.mid))
        self.ni = acl.mdl.get_num_inputs(self.desc); self.no = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(self.ni)]
        self.in_shapes = []; self.out_shapes = []
        for i in range(self.ni):
            d, _ = acl.mdl.get_input_dims(self.desc, i); self.in_shapes.append(tuple(d["dims"][:d["dimCount"]]))
        for i in range(self.no):
            d, _ = acl.mdl.get_output_dims(self.desc, i); self.out_shapes.append(tuple(d["dims"][:d["dimCount"]]))
        self.in_ds = acl.mdl.create_dataset(); self.in_bufs = []
        for i in range(self.ni):
            sz = acl.mdl.get_input_size_by_index(self.desc, i); buf, r = acl.rt.malloc(sz, 2); ck("malloc in", r)
            acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(buf, sz)); self.in_bufs.append((buf, sz))
        self.out_ds = acl.mdl.create_dataset(); self.out_bufs = []
        for i in range(self.no):
            sz = acl.mdl.get_output_size_by_index(self.desc, i); buf, r = acl.rt.malloc(sz, 2); ck("malloc out", r)
            acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(buf, sz)); self.out_bufs.append((buf, sz))
    def run(self, inputs):
        for i, t in enumerate(inputs):
            b = tb(t); ck("memcpy", acl.rt.memcpy(self.in_bufs[i][0], self.in_bufs[i][1], acl.util.bytes_to_ptr(b), len(b), H2D))
        ck("execute", acl.mdl.execute(self.mid, self.in_ds, self.out_ds))
        outs = []
        for i, (buf, sz) in enumerate(self.out_bufs):
            hb, _ = acl.rt.malloc_host(sz); ck("memcpy out", acl.rt.memcpy(hb, sz, buf, sz, D2H))
            ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb); outs.append(b2t(ob, self.out_shapes[i]))
        return outs


def real_emb(model, input_ids):
    """host 查表: embeddings = embed_tokens(input_ids). (qwen3.5 embed_tokens 内部含 scale, 无 PLE)"""
    with torch.no_grad():
        return model.model.embed_tokens(input_ids).detach()


def flatten_prefill(sample, in_names):
    """use-emb sample: (None, mask, pos, past[6对], new_kv, embeddings, output_pos, None, None, delta_states).
    OM 图输入无 input_ids, 有 embeddings (pos 5)."""
    _input_ids, mask, pos, past_kv, new_kv, embeddings, output_pos, _ple, _emb_s, delta_states = sample
    conv_flat, rec_flat = delta_states
    kv_flat = []
    for k, v in past_kv:
        kv_flat.append(k); kv_flat.append(v)
    delta_flat = list(conv_flat) + list(rec_flat)
    by_name = {"embeddings": embeddings, "attention_mask": mask, "position_ids": pos,
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


def build_prefill(model, chunk_ids, real_len, start_pos, kv_prev, delta_prev, wrapper, in_names):
    dtype = torch.bfloat16
    minv = torch.tensor(-1e4, dtype=dtype)
    q_pos = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int32).unsqueeze(1)
    k_pos = torch.arange(KV_MAX, dtype=torch.int32).unsqueeze(0)
    mask = k_pos <= q_pos  # BOOL [seq,kv_max], True=attend (causal, 对齐 58b5d1f BOOL mask)
    if real_len < PREFILL_SEQ:
        mask[real_len:, :] = False  # pad rows: mask-out
    mask = (~mask).unsqueeze(0).unsqueeze(0).contiguous()  # BOOL [1,1,seq,kv_max], True=mask-out
    pos = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int32).unsqueeze(0)
    new_kv = torch.arange(start_pos, start_pos + PREFILL_SEQ, dtype=torch.int32)
    out_pos = torch.tensor([real_len - 1], dtype=torch.int32)
    if kv_prev is not None:
        past = kv_prev
    else:
        full_idx = wrapper._get_full_attn_layer_indices()
        nk = wrapper.config.num_key_value_heads
        past = [(torch.zeros(1, nk, KV_MAX, wrapper._get_layer_head_dim(i), dtype=dtype),
                 torch.zeros(1, nk, KV_MAX, wrapper._get_layer_head_dim(i), dtype=dtype))
                for i in full_idx]
    if delta_prev is not None:
        n_lin = len(delta_prev) // 2
        conv_flat = list(delta_prev[:n_lin]); rec_flat = list(delta_prev[n_lin:])
    else:
        delta = wrapper._build_zero_delta_states()
        conv_flat, rec_flat = list(delta[0]), list(delta[1])
    embeddings = real_emb(model, chunk_ids)   # host 查表 [1, PREFILL_SEQ, hidden]
    sample = (None, mask, pos, past, new_kv, embeddings, out_pos, None, None, (conv_flat, rec_flat))
    return flatten_prefill(sample, in_names)


def flatten_decode(embeddings, mask, pos, kv_pairs, new_kv, output_pos, delta_flat, in_names):
    kv_flat = []
    for k, v in kv_pairs:
        kv_flat.append(k); kv_flat.append(v)
    by_name = {"embeddings": embeddings, "attention_mask": mask, "position_ids": pos,
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
    logits = outs[0]
    kv_flat = outs[1:13]
    kv_pairs = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, 12, 2)]
    delta_out = list(outs[13:49])
    conv_flat = delta_out[0::2]; rec_flat = delta_out[1::2]
    delta_flat = conv_flat + rec_flat
    return logits, kv_pairs, delta_flat


def run_turn(tok, model, wrapper, runner_pf, runner_dc, messages, prompt):
    """单轮: prompt 包 chat 追加 messages, 完整 prefill chat 历史 + decode. 返回 assistant 文本.

    每轮用 tok.apply_chat_template 包裹累积 messages (含历史 user/assistant), 完整 prefill
    该轮 chat 历史 (每轮 KV 重置), decode 生成 assistant. chat template 保证对话格式正确
    (裸 prompt 会退化成续写).
    """
    messages.append({"role": "user", "content": prompt})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_ids = tok(text, return_tensors="pt")["input_ids"][0]  # chat template 包裹后 tokenize
    real_len = len(all_ids)
    if real_len > KV_MAX:
        print(f"[WARN] chat history {real_len} > KV_MAX {KV_MAX}, truncating")
        all_ids = all_ids[:KV_MAX]; real_len = KV_MAX

    num_chunks = (real_len + PREFILL_SEQ - 1) // PREFILL_SEQ
    print(f"  prefill: {real_len} tokens (chat template), {num_chunks} chunk(s), seq={PREFILL_SEQ}")
    kv_prev = None; delta_prev = None; logits = None
    for ci in range(num_chunks):
        start = ci * PREFILL_SEQ
        end = min(start + PREFILL_SEQ, real_len)
        clen = end - start
        chunk_ids = all_ids[start:end]
        if clen < PREFILL_SEQ:
            chunk_ids = torch.cat([chunk_ids, torch.zeros(PREFILL_SEQ - clen, dtype=torch.long)])
        chunk_ids = chunk_ids.unsqueeze(0)
        pf_inputs = build_prefill(model, chunk_ids, clen, start, kv_prev, delta_prev, wrapper, runner_pf.in_names)
        pf_outs = runner_pf.run(pf_inputs)
        logits, kv_pairs, delta_flat = reconstruct_kv(pf_outs)
        kv_prev = kv_pairs; delta_prev = delta_flat
        print(f"    chunk {ci}: tokens [{start}:{end}] (real={clen}) → ok")
    next_tok = logits[0, 0].argmax().item()
    print(f"  prefill → tok={next_tok} ({tok.decode([next_tok])!r})  has_nan={torch.isnan(logits).any().item()}")
    cached_len = real_len
    tokens = [next_tok]
    print(f"  decode ({MAX_NEW} steps, cached_len={cached_len}):")
    for step in range(MAX_NEW):
        nt = torch.tensor([[next_tok]], dtype=torch.int64)
        emb = real_emb(model, nt)   # host 查表 [1,1,hidden]
        mask = torch.zeros(1, 1, 1, KV_MAX, dtype=torch.bool)  # BOOL, False=attend
        if cached_len + 1 < KV_MAX:
            mask[:, :, :, cached_len + 1:] = True  # mask-out k>cached_len
        pos = torch.tensor([[cached_len]], dtype=torch.int32)
        new_kv = torch.tensor([cached_len], dtype=torch.int32)
        out_pos = torch.tensor([0], dtype=torch.int32)
        dc_inputs = flatten_decode(emb, mask, pos, kv_pairs, new_kv, out_pos, delta_flat, runner_dc.in_names)
        dc_outs = runner_dc.run(dc_inputs)
        logits, kv_pairs, delta_flat = reconstruct_kv(dc_outs)
        cached_len += 1
        next_tok = logits[0, 0].argmax().item()
        tokens.append(next_tok)
        print(f"    step {step+1}: tok={next_tok} ({tok.decode([next_tok])!r})")
        if next_tok in (tok.eos_token_id or 0, 0): print("    (EOS/pad)"); break
    text_out = tok.decode(tokens, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": text_out})
    return text_out


def main():
    prompts = sys.argv[1:] or ["Hello, my name is"]
    tok = AutoTokenizer.from_pretrained(HF)
    print("[0] Load qwen3.5-0.8B (CPU) + wrapper (embed_tokens 查表 + sample_inputs helpers)")
    cfg = Qwen3_5TextConfig.from_pretrained(HF)
    model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model = load_qwen3_5_weights(model, HF, device="cpu", dtype=torch.bfloat16)
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=KV_MAX, batch=1).eval()
    wrapper._delta_cache_box[0] = None

    print("\n[1] init ACL + load prefill.om + decode.om (use-emb)")
    r = acl.init()
    if r not in (0, 100002): ck("acl.init", r)
    ck("set_device", acl.rt.set_device(DEV))
    runner_pf = OMRunner(PREFILL_OM); runner_dc = OMRunner(DECODE_OM)
    print(f"  prefill: {runner_pf.ni} in / {runner_pf.no} out  (embeddings: {'embeddings' in runner_pf.in_names})")
    print(f"  decode:  {runner_dc.ni} in / {runner_dc.no} out")

    messages = []
    for turn, prompt in enumerate(prompts):
        print(f"\n{'='*60}\n[Turn {turn+1}/{len(prompts)}] user: {prompt!r}\n{'='*60}")
        text = run_turn(tok, model, wrapper, runner_pf, runner_dc, messages, prompt)
        print(f"\n[Turn {turn+1}] assistant: {text!r}")
    print(f"\n{'='*60}\n完成 ({len(prompts)} 轮, chat template)\n{'='*60}")


if __name__ == "__main__":
    main()
