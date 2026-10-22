#!/usr/bin/env python3
"""单层 gemma4 OM 跑输入输出 (构造 dummy 输入, ACL execute, 打印输出数据)。

不加载完整模型, 不对比 eager; 仅验证单层 OM 可执行 + 输出 shape/数据合理。
"""
import os, sys, argparse, json
import ctypes
from ctypes import c_char, c_uint32, c_int64, c_int32, c_void_p, CFUNCTYPE, Structure, addressof, string_at
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


# ACL dump 回调: acldumpChunk (fileName + bufLen + dataBuf 柔性数组), 回调内写文件
class _AcldumpChunk(Structure):
    _fields_ = [("fileName", c_char * 4096), ("bufLen", c_uint32), ("isLastChunk", c_uint32),
                ("offset", c_int64), ("flag", c_int32)]
_DumpCb = CFUNCTYPE(c_int32, c_void_p, c_int32)
_dump_count = [0]
def _dump_cb(p, flag):
    _dump_count[0] += 1
    try:
        c = ctypes.cast(p, ctypes.POINTER(_AcldumpChunk)).contents
        if c.bufLen > 0:
            data = string_at(addressof(c) + _AcldumpChunk.flag.offset + 4, c.bufLen)
            fn = c.fileName.decode().rstrip("\x00")
            d = os.path.dirname(fn)
            if d: os.makedirs(d, exist_ok=True)
            with open(fn, "ab") as f: f.write(data)
    except Exception as e: print("  dump cb err:", e)
    return 0
_dump_cb_ref = _DumpCb(_dump_cb)
import torch
import torch_npu  # noqa (NPU env, 提供匹配的 CANN GE lib, 避免 import segfault)
import acl

DEFAULT_OM = "/mnt/workspace/CANN-LM_5883/examples/test_converter/export_llm_model/output/e2e_gemma4_1layer/prefill/prefill.om"
DEFAULT_INPUTS_PT = "/mnt/workspace/CANN-LM_5883/examples/test_converter/export_llm_model/output/e2e_gemma4_1layer/inputs.pt"
DEV = 0
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


def make_dummy_inputs():
    """构造单层 prefill OM 的 9 输入 (zeros/arange, 合理 dtype)。"""
    bf = torch.bfloat16
    torch.manual_seed(0)
    return {
        "embeddings": torch.randn(1, 128, 1536, dtype=bf) * 0.1,           # 随机 (非零, 看实际计算)
        "mask_full": torch.zeros(1, 1, 128, 2048, dtype=torch.bool),      # False=attend (dummy 全 attend)
        "mask_sliding": torch.zeros(1, 1, 128, 2048, dtype=torch.bool),
        "position_ids": torch.arange(128, dtype=torch.int32).unsqueeze(0),  # [1,128]
        "past_key_values_0_0": torch.zeros(1, 1, 2048, 256, dtype=bf),
        "past_key_values_0_1": torch.zeros(1, 1, 2048, 256, dtype=bf),
        "new_kv_cache_pos": torch.arange(128, dtype=torch.int32),          # [128]
        "output_pos": torch.tensor([127], dtype=torch.int32),              # [1]
        "per_layer_token_identity": torch.randn(1, 128, 35, 256, dtype=bf) * 0.1,
    }


def load_bin_inputs(bin_dir):
    """从 golden 目录读 .bin 输入 (golden_meta.json + input_*.bin) 构造 by_name dict。

    golden 的 bf16 被无损上转 float32 保存, 这里还原为 bf16 (OM 期望 bf16);
    bool/int32 不变。past_key_values 的 sub_index 0/1 映射为 past_key_values_0_0/_0_1
    (对齐 OM input_names, 与 inputs.pt 的 key 命名一致)。
    """
    with open(os.path.join(bin_dir, "golden_meta.json")) as f:
        meta = json.load(f)
    by_name = {}
    for gi in meta["inputs"]:
        name = gi["name"]; sub = gi.get("sub_index", 0)
        key = f"past_key_values_0_{sub}" if name == "past_key_values" else name
        with open(os.path.join(bin_dir, gi["file"]), "rb") as f:
            raw = f.read()
        if gi["dtype"] == "float32":
            t = torch.frombuffer(raw, dtype=torch.float32).reshape(gi["shape"]).to(torch.bfloat16).clone()
        elif gi["dtype"] == "float16":
            t = torch.frombuffer(raw, dtype=torch.float16).reshape(gi["shape"]).clone()
        elif gi["dtype"] == "bool":
            t = torch.frombuffer(raw, dtype=torch.bool).reshape(gi["shape"]).clone()
        else:
            t = torch.frombuffer(raw, dtype=torch.int32).reshape(gi["shape"]).clone()
        by_name[key] = t
    return by_name


def run_om(om_path, by_name, runs=1, dump_dir=None):
    """单次 load OM → N 次 execute → 收集 N 次输出; dump_dir 非空时 dump 每个算子输出到 dump_dir (aclmdlInitDump/SetDump/FinalizeDump)。"""
    acl_init_tolerant(); check("set_device", acl.rt.set_device(DEV))
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        check("init_dump", acl.mdl.init_dump())
        cfg_path = os.path.join(dump_dir, "dump.json")
        with open(cfg_path, "w") as f:
            json.dump({"dump": {"dump_list": [{"model_name": "delegate_graph"}], "dump_path": dump_dir, "dump_mode": "output", "dump_op_switch": "off", "dump_data": "tensor"}}, f)
        check("set_dump", acl.mdl.set_dump(cfg_path))
        check("dump_reg_callback", acl.mdl.dump_reg_callback(_dump_cb_ref, 0))
        _dump_count[0] = 0
    mid, ret = acl.mdl.load_from_file(om_path); check("load", ret)
    desc = acl.mdl.create_desc(); check("get_desc", acl.mdl.get_desc(desc, mid))
    ni = acl.mdl.get_num_inputs(desc)
    in_names = [acl.mdl.get_input_name_by_index(desc, i) for i in range(ni)]
    no = acl.mdl.get_num_outputs(desc)
    print(f"  OM: {ni} inputs, {no} outputs")
    print(f"  in_names: {in_names}")
    inputs = [by_name[n] for n in in_names]
    in_ds = acl.mdl.create_dataset(); in_bufs = []
    for t in inputs:
        b = t_bytes(t); sz = len(b); buf, ret = acl.rt.malloc(sz, 2); check("malloc", ret)
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(buf, sz)); in_bufs.append((buf, sz, b))
    out_ds = acl.mdl.create_dataset();     out_meta = []
    for i in range(no):
        sz = acl.mdl.get_output_size_by_index(desc, i); buf, ret = acl.rt.malloc(sz, 2); check("malloc out", ret)
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(buf, sz))
        dims, _ = acl.mdl.get_output_dims(desc, i)
        out_dt = acl.mdl.get_output_data_type(desc, i)
        out_meta.append((buf, sz, tuple(dims["dims"][:dims["dimCount"]]), out_dt))
    all_outs = []
    for r in range(runs):
        for buf, sz, b in in_bufs:
            check("memcpy in", acl.rt.memcpy(buf, sz, acl.util.bytes_to_ptr(b), len(b), H2D))
        print(f"  executing OM (run {r+1}/{runs})..."); check("execute", acl.mdl.execute(mid, in_ds, out_ds))
        outs = []
        for buf, sz, shape, out_dt in out_meta:
            hb, _ = acl.rt.malloc_host(sz); check("memcpy out", acl.rt.memcpy(hb, sz, buf, sz, D2H))
            ob = acl.util.ptr_to_bytes(hb, sz); acl.rt.free_host(hb)
            if out_dt == 1:  # DT_FLOAT16
                t = torch.frombuffer(ob, dtype=torch.float16).reshape(shape)
            else:  # DT_BF16 or other int16-viewable
                t = torch.frombuffer(ob, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
            outs.append(t)
        all_outs.append(outs)
    for buf, _, _ in in_bufs: acl.rt.free(buf)
    for buf, _, _, _ in out_meta: acl.rt.free(buf)
    if dump_dir:
        acl.mdl.dump_unreg_callback()  # void 返回, 不 check
        check("finalize_dump", acl.mdl.finalize_dump())
        print(f"  dump callback 被调 {_dump_count[0]} 次 → {dump_dir}")
    acl.mdl.unload(mid); acl.rt.reset_device(DEV); acl.finalize()
    return all_outs


def compare_golden(outs, golden_dir):
    """OM 输出 (bf16) vs golden 输出 (fp32, 源自 bf16 无损上转) 逐个比对。

    bf16 容差: OM 是编译融合后 bf16, golden 是 CPU eager bf16, 二者同精度量级;
    用 maxdiff (绝对) + 相对误差 (mean_abs_diff / golden mean_abs) 判定, <1% 算 PASS。
    """
    meta_path = os.path.join(golden_dir, "golden_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    g_outs = meta["outputs"]
    print(f"\n=== Golden 比对 ({golden_dir}) ===")
    print(f"  golden: {meta['model_type']} layers={meta['num_layers']} dtype={meta['dtype']} seed={meta.get('seed', 'N/A')}")
    n = min(len(outs), len(g_outs))
    if len(outs) != len(g_outs):
        print(f"  [!] 输出数不一致: OM {len(outs)} vs golden {len(g_outs)}, 只比对前 {n} 个")
    all_pass = True
    for i in range(n):
        g = g_outs[i]
        gpath = os.path.join(golden_dir, g["file"])
        with open(gpath, "rb") as fb:
            g_dtype = g.get("dtype", "float32")
            if g_dtype == "float16":
                gt = torch.frombuffer(fb.read(), dtype=torch.float16).reshape(g["shape"]).float()
            else:
                gt = torch.frombuffer(fb.read(), dtype=torch.float32).reshape(g["shape"])
        om = outs[i].float()
        if om.numel() != gt.numel():
            print(f"  out[{i}] numel mismatch: OM {om.numel()} vs golden {gt.numel()} (shape {list(om.shape)} vs {list(gt.shape)}) SKIP")
            all_pass = False
            continue
        if om.shape != gt.shape:
            om = om.reshape(gt.shape)
        maxdiff = (om - gt).abs().max().item()
        meandiff = (om - gt).abs().mean().item()
        gabs = gt.abs().mean().item()
        rel = meandiff / (gabs + 1e-12)
        ok = rel < 0.01
        all_pass = all_pass and ok
        print(f"  out[{i}] shape={list(om.shape)}: maxdiff={maxdiff:.4e} meandiff={meandiff:.4e} "
              f"golden|mean|={gabs:.4e} rel={rel:.4e} {'PASS' if ok else 'CHECK'}")
    print(f"  => {'ALL PASS' if all_pass else '有差异需人工确认'}")
    return all_pass


def save_om_out(outs, out_dir):
    """保存 OM 输出为 float32 .bin + meta (同 golden 格式, 便于离线比对)。"""
    os.makedirs(out_dir, exist_ok=True)
    meta = {"outputs": []}
    for i, t in enumerate(outs):
        tf = t.float().cpu().contiguous()
        fname = f"output_{i}.bin"
        tf.numpy().tofile(os.path.join(out_dir, fname))
        meta["outputs"].append({"file": fname, "shape": list(tf.shape), "dtype": "float32", "index": i})
    with open(os.path.join(out_dir, "om_out_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  OM 输出已保存 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="单层 gemma4 OM 跑输入输出")
    ap.add_argument("--om", default=os.environ.get("OM", DEFAULT_OM), help="OM 路径")
    ap.add_argument("--load", default=None, help="加载固定输入 .pt (不构造 dummy)")
    ap.add_argument("--load-bin", default=None, help="从 golden 目录读 .bin 输入 (golden_meta.json + input_*.bin; bf16 还原)")
    ap.add_argument("--save", default=None, help="保存当前输入到 .pt (固定复现)")
    ap.add_argument("--save-bin", default=None, help="保存为 C++ 可读 .bin (每 tensor 一个 .bin + inputs_meta.json)")
    ap.add_argument("--save-bin-fp16", default=None, help="把 bf16 转 fp16 保存 .bin (int32/bool 不变; 注意 bf16>65504 溢出)")
    ap.add_argument("--runs", type=int, default=1, help="跑 N 次 execute 验证输出一致性")
    ap.add_argument("--dump", default=None, help="dump 每个算子输出到目录 (acl.mdl.init_dump/set_dump)")
    ap.add_argument("--golden", default=None, help="golden 数据目录 (含 golden_meta.json + output_*.bin); 跑完 OM 自动比对")
    ap.add_argument("--save-out", default=None, help="保存 OM 输出到目录 (float32 .bin + meta, 同 golden 格式)")
    args = ap.parse_args()
    print(f"OM: {args.om}")
    if args.load_bin:
        print(f"[1/2] 加载 golden .bin 输入: {args.load_bin}")
        by_name = load_bin_inputs(args.load_bin)
    elif args.load:
        print(f"[1/2] 加载固定输入: {args.load}")
        by_name = torch.load(args.load, weights_only=False)
    else:
        print("[1/2] 构造单层 dummy 输入 (seed=0, 固定可复现)")
        by_name = make_dummy_inputs()
    for k, v in by_name.items():
        print(f"  {k:30s} shape={list(v.shape)} dtype={v.dtype}")
    if args.save:
        torch.save(by_name, args.save)
        print(f"  → 已保存固定输入: {args.save}")
    if args.save_bin:
        os.makedirs(args.save_bin, exist_ok=True)
        meta = {}
        for k, t in by_name.items():
            b = t_bytes(t)
            fname = f"{k}.bin"
            with open(os.path.join(args.save_bin, fname), "wb") as f:
                f.write(b)
            meta[k] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                        "nbytes": len(b), "file": fname}
        with open(os.path.join(args.save_bin, "inputs_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  → 已保存 C++ 可读 .bin + meta: {args.save_bin}")
    if args.save_bin_fp16:
        os.makedirs(args.save_bin_fp16, exist_ok=True)
        meta = {}
        for k, t in by_name.items():
            if t.dtype == torch.bfloat16:
                t2 = t.to(torch.float16)
                if torch.isinf(t2).any() or torch.isnan(t2).any():
                    print(f"  [!] {k}: bf16→fp16 溢出/NaN (源 max={t.float().abs().max().item():.2f})")
            else:
                t2 = t  # int32/bool 不转
            b = t_bytes(t2)
            fname = f"{k}.bin"
            with open(os.path.join(args.save_bin_fp16, fname), "wb") as f:
                f.write(b)
            meta[k] = {"shape": list(t2.shape), "dtype": str(t2.dtype).replace("torch.", ""),
                        "nbytes": len(b), "file": fname, "source_dtype": str(t.dtype).replace("torch.", "")}
        with open(os.path.join(args.save_bin_fp16, "inputs_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  → 已保存 fp16 .bin + meta: {args.save_bin_fp16}")
    print(f"[2/2] ACL load + execute 单层 OM (runs={args.runs})")
    all_outs = run_om(args.om, by_name, runs=args.runs, dump_dir=args.dump)
    outs = all_outs[0]
    print(f"\n输出 {len(outs)} 个 (run 1):")
    for i, o in enumerate(outs):
        of = o.float()
        print(f"  out[{i}] shape={list(o.shape)} dtype={o.dtype}")
        print(f"    前5元素: {o.flatten()[:5].tolist()}")
        print(f"    均值={of.mean().item():.4e} 最大绝对值={of.abs().max().item():.4e} 是否全0={bool((o == 0).all().item())}")
    if args.runs > 1:
        print(f"\n=== {args.runs} 次输出一致性对比 (vs run 1) ===")
        for i in range(len(outs)):
            base = outs[i].float()
            max_d = max((base - all_outs[r][i].float()).abs().max().item() for r in range(1, args.runs))
            print(f"  out[{i}] shape={list(outs[i].shape)}: {args.runs}次间 maxdiff={max_d:.2e} 完全一致={max_d==0}")
    if args.save_out:
        print("\n=== 保存 OM 输出 ===")
        save_om_out(outs, args.save_out)
    if args.golden:
        compare_golden(outs, args.golden)


if __name__ == "__main__":
    main()
