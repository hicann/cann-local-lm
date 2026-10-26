"""算子级 NPU 验证 — Step 3: 加载 OM → NPU 跑 input_act → 存 npu_output.pt (NPU env)。

env: 系统python + torch_npu + acl + CANN (source scripts/cann_lm_compile/set_env_cann.sh --npu, 或 set_env.sh)。
用 desc 路径 (get_desc_from_file) 避免 load_from_file 后 get_num_inputs segfault (AGENTS 已知坑)。
torch_npu 提供 GE lib 避免 import segfault。

输入: 从 golden op_io 取 input_act (dopt fakequant 抓的真实激活, fp16)。
输出: OM QuantLinear 输出 (fp16) → 存 npu_output.pt, 供 verify_op.py 余弦对比。
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
import torch_npu  # noqa: F401  # 提供 GE lib 避免 import segfault
import acl

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))
from ref_w4a8 import INT4_NEG_CLIP  # noqa: F401  # 确认 reference 一致性 (未直接用)

ACL_OK = 0
DEV = int(os.environ.get("DEVICE_ID", "0"))


def _check(n, r):
    if r != ACL_OK:
        raise RuntimeError(f"{n} ret={r}")


def load_input_act(op_name, golden_dir, io_index):
    safe = op_name.replace(".", "_")
    data = torch.load(Path(golden_dir) / "op_io" / f"{safe}.pt", weights_only=True)
    return data["inputs"][io_index].to(torch.float16).contiguous()


def run_om(om_path, x):
    """单输入单输出 OM: load → desc → execute → output tensor (fp16)。"""
    _check("acl.init", acl.init() if True else 0)  # tolerant
    try:
        acl.init()
    except RuntimeError:
        pass
    _check("set_device", acl.rt.set_device(DEV))
    # desc 路径 (避免 load_from_file+get_num_inputs segfault)
    desc = acl.mdl.create_desc()
    _check("get_desc_from_file", acl.mdl.get_desc_from_file(desc, str(om_path)))
    ni = acl.mdl.get_num_inputs(desc)
    no = acl.mdl.get_num_outputs(desc)
    in_name = acl.mdl.get_input_name_by_index(desc, 0)
    out_name = acl.mdl.get_output_name_by_index(desc, 0)
    in_size = acl.mdl.get_input_size_by_index(desc, 0)
    out_size = acl.mdl.get_output_size_by_index(desc, 0)
    print(
        f"[run_npu] OM inputs={ni} ({in_name} {in_size}B) outputs={no} ({out_name} {out_size}B)",
        flush=True,
    )

    mid, ret = acl.mdl.load_from_file(str(om_path))
    _check("load_from_file", ret)

    in_ds = acl.mdl.create_dataset()
    x_bytes = (
        x.contiguous()
        .view(torch.int16 if x.dtype in (torch.float16, torch.bfloat16) else torch.int8)
        .numpy()
        .tobytes()
        if x.dtype != torch.float32
        else x.numpy().tobytes()
    )
    in_buf, ret = acl.mdl.create_dataset_buffer(x_bytes, len(x_bytes))
    _check("create_input_buffer", ret)
    _, ret = acl.mdl.add_dataset_buffer(in_ds, in_buf)
    _check("add_input_buffer", ret)

    out_ds = acl.mdl.create_dataset()
    out_buf, ret = acl.mdl.create_dataset_buffer(out_size, out_size)
    _check("create_output_buffer", ret)
    _, ret = acl.mdl.add_dataset_buffer(out_ds, out_buf)
    _check("add_output_buffer", ret)

    _check("execute", acl.mdl.execute(mid, in_ds, out_ds))

    out_bytes = acl.mdl.get_dataset_buffer(out_ds, 0)[0]
    # 简化: 用 acl.util.numpy_to_ptr 反向; 实际用 ctypes 读 buffer
    import ctypes

    buf_ptr = ctypes.cast(out_bytes, ctypes.POINTER(ctypes.c_uint8))
    out_arr = ctypes.string_at(buf_ptr, out_size)
    out_dtype = torch.float16  # QuantLinear output_dtype=5 (fp16)
    out_t = torch.frombuffer(bytearray(out_arr), dtype=out_dtype)
    # 输出 shape 由 desc 推 (简化: reshape 到 (x.shape[0], x.shape[1], -1))
    out_t = out_t.reshape(x.shape[0], x.shape[1], -1)
    return out_t


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--om", required=True)
    ap.add_argument("--op-name", required=True)
    ap.add_argument("--golden-dir", default=str(BENCH_DIR / "golden"))
    ap.add_argument("--io-index", type=int, default=0)
    ap.add_argument(
        "--output", default=str(Path(__file__).resolve().parent / "npu_output.pt")
    )
    args = ap.parse_args()

    x = load_input_act(args.op_name, args.golden_dir, args.io_index)
    print(
        f"[run_npu] input {tuple(x.shape)} {x.dtype} from golden io[{args.io_index}]",
        flush=True,
    )
    out = run_om(args.om, x)
    torch.save(out, args.output)
    print(f"[run_npu] NPU output {tuple(out.shape)} saved: {args.output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
