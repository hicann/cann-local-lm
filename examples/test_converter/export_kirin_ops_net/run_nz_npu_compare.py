#!/usr/bin/env python3
"""NZ vs ND OM NPU 输出对比脚本(CLI-015 L2 验证用).

加载两个 OM(NZ 权重版 + ND 权重版),用相同输入跑,对比输出 max_diff。
期望:max_diff=0.0(NZ OM 输出 == ND OM 输出)。

用法(③ NPU env):
  source /home/developer/Ascend/cann/set_env.sh
  python3 run_nz_npu_compare.py \\
      --nz-om out/matmul_single_nz.om \\
      --nd-om out/matmul_single_nd.om \\
      --input-shape 1 128 \\
      --input-dtype float16 \\
      --input-value ones

参数:
  --nz-om PATH         NZ 权重 OM 路径(必填)
  --nd-om PATH         ND 权重 OM 路径(必填, 对照)
  --input-shape D1 D2  输入 shape(空格分隔, 默认 "1 128")
  --input-dtype TYPE   输入 dtype: float16/bfloat16/float32(默认 float16)
  --input-value VAL    输入填充: ones/zeros/random(默认 ones)
  --atol TOL           allclose 容差(默认 1e-3)
  --device ID          NPU device id(默认 0)
"""
import argparse
import sys

import numpy as np

try:
    import acl
except ImportError:
    print("ERROR: acl not found. Run: source /home/developer/Ascend/cann/set_env.sh")
    sys.exit(1)

H2D = 1  # aclrtMemcpyHostToDevice
D2H = 2  # aclrtMemcpyDeviceToDeviceToHost

DTYPE_MAP = {
    "float16": (np.float16, 1),   # aclDataType ACL_FLOAT16 = 1
    "bfloat16": (np.uint16, 9),    # ACL_BF16 = 9 (stored as uint16 bytes)
    "float32": (np.float32, 0),   # ACL_FLOAT = 0
}

VALUE_MAP = {
    "ones": lambda shape, dtype: np.ones(shape, dtype=dtype),
    "zeros": lambda shape, dtype: np.zeros(shape, dtype=dtype),
    "random": lambda shape, dtype: np.random.randn(*shape).astype(dtype),
}


def parse_args():
    p = argparse.ArgumentParser(description="NZ vs ND OM NPU output comparison (CLI-015 L2)")
    p.add_argument("--nz-om", required=True, help="NZ weight OM path")
    p.add_argument("--nd-om", required=True, help="ND weight OM path (baseline)")
    p.add_argument("--input-shape", nargs="+", type=int, default=[1, 128], help="Input shape")
    p.add_argument("--input-dtype", default="float16", choices=list(DTYPE_MAP), help="Input dtype")
    p.add_argument("--input-value", default="ones", choices=list(VALUE_MAP), help="Input fill")
    p.add_argument("--atol", type=float, default=1e-3, help="allclose atol")
    p.add_argument("--device", type=int, default=0, help="NPU device id")
    return p.parse_args()


def init_acl(device):
    r = acl.init()
    if r not in (0, 100002):
        raise RuntimeError(f"acl.init failed: {r}")
    r = acl.rt.set_device(device)
    if r != 0:
        raise RuntimeError(f"acl.rt.set_device({device}) failed: {r}")
    return acl.rt.create_stream()[0]


def run_om(om_path, input_bytes, stream):
    mid, ret = acl.mdl.load_from_file(om_path)
    if ret != 0:
        raise RuntimeError(f"acl.mdl.load_from_file({om_path}) failed: {ret}")
    desc = acl.mdl.create_desc()
    acl.mdl.get_desc(desc, mid)
    n_in = acl.mdl.get_num_inputs(desc)
    n_out = acl.mdl.get_num_outputs(desc)
    in_sizes = [acl.mdl.get_input_size_by_index(desc, i) for i in range(n_in)]
    out_sizes = [acl.mdl.get_output_size_by_index(desc, i) for i in range(n_out)]

    in_ds = acl.mdl.create_dataset()
    in_bufs = []
    for i, sz in enumerate(in_sizes):
        buf, ret = acl.rt.malloc(sz, 2)
        if ret != 0:
            raise RuntimeError(f"malloc input {i} failed: {ret}")
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(buf, sz))
        if i == 0:
            copy_sz = min(len(input_bytes), sz)
            acl.rt.memcpy(buf, sz, acl.util.bytes_to_ptr(input_bytes), copy_sz, H2D)
        in_bufs.append(buf)

    out_ds = acl.mdl.create_dataset()
    out_bufs = []
    for sz in out_sizes:
        buf, ret = acl.rt.malloc(sz, 2)
        if ret != 0:
            raise RuntimeError(f"malloc output failed: {ret}")
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(buf, sz))
        out_bufs.append(buf)

    ret = acl.mdl.execute(mid, in_ds, out_ds)
    if ret != 0:
        raise RuntimeError(f"acl.mdl.execute failed: {ret}")

    outputs = []
    for buf, sz in zip(out_bufs, out_sizes):
        hb, _ = acl.rt.malloc_host(sz)
        acl.rt.memcpy(hb, sz, buf, sz, D2H)
        ob = acl.util.ptr_to_bytes(hb, sz)
        acl.rt.free_host(hb)
        outputs.append(ob)

    for buf in in_bufs + out_bufs:
        acl.rt.free(buf)
    acl.mdl.unload(mid)
    return outputs, n_in, n_out


def main():
    args = parse_args()
    np_dtype, _acl_dtype = DTYPE_MAP[args.input_dtype]
    input_arr = VALUE_MAP[args.input_value](args.input_shape, np_dtype)
    input_bytes = input_arr.tobytes()

    print(f"Input: shape={args.input_shape}, dtype={args.input_dtype}, value={args.input_value}")
    print(f"NZ OM: {args.nz_om}")
    print(f"ND OM: {args.nd_om}")

    stream = init_acl(args.device)
    print(f"Device {args.device} ready")

    try:
        print(f"\n[1/2] Running NZ OM...")
        nz_outputs, nz_n_in, nz_n_out = run_om(args.nz_om, input_bytes, stream)
        print(f"  inputs={nz_n_in}, outputs={nz_n_out}")

        print(f"[2/2] Running ND OM...")
        nd_outputs, nd_n_in, nd_n_out = run_om(args.nd_om, input_bytes, stream)
        print(f"  inputs={nd_n_in}, outputs={nd_n_out}")

        if nz_n_out != nd_n_out:
            print(f"\n[FAIL] Output count mismatch: NZ={nz_n_out} vs ND={nd_n_out}")
            sys.exit(1)

        all_pass = True
        for i, (nz_out, nd_out) in enumerate(zip(nz_outputs, nd_outputs)):
            if len(nz_out) != len(nd_out):
                print(f"  Output[{i}] size mismatch: NZ={len(nz_out)}B vs ND={len(nd_out)}B")
                all_pass = False
                continue
            nz_arr = np.frombuffer(nz_out, dtype=np_dtype)
            nd_arr = np.frombuffer(nd_out, dtype=np_dtype)
            diff = np.max(np.abs(nz_arr.astype(np.float32) - nd_arr.astype(np.float32)))
            ok = np.allclose(nz_arr, nd_arr, atol=args.atol)
            status = "PASS" if ok else "FAIL"
            print(f"  Output[{i}]: shape={nz_arr.shape}, max_diff={diff:.6e}, [{status}]")
            if not ok:
                print(f"    NZ first5={nz_arr[:5]}")
                print(f"    ND first5={nd_arr[:5]}")
                all_pass = False

        if all_pass:
            print(f"\n[PASS] All outputs match (max_diff=0 expected). NZ OM == ND OM.")
        else:
            print(f"\n[FAIL] Output mismatch. See details above.")
            sys.exit(1)

    finally:
        acl.rt.reset_device(args.device)
        acl.finalize()


if __name__ == "__main__":
    main()
