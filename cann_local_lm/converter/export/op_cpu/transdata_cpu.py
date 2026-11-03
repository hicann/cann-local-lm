"""TransDataCpu: int4 ND → int4 FRACTAL_Z via HostCpu constant folding.

Input:  [N, K]              DT_INT4,  ND
        (逻辑: N 行 K 个 int4; 物理: N 行 K/2 个 int8 nibble-packed 字节)

Output: [K/64, N/16, 16, 64] DT_INT4,  FRACTAL_Z
        storage_shape = origin_shape = 4D [k1, n1, 16, 64]

FRACTAL_Z 分形布局:
  - K 方向 block = 64 int4 elements = 32 int8 bytes
  - N 方向 block = 16
  - C0 = 64 (最后一维, int4 element count per K block)
  - 物理排列: [k1, n1, 16, 64] int4 = [k1, n1, 16, 32] int8 bytes

排列逻辑:
  输入物理 [N, K/2] int8 bytes (ND 行优先)
  → reshape [n1, 16, k1, 32]
  → permute(2, 0, 1, 3)
  → [k1, n1, 16, 32] int8 bytes = [k1, n1, 16, 64] int4

关键: storage_shape 和 origin_shape 都设为 4D [k1, n1, 16, 64]。
  GE ConstInfer 会用 storage_shape 覆盖 origin_shape, 然后从 origin 重算 FRACTAL_Z。
  如果 storage=origin=[k1,n1,16,64], 重算后仍是 [k1,n1,16,64] (4D → 4D, 不变)。
  TBE InferShape 需要 4D shape, 所以 origin_shape 必须是 4D。

常量折叠在编译期调用 execute 完成字节重排，运行期该节点被折叠为 Const。
"""

import ctypes

import numpy as np

from ge.custom_op import (
    OpBackend,
    get_execute_ctx,
    register_kernel,
    register_op,
    register_op_impl,
)
from ge.graph.types import DataType, Format
from ge.runtime import ExpandDimsType, StorageFormat, StorageShape, Tensor, TensorDesc

_B_FRAC_K_INT4 = 64  # K direction fractal block (int4 count)
_B_FRAC_N = 16  # N direction fractal block
_INT4_PER_INT8 = 2  # 2 int4 per int8 nibble
_C0_BYTES = _B_FRAC_K_INT4 // _INT4_PER_INT8  # 32 int8 bytes per K block


def _ceil_div(a, b):
    return (a + b - 1) // b


def _compute_fractal_shape(n, k):
    """[N, K] int4 → FRACTAL_Z [k1, n1, 16, 64] int4。"""
    k1 = _ceil_div(k, _B_FRAC_K_INT4)  # ceil(K / 64)
    n1 = _ceil_div(n, _B_FRAC_N)  # ceil(N / 16)
    return [k1, n1, _B_FRAC_N, _B_FRAC_K_INT4]


@register_op(op_type="TransData")
def transdata_cpu_infer_meta(
    src: TensorDesc,
    *,
    src_format: str,
    dst_format: str,
) -> TensorDesc:
    """ND [N, K] int4 → FRACTAL_Z [k1, n1, 16, 64] int4。

    storage_shape = origin_shape = 4D [k1, n1, 16, 64]。
    GE ConstInfer 用 storage_shape 覆盖 origin_shape, 4D→4D 不变。
    TBE InferShape 要求 4D, 所以 origin 必须是 4D。
    """

    dims = src.shape.origin_shape.dims
    n = dims[0]
    k = dims[1]
    out_dims = _compute_fractal_shape(n, k)
    return TensorDesc(StorageShape(out_dims, out_dims), src.data_type)


@register_op_impl(op_type="TransData")
class TransDataCpuHostKernel:
    """HostCpu kernel: ND → FRACTAL_Z 字节重排。"""

    @register_kernel(backend=OpBackend.HOST)
    def execute(self, src: Tensor, *, src_format: str, dst_format: str) -> None:
        # 逻辑 shape [N, K] int4, 物理 [N, K/2] int8 bytes
        dims = src.origin_shape.dims
        n = dims[0]
        k = dims[1]
        k_half = k // _INT4_PER_INT8  # 物理 int8 字节数 per row

        out_dims = _compute_fractal_shape(n, k)
        k1, n1 = out_dims[0], out_dims[1]

        print(
            "[Python] HostCpu execute for TransData: "
            f"input ND [{n}, {k}] int4 (physical [{n}, {k_half}] bytes) "
            f"-> FRACTAL_Z {out_dims} int4",
            flush=True,
        )

        # storage_shape = origin_shape = 4D [k1, n1, 16, 64]
        out_shape = StorageShape(out_dims, out_dims)
        out_format = StorageFormat(
            Format.FORMAT_FRACTAL_Z, Format.FORMAT_FRACTAL_Z, ExpandDimsType()
        )
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, out_shape, out_format, src.data_type)

        # 读取输入: [N, K/2] int8 bytes
        src_buf = (ctypes.c_char * int(src.size)).from_address(int(src.addr))
        src_arr = np.frombuffer(src_buf, dtype=np.int8).reshape(n, k_half)

        # FRACTAL_Z 字节重排:
        # [N, K/2] → reshape [n1, 16, k1, 32] → permute(2,0,1,3) → [k1, n1, 16, 32]
        if n % _B_FRAC_N == 0 and k % _B_FRAC_K_INT4 == 0:
            # 精确整除, 无需 padding
            src_reshaped = src_arr.reshape(n1, _B_FRAC_N, k1, _C0_BYTES)
            dst_arr = np.ascontiguousarray(src_reshaped.transpose(2, 0, 1, 3))
        else:
            # 需要 padding (零填充到 block 边界)
            n_padded = n1 * _B_FRAC_N
            k_half_padded = k1 * _C0_BYTES
            padded = np.zeros((n_padded, k_half_padded), dtype=np.int8)
            padded[:n, :k_half] = src_arr
            src_reshaped = padded.reshape(n1, _B_FRAC_N, k1, _C0_BYTES)
            dst_arr = np.ascontiguousarray(src_reshaped.transpose(2, 0, 1, 3))

        # 写入输出 buffer
        dst_bytes = dst_arr.nbytes
        ctypes.memmove(int(output.addr), dst_arr.ctypes.data, dst_bytes)
        print(f"[Python] TransData done: {dst_bytes} bytes rearranged", flush=True)


# ---------------------------------------------------------------------------
# IR 输出名对齐补丁: output0 -> dst
#
# 本算子经 ASCEND_CUSTOM_OPP_PATH 注册进 GE 后, canonical IR 输出名是
# proto.py `_build_outputs` 硬编码的 "output0"。但编译期 pass
# (qbm_x2_transdata_pass.py) 用 ge.es.math.TransData 生成的 replacement
# 节点 IR 输出名是 native 命名 "dst"。libgraph RecoverIrDefinitions 对
# outputs 做严格逐名比对 (非 subset), 两侧必须一致才能通过校验。
#
# HostCpu execute 按 output index 取 tensor (malloc_output_tensor(0, ...)),
# 与名字无关; 本补丁只改注册表里的 canonical IR 名, 不影响 kernel 分发。
# ---------------------------------------------------------------------------
from ge.custom_op.proto import get_registered_op_proto_by_op_type  # noqa: E402

_td_descriptor = get_registered_op_proto_by_op_type("TransData")
if (
    _td_descriptor is not None
    and _td_descriptor.outputs
    and _td_descriptor.outputs[0].name != "dst"
):
    import dataclasses  # noqa: E402

    _new_outputs = tuple(
        dataclasses.replace(item, name=("dst" if item.index == 0 else item.name))
        for item in _td_descriptor.outputs
    )
    object.__setattr__(_td_descriptor, "outputs", _new_outputs)
