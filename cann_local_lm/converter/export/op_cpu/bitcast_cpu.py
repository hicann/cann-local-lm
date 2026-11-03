"""BitcastCpu: int8 ND → int4 ND via HostCpu constant folding.

Input:  [N, K/2] DT_INT8  (nibble-packed, each byte = 2 int4)
Output: [N, K]   DT_INT4  (same bytes, dtype reinterpretation)

Bitcast 不改变字节布局，只改变数据类型解读。常量折叠在编译期调用 execute
完成 memcpy，运行期该节点被折叠为 Const。
"""

import ctypes

from ge.custom_op import (
    OpBackend,
    get_execute_ctx,
    register_kernel,
    register_op,
    register_op_impl,
)
from ge.graph.types import DataType
from ge.runtime import StorageShape, Tensor, TensorDesc


@register_op(op_type="Bitcast")
def bitcast_cpu_infer_meta(x: TensorDesc, *, type: DataType) -> TensorDesc:
    """[N, K/2] int8 → [N, K] int4: 最后一维翻倍 (1 int8 byte = 2 int4 elements)。"""

    dims = x.shape.origin_shape.dims
    out_dims = list(dims)
    out_dims[-1] = out_dims[-1] * 2
    return TensorDesc(StorageShape(out_dims, out_dims), type)


@register_op_impl(op_type="Bitcast")
class BitcastCpuHostKernel:
    """HostCpu kernel: 纯 memcpy，字节布局不变。"""

    @register_kernel(backend=OpBackend.HOST)
    def execute(self, x: Tensor, *, type: DataType) -> None:
        print(
            "[Python] HostCpu execute for Bitcast: "
            f"input shape={x.origin_shape.dims} dtype={x.data_type} "
            f"-> output dtype={type}",
            flush=True,
        )

        dims = x.origin_shape.dims
        out_dims = list(dims)
        out_dims[-1] = out_dims[-1] * 2
        out_shape = StorageShape(out_dims, out_dims)

        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, out_shape, x.format, type)

        # Bitcast: same bytes, just different dtype interpretation
        byte_count = int(x.size)
        ctypes.memmove(int(output.addr), int(x.addr), byte_count)
        print(f"[Python] Bitcast done: copied {byte_count} bytes", flush=True)
