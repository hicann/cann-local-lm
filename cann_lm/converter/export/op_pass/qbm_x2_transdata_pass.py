"""qbm_x2_transdata_pass: BeforeInferShape 阶段为 QuantBatchMatmulV4 的 x2 插 TransData。

背景 (noinferformat 流程, CANN 9.3.0 KirinX90):
  - converter 新路线: 权重 [N,K/2] int8 --Bitcast--> [N,K] int4 ND 直接进 QBMV4
    (不预排 FRACTAL_Z, 不在导出侧插 TransData 节点)
  - installed ops-nn 的 V4 infershape (InferShapeFracZ) 无条件要求 x2 是
    FRACTAL_Z 4D [K/64, N/16, 16, 64], x2 ND 2D 直接编译失败:
    "FRACTAL_Z infershape requires x2 4D shape ..., got dim num 2"
  - SecondInferformat 开关都不帮忙: 它只传播已有格式, 不创建 TransData 节点

机制 (用户指出的环境变量 = ASCEND_GE_PY_PASS_PATH):
  GE GeGenerator::Initialize -> ge::fusion::LoadPassPlugins() 读该环境变量
  (bootstrap.py: load_plugins_from_env(ENV_PY_PASS_PATH)),
  经 pybind 桥调 ge.passes._bridge.load_and_get_pass_descriptors() 收集本目录
  所有 .py 顶层 register 的 pass, 注册进 GE PassRegistry 与内置 fusion pass
  同管线执行 (FailurePassExecutor::InitPassesIfNeed).

本 pass (DecomposePass, op_types=[QuantBatchMatmulV4], stage=BeforeInferShape):
  meet_requirements: x2 = DT_INT4 且 format != FRACTAL_Z 且 shape 是 2D [N,K]
  replacement: 用 ge.es plugin (math.TransData / nn.QuantBatchMatmulV4) 重建子图
    x1 --|                       (x1 直传)
    x2 --TransData(ND->FRACTAL_Z)-+--QuantBatchMatmulV4--> y
    bias/x1_scale/x2_scale 等直传, IR 属性从原节点拷贝
  TransData 的 HostCpu kernel 已由 op_cpu/transdata_cpu.py 注册
  (ASCEND_CUSTOM_OPP_PATH), ConstantFolding 会在编译期执行 ND->FRACTAL_Z
  字节重排并折叠成 Const — 最终图形态与 v10 已验证路线完全一致。

验收: 编译日志出现
  "Python pass[QbmX2TransDataPass] is registered from pybind bridge"
  "[Python] HostCpu execute for TransData ..." + PASS x3 + op_model.omc 生成
"""

from ge.es import GraphBuilder, TensorHolder
from ge.es import get_plugin
from ge.graph import Graph
from ge.graph.types import DataType, Format
from ge.passes import DecomposePass, PassStage
from ge.passes import register_decompose_pass

_math = get_plugin("math")
_nn = get_plugin("nn")


def _frac_z_dims(n: int, k: int):
    """[N, K] int4 -> FRACTAL_Z 元素视角 4D [k1, n1, 16, 64] (K 块 64, N 块 16)。"""
    return [(k + 63) // 64, (n + 15) // 16, 16, 64]


def _resolve_nk(x1_desc, x2_desc):
    """返回 (N, K)。x2 唯一合法口径 = int4 元素视角 [N, K] (Bitcast 输出,
    nibble 原生 [N, K//2] 末维 ×2 而来; swap.py 已去掉 transpose, 不再产
    [K//2, N] 混轴视图)。以 x1 K 锚定: dim1 == x1_K 才接受, 其余 None。"""
    k_ref = int(list(x1_desc.shape)[-1])
    d0, d1 = int(x2_desc.shape[0]), int(x2_desc.shape[1])
    if d1 == k_ref:  # [N, K] int4 元素视角
        return d0, d1, "nk"
    return None


def _get_node_attr(node, key, default=None):
    try:
        val = node.get_attr(key)
        if val is not None:
            return val
    except Exception:
        pass
    return default


@register_decompose_pass(
    name="QbmX2TransDataPass",
    stage=PassStage.BEFORE_INFER_SHAPE,
    op_types=["QuantBatchMatmulV4"],
)
class QbmX2TransDataPass(DecomposePass):
    """QBMV4 x2 int4 ND -> int4 FRACTAL_Z (在 infershape 前补插 TransData)。"""

    def meet_requirements(self, node, context=None) -> bool:
        try:
            x2_desc = node.get_input_desc(1)
            x1_desc = node.get_input_desc(0)
        except Exception:
            return False
        if x2_desc is None or x1_desc is None:
            return False
        if x2_desc.data_type != DataType.DT_INT4:
            return False
        if x2_desc.format == Format.FORMAT_FRACTAL_Z:
            return False
        dims = list(x2_desc.shape)
        if len(dims) != 2:
            return False  # 已是 4D/未知 shape, 不动
        return _resolve_nk(x1_desc, x2_desc) is not None

    def replacement(self, node, context=None) -> Graph:
        x2_desc = node.get_input_desc(1)
        x1_desc = node.get_input_desc(0)
        n_dim, k_dim, _kind = _resolve_nk(
            x1_desc, x2_desc
        )  # 单一 [N,K] 口径, 经 meet 已必真

        op_count = node.get_inputs_size()
        builder = GraphBuilder("qbm_x2_transdata_replacement")
        inputs = builder.create_inputs(op_count)

        # x2: int4 ND -> TransData -> int4 FRACTAL_Z 4D [k1, n1, 16, 64]
        # (真实 N/K 已按 x1 锚定; desc 的 2D 轴序两种口径都已归一)
        trans = _math.TransData(inputs[1], src_format="ND", dst_format="FRACTAL_Z")
        trans.set_data_type(DataType.DT_INT4)
        trans.set_format(Format.FORMAT_FRACTAL_Z)
        trans.set_shape(_frac_z_dims(n_dim, k_dim))

        # 部分输入可能不存在 (optional 未接线), 从 inputs 列表里安全选取:
        # 复用 inputs[i], 对不存在 (size 为 0 或越界) 的走 None
        def optional(i):
            return inputs[i] if i < op_count else None

        y_desc = node.get_output_desc(0)
        y = _nn.QuantBatchMatmulV4(
            inputs[0],
            trans,
            optional(2),  # bias
            optional(3),  # x1_scale
            optional(4),  # x2_scale
            optional(5),  # y_scale
            optional(6),  # x1_offset
            optional(7),  # x2_offset
            optional(8),  # y_offset
            optional(9),  # x2_table
            dtype=_get_node_attr(node, "dtype", 1),
            compute_type=_get_node_attr(node, "compute_type", -1),
            transpose_x1=bool(_get_node_attr(node, "transpose_x1", False)),
            transpose_x2=bool(_get_node_attr(node, "transpose_x2", False)),
            group_size=_get_node_attr(node, "group_size", -1),
        )
        y.set_data_type(y_desc.data_type)
        y.set_format(Format.FORMAT_ND)

        # FormatRefiner (InferOriginFormat 阶段) 会以本 replacement 的 FZ TransData 为
        # 锚点做前向/回推 format 改写: QBMV4 InferFormat 把 y 标 FZ, 再传染到外图消费链
        # (Mul/Reshape), FE transop 插的 trans_TransData 在 noinferformat ops 栈无内核 →
        # unknown shape 子图 → mobile om 拒收。_format_locked (ATTR_NAME_FORMAT_LOCKED,
        # attr/ge_attr_define.cc) 使 refiner 跳过对该节点的改写 (format_refiner.cc
        # AnchorsInferProcess:166 / BackInferProcess:327 改写前均先读此锁)。
        # x2 侧 FZ 是合法语义 (desc 已显式声明), refiner 只是不再重写; y/下游保持 ND。
        # 必须在 build_and_reset 之前挂 (builder 构图期一次性 API)。
        builder.set_node_attr_bool(trans, "_format_locked", True)
        builder.set_node_attr_bool(y, "_format_locked", True)

        builder.set_graph_output(y, 0)
        graph = builder.build_and_reset()

        # 锁使 FormatRefiner 跳过 QBMV4, 其 x2 输入 desc 的 ND->FZ 刷新也随之丢失,
        # 与折叠后的 FZ Const 失配 (FE 判 origin_format 不连续 -> 插 11 条无内核
        # trans_TransData)。此处手工补上该刷新: 用 Node.update_input_desc 把子图内
        # QBMV4 的 x2 desc 显式改为 FZ 4D (storage/origin 同步, 与 stage5 dump 中
        # refiner 刷出的形态完全一致)。y 输出 desc 已在上面 set_format(ND)+锁,
        # refiner 不再前向污染下游。
        fz = Format.FORMAT_FRACTAL_Z
        for nd in graph.get_all_nodes():
            if nd.type == "QuantBatchMatmulV4":
                desc = nd.get_input_desc(1)
                desc.set_format(fz).set_origin_format(fz)
                desc.set_shape(_frac_z_dims(n_dim, k_dim))
                desc.set_origin_shape(_frac_z_dims(n_dim, k_dim))
                nd.update_input_desc(1, desc)
        return graph
