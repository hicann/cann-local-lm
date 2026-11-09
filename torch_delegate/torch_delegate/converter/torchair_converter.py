# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch

from torch.export import ExportedProgram, default_decompositions

from ..core.utils import logger
from .base import NodeIR, NodeIRConverter
from ..epair.config import EpairConfig
from ..epair.schema import DeviceDecision
from ..lite_context import active_config


def _load_torchair_symbols():
    """加载 torchair 转换符号，优先独立 torchair 包，回退 torch_npu 内置实现。

    torch_npu>=2.9 起在 ``torch_npu.dynamo.torchair`` 内置等价实现，且与所配
    torch 版本 ABI 兼容；独立 torchair 包存在时保持原行为不变。
    """
    try:
        from torchair.configs.compiler_config import CompilerConfig
        from torchair.ge.ge_graph_ascend import (
            _GeGraphAscend,
            _NodeIR,
            _optimize_and_convert,
        )
    except ImportError:
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
        from torch_npu.dynamo.torchair.ge.ge_graph_ascend import (
            _GeGraphAscend,
            _NodeIR,
            _optimize_and_convert,
        )
    return CompilerConfig, _GeGraphAscend, _NodeIR, _optimize_and_convert


_CompilerConfig, _GeGraphAscend, _NodeIR, _optimize_and_convert = (
    _load_torchair_symbols()
)


def _convert_torchair_node_ir(torchair_ir: _NodeIR) -> NodeIR:
    return NodeIR(
        source=torchair_ir.source, ops=torchair_ir.ops, mapping=torchair_ir.mapping
    )


_ASSERT_TENSOR_METADATA_TARGET = torch.ops.aten._assert_tensor_metadata.default

# DeviceDecision.rule 取值：
#   "converter"        —— converter 转换成功，映射为 GE 内置算子（NPU 执行）
#   "has_privateuse1"  —— fallback 节点存在 PrivateUse1 kernel（Device 后端回调 NPU）
#   "cpu_only"         —— fallback 节点仅有 CPU/Composite kernel（HostCpu 后端回调 CPU）

# auto_functionalized(_v2) HOP：包裹带可变参数的原位算子，target 不是
# OpOverload，真实算子需从节点参数解包（与 C++ parse_functionalized.cc 的
# inputs[0].arg.as_operator 解包一致）。
_AUTO_FUNCTIONALIZED_TARGET_NAMES = ("auto_functionalized_v2", "auto_functionalized")


def _unwrap_functionalized_target(node: torch.fx.Node) -> torch._ops.OpOverload | None:
    """从 auto_functionalized(_v2) HOP 节点解包被包裹的真实算子。

    auto_functionalized(_v2)(**_mutable_op**, **kwargs) 的第一个位置参数即被
    包裹的算子（torch >= 2.9 导出图中为 OpOverload；兼容携带 ``func`` 属性的
    包装形态）；非上述两类 HOP 或无法解包时返回 None。
    """
    target_name = getattr(node.target, "__name__", str(node.target))
    if target_name not in _AUTO_FUNCTIONALIZED_TARGET_NAMES:
        return None
    if not node.args:
        return None
    first_arg = node.args[0]
    func = getattr(first_arg, "func", first_arg)
    return func if isinstance(func, torch._ops.OpOverload) else None


def _op_namespace(target: torch._ops.OpOverload) -> str:
    """取算子命名空间；不同 torch 版本下 namespace 可能为属性或方法。"""
    namespace = target.namespace
    return namespace() if callable(namespace) else namespace


def _dispatcher_op_name(target: torch._ops.OpOverload) -> str:
    """dispatcher 查询名："namespace::op"（不含 overload）。"""
    return f"{_op_namespace(target)}::{target._schema.name.split('::')[-1]}"


def _touch_op_packet(target: torch._ops.OpOverload) -> None:
    """显式触碰 torch.ops 命名空间，确保算子已注册进 Dispatcher（防惰性注册查询失败）。"""
    try:
        packet = getattr(torch.ops, _op_namespace(target))
        getattr(packet, target._schema.name.split("::")[-1].split(".")[0])
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"cannot materialize op packet for {target}: {exc}") from exc


def _dispatch_has_kernel(target: torch._ops.OpOverload, key: str) -> bool:
    _touch_op_packet(target)
    op_name = _dispatcher_op_name(target)
    try:
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key))
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"query dispatcher kernel failed for op {op_name}: {exc}"
        ) from exc


def _strip_assert_tensor_metadata(ep: ExportedProgram) -> int:
    """移除 torch.export 自动插入的 _assert_tensor_metadata 节点。

    该算子 schema 为 (..., Tensor) -> ()，仅做元数据断言、无计算语义，
    torchair 与 GE 后端均无法处理，需在转换前剥离。
    """
    removed = 0
    for node in list(ep.graph.nodes):
        if node.op == "call_function" and node.target is _ASSERT_TENSOR_METADATA_TARGET:
            ep.graph.erase_node(node)
            removed += 1
    if removed > 0:
        ep.graph.lint()
        ep.graph_module.recompile()
        logger.debug(f"Stripped {removed} _assert_tensor_metadata node(s)")
    return removed


class TorchAirGEConverter(NodeIRConverter):
    """基于 TorchAir 的 NodeIRConverter 实现。

    通过 torchair 的 _optimize_and_convert 将 ExportedProgram 转换为 GE 图，
    并逐节点产出 NodeIR，供 EpairWriter 注入 EP JSON 的 ascend_ir 字段。
    构造时未显式传入 config 则使用默认 EpairConfig()。
    """

    def __init__(self, config: EpairConfig | None = None):
        self._config = config if config is not None else EpairConfig()
        self._ascend_ir_map: dict[str, NodeIR] = {}
        self._ge_graph_ascend: _GeGraphAscend | None = None
        self._device_decisions: list[DeviceDecision] = []
        logger.debug("TorchAirGEConverter initialized")

    def _resolve_device(self, target: torch._ops.OpOverload) -> tuple[str, bool, bool]:
        """按 Dispatcher 注册表自动判定算子执行 Device。

        判定顺序：
        - 有 PrivateUse1 kernel → NPU（Device 后端回调，含注册到全部 backend 的
          CompositeExplicitAutograd 算子）；
        - 有 CPU kernel → CPU（HostCpu 后端回调）；
        - 仅注册 CompositeImplicitAutograd（分解实现，无 backend kernel）→ CPU：
          Dispatcher 对 NPU 设备张量查不到 kernel，回调时须传入 CPU 张量，
          composite kernel 在 CPU 上分解执行；
        - 以上皆无 → 直接报错。

        Returns:
            (device, has_privateuse1, has_cpu)
        """
        has_privateuse1 = _dispatch_has_kernel(target, "PrivateUse1")
        has_cpu = _dispatch_has_kernel(target, "CPU")
        if has_privateuse1:
            return "npu", has_privateuse1, has_cpu
        if has_cpu:
            return "cpu", has_privateuse1, has_cpu
        if _dispatch_has_kernel(target, "CompositeImplicitAutograd"):
            return "cpu", has_privateuse1, has_cpu
        raise RuntimeError(
            f"op {_dispatcher_op_name(target)} has neither PrivateUse1, CPU nor CompositeImplicitAutograd kernel"
        )

    def _make_decision(
        self,
        node: torch.fx.Node,
        target: object,
        device: str,
        has_privateuse1: bool,
        has_cpu: bool,
    ) -> DeviceDecision:
        node_ir = self._ascend_ir_map.get(node.name)
        rule = (
            "converter"
            if (node_ir is not None and node_ir.source == "converter")
            else ("has_privateuse1" if device == "npu" else "cpu_only")
        )
        if isinstance(target, torch._ops.OpOverload):
            namespace = _op_namespace(target)
            op_name = _dispatcher_op_name(target)
            overload = target._schema.overload_name
            # 规范化为 "torch.ops.<ns>.<op>.<overload>"（str(OpOverload) 在部分
            # torch 版本不带前缀，需与 EP JSON/C++ 侧 effective_schema_target 一致）
            target_str = f"torch.ops.{target}"
        else:
            # converter 的 HOP 节点（如 auto_functionalized_v2）：从 target 字符串提取
            namespace = ""
            op_name = str(target)
            overload = ""
            target_str = str(target)
        return DeviceDecision(
            node_name=node.name,
            target=target_str,
            namespace=namespace,
            op_name=op_name,
            overload=overload,
            device=device,
            rule=rule,
            has_privateuse1=has_privateuse1,
            has_cpu=has_cpu,
        )

    def _resolve_mock_node_devices(self) -> None:
        """对 ascend_ir_map 中的全部计算节点记录 Device 归属。

        - converter 节点：映射为 GE 内置算子，固定 NPU，rule="converter"；
        - fallback（mock）节点：按 Dispatcher 自动判定并写回 NodeIR.device；
          auto_functionalized(_v2) HOP 节点先解包被包裹的真实算子再判定。
        判定记录用于导出期归属报告与严格校验，不随 epair 落盘。
        """
        optimized_graph = (
            self._ge_graph_ascend.optimized.graph if self._ge_graph_ascend else None
        )
        if optimized_graph is None:
            return
        for node in optimized_graph.nodes:
            if node.op != "call_function":
                continue
            node_ir = self._ascend_ir_map.get(node.name)
            if node_ir is None:
                continue
            target = node.target
            if node_ir.source == "converter":
                # converter 节点不走 Dispatcher 回调，kernel 探测仅作记录；
                # 惰性注册可能导致探测失败，失败时置占位值（不影响归属结论）。
                has_privateuse1 = has_cpu = True
                if isinstance(target, torch._ops.OpOverload):
                    try:
                        has_privateuse1 = _dispatch_has_kernel(target, "PrivateUse1")
                        has_cpu = _dispatch_has_kernel(target, "CPU")
                    except RuntimeError as exc:
                        logger.debug(
                            f"kernel probe skipped for converter node {node.name}: {exc}"
                        )
                self._device_decisions.append(
                    self._make_decision(node, target, "npu", has_privateuse1, has_cpu)
                )
                logger.debug(
                    f"Device ownership: node={node.name}, target={target}, device=npu (converter)"
                )
                continue
            if not isinstance(target, torch._ops.OpOverload):
                target = _unwrap_functionalized_target(node)
            if not isinstance(target, torch._ops.OpOverload):
                raise RuntimeError(
                    f"fallback node {node.name} has unsupported target type "
                    f"{type(node.target).__name__}, cannot resolve device ownership"
                )
            device, has_privateuse1, has_cpu = self._resolve_device(target)
            node_ir.device = device
            self._device_decisions.append(
                self._make_decision(node, target, device, has_privateuse1, has_cpu)
            )
            logger.debug(
                f"Device ownership: node={node.name}, target={target}, device={device}"
            )

    def get_device_decisions(self) -> list[DeviceDecision]:
        """返回逐节点 Device 归属判定记录（供归属报告与严格校验使用）。"""
        return self._device_decisions

    def _create_compiler_config(self) -> "_CompilerConfig":
        """将 EpairConfig 转换为 CompilerConfig"""
        compiler_config = _CompilerConfig()
        compiler_config.experimental_config.enable_view_optimize = False
        compiler_config.experimental_config.remove_noop_ops = (
            self._config.remove_noop_ops
        )
        pattern_fusion_pass = self._config.pattern_fusion_pass
        if active_config() is not None:
            pattern_fusion_pass = False
            logger.warning_once(
                "Lite EPAIR export disables TorchAir pattern fusion; "
                "related NPU operator fusion will not be applied."
            )
        compiler_config.experimental_config.pattern_fusion_pass = pattern_fusion_pass
        return compiler_config

    def convert_graph(self, ep: ExportedProgram) -> ExportedProgram:
        """使用 torchair 的 _optimize_and_convert API 转换图"""
        logger.info("Starting graph conversion using torchair._optimize_and_convert...")

        compiler_config = self._create_compiler_config()

        logger.debug(
            f"Whitelist decompositions: {len(self._config.whitelist_decompositions)} ops"
        )
        decomps = {}
        table = default_decompositions()

        # 从白名单添加decomposition
        for op in self._config.whitelist_decompositions:
            if op in table:
                decomps[op] = table[op]
                logger.debug(f"Added {op} to decomps from whitelist")

        _strip_assert_tensor_metadata(ep)

        # 调用 torchair API
        self._ge_graph_ascend = _optimize_and_convert(
            ep, config=compiler_config, custom_decompositions=decomps
        )

        # 获取 torchair ascend_ir_map 并转换为本地 NodeIR
        torchair_ir_map = self._ge_graph_ascend.ascend_ir_map
        self._ascend_ir_map = {
            name: _convert_torchair_node_ir(ir) for name, ir in torchair_ir_map.items()
        }

        # decompose之后还会出现 assert_tensor_metadata, 做二次清理
        _strip_assert_tensor_metadata(self._ge_graph_ascend.optimized)

        # 对 fallback（mock）节点执行 Device 归属判定，结果写回 NodeIR.device
        self._resolve_mock_node_devices()

        logger.info(
            f"Conversion complete: {len(self._ascend_ir_map)} nodes with ascend_ir"
        )

        return self._ge_graph_ascend.optimized

    def is_node_supported(self, node: torch.fx.Node) -> bool:
        return node.op == "call_function"

    def convert_node(self, node: torch.fx.Node) -> NodeIR | None:
        """从 ascend_ir_map 获取节点 IR"""
        return self._ascend_ir_map.get(node.name)

    def get_ascend_ir_map(self) -> dict[str, NodeIR]:
        """返回 ascend_ir_map"""
        return self._ascend_ir_map

    def get_ge_graph_ascend(self) -> _GeGraphAscend:
        """返回完整的转换结果"""
        return self._ge_graph_ascend
