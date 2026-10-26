# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch

from torch.export import ExportedProgram, default_decompositions
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge.ge_graph_ascend import _optimize_and_convert, _GeGraphAscend, _NodeIR

from ..core.utils import logger
from .base import NodeIR, NodeIRConverter
from ..epair.config import EpairConfig
from ..lite_context import active_config


def _convert_torchair_node_ir(torchair_ir: _NodeIR) -> NodeIR:
    return NodeIR(
        source=torchair_ir.source, ops=torchair_ir.ops, mapping=torchair_ir.mapping
    )


_ASSERT_TENSOR_METADATA_TARGET = torch.ops.aten._assert_tensor_metadata.default


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
    def __init__(self, config: EpairConfig | None = None):
        self._config = config if config is not None else EpairConfig()
        self._ascend_ir_map: dict[str, NodeIR] = {}
        self._ge_graph_ascend: _GeGraphAscend | None = None
        logger.debug("TorchAirGEConverter initialized")

    def _create_compiler_config(self) -> CompilerConfig:
        """将 EpairConfig 转换为 CompilerConfig"""
        compiler_config = CompilerConfig()
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
