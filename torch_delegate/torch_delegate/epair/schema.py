# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import dataclasses
from typing import Any

from torch.export import ExportedProgram


@dataclasses.dataclass
class EpairManifest:
    """Epair 转换元数据。

    记录 ExportedProgram 到 Ascend IR 的转换统计信息。

    Attributes:
        version: 格式版本号，当前为 2（v1 为全 NPU 语义）。
        backend_name: 后端名称，固定为 "ascend_ge"。
        cann_version: CANN 编译环境版本号，如 "9.2.0"。
            若无法获取 CANN 版本（如缺少 torch_npu），值为 "unknown"。
        total_nodes: EP JSON 中的总算子节点数。
        converted_nodes: 成功转换为 Ascend IR 的节点数。
            converted_nodes < total_nodes 表示存在 fallback 节点。
        cpu_nodes: 判定为 CPU 执行的 fallback 节点数（ascend_ir.device == "cpu"）。
    """

    version: int
    backend_name: str
    cann_version: str
    total_nodes: int
    converted_nodes: int
    cpu_nodes: int = 0


@dataclasses.dataclass
class DeviceDecision:
    """单个计算节点的 Device 归属判定记录。

    Attributes:
        node_name: FX 节点名。
        target: 完整 target，如 "torch.ops.aten.nonzero.default"。
        namespace: 算子命名空间，"aten" 或自定义 namespace。
        op_name: 探测用的 dispatcher 名，如 "aten::nonzero"。
        overload: 精确 overload 名，用于识别"注册到错误 overload"。
        device: 判定结论，"npu" | "cpu"。
        rule: 判定依据，"converter" | "has_privateuse1" | "cpu_only"。
        has_privateuse1: 是否存在 PrivateUse1（NPU）kernel。
        has_cpu: 是否存在 CPU kernel。
    """

    node_name: str
    target: str
    namespace: str
    op_name: str
    overload: str
    device: str
    rule: str
    has_privateuse1: bool
    has_cpu: bool


@dataclasses.dataclass
class EpairContents:
    """Epair 转换结果，包含完整的转换产物。

    由 export_to_epair() 返回，可通过 epair_save() 序列化为 .epair 文件，
    或通过 epair_load() 从文件加载。

    Attributes:
        manifest: 转换元数据，包含节点统计和版本信息。
        ep_json: 含 ascend_ir 注解的 EP JSON dict。
            每个 node 对象可能包含可选的 ascend_ir 字段，
            格式详见 docs/zh/features/epair_format_design.md。
        ep: 可执行的 ExportedProgram。
            可通过 ep.module()(inputs...) 在 Python eager 模式下执行。

    Example:
        >>> contents = export_to_epair(ep)
        >>> # 检查转换覆盖率
        >>> rate = contents.manifest.converted_nodes / contents.manifest.total_nodes
        >>> print(f"转换率: {rate:.1%}")
        >>> # 分析 fallback 节点
        >>> for node in contents.ep_json["graph_module"]["graph"]["nodes"]:
        ...     if node.get("ascend_ir", {}).get("source") == "":
        ...         print(f"Fallback: {node['target']}")
    """

    manifest: EpairManifest
    ep_json: dict[str, Any]
    ep: ExportedProgram
    # 逐节点 Device 归属判定记录，仅存在于导出期（用于归属报告与严格校验），
    # 不随 .epair 文件序列化；epair_load() 返回的对象该字段为空列表。
    decisions: list[DeviceDecision] = dataclasses.field(default_factory=list)
