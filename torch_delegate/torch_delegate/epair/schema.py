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
        version: 格式版本号，当前为 1。
        backend_name: 后端名称，固定为 "ascend_ge"。
        cann_version: CANN 编译环境版本号，如 "9.1.0"。
            若无法获取 CANN 版本（如缺少 torch_npu），值为 "unknown"。
        total_nodes: EP JSON 中的总算子节点数。
        converted_nodes: 成功转换为 Ascend IR 的节点数。
            converted_nodes < total_nodes 表示存在 fallback 节点。
    """

    version: int
    backend_name: str
    cann_version: str
    total_nodes: int
    converted_nodes: int


@dataclasses.dataclass
class EpairContents:
    """Epair 转换结果，包含完整的转换产物。

    由 export_to_epair() 返回，可通过 epair_save() 序列化为 .epair 文件，
    或通过 epair_load() 从文件加载。

    Attributes:
        manifest: 转换元数据，包含节点统计和版本信息。
        ep_json: 含 ascend_ir 注解的 EP JSON dict。
            每个 node 对象可能包含可选的 ascend_ir 字段，
            格式详见 docs/epair.md 第 4 节。
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
