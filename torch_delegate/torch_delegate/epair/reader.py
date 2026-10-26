# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import json
import os

import torch.export

from ..core.utils import logger
from .constants import EPAIR_MANIFEST_FILENAME, EPAIR_ANNOTATED_MODEL_FILENAME
from .schema import EpairContents, EpairManifest


def epair_load(archive_path: str | os.PathLike) -> EpairContents:
    """加载 .epair 文件并返回 EpairContents。

    从 .epair 归档中读取：
    - manifest.json: 转换元数据
    - ep_json_with_ascend_ir.json: 含 ascend_ir 注解的 EP JSON
    - 标准 EP JSON 和权重数据

    Args:
        archive_path: .epair 文件路径，支持 str 或 os.PathLike。

    Returns:
        EpairContents 对象，包含：
        - manifest: 转换元数据
        - ep_json: 含 ascend_ir 注解的 EP JSON dict
        - ep: 可执行的 ExportedProgram（可直接调用 ep.module()(...)）

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        zipfile.BadZipFile: 当文件不是有效的 ZIP 归档时抛出。
        KeyError: 当归档缺少必要文件（manifest.json 等）时抛出。

    Example:
        >>> from torch_delegate import epair_load
        >>> contents = epair_load("model.epair")
        >>> print(f"后端: {contents.manifest.backend_name}")
        >>> # 在 Python eager 模式下执行
        >>> result = contents.ep.module()(input_tensor)

    Note:
        - 加载的 ExportedProgram 可在 Python eager 模式下执行，
          也可传递给 C++ 后端进行 NPU 加速。
        - ep_json 字段包含 ascend_ir 注解，可用于分析哪些算子被转换。
    """
    logger.info(f"Loading epair from {archive_path}")
    extra_files = {EPAIR_MANIFEST_FILENAME: "", EPAIR_ANNOTATED_MODEL_FILENAME: ""}
    ep = torch.export.load(archive_path, extra_files=extra_files)
    manifest = EpairManifest(**json.loads(extra_files[EPAIR_MANIFEST_FILENAME]))
    ep_json_with_ir = json.loads(extra_files[EPAIR_ANNOTATED_MODEL_FILENAME])
    logger.info(
        f"Epair loaded: backend={manifest.backend_name}, "
        f"version={manifest.version}, "
        f"{manifest.converted_nodes}/{manifest.total_nodes} nodes"
    )
    return EpairContents(manifest=manifest, ep_json=ep_json_with_ir, ep=ep)
