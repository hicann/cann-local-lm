# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os

from ._version import __version__
from .api import export_to_epair, epair_save, epair_load
from .model_loader import EpairModelLoader, EpairModelLoaderError
from .epair.schema import EpairContents, EpairManifest
from .epair.config import EpairConfig
from .lite_context import (
    LiteConfig,
    LiteConversionError,
    LiteDependencyError,
    LiteExportError,
    LiteImportError,
    LiteStateError,
    lite_export_context,
)


def cmake_prefix() -> str:
    """返回 wheel 内 C++ 制品前缀的绝对路径。

    返回值即已安装的 torch_delegate 包目录，其下包含 inc/（对外头文件）、
    lib/（libtorch_delegate_backend.so）与 lib/cmake/TorchDelegate/（CMake 配置包），
    可直接加入 CMAKE_PREFIX_PATH 供 find_package(TorchDelegate) 解析。

    Returns:
        包目录绝对路径字符串。

    Example:
        >>> import torch_delegate
        >>> torch_delegate.cmake_prefix()
        '/path/to/site-packages/torch_delegate'
    """
    return os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "__version__",
    "cmake_prefix",
    "export_to_epair",
    "epair_save",
    "epair_load",
    "EpairContents",
    "EpairManifest",
    "EpairConfig",
    "LiteConfig",
    "lite_export_context",
    "EpairModelLoader",
    "EpairModelLoaderError",
    "LiteExportError",
    "LiteImportError",
    "LiteStateError",
    "LiteDependencyError",
    "LiteConversionError",
]
