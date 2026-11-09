# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Patch PyTorch的do_auto_functionalize为v2版本

通过在源头patch，确保ExportedProgram生成auto_functionalized_v2格式的节点
"""

from typing import Any, Callable

import torch._higher_order_ops.auto_functionalize as auto_func

from ..core.utils import logger

_original_do_auto_functionalize = None
_is_patched = False


def do_auto_functionalize_patch(
    mode: Any, op: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """
    Patched version that directly calls do_auto_functionalize_v2

    This ensures ExportedProgram generates v2 format nodes (auto_functionalized_v2)
    instead of v1 format (auto_functionalized), eliminating need for
    runtime conversion in torch.

    Args:
        mode: FunctionalTensorMode instance
        op: The operator being functionalized
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        The result of do_auto_functionalize_v2
    """
    return auto_func.do_auto_functionalize_v2(mode, op, args, kwargs)


def patch_pytorch():
    """
    Patch PyTorch's do_auto_functionalize to use v2

    This replaces torch._subclasses.functional_tensor.do_auto_functionalize
    with our patched version that directly calls do_auto_functionalize_v2.

    Returns:
        The original do_auto_functionalize function for restoration
    """
    global _original_do_auto_functionalize, _is_patched

    if _is_patched:
        return _original_do_auto_functionalize

    _original_do_auto_functionalize = auto_func.do_auto_functionalize
    auto_func.do_auto_functionalize = do_auto_functionalize_patch
    _is_patched = True

    logger.debug("PyTorch do_auto_functionalize patched to v2")
    return _original_do_auto_functionalize


def restore_pytorch(original: Callable[..., Any] | None = None) -> None:
    """
    Restore original PyTorch do_auto_functionalize

    Args:
        original: Optional, specify which function to restore
    """
    global _is_patched

    if not _is_patched:
        return

    if original is not None:
        auto_func.do_auto_functionalize = original
    elif _original_do_auto_functionalize is not None:
        auto_func.do_auto_functionalize = _original_do_auto_functionalize

    _is_patched = False
    logger.debug("PyTorch do_auto_functionalize restored to original")


def is_patched():
    """Check if PyTorch is currently patched"""
    return _is_patched


def get_original_function():
    """Get original do_auto_functionalize function"""
    return _original_do_auto_functionalize
