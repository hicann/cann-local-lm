# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TorchAir converters for the two demo custom ops."""

from __future__ import annotations

from typing import Any

import torch


def register_beam_search_converter() -> None:
    """Register BeamSearch converter (CANN has pre-registered IR).

    Used for:
    - epair export (compile_to_air.py)
    - Reference path (torch.compile)
    """
    from torchair import register_fx_node_ge_converter
    from torchair.ge import custom_op

    @register_fx_node_ge_converter(torch.ops.xllm_ascend.beam_search.default)
    def _convert_beam_search(
        log_probs: Any,
        top_tokens: Any,
        top_probs: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "BeamSearch",
            inputs={
                "log_probs": log_probs,
                "top_tokens": top_tokens,
                "top_probs": top_probs,
            },
            outputs=["out_token_ids", "out_token_index", "out_log_probs"],
        )


def register_assert_tensor_metadata_converter() -> None:
    """Register no-op converter for aten._assert_tensor_metadata.

    作为兜底保护：若上游未移除该节点，torchair 不会拒绝 PyTorch 内置 op。
    _assert_tensor_metadata 是 torch.export 自动插入的 metadata 校验节点，
    schema 为 -> ()，无计算语义，映射为 GE no-op 即可。
    """
    from torchair import register_fx_node_ge_converter

    @register_fx_node_ge_converter(torch.ops.aten._assert_tensor_metadata.default)
    def _convert_assert_tensor_metadata(*args: Any, **kwargs: Any) -> None:
        pass


def register_rope_in_place_converter() -> None:
    """Register RopeInPlace converter (only for Reference torch.compile path).

    NOT used during epair export to allow fallback codegen in Delegate path.
    """
    from torchair import register_fx_node_ge_converter
    from torchair.ge import custom_op

    @register_fx_node_ge_converter(torch.ops.xllm_ascend.rope_in_place.default)
    def _convert_rope_in_place(
        input: Any,
        sin_cache: Any,
        cos_cache: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "RopeInPlace",
            inputs={"input": input, "sin_cache": sin_cache, "cos_cache": cos_cache},
            outputs=["output"],
        )


def register_all_converters() -> None:
    """Register all converters (for Reference torch.compile path).

    Called in accuracy_compare.py before torch.compile.
    """
    register_assert_tensor_metadata_converter()
    register_beam_search_converter()
    register_rope_in_place_converter()
