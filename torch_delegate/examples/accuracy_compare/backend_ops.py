# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Platform custom ops used by the compile demo."""

from __future__ import annotations

import torch

_LIBRARY = torch.library.Library("xllm_ascend", "DEF")
_LIBRARY.define(
    "rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()"
)
_LIBRARY.define(
    "beam_search(Tensor log_probs, Tensor top_tokens, Tensor top_probs) -> "
    "(Tensor, Tensor, Tensor)"
)


@torch.library.register_fake("xllm_ascend::rope_in_place")
def _rope_in_place_meta(
    input: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
):
    pass


@torch.library.impl("xllm_ascend::rope_in_place", "PrivateUse1")
def _rope_in_place_stub(
    input: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
) -> None:
    pass


@torch.library.register_fake("xllm_ascend::beam_search")
def _beam_search_meta(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del top_probs
    # Initialize outputs with -1 to verify if kernel writes data
    out_token_ids = torch.full(
        log_probs.shape, -1, dtype=torch.int32, device=log_probs.device
    )
    out_token_index = torch.full(
        log_probs.shape, -1, dtype=torch.int32, device=log_probs.device
    )
    out_log_probs = torch.full(
        log_probs.shape, -1.0, dtype=log_probs.dtype, device=log_probs.device
    )
    return (out_token_ids, out_token_index, out_log_probs)


torch.library.impl("xllm_ascend::beam_search", "PrivateUse1", _beam_search_meta)
