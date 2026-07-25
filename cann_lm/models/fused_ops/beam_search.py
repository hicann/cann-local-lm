# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cann_lm::beam_search - beam search op."""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "beam_search(Tensor log_probs, Tensor top_tokens, Tensor top_probs) -> "
    "(Tensor, Tensor, Tensor)"
)


@torch.library.register_fake("cann_lm::beam_search")
def _beam_search_meta(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del top_probs
    token_ids = top_tokens.new_empty(log_probs.shape)
    beam_indices = top_tokens.new_empty(log_probs.shape)
    scores = log_probs.new_empty(log_probs.shape)
    return token_ids, beam_indices, scores
