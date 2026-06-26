# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3.5 model (modular)."""

from modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5VisionModel,
    Qwen3_5TextModel,
    Qwen3_5Model,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextForSequenceClassification,
    Qwen3_5ForSequenceClassification,
    Qwen3_5ForTokenClassification,
)
from configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
    "Qwen3_5VisionModel",
    "Qwen3_5TextModel",
    "Qwen3_5Model",
    "Qwen3_5ForCausalLM",
    "Qwen3_5TextForSequenceClassification",
    "Qwen3_5ForSequenceClassification",
    "Qwen3_5ForTokenClassification",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5PreTrainedModel",
]
