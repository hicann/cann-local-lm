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

"""Shared torch.library namespaces for fused ops.

npu 命名空间: stubs 环境 (无 torch_npu) 自定义 schema + fake + CPU impl;
  真实 torch_npu 环境 schema 由 torch_npu 提供, 不 define (避免冲突).
cann_lm 命名空间: 无 torchair 对应的算子 (如 chunk_gated_delta_rule, torchair raise).
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

# 检测 torch_npu 是否可用 (真实 NPU 环境)
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

# npu 命名空间: stubs 环境 define schema (torchair custom/ converter 自动注册);
# 真实 torch_npu 跳过 (torch_npu 已 define, 避免 RuntimeError: already registered)
NPU_LIB = torch.library.Library("npu", "DEF") if not _HAS_TORCH_NPU else None

# cann_lm 命名空间: 无 torchair 对应的算子 (chunk_gated_delta_rule, torchair raise RuntimeError)
CANN_LM_LIB = torch.library.Library("cann_lm", "DEF")
