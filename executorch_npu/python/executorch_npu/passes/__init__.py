# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Graph rewrites applied before torch-delegate conversion."""

from .edge_copy_pass import lower_edge_copy_ops
from .edge_ops_pass import lower_edge_ops_to_aten
from .scalar_constant_cast_pass import fold_scalar_constant_casts
from .softmax_pass import lower_softmax_int
from .static_split_pass import lower_static_split_with_sizes_copy


__all__ = [
    "fold_scalar_constant_casts",
    "lower_edge_copy_ops",
    "lower_edge_ops_to_aten",
    "lower_softmax_int",
    "lower_static_split_with_sizes_copy",
]
