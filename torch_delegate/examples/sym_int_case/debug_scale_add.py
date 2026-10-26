#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Debug scale_add computation
"""

import sys
from pathlib import Path
import torch

# 添加 custom_ops 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 测试 scale_add
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0])
scale = 0.5
shift = 2

# 调用 scale_add
result = torch.ops.my_ops.scale_add(x, y, scale, shift)

print(f"x: {x}")
print(f"y: {y}")
print(f"scale: {scale}")
print(f"shift: {shift}")
print(f"result: {result}")
print(f"expected (x * scale + y + shift): {x * scale + y + shift}")

# 测试第二个 scale_add
x2 = torch.tensor([4.9363, 2.3066, 0.5537])
y2 = torch.tensor([4.9363, 2.3066, 0.5537])
scale2 = 1.0
shift2 = 3

result2 = torch.ops.my_ops.scale_add(x2, y2, scale2, shift2)

print("\nSecond scale_add:")
print(f"x2: {x2}")
print(f"y2: {y2}")
print(f"scale2: {scale2}")
print(f"shift2: {shift2}")
print(f"result2: {result2}")
print(f"expected (x2 * scale2 + y2 + shift2): {x2 * scale2 + y2 + shift2}")
