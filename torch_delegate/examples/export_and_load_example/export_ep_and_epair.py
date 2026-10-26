# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch
import torch_delegate as td
from pathlib import Path


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3))
        self.register_buffer("bias", torch.ones(3, 3))

    def forward(self, x):
        y = torch.mm(x, self.weight) + self.bias
        return torch.transpose(y, 0, 1)


model = MyModel().eval().npu()
example_input = (torch.randn(3, 4).npu(),)


# 1. 导出 ExportedProgram
ep = torch.export.export(model, example_input)

# 2. 保存为 pt2
root = Path(__file__).resolve().parent
pt2_path = root / "ep_model.pt2"
torch.export.save(ep, pt2_path)
print(f"Successfully saved {pt2_path}")

# 3. 导出 epair
epair_content = td.export_to_epair(ep)
epair_path = root / "epair_model.epair"
td.epair_save(epair_content, epair_path)
print(f"Successfully saved {epair_path}")

# 4. 将 example_input 保存为二进制文件（C++ 和精度比对共用）
for i, inp in enumerate(example_input):
    path = root / f"input_{i}.bin"
    inp.cpu().float().detach().numpy().tofile(path)
    print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")
