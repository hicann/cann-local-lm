# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export the example model to EPAIR without a local CANN runtime.

This script only builds the GE IR used by EPAIR. It does not load, compile,
or execute the graph on an NPU.
"""

import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

import torch_delegate as td


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3))
        self.register_buffer("bias", torch.ones(3, 3))

    def forward(self, x):
        y = torch.mm(x, self.weight) + self.bias
        return torch.transpose(y, 0, 1)


def main() -> None:
    root = Path(__file__).resolve().parent
    model = MyModel().eval()
    example_input = (torch.randn(3, 4),)

    ep = torch.export.export(model, example_input)

    pt2_path = root / "ep_model_lite.pt2"
    torch.export.save(ep, pt2_path)
    print(f"Successfully saved {pt2_path}")

    lite_config = td.LiteConfig(
        target_soc="Ascend910_9362",
        allow_hf32=False,
    )
    with td.lite_export_context(lite_config):
        epair_content = td.export_to_epair(ep)

    epair_path = root / "epair_model_lite.epair"
    td.epair_save(epair_content, epair_path)
    print(
        "Successfully saved "
        f"{epair_path} ({epair_content.manifest.converted_nodes}/"
        f"{epair_content.manifest.total_nodes} nodes converted)"
    )

    for i, inp in enumerate(example_input):
        path = root / f"input_lite_{i}.bin"
        inp.detach().numpy().tofile(path)
        print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")


if __name__ == "__main__":
    main()
