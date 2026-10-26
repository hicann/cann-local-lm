/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/library.h>
#include <torch/torch.h>

TORCH_LIBRARY(my_ops, m) {
  m.def("inplace_scale_out(Tensor(a!) a, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("inplace_scale_out", [](at::Tensor &a, double scale) -> at::Tensor {
    a.mul_(scale);
    return a;
  });
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("inplace_scale_out", [](at::Tensor &a, double scale) -> at::Tensor {
    a.mul_(scale);
    return a;
  });
}
