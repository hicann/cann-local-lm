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
  m.def("dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()");
  m.def("multi_optional_op(Tensor x, float alpha=1.0, float beta=0.0, float gamma=0.0, float delta=0.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("dual_inplace_", [](at::Tensor &a, at::Tensor &b, double scale_a, double scale_b) {
    a.mul_(scale_a);
    b.mul_(scale_b);
  });

  m.impl("multi_optional_op",
         [](const at::Tensor &x, double alpha, double beta, double gamma, double delta) -> at::Tensor {
           return x * alpha + beta + gamma + delta;
         });
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("dual_inplace_", [](at::Tensor &a, at::Tensor &b, double scale_a, double scale_b) {
    a.mul_(scale_a);
    b.mul_(scale_b);
  });

  m.impl("multi_optional_op",
         [](const at::Tensor &x, double alpha, double beta, double gamma, double delta) -> at::Tensor {
           return x * alpha + beta + gamma + delta;
         });
}
