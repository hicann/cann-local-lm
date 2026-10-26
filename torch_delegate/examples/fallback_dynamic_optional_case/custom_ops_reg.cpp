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
  m.def("my_addn(Tensor[] tensors) -> Tensor");
  m.def("my_addmm(Tensor input, Tensor mat1, Tensor mat2, Tensor? bias, *, Scalar beta=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("my_addn", [](const c10::List<at::Tensor> &tensors) -> at::Tensor {
    at::Tensor result = tensors[0].clone();
    for (size_t i = 1; i < tensors.size(); ++i) {
      result = result + tensors[i];
    }
    return result;
  });

  m.impl("my_addmm",
         [](const at::Tensor &input, const at::Tensor &mat1, const at::Tensor &mat2,
            const c10::optional<at::Tensor> &bias, const at::Scalar &beta) -> at::Tensor {
           at::Tensor result = input + beta * at::mm(mat1, mat2);
           if (bias.has_value()) {
             result = result + bias.value();
           }
           return result;
         });
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("my_addn", [](const c10::List<at::Tensor> &tensors) -> at::Tensor {
    at::Tensor result = tensors[0].clone();
    for (size_t i = 1; i < tensors.size(); ++i) {
      result = result + tensors[i];
    }
    return result;
  });

  m.impl("my_addmm",
         [](const at::Tensor &input, const at::Tensor &mat1, const at::Tensor &mat2,
            const c10::optional<at::Tensor> &bias, const at::Scalar &beta) -> at::Tensor {
           at::Tensor result = input + beta * at::mm(mat1, mat2);
           if (bias.has_value()) {
             result = result + bias.value();
           }
           return result;
         });
}
