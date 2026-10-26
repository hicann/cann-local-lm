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
  m.def("scale_add(Tensor x, Tensor y, float scale, SymInt shift=0) -> Tensor");
  m.def("custom_clamp(Tensor x, float min_val, float max_val, str mode=\"soft\") -> Tensor");
  m.def("multi_reduce(Tensor x, int[] dims, bool keepdim, float scale=1.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("scale_add", [](const at::Tensor &x, const at::Tensor &y, double scale, int64_t shift) -> at::Tensor {
    return x * scale + y + shift;
  });

  m.impl("custom_clamp", [](const at::Tensor &x, double min_val, double max_val, c10::string_view mode) -> at::Tensor {
    return torch::clamp(x, min_val, max_val);
  });

  m.impl("multi_reduce", [](const at::Tensor &x, at::IntArrayRef dims, bool keepdim, double scale) -> at::Tensor {
    return torch::sum(x, dims, keepdim) * scale;
  });
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("scale_add", [](const at::Tensor &x, const at::Tensor &y, double scale, int64_t shift) -> at::Tensor {
    return x * scale + y + shift;
  });

  m.impl("custom_clamp", [](const at::Tensor &x, double min_val, double max_val, c10::string_view mode) -> at::Tensor {
    return torch::clamp(x, min_val, max_val);
  });

  m.impl("multi_reduce", [](const at::Tensor &x, at::IntArrayRef dims, bool keepdim, double scale) -> at::Tensor {
    return torch::sum(x, dims, keepdim) * scale;
  });
}
