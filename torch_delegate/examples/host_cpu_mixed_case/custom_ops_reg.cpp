/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * HostCpu 混合调度示例 - C++ 侧自定义算子注册
 *
 * C++ 部署进程与 Python 导出进程不共享 Dispatcher 注册表，需以 TORCH_LIBRARY
 * 注册与 custom_ops.py 相同的 schema 与 CPU 实现，供 HostCpu 生成算子
 * callBoxed 回调。dlopen 本库后注册生效。
 */

#include <torch/library.h>
#include <torch/torch.h>

TORCH_LIBRARY(hcpu_ops, m) {
  m.def("feature_scale(Tensor self, Scalar factor) -> Tensor");
  m.def("feature_merge(Tensor a, Tensor b) -> Tensor");
  // ids 以 float32 传入、kernel 内转回 int64 查表（规避 GE HostCpu staging 对
  // DT_INT64 Data 输入的截断，见 custom_ops.py feature_gather 注释）
  m.def("feature_gather(Tensor table, Tensor ids_f) -> Tensor");
  m.def("score_rank(Tensor scores, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(hcpu_ops, CPU, m) {
  m.impl("feature_scale", [](const at::Tensor &self, const at::Scalar &factor) { return self * factor.toDouble(); });
  m.impl("feature_merge", [](const at::Tensor &a, const at::Tensor &b) { return a + b; });
  m.impl("feature_gather",
         [](const at::Tensor &table, const at::Tensor &ids_f) { return table.index_select(0, ids_f.to(at::kLong)); });
  m.impl("score_rank", [](const at::Tensor &scores, const at::Tensor &bias) { return scores + bias; });
}
