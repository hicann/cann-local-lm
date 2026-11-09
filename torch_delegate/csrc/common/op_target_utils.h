/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <string>

namespace td {

// FX graph 算子 target 的三段式拆分结果。
struct OpTargetParts {
  std::string namespace_name;
  std::string op_name;
  std::string overload;
};

// 拆分 FX graph 的算子 target（torch.ops.<namespace>.<op>.<overload>）：
//   torch.ops.aten.topk.default   -> namespace="aten",      op="topk",      overload=""
//   torch.ops.aten.mul.Tensor     -> namespace="aten",      op="mul",       overload="Tensor"
//   torch.ops.my_ns.custom_add.default -> namespace="my_ns", op="custom_add", overload=""
// 末段 "default" 映射为空串（Dispatcher 以空串表示 default overload）。
// 段数不足时尽力拆分：无点号时整串视为 op_name，namespace 留空。
inline OpTargetParts ParseOpTarget(const std::string &target) {
  OpTargetParts parts;
  std::string op_str = target;
  const std::string prefix = "torch.ops.";
  if (op_str.rfind(prefix, 0) == 0) {
    op_str = op_str.substr(prefix.size());
  }

  const size_t last_dot = op_str.find_last_of('.');
  if (last_dot == std::string::npos) {
    parts.op_name = op_str;
    return parts;
  }
  parts.overload = op_str.substr(last_dot + 1);
  const std::string ns_op = op_str.substr(0, last_dot);

  const size_t ns_dot = ns_op.find_last_of('.');
  if (ns_dot == std::string::npos) {
    parts.op_name = ns_op;
  } else {
    parts.namespace_name = ns_op.substr(0, ns_dot);
    parts.op_name = ns_op.substr(ns_dot + 1);
  }

  if (parts.overload == "default") {
    parts.overload.clear();
  }
  return parts;
}

}  // namespace td
