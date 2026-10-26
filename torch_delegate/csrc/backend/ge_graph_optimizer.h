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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <graph/graph.h>

#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

namespace td {

struct GeInputBinding {
  std::vector<int64_t> source_user_indices;
};

using InputBindingMap = std::vector<GeInputBinding>;

class GeGraphOptimizer {
 public:
  static Status Optimize(ge::Graph &graph, InputBindingMap &binding_map);

 private:
  static Status SymPackMergePass(ge::Graph &graph, InputBindingMap &binding_map);
  static Status RemoveDeadDataAndReorderPass(ge::Graph &graph, InputBindingMap &binding_map);
};

}  // namespace td
