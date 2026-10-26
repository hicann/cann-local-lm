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

#include "backend/graph_build_context.h"
#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

namespace td {

class ConverterNodePlanner {
 public:
  ConverterNodePlanner(GraphBuildContext &ctx, const GraphPlanningDeps &deps);
  Status Plan(const ParsedNode &node);
  Status PlanNoOp(const ParsedNode &node);

 private:
  Status ProcessConstOp(const ParsedNode &node, const ParsedAscendIrOp &op);
  Status ProcessBuildableOp(const ParsedNode &node, const ParsedAscendIrOp &op);
  Status RegisterOutputMappings(const ParsedNode &node);
  Status PlanNoOpWithMappings(const ParsedNode &node);
  Status PlanNoOpPassthrough(const ParsedNode &node);

  GraphBuildContext &ctx_;
  const GraphPlanningDeps &deps_;
};

}  // namespace td
