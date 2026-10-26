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
#include <utility>
#include <vector>

#include <graph/attr_value.h>
#include <graph/tensor.h>

#include "backend/aten_tensor_desc_mapper.h"
#include "backend/graph_build_context.h"
#include "backend/parsed_model.h"
#include "backend/registered_ir_cache.h"
#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

namespace td {

class FallbackNodePlanner {
 public:
  FallbackNodePlanner(GraphBuildContext &ctx, const GraphPlanningDeps &deps);
  Status Plan(const ParsedNode &node);

 private:
  Status ResolveIrDef(const ParsedNode &node);
  Status RegisterScalarInputs(const ParsedNode &node);
  Status BuildInputDefs(const ParsedNode &node);
  Status BuildOutputDefs(const ParsedNode &node);
  Status BuildAttrs(const ParsedNode &node);
  Status BuildAndRegisterNode(const ParsedNode &node);
  Status BuildScalarConstHolder(const ParsedNodeInput &input, TensorHolder &out);
  Status CollectFallbackAttrs(const ParsedNode &node, OrderedJson &out);

  GraphBuildContext &ctx_;
  const GraphPlanningDeps &deps_;
  const RegisteredIrDef *ir_def_ = nullptr;
  std::string op_type_;
  std::string context_;
  std::vector<PlannedEdgeBinding> bindings_;
  std::vector<std::pair<std::string, ge::TensorDesc>> inputs_;
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs_;
  std::vector<std::pair<std::string, ge::AttrValue>> attrs_;
};

}  // namespace td
