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

#include <memory>
#include <string>
#include <vector>

#include <graph/gnode.h>
#include <graph/graph.h>

#include "backend/aten_schema_resolver.h"
#include "backend/aten_tensor_desc_mapper.h"
#include "backend/parsed_model.h"
#include "backend/registered_ir_cache.h"
#include "backend/value_registry.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"

namespace td {

struct PlannedEdgeBinding {
  std::string src_ref;
  std::string input_name;
  int32_t dst_index = 0;
};

struct DeferredEdge {
  std::string src_ref;
  ge::GNode dst_node;
  int32_t dst_index = 0;
};

struct DeferredOutputBinding {
  std::string alias;
  std::string source_ref;
};

struct GraphBuildContext {
  explicit GraphBuildContext(const std::string &graph_name) : graph(std::make_shared<ge::Graph>(graph_name.c_str())) {}

  std::shared_ptr<ge::Graph> graph;
  td::ValueRegistry registry;
  RegisteredIrCache ir_cache;
  std::vector<DeferredEdge> deferred_edges;
  std::vector<DeferredOutputBinding> deferred_outputs;
};

struct GraphPlanningDeps {
  const ParsedModel &model;
  const td::AtenSchemaResolver &schema_resolver;
  const td::AtenTensorDescMapper &tensor_desc_mapper;
};

}  // namespace td
