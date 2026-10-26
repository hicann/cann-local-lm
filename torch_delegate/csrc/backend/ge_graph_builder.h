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

#include <graph/graph.h>

#include "backend/aten_schema_resolver.h"
#include "backend/aten_tensor_desc_mapper.h"
#include "backend/ge_graph_optimizer.h"
#include "backend/ge_weight_loader.h"
#include "backend/graph_build_context.h"
#include "backend/parsed_model.h"
#include "backend/value_registry.h"
#include "common/logging/logging.h"
#include "epair_archive.h"
#include "torch_delegate/td_types.h"

namespace td {

class GeGraphBuilder {
 public:
  explicit GeGraphBuilder(const EpairArchive &archive, Status &status);

  std::shared_ptr<ge::Graph> Build(InputBindingMap &binding_map);
  bool IsFullyOffloaded() const noexcept;

 private:
  Status PreparePlaceholders();
  Status PlanNodes();
  Status BindDeferredOutputs();
  Status BindDeferredEdges();
  Status FinalizeGraph();
  Status NormalizeGraph();

  const EpairArchive &archive_;
  ParsedModel model_;
  GeWeightLoader loader_;
  AtenSchemaResolver schema_resolver_;
  AtenTensorDescMapper tensor_desc_mapper_;
  GraphBuildContext ctx_;
  bool initialized_ = false;
};

}  // namespace td
