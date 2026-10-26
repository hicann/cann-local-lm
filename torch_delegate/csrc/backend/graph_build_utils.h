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
#include <vector>

#include <graph/gnode.h>
#include <graph/graph.h>
#include <graph/tensor.h>

#include "backend/aten_tensor_desc_mapper.h"
#include "backend/graph_build_context.h"
#include "backend/model_types.h"
#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

namespace td {

struct NodeBuildSpec {
  std::string type;
  std::string name;
  std::vector<std::pair<std::string, ge::TensorDesc>> inputs;
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs;
  std::vector<std::pair<std::string, ge::AttrValue>> attrs;
  std::string context;
};

struct ConstNodeSpec {
  std::string node_name;
  TensorHolder holder;
  ge::Placement placement = ge::kPlacementDevice;
};

struct NodeErrorContext {
  std::string fx_tensor;
  std::string ge_tensor;
  std::string source_ref;
  int32_t dst_index = -1;

  NodeErrorContext() = default;
  NodeErrorContext(std::string fx, std::string ge = {}, std::string src = {}, int32_t idx = -1)
      : fx_tensor(std::move(fx)), ge_tensor(std::move(ge)), source_ref(std::move(src)), dst_index(idx) {}
};

size_t ElementSizeForGeDtype(ge::DataType dtype);
ge::TensorDesc MakeTensorDesc(const td::GeTensorDescInfo &info);
std::vector<uint8_t> TensorBytes(const TensorHolder &holder);

Status CreateConstNode(ge::Graph &graph, const ConstNodeSpec &spec, const td::AtenTensorDescMapper &mapper,
                       RegisteredValue &out);
Status ValidateConstOutputDesc(const ParsedAscendIrOp &op, const TensorHolder &holder,
                               const td::AtenTensorDescMapper &mapper);

Status BuildRecoverableNode(ge::Graph &graph, const NodeBuildSpec &spec, std::unique_ptr<ge::GNode> &out);
Status LookupInputIndex(ge::GNode &node, const std::string &input_name, int32_t &input_index,
                        const std::string &context);
Status LookupOutputIndex(ge::GNode &node, const std::string &output_name, int32_t &output_index,
                         const std::string &context);

bool IsConstOp(const ParsedAscendIrOp &op);
bool IsAscendIrPresent(const ParsedNode &node);
bool IsConverterNode(const ParsedNode &node);
bool HasBuildableConverterOps(const ParsedNode &node);
bool IsConverterNoOpNode(const ParsedNode &node);
bool IsAssertTensorMetadataNode(const ParsedNode &node);
bool IsSameTensorMeta(const ParsedTensorMeta &lhs, const ParsedTensorMeta &rhs);

std::vector<std::string> CollectOutputTensorNames(const ParsedNodeOutput &output);
std::vector<std::string> CollectNodeOutputTensorNames(const ParsedNode &node);
std::string ResolveBoundaryInputSourceRef(const ParsedNode &node, const std::string &input_ref);
std::string GetNodeNameForLog(const ge::GNode &node);

std::string MakeConverterNodeError(const ParsedNode &node, const ParsedAscendIrOp *op, const std::string &reason,
                                   const NodeErrorContext &ctx = {});
std::string MakeFallbackNodeError(const ParsedNode &node, const std::string &op_type, const std::string &reason,
                                  const NodeErrorContext &ctx = {});

bool IsScalarInput(const ParsedNodeInput &input);
bool IsSymInput(const ParsedNodeInput &input);
ge::DataType GetScalarDtype(const ParsedNodeInput &input);

std::string NodeDebugContext(const ParsedNode &node);

}  // namespace td
