/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/aten_schema_resolver.h"

#include "backend/graph_build_utils.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {

Status AtenSchemaResolver::ParseTarget(const std::string &target, const std::string &context,
                                       AtenTargetInfo &out) const {
  out.target = target;
  std::string op_str = target;
  constexpr char kPrefix[] = "torch.ops.";
  if (op_str.rfind(kPrefix, 0) == 0) {
    op_str = op_str.substr(sizeof(kPrefix) - 1);
  }
  const size_t last_dot = op_str.find_last_of('.');
  if (last_dot == std::string::npos) {
    TDLOGE("Target missing overload separator, context=%s, target=%s", context.c_str(), target.c_str());
    return INVALID_PARAM;
  }

  const std::string overload = op_str.substr(last_dot + 1);
  const std::string ns_op = op_str.substr(0, last_dot);
  const size_t second_dot = ns_op.find_last_of('.');
  if (second_dot == std::string::npos) {
    TDLOGE("Target missing namespace separator, context=%s, target=%s", context.c_str(), target.c_str());
    return INVALID_PARAM;
  }

  out.namespace_name = ns_op.substr(0, second_dot);
  out.op_name = ns_op.substr(second_dot + 1);
  out.overload = overload == "default" ? "" : overload;
  out.dispatcher_name = out.namespace_name + "::" + out.op_name;
  return SUCCESS;
}

Status AtenSchemaResolver::Resolve(const ParsedNode &node, AtenSchemaInfo &out) const {
  const std::string context = NodeDebugContext(node);
  TD_CHK_STATUS(ParseTarget(node.effective_schema_target, context, out.target));
  out.is_out = out.target.overload == "out";
  out.is_inplace = !out.is_out && !out.target.op_name.empty() && out.target.op_name.back() == '_';

  if (node.inputs.size() < node.schema_metadata_input_count) {
    TDLOGE("schema metadata input count is invalid, context=%s, inputs=%zu, schema_metadata_input_count=%zu",
           context.c_str(), node.inputs.size(), node.schema_metadata_input_count);
    return INVALID_PARAM;
  }

  for (size_t index = node.schema_metadata_input_count; index < node.inputs.size(); ++index) {
    const ParsedNodeInput &input = node.inputs[index];
    AtenSchemaArgInfo arg_info;
    arg_info.name = input.name.empty() ? "input_" + std::to_string(out.arguments.size() + 1) : input.name;
    arg_info.is_tensor = input.input_kind == NodeInputKind::Tensor;
    arg_info.is_optional_tensor = false;
    arg_info.is_tensor_list = false;
    arg_info.is_out = false;
    arg_info.has_default = false;
    if (!arg_info.is_tensor) {
      TDLOGE("Fallback node currently only supports tensor JSON inputs, context=%s, input=%s", context.c_str(),
             arg_info.name.c_str());
      return UNIMPLEMENTED;
    }
    out.arguments.push_back(std::move(arg_info));
  }
  return SUCCESS;
}

std::string AtenSchemaResolver::MakeFallbackOpType(const AtenTargetInfo &info) const {
  std::string id = info.namespace_name;
  if (!id.empty()) {
    id += "__";
  }
  id += info.op_name;
  if (!info.overload.empty()) {
    id += "_" + info.overload;
  }
  return id;
}

}  // namespace td
