/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/registered_ir_cache.h"

#include <ge/ge_api_v2.h>
#include <graph/graph.h>
#include <graph/operator_reg.h>

#include "common/abi_compat_utils.h"
#include "common/logging/logging.h"

namespace td {

const RegisteredIrDef *RegisteredIrCache::Resolve(const std::string &op_type, const std::string &context) {
  const auto it = cache_.find(op_type);
  if (it != cache_.end()) {
    return &it->second;
  }
  std::vector<std::pair<ge::AscendString, ge::AscendString>> inputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> outputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> attrs;
  if (::GetRegisteredIrDefV2(op_type.c_str(), inputs, outputs, attrs) != ge::GRAPH_SUCCESS) {
    TDLOGE("GetRegisteredIrDefV2 failed, context=%s, op_type=%s", context.c_str(), op_type.c_str());
    return nullptr;
  }
  RegisteredIrDef ir_def;
  ir_def.input_names.reserve(inputs.size());
  ir_def.output_names.reserve(outputs.size());
  ir_def.attr_names.reserve(attrs.size());
  for (const auto &input : inputs) {
    const std::string ir_name = td::internal::AscendStringToString(input.first);
    if (ir_name.empty()) {
      TDLOGE("Registered IR input name is empty, context=%s, op_type=%s", context.c_str(), op_type.c_str());
      return nullptr;
    }
    ir_def.input_names.push_back(ir_name);
  }
  for (const auto &output : outputs) {
    const std::string ir_name = td::internal::AscendStringToString(output.first);
    if (ir_name.empty()) {
      TDLOGE("Registered IR output name is empty, context=%s, op_type=%s", context.c_str(), op_type.c_str());
      return nullptr;
    }
    ir_def.output_names.push_back(ir_name);
  }
  for (const auto &attr : attrs) {
    const std::string ir_name = td::internal::AscendStringToString(attr.first);
    if (!ir_name.empty()) {
      ir_def.attr_names.push_back(ir_name);
    }
  }
  const auto [inserted_it, inserted] = cache_.emplace(op_type, std::move(ir_def));
  return &inserted_it->second;
}

}  // namespace td
