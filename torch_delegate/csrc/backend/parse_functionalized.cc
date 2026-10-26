/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parse_functionalized.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/logging/logging.h"

namespace td {

std::vector<std::string> ParseTensorListValue(const OrderedJson &value);

namespace {

bool IsFunctionalizedTarget(const std::string &target) {
  return target == "torch.ops.higher_order.auto_functionalized_v2";
}

bool IsLegacyFunctionalizedTarget(const std::string &target) {
  return target == "torch.ops.higher_order.auto_functionalized";
}

std::optional<int64_t> ParseIntArg(const OrderedJson &arg) {
  if (!arg.is_object() || !arg.contains("as_int") || !arg["as_int"].is_number_integer()) {
    return std::nullopt;
  }
  return arg["as_int"].get<int64_t>();
}

std::optional<std::vector<ParsedNodeInput>> RebuildFunctionalizedInputs(const ParsedNode &parsed) {
  std::vector<std::string> all_bases;
  std::vector<int64_t> base_indices;
  std::vector<ParsedNodeInput> other_inputs;
  other_inputs.reserve(parsed.inputs.size());
  for (size_t index = 1; index < parsed.inputs.size(); ++index) {
    const auto &input = parsed.inputs[index];
    if (input.name == "_all_bases") {
      all_bases = ParseTensorListValue(input.raw_arg);
      continue;
    }
    constexpr std::string_view kBaseIndexSuffix = "_base_index";
    if (input.name.size() >= kBaseIndexSuffix.size() &&
        input.name.compare(input.name.size() - kBaseIndexSuffix.size(), kBaseIndexSuffix.size(), kBaseIndexSuffix) ==
            0) {
      const std::optional<int64_t> parsed_index = ParseIntArg(input.raw_arg);
      if (!parsed_index.has_value()) {
        TDLOGE("_base_index must be integer, target=%s", parsed.target.c_str());
        return std::nullopt;
      }
      base_indices.push_back(parsed_index.value());
      continue;
    }
    other_inputs.push_back(input);
  }
  std::vector<ParsedNodeInput> normalized_inputs;
  normalized_inputs.reserve(base_indices.size() + other_inputs.size());
  for (size_t index = 0; index < base_indices.size(); ++index) {
    const int64_t base_index = base_indices[index];
    if (base_index < 0 || static_cast<size_t>(base_index) >= all_bases.size()) {
      TDLOGE("_base_index is out of range, target=%s", parsed.target.c_str());
      return std::nullopt;
    }
    ParsedNodeInput base_input;
    base_input.name = index == 0 ? "self" : "self_" + std::to_string(index);
    base_input.input_kind = NodeInputKind::Tensor;
    base_input.tensor_name = all_bases[static_cast<size_t>(base_index)];
    normalized_inputs.push_back(std::move(base_input));
  }
  normalized_inputs.insert(normalized_inputs.end(), other_inputs.begin(), other_inputs.end());
  return normalized_inputs;
}

}  // namespace

std::string StripTorchOpsPrefix(const std::string &value) {
  constexpr char kPrefix[] = "torch.ops.";
  return value.rfind(kPrefix, 0) == 0 ? value.substr(sizeof(kPrefix) - 1) : value;
}

Status PopulateEffectiveSchemaInfo(ParsedNode &parsed) {
  parsed.effective_schema_target = parsed.target;
  parsed.schema_metadata_input_count = 0;
  parsed.functionalize.is_functionalized = IsFunctionalizedTarget(parsed.target);
  parsed.functionalize.wrapped_target.clear();
  if (IsLegacyFunctionalizedTarget(parsed.target)) {
    TDLOGE("auto_functionalized v1 is not supported; please export auto_functionalized_v2, target=%s",
           parsed.target.c_str());
    return INVALID_PARAM;
  }
  if (!IsFunctionalizedTarget(parsed.target) || parsed.inputs.empty()) {
    parsed.functionalize.runtime_input_tensor_names = parsed.fx_input_tensor_names;
    parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
    return SUCCESS;
  }
  const ParsedNodeInput &first_input = parsed.inputs.front();
  if (first_input.input_kind != NodeInputKind::Operator || first_input.operator_name.empty()) {
    TDLOGE("functionalized node missing wrapped operator, target=%s", parsed.target.c_str());
    return INVALID_PARAM;
  }
  parsed.effective_schema_target = first_input.operator_name;
  parsed.functionalize.wrapped_target = first_input.operator_name;
  const auto normalized_inputs = RebuildFunctionalizedInputs(parsed);
  if (!normalized_inputs.has_value()) {
    return INVALID_PARAM;
  }
  if (!normalized_inputs.value().empty()) {
    parsed.inputs = normalized_inputs.value();
  }
  parsed.functionalize.metadata_input_count = parsed.schema_metadata_input_count = 1;
  parsed.functionalize.runtime_input_tensor_names.clear();
  for (const auto &input : parsed.inputs) {
    if (input.input_kind == NodeInputKind::Tensor && !input.tensor_name.empty()) {
      parsed.functionalize.runtime_input_tensor_names.push_back(input.tensor_name);
    }
  }
  parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
  return SUCCESS;
}

}  // namespace td
