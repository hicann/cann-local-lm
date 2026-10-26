/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parse_node_io.h"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace td {

std::vector<std::string> ParseTensorListValue(const OrderedJson &value) {
  std::vector<std::string> names;
  if (!value.is_object() || !value.contains("as_tensors") || !value["as_tensors"].is_array()) {
    return names;
  }
  for (const auto &tensor : value["as_tensors"]) {
    if (!tensor.is_object() || !tensor.contains("name") || !tensor["name"].is_string()) {
      return {};
    }
    names.push_back(tensor["name"].get<std::string>());
  }
  return names;
}

std::vector<std::string> OutputTensorNames(const ParsedNodeOutput &output) {
  if (output.is_tensor) {
    return {output.tensor_name};
  }
  if (output.is_tensor_list) {
    return output.tensor_names;
  }
  return {};
}

std::vector<std::string> InputTensorNames(const ParsedNodeInput &input) {
  if (input.input_kind == NodeInputKind::Tensor) {
    return {input.tensor_name};
  }
  if (input.input_kind == NodeInputKind::TensorList) {
    return input.tensor_names;
  }
  return {};
}

bool ParseSymOrScalarInputArg(const OrderedJson &arg, ParsedNodeInput &parsed_input, ParsedNode &parsed) {
  if (!arg.is_object()) return false;
  // Helper to attempt extracting as_name from a single sym argument object
  auto try_sym_name = [](const OrderedJson &obj, std::string &out_name) -> bool {
    if (!obj.is_object()) return false;
    if (obj.contains("as_name") && obj["as_name"].is_string()) {
      out_name = obj["as_name"].get<std::string>();
      return true;
    }
    return false;
  };
  // Check single sym types with as_name
  static const std::vector<const char *> kSingleSymFields = {"as_sym_int", "as_sym_float", "as_sym_bool"};
  for (const auto &field : kSingleSymFields) {
    if (arg.contains(field) && arg[field].is_object()) {
      std::string name;
      if (try_sym_name(arg[field], name)) {
        parsed.fx_input_tensor_names.push_back(name);
        parsed_input.tensor_name = name;
        if (std::string(field) == "as_sym_int") {
          parsed_input.input_kind = NodeInputKind::SymInt;
        } else if (std::string(field) == "as_sym_float") {
          parsed_input.input_kind = NodeInputKind::SymFloat;
        } else if (std::string(field) == "as_sym_bool") {
          parsed_input.input_kind = NodeInputKind::SymBool;
        }
      }
      return true;
    }
  }
  // Check list sym types with as_name
  static const std::vector<const char *> kListSymFields = {"as_sym_ints", "as_sym_floats", "as_sym_bools"};
  for (const auto &field : kListSymFields) {
    if (arg.contains(field) && arg[field].is_array()) {
      for (const auto &item : arg[field]) {
        std::string name;
        if (try_sym_name(item, name)) {
          parsed.fx_input_tensor_names.push_back(name);
          parsed_input.tensor_names.push_back(name);
        }
      }
      if (std::string(field) == "as_sym_ints") {
        parsed_input.input_kind = NodeInputKind::SymListInt;
      } else if (std::string(field) == "as_sym_floats") {
        parsed_input.input_kind = NodeInputKind::SymListFloat;
      } else if (std::string(field) == "as_sym_bools") {
        parsed_input.input_kind = NodeInputKind::SymListBool;
      }
      return true;
    }
  }
  // Check scalar types (no edge, inline value)
  if (arg.contains("as_int")) {
    parsed_input.input_kind = NodeInputKind::ScalarInt;
    return true;
  }
  if (arg.contains("as_ints")) {
    parsed_input.input_kind = NodeInputKind::ScalarListInt;
    return true;
  }
  if (arg.contains("as_float")) {
    parsed_input.input_kind = NodeInputKind::ScalarFloat;
    return true;
  }
  if (arg.contains("as_floats")) {
    parsed_input.input_kind = NodeInputKind::ScalarListFloat;
    return true;
  }
  if (arg.contains("as_bool")) {
    parsed_input.input_kind = NodeInputKind::ScalarBool;
    return true;
  }
  if (arg.contains("as_bools")) {
    parsed_input.input_kind = NodeInputKind::ScalarListBool;
    return true;
  }
  if (arg.contains("as_string") || arg.contains("as_strings")) {
    parsed_input.input_kind = NodeInputKind::ScalarString;
    return true;
  }
  if (arg.contains("as_scalar_type")) {
    parsed_input.input_kind = NodeInputKind::Enum;
    return true;
  }
  // Other scalar types that don't need special handling
  static const std::vector<const char *> kOtherScalarFields = {
      "as_memory_format", "as_layout", "as_device", "as_complex", "as_optional_tensor", "as_optional_tensors"};
  for (const auto &field : kOtherScalarFields) {
    if (arg.contains(field)) return true;
  }
  // Check as_none
  if (arg.contains("as_none") && arg["as_none"].is_boolean() && arg["as_none"].get<bool>()) {
    parsed_input.input_kind = NodeInputKind::None;
    return true;
  }
  return false;
}

}  // namespace td
