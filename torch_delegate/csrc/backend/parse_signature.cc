/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parse_signature.h"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/logging/logging.h"

namespace td {
namespace {

bool IsNoneValue(const OrderedJson &value) {
  return value.is_object() && value.contains("as_none") && value["as_none"].is_boolean() &&
         value["as_none"].get<bool>();
}

bool IsConstantInputNone(const OrderedJson &spec) {
  return spec.contains("constant_input") && spec["constant_input"].is_object() &&
         spec["constant_input"].contains("value") && IsNoneValue(spec["constant_input"]["value"]);
}

std::string GetConstantInputName(const OrderedJson &spec) {
  if (spec.contains("constant_input") && spec["constant_input"].is_object() &&
      spec["constant_input"].contains("name") && spec["constant_input"]["name"].is_string()) {
    return spec["constant_input"]["name"].get<std::string>();
  }
  return "";
}

bool OutputSpecHasNoneArg(const OrderedJson &spec) {
  static const std::vector<const char *> kOutputSpecKinds = {"user_output", "user_input_mutation", "buffer_mutation"};
  for (const char *kind : kOutputSpecKinds) {
    if (spec.contains(kind) && spec[kind].is_object() && spec[kind].contains("arg") && IsNoneValue(spec[kind]["arg"])) {
      return true;
    }
  }
  return false;
}

PlaceholderKind ParseInputKind(const OrderedJson &spec, std::string &placeholder_name, std::string &logical_name,
                               bool &persistent, NodeInputKind &input_kind, bool &ok) {
  placeholder_name.clear();
  logical_name.clear();
  persistent = false;
  input_kind = NodeInputKind::Tensor;

  auto parse_named_arg = [&ok](const OrderedJson &obj, const char *arg_key) -> std::string {
    if (!obj.contains(arg_key) || !obj[arg_key].is_object()) {
      TDLOGE("Input spec missing arg object");
      ok = false;
      return "";
    }
    const auto &arg = obj[arg_key];
    if (arg.contains("name") && arg["name"].is_string()) {
      return arg["name"].get<std::string>();
    }
    if (arg.contains("as_tensor") && arg["as_tensor"].is_object() && arg["as_tensor"].contains("name") &&
        arg["as_tensor"]["name"].is_string()) {
      return arg["as_tensor"]["name"].get<std::string>();
    }
    static const std::vector<const char *> kSymFields = {"as_sym_int", "as_sym_float", "as_sym_bool"};
    for (const auto &field : kSymFields) {
      if (arg.contains(field) && arg[field].is_object() && arg[field].contains("as_name") &&
          arg[field]["as_name"].is_string()) {
        return arg[field]["as_name"].get<std::string>();
      }
    }
    TDLOGE("Input spec arg missing name");
    ok = false;
    return "";
  };

  if (spec.contains("parameter") && spec["parameter"].is_object()) {
    const auto &parameter = spec["parameter"];
    placeholder_name = parse_named_arg(parameter, "arg");
    if (parameter.contains("parameter_name") && parameter["parameter_name"].is_string()) {
      logical_name = parameter["parameter_name"].get<std::string>();
    }
    return PlaceholderKind::Parameter;
  }
  if (spec.contains("buffer") && spec["buffer"].is_object()) {
    const auto &buffer = spec["buffer"];
    placeholder_name = parse_named_arg(buffer, "arg");
    if (buffer.contains("buffer_name") && buffer["buffer_name"].is_string()) {
      logical_name = buffer["buffer_name"].get<std::string>();
    }
    if (buffer.contains("persistent") && buffer["persistent"].is_boolean()) {
      persistent = buffer["persistent"].get<bool>();
    }
    return PlaceholderKind::Buffer;
  }
  if (spec.contains("tensor_constant") && spec["tensor_constant"].is_object()) {
    const auto &tensor_constant = spec["tensor_constant"];
    placeholder_name = parse_named_arg(tensor_constant, "arg");
    if (tensor_constant.contains("tensor_constant_name") && tensor_constant["tensor_constant_name"].is_string()) {
      logical_name = tensor_constant["tensor_constant_name"].get<std::string>();
    }
    return PlaceholderKind::TensorConstant;
  }
  if (spec.contains("user_input") && spec["user_input"].is_object()) {
    const auto &user_input = spec["user_input"];
    placeholder_name = parse_named_arg(user_input, "arg");
    logical_name = placeholder_name;

    // Identify value type from arg
    if (user_input.contains("arg") && user_input["arg"].is_object()) {
      const auto &arg = user_input["arg"];
      if (arg.contains("as_tensor")) {
        input_kind = NodeInputKind::Tensor;
      } else if (arg.contains("as_sym_int")) {
        input_kind = NodeInputKind::SymInt;
      } else if (arg.contains("as_sym_float")) {
        input_kind = NodeInputKind::SymFloat;
      } else if (arg.contains("as_sym_bool")) {
        input_kind = NodeInputKind::SymBool;
      } else if (arg.contains("as_int")) {
        input_kind = NodeInputKind::ScalarInt;
      } else if (arg.contains("as_float")) {
        input_kind = NodeInputKind::ScalarFloat;
      } else if (arg.contains("as_bool")) {
        input_kind = NodeInputKind::ScalarBool;
      } else if (arg.contains("as_string")) {
        input_kind = NodeInputKind::ScalarString;
      }
    }

    return PlaceholderKind::UserInput;
  }
  if (spec.contains("constant_input") && spec["constant_input"].is_object()) {
    const auto &constant_input = spec["constant_input"];
    if (constant_input.contains("name") && constant_input["name"].is_string()) {
      placeholder_name = constant_input["name"].get<std::string>();
    }
    logical_name = placeholder_name;

    // Identify value type from value
    if (constant_input.contains("value") && constant_input["value"].is_object()) {
      const auto &value = constant_input["value"];
      if (value.contains("as_int")) {
        input_kind = NodeInputKind::ScalarInt;
      } else if (value.contains("as_float")) {
        input_kind = NodeInputKind::ScalarFloat;
      } else if (value.contains("as_bool")) {
        input_kind = NodeInputKind::ScalarBool;
      } else if (value.contains("as_string")) {
        input_kind = NodeInputKind::ScalarString;
      }
    }

    return PlaceholderKind::ConstantInput;
  }
  TDLOGE("Unsupported input spec kind");
  ok = false;
  return PlaceholderKind::Unknown;
}

std::string ParseNamedValueRef(const OrderedJson &item, const std::string &code, bool &ok) {
  if (!item.is_object()) {
    TDLOGE("%s: Value reference must be object", code.c_str());
    ok = false;
    return "";
  }
  if (item.contains("name") && item["name"].is_string()) {
    return item["name"].get<std::string>();
  }
  if (item.contains("as_tensor") && item["as_tensor"].is_object() && item["as_tensor"].contains("name") &&
      item["as_tensor"]["name"].is_string()) {
    return item["as_tensor"]["name"].get<std::string>();
  }
  if (item.contains("as_sym_int") && item["as_sym_int"].is_object() && item["as_sym_int"].contains("as_name") &&
      item["as_sym_int"]["as_name"].is_string()) {
    return item["as_sym_int"]["as_name"].get<std::string>();
  }
  if (item.contains("as_sym_float") && item["as_sym_float"].is_object() && item["as_sym_float"].contains("as_name") &&
      item["as_sym_float"]["as_name"].is_string()) {
    return item["as_sym_float"]["as_name"].get<std::string>();
  }
  if (item.contains("as_sym_bool") && item["as_sym_bool"].is_object() && item["as_sym_bool"].contains("as_name") &&
      item["as_sym_bool"]["as_name"].is_string()) {
    return item["as_sym_bool"]["as_name"].get<std::string>();
  }
  if (item.contains("as_int") || item.contains("as_float") || item.contains("as_bool") || item.contains("as_string")) {
    return "";
  }
  TDLOGE("%s: Value reference must contain a supported name field", code.c_str());
  ok = false;
  return "";
}

std::string ParseOutputSpecName(const OrderedJson &spec, bool &ok) {
  static const std::vector<const char *> kOutputSpecKinds = {"user_output", "user_input_mutation", "buffer_mutation"};
  for (const char *kind : kOutputSpecKinds) {
    if (spec.contains(kind) && spec[kind].is_object() && spec[kind].contains("arg")) {
      return ParseNamedValueRef(spec[kind]["arg"], "INVALID_OUTPUT_SPEC", ok);
    }
  }
  TDLOGE("Unsupported output spec kind");
  ok = false;
  return "";
}

}  // namespace

Status ParseInputSpecs(const OrderedJson &signature, std::vector<ParsedInputSpec> &specs) {
  specs.clear();
  if (!signature.contains("input_specs") || !signature["input_specs"].is_array()) {
    return SUCCESS;
  }
  const auto &input_specs = signature["input_specs"];
  for (size_t i = 0; i < input_specs.size(); ++i) {
    const auto &spec = input_specs[i];
    if (IsConstantInputNone(spec)) {
      TDLOGE("Unsupported input type 'None' at input_specs[%zu], name='%s'", i, GetConstantInputName(spec).c_str());
      return INVALID_PARAM;
    }
    ParsedInputSpec parsed;
    parsed.raw_json = spec;
    bool ok = true;
    parsed.kind =
        ParseInputKind(spec, parsed.placeholder_name, parsed.logical_name, parsed.persistent, parsed.input_kind, ok);
    if (!ok) {
      return INVALID_PARAM;
    }
    specs.push_back(std::move(parsed));
  }
  return SUCCESS;
}

Status ParseOutputSpecs(const OrderedJson &signature, std::vector<ParsedOutputSpec> &specs) {
  specs.clear();
  if (!signature.contains("output_specs") || !signature["output_specs"].is_array()) {
    return SUCCESS;
  }
  const auto &output_specs = signature["output_specs"];
  for (size_t i = 0; i < output_specs.size(); ++i) {
    const auto &spec = output_specs[i];
    if (OutputSpecHasNoneArg(spec)) {
      TDLOGE("Unsupported output type 'None' at output_specs[%zu]", i);
      return INVALID_PARAM;
    }
    ParsedOutputSpec parsed;
    parsed.raw_json = spec;
    bool ok = true;
    parsed.name = ParseOutputSpecName(spec, ok);
    if (!ok) {
      return INVALID_PARAM;
    }
    specs.push_back(std::move(parsed));
  }
  return SUCCESS;
}

Status ParseGraphInputs(const OrderedJson &array, const std::vector<ParsedInputSpec> &input_specs,
                        std::vector<ParsedValueRef> &values) {
  values.clear();
  if (!array.is_array()) {
    TDLOGE("graph.inputs: Expected JSON array");
    return INVALID_PARAM;
  }
  if (!array.empty() && input_specs.empty()) {
    TDLOGE("signature.input_specs is required when graph.inputs is present");
    return INVALID_PARAM;
  }
  if (array.size() != input_specs.size()) {
    TDLOGE("graph.inputs and signature.input_specs size mismatch: graph.inputs=%zu, signature.input_specs=%zu",
           array.size(), input_specs.size());
    return INVALID_PARAM;
  }
  for (size_t i = 0; i < array.size(); ++i) {
    if (IsNoneValue(array[i])) {
      TDLOGE("Unsupported input type 'None' at graph.inputs[%zu], name='%s'", i,
             input_specs[i].placeholder_name.c_str());
      return INVALID_PARAM;
    }
    bool ok = true;
    const std::string graph_name = ParseNamedValueRef(array[i], "INVALID_GRAPH_INPUTS", ok);
    if (!ok) {
      return INVALID_PARAM;
    }
    const std::string &spec_name = input_specs[i].placeholder_name;
    if (!spec_name.empty()) {
      values.push_back({spec_name});
      continue;
    }
    if (graph_name.empty()) {
      TDLOGE("Graph input has no resolvable name at graph.inputs[%zu]", i);
      return INVALID_PARAM;
    }
    values.push_back({graph_name});
  }
  return SUCCESS;
}

Status ParseGraphOutputs(const OrderedJson &array, const std::vector<ParsedOutputSpec> &output_specs,
                         std::vector<ParsedValueRef> &values) {
  values.clear();
  if (!array.is_array()) {
    TDLOGE("graph.outputs: Expected JSON array");
    return INVALID_PARAM;
  }
  if (!array.empty() && output_specs.empty()) {
    TDLOGE("signature.output_specs is required when graph.outputs is present");
    return INVALID_PARAM;
  }
  if (array.size() != output_specs.size()) {
    TDLOGE("graph.outputs and signature.output_specs size mismatch: graph.outputs=%zu, signature.output_specs=%zu",
           array.size(), output_specs.size());
    return INVALID_PARAM;
  }
  for (size_t i = 0; i < array.size(); ++i) {
    if (IsNoneValue(array[i])) {
      TDLOGE("Unsupported output type 'None' at graph.outputs[%zu]", i);
      return INVALID_PARAM;
    }
    bool ok = true;
    const std::string graph_name = ParseNamedValueRef(array[i], "INVALID_GRAPH_OUTPUTS", ok);
    if (!ok) {
      return INVALID_PARAM;
    }
    const std::string &spec_name = output_specs[i].name;
    if (!spec_name.empty()) {
      values.push_back({spec_name});
      continue;
    }
    if (graph_name.empty()) {
      TDLOGE("Graph output has no resolvable name at graph.outputs[%zu]", i);
      return INVALID_PARAM;
    }
    values.push_back({graph_name});
  }
  return SUCCESS;
}

}  // namespace td
