/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parsed_model.h"

#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "backend/node_naming.h"
#include "backend/parse_ascend_ir.h"
#include "backend/parse_functionalized.h"
#include "backend/parse_node_io.h"
#include "backend/parse_signature.h"
#include "backend/parse_value_store.h"
#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {

namespace {

Status ParseNodeJsonImpl(const OrderedJson &node, size_t json_node_index,
                         std::unordered_map<std::string, size_t> &fallback_name_counters, ParsedNode &out);

Status ParseAscendIrJsonImpl(const OrderedJson &ascend_ir, const ParsedNode &parsed_node, ParsedAscendIr &out);

Status ParseRootJsonImpl(const std::string &model_json, OrderedJson &out) {
  TDLOGI("ParseRootJson enter: json_size=%zu", model_json.size());
  try {
    out = OrderedJson::parse(model_json);
    TDLOGI("ParseRootJson exit: success");
    return SUCCESS;
  } catch (const OrderedJson::parse_error &error) {
    TDLOGE("ParseRootJson failed: %s", error.what());
    out = OrderedJson::object();
    return INVALID_PARAM;
  }
}

struct ValueStoreRefs {
  std::unordered_map<std::string, ParsedTensorMeta> &tensor_values;
  std::unordered_map<std::string, int64_t> &sym_int_values;
  std::unordered_map<std::string, double> &sym_float_values;
  std::unordered_map<std::string, bool> &sym_bool_values;
  std::unordered_map<std::string, int64_t> &int_values;
  std::unordered_map<std::string, double> &float_values;
  std::unordered_map<std::string, bool> &bool_values;
  std::unordered_map<std::string, std::string> &string_values;
};

void ParseModelMetadata(const OrderedJson &root, SchemaVersion &schema_version, OrderedJson &opset_version,
                        std::string &torch_version,
                        std::unordered_map<std::string, std::pair<int64_t, int64_t>> &range_constraints) {
  if (root.contains("schema_version") && root["schema_version"].is_object()) {
    const auto &schema = root["schema_version"];
    if (schema.contains("major") && schema["major"].is_number_integer()) {
      schema_version.major = schema["major"].get<int64_t>();
    }
    if (schema.contains("minor") && schema["minor"].is_number_integer()) {
      schema_version.minor = schema["minor"].get<int64_t>();
    }
    TDLOGD("ParseModelJson: schema_version major=%ld, minor=%ld", static_cast<long>(schema_version.major),
           static_cast<long>(schema_version.minor));
  }
  if (root.contains("opset_version")) {
    opset_version = root["opset_version"];
  }
  if (root.contains("torch_version") && root["torch_version"].is_string()) {
    torch_version = root["torch_version"].get<std::string>();
    TDLOGD("ParseModelJson: torch_version=%s", torch_version.c_str());
  }
  if (root.contains("range_constraints") && root["range_constraints"].is_object()) {
    for (const auto &[name, constraint] : root["range_constraints"].items()) {
      if (!constraint.is_object() || !constraint.contains("min_val") || !constraint.contains("max_val")) {
        continue;
      }
      constexpr int64_t kDefaultMin = 0;
      constexpr int64_t kDefaultMax = -1;
      const int64_t min_val =
          constraint["min_val"].is_number_integer() ? constraint["min_val"].get<int64_t>() : kDefaultMin;
      const int64_t max_val =
          constraint["max_val"].is_number_integer() ? constraint["max_val"].get<int64_t>() : kDefaultMax;
      range_constraints[name] = {min_val, max_val};
      TDLOGD("ParseModelJson: range_constraint %s=[%ld, %ld]", name.c_str(), static_cast<long>(min_val),
             static_cast<long>(max_val));
    }
    TDLOGD("ParseModelJson: range_constraints=%zu", range_constraints.size());
  }
}

Status ParseGraphValueStores(const OrderedJson &graph, ValueStoreRefs &stores) {
  if (graph.contains("tensor_values")) {
    TD_CHK_STATUS(ParseTensorValues(graph["tensor_values"], stores.tensor_values));
    TDLOGD("ParseModelJson: tensor_values=%zu", stores.tensor_values.size());
  }
  if (graph.contains("sym_int_values")) {
    TD_CHK_STATUS(ParseSymIntValues(graph["sym_int_values"], stores.sym_int_values));
    TDLOGD("ParseModelJson: sym_int_values=%zu", stores.sym_int_values.size());
  }
  if (graph.contains("sym_float_values")) {
    TD_CHK_STATUS(ParseSymFloatValues(graph["sym_float_values"], stores.sym_float_values));
    TDLOGD("ParseModelJson: sym_float_values=%zu", stores.sym_float_values.size());
  }
  if (graph.contains("sym_bool_values")) {
    TD_CHK_STATUS(ParseSymBoolValues(graph["sym_bool_values"], stores.sym_bool_values));
    TDLOGD("ParseModelJson: sym_bool_values=%zu", stores.sym_bool_values.size());
  }
  if (graph.contains("int_values")) {
    TD_CHK_STATUS(ParseIntValues(graph["int_values"], stores.int_values));
    TDLOGD("ParseModelJson: int_values=%zu", stores.int_values.size());
  }
  if (graph.contains("float_values")) {
    TD_CHK_STATUS(ParseFloatValues(graph["float_values"], stores.float_values));
    TDLOGD("ParseModelJson: float_values=%zu", stores.float_values.size());
  }
  if (graph.contains("bool_values")) {
    TD_CHK_STATUS(ParseBoolValues(graph["bool_values"], stores.bool_values));
    TDLOGD("ParseModelJson: bool_values=%zu", stores.bool_values.size());
  }
  if (graph.contains("string_values")) {
    TD_CHK_STATUS(ParseStringValues(graph["string_values"], stores.string_values));
    TDLOGD("ParseModelJson: string_values=%zu", stores.string_values.size());
  }
  return SUCCESS;
}

Status ParseNodeInputs(const OrderedJson &node, size_t json_node_index, ParsedNode &parsed) {
  if (!node.contains("inputs") || !node["inputs"].is_array()) {
    return SUCCESS;
  }
  for (const auto &input : node["inputs"]) {
    ParsedNodeInput parsed_input;
    if (input.contains("name") && input["name"].is_string()) {
      parsed_input.name = input["name"].get<std::string>();
    }
    if (input.contains("kind")) {
      if (input["kind"].is_number_integer()) {
        parsed_input.kind = input["kind"].get<int64_t>();
      } else if (input["kind"].is_string()) {
        parsed_input.kind = -1;
      }
    }
    if (input.contains("arg")) {
      parsed_input.raw_arg = input["arg"];
      if (input["arg"].is_object() && input["arg"].contains("as_tensor") && input["arg"]["as_tensor"].is_object() &&
          input["arg"]["as_tensor"].contains("name") && input["arg"]["as_tensor"]["name"].is_string()) {
        parsed_input.input_kind = NodeInputKind::Tensor;
        parsed_input.tensor_name = input["arg"]["as_tensor"]["name"].get<std::string>();
        parsed.fx_input_tensor_names.push_back(parsed_input.tensor_name);
      } else if (input["arg"].is_object() && input["arg"].contains("as_tensors") &&
                 input["arg"]["as_tensors"].is_array()) {
        parsed_input.input_kind = NodeInputKind::TensorList;
        parsed_input.tensor_names = ParseTensorListValue(input["arg"]);
        if (parsed_input.tensor_names.empty()) {
          TDLOGE("ParseNode failed: tensor-list input missing names, node=%zu, target=%s", json_node_index,
                 parsed.target.c_str());
          return INVALID_PARAM;
        }
        parsed.fx_input_tensor_names.insert(parsed.fx_input_tensor_names.end(), parsed_input.tensor_names.begin(),
                                            parsed_input.tensor_names.end());
      } else if (input["arg"].is_object() && input["arg"].contains("as_operator") &&
                 input["arg"]["as_operator"].is_string()) {
        parsed_input.input_kind = NodeInputKind::Operator;
        parsed_input.operator_name = input["arg"]["as_operator"].get<std::string>();
      } else if (ParseSymOrScalarInputArg(input["arg"], parsed_input, parsed)) {
        // sym/scalar input handled
      }
    }
    parsed.inputs.push_back(std::move(parsed_input));
  }
  return SUCCESS;
}

bool TryParseNonTensorOutput(const OrderedJson &output, ParsedNode &parsed, ParsedNodeOutput &parsed_output) {
  static const std::vector<std::pair<const char *, const char *>> kSymFields = {
      {"as_sym_int", "as_name"}, {"as_sym_float", "as_name"}, {"as_sym_bool", "as_name"}};
  for (const auto &[sym_field, name_field] : kSymFields) {
    if (output.contains(sym_field) && output[sym_field].is_object() && output[sym_field].contains(name_field) &&
        output[sym_field][name_field].is_string()) {
      const std::string sym_name = output[sym_field][name_field].get<std::string>();
      parsed_output.is_tensor = true;
      parsed_output.tensor_name = sym_name;
      parsed.outputs.push_back(parsed_output);
      parsed.fx_output_tensor_names.push_back(sym_name);
      if (parsed.name.empty()) {
        parsed.name = sym_name;
      }
      return true;
    }
  }
  static const std::vector<const char *> kListSymFields = {"as_sym_ints", "as_sym_floats", "as_sym_bools"};
  for (const auto &field : kListSymFields) {
    if (output.contains(field) && output[field].is_array()) {
      for (const auto &item : output[field]) {
        if (item.is_object() && item.contains("as_name") && item["as_name"].is_string()) {
          parsed.fx_output_tensor_names.push_back(item["as_name"].get<std::string>());
        }
      }
      return true;
    }
  }
  static const std::vector<const char *> kSymConcreteFields = {"as_sym_int", "as_sym_float", "as_sym_bool"};
  for (const auto &field : kSymConcreteFields) {
    if (output.contains(field) && output[field].is_object() && !output[field].contains("as_name")) {
      return true;
    }
  }
  static const std::vector<const char *> kScalarFields = {"as_int",  "as_ints",  "as_float",  "as_floats",
                                                          "as_bool", "as_bools", "as_string", "as_strings"};
  for (const auto &field : kScalarFields) {
    if (output.contains(field)) {
      return true;
    }
  }
  return false;
}

Status ParseSingleNodeOutput(const OrderedJson &output, size_t json_node_index, const std::string &target,
                             ParsedNode &parsed) {
  if (output.is_object() && output.contains("as_none") && output["as_none"].is_boolean() &&
      output["as_none"].get<bool>()) {
    return SUCCESS;
  }
  ParsedNodeOutput parsed_output;
  if (output.is_object() && output.contains("as_tensor") && output["as_tensor"].is_object() &&
      output["as_tensor"].contains("name") && output["as_tensor"]["name"].is_string()) {
    const std::string output_name = output["as_tensor"]["name"].get<std::string>();
    parsed_output.is_tensor = true;
    parsed_output.tensor_name = output_name;
    parsed.outputs.push_back(parsed_output);
    parsed.fx_output_tensor_names.push_back(output_name);
    if (parsed.name.empty()) {
      parsed.name = output_name;
    }
    return SUCCESS;
  }
  if (output.is_object() && output.contains("as_tensors") && output["as_tensors"].is_array()) {
    parsed_output.is_tensor_list = true;
    parsed_output.tensor_names = ParseTensorListValue(output);
    if (parsed_output.tensor_names.empty()) {
      TDLOGE("ParseNode failed: tensor-list output missing names, node=%zu, target=%s", json_node_index,
             target.c_str());
      return INVALID_PARAM;
    }
    parsed.outputs.push_back(parsed_output);
    parsed.fx_output_tensor_names.insert(parsed.fx_output_tensor_names.end(), parsed_output.tensor_names.begin(),
                                         parsed_output.tensor_names.end());
    if (parsed.name.empty()) {
      parsed.name = parsed_output.tensor_names.front();
    }
    return SUCCESS;
  }
  if (output.is_object() && TryParseNonTensorOutput(output, parsed, parsed_output)) {
    return SUCCESS;
  }
  TDLOGE("ParseNode failed: unsupported output format, node=%zu, target=%s", json_node_index, target.c_str());
  return INVALID_PARAM;
}

Status ParseNodeOutputs(const OrderedJson &node, size_t json_node_index, const std::string &target,
                        ParsedNode &parsed) {
  if (!node.contains("outputs") || !node["outputs"].is_array()) {
    return SUCCESS;
  }
  for (const auto &output : node["outputs"]) {
    TD_CHK_STATUS(ParseSingleNodeOutput(output, json_node_index, target, parsed));
  }
  return SUCCESS;
}

Status PopulateNodeSourceAndName(ParsedNode &parsed, std::unordered_map<std::string, size_t> &fallback_name_counters) {
  TD_CHK_STATUS(PopulateEffectiveSchemaInfo(parsed));
  parsed.source_info.source = parsed.ascend_ir.has_value() ? parsed.ascend_ir->source : std::string();
  parsed.source_info.is_converter = parsed.ascend_ir.has_value() && parsed.ascend_ir->source == "converter";
  parsed.source_info.is_fallback = IsFallbackNode(parsed);
  std::optional<std::string> fallback_op_type;
  TD_CHK_STATUS(BuildFallbackOpType(parsed, fallback_op_type));
  if (fallback_op_type.has_value()) {
    parsed.name = BuildUniqueFallbackNodeName(fallback_op_type.value(), fallback_name_counters);
    TDLOGD("ParseNode: fallback node name=%s, type=%s", parsed.name.c_str(), fallback_op_type.value().c_str());
  } else {
    parsed.name = BuildNodeName(parsed);
  }
  return SUCCESS;
}

void PopulateNodeBoundaries(ParsedNode &parsed) {
  if (!parsed.ascend_ir.has_value()) {
    return;
  }
  parsed.ascend_ir->boundary_outputs.clear();
  for (size_t index = 0; index < parsed.fx_output_tensor_names.size(); ++index) {
    const std::string &fx_output_name = parsed.fx_output_tensor_names[index];
    auto output_it = parsed.ascend_ir->output_mappings.find(fx_output_name);
    if (output_it == parsed.ascend_ir->output_mappings.end()) {
      output_it = parsed.ascend_ir->output_mappings.find(std::to_string(index));
    }
    if (output_it != parsed.ascend_ir->output_mappings.end()) {
      parsed.ascend_ir->boundary_outputs.push_back({fx_output_name, output_it->second});
    }
  }
  for (const auto &input : parsed.inputs) {
    const std::vector<std::string> tensor_names = InputTensorNames(input);
    if (tensor_names.empty()) {
      continue;
    }
    for (const auto &tensor_name : tensor_names) {
      ParsedBoundaryInput boundary_input;
      boundary_input.fx_tensor_name = tensor_name;
      boundary_input.schema_input_name = input.name;
      const auto input_it = parsed.ascend_ir->input_mappings.find(tensor_name);
      boundary_input.ge_tensor_name =
          input_it == parsed.ascend_ir->input_mappings.end() ? tensor_name : input_it->second;
      parsed.ascend_ir->boundary_inputs.push_back(std::move(boundary_input));
    }
  }
}

Status ParseNodesArray(const OrderedJson &graph, std::vector<ParsedNode> &nodes) {
  if (!graph.contains("nodes") || !graph["nodes"].is_array()) {
    return SUCCESS;
  }
  TDLOGI("ParseModelJson step: parsing nodes, count=%zu", graph["nodes"].size());
  std::unordered_map<std::string, size_t> fallback_node_name_counters;
  for (const auto &node : graph["nodes"]) {
    if (!node.is_object()) {
      TDLOGE("ParseModelJson: node[%zu] is not object", nodes.size());
      return INVALID_PARAM;
    }
    ParsedNode parsed_node;
    const Status node_status = ParseNodeJsonImpl(node, nodes.size(), fallback_node_name_counters, parsed_node);
    if (node_status != SUCCESS) {
      TDLOGE("ParseModelJson: ParseNodeJson failed for node[%zu], status=%lld", nodes.size(),
             static_cast<long long>(node_status));
      return node_status;
    }
    nodes.push_back(std::move(parsed_node));
  }
  TDLOGI("ParseModelJson: nodes parsed=%zu", nodes.size());
  return SUCCESS;
}

Status ParseNodeJsonImpl(const OrderedJson &node, size_t json_node_index,
                         std::unordered_map<std::string, size_t> &fallback_name_counters, ParsedNode &out) {
  out = ParsedNode{};
  ParsedNode &parsed = out;
  parsed.json_node_index = json_node_index;
  if (!node.is_object()) {
    TDLOGE("ParseNode failed: node[%zu] is not object", json_node_index);
    return INVALID_PARAM;
  }
  if (node.contains("target") && node["target"].is_string()) {
    parsed.target = node["target"].get<std::string>();
  }
  TDLOGD("ParseNode: index=%zu, target=%s", json_node_index, parsed.target.c_str());
  if (node.contains("metadata")) {
    parsed.metadata = node["metadata"];
  }
  TD_CHK_STATUS(ParseNodeInputs(node, json_node_index, parsed));
  TD_CHK_STATUS(ParseNodeOutputs(node, json_node_index, parsed.target, parsed));
  if (node.contains("ascend_ir")) {
    ParsedAscendIr ascend_ir;
    const Status ir_status = ParseAscendIrJsonImpl(node["ascend_ir"], parsed, ascend_ir);
    if (ir_status == SUCCESS) {
      parsed.ascend_ir = std::move(ascend_ir);
      TDLOGD("ParseNode: ascend_ir parsed, source=%s, ops=%zu", parsed.ascend_ir->source.c_str(),
             parsed.ascend_ir->ops.size());
    } else {
      return ir_status;
    }
  }
  TD_CHK_STATUS(PopulateNodeSourceAndName(parsed, fallback_name_counters));
  if (parsed.functionalize.runtime_input_tensor_names.empty()) {
    parsed.functionalize.runtime_input_tensor_names = parsed.fx_input_tensor_names;
  }
  if (parsed.functionalize.runtime_output_tensor_names.empty()) {
    parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
  }
  PopulateNodeBoundaries(parsed);
  return SUCCESS;
}

Status ParseAscendIrJsonImpl(const OrderedJson &ascend_ir, const ParsedNode &parsed_node, ParsedAscendIr &out) {
  const Status status = ParseAscendIrImpl(ascend_ir, BuildNodeJsonContext(parsed_node), out);
  if (status != SUCCESS) {
    TDLOGE("ParseAscendIrJson failed: context=%s", BuildNodeJsonContext(parsed_node).c_str());
    return status;
  }
  return SUCCESS;
}

}  // namespace

Status ParseEpairModelImpl(const OrderedJson &root, ParsedModel &out) {
  TDLOGI("ParseModelJson enter");
  out = ParsedModel{};
  ParseModelMetadata(root, out.schema_version_, out.opset_version_, out.torch_version_, out.range_constraints_);
  if (!root.contains("graph_module") || !root["graph_module"].is_object()) {
    TDLOGE("ParseModelJson failed: graph_module missing or invalid");
    return INVALID_PARAM;
  }
  const auto &graph_module = root["graph_module"];
  if (!graph_module.contains("graph") || !graph_module["graph"].is_object()) {
    TDLOGE("ParseModelJson failed: graph_module.graph missing or invalid");
    return INVALID_PARAM;
  }
  TDLOGI("ParseModelJson step: parsing graph");
  const auto &graph = graph_module["graph"];
  if (graph_module.contains("signature") && graph_module["signature"].is_object()) {
    TDLOGD("ParseModelJson step: parsing signature");
    const auto &signature = graph_module["signature"];
    TD_CHK_STATUS(ParseInputSpecs(signature, out.input_specs_));
    TDLOGD("ParseModelJson: input_specs=%zu", out.input_specs_.size());
    TD_CHK_STATUS(ParseOutputSpecs(signature, out.output_specs_));
    TDLOGD("ParseModelJson: output_specs=%zu", out.output_specs_.size());
  }
  if (graph.contains("inputs")) {
    TD_CHK_STATUS(ParseGraphInputs(graph["inputs"], out.input_specs_, out.graph_inputs_));
    TDLOGD("ParseModelJson: graph_inputs=%zu", out.graph_inputs_.size());
  }
  if (graph.contains("outputs")) {
    TD_CHK_STATUS(ParseGraphOutputs(graph["outputs"], out.output_specs_, out.graph_outputs_));
    TDLOGD("ParseModelJson: graph_outputs=%zu", out.graph_outputs_.size());
  }
  ValueStoreRefs stores{out.tensor_values_, out.sym_int_values_, out.sym_float_values_, out.sym_bool_values_,
                        out.int_values_,    out.float_values_,   out.bool_values_,      out.string_values_};
  TD_CHK_STATUS(ParseGraphValueStores(graph, stores));
  TD_CHK_STATUS(ParseNodesArray(graph, out.nodes_));
  TDLOGI("ParseModelJson exit: success");
  return SUCCESS;
}

Status ParsedModel::Parse(const std::string &model_json, ParsedModel &out) {
  out = ParsedModel{};
  OrderedJson root;
  Status status = ParseRootJsonImpl(model_json, root);
  if (status != SUCCESS) {
    TDLOGE("Parse failed: ParseRootJson returned %s", StatusName(status));
    return status;
  }
  status = ParseEpairModelImpl(root, out);
  if (status != SUCCESS) {
    TDLOGE("Parse failed: ParseModelJson returned %s", StatusName(status));
    return status;
  }
  for (size_t node_index = 0; node_index < out.nodes_.size(); ++node_index) {
    const ParsedNode &node = out.nodes_[node_index];
    out.node_name_index_[node.name] = node_index;
    for (const auto &output : node.outputs) {
      for (const auto &tensor_name : OutputTensorNames(output)) {
        out.output_name_index_[tensor_name] = node_index;
      }
    }
  }
  for (size_t input_index = 0; input_index < out.input_specs_.size(); ++input_index) {
    const ParsedInputSpec &spec = out.input_specs_[input_index];
    if (!spec.placeholder_name.empty()) {
      out.input_spec_index_[spec.placeholder_name] = input_index;
    }
  }
  for (size_t output_index = 0; output_index < out.output_specs_.size(); ++output_index) {
    const ParsedOutputSpec &spec = out.output_specs_[output_index];
    if (!spec.name.empty()) {
      out.output_spec_index_[spec.name] = output_index;
    }
  }
  return SUCCESS;
}

const SchemaVersion &ParsedModel::SchemaVersion() const {
  return schema_version_;
}

const std::vector<ParsedValueRef> &ParsedModel::GraphInputs() const {
  return graph_inputs_;
}

const std::vector<ParsedValueRef> &ParsedModel::GraphOutputs() const {
  return graph_outputs_;
}

const std::vector<ParsedNode> &ParsedModel::Nodes() const {
  return nodes_;
}

const std::vector<ParsedInputSpec> &ParsedModel::InputSpecs() const {
  return input_specs_;
}

const ParsedNode *ParsedModel::FindNodeByName(const std::string &name) const {
  const auto it = node_name_index_.find(name);
  if (it == node_name_index_.end()) {
    return nullptr;
  }
  return &nodes_[it->second];
}

const ParsedNode *ParsedModel::FindNodeByOutput(const std::string &output_name) const {
  const auto it = output_name_index_.find(output_name);
  if (it == output_name_index_.end()) {
    return nullptr;
  }
  return &nodes_[it->second];
}

const ParsedTensorMeta *ParsedModel::FindTensorMeta(const std::string &name) const {
  const auto it = tensor_values_.find(name);
  if (it == tensor_values_.end()) {
    return nullptr;
  }
  return &it->second;
}

const ParsedInputSpec *ParsedModel::FindInputSpec(const std::string &placeholder_name) const {
  const auto it = input_spec_index_.find(placeholder_name);
  if (it == input_spec_index_.end()) {
    return nullptr;
  }
  return &input_specs_[it->second];
}

const ParsedOutputSpec *ParsedModel::FindOutputSpec(const std::string &output_name) const {
  const auto it = output_spec_index_.find(output_name);
  if (it == output_spec_index_.end()) {
    return nullptr;
  }
  return &output_specs_[it->second];
}

std::optional<int64_t> ParsedModel::FindSymInt(const std::string &name) const {
  const auto it = sym_int_values_.find(name);
  if (it == sym_int_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<double> ParsedModel::FindSymFloat(const std::string &name) const {
  const auto it = sym_float_values_.find(name);
  if (it == sym_float_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<bool> ParsedModel::FindSymBool(const std::string &name) const {
  const auto it = sym_bool_values_.find(name);
  if (it == sym_bool_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<int64_t> ParsedModel::FindInt(const std::string &name) const {
  const auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<double> ParsedModel::FindFloat(const std::string &name) const {
  const auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<bool> ParsedModel::FindBool(const std::string &name) const {
  const auto it = bool_values_.find(name);
  if (it == bool_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> ParsedModel::FindString(const std::string &name) const {
  const auto it = string_values_.find(name);
  if (it == string_values_.end()) {
    return std::nullopt;
  }
  return it->second;
}

PlaceholderKind ParsedModel::PlaceholderKindOf(const std::string &name) const {
  const ParsedInputSpec *spec = FindInputSpec(name);
  return spec == nullptr ? PlaceholderKind::Unknown : spec->kind;
}

std::string ParsedModel::LogicalNameOf(const std::string &name) const {
  const ParsedInputSpec *spec = FindInputSpec(name);
  return spec == nullptr ? std::string() : spec->logical_name;
}

const std::pair<int64_t, int64_t> *ParsedModel::FindRangeConstraint(const std::string &symbol) const {
  const auto it = range_constraints_.find(symbol);
  if (it == range_constraints_.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace td
