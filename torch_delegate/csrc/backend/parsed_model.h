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
#include <unordered_map>
#include <vector>

#include "backend/model_types.h"
#include "torch_delegate/td_types.h"

namespace td {

class ParsedModel {
 public:
  static Status Parse(const std::string &json, ParsedModel &out);

  auto SchemaVersion() const -> const ::td::SchemaVersion &;
  const std::vector<ParsedValueRef> &GraphInputs() const;
  const std::vector<ParsedValueRef> &GraphOutputs() const;
  const std::vector<ParsedNode> &Nodes() const;
  const std::vector<ParsedInputSpec> &InputSpecs() const;

  const ParsedTensorMeta *FindTensorMeta(const std::string &name) const;
  std::optional<int64_t> FindSymInt(const std::string &name) const;
  std::optional<double> FindSymFloat(const std::string &name) const;
  std::optional<bool> FindSymBool(const std::string &name) const;
  std::optional<int64_t> FindInt(const std::string &name) const;
  std::optional<double> FindFloat(const std::string &name) const;
  std::optional<bool> FindBool(const std::string &name) const;
  std::optional<std::string> FindString(const std::string &name) const;

  const ParsedNode *FindNodeByName(const std::string &name) const;
  const ParsedNode *FindNodeByOutput(const std::string &output_name) const;

  const ParsedInputSpec *FindInputSpec(const std::string &placeholder_name) const;
  const ParsedOutputSpec *FindOutputSpec(const std::string &output_name) const;
  PlaceholderKind PlaceholderKindOf(const std::string &name) const;
  std::string LogicalNameOf(const std::string &name) const;

  const std::pair<int64_t, int64_t> *FindRangeConstraint(const std::string &symbol) const;

 private:
  ::td::SchemaVersion schema_version_;
  OrderedJson opset_version_;
  std::string torch_version_;
  std::vector<ParsedValueRef> graph_inputs_;
  std::vector<ParsedValueRef> graph_outputs_;
  std::vector<ParsedNode> nodes_;
  std::unordered_map<std::string, ParsedTensorMeta> tensor_values_;
  std::vector<ParsedInputSpec> input_specs_;
  std::vector<ParsedOutputSpec> output_specs_;
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> range_constraints_;

  std::unordered_map<std::string, int64_t> sym_int_values_;
  std::unordered_map<std::string, double> sym_float_values_;
  std::unordered_map<std::string, bool> sym_bool_values_;
  std::unordered_map<std::string, int64_t> int_values_;
  std::unordered_map<std::string, double> float_values_;
  std::unordered_map<std::string, bool> bool_values_;
  std::unordered_map<std::string, std::string> string_values_;

  std::unordered_map<std::string, size_t> node_name_index_;
  std::unordered_map<std::string, size_t> output_name_index_;
  std::unordered_map<std::string, size_t> input_spec_index_;
  std::unordered_map<std::string, size_t> output_spec_index_;

  friend Status ParseEpairModelImpl(const OrderedJson &root, ParsedModel &out);
};

}  // namespace td
