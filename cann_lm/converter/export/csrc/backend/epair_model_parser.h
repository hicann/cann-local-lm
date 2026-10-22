#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/logging/logging.h"
#include "backend/model_types.h"

namespace td {

class EpairModelParser {
 public:
  static EpairModelParser Parse(const std::string &model_json, ValidationResult &result);

  static OrderedJson ParseRootJson(const std::string &model_json, ValidationResult &result);

  static ParsedModel ParseModelJson(const OrderedJson &root, ValidationResult &result);

  static ParsedNode ParseNodeJson(const OrderedJson &node, size_t json_node_index,
                                  std::unordered_map<std::string, size_t> &fallback_name_counters,
                                  ValidationResult &result);

  static std::optional<ParsedAscendIr> ParseAscendIrJson(const OrderedJson &ascend_ir, const ParsedNode &parsed_node,
                                                         ValidationResult &result);

  const ParsedModel &Model() const;
  const SchemaVersion &SchemaVersionValue() const;
  const std::vector<ParsedValueRef> &GraphInputs() const;
  const std::vector<ParsedValueRef> &GraphOutputs() const;
  const std::vector<ParsedNode> &Nodes() const;

  const ParsedNode *FindNodeByName(const std::string &name) const;
  const ParsedNode *FindNodeByOutput(const std::string &output_name) const;
  const ParsedTensorMeta *FindTensorValue(const std::string &name) const;
  const ParsedInputSpec *FindInputSpec(const std::string &placeholder_name) const;
  const ParsedOutputSpec *FindOutputSpec(const std::string &output_name) const;
  const ParsedAscendIr *FindAscendIr(const std::string &node_name) const;

  // Find different value types
  std::optional<int64_t> FindSymIntValue(const std::string &name) const;
  std::optional<double> FindSymFloatValue(const std::string &name) const;
  std::optional<bool> FindSymBoolValue(const std::string &name) const;
  std::optional<int64_t> FindIntValue(const std::string &name) const;
  std::optional<double> FindFloatValue(const std::string &name) const;
  std::optional<bool> FindBoolValue(const std::string &name) const;
  std::optional<std::string> FindStringValue(const std::string &name) const;

  PlaceholderKind PlaceholderKindOf(const std::string &name) const;
  std::string LogicalNameOfPlaceholder(const std::string &name) const;

 private:
  ParsedModel model_;
  std::unordered_map<std::string, size_t> node_name_to_index_;
  std::unordered_map<std::string, size_t> output_name_to_index_;
  std::unordered_map<std::string, size_t> input_spec_to_index_;
  std::unordered_map<std::string, size_t> output_spec_to_index_;
};

std::string NodeDebugContext(const ParsedNode &node);

}  // namespace td
