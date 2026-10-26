/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/node_naming.h"

#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/parse_functionalized.h"
#include "common/json_parse_utils.h"
#include "common/logging/logging.h"

namespace td {

std::vector<std::string> OutputTensorNames(const ParsedNodeOutput &output);

namespace {

std::string CollapseUnderscores(const std::string &value) {
  std::string normalized;
  normalized.reserve(value.size());
  bool last_was_underscore = false;
  for (const char ch : value) {
    if (ch == '_') {
      if (!last_was_underscore) {
        normalized.push_back(ch);
      }
      last_was_underscore = true;
      continue;
    }
    normalized.push_back(ch);
    last_was_underscore = false;
  }
  while (!normalized.empty() && normalized.front() == '_') {
    normalized.erase(normalized.begin());
  }
  while (!normalized.empty() && normalized.back() == '_') {
    normalized.pop_back();
  }
  return normalized;
}

std::string SanitizeNodeStem(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalnum(ch)) {
      sanitized.push_back(static_cast<char>(ch));
    } else {
      sanitized.push_back('_');
    }
  }
  sanitized = CollapseUnderscores(sanitized);
  if (sanitized.empty()) {
    return "unknown";
  }
  constexpr size_t kMaxStemLength = 64;
  if (sanitized.size() > kMaxStemLength) {
    sanitized.resize(kMaxStemLength);
    sanitized = CollapseUnderscores(sanitized);
  }
  return sanitized.empty() ? "unknown" : sanitized;
}

}  // namespace

std::string SelectNodeStem(const ParsedNode &parsed) {
  if (!parsed.effective_schema_target.empty()) {
    return SanitizeNodeStem(StripTorchOpsPrefix(parsed.effective_schema_target));
  }
  if (!parsed.target.empty()) {
    return SanitizeNodeStem(StripTorchOpsPrefix(parsed.target));
  }
  if (!parsed.outputs.empty()) {
    const std::vector<std::string> output_names = OutputTensorNames(parsed.outputs.front());
    if (!output_names.empty()) {
      return SanitizeNodeStem(output_names.front());
    }
  }
  return "unknown";
}

std::string BuildNodeName(const ParsedNode &parsed) {
  return "node_" + std::to_string(parsed.json_node_index) + "_" + SelectNodeStem(parsed);
}

std::string BuildNodeJsonContext(const ParsedNode &node) {
  return "graph_module.graph.nodes[" + std::to_string(node.json_node_index) + "]";
}

bool IsFallbackNode(const ParsedNode &node) {
  return node.ascend_ir.has_value() && node.ascend_ir->source != "converter" && !node.effective_schema_target.empty();
}

Status BuildFallbackOpType(const ParsedNode &node, std::optional<std::string> &out) {
  out.reset();
  if (!IsFallbackNode(node)) {
    return SUCCESS;
  }

  std::string op_str = node.effective_schema_target;
  constexpr char kPrefix[] = "torch.ops.";
  if (op_str.rfind(kPrefix, 0) == 0) {
    op_str = op_str.substr(sizeof(kPrefix) - 1);
  }

  const size_t overload_separator = op_str.find_last_of('.');
  if (overload_separator == std::string::npos) {
    TDLOGE("Target missing overload separator, context=%s", BuildNodeJsonContext(node).c_str());
    return INVALID_PARAM;
  }

  const std::string overload = op_str.substr(overload_separator + 1);
  const std::string ns_and_op = op_str.substr(0, overload_separator);
  const size_t namespace_separator = ns_and_op.find_last_of('.');
  if (namespace_separator == std::string::npos) {
    TDLOGE("Target missing namespace separator, context=%s", BuildNodeJsonContext(node).c_str());
    return INVALID_PARAM;
  }

  const std::string op_name = ns_and_op.substr(namespace_separator + 1);
  std::string op_type = ToCamelCase(op_name);
  if (overload == "out") {
    op_type += "Out";
  } else if (!op_name.empty() && op_name.back() == '_') {
    op_type += "Inplace";
  }
  if (op_type.empty()) {
    TDLOGE("Target op_name is empty, context=%s", BuildNodeJsonContext(node).c_str());
    return INVALID_PARAM;
  }
  out = op_type;
  return SUCCESS;
}

std::string BuildUniqueFallbackNodeName(const std::string &op_type, std::unordered_map<std::string, size_t> &counters) {
  const size_t next_index = counters[op_type]++;
  if (next_index == 0) {
    return op_type;
  }
  return op_type + "_" + std::to_string(next_index);
}

}  // namespace td
