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
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/pt2_type_constants.h"

namespace td {

using OrderedJson = nlohmann::ordered_json;

enum class PlaceholderKind {
  UserInput,
  Parameter,
  Buffer,
  TensorConstant,
  ConstantInput,
  Unknown,
};

struct SchemaVersion {
  int64_t major = 0;
  int64_t minor = 0;
};

struct ParsedDevice {
  std::string type;
  std::optional<int64_t> index;
};

struct ParsedTensorMeta {
  int64_t dtype = 0;
  std::vector<int64_t> sizes;
  bool requires_grad = false;
  ParsedDevice device;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
  int64_t layout = 0;
  std::vector<std::string> dynamic_dim_symbols;
};

struct ParsedValueRef {
  std::string name;
};

struct ParsedNodeInput {
  std::string name;
  int64_t kind = -1;
  NodeInputKind input_kind = NodeInputKind::Tensor;
  std::string tensor_name;
  std::vector<std::string> tensor_names;
  std::string operator_name;
  OrderedJson raw_arg;
};

struct ParsedNodeOutput {
  bool is_tensor = false;
  bool is_tensor_list = false;
  std::string tensor_name;
  std::vector<std::string> tensor_names;
};

struct ParsedBoundaryOutput {
  std::string fx_tensor_name;
  std::string ge_tensor_name;
};

struct ParsedBoundaryInput {
  std::string fx_tensor_name;
  std::string schema_input_name;
  std::string ge_tensor_name;
};

struct ParsedFunctionalizeInfo {
  bool is_functionalized = false;
  std::string wrapped_target;
  size_t metadata_input_count = 0;
  std::vector<std::string> runtime_input_tensor_names;
  std::vector<std::string> runtime_output_tensor_names;
};

struct ParsedNodeSource {
  std::string source;
  bool is_converter = false;
  bool is_fallback = false;
};

struct ParsedAscendIrOpDesc {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string layout;
  std::string device_type;
  OrderedJson attr;
  OrderedJson raw_json;
};

struct ParsedAscendIrOp {
  std::string name;
  std::string type;
  std::vector<std::string> inputs;
  OrderedJson attrs;
  std::vector<ParsedAscendIrOpDesc> input_descs;
  std::vector<ParsedAscendIrOpDesc> output_descs;
  OrderedJson raw_json;
};

struct ParsedAscendIr {
  std::string source;
  // 执行 Device 归属："npu"（缺省）| "cpu"；由导出期按 Dispatcher kernel
  // 注册情况自动判定并固化，缺失时按 npu 兼容老 epair。CPU 节点判定统一
  // 用 device == "cpu"（IsCpuNode），不再另设派生布尔字段。
  std::string device = "npu";
  std::vector<ParsedAscendIrOp> ops;
  std::unordered_map<std::string, std::string> input_mappings;
  std::unordered_map<std::string, std::string> output_mappings;
  std::vector<ParsedBoundaryOutput> boundary_outputs;
  std::vector<ParsedBoundaryInput> boundary_inputs;
  OrderedJson raw_json;
};

struct ParsedNode {
  std::string name;
  size_t json_node_index = 0;
  std::string target;
  std::string effective_schema_target;
  size_t schema_metadata_input_count = 0;
  std::vector<ParsedNodeInput> inputs;
  std::vector<ParsedNodeOutput> outputs;
  std::vector<std::string> fx_input_tensor_names;
  std::vector<std::string> fx_output_tensor_names;
  OrderedJson metadata;
  ParsedFunctionalizeInfo functionalize;
  ParsedNodeSource source_info;
  std::optional<ParsedAscendIr> ascend_ir;
};

struct ParsedInputSpec {
  PlaceholderKind kind = PlaceholderKind::Unknown;
  std::string placeholder_name;
  std::string logical_name;
  bool persistent = false;
  NodeInputKind input_kind = NodeInputKind::Tensor;
  OrderedJson raw_json;
};

struct ParsedOutputSpec {
  std::string name;
  OrderedJson raw_json;
};

using TensorData =
    std::variant<std::monostate, std::vector<double>, std::vector<float>, std::vector<uint16_t>, std::vector<int64_t>,
                 std::vector<int32_t>, std::vector<int16_t>, std::vector<int8_t>, std::vector<uint8_t>>;

struct TensorHolder {
  std::vector<int64_t> dims;
  int64_t datatype = 0;
  int64_t format = 0;
  TensorData value;
  bool requires_grad = false;

  size_t ByteSize() const {
    return std::visit(
        [](const auto &values) -> size_t {
          using ValueType = std::decay_t<decltype(values)>;
          if constexpr (std::is_same_v<ValueType, std::monostate>) {
            return 0;
          } else {
            return values.size() * sizeof(typename ValueType::value_type);
          }
        },
        value);
  }

  bool HasValue() const {
    return !std::holds_alternative<std::monostate>(value);
  }
};

struct TensorRecordInfo {
  std::string logical_name;
  std::string record_path;
  bool is_parameter = false;
  bool use_pickle = false;
  ParsedTensorMeta tensor_meta;
};

}  // namespace td
