/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "executorch_npu/manifest.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace executorch_npu {
namespace {

using Json = nlohmann::json;

constexpr char kMagic[] = "ASCD";
constexpr uint16_t kFramingVersion = 1;
constexpr uint32_t kRuntimeProtocolVersion = 1;
constexpr uint32_t kArtifactAbi = 1;
constexpr size_t kWeightAlignment = 32;

class ManifestParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void Fail(std::string message) {
  throw ManifestParseError(std::move(message));
}

void SetError(std::string message, std::string *error) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

uint16_t ReadLittleEndian16(const uint8_t *data) {
  return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8U);
}

uint32_t ReadLittleEndian32(const uint8_t *data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8U) |
         (static_cast<uint32_t>(data[2]) << 16U) | (static_cast<uint32_t>(data[3]) << 24U);
}

bool IsFullSha256(const std::string &value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
  });
}

bool IsBasename(const std::string &value) {
  return !value.empty() && value != "." && value != ".." && value.find('/') == std::string::npos &&
         value.find('\\') == std::string::npos;
}

bool IsSupportedDtype(const std::string &value) {
  static const std::set<std::string> supported = {
      "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64",
  };
  return supported.count(value) != 0;
}

size_t SizeValue(const Json &value, const std::string &path) {
  if (!value.is_number_unsigned()) {
    Fail(path + ": expected a non-negative integer");
  }
  const uint64_t raw = value.get<uint64_t>();
  if (raw > std::numeric_limits<size_t>::max()) {
    Fail(path + ": integer exceeds size_t");
  }
  return static_cast<size_t>(raw);
}

int64_t Int64Value(const Json &value, const std::string &path) {
  if (value.is_number_unsigned()) {
    const uint64_t raw = value.get<uint64_t>();
    if (raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      Fail(path + ": integer exceeds int64_t");
    }
    return static_cast<int64_t>(raw);
  }
  if (!value.is_number_integer()) {
    Fail(path + ": expected an integer");
  }
  return value.get<int64_t>();
}

class ObjectReader {
 public:
  ObjectReader(const Json &value, std::string path) : value_(value), path_(std::move(path)) {
    if (!value_.is_object()) {
      Fail(path_ + ": expected an object");
    }
  }

  const Json &Object(std::string_view key) {
    const Json &value = Take(key);
    if (!value.is_object()) {
      Fail(FieldPath(key) + ": expected an object");
    }
    return value;
  }

  const Json &Array(std::string_view key) {
    const Json &value = Take(key);
    if (!value.is_array()) {
      Fail(FieldPath(key) + ": expected an array");
    }
    return value;
  }

  const Json *OptionalArray(std::string_view key) {
    const auto iterator = value_.find(std::string(key));
    if (iterator == value_.end()) {
      return nullptr;
    }
    if (!iterator->is_array()) {
      Fail(FieldPath(key) + ": expected an array");
    }
    return &*iterator;
  }

  std::string String(std::string_view key) {
    const Json &value = Take(key);
    if (!value.is_string()) {
      Fail(FieldPath(key) + ": expected a string");
    }
    return value.get<std::string>();
  }

  std::string NonEmptyString(std::string_view key) {
    std::string value = String(key);
    if (value.empty()) {
      Fail(FieldPath(key) + ": must not be empty");
    }
    return value;
  }

  std::string Sha256(std::string_view key) {
    std::string value = String(key);
    if (!IsFullSha256(value)) {
      Fail(FieldPath(key) + ": expected a lowercase SHA-256 digest");
    }
    return value;
  }

  std::string Enum(std::string_view key, std::initializer_list<std::string_view> allowed) {
    std::string value = String(key);
    if (std::find(allowed.begin(), allowed.end(), value) == allowed.end()) {
      Fail(FieldPath(key) + ": unsupported value " + value);
    }
    return value;
  }

  std::optional<std::string> OptionalNonEmptyString(std::string_view key) {
    const auto iterator = value_.find(std::string(key));
    if (iterator == value_.end()) {
      return std::nullopt;
    }
    if (!iterator->is_string() || iterator->get_ref<const std::string &>().empty()) {
      Fail(FieldPath(key) + ": expected a non-empty string");
    }
    return iterator->get<std::string>();
  }

  size_t Size(std::string_view key) {
    return SizeValue(Take(key), FieldPath(key));
  }

  size_t PositiveSize(std::string_view key) {
    const size_t value = Size(key);
    if (value == 0) {
      Fail(FieldPath(key) + ": must be positive");
    }
    return value;
  }

  int64_t Int64(std::string_view key) {
    return Int64Value(Take(key), FieldPath(key));
  }

  bool Bool(std::string_view key) {
    const Json &value = Take(key);
    if (!value.is_boolean()) {
      Fail(FieldPath(key) + ": expected a boolean");
    }
    return value.get<bool>();
  }

 private:
  const Json &Take(std::string_view key) const {
    const auto iterator = value_.find(std::string(key));
    if (iterator == value_.end()) {
      Fail(FieldPath(key) + ": required field is missing");
    }
    return *iterator;
  }

  std::string FieldPath(std::string_view key) const {
    return path_ + "." + std::string(key);
  }

  const Json &value_;
  std::string path_;
};

std::string ItemPath(const std::string &path, size_t index) {
  return path + "[" + std::to_string(index) + "]";
}

std::vector<int64_t> ParseShape(const Json &json, const std::string &path, int64_t minimum) {
  std::vector<int64_t> shape;
  shape.reserve(json.size());
  for (size_t index = 0; index < json.size(); ++index) {
    const int64_t dimension = Int64Value(json[index], ItemPath(path, index));
    if (dimension < minimum) {
      Fail(ItemPath(path, index) + ": dimension is out of range");
    }
    shape.push_back(dimension);
  }
  return shape;
}

void ParseRuntimeProtocol(const Json &json) {
  ObjectReader fields(json, "manifest.runtime_protocol");
  const std::string name = fields.String("name");
  const size_t version = fields.Size("version");
  const size_t artifact_abi = fields.Size("artifact_abi");
  if (name != "AscendBackend" || version != kRuntimeProtocolVersion || artifact_abi != kArtifactAbi) {
    Fail("manifest.runtime_protocol: incompatible protocol or artifact ABI");
  }
}

ModelArtifact ParseModel(const Json &json) {
  ObjectReader fields(json, "manifest.model");
  ModelArtifact model;
  model.named_data_key = fields.NonEmptyString("named_data_key");
  model.sha256 = fields.Sha256("sha256");
  model.nbytes = fields.PositiveSize("nbytes");
  return model;
}

void ParseWeightGroups(const Json &json, Manifest *manifest, std::unordered_map<std::string, size_t> *group_sizes) {
  if (json.empty()) {
    Fail("manifest.weight_groups: must not be empty");
  }
  for (size_t index = 0; index < json.size(); ++index) {
    const std::string path = ItemPath("manifest.weight_groups", index);
    ObjectReader fields(json[index], path);
    WeightGroup group;
    group.key = fields.NonEmptyString("weight_group_key");
    group.sha256 = fields.Sha256("sha256");
    group.nbytes = fields.PositiveSize("total_nbytes");
    if (group.key != "ascend.weights." + group.sha256 || !group_sizes->emplace(group.key, group.nbytes).second) {
      Fail(path + ": key is not content-addressed or is duplicated");
    }
    manifest->weight_groups.push_back(std::move(group));
  }
}

void ParseFileConstants(const Json &json, const std::unordered_map<std::string, size_t> &group_sizes,
                        Manifest *manifest) {
  if (json.empty()) {
    Fail("manifest.file_constant_bindings: must not be empty");
  }
  std::unordered_map<std::string, FileConstantBinding> binding_by_id;
  for (size_t index = 0; index < json.size(); ++index) {
    const std::string path = ItemPath("manifest.file_constant_bindings", index);
    ObjectReader fields(json[index], path);
    FileConstantBinding binding;
    binding.binding_id = fields.NonEmptyString("binding_id");
    binding.file_name = fields.NonEmptyString("file_name");
    binding.weight_group_key = fields.NonEmptyString("weight_group_key");
    binding.offset = fields.Size("offset");
    binding.nbytes = fields.PositiveSize("nbytes");
    const Json &shape = fields.Array("shape");
    const std::optional<std::string> alias = fields.OptionalNonEmptyString("alias_of");
    binding.alias_of = alias.value_or("");

    const auto group = group_sizes.find(binding.weight_group_key);
    if (!IsBasename(binding.file_name) || group == group_sizes.end() || binding.offset % kWeightAlignment != 0 ||
        binding.offset > group->second || binding.nbytes > group->second - binding.offset || shape.size() != 1 ||
        SizeValue(shape[0], path + ".shape[0]") != binding.nbytes || binding_by_id.count(binding.binding_id) != 0) {
      Fail(path + ": identity or byte range is invalid");
    }
    if (alias.has_value()) {
      const auto canonical = binding_by_id.find(*alias);
      if (canonical == binding_by_id.end() || !canonical->second.alias_of.empty() ||
          canonical->second.weight_group_key != binding.weight_group_key ||
          canonical->second.offset != binding.offset || canonical->second.nbytes != binding.nbytes) {
        Fail(path + ".alias_of: alias is invalid");
      }
    }
    binding_by_id.emplace(binding.binding_id, binding);
    manifest->file_constant_bindings.push_back(std::move(binding));
  }
}

void ParseShapeRange(const Json *json, const std::vector<int64_t> &shape, const std::string &path,
                     std::vector<DimensionRange> *output) {
  const bool dynamic = std::find(shape.begin(), shape.end(), -1) != shape.end();
  if (dynamic != (json != nullptr)) {
    Fail(path + ": required exactly for dynamic shapes");
  }
  if (json == nullptr) {
    return;
  }
  if (json->size() != shape.size()) {
    Fail(path + ": must match the shape rank");
  }
  for (size_t index = 0; index < json->size(); ++index) {
    const Json &range = (*json)[index];
    const std::string item_path = ItemPath(path, index);
    if (!range.is_array() || range.size() != 2) {
      Fail(item_path + ": expected [minimum, maximum]");
    }
    const int64_t minimum = Int64Value(range[0], item_path + "[0]");
    const int64_t maximum = Int64Value(range[1], item_path + "[1]");
    if (minimum < 0 || maximum < minimum ||
        (shape[index] >= 0 && (minimum != shape[index] || maximum != shape[index]))) {
      Fail(item_path + ": shape_range is invalid");
    }
    output->push_back(DimensionRange{minimum, maximum});
  }
}

void ParseIoBindings(const Json &json, Direction expected_direction, const std::string &path,
                     std::set<size_t> *argument_indices, std::vector<IoBinding> *output) {
  const std::string expected = expected_direction == Direction::kInput ? "input" : "output";
  for (size_t index = 0; index < json.size(); ++index) {
    const std::string item_path = ItemPath(path, index);
    ObjectReader fields(json[index], item_path);
    IoBinding binding;
    binding.et_arg_index = fields.Size("et_arg_index");
    const std::string direction = fields.Enum("direction", {"input", "output"});
    binding.ascend_name = fields.NonEmptyString("ascend_name");
    binding.dtype = fields.String("dtype");
    const Json &shape = fields.Array("shape");
    binding.semantic_role = fields.Enum("semantic_role", {"data", "state_current", "state_next"});
    const Json *shape_range = fields.OptionalArray("shape_range");
    if (direction != expected) {
      Fail(item_path + ".direction: does not match its containing array");
    }
    if (!IsSupportedDtype(binding.dtype)) {
      Fail(item_path + ".dtype: unsupported value " + binding.dtype);
    }
    if (!argument_indices->insert(binding.et_arg_index).second) {
      Fail(item_path + ".et_arg_index: duplicated argument index");
    }
    binding.direction = expected_direction;
    binding.shape = ParseShape(shape, item_path + ".shape", -1);
    ParseShapeRange(shape_range, binding.shape, item_path + ".shape_range", &binding.shape_range);
    output->push_back(std::move(binding));
  }
}

void ParseIoContract(const Json &json, Manifest *manifest) {
  ObjectReader fields(json, "manifest.io_contract");
  const Json &inputs = fields.Array("inputs");
  const Json &outputs = fields.Array("outputs");
  std::set<size_t> argument_indices;
  ParseIoBindings(inputs, Direction::kInput, "manifest.io_contract.inputs", &argument_indices, &manifest->inputs);
  ParseIoBindings(outputs, Direction::kOutput, "manifest.io_contract.outputs", &argument_indices, &manifest->outputs);
  for (size_t index = 0; index < argument_indices.size(); ++index) {
    if (argument_indices.count(index) == 0) {
      Fail("manifest.io_contract: argument indices must be contiguous");
    }
  }
}

void ParseState(const Json &json, Manifest *manifest) {
  ObjectReader state(json, "manifest.state");
  const Json &bindings = state.Array("bindings");
  std::unordered_map<size_t, const IoBinding *> io_by_argument;
  for (const IoBinding &binding : manifest->inputs) {
    io_by_argument.emplace(binding.et_arg_index, &binding);
  }
  for (const IoBinding &binding : manifest->outputs) {
    io_by_argument.emplace(binding.et_arg_index, &binding);
  }
  std::set<std::string> state_ids;
  std::set<size_t> state_arguments;
  for (size_t index = 0; index < bindings.size(); ++index) {
    const std::string path = ItemPath("manifest.state.bindings", index);
    ObjectReader fields(bindings[index], path);
    StateBinding binding;
    binding.state_id = fields.NonEmptyString("state_id");
    binding.current_input_arg = fields.Size("current_input_arg");
    const int64_t next_output_arg = fields.Int64("next_output_arg");
    binding.dtype = fields.String("dtype");
    const Json &shape = fields.Array("shape");
    binding.reset_policy = fields.Enum("reset_policy", {"on_prefill", "explicit", "never"});
    binding.reset_owner = fields.Bool("reset_owner");
    binding.host_visibility = fields.Enum("host_visibility", {"device_only", "synchronized"});
    if (bindings[index].contains("buffer_mode")) {
      Fail(path + ".buffer_mode is unsupported; re-export for the double-buffer runtime");
    }
    if (!IsSupportedDtype(binding.dtype)) {
      Fail(path + ".dtype: unsupported value " + binding.dtype);
    }
    if (!state_ids.insert(binding.state_id).second) {
      Fail(path + ".state_id: duplicated state identifier");
    }
    if (next_output_arg < -1 || (binding.reset_owner && manifest->role != "prefill") ||
        (binding.reset_owner && binding.reset_policy != "on_prefill")) {
      Fail(path + ": state identity or reset policy is invalid");
    }
    binding.next_output_arg =
        next_output_arg < 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(next_output_arg);
    if (binding.current_input_arg == binding.next_output_arg || state_arguments.count(binding.current_input_arg) != 0 ||
        (next_output_arg >= 0 && state_arguments.count(binding.next_output_arg) != 0)) {
      Fail(path + ": state argument is reused");
    }
    state_arguments.insert(binding.current_input_arg);
    if (next_output_arg >= 0) {
      state_arguments.insert(binding.next_output_arg);
    }
    binding.shape = ParseShape(shape, path + ".shape", 0);

    const auto current = io_by_argument.find(binding.current_input_arg);
    const auto next = io_by_argument.find(binding.next_output_arg);
    if (current == io_by_argument.end() || current->second->direction != Direction::kInput ||
        current->second->semantic_role != "state_current" || current->second->dtype != binding.dtype ||
        current->second->shape != binding.shape ||
        (next_output_arg >= 0 && (next == io_by_argument.end() || next->second->direction != Direction::kOutput ||
                                  next->second->semantic_role != "state_next" || next->second->dtype != binding.dtype ||
                                  next->second->shape != binding.shape))) {
      Fail(path + ": state binding does not match the I/O contract");
    }
    manifest->state_bindings.push_back(std::move(binding));
  }

  auto check_coverage = [&](const IoBinding &binding) {
    if ((binding.semantic_role == "data") != (state_arguments.count(binding.et_arg_index) == 0)) {
      Fail("manifest.state: bindings do not cover every state I/O argument");
    }
  };
  std::for_each(manifest->inputs.begin(), manifest->inputs.end(), check_coverage);
  std::for_each(manifest->outputs.begin(), manifest->outputs.end(), check_coverage);
}

Json ParseEnvelope(const void *data, size_t size) {
  constexpr size_t kHeaderSize = 10;
  if (data == nullptr || size < kHeaderSize) {
    Fail("processed data is shorter than the ASCD header");
  }
  const auto *bytes = static_cast<const uint8_t *>(data);
  if (std::memcmp(bytes, kMagic, 4) != 0) {
    Fail("processed data has invalid ASCD magic");
  }
  if (ReadLittleEndian16(bytes + 4) != kFramingVersion) {
    Fail("unsupported ASCD framing version");
  }
  const uint32_t json_size = ReadLittleEndian32(bytes + 6);
  if (json_size > kMaximumManifestBytes || static_cast<size_t>(json_size) != size - kHeaderSize) {
    Fail("ASCD JSON length is invalid");
  }
  Json root = Json::parse(bytes + kHeaderSize, bytes + size);
  if (!root.is_object()) {
    Fail("ASCD payload must be a JSON object");
  }
  return root;
}

Manifest ParseManifestObject(const Json &root) {
  ObjectReader fields(root, "manifest");
  Manifest manifest;
  ParseRuntimeProtocol(fields.Object("runtime_protocol"));
  manifest.method = fields.NonEmptyString("method");
  manifest.delegate_index = fields.Size("delegate_index");
  manifest.role = fields.Enum("role", {"inference", "prefill", "decode"});
  manifest.acl_soc_name = fields.NonEmptyString("acl_soc_name");
  manifest.model = ParseModel(fields.Object("model"));

  std::unordered_map<std::string, size_t> group_sizes;
  ParseWeightGroups(fields.Array("weight_groups"), &manifest, &group_sizes);
  ParseFileConstants(fields.Array("file_constant_bindings"), group_sizes, &manifest);
  ParseIoContract(fields.Object("io_contract"), &manifest);
  ParseState(fields.Object("state"), &manifest);
  return manifest;
}

}  // namespace

bool ParseManifest(const void *data, size_t size, Manifest *manifest, std::string *error) {
  if (manifest == nullptr) {
    SetError("manifest output pointer is null", error);
    return false;
  }
  try {
    Manifest parsed = ParseManifestObject(ParseEnvelope(data, size));
    *manifest = std::move(parsed);
    if (error != nullptr) {
      error->clear();
    }
    return true;
  } catch (const ManifestParseError &exception) {
    SetError(exception.what(), error);
  } catch (const Json::exception &exception) {
    SetError(std::string("invalid manifest JSON: ") + exception.what(), error);
  }
  return false;
}

}  // namespace executorch_npu
