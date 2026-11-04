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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch_npu {

constexpr size_t kMaximumManifestBytes = 1024 * 1024;

struct ModelArtifact {
  std::string named_data_key;
  std::string sha256;
  size_t nbytes = 0;
};

struct WeightGroup {
  std::string key;
  std::string sha256;
  size_t nbytes = 0;
};

struct FileConstantBinding {
  std::string binding_id;
  std::string file_name;
  std::string weight_group_key;
  size_t offset = 0;
  size_t nbytes = 0;
  std::string alias_of;
};

enum class Direction {
  kInput,
  kOutput,
};

struct DimensionRange {
  int64_t minimum = 0;
  int64_t maximum = 0;
};

struct IoBinding {
  size_t et_arg_index = 0;
  Direction direction = Direction::kInput;
  std::string ascend_name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<DimensionRange> shape_range;
  std::string semantic_role;
};

struct StateBinding {
  std::string state_id;
  size_t current_input_arg = 0;
  size_t next_output_arg = 0;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string reset_policy;
  bool reset_owner = false;
  std::string host_visibility;
};

struct Manifest {
  std::string method;
  size_t delegate_index = 0;
  std::string role;
  std::string acl_soc_name;
  ModelArtifact model;
  std::vector<WeightGroup> weight_groups;
  std::vector<FileConstantBinding> file_constant_bindings;
  std::vector<IoBinding> inputs;
  std::vector<IoBinding> outputs;
  std::vector<StateBinding> state_bindings;
};

bool ParseManifest(const void *data, size_t size, Manifest *manifest, std::string *error);

}  // namespace executorch_npu
