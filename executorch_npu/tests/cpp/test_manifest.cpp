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

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

std::vector<uint8_t> Frame(const std::string &json) {
  const uint32_t size = static_cast<uint32_t>(json.size());
  std::vector<uint8_t> output = {'A',
                                 'S',
                                 'C',
                                 'D',
                                 1,
                                 0,
                                 static_cast<uint8_t>(size),
                                 static_cast<uint8_t>(size >> 8U),
                                 static_cast<uint8_t>(size >> 16U),
                                 static_cast<uint8_t>(size >> 24U)};
  output.insert(output.end(), json.begin(), json.end());
  return output;
}

std::string ManifestJson() {
  const std::string weight_hash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  const std::string model_hash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
  return std::string("{") + R"("runtime_protocol":{"name":"AscendBackend","version":1,"artifact_abi":1},)" +
         R"("method":"forward","delegate_index":0,"role":"inference","acl_soc_name":"Ascend910_9362","model":{"named_data_key":"model","sha256":")" +
         model_hash + R"(","nbytes":7},)" + R"("weight_groups":[{"weight_group_key":"ascend.weights.)" + weight_hash +
         R"(","sha256":")" + weight_hash + R"(","total_nbytes":11}],)" +
         R"("file_constant_bindings":[{"binding_id":"fc0","file_name":"weight_0","weight_group_key":"ascend.weights.)" +
         weight_hash + R"(","offset":0,"nbytes":11,"shape":[11]}],)" +
         R"("io_contract":{"inputs":[{"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[1,4],"semantic_role":"data"}],"outputs":[{"et_arg_index":1,"direction":"output","ascend_name":"y","dtype":"float32","shape":[1,4],"semantic_role":"data"}]},)" +
         R"("state":{"bindings":[]}})";
}

std::string ReplaceOnce(std::string value, const std::string &from, const std::string &to) {
  const size_t position = value.find(from);
  assert(position != std::string::npos);
  value.replace(position, from.size(), to);
  return value;
}

}  // namespace

int main(int argc, char **argv) {
  // Accept Python-produced ASCD payloads for producer/runtime compatibility tests.
  if (argc > 1) {
    for (int index = 1; index < argc; ++index) {
      std::ifstream input(argv[index], std::ios::binary);
      assert(input.is_open());
      const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
      executorch_npu::Manifest parsed;
      std::string error;
      if (!executorch_npu::ParseManifest(bytes.data(), bytes.size(), &parsed, &error)) {
        std::cerr << argv[index] << ": " << error << '\n';
        return 1;
      }
    }
    return 0;
  }
  std::vector<uint8_t> framed = Frame(ManifestJson());
  executorch_npu::Manifest manifest;
  std::string error;
  assert(executorch_npu::ParseManifest(framed.data(), framed.size(), &manifest, &error));
  assert(error.empty());
  assert(manifest.method == "forward");
  assert(manifest.delegate_index == 0);
  assert(manifest.weight_groups.size() == 1);
  assert(manifest.file_constant_bindings.size() == 1);
  assert(manifest.inputs.size() == 1);
  assert(manifest.outputs.size() == 1);
  assert(manifest.state_bindings.empty());

  Json missing_method_json = Json::parse(ManifestJson());
  missing_method_json.erase("method");
  std::vector<uint8_t> missing_method = Frame(missing_method_json.dump());
  assert(!executorch_npu::ParseManifest(missing_method.data(), missing_method.size(), &manifest, &error));
  assert(error.find("manifest.method") != std::string::npos);
  assert(error.find("required field is missing") != std::string::npos);

  Json unknown_field_json = Json::parse(ManifestJson());
  unknown_field_json["unexpected"] = true;
  unknown_field_json["model"]["producer_note"] = "future metadata";
  unknown_field_json["io_contract"]["inputs"][0]["future_hint"] = 1;
  std::vector<uint8_t> unknown_field = Frame(unknown_field_json.dump());
  assert(executorch_npu::ParseManifest(unknown_field.data(), unknown_field.size(), &manifest, &error));
  assert(error.empty());
  assert(manifest.method == "forward");

  Json wrong_type_json = Json::parse(ManifestJson());
  wrong_type_json["model"]["nbytes"] = "7";
  std::vector<uint8_t> wrong_type = Frame(wrong_type_json.dump());
  assert(!executorch_npu::ParseManifest(wrong_type.data(), wrong_type.size(), &manifest, &error));
  assert(error.find("manifest.model.nbytes") != std::string::npos);
  assert(error.find("non-negative integer") != std::string::npos);

  const std::string manifest_310b_json =
      ReplaceOnce(ManifestJson(), R"("acl_soc_name":"Ascend910_9362")", R"("acl_soc_name":"Ascend310B1")");
  std::vector<uint8_t> manifest_310b = Frame(manifest_310b_json);
  assert(executorch_npu::ParseManifest(manifest_310b.data(), manifest_310b.size(), &manifest, &error));
  assert(manifest.acl_soc_name == "Ascend310B1");

  const std::string static_input =
      R"({"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[1,4],"semantic_role":"data"})";
  const std::string dynamic_input =
      R"({"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[-1,4],"shape_range":[[2,4],[4,4]],"semantic_role":"data"})";
  std::vector<uint8_t> dynamic = Frame(ReplaceOnce(ManifestJson(), static_input, dynamic_input));
  assert(executorch_npu::ParseManifest(dynamic.data(), dynamic.size(), &manifest, &error));
  assert(manifest.inputs[0].shape[0] == -1);
  assert(manifest.inputs[0].shape_range.size() == 2);
  assert(manifest.inputs[0].shape_range[0].minimum == 2);
  assert(manifest.inputs[0].shape_range[0].maximum == 4);

  const std::string invalid_dynamic_input =
      R"({"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[-1,4],"shape_range":[[4,2],[4,4]],"semantic_role":"data"})";
  std::vector<uint8_t> invalid_dynamic = Frame(ReplaceOnce(ManifestJson(), static_input, invalid_dynamic_input));
  assert(!executorch_npu::ParseManifest(invalid_dynamic.data(), invalid_dynamic.size(), &manifest, &error));
  assert(error.find("shape_range is invalid") != std::string::npos);

  const std::string original_io =
      R"("io_contract":{"inputs":[{"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[1,4],"semantic_role":"data"}],"outputs":[{"et_arg_index":1,"direction":"output","ascend_name":"y","dtype":"float32","shape":[1,4],"semantic_role":"data"}]},)" +
      std::string(R"("state":{"bindings":[]})");
  const std::string stateful_io =
      R"("io_contract":{"inputs":[{"et_arg_index":0,"direction":"input","ascend_name":"x","dtype":"float32","shape":[1,4],"semantic_role":"state_current"}],"outputs":[{"et_arg_index":1,"direction":"output","ascend_name":"y","dtype":"float32","shape":[1,4],"semantic_role":"state_next"}]},)" +
      std::string(
          R"("state":{"bindings":[{"state_id":"kv","current_input_arg":0,"next_output_arg":1,"dtype":"float32","shape":[1,4],"reset_policy":"on_prefill","reset_owner":true,"host_visibility":"synchronized"}]})");
  std::string stateful_json = ReplaceOnce(ManifestJson(), original_io, stateful_io);
  stateful_json = ReplaceOnce(stateful_json, R"("role":"inference")", R"("role":"prefill")");
  std::vector<uint8_t> stateful = Frame(stateful_json);
  assert(executorch_npu::ParseManifest(stateful.data(), stateful.size(), &manifest, &error));
  assert(manifest.state_bindings.size() == 1);
  assert(manifest.state_bindings[0].state_id == "kv");
  for (const char *mode : {"single", "double", "guess"}) {
    Json retired_state = Json::parse(stateful_json);
    retired_state["state"]["bindings"][0]["buffer_mode"] = mode;
    const auto retired_bytes = Frame(retired_state.dump());
    assert(!executorch_npu::ParseManifest(retired_bytes.data(), retired_bytes.size(), &manifest, &error));
    assert(error.find("buffer_mode is unsupported") != std::string::npos);
  }

  const std::string scalar_stateful_io =
      R"("io_contract":{"inputs":[{"et_arg_index":0,"direction":"input","ascend_name":"counter","dtype":"int64","shape":[],"semantic_role":"state_current"}],"outputs":[{"et_arg_index":1,"direction":"output","ascend_name":"counter_next","dtype":"int64","shape":[],"semantic_role":"state_next"}]},)" +
      std::string(
          R"("state":{"bindings":[{"state_id":"counter","current_input_arg":0,"next_output_arg":1,"dtype":"int64","shape":[],"reset_policy":"never","reset_owner":false,"host_visibility":"device_only"}]})");
  const std::vector<uint8_t> scalar_stateful = Frame(ReplaceOnce(ManifestJson(), original_io, scalar_stateful_io));
  assert(executorch_npu::ParseManifest(scalar_stateful.data(), scalar_stateful.size(), &manifest, &error));
  assert(manifest.state_bindings.size() == 1);
  assert(manifest.state_bindings[0].shape.empty());

  const std::string orphan_state_role =
      ReplaceOnce(ManifestJson(), R"("semantic_role":"data")", R"("semantic_role":"state_current")");
  std::vector<uint8_t> orphan = Frame(orphan_state_role);
  assert(!executorch_npu::ParseManifest(orphan.data(), orphan.size(), &manifest, &error));
  assert(error.find("do not cover") != std::string::npos);

  const std::string misaligned_weight =
      ReplaceOnce(ManifestJson(), R"("offset":0,"nbytes":11,"shape":[11])", R"("offset":1,"nbytes":10,"shape":[10])");
  std::vector<uint8_t> misaligned = Frame(misaligned_weight);
  assert(!executorch_npu::ParseManifest(misaligned.data(), misaligned.size(), &manifest, &error));
  assert(error.find("range is invalid") != std::string::npos);

  framed[0] = 'X';
  assert(!executorch_npu::ParseManifest(framed.data(), framed.size(), &manifest, &error));
  assert(!error.empty());
  return 0;
}
