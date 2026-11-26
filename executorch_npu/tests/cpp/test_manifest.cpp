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
#include <limits>
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

  // The supported dtype set has one definition (DtypeElementSize); parsing must
  // reject anything outside it instead of accepting a name the backend cannot size.
  for (const char *bad_dtype : {"float8_e4m3fn", "FLOAT32", ""}) {
    Json bad_dtype_json = Json::parse(ManifestJson());
    bad_dtype_json["io_contract"]["inputs"][0]["dtype"] = bad_dtype;
    const auto bad_dtype_bytes = Frame(bad_dtype_json.dump());
    assert(!executorch_npu::ParseManifest(bad_dtype_bytes.data(), bad_dtype_bytes.size(), &manifest, &error));
    assert(error.find("manifest.io_contract.inputs[0].dtype") != std::string::npos);
    assert(error.find("unsupported value") != std::string::npos);
  }
  // int16 and float64 are part of the supported set and must parse.
  for (const char *good_dtype : {"int16", "float64"}) {
    Json good_dtype_json = Json::parse(ManifestJson());
    good_dtype_json["io_contract"]["inputs"][0]["dtype"] = good_dtype;
    const auto good_dtype_bytes = Frame(good_dtype_json.dump());
    assert(executorch_npu::ParseManifest(good_dtype_bytes.data(), good_dtype_bytes.size(), &manifest, &error));
    assert(manifest.inputs[0].dtype == good_dtype);
  }

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

  // ExecuTorch allocates one argument slot per binding regardless of state host
  // visibility, so the count must cover device_only states in trailing slots.
  // Under-counting them made the backend reject a valid invocation with
  // Error::InvalidArgument before it ever reached the device.
  {
    const auto io = [](size_t index, const char *role) {
      executorch_npu::IoBinding binding;
      binding.et_arg_index = index;
      binding.semantic_role = role;
      return binding;
    };
    const auto state = [](size_t current, size_t next, const char *visibility) {
      executorch_npu::StateBinding binding;
      binding.state_id = "s";
      binding.current_input_arg = current;
      binding.next_output_arg = next;
      binding.host_visibility = visibility;
      return std::vector<executorch_npu::StateBinding>{binding};
    };
    executorch_npu::Manifest counted;

    // synchronized explicit contract: the state reuses user I/O slots.
    counted.inputs = {io(0, "data"), io(1, "state_current")};
    counted.outputs = {io(2, "data"), io(3, "state_next")};
    counted.state_bindings = state(1, 3, "synchronized");
    assert(executorch_npu::ExpectedArgumentCount(counted) == 4);

    // device_only read-only state: there is no state_next slot at all.
    counted.inputs = {io(0, "data"), io(1, "state_current")};
    counted.outputs = {io(2, "data")};
    counted.state_bindings = state(1, std::numeric_limits<size_t>::max(), "device_only");
    assert(executorch_npu::ExpectedArgumentCount(counted) == 3);

    // device_only mutated buffer: the state occupies the trailing slots and the
    // manifest lists it before the user data. This is the shape that used to be
    // under-counted, because every trailing slot is device_only.
    counted.inputs = {io(2, "state_current"), io(0, "data")};
    counted.outputs = {io(3, "state_next"), io(1, "data")};
    counted.state_bindings = state(2, 3, "device_only");
    assert(executorch_npu::ExpectedArgumentCount(counted) == 4);

    // stateless delegate.
    counted.inputs = {io(0, "data")};
    counted.outputs = {io(1, "data")};
    counted.state_bindings.clear();
    assert(executorch_npu::ExpectedArgumentCount(counted) == 2);
  }

  // Every dtype the manifest accepts must have a byte width, otherwise a
  // device_only state of that dtype cannot be zero-initialized and the backend
  // reports the unrelated "requires an initializer" error. int16 and float64
  // were missing from the table that used to live in ascend_backend.cpp.
  {
    const struct {
      const char *dtype;
      size_t element_size;
    } kExpected[] = {
        {"bool", 1},  {"uint8", 1},   {"int8", 1},     {"int16", 2},   {"int32", 4},
        {"int64", 8}, {"float16", 2}, {"bfloat16", 2}, {"float32", 4}, {"float64", 8},
    };
    for (const auto &entry : kExpected) {
      assert(executorch_npu::DtypeElementSize(entry.dtype) == entry.element_size);

      executorch_npu::StateBinding binding;
      binding.dtype = entry.dtype;
      binding.shape = {2, 3};
      size_t nbytes = 0;
      assert(executorch_npu::StateZeroInitializerBytes(binding, &nbytes));
      assert(nbytes == 6 * entry.element_size);
    }
    // Names outside the supported set must be rejected rather than silently sized.
    assert(executorch_npu::DtypeElementSize("float8_e4m3fn") == 0);
    assert(executorch_npu::DtypeElementSize("") == 0);
    assert(executorch_npu::DtypeElementSize("FLOAT32") == 0);

    executorch_npu::StateBinding unsupported;
    unsupported.dtype = "float8_e4m3fn";
    unsupported.shape = {4};
    size_t ignored = 0;
    assert(!executorch_npu::StateZeroInitializerBytes(unsupported, &ignored));

    // A non-positive or overflowing dimension must be refused, not truncated.
    executorch_npu::StateBinding non_positive;
    non_positive.dtype = "float32";
    non_positive.shape = {0};
    assert(!executorch_npu::StateZeroInitializerBytes(non_positive, &ignored));

    executorch_npu::StateBinding overflow;
    overflow.dtype = "float64";
    overflow.shape = {static_cast<int64_t>(1) << 62, static_cast<int64_t>(1) << 62};
    assert(!executorch_npu::StateZeroInitializerBytes(overflow, &ignored));

    // A scalar state has an empty shape and still occupies one element.
    executorch_npu::StateBinding scalar;
    scalar.dtype = "int64";
    size_t scalar_bytes = 0;
    assert(executorch_npu::StateZeroInitializerBytes(scalar, &scalar_bytes));
    assert(scalar_bytes == 8);
  }

  framed[0] = 'X';
  assert(!executorch_npu::ParseManifest(framed.data(), framed.size(), &manifest, &error));
  assert(!error.empty());
  return 0;
}
