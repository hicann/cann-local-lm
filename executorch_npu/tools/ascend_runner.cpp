/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

using executorch::aten::BFloat16;
using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::module::Module;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::LoadBackendOptionsMap;

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return {};
  }
  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    return {};
  }
  stream.seekg(0);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!stream.read(reinterpret_cast<char *>(data.data()), size)) {
    return {};
  }
  return data;
}

int Fail(const std::string &message, Error error = Error::Ok) {
  std::cerr << message;
  if (error != Error::Ok) {
    std::cerr << ": 0x" << std::hex << static_cast<uint32_t>(error);
  }
  std::cerr << '\n';
  return 1;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: " << argv[0] << " MODEL.pte WEIGHTS.ptd INPUT.bin EXPECTED.bin DEVICE_ID "
              << "STATE_NAMESPACE [BATCH_SIZE]\n";
    return 2;
  }
  executorch::runtime::runtime_init();
  const int device_id = std::stoi(argv[5]);
  const char *state_namespace = argv[6];
  if (device_id < 0 || state_namespace[0] == '\0') {
    return Fail("device_id must be non-negative and state_namespace non-empty");
  }

  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", device_id) != Error::Ok) {
    return Fail("failed to create Ascend backend options");
  }
  if (ascend_options.set_option("state_namespace", state_namespace) != Error::Ok) {
    return Fail("failed to create Ascend backend options");
  }
  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) != Error::Ok) {
    return Fail("failed to install Ascend backend options");
  }

  Module module(argv[1], argv[2]);
  Error status = module.load(backend_options);
  if (status != Error::Ok) {
    return Fail("failed to load PTE", status);
  }
  auto metadata = module.method_meta("forward");
  if (!metadata.ok() || metadata->num_inputs() != 1 || metadata->num_outputs() != 1) {
    return Fail("smoke runner requires one input and one output");
  }
  auto input_metadata = metadata->input_tensor_meta(0);
  if (!input_metadata.ok()) {
    return Fail("input is not a tensor", input_metadata.error());
  }

  std::vector<uint8_t> input = ReadFile(argv[3]);
  std::vector<executorch::aten::SizesType> input_shape;
  for (int32_t dimension : input_metadata->sizes()) {
    input_shape.push_back(dimension);
  }
  size_t expected_input_nbytes = input_metadata->nbytes();
  if (argc == 8) {
    const int64_t requested_batch = std::stoll(argv[7]);
    if (input_shape.empty() || input_shape.front() <= 0 || requested_batch <= 0 ||
        requested_batch > std::numeric_limits<executorch::aten::SizesType>::max()) {
      return Fail("invalid batch size override");
    }
    const size_t metadata_batch = static_cast<size_t>(input_shape.front());
    const size_t requested = static_cast<size_t>(requested_batch);
    if (expected_input_nbytes % metadata_batch != 0 ||
        expected_input_nbytes / metadata_batch > std::numeric_limits<size_t>::max() / requested) {
      return Fail("input byte size overflow for batch override");
    }
    expected_input_nbytes = expected_input_nbytes / metadata_batch * requested;
    input_shape.front() = static_cast<executorch::aten::SizesType>(requested_batch);
  }
  if (input.size() != expected_input_nbytes) {
    return Fail("input file size differs from requested input shape");
  }
  std::cout << "ASCEND_EXECUTORCH_INPUT shape=[";
  for (size_t index = 0; index < input_shape.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << input_shape[index];
  }
  std::cout << "] bytes=" << input.size() << '\n';
  auto input_tensor = from_blob(input.data(), input_shape, input_metadata->scalar_type());
  auto outputs = module.forward(EValue(*input_tensor));
  if (!outputs.ok() || outputs->size() != 1 || !outputs->front().isTensor()) {
    return Fail("Ascend execution failed", outputs.ok() ? Error::InvalidProgram : outputs.error());
  }

  const auto &output = outputs->front().toTensor();
  std::vector<uint8_t> expected = ReadFile(argv[4]);
  if (expected.size() != output.nbytes()) {
    return Fail("expected output size differs from runtime output");
  }
  size_t count = 0;
  if (output.scalar_type() == ScalarType::Float && output.nbytes() % sizeof(float) == 0) {
    const auto *actual_values = output.const_data_ptr<float>();
    count = output.nbytes() / sizeof(float);
    for (size_t index = 0; index < count; ++index) {
      float expected_value = 0.0F;
      std::memcpy(&expected_value, expected.data() + index * sizeof(float), sizeof(float));
      const float difference = std::abs(actual_values[index] - expected_value);
      const float tolerance = 1.0e-3F + 1.0e-3F * std::abs(expected_value);
      if (!std::isfinite(actual_values[index]) || difference > tolerance) {
        std::cerr << "output mismatch at " << index << ": actual=" << actual_values[index]
                  << ", expected=" << expected_value << '\n';
        return 1;
      }
    }
  } else if (output.scalar_type() == ScalarType::BFloat16 && output.nbytes() % sizeof(uint16_t) == 0) {
    const auto *actual_values = output.const_data_ptr<BFloat16>();
    count = output.nbytes() / sizeof(uint16_t);
    size_t mismatch_count = 0;
    size_t max_difference_index = 0;
    size_t actual_argmax = 0;
    size_t expected_argmax = 0;
    float max_difference = 0.0F;
    float actual_max = -INFINITY;
    float expected_max = -INFINITY;
    for (size_t index = 0; index < count; ++index) {
      uint16_t expected_bits = 0;
      std::memcpy(&expected_bits, expected.data() + index * sizeof(uint16_t), sizeof(uint16_t));
      const float actual_value = static_cast<float>(actual_values[index]);
      const float expected_value = static_cast<float>(BFloat16(expected_bits, BFloat16::from_bits()));
      const float difference = std::abs(actual_value - expected_value);
      const float tolerance = 1.0e-2F + 1.0e-2F * std::abs(expected_value);
      if (!std::isfinite(actual_value) || difference > tolerance) {
        ++mismatch_count;
      }
      if (difference > max_difference) {
        max_difference = difference;
        max_difference_index = index;
      }
      if (actual_value > actual_max) {
        actual_max = actual_value;
        actual_argmax = index;
      }
      if (expected_value > expected_max) {
        expected_max = expected_value;
        expected_argmax = index;
      }
    }
    if (mismatch_count != 0) {
      std::cerr << "bfloat16 output mismatch: count=" << mismatch_count << ", max_abs_diff=" << max_difference
                << ", max_abs_diff_index=" << max_difference_index << ", actual_argmax=" << actual_argmax
                << ", expected_argmax=" << expected_argmax << '\n';
      return 1;
    }
  } else if (output.scalar_type() == ScalarType::Long && output.nbytes() % sizeof(int64_t) == 0) {
    const auto *actual_values = output.const_data_ptr<int64_t>();
    count = output.nbytes() / sizeof(int64_t);
    for (size_t index = 0; index < count; ++index) {
      int64_t expected_value = 0;
      std::memcpy(&expected_value, expected.data() + index * sizeof(int64_t), sizeof(int64_t));
      if (actual_values[index] != expected_value) {
        std::cerr << "int64 output mismatch at " << index << ": actual=" << actual_values[index]
                  << ", expected=" << expected_value << '\n';
        return 1;
      }
    }
  } else {
    return Fail("smoke runner supports float32, bfloat16, and int64 output");
  }
  std::cout << "ASCEND_EXECUTORCH_OK values=" << count << '\n';
  return 0;
}
