/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
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

#include "executorch_npu/state.h"
#include "session_sampling.h"

namespace {

using executorch::aten::BFloat16;
using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::module::Module;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::LoadBackendOptionsMap;

int Fail(const std::string &message, Error error = Error::Ok) {
  std::cerr << message;
  if (error != Error::Ok) {
    std::cerr << ": 0x" << std::hex << static_cast<uint32_t>(error);
  }
  std::cerr << '\n';
  return 1;
}

bool ReadFile(const std::string &path, std::vector<uint8_t> *data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return false;
  }
  const std::streamsize size = stream.tellg();
  if (size < 0) {
    return false;
  }
  stream.seekg(0);
  data->resize(static_cast<size_t>(size));
  return size == 0 || stream.read(reinterpret_cast<char *>(data->data()), size);
}

bool WriteFile(const std::filesystem::path &path, const void *data, size_t size) {
  std::ofstream stream(path, std::ios::binary);
  return stream && stream.write(reinterpret_cast<const char *>(data), size);
}

std::vector<executorch::aten::SizesType> TensorShape(const executorch::runtime::TensorInfo &metadata) {
  std::vector<executorch::aten::SizesType> shape;
  shape.reserve(metadata.sizes().size());
  for (int32_t dimension : metadata.sizes()) {
    shape.push_back(dimension);
  }
  return shape;
}

bool ParseInt(const char *value, int *result) {
  try {
    size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (value[consumed] != '\0') {
      return false;
    }
    *result = parsed;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseFloat(const char *value, float *result) {
  try {
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (value[consumed] != '\0' || !std::isfinite(parsed)) {
      return false;
    }
    *result = parsed;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseUint64(const char *value, uint64_t *result) {
  try {
    if (value[0] == '-') {
      return false;
    }
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0') {
      return false;
    }
    *result = static_cast<uint64_t>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseShape(const char *value, std::vector<executorch::aten::SizesType> *shape) {
  std::stringstream stream(value);
  std::string dimension;
  while (std::getline(stream, dimension, ',')) {
    try {
      size_t consumed = 0;
      const long long parsed = std::stoll(dimension, &consumed);
      if (consumed != dimension.size() || parsed <= 0 ||
          parsed > std::numeric_limits<executorch::aten::SizesType>::max()) {
        return false;
      }
      shape->push_back(static_cast<executorch::aten::SizesType>(parsed));
    } catch (const std::exception &) {
      return false;
    }
  }
  return !shape->empty();
}

size_t ElementCount(const std::vector<executorch::aten::SizesType> &shape) {
  size_t count = 1;
  for (const auto dimension : shape) {
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void FillCausalMasks(int seq_len, size_t kv_len, int mask_window, std::vector<uint8_t> *full,
                     std::vector<uint8_t> *sliding) {
  const size_t elements = static_cast<size_t>(seq_len) * kv_len;
  full->assign(elements * sizeof(BFloat16), 0);
  if (sliding != nullptr) {
    sliding->assign(elements * sizeof(BFloat16), 0);
  }
  for (int row = 0; row < seq_len; ++row) {
    for (size_t col = 0; col < kv_len; ++col) {
      const bool causal = col <= static_cast<size_t>(row);
      const bool in_window = sliding == nullptr ||
                             static_cast<int64_t>(col) > static_cast<int64_t>(row) - static_cast<int64_t>(mask_window);
      const size_t offset = (static_cast<size_t>(row) * kv_len + col) * 2;
      // BF16 finfo.min has bits 0xFF7F, stored as 0x7F, 0xFF.
      if (!causal) {
        (*full)[offset] = 0x7F;
        (*full)[offset + 1] = 0xFF;
      }
      if (sliding != nullptr && (!causal || !in_window)) {
        (*sliding)[offset] = 0x7F;
        (*sliding)[offset + 1] = 0xFF;
      }
    }
  }
}

bool IsInt64Tensor(const executorch::runtime::TensorInfo &metadata) {
  return metadata.scalar_type() == ScalarType::Long;
}

bool GreedyToken(const executorch::aten::Tensor &logits, int64_t *token) {
  const auto &sizes = logits.sizes();
  if (sizes.empty() || sizes.back() <= 0) {
    return false;
  }
  const size_t vocabulary_size = static_cast<size_t>(sizes.back());
  size_t element_size = 0;
  if (logits.scalar_type() == ScalarType::Float) {
    element_size = sizeof(float);
  } else if (logits.scalar_type() == ScalarType::BFloat16) {
    element_size = sizeof(BFloat16);
  } else {
    return false;
  }
  const size_t value_count = logits.nbytes() / element_size;
  if (value_count < vocabulary_size) {
    return false;
  }
  size_t best_index = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  if (logits.scalar_type() == ScalarType::Float) {
    const auto *values = logits.const_data_ptr<float>() + value_count - vocabulary_size;
    for (size_t index = 0; index < vocabulary_size; ++index) {
      if (values[index] > best_value) {
        best_value = values[index];
        best_index = index;
      }
    }
  } else if (logits.scalar_type() == ScalarType::BFloat16) {
    const auto *values = logits.const_data_ptr<BFloat16>() + value_count - vocabulary_size;
    for (size_t index = 0; index < vocabulary_size; ++index) {
      const float value = static_cast<float>(values[index]);
      if (value > best_value) {
        best_value = value;
        best_index = index;
      }
    }
  } else {
    return false;
  }
  *token = static_cast<int64_t>(best_index);
  return std::isfinite(best_value);
}

bool ChoiceScores(const executorch::aten::Tensor &logits, const std::array<size_t, 4> &choice_ids, float scores[4]) {
  const auto &sizes = logits.sizes();
  if (sizes.empty() || sizes.back() <= 0) {
    return false;
  }
  const size_t vocabulary_size = static_cast<size_t>(sizes.back());
  const size_t element_size = logits.scalar_type() == ScalarType::Float
                                  ? sizeof(float)
                                  : (logits.scalar_type() == ScalarType::BFloat16 ? sizeof(BFloat16) : 0);
  if (element_size == 0 || logits.nbytes() / element_size < vocabulary_size) {
    return false;
  }
  const size_t base = logits.nbytes() / element_size - vocabulary_size;
  for (size_t index = 0; index < 4; ++index) {
    if (choice_ids[index] >= vocabulary_size) {
      return false;
    }
    const size_t offset = base + choice_ids[index];
    scores[index] = logits.scalar_type() == ScalarType::Float
                        ? logits.const_data_ptr<float>()[offset]
                        : static_cast<float>(logits.const_data_ptr<BFloat16>()[offset]);
    if (!std::isfinite(scores[index])) {
      return false;
    }
  }
  return true;
}

bool SampleToken(const executorch::aten::Tensor &logits, float temperature, size_t top_k, std::mt19937_64 *generator,
                 int64_t *token) {
  const auto &sizes = logits.sizes();
  if (sizes.empty() || sizes.back() <= 0) {
    return false;
  }
  const size_t vocabulary_size = static_cast<size_t>(sizes.back());
  size_t element_size = 0;
  if (logits.scalar_type() == ScalarType::Float) {
    element_size = sizeof(float);
  } else if (logits.scalar_type() == ScalarType::BFloat16) {
    element_size = sizeof(BFloat16);
  } else {
    return false;
  }
  const size_t value_count = logits.nbytes() / element_size;
  if (value_count < vocabulary_size) {
    return false;
  }
  std::vector<float> values(vocabulary_size);
  if (logits.scalar_type() == ScalarType::Float) {
    const auto *source = logits.const_data_ptr<float>() + value_count - vocabulary_size;
    values.assign(source, source + vocabulary_size);
  } else {
    const auto *source = logits.const_data_ptr<BFloat16>() + value_count - vocabulary_size;
    for (size_t index = 0; index < vocabulary_size; ++index) {
      values[index] = static_cast<float>(source[index]);
    }
  }
  return executorch_npu::session_sampling::SampleToken(values, temperature, top_k, generator, token);
}

class StateNamespaceGuard {
 public:
  StateNamespaceGuard(int device_id, const char *state_namespace)
      : device_id_(device_id), state_namespace_(state_namespace) {}

  ~StateNamespaceGuard() {
    const auto status = executorch_npu::ReleaseStateNamespace(device_id_, state_namespace_.c_str());
    if (status != executorch_npu::StateStatus::kOk && status != executorch_npu::StateStatus::kNotFound) {
      std::cerr << "failed to release state_namespace\n";
    }
  }

 private:
  int device_id_;
  std::string state_namespace_;
};

// Optional lightweight regression mode. Logit snapshots and reporting are
// outside measured intervals; ordinary `session` behavior is unchanged.
struct LightEval {
  using Clock = std::chrono::steady_clock;
  int repeats = 1;
  int warmup = 0;
  std::string output;
  std::string forced_path;
  std::string manifest;
  std::vector<int64_t> forced;
  std::vector<int64_t> predictions;
  std::vector<float> logits;
  std::vector<double> step_ms;
  Clock::time_point start, phase;
  bool enabled() const {
    return !output.empty();
  }
  void begin() {
    predictions.clear();
    logits.clear();
    step_ms.clear();
    start = phase = Clock::now();
  }
  bool observe(const executorch::aten::Tensor &tensor, int64_t *token) {
    const auto end = Clock::now();
    step_ms.push_back(std::chrono::duration<double, std::milli>(end - phase).count());
    predictions.push_back(*token);
    const auto sizes = tensor.sizes();
    if (sizes.empty() || sizes.back() <= 0) return false;
    const size_t width = sizes.back();
    const size_t item =
        tensor.scalar_type() == ScalarType::Float ? 4 : (tensor.scalar_type() == ScalarType::BFloat16 ? 2 : 0);
    if (!item || tensor.nbytes() / item < width) return false;
    const size_t base = tensor.nbytes() / item - width;
    for (size_t i = 0; i < width; ++i) {
      const float value = item == 4 ? tensor.const_data_ptr<float>()[base + i]
                                    : static_cast<float>(tensor.const_data_ptr<BFloat16>()[base + i]);
      if (!std::isfinite(value)) return false;
      logits.push_back(value);
    }
    if (!forced.empty()) {
      *token = forced.at(predictions.size() - 1);
      if (static_cast<size_t>(*token) >= width) return false;
    }
    phase = Clock::now();
    return true;
  }
  bool finish(int trial) {
    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (forced.empty()) forced = predictions;
    if (trial < warmup) {
      std::cerr << "LIGHT_EVAL warmup complete " << output << std::endl;
      return true;
    }
    const auto dir = std::filesystem::path(output);
    std::filesystem::create_directories(dir);
    const int index = trial - warmup;
    const auto stem = dir / ("trial_" + std::to_string(index));
    if (!WriteFile(stem.string() + ".f32", logits.data(), logits.size() * sizeof(float)) ||
        !WriteFile(dir / "trajectory.i64", forced.data(), forced.size() * sizeof(int64_t)))
      return false;
    double total = 0;
    for (double step : step_ms) total += step;
    std::ofstream stream(stem.string() + ".json");
    stream << std::setprecision(12) << "{\"trial\":" << index << ",\"output_tokens\":" << predictions.size()
           << ",\"vocab_size\":" << logits.size() / predictions.size() << ",\"ttft_ms\":" << step_ms.front()
           << ",\"tpot_ms\":" << (total - step_ms.front()) / (predictions.size() - 1) << ",\"total_ms\":" << total
           << ",\"throughput_tps\":" << predictions.size() * 1000.0 / total << ",\"wall_with_capture_ms\":" << wall_ms
           << ",\"predictions\":[";
    for (size_t i = 0; i < predictions.size(); ++i) stream << (i ? "," : "") << predictions[i];
    stream << "],\"step_ms\":[";
    for (size_t i = 0; i < step_ms.size(); ++i) stream << (i ? "," : "") << step_ms[i];
    stream << "]}\n";
    std::cout << "LIGHT_EVAL trial=" << index << " ttft_ms=" << step_ms.front() << " total_ms=" << total << '\n';
    return static_cast<bool>(stream);
  }
};

int RunSession(int argc, char **argv) {
  if (argc < 13) {
    std::cerr << "usage: " << argv[0]
              << " session MODEL.pte WEIGHTS.ptd PREFILL_METHOD "
                 "PREFILL_TOKEN_IDS.bin PREFILL_POSITIONS.bin "
                 "DECODE_METHOD_OR_PREFIX START_CACHE_LEN DECODE_INPUT_SHAPE "
                 "MAX_NEW_TOKENS DEVICE_ID STATE_NAMESPACE "
                 "[--temperature VALUE] [--top-k VALUE] [--seed VALUE] "
                 "--max-seq-len VALUE [--eos-token TOKEN_ID] "
                 "[--mask-window W | --causal-mask] "
                 "[--eval-output DIR --forced-tokens FILE --warmup N --repeats N] [--eval-manifest TSV]\n";
    return 2;
  }
  LightEval bench;
  int start_cache_length = 0;
  int max_new_tokens = 0;
  int device_id = 0;
  if (!ParseInt(argv[8], &start_cache_length) || !ParseInt(argv[10], &max_new_tokens) ||
      !ParseInt(argv[11], &device_id) || start_cache_length < 0 || max_new_tokens < 0 || device_id < 0 ||
      argv[12][0] == '\0' || std::string(argv[4]).empty() || std::string(argv[7]).empty()) {
    return Fail("invalid session arguments");
  }
  std::vector<executorch::aten::SizesType> decode_shape;
  if (!ParseShape(argv[9], &decode_shape)) {
    return Fail("decode_input_shape must be comma-separated positive dimensions");
  }
  const size_t decode_elements = ElementCount(decode_shape);
  float temperature = 1.0F;
  size_t top_k = 0;
  uint64_t seed = 0;
  bool sampling = false;
  bool seen_temperature = false;
  bool seen_top_k = false;
  bool seen_seed = false;
  bool seen_eos_token = false;
  bool seen_max_seq_len = false;
  int eos_token = 0;
  int max_seq_len = 0;
  int mask_window = 0;
  bool causal_mask = false;
  for (int index = 13; index < argc;) {
    const std::string option = argv[index];
    if (option == "--causal-mask" && !causal_mask) {
      causal_mask = true;
      ++index;
      continue;
    }
    if (index + 1 >= argc) {
      return Fail("session option requires a value: " + option);
    }
    if (option == "--eval-output" && bench.output.empty()) {
      bench.output = argv[index + 1];
    } else if (option == "--eval-manifest" && bench.manifest.empty()) {
      bench.manifest = argv[index + 1];
    } else if (option == "--forced-tokens" && bench.forced_path.empty()) {
      bench.forced_path = argv[index + 1];
    } else if (option == "--repeats") {
      if (!ParseInt(argv[index + 1], &bench.repeats) || bench.repeats < 1) return Fail("repeats must be positive");
    } else if (option == "--warmup") {
      if (!ParseInt(argv[index + 1], &bench.warmup) || bench.warmup < 0) return Fail("warmup must be nonnegative");
    } else if (option == "--temperature" && !seen_temperature) {
      if (!ParseFloat(argv[index + 1], &temperature) || temperature <= 0) {
        return Fail("temperature must be a finite positive number");
      }
      seen_temperature = true;
    } else if (option == "--top-k" && !seen_top_k) {
      int parsed_top_k = 0;
      if (!ParseInt(argv[index + 1], &parsed_top_k) || parsed_top_k <= 0) {
        return Fail("top-k must be a positive integer");
      }
      top_k = static_cast<size_t>(parsed_top_k);
      seen_top_k = true;
    } else if (option == "--seed" && !seen_seed) {
      if (!ParseUint64(argv[index + 1], &seed)) {
        return Fail("seed must be an unsigned integer");
      }
      seen_seed = true;
    } else if (option == "--eos-token" && !seen_eos_token) {
      if (!ParseInt(argv[index + 1], &eos_token)) {
        return Fail("eos token must be an integer");
      }
      seen_eos_token = true;
    } else if (option == "--max-seq-len" && !seen_max_seq_len) {
      if (!ParseInt(argv[index + 1], &max_seq_len) || max_seq_len < start_cache_length) {
        return Fail("max sequence length must be at least start cache length");
      }
      seen_max_seq_len = true;
    } else if (option == "--mask-window" && mask_window == 0) {
      if (!ParseInt(argv[index + 1], &mask_window) || mask_window <= 0) {
        return Fail("mask window must be a positive integer");
      }
    } else {
      return Fail("unknown or duplicate sampling option: " + option);
    }
    if (option == "--temperature" || option == "--top-k" || option == "--seed") {
      sampling = true;
    }
    index += 2;
  }
  if (causal_mask && mask_window > 0) {
    return Fail("choose either --causal-mask or --mask-window");
  }
  if (!seen_max_seq_len || max_new_tokens > max_seq_len - start_cache_length) {
    return Fail("max_new_tokens exceeds the static artifact capacity");
  }
  if (sampling && !seen_seed) {
    seed = std::random_device{}();
  }
  if (bench.enabled()) {
    if (std::filesystem::exists(std::filesystem::path(bench.output) / "trial_0.json"))
      return Fail("evaluation output already contains results");
    if (sampling || seen_eos_token || max_new_tokens < 2 || decode_elements != 1)
      return Fail("eval requires greedy, no EOS stop, >=2 output tokens and one-token decode");

  } else if (bench.repeats != 1 || bench.warmup || !bench.forced_path.empty() || !bench.manifest.empty()) {
    return Fail("eval options require --eval-output");
  }
  struct EvalCase {
    std::string name, tokens, positions, forced;
    int length;
  };
  std::vector<EvalCase> cases;
  if (!bench.manifest.empty()) {
    std::ifstream manifest(bench.manifest);
    if (!manifest) return Fail("cannot open evaluation manifest");
    std::string line;
    std::set<std::string> names;
    while (std::getline(manifest, line)) {
      if (line.empty()) continue;
      std::istringstream row(line);
      EvalCase item;
      std::string extra;
      if (!(row >> item.name >> item.tokens >> item.positions >> item.length >> item.forced) || (row >> extra) ||
          item.length < 1 || item.length > max_seq_len - max_new_tokens || item.name == "." || item.name == ".." ||
          std::filesystem::path(item.name).filename().string() != item.name || !names.insert(item.name).second)
        return Fail("invalid or duplicate evaluation manifest row");
      if (item.forced == "-") item.forced.clear();
      if (std::filesystem::exists(std::filesystem::path(bench.output) / item.name / "trial_0.json"))
        return Fail("evaluation case already contains results");
      cases.push_back(item);
    }
    if (cases.empty()) return Fail("empty evaluation manifest");
  } else {
    cases.push_back({"", argv[5], argv[6], bench.forced_path, start_cache_length});
  }
  const std::string base_output = bench.output;
  std::mt19937_64 generator(seed);

  executorch::runtime::runtime_init();
  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", device_id) != Error::Ok ||
      ascend_options.set_option("state_namespace", argv[12]) != Error::Ok) {
    return Fail("failed to create Ascend backend options");
  }
  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) != Error::Ok) {
    return Fail("failed to install Ascend backend options");
  }
  StateNamespaceGuard state_guard(device_id, argv[12]);
  Module module(argv[2], argv[3], Module::LoadMode::File, nullptr, nullptr, nullptr, true);
  if (bench.enabled()) std::cerr << "LIGHT_EVAL loading model" << std::endl;
  Error status = module.load(backend_options);
  if (status != Error::Ok) {
    return Fail("failed to load PTE", status);
  }
  // New fixed-capacity Qwen artifacts expose one tensor-position-driven
  // `decode` method. Keep the legacy decode_<cache_len> fallback so existing
  // bucketed artifacts remain runnable.
  const bool single_decode_method = module.method_meta(argv[7]).ok();
  const bool shared_model_method = std::string(argv[4]) == std::string(argv[7]);
  if (max_new_tokens > 1) {
    const std::string first_decode_method =
        single_decode_method ? std::string(argv[7]) : std::string(argv[7]) + "_" + std::to_string(start_cache_length);
    // Explicitly load the first method before prefill. Loading every method here
    // changes the multi-method Module execution state.
    status = module.load_method(first_decode_method);
    if (status != Error::Ok) {
      return Fail("failed to load initial decode method before prefill", status);
    }
  }

  if (!bench.manifest.empty() && !single_decode_method)
    return Fail("multi-case evaluation requires a single fixed-capacity decode method");
  for (const auto &eval_case : cases) {
    start_cache_length = eval_case.length;
    bench.forced_path = eval_case.forced;
    bench.forced.clear();
    if (bench.enabled()) {
      bench.output =
          eval_case.name.empty() ? base_output : (std::filesystem::path(base_output) / eval_case.name).string();
      if (!bench.forced_path.empty()) {
        std::vector<uint8_t> data;
        if (!ReadFile(bench.forced_path, &data) || data.size() != max_new_tokens * sizeof(int64_t))
          return Fail("forced token file length must equal max_new_tokens");
        bench.forced.resize(max_new_tokens);
        std::memcpy(bench.forced.data(), data.data(), data.size());
        for (auto value : bench.forced)
          if (value < 0) return Fail("negative forced token");
      }
    }
    const auto prefill_metadata = module.method_meta(argv[4]);
    if (!prefill_metadata.ok() || prefill_metadata->num_inputs() < 2) {
      return Fail("prefill method must have token IDs and positions as inputs 0 and 1");
    }
    auto prefill_tokens_metadata = prefill_metadata->input_tensor_meta(0);
    auto prefill_positions_metadata = prefill_metadata->input_tensor_meta(1);
    if (!prefill_tokens_metadata.ok() || !prefill_positions_metadata.ok() || !IsInt64Tensor(*prefill_tokens_metadata) ||
        !IsInt64Tensor(*prefill_positions_metadata)) {
      return Fail("prefill token IDs and positions must be int64 tensors");
    }
    std::vector<uint8_t> prefill_tokens;
    std::vector<uint8_t> prefill_positions;
    const std::vector<executorch::aten::SizesType> prefill_token_shape = {
        1, static_cast<executorch::aten::SizesType>(start_cache_length)};
    const std::vector<executorch::aten::SizesType> prefill_position_shape = {
        static_cast<executorch::aten::SizesType>(start_cache_length)};
    const size_t expected_prefill_bytes = ElementCount(prefill_token_shape) * sizeof(int64_t);
    const size_t expected_position_bytes = ElementCount(prefill_position_shape) * sizeof(int64_t);
    if (!ReadFile(eval_case.tokens, &prefill_tokens) || !ReadFile(eval_case.positions, &prefill_positions) ||
        prefill_tokens.size() != expected_prefill_bytes || prefill_positions.size() != expected_position_bytes) {
      return Fail("prefill input file size differs from method metadata");
    }
    // Load prefill before timed requests, as decode was loaded above.
    if (bench.enabled() && module.load_method(argv[4]) != Error::Ok)
      return Fail("failed to load prefill before evaluation");
    if (bench.enabled()) std::cerr << "LIGHT_EVAL ready " << bench.output << std::endl;
    for (int trial = 0; trial < bench.warmup + bench.repeats; ++trial) {
      if (bench.enabled()) {
        bench.begin();
        const auto reset = executorch_npu::ReleaseStateNamespace(device_id, argv[12]);
        if (reset != executorch_npu::StateStatus::kOk && reset != executorch_npu::StateStatus::kNotFound)
          return Fail("failed to reset evaluation state");
      }
      const bool sliding_mask = mask_window > 0;
      const size_t state_base = 2 + (sliding_mask ? 2 : (causal_mask ? 1 : 0));
      if (prefill_metadata->num_inputs() < state_base) {
        return Fail("prefill method has fewer inputs than the mask policy requires");
      }
      std::vector<uint8_t> prefill_sliding_data;
      std::vector<uint8_t> prefill_full_data;
      std::vector<executorch::aten::SizesType> prefill_mask_shape;
      if (sliding_mask || causal_mask) {
        auto mask_metadata = prefill_metadata->input_tensor_meta(2);
        if (!mask_metadata.ok() || mask_metadata->scalar_type() != ScalarType::BFloat16) {
          return Fail("prefill mask input must be bfloat16");
        }
        if (sliding_mask) {
          auto full_metadata = prefill_metadata->input_tensor_meta(3);
          if (!full_metadata.ok() || full_metadata->scalar_type() != ScalarType::BFloat16) {
            return Fail("prefill full mask input must be bfloat16");
          }
        }
        prefill_mask_shape = TensorShape(*mask_metadata);
        if (prefill_mask_shape.size() != 4 || prefill_mask_shape[0] != 1 || prefill_mask_shape[1] != 1 ||
            prefill_mask_shape[3] < start_cache_length) {
          return Fail("prefill mask shape must be [1,1,query,kv_capacity]");
        }
        if (bench.enabled() && prefill_mask_shape[3] != max_seq_len)
          return Fail("configured capacity differs from artifact mask capacity");
        prefill_mask_shape[2] = static_cast<executorch::aten::SizesType>(start_cache_length);
        FillCausalMasks(start_cache_length, static_cast<size_t>(prefill_mask_shape[3]), mask_window, &prefill_full_data,
                        sliding_mask ? &prefill_sliding_data : nullptr);
      }
      std::vector<std::vector<uint8_t>> prefill_state(prefill_metadata->num_inputs() - state_base);
      std::vector<executorch::extension::TensorPtr> prefill_tensors;
      std::vector<EValue> prefill_inputs;
      prefill_tensors.reserve(prefill_metadata->num_inputs());
      prefill_inputs.reserve(prefill_metadata->num_inputs());
      for (size_t index = 0; index < prefill_metadata->num_inputs(); ++index) {
        auto tensor_metadata = prefill_metadata->input_tensor_meta(index);
        if (!tensor_metadata.ok()) {
          return Fail("all prefill inputs must be tensors", tensor_metadata.error());
        }
        void *data = nullptr;
        if (index == 0) {
          data = prefill_tokens.data();
        } else if (index == 1) {
          data = prefill_positions.data();
        } else if (sliding_mask && index == 2) {
          data = prefill_sliding_data.data();
        } else if ((sliding_mask && index == 3) || (causal_mask && index == 2)) {
          data = prefill_full_data.data();
        } else {
          prefill_state[index - state_base].assign(tensor_metadata->nbytes(), 0);
          data = prefill_state[index - state_base].data();
        }
        const auto shape =
            index == 0 ? prefill_token_shape
                       : (index == 1 ? prefill_position_shape
                                     : (index < state_base ? prefill_mask_shape : TensorShape(*tensor_metadata)));
        prefill_tensors.push_back(from_blob(data, shape, tensor_metadata->scalar_type()));
        prefill_inputs.emplace_back(*prefill_tensors.back());
      }
      auto prefill_outputs = module.execute(argv[4], prefill_inputs);
      if (!prefill_outputs.ok() || prefill_outputs->empty() || !prefill_outputs->front().isTensor()) {
        return Fail("prefill execution failed", prefill_outputs.ok() ? Error::InvalidProgram : prefill_outputs.error());
      }
      int64_t token = 0;
      if (!(sampling ? SampleToken(prefill_outputs->front().toTensor(), temperature, top_k, &generator, &token)
                     : GreedyToken(prefill_outputs->front().toTensor(), &token))) {
        return Fail("prefill logits must be valid float32 or bfloat16 with vocabulary last");
      }

      if (bench.enabled() && !bench.observe(prefill_outputs->front().toTensor(), &token))
        return Fail("invalid prefill logits during evaluation");
      size_t emitted_tokens = 0;
      if (max_new_tokens > 0) {
        if (!bench.enabled()) std::cout << "ASCEND_SESSION_TOKEN index=0 token_id=" << token << '\n';
        emitted_tokens = 1;
        if (executorch_npu::session_sampling::ShouldStop(token, seen_eos_token, eos_token)) {
          std::cout << "ASCEND_SESSION_OK generated_tokens=" << emitted_tokens << '\n';
          return 0;
        }
      }
      for (int generated = 1; generated < max_new_tokens; ++generated) {
        const int cache_length = start_cache_length + static_cast<int>((generated - 1) * decode_elements);
        const std::string decode_method =
            single_decode_method ? std::string(argv[7]) : std::string(argv[7]) + "_" + std::to_string(cache_length);
        const auto decode_metadata = module.method_meta(decode_method);
        if (!decode_metadata.ok()) {
          return Fail("failed to inspect decode method metadata", decode_metadata.error());
        }
        if (decode_metadata->num_inputs() < 2) {
          return Fail("decode method has fewer than two inputs");
        }
        auto token_metadata = decode_metadata->input_tensor_meta(0);
        auto position_metadata = decode_metadata->input_tensor_meta(1);
        if (!token_metadata.ok() || !position_metadata.ok() || !IsInt64Tensor(*token_metadata) ||
            !IsInt64Tensor(*position_metadata) ||
            (!shared_model_method && (TensorShape(*token_metadata) != decode_shape ||
                                      ElementCount(TensorShape(*position_metadata)) != decode_elements))) {
          return Fail("decode inputs 0 and 1 do not match the session ABI");
        }
        constexpr size_t token_input_index = 0;
        constexpr size_t position_input_index = 1;
        std::vector<int64_t> tokens(decode_elements, token);
        std::vector<int64_t> positions(decode_elements);
        for (size_t index = 0; index < decode_elements; ++index) {
          positions[index] = static_cast<int64_t>(cache_length) + index;
        }
        std::vector<std::vector<uint8_t>> state(decode_metadata->num_inputs());
        std::vector<executorch::extension::TensorPtr> tensors;
        std::vector<EValue> inputs;
        tensors.reserve(decode_metadata->num_inputs());
        inputs.reserve(decode_metadata->num_inputs());
        for (size_t index = 0; index < decode_metadata->num_inputs(); ++index) {
          auto tensor_metadata = decode_metadata->input_tensor_meta(index);
          if (!tensor_metadata.ok()) {
            return Fail("all decode inputs must be tensors", tensor_metadata.error());
          }
          void *data = nullptr;
          if (index == token_input_index) {
            data = tokens.data();
          } else if (index == position_input_index) {
            data = positions.data();
          } else {
            state[index].assign(tensor_metadata->nbytes(), 0);
            data = state[index].data();
          }
          const auto shape = shared_model_method && index < 2 ? decode_shape : TensorShape(*tensor_metadata);
          tensors.push_back(from_blob(data, shape, tensor_metadata->scalar_type()));
          inputs.emplace_back(*tensors.back());
        }
        auto outputs = module.execute(decode_method, inputs);
        if (!outputs.ok() || outputs->empty() || !outputs->front().isTensor()) {
          return Fail("decode execution failed", outputs.ok() ? Error::InvalidProgram : outputs.error());
        }
        if (!(sampling ? SampleToken(outputs->front().toTensor(), temperature, top_k, &generator, &token)
                       : GreedyToken(outputs->front().toTensor(), &token))) {
          return Fail("decode logits must be valid float32 or bfloat16 with vocabulary last");
        }
        if (bench.enabled() && !bench.observe(outputs->front().toTensor(), &token))
          return Fail("invalid decode logits during evaluation");
        if (!bench.enabled()) std::cout << "ASCEND_SESSION_TOKEN index=" << generated << " token_id=" << token << '\n';
        emitted_tokens = static_cast<size_t>(generated + 1);
        if (executorch_npu::session_sampling::ShouldStop(token, seen_eos_token, eos_token)) {
          break;
        }
      }
      if (bench.enabled()) {
        if (!bench.finish(trial)) return Fail("failed to save evaluation outputs");
      } else {
        std::cout << "ASCEND_SESSION_OK generated_tokens=" << emitted_tokens << '\n';
      }
    }  // repeated requests, one resident Module
  }  // all evaluation cases share the loaded model
  return 0;
}

int RunBatch(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "usage: " << argv[0]
              << " batch MODEL.pte WEIGHTS.ptd PREFILL_METHOD MANIFEST "
                 "DEVICE_ID STATE_NAMESPACE --max-seq-len VALUE "
                 "[--choice-token-ids A,B,C,D] [--mask-window W | --causal-mask]\n";
    return 2;
  }
  int device_id = 0;
  int max_seq_len = 0;
  if (!ParseInt(argv[6], &device_id) || device_id < 0 || std::string(argv[7]).empty()) {
    return Fail("invalid batch device or state namespace");
  }
  if (argc < 10 || std::string(argv[8]) != "--max-seq-len" || !ParseInt(argv[9], &max_seq_len) || max_seq_len <= 0) {
    return Fail("batch requires --max-seq-len VALUE");
  }
  std::array<size_t, 4> choice_ids = {32, 33, 34, 35};
  // Build attention masks on the host to keep GE from constructing them in-graph.
  int mask_window = 0;
  bool causal_mask = false;
  for (int index = 10; index < argc;) {
    const std::string option = argv[index];
    if (option == "--choice-token-ids" && index + 1 < argc) {
      std::vector<executorch::aten::SizesType> parsed_ids;
      if (!ParseShape(argv[index + 1], &parsed_ids) || parsed_ids.size() != 4) {
        return Fail("choice token IDs must contain four comma-separated values");
      }
      for (size_t slot = 0; slot < choice_ids.size(); ++slot) {
        choice_ids[slot] = static_cast<size_t>(parsed_ids[slot]);
      }
      index += 2;
    } else if (option == "--mask-window" && index + 1 < argc) {
      if (!ParseInt(argv[index + 1], &mask_window) || mask_window < 0) {
        return Fail("invalid --mask-window");
      }
      index += 2;
    } else if (option == "--causal-mask") {
      causal_mask = true;
      ++index;
    } else {
      return Fail("unknown batch option");
    }
  }
  if (causal_mask && mask_window > 0) {
    return Fail("choose either --causal-mask or --mask-window");
  }

  std::ifstream manifest(argv[5]);
  if (!manifest) {
    return Fail("failed to open batch manifest");
  }
  executorch::runtime::runtime_init();
  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", device_id) != Error::Ok ||
      ascend_options.set_option("state_namespace", argv[7]) != Error::Ok) {
    return Fail("failed to create Ascend backend options");
  }
  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) != Error::Ok) {
    return Fail("failed to install Ascend backend options");
  }
  StateNamespaceGuard state_guard(device_id, argv[7]);
  Module module(argv[2], argv[3], Module::LoadMode::File, nullptr, nullptr, nullptr, true);
  Error status = module.load(backend_options);
  if (status != Error::Ok) {
    return Fail("failed to load PTE", status);
  }
  const auto metadata = module.method_meta(argv[4]);
  if (!metadata.ok() || metadata->num_inputs() < 2) {
    return Fail("prefill method must have token IDs and positions");
  }
  auto token_metadata = metadata->input_tensor_meta(0);
  auto position_metadata = metadata->input_tensor_meta(1);
  if (!token_metadata.ok() || !position_metadata.ok() || !IsInt64Tensor(*token_metadata) ||
      !IsInt64Tensor(*position_metadata)) {
    return Fail("batch prefill inputs must be int64 tensors");
  }

  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream fields(line);
    int sample_id = 0;
    int seq_len = 0;
    std::string token_path;
    std::string position_path;
    if (!(fields >> sample_id >> token_path >> position_path >> seq_len) || sample_id < 0 || seq_len < 1 ||
        seq_len > max_seq_len) {
      return Fail("invalid batch manifest line");
    }
    std::vector<uint8_t> token_data;
    std::vector<uint8_t> position_data;
    if (!ReadFile(token_path, &token_data) || !ReadFile(position_path, &position_data) ||
        token_data.size() != static_cast<size_t>(seq_len) * sizeof(int64_t) ||
        position_data.size() != static_cast<size_t>(seq_len) * sizeof(int64_t)) {
      return Fail("batch input size differs from sequence length");
    }
    const std::vector<executorch::aten::SizesType> token_shape = {1, static_cast<executorch::aten::SizesType>(seq_len)};
    const std::vector<executorch::aten::SizesType> position_shape = {static_cast<executorch::aten::SizesType>(seq_len)};
    const bool synth_masks = mask_window > 0;
    const size_t state_base = 2 + (synth_masks ? 2 : (causal_mask ? 1 : 0));
    if (metadata->num_inputs() < state_base) {
      return Fail("batch method has fewer inputs than the manifest describes");
    }
    std::vector<uint8_t> sliding_data;
    std::vector<uint8_t> full_data;
    std::vector<executorch::aten::SizesType> mask_shape;
    if (synth_masks || causal_mask) {
      auto first_mask_metadata = metadata->input_tensor_meta(2);
      if (!first_mask_metadata.ok() || first_mask_metadata->scalar_type() != executorch::aten::ScalarType::BFloat16) {
        return Fail("synthesized masks must be bfloat16");
      }
      if (synth_masks) {
        auto full_metadata = metadata->input_tensor_meta(3);
        if (!full_metadata.ok() || full_metadata->scalar_type() != executorch::aten::ScalarType::BFloat16) {
          return Fail("full mask input must be bfloat16");
        }
      }
      const auto &sizes = first_mask_metadata->sizes();
      if (sizes.size() < 2) {
        return Fail("synthesized mask has invalid rank");
      }
      mask_shape.assign(sizes.begin(), sizes.end());
      mask_shape[mask_shape.size() - 2] = static_cast<executorch::aten::SizesType>(seq_len);
      const size_t kv_len = static_cast<size_t>(mask_shape.back());
      FillCausalMasks(seq_len, kv_len, mask_window, &full_data, synth_masks ? &sliding_data : nullptr);
    }
    std::vector<std::vector<uint8_t>> state(metadata->num_inputs() - state_base);
    std::vector<executorch::extension::TensorPtr> tensors;
    std::vector<EValue> inputs;
    tensors.reserve(metadata->num_inputs());
    inputs.reserve(metadata->num_inputs());
    for (size_t index = 0; index < metadata->num_inputs(); ++index) {
      auto input_metadata = metadata->input_tensor_meta(index);
      if (!input_metadata.ok()) {
        return Fail("all batch prefill inputs must be tensors");
      }
      void *data = nullptr;
      const std::vector<executorch::aten::SizesType> *shape = nullptr;
      if (index == 0) {
        data = token_data.data();
        shape = &token_shape;
      } else if (index == 1) {
        data = position_data.data();
        shape = &position_shape;
      } else if (synth_masks && index == 2) {
        data = sliding_data.data();
        shape = &mask_shape;
      } else if ((synth_masks && index == 3) || (causal_mask && index == 2)) {
        data = full_data.data();
        shape = &mask_shape;
      } else {
        state[index - state_base].assign(input_metadata->nbytes(), 0);
        data = state[index - state_base].data();
      }
      tensors.push_back(
          from_blob(data, shape == nullptr ? TensorShape(*input_metadata) : *shape, input_metadata->scalar_type()));
      inputs.emplace_back(*tensors.back());
    }
    auto outputs = module.execute(argv[4], inputs);
    if (!outputs.ok() || outputs->empty()) {
      return Fail("batch prefill execution failed");
    }
    // Stateful methods may return cache deltas alongside logits. Select the
    // tensor with the largest final dimension as the vocabulary logits.
    size_t logits_index = outputs->size();
    executorch::aten::SizesType logits_width = -1;
    for (size_t index = 0; index < outputs->size(); ++index) {
      const EValue &value = (*outputs)[index];
      if (!value.isTensor()) {
        continue;
      }
      const auto &tensor = value.toTensor();
      const auto width = tensor.dim() > 0 ? tensor.size(tensor.dim() - 1) : 0;
      if (logits_index == outputs->size() || width > logits_width) {
        logits_index = index;
        logits_width = width;
      }
    }
    if (logits_index == outputs->size()) {
      return Fail("batch prefill produced no tensor output");
    }
    float scores[4] = {};
    if (!ChoiceScores((*outputs)[logits_index].toTensor(), choice_ids, scores)) {
      return Fail("batch logits do not contain valid choice scores");
    }
    std::cout << "ASCEND_BATCH_SCORES id=" << sample_id << " A=" << scores[0] << " B=" << scores[1]
              << " C=" << scores[2] << " D=" << scores[3] << '\n';
    std::cout.flush();
    const auto release_status = executorch_npu::ReleaseStateNamespace(device_id, argv[7]);
    if (release_status != executorch_npu::StateStatus::kOk &&
        release_status != executorch_npu::StateStatus::kNotFound) {
      return Fail("failed to reset batch state namespace");
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "session") {
    return RunSession(argc, argv);
  }
  if (argc > 1 && std::string(argv[1]) == "batch") {
    return RunBatch(argc, argv);
  }
  if (argc < 7) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.pte WEIGHTS.ptd METHOD DEVICE_ID STATE_NAMESPACE "
                 "OUTPUT_DIR [ORDINARY_INPUT.bin ...]\n";
    return 2;
  }

  int device_id = 0;
  try {
    device_id = std::stoi(argv[4]);
  } catch (const std::exception &) {
    return Fail("device_id must be an integer");
  }
  const std::string method_name = argv[3];
  const char *state_namespace = argv[5];
  if (device_id < 0 || method_name.empty() || state_namespace[0] == '\0') {
    return Fail("method, non-negative device_id, and state_namespace are required");
  }

  executorch::runtime::runtime_init();
  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", device_id) != Error::Ok ||
      ascend_options.set_option("state_namespace", state_namespace) != Error::Ok) {
    return Fail("failed to create Ascend backend options");
  }
  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) != Error::Ok) {
    return Fail("failed to install Ascend backend options");
  }

  Module module(argv[1], argv[2], Module::LoadMode::File, nullptr, nullptr, nullptr, true);
  Error status = module.load(backend_options);
  if (status != Error::Ok) {
    return Fail("failed to load PTE", status);
  }
  auto metadata = module.method_meta(method_name);
  if (!metadata.ok()) {
    return Fail("failed to inspect method metadata", metadata.error());
  }

  const size_t ordinary_input_count = static_cast<size_t>(argc - 7);
  if (ordinary_input_count > metadata->num_inputs()) {
    return Fail("more ordinary input files than method inputs");
  }

  std::vector<std::vector<uint8_t>> input_storage(metadata->num_inputs());
  std::vector<executorch::extension::TensorPtr> input_tensors;
  std::vector<EValue> inputs;
  input_tensors.reserve(metadata->num_inputs());
  inputs.reserve(metadata->num_inputs());
  for (size_t index = 0; index < metadata->num_inputs(); ++index) {
    auto tensor_metadata = metadata->input_tensor_meta(index);
    if (!tensor_metadata.ok()) {
      return Fail("all method inputs must be tensors", tensor_metadata.error());
    }
    if (index < ordinary_input_count) {
      if (!ReadFile(argv[7 + index], &input_storage[index])) {
        return Fail("failed to read ordinary input file: " + std::string(argv[7 + index]));
      }
      if (input_storage[index].size() != tensor_metadata->nbytes()) {
        return Fail("ordinary input file size differs from method metadata");
      }
    } else {
      // The backend replaces state_current buffers with its state cache.
      input_storage[index].assign(tensor_metadata->nbytes(), 0);
    }
    input_tensors.push_back(
        from_blob(input_storage[index].data(), TensorShape(*tensor_metadata), tensor_metadata->scalar_type()));
    inputs.emplace_back(*input_tensors.back());
  }

  auto outputs = module.execute(method_name, inputs);
  if (!outputs.ok()) {
    return Fail("Ascend execution failed", outputs.error());
  }
  if (outputs->size() != metadata->num_outputs()) {
    return Fail("runtime output count differs from method metadata");
  }

  const size_t state_input_count = metadata->num_inputs() - ordinary_input_count;
  const size_t total_output_count = metadata->num_outputs();
  // Some states may have next_output_arg = -1 (not updated), so the number of
  // state outputs can be less than state_input_count.
  const size_t ordinary_output_count =
      total_output_count > state_input_count ? total_output_count - state_input_count : 0;
  std::error_code error;
  std::filesystem::create_directories(argv[6], error);
  if (error) {
    return Fail("failed to create output directory: " + error.message());
  }
  for (size_t index = 0; index < ordinary_output_count; ++index) {
    if (!outputs->at(index).isTensor()) {
      return Fail("non-state output is not a tensor");
    }
    const auto &output = outputs->at(index).toTensor();
    const std::filesystem::path path = std::filesystem::path(argv[6]) / ("output_" + std::to_string(index) + ".bin");
    if (!WriteFile(path, output.const_data_ptr(), output.nbytes())) {
      return Fail("failed to write output file: " + path.string());
    }
    std::cout << "ASCEND_STATEFUL_OUTPUT index=" << index << " path=" << path.string() << " bytes=" << output.nbytes()
              << '\n';
  }
  std::cout << "ASCEND_STATEFUL_OK method=" << method_name << " ordinary_outputs=" << ordinary_output_count
            << " synthesized_state_inputs=" << state_input_count << '\n';
  return 0;
}
