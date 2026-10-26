/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class ModelType { Simple, Chain, Param, Unknown };

ModelType ParseModelType(const std::string &name) {
  if (name == "simple") return ModelType::Simple;
  if (name == "chain") return ModelType::Chain;
  if (name == "param") return ModelType::Param;
  return ModelType::Unknown;
}

std::string ModelTypeName(ModelType t) {
  switch (t) {
    case ModelType::Simple:
      return "simple";
    case ModelType::Chain:
      return "chain";
    case ModelType::Param:
      return "param";
    default:
      return "unknown";
  }
}

int64_t ModelInputCount(ModelType t) {
  return (t == ModelType::Simple) ? 2 : 1;
}

// All models use float32 {4,4} input.
std::vector<int64_t> kInputShape = {4, 4};
int64_t kInputNumel = 4 * 4;

float MaxAbsDiff(const at::Tensor &a, const at::Tensor &b) {
  return a.sub(b).abs().max().item<float>();
}

std::string TensorStr(const at::Tensor &t, size_t limit = 8) {
  auto f = t.flatten();
  size_t n = f.numel();
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < std::min(n, limit); ++i) {
    if (i > 0) os << ", ";
    os << f[i].item<float>();
  }
  if (n > limit) os << ", ...";
  os << "]";
  return os.str();
}

std::vector<at::Tensor> ReadInputBin(const std::string &input_dir, ModelType t, const std::string &model_name) {
  int64_t count = ModelInputCount(t);
  std::vector<at::Tensor> inputs;
  for (int64_t i = 0; i < count; ++i) {
    std::string path = input_dir + "/" + model_name + "_input_" + std::to_string(i) + ".bin";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "[ERROR] Cannot open: " << path << std::endl;
      std::exit(1);
    }
    std::vector<float> buf(kInputNumel);
    file.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(kInputNumel * sizeof(float)));
    if (!file) {
      std::cerr << "[ERROR] Failed to read " << path << std::endl;
      std::exit(1);
    }
    auto t = at::from_blob(buf.data(), kInputShape, at::kFloat).clone();
    inputs.push_back(t);
  }
  return inputs;
}

void WriteTensorBin(const std::string &path, const at::Tensor &t) {
  auto f = t.flatten().contiguous();
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char *>(f.data_ptr()), static_cast<std::streamsize>(f.numel() * f.element_size()));
  if (!file) {
    std::cerr << "[ERROR] Failed to write: " << path << std::endl;
    std::exit(1);
  }
}

void PrintUsage(const char *prog) {
  std::cerr << "Usage: " << prog << std::endl;
  std::cerr << "  --model-path <pt2_path>" << std::endl;
  std::cerr << "  --model-name <simple|chain|param>" << std::endl;
  std::cerr << "  [--input-dir <dir>]   (default: generate inputs with seed=42)" << std::endl;
  std::cerr << "  [--output-dir <dir>]  (write aoti_output/input_<idx>.bin)" << std::endl;
}

}  // anonymous namespace

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string model_name;
  std::string input_dir;
  std::string output_dir;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--model-path" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--model-name" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--input-dir" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "[ERROR] Unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (model_path.empty() || model_name.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  ModelType t = ParseModelType(model_name);
  if (t == ModelType::Unknown) {
    std::cerr << "[ERROR] Unknown model: " << model_name << std::endl;
    return 1;
  }

  std::cout << "============================================" << std::endl;
  std::cout << " Verify AOTI Load (C++)" << std::endl;
  std::cout << " Model: " << ModelTypeName(t) << std::endl;
  std::cout << " Path:  " << model_path << std::endl;
  std::cout << "============================================" << std::endl;

  // ---- Step 1: Load AOTI model ----
  std::cout << "\n[Step 1] Loading AOTI package..." << std::endl;
  torch::inductor::AOTIModelPackageLoader loader(model_path);
  std::cout << "[PASS] AOTI model loaded" << std::endl;

  // ---- Step 2: Get inputs ----
  std::cout << "\n[Step 2] Getting inputs..." << std::endl;
  std::vector<at::Tensor> inputs;
  if (!input_dir.empty()) {
    std::cout << "  reading from: " << input_dir << std::endl;
    inputs = ReadInputBin(input_dir, t, model_name);
  } else {
    std::cout << "  generating with seed=42" << std::endl;
    at::manual_seed(42);
    inputs.push_back(at::randn({4, 4}, at::kFloat));
    if (t == ModelType::Simple) {
      inputs.push_back(at::randn({4, 4}, at::kFloat));
    }
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::cout << "  input[" << i << "] data=" << TensorStr(inputs[i]) << std::endl;
  }

  // Save copies for reference computation.
  std::vector<at::Tensor> saved_inputs;
  for (auto &in : inputs) {
    saved_inputs.push_back(in.clone());
  }

  // ---- Step 3: Run AOTI model ----
  std::cout << "\n[Step 3] Running AOTI model..." << std::endl;
  auto outputs = loader.run(inputs);
  std::cout << "[PASS] Model executed" << std::endl;
  std::cout << "  output[0] data=" << TensorStr(outputs[0]) << std::endl;

  // ---- Step 4: Write results to .bin files ----
  if (!output_dir.empty()) {
    std::cout << "\n[Step 4] Writing results to: " << output_dir << std::endl;
    WriteTensorBin(output_dir + "/" + model_name + "_aoti_output_0.bin", outputs[0]);
    for (size_t i = 0; i < inputs.size(); ++i) {
      WriteTensorBin(output_dir + "/" + model_name + "_aoti_input_" + std::to_string(i) + ".bin", inputs[i]);
    }
    std::cout << "  wrote " << model_name << "_aoti_output_0.bin" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::cout << "  wrote " << model_name << "_aoti_input_" << i << ".bin" << std::endl;
    }
  }

  // ---- Step 5: Compare output vs reference ----
  if (!output_dir.empty()) {
    std::cout << "\n[Step 5] Comparing..." << std::endl;
  } else {
    std::cout << "\n[Step 4] Comparing..." << std::endl;
  }

  // Reference: what the output SHOULD be
  at::Tensor x_ref = saved_inputs[0];
  at::Tensor output_ref;
  if (t == ModelType::Simple) {
    output_ref = (saved_inputs[0] + saved_inputs[1]) * 2.0f;
  } else if (t == ModelType::Chain) {
    output_ref = ((x_ref + 1.0f) * 2.0f) / 4.0f;
  } else {  // Param
    output_ref = (x_ref + 1.0f) * 2.0f;
  }

  float max_diff = MaxAbsDiff(outputs[0], output_ref);
  std::cout << "  output max diff: " << max_diff << std::endl;

  constexpr float kTolerance = 1e-5f;
  bool output_ok = max_diff < kTolerance;
  if (output_ok) {
    std::cout << "  [PASS] Output matches reference" << std::endl;
  } else {
    std::cout << "  [FAIL] Output differs from reference" << std::endl;
  }

  // ---- Step 6: Inplace verification ----
  std::cout << "\n[Step 6] Inplace verification..." << std::endl;
  bool inplace_ok = true;
  for (size_t i = 0; i < inputs.size(); ++i) {
    at::Tensor expected;
    if (t == ModelType::Simple && i == 0) {
      expected = (saved_inputs[0] + saved_inputs[1]) * 2.0f;
    } else if (t == ModelType::Simple && i == 1) {
      expected = saved_inputs[1].clone();  // y is unchanged
    } else if (t == ModelType::Chain) {
      expected = ((saved_inputs[0] + 1.0f) * 2.0f) / 4.0f;
    } else {  // Param
      expected = (saved_inputs[0] + 1.0f) * 2.0f;
    }

    float diff = MaxAbsDiff(inputs[i], expected);
    bool ok = diff < kTolerance;
    std::cout << "  input[" << i << "] modified? " << (diff > 1e-7f ? "yes" : "no") << " max diff from eager: " << diff
              << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;
    if (!ok) inplace_ok = false;
  }

  bool passed = output_ok && inplace_ok;
  std::cout << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << " [" << ModelTypeName(t) << "] Verification: " << (passed ? "PASS" : "FAIL") << std::endl;
  std::cout << "============================================" << std::endl;

  return passed ? 0 : 1;
}
