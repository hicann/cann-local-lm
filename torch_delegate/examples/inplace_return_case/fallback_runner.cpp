/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * Inplace+Return 测试 - C++ Runner
 *
 * 加载 epair 文件并执行，支持多组输入和动态 shape。
 * 单 tensor 输入 (x)，scale_val 作为 Const 节点内嵌在 GE 图中。
 *
 * 用法:
 *   ./fallback_runner [--custom-ops <path>] <epair>
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

const char *kStageSep = "========================================";

void PrintStage(size_t index, size_t total, const std::string &name) {
  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " [" << index << "/" << total << "] " << name << std::endl;
  std::cout << kStageSep << std::endl;
}

std::string Dirname(const std::string &path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos) return ".";
  return path.substr(0, pos);
}

std::string Basename(const std::string &path) {
  auto pos = path.rfind('/');
  std::string name = (pos == std::string::npos) ? path : path.substr(pos + 1);
  auto dot = name.rfind('.');
  if (dot != std::string::npos) name = name.substr(0, dot);
  return name;
}

std::vector<int64_t> ReadShapeFile(const std::string &path) {
  std::vector<int64_t> shape;
  std::ifstream file(path);
  if (!file) return shape;
  std::string line;
  std::getline(file, line);
  std::istringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    shape.push_back(std::stoll(token));
  }
  return shape;
}

std::vector<int> FindInputGroups(const std::string &dir, const std::string &prefix) {
  std::vector<int> groups;
  for (int i = 0; i < 100; ++i) {
    std::string x_path = dir + "/input_" + prefix + "_" + std::to_string(i) + "_x.bin";
    std::ifstream test(x_path);
    if (!test.good()) break;
    groups.push_back(i);
  }
  return groups;
}

gert::Tensor LoadTensorFromFile(const std::string &bin_path, const std::string &shape_path) {
  std::vector<int64_t> dims = ReadShapeFile(shape_path);
  if (dims.empty()) {
    throw std::runtime_error("Failed to read shape file: " + shape_path);
  }

  size_t element_count = 1;
  for (auto d : dims) element_count *= d;
  size_t data_size = element_count * sizeof(float);

  std::vector<float> data(element_count);
  std::ifstream file(bin_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open input file: " + bin_path);
  }
  file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data_size));

  void *device_ptr = nullptr;
  aclError err = aclrtMalloc(&device_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (err != ACL_ERROR_NONE) {
    throw std::runtime_error("aclrtMalloc failed");
  }
  err = aclrtMemcpy(device_ptr, data_size, data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (err != ACL_ERROR_NONE) {
    aclrtFree(device_ptr);
    throw std::runtime_error("aclrtMemcpy H2D failed");
  }

  gert::Tensor tensor;
  auto &shape = tensor.GetShape();
  for (auto d : dims) {
    shape.MutableOriginShape().AppendDim(d);
    shape.MutableStorageShape().AppendDim(d);
  }
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  tensor.SetDataType(ge::DT_FLOAT);
  gert::TensorData tensor_data(device_ptr, nullptr, data_size, gert::kOnDeviceHbm);
  tensor.SetData(std::move(tensor_data));

  return tensor;
}

bool RunEpairModel(const std::string &epair_path, const std::string &prefix) {
  std::string dir = Dirname(epair_path);

  auto input_groups = FindInputGroups(dir, prefix);
  if (input_groups.empty()) {
    std::cerr << "[WARN] No input files found: " << dir << "/input_" << prefix << "_*_x.bin" << std::endl;
    return true;
  }

  std::cout << std::endl;
  std::cout << "Loading: " << epair_path << std::endl;
  std::cout << "Input group count: " << input_groups.size() << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  td::EpairModelLoader loader(epair_path.c_str());

  ge::Graph test_graph("test");
  td::Status init_check = loader.GetGEGraph(&test_graph);
  if (init_check != td::SUCCESS) {
    std::cerr << "[ERROR] EpairModelLoader initialization failed, status code: " << init_check << std::endl;
    return false;
  }
  std::cout << "  ✓ epair loaded successfully" << std::endl;

  static aclrtStream stream = nullptr;
  if (stream == nullptr) {
    aclError acl_ret = aclrtCreateStream(&stream);
    if (acl_ret != ACL_ERROR_NONE) {
      std::cerr << "[ERROR] aclrtCreateStream failed, error: " << acl_ret << std::endl;
      return false;
    }
  }
  std::cout << "  ✓ ACL stream created successfully" << std::endl;

  std::map<ge::AscendString, ge::AscendString> session_options = {};
  td::Status compile_status = loader.CompileAndLoad(session_options, stream);
  if (compile_status != td::SUCCESS) {
    std::cerr << "[ERROR] CompileAndLoad failed, status: " << compile_status << std::endl;
    aclrtDestroyStream(stream);
    return false;
  }
  std::cout << "  ✓ CompileAndLoad succeeded" << std::endl;

  for (size_t group_idx = 0; group_idx < input_groups.size(); ++group_idx) {
    int group = input_groups[group_idx];
    std::string x_bin = dir + "/input_" + prefix + "_" + std::to_string(group) + "_x.bin";
    std::string x_shape = dir + "/input_" + prefix + "_" + std::to_string(group) + "_x_shape.txt";

    gert::Tensor x_tensor = LoadTensorFromFile(x_bin, x_shape);
    std::vector<int64_t> x_dims = ReadShapeFile(x_shape);
    std::cout << "  [DEBUG] x shape: [";
    for (size_t i = 0; i < x_dims.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << x_dims[i];
    }
    std::cout << "]" << std::endl;

    // 读取 scale_val 用于调试输出（但不作为图输入，它是 Const 节点）
    std::string scale_path = dir + "/input_" + prefix + "_" + std::to_string(group) + "_scale.txt";
    float scale_val = 1.0f;
    {
      std::ifstream scale_file(scale_path);
      if (scale_file) {
        scale_file >> scale_val;
      }
    }
    std::cout << "  [DEBUG] scale_val: " << scale_val << std::endl;
    std::cout << "  [INFO] scale_val is embedded in the GE graph as a Const node, not used as a runtime input"
              << std::endl;

    // 构建输入列表（只传 x，scale_val 是 Const 节点）
    std::vector<gert::Tensor> inputs;
    inputs.push_back(std::move(x_tensor));

    std::vector<gert::Tensor> outputs;
    td::Status run_status = loader.RunModelWithStreamAsync(stream, inputs, outputs);

    if (run_status != td::SUCCESS) {
      std::cerr << "[ERROR] RunModelWithStreamAsync failed (group " << group << "), status: " << run_status
                << std::endl;
      aclrtDestroyStream(stream);
      return false;
    }

    std::cout << "  ✓ Input group " << group << " executed successfully (outputs=" << outputs.size() << ")"
              << std::endl;

    // 保存所有输出
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::string out_path =
          dir + "/delegate_output_" + prefix + "_" + std::to_string(group) + "_" + std::to_string(i) + ".bin";

      auto out_shape = outputs[i].GetStorageShape();
      size_t data_cnt = 1;
      for (size_t d = 0; d < out_shape.GetDimNum(); ++d) {
        data_cnt *= out_shape.GetDim(d);
      }
      size_t data_size = data_cnt * sizeof(float);

      if (data_cnt == 0 || data_size > 1024ULL * 1024 * 1024) {
        std::cerr << "[ERROR] Invalid output data_cnt: " << data_cnt
                  << ", expected 1 <= data_cnt <= 268435456 (data size <= 1 GB)" << std::endl;
        aclrtDestroyStream(stream);
        return false;
      }

      std::vector<float> host_buf(data_cnt);
      aclrtMemcpy(host_buf.data(), data_size, outputs[i].GetAddr(), data_size, ACL_MEMCPY_DEVICE_TO_HOST);

      std::ofstream out_file(out_path, std::ios::binary);
      out_file.write(reinterpret_cast<const char *>(host_buf.data()), static_cast<std::streamsize>(data_size));

      std::string out_shape_path = out_path.substr(0, out_path.size() - 4) + "_shape.txt";
      std::ofstream shape_file(out_shape_path);
      for (size_t d = 0; d < out_shape.GetDimNum(); ++d) {
        if (d > 0) shape_file << ",";
        shape_file << out_shape.GetDim(d);
      }

      std::cout << "    output[" << i << "]: shape=[";
      for (size_t d = 0; d < out_shape.GetDimNum(); ++d) {
        if (d > 0) std::cout << ",";
        std::cout << out_shape.GetDim(d);
      }
      std::cout << "], size=" << data_cnt << " elements → " << out_path << std::endl;
    }
  }

  aclrtDestroyStream(stream);

  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string custom_ops_path;
  std::vector<std::string> epair_paths;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--custom-ops" && i + 1 < argc) {
      custom_ops_path = argv[++i];
    } else {
      epair_paths.push_back(argv[i]);
    }
  }

  if (epair_paths.empty()) {
    std::cerr << "[ERROR] Usage: " << argv[0] << " [--custom-ops <path>] <epair1> [<epair2> ...]" << std::endl;
    return -1;
  }

  if (!custom_ops_path.empty()) {
    std::cout << "Loading custom op library: " << custom_ops_path << std::endl;
    void *handle = dlopen(custom_ops_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "[ERROR] dlopen failed: " << dlerror() << std::endl;
      return -1;
    }
    std::cout << "  ✓ Custom op library loaded successfully" << std::endl;
  }

  PrintStage(1, 2, "Initialize GE");
  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
  auto init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[ERROR] GEInitializeV2 failed, status: " << init_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE initialized successfully" << std::endl;

  aclError acl_ret = aclrtSetDevice(0);
  if (acl_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtSetDevice failed, error: " << acl_ret << std::endl;
    return -1;
  }

  PrintStage(2, 2, "Run models");
  bool all_pass = true;

  for (const auto &epair_path : epair_paths) {
    std::string model_name = Basename(epair_path);
    std::string prefix = model_name;
    if (prefix.substr(0, 6) == "model_") prefix = prefix.substr(6);

    std::cout << std::endl;
    std::cout << kStageSep << std::endl;
    std::cout << " Running: " << model_name << " (prefix=" << prefix << ")" << std::endl;
    std::cout << kStageSep << std::endl;

    if (!RunEpairModel(epair_path, prefix)) {
      std::cerr << "[FAIL] " << model_name << " execution failed" << std::endl;
      all_pass = false;
    } else {
      std::cout << "[PASS] " << model_name << " execution succeeded" << std::endl;
    }
  }

  ge::GEFinalizeV2();

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  if (all_pass) {
    std::cout << " All models executed successfully" << std::endl;
  } else {
    std::cout << " Some models failed to execute" << std::endl;
  }
  std::cout << kStageSep << std::endl;

  return all_pass ? 0 : -1;
}
