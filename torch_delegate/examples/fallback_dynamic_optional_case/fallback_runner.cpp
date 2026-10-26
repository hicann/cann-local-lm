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
 * Fallback Dynamic/Optional 测试 - C++ Runner
 *
 * 加载 epair 文件并执行，支持动态输入 (Tensor[]) 和可选输入 (Tensor?)。
 *
 * 输入文件命名规则: input_{prefix}_{name}.bin + input_{prefix}_{name}_shape.txt
 *
 * 用法:
 *   ./fallback_runner [--custom-ops <path>] <epair1> [<epair2> ...]
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

std::vector<std::string> DiscoverInputNames(const std::string &dir, const std::string &prefix) {
  std::vector<std::string> names;
  std::string pattern = "input_" + prefix + "_";
  for (int attempt = 0; attempt < 20; ++attempt) {
    for (const auto &name : {"x", "y", "input", "mat1", "mat2"}) {
      std::string bin = dir + "/" + pattern + name + ".bin";
      std::ifstream test(bin);
      if (test.good()) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
          names.push_back(name);
        }
      }
    }
    break;
  }
  return names;
}

bool RunEpairModel(const std::string &epair_path, const std::string &prefix) {
  std::string dir = Dirname(epair_path);

  auto input_names = DiscoverInputNames(dir, prefix);
  if (input_names.empty()) {
    std::cerr << "[WARN] 未找到输入文件: " << dir << "/input_" << prefix << "_*.bin" << std::endl;
    return false;
  }

  std::cout << std::endl;
  std::cout << "加载: " << epair_path << std::endl;
  std::cout << "输入数量: " << input_names.size() << std::endl;
  for (const auto &name : input_names) {
    std::cout << "  - " << name << " (" << dir << "/input_" << prefix << "_" << name << ".bin)" << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;

  td::EpairModelLoader loader(epair_path.c_str());

  ge::Graph test_graph(prefix.c_str());
  td::Status init_check = loader.GetGEGraph(&test_graph);
  if (init_check != td::SUCCESS) {
    std::cerr << "[ERROR] EpairModelLoader 初始化失败, status code: " << init_check << std::endl;
    return false;
  }
  std::cout << "  ✓ epair 加载成功" << std::endl;

  static aclrtStream stream = nullptr;
  if (stream == nullptr) {
    aclError acl_ret = aclrtCreateStream(&stream);
    if (acl_ret != ACL_ERROR_NONE) {
      std::cerr << "[ERROR] aclrtCreateStream 失败, error: " << acl_ret << std::endl;
      return false;
    }
  }
  std::cout << "  ✓ ACL stream 创建成功" << std::endl;

  std::map<ge::AscendString, ge::AscendString> session_options = {};
  td::Status compile_status = loader.CompileAndLoad(session_options, stream);
  if (compile_status != td::SUCCESS) {
    std::cerr << "[ERROR] CompileAndLoad 失败, status: " << compile_status << std::endl;
    aclrtDestroyStream(stream);
    return false;
  }
  std::cout << "  ✓ CompileAndLoad 成功" << std::endl;

  std::vector<gert::Tensor> inputs;
  for (const auto &name : input_names) {
    std::string bin = dir + "/input_" + prefix + "_" + name + ".bin";
    std::string shape = dir + "/input_" + prefix + "_" + name + "_shape.txt";
    inputs.push_back(LoadTensorFromFile(bin, shape));
  }

  // 静态 shape 模型（IsStatic=true）GE 严格校验 output_tensor 数量，
  // 需预分配 output tensor（输出与第一个输入同 shape、dtype=float32）
  std::vector<int64_t> out_dims;
  {
    std::string first_shape = dir + "/input_" + prefix + "_" + input_names[0] + "_shape.txt";
    out_dims = ReadShapeFile(first_shape);
  }
  size_t out_element_count = 1;
  for (auto d : out_dims) {
    out_element_count *= static_cast<size_t>(d);
  }
  size_t out_size = out_element_count * sizeof(float);

  std::vector<gert::Tensor> outputs;
  void *prealloc_out_ptr = nullptr;
  {
    gert::Tensor out_tensor;
    auto &out_shape = out_tensor.GetShape();
    for (auto d : out_dims) {
      out_shape.MutableOriginShape().AppendDim(d);
      out_shape.MutableStorageShape().AppendDim(d);
    }
    out_tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    out_tensor.SetDataType(ge::DT_FLOAT);
    aclrtMalloc(&prealloc_out_ptr, out_size, ACL_MEM_MALLOC_HUGE_FIRST);
    gert::TensorData out_data(prealloc_out_ptr, nullptr, out_size, gert::kOnDeviceHbm);
    out_tensor.SetData(std::move(out_data));
    outputs.push_back(std::move(out_tensor));
  }
  td::Status run_status = loader.RunModelWithStreamAsync(stream, inputs, outputs);

  aclError sync_ret = aclrtSynchronizeStream(stream);
  if (sync_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtSynchronizeStream 失败, error: " << sync_ret << std::endl;
    if (prealloc_out_ptr) {
      aclrtFree(prealloc_out_ptr);
    }
    aclrtDestroyStream(stream);
    stream = nullptr;
    return false;
  }

  if (run_status != td::SUCCESS) {
    std::cerr << "[ERROR] RunModelWithStreamAsync 失败, status: " << run_status << std::endl;
    if (prealloc_out_ptr) {
      aclrtFree(prealloc_out_ptr);
    }
    aclrtDestroyStream(stream);
    return false;
  }

  std::cout << "  ✓ 执行成功 (outputs=" << outputs.size() << ")" << std::endl;

  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string out_path = dir + "/delegate_output_" + prefix + "_" + std::to_string(i) + ".bin";

    auto out_shape = outputs[i].GetStorageShape();
    size_t data_cnt = 1;
    for (size_t d = 0; d < out_shape.GetDimNum(); ++d) {
      data_cnt *= out_shape.GetDim(d);
    }
    size_t data_size = data_cnt * sizeof(float);

    if (data_cnt == 0 || data_size > 1024ULL * 1024 * 1024) {
      std::cerr << "[ERROR] Invalid output data_cnt: " << data_cnt << std::endl;
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

    std::cout << "  写出 output[" << i << "]: " << data_cnt << " floats → " << out_path << std::endl;
  }

  if (prealloc_out_ptr) {
    aclrtFree(prealloc_out_ptr);
  }
  aclrtDestroyStream(stream);
  stream = nullptr;

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
    std::cerr << "[ERROR] 用法: " << argv[0] << " [--custom-ops <path>] <epair1> [<epair2> ...]" << std::endl;
    return -1;
  }

  if (!custom_ops_path.empty()) {
    std::cout << "加载自定义算子库: " << custom_ops_path << std::endl;
    void *handle = dlopen(custom_ops_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "[ERROR] dlopen 失败: " << dlerror() << std::endl;
      return -1;
    }
    std::cout << "  ✓ 自定义算子库加载成功" << std::endl;
  }

  PrintStage(1, 2, "初始化 GE");
  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
  auto init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[ERROR] GEInitializeV2 失败, status: " << init_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE 初始化成功" << std::endl;

  aclError acl_ret = aclrtSetDevice(0);
  if (acl_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtSetDevice 失败, error: " << acl_ret << std::endl;
    return -1;
  }

  PrintStage(2, 2, "执行模型");
  bool all_pass = true;

  for (const auto &epair_path : epair_paths) {
    std::string model_name = Basename(epair_path);
    std::string prefix = model_name;
    if (prefix.substr(0, 6) == "model_") prefix = prefix.substr(6);

    std::cout << std::endl;
    std::cout << kStageSep << std::endl;
    std::cout << " 执行: " << model_name << " (prefix=" << prefix << ")" << std::endl;
    std::cout << kStageSep << std::endl;

    if (!RunEpairModel(epair_path, prefix)) {
      std::cerr << "[FAIL] " << model_name << " 执行失败" << std::endl;
      all_pass = false;
    } else {
      std::cout << "[PASS] " << model_name << " 执行成功" << std::endl;
    }
  }

  ge::GEFinalizeV2();

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  if (all_pass) {
    std::cout << " 所有模型执行完成" << std::endl;
  } else {
    std::cout << " 部分模型执行失败" << std::endl;
  }
  std::cout << kStageSep << std::endl;

  return all_pass ? 0 : -1;
}
