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
 * E2E 测试 - C++ Runner
 *
 * 加载 epair 文件并执行，支持多组输入和动态 shape。
 * 支持多 tensor 输入（x + static_tensor）。
 *
 * 用法:
 *   ./fallback_runner [--custom-ops <path>] <epair>
 *
 * 对于每个 epair 文件，自动查找对应的输入文件：
 *   输入: input_{prefix}_{group_idx}_x.bin + input_{prefix}_{group_idx}_x_shape.txt
 *         input_{prefix}_{group_idx}_static_tensor.bin + input_{prefix}_{group_idx}_static_tensor_shape.txt
 *   输出: delegate_output_{prefix}_{group_idx}_0.bin
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/csrc/logging/LogContext.h>
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

  // 加载 epair
  td::EpairModelLoader loader(epair_path.c_str());

  // 验证初始化是否成功
  ge::Graph test_graph("test");
  td::Status init_check = loader.GetGEGraph(&test_graph);
  if (init_check != td::SUCCESS) {
    std::cerr << "[ERROR] EpairModelLoader initialization failed, status code: " << init_check << std::endl;
    std::cerr << "[ERROR] Please check the epair file format and log output" << std::endl;
    std::cerr << "[ERROR] epair path: " << epair_path << std::endl;
    return false;
  }
  std::cout << "  ✓ epair loaded successfully" << std::endl;

  // 统计 GE 图中 Data 节点数量，确定模型期望的输入个数
  // 静态 shape 模型中 SymInt/int 常量被烘焙为 Const 节点，不是 Data 输入
  auto direct_nodes = test_graph.GetDirectNode();
  size_t expected_input_count = 0;
  for (const auto &node : direct_nodes) {
    ge::AscendString type;
    if (node.GetType(type) == ge::GRAPH_SUCCESS && std::string(type.GetString()) == "Data") {
      expected_input_count++;
    }
  }
  std::cout << "  [INFO] Model expected input count: " << expected_input_count << std::endl;

  // 创建 ACL stream
  aclrtStream stream = nullptr;
  aclError acl_ret = aclrtCreateStream(&stream);
  if (acl_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtCreateStream failed, error: " << acl_ret << std::endl;
    return false;
  }
  std::cout << "  ✓ ACL stream created successfully" << std::endl;

  // 编译并加载
  std::map<ge::AscendString, ge::AscendString> session_options = {};
  td::Status compile_status = loader.CompileAndLoad(session_options, stream);
  if (compile_status != td::SUCCESS) {
    std::cerr << "[ERROR] CompileAndLoad failed, status: " << compile_status << std::endl;
    aclrtDestroyStream(stream);
    return false;
  }
  std::cout << "  ✓ CompileAndLoad succeeded" << std::endl;

  // 对每组输入执行
  for (size_t group_idx = 0; group_idx < input_groups.size(); ++group_idx) {
    int group = input_groups[group_idx];
    std::string x_bin = dir + "/input_" + prefix + "_" + std::to_string(group) + "_x.bin";
    std::string x_shape = dir + "/input_" + prefix + "_" + std::to_string(group) + "_x_shape.txt";
    std::string st_bin = dir + "/input_" + prefix + "_" + std::to_string(group) + "_static_tensor.bin";
    std::string st_shape = dir + "/input_" + prefix + "_" + std::to_string(group) + "_static_tensor_shape.txt";

    // 加载 x
    gert::Tensor x_tensor = LoadTensorFromFile(x_bin, x_shape);
    std::vector<int64_t> x_dims = ReadShapeFile(x_shape);
    std::cout << "  [DEBUG] x shape: [";
    for (size_t i = 0; i < x_dims.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << x_dims[i];
    }
    std::cout << "]" << std::endl;

    // 加载 static_tensor
    gert::Tensor st_tensor = LoadTensorFromFile(st_bin, st_shape);
    std::vector<int64_t> st_dims = ReadShapeFile(st_shape);
    std::cout << "  [DEBUG] static_tensor shape: [";
    for (size_t i = 0; i < st_dims.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << st_dims[i];
    }
    std::cout << "]" << std::endl;

    // 加载 multiplier (SymInt, 作为 int64 tensor)
    std::string mult_path = dir + "/input_" + prefix + "_" + std::to_string(group) + "_multiplier.txt";
    int64_t multiplier_val = 0;
    {
      std::ifstream mult_file(mult_path);
      if (mult_file) {
        mult_file >> multiplier_val;
      }
    }
    // SymInt 输入在 GE 图中为 host placement 的 Data 节点，必须以 host tensor 传入
    // （EpairModelLoader 运行期校验 sym 输入 placement，device tensor 返回 INVALID_PARAM）
    gert::Tensor mult_tensor;
    mult_tensor.GetShape().MutableOriginShape().AppendDim(1);
    mult_tensor.GetShape().MutableStorageShape().AppendDim(1);
    mult_tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    mult_tensor.SetDataType(ge::DT_INT64);
    gert::TensorData mult_data(&multiplier_val, nullptr, sizeof(int64_t), gert::kOnHost);
    mult_tensor.SetData(std::move(mult_data));
    std::cout << "  [DEBUG] multiplier: " << multiplier_val << std::endl;

    // 构建输入列表（顺序：x, static_tensor, multiplier）
    // 仅发送模型期望数量的输入（静态 shape 模型可能不需要 multiplier）
    std::vector<gert::Tensor> inputs;
    inputs.push_back(std::move(x_tensor));
    inputs.push_back(std::move(st_tensor));
    if (expected_input_count > 2) {
      inputs.push_back(std::move(mult_tensor));
    }

    // 执行
    // 静态 shape 模型（IsStatic=true）GE 严格校验 output_tensor 数量，
    // 需预分配 output tensor（输出与 x 同 shape、dtype=float32）
    // 动态 shape 模型不需要预分配，GE 会自动填充 outputs
    std::vector<gert::Tensor> outputs;
    void *prealloc_out_ptr = nullptr;
    if (expected_input_count <= 2) {
      gert::Tensor out_tensor;
      auto &out_shape = out_tensor.GetShape();
      for (auto d : x_dims) {
        out_shape.MutableOriginShape().AppendDim(d);
        out_shape.MutableStorageShape().AppendDim(d);
      }
      out_tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
      out_tensor.SetDataType(ge::DT_FLOAT);
      size_t out_size = 1;
      for (auto d : x_dims) {
        out_size *= static_cast<size_t>(d);
      }
      out_size *= sizeof(float);
      aclrtMalloc(&prealloc_out_ptr, out_size, ACL_MEM_MALLOC_HUGE_FIRST);
      gert::TensorData out_data(prealloc_out_ptr, nullptr, out_size, gert::kOnDeviceHbm);
      out_tensor.SetData(std::move(out_data));
      outputs.push_back(std::move(out_tensor));
    }
    td::Status run_status = loader.RunModelWithStreamAsync(stream, inputs, outputs);

    // 异步接口，需同步 stream 后再读取输出或释放输入内存
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_ERROR_NONE) {
      std::cerr << "[ERROR] aclrtSynchronizeStream failed (group " << group << "), error: " << sync_ret << std::endl;
      if (prealloc_out_ptr) {
        aclrtFree(prealloc_out_ptr);
        prealloc_out_ptr = nullptr;
      }
      aclrtDestroyStream(stream);
      return false;
    }

    if (run_status != td::SUCCESS) {
      std::cerr << "[ERROR] RunModelWithStreamAsync failed (group " << group << "), status: " << run_status
                << std::endl;
      if (prealloc_out_ptr) {
        aclrtFree(prealloc_out_ptr);
        prealloc_out_ptr = nullptr;
      }
      aclrtDestroyStream(stream);
      return false;
    }

    std::cout << "  ✓ Input group " << group << " executed successfully (outputs=" << outputs.size() << ")"
              << std::endl;

    // 保存输出
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::string out_path =
          dir + "/delegate_output_" + prefix + "_" + std::to_string(group) + "_" + std::to_string(i) + ".bin";

      // 从 shape 计算 output size（GetSize() 可能因 dtype=DT_UNDEFINED 返回垃圾值）
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
    }

    // 释放预分配的 output 设备内存
    if (prealloc_out_ptr) {
      aclrtFree(prealloc_out_ptr);
      prealloc_out_ptr = nullptr;
    }
  }

  // 销毁 ACL stream
  aclrtDestroyStream(stream);

  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string custom_ops_path;
  std::vector<std::string> epair_paths;

  // 解析参数
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

  // 加载自定义算子库
  if (!custom_ops_path.empty()) {
    std::cout << "Loading custom op library: " << custom_ops_path << std::endl;
    void *handle = dlopen(custom_ops_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "[ERROR] dlopen failed: " << dlerror() << std::endl;
      return -1;
    }
    std::cout << "  ✓ Custom op library loaded successfully" << std::endl;
  }

  // 初始化 GE
  PrintStage(1, 2, "Initialize GE");
  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
  auto init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[ERROR] GEInitializeV2 failed, status: " << init_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE initialized successfully" << std::endl;

  // 设置 torch_npu 日志级别为 DEBUG
  std::cout << "[INFO] Setting torch_npu log level to DEBUG..." << std::endl;
  std::unordered_map<std::string, int> log_levels = {{"torch_npu", 10},  // DEBUG = 10
                                                     {"torch_npu.dispatch", 10},
                                                     {"torch_npu.acl", 10},
                                                     {"torch_npu.memory", 10}};
  // npu_logging::LogContext::GetInstance().setLogs(log_levels);
  // std::cout << "[PASS] torch_npu 日志级别设置完成" << std::endl;

  // 设置设备
  aclError acl_ret = aclrtSetDevice(0);
  if (acl_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtSetDevice failed, error: " << acl_ret << std::endl;
    return -1;
  }

  // 执行所有模型
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

  // 清理
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
