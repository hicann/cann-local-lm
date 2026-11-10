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
 * HostCpu 混合调度示例 - C++ Runner
 *
 * 加载含 CPU 节点的 epair 并执行一次 RunGraphWithStreamAsync：
 * HostCpu 自定义算子由 FallbackCodegen 在加载期生成并注册（kHostCPU 后端），
 * CPU 区段与 NPU 区段同图，由 GE 统一调度。
 *
 * 用法:
 *   ./host_cpu_runner <model.epair> [<model2.epair> ...]
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *kStageSep = "========================================";

std::string Dirname(const std::string &path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos) return ".";
  return path.substr(0, pos);
}

std::string Basename(const std::string &path) {
  std::string name = path;
  auto pos = path.rfind('/');
  if (pos != std::string::npos) name = path.substr(pos + 1);
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

gert::Tensor LoadTensorFromFile(const std::string &bin_path, const std::string &shape_path,
                                const std::string &dtype_path) {
  std::vector<int64_t> dims = ReadShapeFile(shape_path);
  if (dims.empty()) {
    throw std::runtime_error("Failed to read shape file: " + shape_path);
  }
  bool is_int64 = false;
  {
    std::ifstream dtype_file(dtype_path);
    std::string line;
    if (dtype_file && std::getline(dtype_file, line) && line.find("int64") != std::string::npos) {
      is_int64 = true;
    }
  }
  const size_t element_count = std::accumulate(dims.begin(), dims.end(), size_t{1},
                                               [](size_t a, int64_t d) { return a * static_cast<size_t>(d); });
  const size_t data_size = element_count * (is_int64 ? sizeof(int64_t) : sizeof(float));

  std::vector<char> data(data_size);
  std::ifstream file(bin_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open input file: " + bin_path);
  }
  file.read(data.data(), static_cast<std::streamsize>(data_size));

  void *device_ptr = nullptr;
  if (aclrtMalloc(&device_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
    throw std::runtime_error("aclrtMalloc failed");
  }
  if (aclrtMemcpy(device_ptr, data_size, data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
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
  tensor.SetDataType(is_int64 ? ge::DT_INT64 : ge::DT_FLOAT);
  gert::TensorData tensor_data(device_ptr, nullptr, data_size, gert::kOnDeviceHbm);
  tensor.SetData(std::move(tensor_data));
  return tensor;
}

bool RunEpairModel(const std::string &epair_path) {
  const std::string dir = Dirname(epair_path);
  const std::string prefix =
      Basename(epair_path).substr(0, 6) == "model_" ? Basename(epair_path).substr(6) : Basename(epair_path);
  const auto t_start = std::chrono::steady_clock::now();

  const std::string input_bin = dir + "/input_" + prefix + "_x.bin";
  const std::string input_shape = input_bin + "_shape.txt";

  std::cout << "Loading: " << epair_path << std::endl;
  td::EpairModelLoader loader(epair_path.c_str());
  ge::Graph test_graph(prefix.c_str());
  if (loader.GetGEGraph(&test_graph) != td::SUCCESS) {
    std::cerr << "[ERROR] EpairModelLoader init failed" << std::endl;
    return false;
  }
  const auto t_loaded = std::chrono::steady_clock::now();
  const double load_ms = std::chrono::duration<double, std::milli>(t_loaded - t_start).count();
  std::cout << "  [OK] epair loaded (incl. codegen compile): " << load_ms << " ms" << std::endl;

  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
    return false;
  }

  std::map<ge::AscendString, ge::AscendString> session_options = {};
  if (loader.CompileAndLoad(session_options, stream) != td::SUCCESS) {
    std::cerr << "[ERROR] CompileAndLoad failed" << std::endl;
    aclrtDestroyStream(stream);
    return false;
  }
  const auto t_compiled = std::chrono::steady_clock::now();
  const double compile_ms = std::chrono::duration<double, std::milli>(t_compiled - t_loaded).count();
  std::cout << "  [OK] CompileAndLoad succeeded: " << compile_ms << " ms" << std::endl;

  std::vector<gert::Tensor> inputs;
  inputs.push_back(LoadTensorFromFile(input_bin, input_shape, input_bin + "_dtype.txt"));

  // 输出尺寸优先用 eager 基线的 shape 文件（推荐网络输出形状与输入不同）
  const std::string ref_shape = dir + "/ref_" + prefix + ".bin_shape.txt";
  const std::string out_shape_file = std::ifstream(ref_shape).good() ? ref_shape : input_shape;
  std::vector<int64_t> out_dims = ReadShapeFile(out_shape_file);
  size_t out_element_count = 1;
  for (auto d : out_dims) out_element_count *= static_cast<size_t>(d);
  const size_t out_size = out_element_count * sizeof(float);

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

  const auto t_run_start = std::chrono::steady_clock::now();
  const td::Status run_status = loader.RunModelWithStreamAsync(stream, inputs, outputs);
  const aclError sync_ret = aclrtSynchronizeStream(stream);
  const auto t_run_end = std::chrono::steady_clock::now();
  const double run_ms = std::chrono::duration<double, std::milli>(t_run_end - t_run_start).count();
  if (run_status != td::SUCCESS || sync_ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] RunModelWithStreamAsync failed, status=" << run_status << ", sync=" << sync_ret << std::endl;
    if (prealloc_out_ptr != nullptr) aclrtFree(prealloc_out_ptr);
    aclrtDestroyStream(stream);
    return false;
  }
  std::cout << "  [OK] Executed (CPU/NPU mixed graph in one call)" << std::endl;

  // 性能观测数据：编译时长 / 端到端时延 / HBM 占用
  size_t hbm_free = 0;
  size_t hbm_total = 0;
  if (aclrtGetMemInfo(ACL_HBM_MEM, &hbm_free, &hbm_total) == ACL_ERROR_NONE) {
    std::cout << "  [PERF] HBM used=" << (hbm_total - hbm_free) / (1024 * 1024)
              << " MB / total=" << hbm_total / (1024 * 1024) << " MB" << std::endl;
  }
  std::cout << "  [PERF] " << prefix << ": load+codegen=" << load_ms << " ms, compile_load=" << compile_ms
            << " ms, first_run=" << run_ms << " ms" << std::endl;

  const std::string out_path = dir + "/delegate_output_" + prefix + ".bin";
  std::vector<float> host_buf(out_element_count);
  aclrtMemcpy(host_buf.data(), out_size, outputs[0].GetAddr(), out_size, ACL_MEMCPY_DEVICE_TO_HOST);
  std::ofstream out_file(out_path, std::ios::binary);
  out_file.write(reinterpret_cast<const char *>(host_buf.data()), static_cast<std::streamsize>(out_size));
  std::cout << "  Writing output to: " << out_path << std::endl;

  if (prealloc_out_ptr != nullptr) aclrtFree(prealloc_out_ptr);
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
    std::cerr << "[ERROR] Usage: " << argv[0] << " [--custom-ops <libcustom_ops.so>] <model.epair> [...]" << std::endl;
    return -1;
  }

  // C++ 进程与 Python 导出进程不共享 Dispatcher 注册表，先 dlopen 自定义算子库。
  if (!custom_ops_path.empty()) {
    std::cout << "Loading custom ops library: " << custom_ops_path << std::endl;
    void *handle = dlopen(custom_ops_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      std::cerr << "[ERROR] dlopen failed: " << dlerror() << std::endl;
      return -1;
    }
  }

  // aclInit 注册应用日志（TDLOG 走 aclAppLog）与 ACL 运行时。
  if (aclInit(nullptr) != ACL_ERROR_NONE) {
    std::cerr << "[WARN] aclInit failed, continue with GE init" << std::endl;
  }

  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
  if (ge::GEInitializeV2(ge_options) != ge::SUCCESS) {
    std::cerr << "[ERROR] GEInitializeV2 failed" << std::endl;
    return -1;
  }
  if (aclrtSetDevice(0) != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] aclrtSetDevice failed" << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }

  bool all_pass = true;
  for (const auto &epair_path : epair_paths) {
    std::cout << std::endl
              << kStageSep << std::endl
              << " Running: " << epair_path << std::endl
              << kStageSep << std::endl;
    if (!RunEpairModel(epair_path)) {
      std::cerr << "[FAIL] " << epair_path << " execution failed" << std::endl;
      all_pass = false;
    } else {
      std::cout << "[PASS] " << epair_path << " execution succeeded" << std::endl;
    }
  }

  ge::GEFinalizeV2();
  (void)aclFinalize();
  return all_pass ? 0 : -1;
}
