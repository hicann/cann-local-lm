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
 * inplace_input C++ GE Graph 运行示例
 *
 * 功能：
 * 1. 通过 EpairModelLoader 加载 epair
 * 2. 使用 GE Graph 在 NPU 上直接执行
 * 3. 支持 Simple/Chain/Param 三种模型
 * 4. 精度验证（对比 C++ 计算的参考值）
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <map>
#include <random>

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"

#include "reference_data.h"

namespace {

template <typename Iterator>
static std::string Join(Iterator begin, Iterator end, const std::string &sep) {
  if (begin == end) {
    return "";
  }
  std::stringstream ss;
  ss << *begin;
  for (auto iter = std::next(begin); iter != end; ++iter) {
    ss << sep << *iter;
  }
  return ss.str();
}

void PrintSeparator(const std::string &title) {
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(60, '=') << std::endl;
}

template <typename T>
void PrintTensorToConsole(const gert::Tensor &tensor, size_t max_elements = 16) {
  auto shape = tensor.GetStorageShape();
  if (shape.GetDimNum() == 0U) {
    std::cout << "Shape: scalar" << std::endl;
  } else {
    std::vector<int64_t> dims;
    dims.reserve(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
      dims.push_back(shape.GetDim(i));
    }
    std::cout << "Shape: [" << Join(dims.begin(), dims.end(), ", ") << "]" << std::endl;
  }

  std::cout << "Data: ";
  auto data_cnt = tensor.GetSize() / sizeof(T);
  auto data_begin = reinterpret_cast<const T *>(tensor.GetAddr());

  if (data_begin == nullptr || data_cnt == 0U) {
    std::cout << "<empty>" << std::endl;
    return;
  }

  size_t print_count = std::min(data_cnt, max_elements);
  std::cout << "[" << data_begin[0];
  for (size_t i = 1; i < print_count; ++i) {
    std::cout << ", " << data_begin[i];
  }
  if (data_cnt > max_elements) {
    std::cout << ", ...";
  }
  std::cout << "]" << std::endl;
}

float ComputeMaxDiff(const gert::Tensor &a, const gert::Tensor &b) {
  const float *data_a = reinterpret_cast<const float *>(a.GetAddr());
  const float *data_b = reinterpret_cast<const float *>(b.GetAddr());
  size_t count = a.GetSize() / sizeof(float);

  float max_diff = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float diff = std::abs(data_a[i] - data_b[i]);
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

bool VerifyPrecision(const gert::Tensor &output, const gert::Tensor &reference, float tolerance = 1e-5f) {
  float max_diff = ComputeMaxDiff(output, reference);
  std::cout << "Max diff: " << max_diff << std::endl;

  if (max_diff < tolerance) {
    std::cout << "[PASS] Precision verification passed (tolerance=" << tolerance << ")" << std::endl;
    return true;
  } else {
    std::cout << "[FAIL] Precision verification failed (tolerance=" << tolerance << ")" << std::endl;
    return false;
  }
}

bool PrepareInputTensor(const std::vector<int64_t> &shape, gert::Tensor &tensor, std::vector<float> &buffer,
                        uint32_t seed) {
  // 设置 shape - 对于 {4, 4} 固定形状，直接使用 initializer_list
  gert::StorageShape storage_shape({4, 4}, {4, 4});
  tensor.GetShape() = storage_shape;

  // 设置 format
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});

  // 设置数据类型
  tensor.SetDataType(ge::DT_FLOAT);

  // 生成数据
  size_t element_count = storage_shape.GetStorageShape().GetShapeSize();
  buffer.resize(element_count);

  // 固定 seed 保证跨进程数据一致
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < element_count; ++i) {
    buffer[i] = dist(rng);
  }

  // 设置 TensorData (Host memory)
  size_t data_size = element_count * sizeof(float);
  gert::TensorData tensor_data(buffer.data(), nullptr, data_size, gert::kOnHost);
  tensor.SetData(std::move(tensor_data));

  return true;
}

bool PrepareInputs(inplace::ModelType model_type, std::vector<gert::Tensor> &inputs,
                   std::vector<std::vector<float>> &input_buffers, uint32_t seed = 42) {
  inplace::ModelConfig config = inplace::GetModelConfig(model_type);

  std::cout << "[Step 4] Preparing inputs for " << config.name << " model..." << std::endl;
  std::cout << "  Input shapes: ";
  auto shapes = config.GetInputShapes();
  for (size_t i = 0; i < shapes.size(); ++i) {
    std::cout << "[" << Join(shapes[i].begin(), shapes[i].end(), ", ") << "]";
    if (i < shapes.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;

  // 创建输入 x
  gert::Tensor input_x;
  std::vector<float> buffer_x;
  PrepareInputTensor(config.input_shape_x, input_x, buffer_x, seed);
  inputs.push_back(std::move(input_x));
  input_buffers.push_back(std::move(buffer_x));

  std::cout << "  Input x: ";
  PrintTensorToConsole<float>(inputs.back(), 8);

  // Simple 模型需要第二个输入 y
  if (model_type == inplace::ModelType::Simple) {
    gert::Tensor input_y;
    std::vector<float> buffer_y;
    PrepareInputTensor(config.input_shape_y, input_y, buffer_y, seed + 1);
    inputs.push_back(std::move(input_y));
    input_buffers.push_back(std::move(buffer_y));

    std::cout << "  Input y: ";
    PrintTensorToConsole<float>(inputs.back(), 8);
  }

  return true;
}

bool ComputeReferenceOutput(inplace::ModelType model_type, const std::vector<std::vector<float>> &input_buffers,
                            std::vector<float> &reference_buffer) {
  inplace::ModelConfig config = inplace::GetModelConfig(model_type);
  size_t count = input_buffers[0].size();

  std::cout << "[Step 5] Computing reference output..." << std::endl;

  reference_buffer.resize(count);

  // 获取输入数据指针
  std::vector<const float *> input_ptrs;
  for (const auto &buffer : input_buffers) {
    input_ptrs.push_back(buffer.data());
  }

  inplace::ComputeReference(model_type, input_ptrs, reference_buffer.data(), count, config.scale, config.bias);

  std::cout << "  Reference output: [";
  size_t print_count = std::min(count, static_cast<size_t>(8));
  for (size_t i = 0; i < print_count; ++i) {
    std::cout << reference_buffer[i];
    if (i < print_count - 1) std::cout << ", ";
  }
  if (count > 8) std::cout << ", ...";
  std::cout << "]" << std::endl;

  return true;
}

bool MakeReferenceTensor(const std::vector<float> &buffer, gert::Tensor &tensor) {
  // 设置 shape - 对于 {4, 4} 固定形状
  gert::StorageShape storage_shape({4, 4}, {4, 4});
  tensor.GetShape() = storage_shape;

  // 设置 format
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});

  // 设置数据类型
  tensor.SetDataType(ge::DT_FLOAT);

  // 设置 TensorData (Host memory)
  size_t data_size = buffer.size() * sizeof(float);
  float *data_ptr = const_cast<float *>(buffer.data());
  gert::TensorData tensor_data(data_ptr, nullptr, data_size, gert::kOnHost);
  tensor.SetData(std::move(tensor_data));

  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <epair_path> <model_type>" << std::endl;
    std::cerr << "  model_type: simple, chain, param" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  " << argv[0] << " ../generated/epair_model_simple.epair simple" << std::endl;
    return -1;
  }

  std::string epair_path = argv[1];
  std::string model_type_str = argv[2];

  inplace::ModelType model_type = inplace::ParseModelType(model_type_str);
  if (model_type == inplace::ModelType::Unknown) {
    std::cerr << "[Error] Unknown model type: " << model_type_str << std::endl;
    std::cerr << "  Available: simple, chain, param" << std::endl;
    return -1;
  }

  PrintSeparator("inplace_input C++ GE Graph Verification");
  std::cout << "Model: " << model_type_str << std::endl;
  std::cout << "Epair: " << epair_path << std::endl;
  std::cout << "Tolerance: 1e-5" << std::endl;

  // 1. 初始化 GE
  std::cout << "[Step 1] Initializing GE..." << std::endl;
  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
  ge::Status init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[Error] GEInitialize failed, status: " << init_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[Error] Message: " << err_msg.GetString() << std::endl;
    return -1;
  }
  std::cout << "[Success] GE initialized" << std::endl;

  // 2. 加载 epair
  std::cout << "[Step 2] Loading epair from: " << epair_path << std::endl;
  td::EpairModelLoader loader(epair_path.c_str());
  std::cout << "[Success] Epair loaded" << std::endl;

  // 3. 获取 GE Graph
  std::cout << "[Step 3] Getting GE graph..." << std::endl;
  ge::Graph graph("inplace_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[Error] GetGEGraph failed: " << graph_status << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Graph materialized" << std::endl;

  // 4. 创建 GeSession 并添加 Graph
  std::cout << "[Step 4] Creating GeSession..." << std::endl;
  std::map<ge::AscendString, ge::AscendString> session_options;
  ge::GeSession ge_session(session_options);

  uint32_t graph_id = 0;
  std::cout << "[Step 5] Adding graph (id=" << graph_id << ")..." << std::endl;
  ge::Status add_status = ge_session.AddGraph(graph_id, graph);
  if (add_status != ge::SUCCESS) {
    std::cerr << "[Error] AddGraph failed, status: " << add_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[Error] Message: " << err_msg.GetString() << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Graph added" << std::endl;

  // 5. 准备输入（使用固定 seed=42）
  std::vector<gert::Tensor> inputs;
  std::vector<std::vector<float>> input_buffers;
  if (!PrepareInputs(model_type, inputs, input_buffers, 42)) {
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }

  // 6. 计算参考输出
  std::vector<float> reference_buffer;
  if (!ComputeReferenceOutput(model_type, input_buffers, reference_buffer)) {
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }

  // 7. 执行 Graph
  std::cout << "[Step 6] Running graph..." << std::endl;
  std::vector<gert::Tensor> outputs;
  ge::Status run_status = ge_session.RunGraph(graph_id, inputs, outputs);
  if (run_status != ge::SUCCESS) {
    std::cerr << "[Error] RunGraph failed, status: " << run_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[Error] Message: " << err_msg.GetString() << std::endl;
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Graph executed" << std::endl;

  std::cout << "[Step 7] Output tensors count: " << outputs.size() << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::cout << "  Output[" << i << "]: ";
    PrintTensorToConsole<float>(outputs[i], 8);
  }

  // 8. 精度验证
  std::cout << "[Step 8] Verifying accuracy..." << std::endl;
  inplace::ModelConfig config = inplace::GetModelConfig(model_type);

  gert::Tensor reference_tensor;
  MakeReferenceTensor(reference_buffer, reference_tensor);

  bool all_passed = false;
  if (!outputs.empty()) {
    all_passed = VerifyPrecision(outputs[0], reference_tensor, 1e-5f);
  } else {
    std::cerr << "[Error] No output from graph" << std::endl;
  }

  // 清理
  std::cout << "[Step 9] Cleaning up..." << std::endl;
  ge_session.RemoveGraph(graph_id);
  ge::GEFinalizeV2();
  std::cout << "[Success] Cleanup complete" << std::endl;

  PrintSeparator("Result");
  std::cout << "[" << model_type_str << "] Verification: " << (all_passed ? "PASS" : "FAIL") << std::endl;

  return all_passed ? 0 : -1;
}
