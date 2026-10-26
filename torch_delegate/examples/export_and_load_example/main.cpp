/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
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

template <typename T>
static void PrintTensorToConsole(const gert::Tensor &tensor) {
  auto shape = tensor.GetStorageShape();
  if (shape.GetDimNum() == 0U) {
    std::cout << "  shape: scalar" << std::endl;
  } else {
    std::vector<int64_t> dims;
    dims.reserve(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
      dims.push_back(shape.GetDim(i));
    }
    std::cout << "  shape: [" << Join(dims.begin(), dims.end(), ", ") << "]" << std::endl;
  }

  std::cout << "  data:  ";
  auto data_cnt = tensor.GetSize() / sizeof(T);
  auto data_begin = reinterpret_cast<const T *>(tensor.GetAddr());

  if (data_begin == nullptr || data_cnt == 0U) {
    std::cout << "<empty>" << std::endl;
    return;
  }

  for (size_t i = 0; i < data_cnt; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << data_begin[i];
  }
  std::cout << std::endl;
}

std::string Dirname(const std::string &path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos) {
    return ".";
  }
  return path.substr(0, pos);
}

bool ReadInputBinary(const std::string &epair_path, std::vector<gert::Tensor> &inputs,
                     std::vector<std::vector<float>> &input_buffers) {
  static const int64_t kDims[] = {3, 4};
  static const size_t kRank = 2;
  size_t element_count = 1;
  for (size_t i = 0; i < kRank; ++i) {
    element_count *= kDims[i];
  }
  size_t data_size = element_count * sizeof(float);

  std::string bin_path = Dirname(epair_path) + "/input_0.bin";
  std::cout << "  reading input from: " << bin_path << std::endl;

  std::ifstream file(bin_path, std::ios::binary);
  if (!file) {
    std::cerr << "[ERROR] Cannot open input file: " << bin_path << std::endl;
    return false;
  }

  input_buffers.emplace_back(element_count);
  auto &data = input_buffers.back();

  file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data_size));
  if (!file) {
    std::cerr << "[ERROR] Failed to read " << data_size << " bytes from " << bin_path << std::endl;
    return false;
  }

  gert::StorageShape shape({3, 4}, {3, 4});
  gert::Tensor tensor;
  tensor.GetShape() = shape;
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  tensor.SetDataType(ge::DT_FLOAT);

  gert::TensorData tensor_data(data.data(), nullptr, data_size, gert::kOnHost);
  tensor.SetData(std::move(tensor_data));
  inputs.push_back(std::move(tensor));

  PrintTensorToConsole<float>(inputs.back());
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "[ERROR] Usage: " << argv[0] << " <epair_file_path>" << std::endl;
    return -1;
  }

  constexpr size_t kTotalStages = 6;

  // =========================================================
  // [1/6] Initialize GE
  // =========================================================
  PrintStage(1, kTotalStages, "Initialize GE");

  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
  auto init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[ERROR] GEInitializeV2 failed, status: " << init_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[ERROR] " << err_msg.GetString() << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE initialized successfully" << std::endl;

  // =========================================================
  // [2/6] Load epair archive
  // =========================================================
  PrintStage(2, kTotalStages, "Load epair archive");

  const char *epair_file = argv[1];
  std::cout << "  path: " << epair_file << std::endl;

  td::EpairModelLoader loader(epair_file);
  std::cout << "[PASS] Epair loaded, GE Graph built" << std::endl;

  // =========================================================
  // [3/6] Extract GE Graph
  // =========================================================
  PrintStage(3, kTotalStages, "Extract GE Graph");

  ge::Graph graph("delegate_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[ERROR] GetGEGraph failed, status: " << graph_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE Graph extracted successfully" << std::endl;

  // =========================================================
  // [4/6] Add graph to session
  // =========================================================
  PrintStage(4, kTotalStages, "Add graph to session");

  std::map<ge::AscendString, ge::AscendString> session_options;
  ge::GeSession ge_session(session_options);

  static uint32_t next_graph_id = 0;
  const uint32_t graph_id = next_graph_id++;

  auto add_status = ge_session.AddGraph(graph_id, graph);
  if (add_status != ge::SUCCESS) {
    std::cerr << "[ERROR] AddGraph failed, status: " << add_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[ERROR] " << err_msg.GetString() << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[PASS] Graph added (id=" << graph_id << ")" << std::endl;

  // =========================================================
  // [5/6] Prepare input and run graph
  // =========================================================
  PrintStage(5, kTotalStages, "Prepare input and run graph");

  std::vector<gert::Tensor> inputs;
  std::vector<std::vector<float>> input_buffers;
  input_buffers.reserve(1);
  if (!ReadInputBinary(epair_file, inputs, input_buffers)) {
    (void)ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }

  std::vector<gert::Tensor> outputs;
  auto run_status = ge_session.RunGraph(graph_id, inputs, outputs);
  if (run_status != ge::SUCCESS) {
    std::cerr << "[ERROR] RunGraph failed, status: " << run_status << std::endl;
    auto err_msg = ge::GEGetErrorMsgV3();
    std::cerr << "[ERROR] " << err_msg.GetString() << std::endl;
    (void)ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[PASS] Graph executed successfully" << std::endl;

  std::cout << std::endl;
  std::cout << "  output tensors: " << outputs.size() << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::cout << "  output[" << i << "]:" << std::endl;
    PrintTensorToConsole<float>(outputs[i]);
  }

  // Write output to binary files for accuracy comparison
  {
    std::string out_dir = Dirname(epair_file);
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::string out_path = out_dir + "/delegate_output_" + std::to_string(i) + ".bin";
      auto data_cnt = outputs[i].GetSize() / sizeof(float);
      auto data_begin = reinterpret_cast<const float *>(outputs[i].GetAddr());
      std::ofstream out_file(out_path, std::ios::binary);
      out_file.write(reinterpret_cast<const char *>(data_begin),
                     static_cast<std::streamsize>(data_cnt * sizeof(float)));
      std::cout << "  wrote " << out_path << " (" << data_cnt << " floats)" << std::endl;
    }
  }

  // =========================================================
  // [6/6] Cleanup
  // =========================================================
  PrintStage(6, kTotalStages, "Cleanup");

  (void)ge_session.RemoveGraph(graph_id);
  std::cout << "[PASS] Graph removed" << std::endl;

  ge::GEFinalizeV2();
  std::cout << "[PASS] GE finalized" << std::endl;

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " All stages passed. Sample completed successfully" << std::endl;
  std::cout << kStageSep << std::endl;
  return 0;
}
