/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tensor_io.h"
#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <cstring>

namespace {

ge::DataType ToGeDataType(xllm::GeDataType dtype) {
  return static_cast<ge::DataType>(static_cast<uint32_t>(dtype));
}

xllm::GeDataType FromGeDataType(ge::DataType dtype) {
  return static_cast<xllm::GeDataType>(static_cast<uint32_t>(dtype));
}

bool PrepareInputs(const std::vector<std::string> &input_files, std::vector<gert::Tensor> &inputs,
                   std::vector<std::vector<uint8_t>> &input_buffers) {
  std::cout << "[Info] Loading " << input_files.size() << " inputs..." << std::endl;

  inputs.resize(input_files.size());
  input_buffers.resize(input_files.size());

  for (size_t i = 0; i < input_files.size(); ++i) {
    xllm::TensorData td;
    if (!xllm::LoadTensorFromBin(input_files[i], td)) {
      std::cerr << "[Error] Failed to load input " << i << ": " << input_files[i] << std::endl;
      return false;
    }

    input_buffers[i] = td.data;

    inputs[i].GetShape() = gert::StorageShape();
    for (auto dim : td.meta.shape) {
      inputs[i].GetShape().MutableOriginShape().AppendDim(dim);
      inputs[i].GetShape().MutableStorageShape().AppendDim(dim);
    }
    inputs[i].MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    inputs[i].SetDataType(ToGeDataType(td.meta.dtype));

    gert::TensorData tensor_data(input_buffers[i].data(), nullptr, input_buffers[i].size(), gert::kOnHost);
    inputs[i].SetData(std::move(tensor_data));
  }

  std::cout << "[Success] All inputs prepared" << std::endl;
  return true;
}

bool SaveOutputs(const std::string &output_dir, const std::vector<gert::Tensor> &outputs) {
  std::cout << "[Info] Saving " << outputs.size() << " outputs to: " << output_dir << std::endl;

  for (size_t i = 0; i < outputs.size(); ++i) {
    xllm::TensorData td;

    auto shape = outputs[i].GetStorageShape();
    for (size_t j = 0; j < shape.GetDimNum(); ++j) {
      td.meta.shape.push_back(shape.GetDim(j));
    }
    td.meta.dtype = FromGeDataType(outputs[i].GetDataType());
    td.meta.name = "output_" + std::to_string(i);

    size_t data_size = outputs[i].GetSize();
    td.data.resize(data_size);

    const void *src_ptr = outputs[i].GetAddr();
    if (src_ptr != nullptr) {
      std::memcpy(td.data.data(), src_ptr, data_size);
    }

    std::string bin_path = output_dir + "/output_" + std::to_string(i) + ".bin";
    xllm::SaveTensorToBin(bin_path, td);
  }

  std::cout << "[Success] All outputs saved" << std::endl;
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " <epair_file> <output_dir> <device_id>" << std::endl;
    std::cerr << "       <input0.bin> <input1.bin> <input2.bin> <input3.bin> <input4.bin>" << std::endl;
    return -1;
  }

  std::string epair_file = argv[1];
  std::string output_dir = argv[2];
  std::string device_id = argv[3];

  std::vector<std::string> input_files;
  for (int i = 4; i < argc; ++i) {
    input_files.push_back(argv[i]);
  }

  std::cout << "[Info] Epair: " << epair_file << std::endl;
  std::cout << "[Info] Output dir: " << output_dir << std::endl;
  std::cout << "[Info] Device: " << device_id << std::endl;
  std::cout << "[Info] Inputs: " << input_files.size() << " files" << std::endl;

  std::cout << "[Info] Initializing GE..." << std::endl;

  const char *custom_op_path_env = std::getenv("CUSTOM_OP_LIBRARY_PATH");
  std::string custom_op_path = custom_op_path_env ? custom_op_path_env : "";

  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString(device_id.c_str())}};

  if (!custom_op_path.empty()) {
    std::cout << "[Info] Setting custom op plugin path: " << custom_op_path << std::endl;
    ge_options[ge::AscendString("ge.TBE_plugin_path")] = ge::AscendString(custom_op_path.c_str());
  }

  auto init_status = ge::GEInitializeV2(ge_options);
  if (init_status != ge::SUCCESS) {
    std::cerr << "[Error] GEInitialize failed: " << init_status << std::endl;
    return -1;
  }
  std::cout << "[Success] GE initialized" << std::endl;

  std::cout << "[Info] Loading epair..." << std::endl;
  td::EpairModelLoader loader(epair_file.c_str());
  std::cout << "[Success] Epair loaded" << std::endl;

  std::cout << "[Info] Getting delegate graph..." << std::endl;
  ge::Graph graph("delegate_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[Error] GetGEGraph failed, status: " << graph_status << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Delegate graph materialized" << std::endl;

  std::cout << "[Info] Creating GeSession..." << std::endl;
  std::map<ge::AscendString, ge::AscendString> session_options;
  ge::GeSession ge_session(session_options);

  static uint32_t next_graph_id = 0;
  uint32_t graph_id = next_graph_id++;

  std::cout << "[Info] Adding graph (id=" << graph_id << ")..." << std::endl;
  auto add_status = ge_session.AddGraph(graph_id, graph);
  if (add_status != ge::SUCCESS) {
    std::cerr << "[Error] AddGraph failed: " << add_status << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Graph added" << std::endl;

  std::vector<gert::Tensor> inputs;
  std::vector<std::vector<uint8_t>> input_buffers;
  if (!PrepareInputs(input_files, inputs, input_buffers)) {
    std::cerr << "[Error] Failed to prepare inputs" << std::endl;
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }

  std::cout << "[Info] Running graph..." << std::endl;
  std::vector<gert::Tensor> outputs;
  auto run_status = ge_session.RunGraph(graph_id, inputs, outputs);
  if (run_status != ge::SUCCESS) {
    std::cerr << "[Error] RunGraph failed: " << run_status << std::endl;
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "[Success] Graph executed, outputs count: " << outputs.size() << std::endl;

  if (!SaveOutputs(output_dir, outputs)) {
    std::cerr << "[Error] Failed to save outputs" << std::endl;
    ge_session.RemoveGraph(graph_id);
    ge::GEFinalizeV2();
    return -1;
  }

  ge_session.RemoveGraph(graph_id);

  std::cout << "[Info] Finalizing GE..." << std::endl;
  ge::GEFinalizeV2();
  std::cout << "[Success] GE finalized" << std::endl;

  std::cout << "[Done] Epair runner completed successfully" << std::endl;
  return 0;
}
