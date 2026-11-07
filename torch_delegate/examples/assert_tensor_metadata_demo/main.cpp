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
 * EpairModelLoader 加载和执行示例 — _assert_tensor_metadata 场景
 *
 * 模型: SimpleModel
 *   x: [2, 4]  float32  NPU tensor
 *   weight: [4, 3] (parameter, 已打包在 epair 中)
 *   forward: mm(x, weight) -> to(f32) -> relu -> to(bf16)
 *   output: [2, 3]  bfloat16
 *
 * 本示例演示:
 *   1. 加载含 _assert_tensor_metadata 节点的 epair
 *   2. 编译执行, 观察 GE 报错 (节点无对应 GE IR)
 *
 * 参考: pack_pass/main.cpp
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include "acl/acl_rt.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

const char *kSep = "========================================";

void PrintStage(int idx, int total, const std::string &name) {
  std::cout << "\n"
            << kSep << "\n"
            << " [" << idx << "/" << total << "] " << name << "\n"
            << kSep << std::endl;
}

// ---- 文件读取辅助 ----

std::string Dirname(const std::string &path) {
  auto pos = path.rfind('/');
  return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

std::vector<char> ReadBinaryFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

// ---- 输入元数据解析 ----

struct TensorMeta {
  std::vector<int64_t> shape;
  std::string dtype;
};

std::map<int, TensorMeta> LoadInputMeta(const std::string &dir) {
  std::map<int, TensorMeta> result;
  std::string path = dir + "/input_meta.txt";
  std::ifstream f(path);
  if (!f) return result;
  std::string line;
  while (std::getline(f, line)) {
    size_t idx_end = line.find(':');
    if (idx_end == std::string::npos) continue;
    int idx = std::stoi(line.substr(0, idx_end));

    TensorMeta meta;
    size_t dt_pos = line.find("dtype=");
    if (dt_pos != std::string::npos) {
      size_t dt_end = line.find(' ', dt_pos + 6);
      meta.dtype = line.substr(dt_pos + 6, dt_end - dt_pos - 6);
    }
    size_t sh_pos = line.find("shape=[");
    if (sh_pos != std::string::npos) {
      sh_pos += 7;
      size_t sh_end = line.find(']', sh_pos);
      std::string sh_str = line.substr(sh_pos, sh_end - sh_pos);
      std::stringstream ss(sh_str);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) meta.shape.push_back(std::stoll(tok));
      }
    }
    result[idx] = meta;
  }
  return result;
}

// ---- 输入构建 ----

struct InputSpec {
  int index;
  std::string name;
  ge::DataType dtype;
  bool on_device;
};

// SimpleModel 仅 1 个用户输入: x (float32 [2, 4])
const std::vector<InputSpec> kInputSpecs = {
    {0, "x", ge::DT_FLOAT, true},
};

gert::Tensor BuildInput(const InputSpec &spec, const std::string &bin_path, const TensorMeta &meta,
                        std::vector<std::vector<char>> &buffers) {
  auto raw = ReadBinaryFile(bin_path);

  std::vector<int64_t> dims;
  if (spec.on_device) {
    dims = meta.shape.empty() ? std::vector<int64_t>{1} : meta.shape;
  } else {
    dims = {};
  }

  std::cout << "    input[" << spec.index << "] " << spec.name << "  dtype=" << meta.dtype << "  shape=[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << dims[i];
  }
  std::cout << "]  placement=" << (spec.on_device ? "NPU" : "CPU") << std::endl;

  gert::StorageShape shape;
  for (int64_t d : dims) {
    shape.MutableOriginShape().AppendDim(d);
    shape.MutableStorageShape().AppendDim(d);
  }

  gert::Tensor tensor;
  tensor.GetShape() = shape;
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  tensor.SetDataType(spec.dtype);

  buffers.emplace_back(std::move(raw));
  auto &buf = buffers.back();
  gert::TensorData tensor_data(buf.data(), nullptr, buf.size(), gert::kOnHost);
  tensor.SetData(std::move(tensor_data));

  return tensor;
}

std::vector<gert::Tensor> BuildAllInputs(const std::string &epair_dir, std::vector<std::vector<char>> &buffers) {
  auto input_metas = LoadInputMeta(epair_dir);
  std::vector<gert::Tensor> inputs;
  inputs.reserve(kInputSpecs.size());

  for (const auto &spec : kInputSpecs) {
    std::string bin_path = epair_dir + "/input_" + std::to_string(spec.index) + ".bin";
    auto meta_it = input_metas.find(spec.index);
    if (meta_it == input_metas.end()) {
      throw std::runtime_error("No meta for input " + std::to_string(spec.index));
    }
    inputs.push_back(BuildInput(spec, bin_path, meta_it->second, buffers));
  }
  return inputs;
}

// ---- dtype 辅助 ----

size_t GetDtypeSize(ge::DataType dt) {
  switch (dt) {
    case ge::DT_FLOAT:
      return 4;
    case ge::DT_BF16:
      return 2;
    case ge::DT_FLOAT16:
      return 2;
    case ge::DT_INT64:
      return 8;
    case ge::DT_INT32:
      return 4;
    case ge::DT_INT8:
      return 1;
    case ge::DT_UINT8:
      return 1;
    case ge::DT_BOOL:
      return 1;
    default:
      return 0;
  }
}

// 按 shape × dtype 计算实际数据字节数 (GetSize 返回的是 GE 分配的 buffer 大小, 可能远大于实际数据)
size_t GetActualByteSize(const gert::Tensor &tensor) {
  auto shape = tensor.GetStorageShape();
  size_t num_elements = 1;
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    num_elements *= static_cast<size_t>(shape.GetDim(i));
  }
  return num_elements * GetDtypeSize(tensor.GetDataType());
}

// bf16 -> float32 转换
float Bf16ToFloat(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float fval;
  std::memcpy(&fval, &bits, sizeof(float));
  return fval;
}

// ---- 输出打印 ----

void PrintTensor(const gert::Tensor &tensor, int index) {
  auto shape = tensor.GetStorageShape();
  std::cout << "  output[" << index << "] dtype=" << tensor.GetDataType() << " shape: [";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << shape.GetDim(i);
  }
  std::cout << "]" << std::endl;

  size_t byte_size = GetActualByteSize(tensor);
  std::cout << "  (buffer size=" << tensor.GetSize() << ", actual data=" << byte_size << " bytes)" << std::endl;

  if (byte_size == 0 || tensor.GetAddr() == nullptr) {
    std::cout << "  output[" << index << "] data: <empty>" << std::endl;
    return;
  }

  // 输出在 device 侧, 拷贝到 host 再读取 (仅拷贝实际数据大小)
  std::vector<char> host_buf(byte_size);
  aclrtMemcpy(host_buf.data(), byte_size, tensor.GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);

  std::cout << "  output[" << index << "] data: [";
  if (tensor.GetDataType() == ge::DT_FLOAT) {
    size_t count = byte_size / sizeof(float);
    const float *fdata = reinterpret_cast<const float *>(host_buf.data());
    size_t print_cnt = std::min(count, size_t(16));
    for (size_t i = 0; i < print_cnt; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << fdata[i];
    }
    if (count > print_cnt) std::cout << ", ...";
  } else if (tensor.GetDataType() == ge::DT_BF16) {
    size_t count = byte_size / sizeof(uint16_t);
    const uint16_t *bdata = reinterpret_cast<const uint16_t *>(host_buf.data());
    size_t print_cnt = std::min(count, size_t(16));
    for (size_t i = 0; i < print_cnt; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << Bf16ToFloat(bdata[i]);
    }
    if (count > print_cnt) std::cout << ", ...";
  } else {
    std::cout << "<dtype " << tensor.GetDataType() << ", " << byte_size << " bytes>";
  }
  std::cout << "]" << std::endl;
}

// ---- CompileAndLoad + RunModelWithStreamAsync ----

void RunWithCompileAndLoad(const char *epair_path, const std::vector<gert::Tensor> &inputs) {
  PrintStage(2, 4, "CompileAndLoad + RunModelWithStreamAsync");

  td::EpairModelLoader loader(epair_path);
  std::cout << "  [OK] EpairModelLoader constructed" << std::endl;

  std::map<ge::AscendString, ge::AscendString> options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };

  // 创建 ACL stream
  aclrtStream stream = nullptr;
  auto acl_ret = aclrtCreateStream(&stream);
  if (acl_ret != ACL_SUCCESS) {
    std::cerr << "  [FAIL] aclrtCreateStream failed, ret=" << acl_ret << std::endl;
    return;
  }
  std::cout << "  [OK] aclrtCreateStream" << std::endl;

  auto status = loader.CompileAndLoad(options, stream);
  if (status != td::SUCCESS) {
    std::cerr << "  [FAIL] CompileAndLoad failed, status=" << status << std::endl;
    auto err = ge::GEGetErrorMsgV3();
    std::cerr << "  " << err.GetString() << std::endl;
    aclrtDestroyStream(stream);
    return;
  }
  std::cout << "  [OK] Compile and load finished" << std::endl;

  std::vector<gert::Tensor> outputs;
  status = loader.RunModelWithStreamAsync(stream, inputs, outputs);
  if (status != td::SUCCESS) {
    std::cerr << "  [FAIL] RunModelWithStreamAsync failed, status=" << status << std::endl;
    auto err = ge::GEGetErrorMsgV3();
    std::cerr << "  " << err.GetString() << std::endl;
    aclrtDestroyStream(stream);
    return;
  }

  // 同步 stream, 确保异步推理完成
  acl_ret = aclrtSynchronizeStream(stream);
  if (acl_ret != ACL_SUCCESS) {
    std::cerr << "  [FAIL] aclrtSynchronizeStream failed, ret=" << acl_ret << std::endl;
  }
  std::cout << "  [OK] Inference finished, output count=" << outputs.size() << std::endl;

  for (size_t i = 0; i < outputs.size(); ++i) {
    PrintTensor(outputs[i], i);
  }

  // 保存输出供比对 (转换为 float32, 与 Python ref_output 一致)
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string out_path = Dirname(epair_path) + "/cpp_output_a_" + std::to_string(i) + ".bin";
    size_t actual_sz = GetActualByteSize(outputs[i]);
    std::vector<char> host_out(actual_sz);
    aclrtMemcpy(host_out.data(), actual_sz, outputs[i].GetAddr(), actual_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    // 转换为 float32 后保存
    auto dt = outputs[i].GetDataType();
    if (dt == ge::DT_FLOAT) {
      std::ofstream ofs(out_path, std::ios::binary);
      ofs.write(host_out.data(), static_cast<std::streamsize>(actual_sz));
    } else if (dt == ge::DT_BF16) {
      size_t count = actual_sz / sizeof(uint16_t);
      std::vector<float> f32_buf(count);
      const uint16_t *bdata = reinterpret_cast<const uint16_t *>(host_out.data());
      for (size_t j = 0; j < count; ++j) {
        f32_buf[j] = Bf16ToFloat(bdata[j]);
      }
      std::ofstream ofs(out_path, std::ios::binary);
      ofs.write(reinterpret_cast<const char *>(f32_buf.data()), static_cast<std::streamsize>(count * sizeof(float)));
    } else {
      std::ofstream ofs(out_path, std::ios::binary);
      ofs.write(host_out.data(), static_cast<std::streamsize>(actual_sz));
    }
    std::cout << "  [save] " << out_path << std::endl;
  }

  // 销毁 stream
  aclrtDestroyStream(stream);
}

}  // namespace

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.epair>" << std::endl;
    std::cerr << "\n  Please run export_demo.py on the Python side first to generate simple_model.epair "
                 "and input files"
              << std::endl;
    return -1;
  }

  const char *epair_path = argv[1];
  std::string epair_dir = Dirname(epair_path);

  // =========================================================
  // [1/4] 初始化 GE + 准备输入
  // =========================================================
  PrintStage(1, 4, "GE init + prepare input data");
  std::cout << "  epair: " << epair_path << std::endl;
  std::cout << "  input dir: " << epair_dir << std::endl;

  std::map<ge::AscendString, ge::AscendString> ge_opts = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
  if (ge::GEInitializeV2(ge_opts) != ge::SUCCESS) {
    std::cerr << "[FAIL] GEInitializeV2" << std::endl;
    return -1;
  }
  std::cout << "  [OK] GE initialized, deviceId=0" << std::endl;

  std::vector<std::vector<char>> input_buffers;
  std::vector<gert::Tensor> inputs;
  try {
    inputs = BuildAllInputs(epair_dir, input_buffers);
  } catch (const std::exception &e) {
    std::cerr << "[FAIL] Failed to build inputs: " << e.what() << std::endl;
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "  [OK] Built " << inputs.size() << " input(s)" << std::endl;

  // =========================================================
  // [2/4] CompileAndLoad + RunModelWithStreamAsync
  // =========================================================
  RunWithCompileAndLoad(epair_path, inputs);

  // =========================================================
  // [3/4] 精度比对
  // =========================================================
  PrintStage(3, 4, "Accuracy comparison (vs Python reference output)");

  std::string ref_path = epair_dir + "/ref_output.bin";
  std::ifstream ref_file(ref_path, std::ios::binary);
  if (!ref_file) {
    std::cout << "  [SKIP] Reference output not found: " << ref_path << std::endl;
  } else {
    ref_file.seekg(0, std::ios::end);
    size_t ref_size = ref_file.tellg();
    ref_file.seekg(0, std::ios::beg);
    std::vector<char> ref_data(ref_size);
    ref_file.read(ref_data.data(), ref_size);

    std::cout << "  Reference output size: " << ref_size << " bytes (" << (ref_size / sizeof(float)) << " floats)"
              << std::endl;

    // 打印参考输出数据
    {
      size_t count = ref_size / sizeof(float);
      const float *ref_f = reinterpret_cast<const float *>(ref_data.data());
      std::cout << "  Reference output data: [";
      size_t print_cnt = std::min(count, size_t(16));
      for (size_t i = 0; i < print_cnt; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ref_f[i];
      }
      if (count > print_cnt) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }

    std::string cpp_out = epair_dir + "/cpp_output_a_0.bin";
    std::ifstream cpp_file(cpp_out, std::ios::binary);
    if (!cpp_file) {
      std::cout << "  [SKIP] " << cpp_out << " does not exist" << std::endl;
    } else {
      cpp_file.seekg(0, std::ios::end);
      size_t cpp_size = cpp_file.tellg();
      cpp_file.seekg(0, std::ios::beg);
      std::vector<char> cpp_data(cpp_size);
      cpp_file.read(cpp_data.data(), cpp_size);

      // 打印 C++ 输出数据
      {
        size_t count = cpp_size / sizeof(float);
        const float *cpp_f = reinterpret_cast<const float *>(cpp_data.data());
        std::cout << "  C++ output size: " << cpp_size << " bytes (" << count << " floats)" << std::endl;
        std::cout << "  C++ output data: [";
        size_t print_cnt = std::min(count, size_t(16));
        for (size_t i = 0; i < print_cnt; ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << cpp_f[i];
        }
        if (count > print_cnt) std::cout << ", ...";
        std::cout << "]" << std::endl;
      }
      if (cpp_size != ref_size) {
        std::cout << "  [DIFF] Size mismatch (ref=" << ref_size << " vs cpp=" << cpp_size << ")" << std::endl;
      } else {
        size_t n = ref_size / sizeof(float);
        const float *ref_f = reinterpret_cast<const float *>(ref_data.data());
        const float *cpp_f = reinterpret_cast<const float *>(cpp_data.data());
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          max_err = std::max(max_err, std::abs(ref_f[i] - cpp_f[i]));
        }
        if (max_err < 1e-3f) {
          std::cout << "  [PASS] Matches reference (max_err=" << max_err << ")" << std::endl;
        } else {
          std::cout << "  [DIFF] Exceeds tolerance (max_err=" << max_err << ")" << std::endl;
        }
      }
    }
  }

  // =========================================================
  // [4/4] 清理
  // =========================================================
  PrintStage(4, 4, "Cleanup");
  ge::GEFinalizeV2();
  std::cout << "  [OK] GE finalized" << std::endl;

  std::cout << "\n"
            << kSep << "\n"
            << " All done"
            << "\n"
            << kSep << std::endl;
  return 0;
}
