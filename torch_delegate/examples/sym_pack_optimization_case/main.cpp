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
 * EpairModelLoader 加载和执行示例 — SymPack 优化验证
 *
 * 从 export_pack_model.py 导出的 model.epair 加载模型并执行推理:
 *   EpairModelLoader::CompileAndLoad + RunModelWithStreamAsync
 *
 * 模型: PackReproduceModel(x, m, n) → scalar
 *   x: [a, b]  float32 NPU tensor (device 内存)
 *   m: int64   CPU scalar (不等于 x 任何维度)
 *   n: int64   CPU scalar (不等于 x 任何维度), m*n == a*b
 *   Group A: view([m, n]) × 4 → 4 冗余 Pack[Data_m, Data_n]
 *   Group B: view([n, m]) × 2 → 2 冗余 Pack[Data_n, Data_m]
 *   return sum_a + sum_b → float32 scalar
 *
 * SymPack 优化后:
 *   GE Graph 输入变为 3 个 Data 节点:
 *     [0] x       float32  device  shape=[a, b]
 *     [1] merged_A int64    host    shape=[2]  (由 m,n 打包)
 *     [2] merged_B int64    host    shape=[2]  (由 n,m 打包)
 *   原始 m/n 标量被删除, 按输入组合去重后创建合并 Data 节点
 *
 * EpairModelLoader 内部通过 InputBindingMap 自动适配:
 *   user_inputs[x, m, n] → ge_inputs[x, merged_A([m,n]), merged_B([n,m])]
 *
 * 验证项:
 *   1. EpairModelLoader 自动适配输入, 推理正确
 *   2. 精度比对 max_err < 1e-5
 *
 * TODO: 补充 GetGEGraph + GeSession::RunGraph 手动管理示例,
 *   待 InputBindingMap 对外暴露或找到可靠方式获取 merged Data 打包顺序后补充。
 *
 * 参考: torch-delegate/examples/export_and_load_example/main.cpp
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/gnode.h"
#include "torch_delegate/epair_model_loader.h"
#include "acl/acl_rt.h"

#include <algorithm>
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
  std::string line;
  while (std::getline(f, line)) {
    // 格式: "idx: name  dtype=xxx  shape=[a, b]"
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

// ---- GE Graph Data 节点分析 ----

struct DataNodeInfo {
  std::string name;
  int64_t index = 0;
  ge::DataType dtype = ge::DT_UNDEFINED;
  std::vector<int64_t> shape_dims;
  ge::Placement placement = ge::kPlacementDevice;
};

std::vector<DataNodeInfo> CollectDataNodes(const ge::Graph &graph) {
  std::vector<DataNodeInfo> data_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    ge::AscendString type_str;
    if (node.GetType(type_str) != ge::GRAPH_SUCCESS) {
      continue;
    }
    std::string type = (type_str.GetString() != nullptr) ? type_str.GetString() : "";
    if (type != "Data") {
      continue;
    }
    DataNodeInfo info;
    ge::AscendString name_str;
    if (node.GetName(name_str) == ge::GRAPH_SUCCESS) {
      info.name = (name_str.GetString() != nullptr) ? name_str.GetString() : "";
    }
    node.GetAttr(ge::AscendString("index"), info.index);
    ge::TensorDesc desc;
    if (node.GetInputDesc(0, desc) == ge::GRAPH_SUCCESS) {
      info.dtype = desc.GetDataType();
      info.placement = desc.GetPlacement();
      auto shape = desc.GetShape();
      for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        info.shape_dims.push_back(shape.GetDim(i));
      }
    }
    data_nodes.push_back(std::move(info));
  }
  std::sort(data_nodes.begin(), data_nodes.end(),
            [](const DataNodeInfo &a, const DataNodeInfo &b) { return a.index < b.index; });
  return data_nodes;
}

void PrintDataNodes(const std::vector<DataNodeInfo> &data_nodes) {
  for (const auto &info : data_nodes) {
    std::cout << "    Data index=" << info.index << " name=" << info.name << " dtype=" << static_cast<int>(info.dtype)
              << " shape=[";
    for (size_t i = 0; i < info.shape_dims.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << info.shape_dims[i];
    }
    std::cout << "] placement=" << (info.placement == ge::kPlacementHost ? "Host" : "Device") << std::endl;
  }
}

// ---- 输入构建 ----

struct InputSpec {
  int index;
  std::string name;
  ge::DataType dtype;
  bool on_device;  // true: NPU, false: CPU
};

// 原始用户输入签名 (export 时):
//   idx=0: x  float  NPU  shape=[a, b]
//   idx=1: m  int64  CPU  scalar
//   idx=2: n  int64  CPU  scalar
const std::vector<InputSpec> kUserInputSpecs = {
    {0, "x", ge::DT_FLOAT, true},
    {1, "m", ge::DT_INT64, false},
    {2, "n", ge::DT_INT64, false},
};

gert::Tensor BuildUserInput(const InputSpec &spec, const std::string &bin_path, const TensorMeta &meta,
                            std::vector<std::vector<char>> &host_buffers, std::vector<void *> &device_buffers) {
  auto raw = ReadBinaryFile(bin_path);

  std::vector<int64_t> dims;
  if (spec.on_device) {
    dims = meta.shape.empty() ? std::vector<int64_t>{1} : meta.shape;
  } else {
    // CPU SymInt: 0维标量
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

  if (spec.on_device) {
    // NPU tensor: 数据放到 device
    void *device_ptr = nullptr;
    aclError acl_ret = aclrtMalloc(&device_ptr, raw.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtMalloc failed for input " + spec.name + ", ret=" + std::to_string(acl_ret));
    }
    acl_ret = aclrtMemcpy(device_ptr, raw.size(), raw.data(), raw.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
      aclrtFree(device_ptr);
      throw std::runtime_error("aclrtMemcpy H2D failed for input " + spec.name + ", ret=" + std::to_string(acl_ret));
    }
    device_buffers.push_back(device_ptr);
    gert::TensorData tensor_data(device_ptr, nullptr, raw.size(), gert::kOnDeviceHbm);
    tensor.SetData(std::move(tensor_data));
  } else {
    // CPU scalar: 数据放在 host
    host_buffers.emplace_back(std::move(raw));
    auto &buf = host_buffers.back();
    gert::TensorData tensor_data(buf.data(), nullptr, buf.size(), gert::kOnHost);
    tensor.SetData(std::move(tensor_data));
  }

  return tensor;
}

std::vector<gert::Tensor> BuildAllUserInputs(const std::string &epair_dir, std::vector<std::vector<char>> &host_buffers,
                                             std::vector<void *> &device_buffers) {
  auto input_metas = LoadInputMeta(epair_dir);
  std::vector<gert::Tensor> inputs;
  inputs.reserve(kUserInputSpecs.size());

  for (const auto &spec : kUserInputSpecs) {
    std::string bin_path = epair_dir + "/input_" + std::to_string(spec.index) + ".bin";
    auto meta_it = input_metas.find(spec.index);
    if (meta_it == input_metas.end()) {
      throw std::runtime_error("No meta for input " + std::to_string(spec.index));
    }
    inputs.push_back(BuildUserInput(spec, bin_path, meta_it->second, host_buffers, device_buffers));
  }
  return inputs;
}

// ---- 输出打印 ----

void PrintTensor(const gert::Tensor &tensor, int index) {
  auto shape = tensor.GetStorageShape();
  std::cout << "  output[" << index << "] shape: [";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << shape.GetDim(i);
  }
  std::cout << "]" << std::endl;

  size_t byte_size = tensor.GetSize();
  size_t count = byte_size / sizeof(float);
  if (count == 0 || tensor.GetAddr() == nullptr) {
    std::cout << "  output[" << index << "] data: <empty>" << std::endl;
    return;
  }

  // 输出数据可能在 device 侧, 拷贝到 host 再读取
  std::vector<float> host_buf(count);
  aclrtMemcpy(host_buf.data(), byte_size, tensor.GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);

  std::cout << "  output[" << index << "] data: [";
  size_t print_cnt = std::min(count, size_t(16));
  for (size_t i = 0; i < print_cnt; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << host_buf[i];
  }
  if (count > print_cnt) std::cout << ", ...";
  std::cout << "]" << std::endl;
}

// ---- GE options 构建 ----

std::map<ge::AscendString, ge::AscendString> MakeGeOptions() {
  return {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
}

}  // namespace

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.epair>" << std::endl;
    std::cerr << "\n  请先在 Python 侧运行 export_pack_model.py 生成 model.epair 和输入文件" << std::endl;
    return -1;
  }

  const char *epair_path = argv[1];
  std::string epair_dir = Dirname(epair_path);
  const int kTotalStages = 5;

  // =========================================================
  // [1/5] 初始化 GE + ACL
  // =========================================================
  PrintStage(1, kTotalStages, "GE + ACL 初始化");

  auto ge_opts = MakeGeOptions();
  if (ge::GEInitializeV2(ge_opts) != ge::SUCCESS) {
    std::cerr << "[FAIL] GEInitializeV2" << std::endl;
    return -1;
  }
  std::cout << "  [OK] GE 初始化完成" << std::endl;

  if (aclInit(nullptr) != ACL_SUCCESS) {
    std::cerr << "[WARN] aclInit failed or already initialized" << std::endl;
  }

  // =========================================================
  // [准备] 读取输入 (x 使用 device 内存)
  // =========================================================
  PrintStage(0, kTotalStages, "准备输入数据 (x → device)");
  std::cout << "  epair: " << epair_path << std::endl;
  std::cout << "  input dir: " << epair_dir << std::endl;

  std::vector<std::vector<char>> host_buffers;
  std::vector<void *> device_buffers;
  std::vector<gert::Tensor> user_inputs;
  try {
    user_inputs = BuildAllUserInputs(epair_dir, host_buffers, device_buffers);
  } catch (const std::exception &e) {
    std::cerr << "[FAIL] 构建输入失败: " << e.what() << std::endl;
    for (void *ptr : device_buffers) {
      aclrtFree(ptr);
    }
    aclFinalize();
    ge::GEFinalizeV2();
    return -1;
  }
  std::cout << "  [OK] 共构建 " << user_inputs.size() << " 个输入 (x 在 device)" << std::endl;

  // =========================================================
  // [2/5] 校验 GE Graph: SymPack 优化后的 Data 节点
  // =========================================================
  PrintStage(2, kTotalStages, "校验 GE Graph: SymPack 优化后的 Data 节点");
  {
    td::EpairModelLoader loader(epair_path);
    ge::Graph graph("verify_graph");
    auto st = loader.GetGEGraph(&graph);
    if (st != td::SUCCESS) {
      std::cerr << "  [FAIL] GetGEGraph failed, status=" << st << std::endl;
    } else {
      std::cout << "  [OK] GE Graph 提取成功" << std::endl;
      auto data_nodes = CollectDataNodes(graph);
      std::cout << "  Data 节点 (优化后, 共 " << data_nodes.size() << " 个):" << std::endl;
      PrintDataNodes(data_nodes);
    }
  }

  // =========================================================
  // [3/5] CompileAndLoad + RunModelWithStreamAsync
  //   EpairModelLoader 内部自动按 InputBindingMap 适配输入
  //   user_inputs[x, m, n] → ge_inputs[x, merged_A([m,n]), merged_B([n,m])]
  // =========================================================
  PrintStage(3, kTotalStages, "CompileAndLoad + RunModelWithStreamAsync");

  td::EpairModelLoader loader(epair_path);
  std::cout << "  [OK] EpairModelLoader 构造完成" << std::endl;

  auto options = MakeGeOptions();

  aclrtStream stream = nullptr;
  aclError acl_ret = aclrtCreateStream(&stream);
  if (acl_ret != ACL_ERROR_NONE) {
    std::cerr << "  [FAIL] aclrtCreateStream failed, error=" << acl_ret << std::endl;
  } else {
    auto status = loader.CompileAndLoad(options, stream);
    if (status != td::SUCCESS) {
      std::cerr << "  [FAIL] CompileAndLoad failed, status=" << status << std::endl;
    } else {
      std::cout << "  [OK] 编译加载完成" << std::endl;

      std::vector<gert::Tensor> outputs;
      status = loader.RunModelWithStreamAsync(stream, user_inputs, outputs);
      if (status != td::SUCCESS) {
        std::cerr << "  [FAIL] RunModelWithStreamAsync failed, status=" << status << std::endl;
      } else {
        acl_ret = aclrtSynchronizeStream(stream);
        if (acl_ret != ACL_ERROR_NONE) {
          std::cerr << "  [FAIL] aclrtSynchronizeStream failed, error=" << acl_ret << std::endl;
        } else {
          std::cout << "  [OK] 推理完成, 输出数=" << outputs.size() << std::endl;
          for (size_t i = 0; i < outputs.size(); ++i) {
            PrintTensor(outputs[i], i);
          }
          // 保存输出供比对
          for (size_t i = 0; i < outputs.size(); ++i) {
            std::string out_path = epair_dir + "/cpp_output_a_" + std::to_string(i) + ".bin";
            size_t sz = outputs[i].GetSize();
            std::vector<char> host_out(sz);
            aclrtMemcpy(host_out.data(), sz, outputs[i].GetAddr(), sz, ACL_MEMCPY_DEVICE_TO_HOST);
            std::ofstream ofs(out_path, std::ios::binary);
            ofs.write(host_out.data(), static_cast<std::streamsize>(sz));
            std::cout << "  [save] " << out_path << std::endl;
          }
        }
      }
    }
    aclrtDestroyStream(stream);
  }

  // =========================================================
  // [4/5] 精度比对
  // =========================================================
  PrintStage(4, kTotalStages, "精度比对 (vs Python 参考输出)");

  std::string ref_path = epair_dir + "/ref_output.bin";
  std::ifstream ref_file(ref_path, std::ios::binary);
  if (!ref_file) {
    std::cout << "  [SKIP] 未找到参考输出: " << ref_path << std::endl;
  } else {
    ref_file.seekg(0, std::ios::end);
    size_t ref_size = ref_file.tellg();
    ref_file.seekg(0, std::ios::beg);
    std::vector<char> ref_data(ref_size);
    ref_file.read(ref_data.data(), ref_size);

    std::cout << "  参考输出大小: " << ref_size << " bytes (" << (ref_size / sizeof(float)) << " floats)" << std::endl;

    std::string cpp_out = epair_dir + "/cpp_output_a_0.bin";
    std::ifstream cpp_file(cpp_out, std::ios::binary);
    if (!cpp_file) {
      std::cout << "  [SKIP] " << cpp_out << " 不存在" << std::endl;
    } else {
      cpp_file.seekg(0, std::ios::end);
      size_t cpp_size = cpp_file.tellg();
      cpp_file.seekg(0, std::ios::beg);
      std::vector<char> cpp_data(cpp_size);
      cpp_file.read(cpp_data.data(), cpp_size);

      if (cpp_size != ref_size) {
        std::cout << "  [DIFF] 大小不一致 (ref=" << ref_size << " vs cpp=" << cpp_size << ")" << std::endl;
      } else {
        size_t n = ref_size / sizeof(float);
        const float *ref_f = reinterpret_cast<const float *>(ref_data.data());
        const float *cpp_f = reinterpret_cast<const float *>(cpp_data.data());
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          max_err = std::max(max_err, std::abs(ref_f[i] - cpp_f[i]));
        }

        if (max_err < 1e-5f) {
          std::cout << "  [PASS] 与参考一致 (max_err=" << max_err << ")" << std::endl;
        } else {
          std::cout << "  [DIFF] 超出容差 (max_err=" << max_err << ")" << std::endl;
        }
      }
    }
  }

  // =========================================================
  // [5/5] 清理
  // =========================================================
  PrintStage(5, kTotalStages, "清理");

  for (void *ptr : device_buffers) {
    aclrtFree(ptr);
  }
  std::cout << "  [OK] device 内存已释放" << std::endl;

  aclFinalize();
  ge::GEFinalizeV2();
  std::cout << "  [OK] GE + ACL 已释放" << std::endl;

  std::cout << "\n"
            << kSep << "\n"
            << " 全部完成"
            << "\n"
            << kSep << std::endl;
  return 0;
}
