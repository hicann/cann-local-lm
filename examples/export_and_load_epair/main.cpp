#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/ge_error_codes.h"
#include "graph/ascend_string.h"
#include "ge/ge_ir_build.h"
#include "torch_delegate/epair_model_loader.h"
#include "reg_register.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

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
  if (begin == end) return "";
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
  auto data_cnt = tensor.GetSize() / sizeof(T);
  auto data_begin = reinterpret_cast<const T *>(tensor.GetAddr());
  if (data_begin == nullptr || data_cnt == 0U) {
    std::cout << "  data:  <empty>" << std::endl;
    return;
  }
  std::cout << "  data:  ";
  for (size_t i = 0; i < data_cnt && i < 16; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << data_begin[i];
  }
  if (data_cnt > 16) std::cout << ", ...";
  std::cout << std::endl;
}

std::string Dirname(const std::string &path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos) return ".";
  return path.substr(0, pos);
}

bool ReadInputBinary(const std::string &epair_path,
                     std::vector<gert::Tensor> &inputs,
                     std::vector<std::vector<float>> &input_buffers) {
  static const int64_t kDims[] = {3, 4};
  static const size_t kRank = 2;
  size_t element_count = 1;
  for (size_t i = 0; i < kRank; ++i) element_count *= kDims[i];
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

std::map<ge::AscendString, ge::AscendString> GlobalOptionsWithSoc(const std::string &soc_version) {
  std::map<ge::AscendString, ge::AscendString> opts;
  if (!soc_version.empty()) {
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version.c_str()));
  }
  return opts;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "[ERROR] Usage: " << argv[0]
              << " <model_json_path> <weights_index_path> [output_basename]" << std::endl;
    std::cerr << "         output_basename defaults to \"cann_lm_graph\"; produces "
                 "<basename>.onnx and <basename>.om" << std::endl;
    return -1;
  }

  constexpr size_t kTotalStages = 4;
  const char *model_json_path = argv[1];
  const char *weights_index_path = argv[2];
  // 可选第 3 参数：输出文件 basename（默认 cann_lm_graph），避免不同模型互相覆盖。
  const std::string output_basename = (argc >= 4) ? argv[3] : "cann_lm_graph";

  // [1/4] Load epair and build GE Graph
  PrintStage(1, kTotalStages, "Load epair and build GE Graph");
  std::cout << "  model_json: " << model_json_path << std::endl;
  std::cout << "  weights_index: " << weights_index_path << std::endl;

  td::EpairModelLoader loader(model_json_path, weights_index_path);
  std::cout << "[PASS] EpairModelLoader constructed" << std::endl;

  // [2/4] Extract GE Graph (also validates initialization)
  PrintStage(2, kTotalStages, "Extract GE Graph");

  ge::Graph graph("delegate_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[ERROR] GetGEGraph failed, status: " << graph_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE Graph extracted successfully" << std::endl;

  // Dump graph to ONNX (for debugging)
  PrintStage(3, kTotalStages, "Dump graph to ONNX");

  ge::graphStatus dump_status =
      graph.DumpToFile(ge::Graph::DumpFormat::kOnnx, output_basename.c_str());
  if (dump_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] DumpToFile failed, status: " << dump_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] Graph dumped to " << output_basename << ".onnx" << std::endl;

  // [4/4] Compile graph
  PrintStage(4, kTotalStages, "Compile graph");

  std::map<ge::AscendString, ge::AscendString> init_options = GlobalOptionsWithSoc("Ascend950DT_950x");
  ge::graphStatus init_status = aclgrphBuildInitialize(init_options);
  if (init_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphBuildInitialize failed, status: " << init_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] aclgrphBuildInitialize succeeded" << std::endl;

  ge::ModelBufferData model;
  std::map<ge::AscendString, ge::AscendString> build_options;
  build_options.emplace(ge::AscendString("input_format"), ge::AscendString("ND"));
  ge::graphStatus build_status = aclgrphBuildModel(graph, build_options, model);
  if (build_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphBuildModel failed, status: " << build_status << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[PASS] aclgrphBuildModel succeeded, model size: " << model.length << " bytes" << std::endl;

  const std::string output_file = output_basename;
  ge::graphStatus save_status = ge::aclgrphSaveModel(output_file.c_str(), model);
  if (save_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphSaveModel failed, status: " << save_status << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[PASS] aclgrphSaveModel succeeded, " << output_file << ".om generated" << std::endl;

  ge::aclgrphBuildFinalize();
  std::cout << "[PASS] aclgrphBuildFinalize succeeded" << std::endl;

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " All 4 stages passed" << std::endl;
  std::cout << kStageSep << std::endl;
  return 0;
}
