#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/ge_error_codes.h"
#include "graph/ascend_string.h"
#include "ge/ge_ir_build.h"
#include "torch_delegate/epair_model_loader.h"
#include "reg_register.h"
#include <iostream>
#include <string>

namespace {

const char *kStageSep = "========================================";

void PrintStage(size_t index, size_t total, const std::string &name) {
  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " [" << index << "/" << total << "] " << name << std::endl;
  std::cout << kStageSep << std::endl;
}

std::map<ge::AscendString, ge::AscendString> GlobalOptionsWithSoc(const std::string &soc_version) {
  std::map<ge::AscendString, ge::AscendString> opts;
  if (!soc_version.empty()) {
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version.c_str()));
  }
  return opts;
}

// Derive output OM name from model_json_path when the caller doesn't pass one.
// Strips the "_ep_json_with_ascend_ir.json" suffix from the filename.
//   prefill_ep_json_with_ascend_ir.json -> "prefill"
//   decode_ep_json_with_ascend_ir.json  -> "decode"
//   ep_json_with_ascend_ir.json         -> "model"
// Falls back to the filename sans ".json", or "model" if that's empty.
// 算子注册参考/mnt/workspace/ge/tests/framework/ge_running_env/include/ge_running_env/op_reg.h 和/mnt/workspace/torchair-full/build/dist/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py
std::string DeriveOutputName(const std::string &model_json_path) {
  std::string filename = model_json_path;
  size_t last_sep = filename.find_last_of('/');
  if (last_sep != std::string::npos) {
    filename = filename.substr(last_sep + 1);
  }
  const std::string kEpSuffix = "_ep_json_with_ascend_ir.json";
  if (filename.size() >= kEpSuffix.size() &&
      filename.compare(filename.size() - kEpSuffix.size(), kEpSuffix.size(), kEpSuffix) == 0) {
    std::string stem = filename.substr(0, filename.size() - kEpSuffix.size());
    return stem.empty() ? std::string("model") : stem;
  }
  const std::string kJsonSuffix = ".json";
  if (filename.size() >= kJsonSuffix.size() &&
      filename.compare(filename.size() - kJsonSuffix.size(), kJsonSuffix.size(), kJsonSuffix) == 0) {
    filename = filename.substr(0, filename.size() - kJsonSuffix.size());
  }
  return filename.empty() ? std::string("model") : filename;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "[ERROR] Usage: " << argv[0]
              << " <model_json_path> <weights_index_path> [output_name]" << std::endl;
    std::cerr << "  output_name defaults to the model_json filename stem" << std::endl;
    return -1;
  }

  constexpr size_t kTotalStages = 3;
  const char *model_json_path = argv[1];
  const char *weights_index_path = argv[2];
  const std::string output_name =
      (argc >= 4) ? std::string(argv[3]) : DeriveOutputName(model_json_path);

  // [1/3] Load epair and build GE Graph
  PrintStage(1, kTotalStages, "Load epair and build GE Graph");
  std::cout << "  model_json:   " << model_json_path << std::endl;
  std::cout << "  weights_index:" << weights_index_path << std::endl;
  std::cout << "  output_name: " << output_name << std::endl;

  td::EpairModelLoader loader(model_json_path, weights_index_path);
  std::cout << "[PASS] EpairModelLoader constructed" << std::endl;

  // [2/3] Extract GE Graph (also validates initialization)
  PrintStage(2, kTotalStages, "Extract GE Graph");

  ge::Graph graph("delegate_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[ERROR] GetGEGraph failed, status: " << graph_status << std::endl;
    return -1;
  }
  std::cout << "[PASS] GE Graph extracted successfully" << std::endl;

  // [3/3] Compile graph
  PrintStage(3, kTotalStages, "Compile graph");

  // SOC version: 本机芯片 Ascend910_9362 (npu-smi 显示), CANN platform_config 有 Ascend910_9362.ini
  // 用环境变量 SOC_VERSION 覆盖 (默认 Ascend910_9362)
  const char *soc_env = std::getenv("SOC_VERSION");
  const std::string soc_version = (soc_env != nullptr) ? std::string(soc_env) : std::string("Ascend910_9362");
  std::map<ge::AscendString, ge::AscendString> init_options = GlobalOptionsWithSoc(soc_version);
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

  ge::graphStatus save_status = ge::aclgrphSaveModel(output_name.c_str(), model);
  if (save_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphSaveModel failed, status: " << save_status << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[PASS] aclgrphSaveModel succeeded, " << output_name << ".om generated" << std::endl;

  ge::aclgrphBuildFinalize();
  std::cout << "[PASS] aclgrphBuildFinalize succeeded" << std::endl;

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " All " << kTotalStages << " stages passed" << std::endl;
  std::cout << kStageSep << std::endl;
  return 0;
}
