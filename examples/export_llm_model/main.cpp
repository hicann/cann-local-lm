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

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "[ERROR] Usage: " << argv[0]
              << " <model_json_path> <weights_index_path>" << std::endl;
    return -1;
  }

  constexpr size_t kTotalStages = 3;
  const char *model_json_path = argv[1];
  const char *weights_index_path = argv[2];

  // [1/3] Load epair and build GE Graph
  PrintStage(1, kTotalStages, "Load epair and build GE Graph");
  std::cout << "  model_json: " << model_json_path << std::endl;
  std::cout << "  weights_index: " << weights_index_path << std::endl;

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

  const char *const output_file = "qwen3_model";
  ge::graphStatus save_status = ge::aclgrphSaveModel(output_file, model);
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
  std::cout << " All " << kTotalStages << " stages passed" << std::endl;
  std::cout << kStageSep << std::endl;
  return 0;
}
