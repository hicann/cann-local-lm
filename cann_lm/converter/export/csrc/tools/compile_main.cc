#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/ascend_string.h"
#include "ge/ge_ir_build.h"
#include "torch_delegate/epair_model_loader.h"
#include "reg_register.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

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
  // fusion switch 文件 (ge.fusionSwitchFile): 关/开指定 GraphFusion/UBFusion pass。
  // env GE_FUSION_SWITCH_FILE=/path/cfg; cfg 格式见 CANN 图融合文档 (Switch.GraphFusion.ALL=off 等)。
  if (const char *v = std::getenv("GE_FUSION_SWITCH_FILE")) {
    opts.emplace(ge::AscendString("ge.fusionSwitchFile"), ge::AscendString(v));
  }
  // 在线编译模式 (全局 init option): GE_JIT_COMPILE=0(二进制优先)/1(在线,默认)/2(静态在线+动态二进制)
  // =2 对具体 shape 在线编译产静态 kernel, 可减少 _is_op_dynamic_impl 动态算子
  if (const char *v = std::getenv("GE_JIT_COMPILE")) {
    opts.emplace(ge::AscendString("ge.jit_compile"), ge::AscendString(v));
  }
  // AutoTune 调优算子库 (全局): GE_OP_BANK_PATH=<dir>
  if (const char *v = std::getenv("GE_OP_BANK_PATH")) {
    opts.emplace(ge::AscendString("ge.op_bank_path"), ge::AscendString(v));
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
  // SOC version: 本机芯片 Ascend910_9362 (npu-smi 显示), CANN platform_config 有 Ascend910_9362.ini
  // 用环境变量 SOC_VERSION 覆盖 (默认 Ascend910_9362)
  const char *soc_env = std::getenv("SOC_VERSION");
  const std::string soc_version = (soc_env != nullptr) ? std::string(soc_env) : std::string("Ascend950DT");
  std::map<ge::AscendString, ge::AscendString> init_options = GlobalOptionsWithSoc(soc_version);
  // 权值分离: ge.externalWeightDir 走 init_options (全局 context, build 期经 GetThreadLocalContext 读到);
  // 放 build_options 会被拒 (50331649 options illegal). 指定权值落盘到 output 同目录 weight/,
  // 避免 mobile 路径用 tmp_weight_<pid>/ 临时目录后被清 → .omc 引用丢失权重.
  if (const char *w = std::getenv("GE_EXTERNAL_WEIGHT"); w != nullptr && w[0] != '0') {
    std::filesystem::path weight_dir = std::filesystem::path(output_name).parent_path() / "weight";
    std::filesystem::create_directories(weight_dir);
    init_options.emplace(ge::AscendString("ge.externalWeightDir"), ge::AscendString(weight_dir.string().c_str()));
    std::cout << "[INFO] ge.externalWeightDir=" << weight_dir.string() << std::endl;
  }
  ge::graphStatus init_status = aclgrphBuildInitialize(init_options);
  if (init_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphBuildInitialize failed, status: " << init_status << std::endl;
    return -1;
  }
  td::EpairModelLoader loader(model_json_path, weights_index_path);
  std::cout << "[PASS] EpairModelLoader constructed" << std::endl;

  // [2/3] Extract GE Graph (also validates initialization)
  PrintStage(2, kTotalStages, "Extract GE Graph");

  ge::Graph graph("delegate_graph");
  td::Status graph_status = loader.GetGEGraph(&graph);
  if (graph_status != td::SUCCESS) {
    std::cerr << "[ERROR] GetGEGraph failed, status: " << graph_status << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[PASS] GE Graph extracted successfully" << std::endl;

  // 可选: 保存编译前 GE 图为 .air (ge::Graph::SaveToFile, 二进制 protobuf).
  // env SAVE_AIR_FILE=<path> 触发; 若仅设 SAVE_AIR=1 则用 <output_name>.air.
  if (const char *air_env = std::getenv("SAVE_AIR_FILE"); air_env != nullptr && air_env[0] != '\0') {
    const std::string air_path = air_env;
    const ge::graphStatus save_status = graph.SaveToFile(air_path.c_str());
    if (save_status == ge::GRAPH_SUCCESS) {
      std::cout << "[PASS] graph.SaveToFile saved: " << air_path << std::endl;
    } else {
      std::cerr << "[WARN] graph.SaveToFile failed: " << save_status << std::endl;
    }
  } else if (const char *air = std::getenv("SAVE_AIR"); air != nullptr && air[0] == '1') {
    const std::string air_path = output_name + ".air";
    const ge::graphStatus save_status = graph.SaveToFile(air_path.c_str());
    if (save_status == ge::GRAPH_SUCCESS) {
      std::cout << "[PASS] graph.SaveToFile saved: " << air_path << std::endl;
    } else {
      std::cerr << "[WARN] graph.SaveToFile failed: " << save_status << std::endl;
    }
  }

  // [3/3] Compile graph
  PrintStage(3, kTotalStages, "Compile graph");


  std::cout << "[PASS] aclgrphBuildInitialize succeeded" << std::endl;

  ge::ModelBufferData model;
  std::map<ge::AscendString, ge::AscendString> build_options;
  build_options.emplace(ge::AscendString("input_format"), ge::AscendString("ND"));
  // env 开关 (免重编即可调图优化): GE_BUFFER_OPT=off_optimize/l1_optimize/l2_optimize,
  // GE_OO_LEVEL=O1/O3, GE_OPT_SWITCH="pass:off;..."。便于实验 AutomaticBufferFusionOp 等融合开关。
  if (const char *v = std::getenv("GE_BUFFER_OPT")) {
    build_options.emplace(ge::AscendString("ge.bufferOptimize"), ge::AscendString(v));
  }
  if (const char *v = std::getenv("GE_OO_LEVEL")) {
    build_options.emplace(ge::AscendString("ge.oo.level"), ge::AscendString(v));
  }
  if (const char *v = std::getenv("GE_OPT_SWITCH")) {
    build_options.emplace(ge::AscendString("ge.optimizationSwitch"), ge::AscendString(v));
  }
  // ge.graphRunMode: PREDICTION=0 (推理/静态) / TRAIN=1 (训练/动态). 设 0 可能避免动态 shape 分区.
  if (const char *v = std::getenv("GE_RUN_MODE")) {
    build_options.emplace(ge::AscendString("ge.graphRunMode"), ge::AscendString(v));
    std::cout << "[INFO] ge.graphRunMode=" << v << std::endl;
  }
  // 跳过动态 shape 分区 (设 graph 属性 _no_need_dynamic_shape_partition=true, 避免 PartitionedCall)
  if (const char *v = std::getenv("GE_SKIP_DSP"); v != nullptr && v[0] == '1') {
    ge::AttrValue attr_val;
    attr_val.SetAttrValue(true);
    graph.SetAttr(ge::AscendString("_no_need_dynamic_shape_partition"), attr_val);
    std::cout << "[INFO] Set _no_need_dynamic_shape_partition=true (skip dynamic shape partition)" << std::endl;
  }
  // 权值分离: GE_EXTERNAL_WEIGHT=0(内嵌,默认)/1(每个权值分文件)/2(全合一 .bin)
  // ge.externalWeight 走 build_options; ge.externalWeightDir 走 init_options (见上)
  if (const char *v = std::getenv("GE_EXTERNAL_WEIGHT")) {
    build_options.emplace(ge::AscendString("ge.externalWeight"), ge::AscendString(v));
    std::cout << "[INFO] ge.externalWeight=" << v << " (0=内嵌/1=分文件/2=全合一)" << std::endl;
  }
  ge::graphStatus build_status = aclgrphBuildModel(graph, build_options, model);
  if (build_status != ge::GRAPH_SUCCESS) {
    std::cerr << "[ERROR] aclgrphBuildModel failed, status: " << build_status << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[PASS] aclgrphBuildModel succeeded" << std::endl;

  // Mobile model path (kirinx90/HarmonyOS): aclgrphBuildModel via mobile_model_helper
  // 把模型直接写到 CWD 的 "onlinec", 不填 ModelBufferData.data (kirinx90 上 model.length
  // 还会被堆损坏成垃圾值) → aclgrphSaveModel 的 null-data 校验失败。检测 "onlinec"
  // 改名为目标, 跳过多余的 SaveModel, size 取实际文件大小 (非垃圾的 model.length)。
  // (Ascend 走标准路径, 不产生 onlinec, model.length 正常。)
  const std::string mobile_tmp = "onlinec";
  if (std::filesystem::exists(mobile_tmp)) {
    const std::string out_ext = (soc_version.rfind("Kirin", 0) == 0) ? ".omc" : ".om";
    const std::string dst = output_name + out_ext;
    std::filesystem::rename(mobile_tmp, dst);
    std::cout << "[PASS] mobile model path: renamed " << mobile_tmp << " -> " << dst
              << " (size: " << std::filesystem::file_size(dst) << " bytes)" << std::endl;
  } else {
    std::cout << "[INFO] model size: " << model.length << " bytes" << std::endl;
    ge::graphStatus save_status = ge::aclgrphSaveModel(output_name.c_str(), model);
    if (save_status != ge::GRAPH_SUCCESS) {
      std::cerr << "[ERROR] aclgrphSaveModel failed, status: " << save_status << std::endl;
      ge::aclgrphBuildFinalize();
      return -1;
    }
    std::cout << "[PASS] aclgrphSaveModel succeeded, " << output_name << ".om generated" << std::endl;
  }

  ge::aclgrphBuildFinalize();
  std::cout << "[PASS] aclgrphBuildFinalize succeeded" << std::endl;

  std::cout << std::endl;
  std::cout << kStageSep << std::endl;
  std::cout << " All " << kTotalStages << " stages passed" << std::endl;
  std::cout << kStageSep << std::endl;
  return 0;
}
