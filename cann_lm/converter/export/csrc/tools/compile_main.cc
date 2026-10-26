#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/ascend_string.h"
#include "ge/ge_ir_build.h"
#include "torch_delegate/epair_model_loader.h"
#include "reg_register.h"
#include <cctype>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>

namespace {

const char* kStageSep = "========================================";

void PrintStage(size_t index, size_t total, const std::string& name)
{
    std::cout << std::endl;
    std::cout << kStageSep << std::endl;
    std::cout << " [" << index << "/" << total << "] " << name << std::endl;
    std::cout << kStageSep << std::endl;
}

// Split a tap list value by ';' or ',' (trim whitespace). 支持 --tap "a;b;c" / --tap=a,b / 多次 --tap 混用.
std::vector<std::string> SplitTapList(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&out, &cur]() {
        size_t b = 0, e = cur.size();
        while (b < e && std::isspace(static_cast<unsigned char>(cur[b])))
            ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(cur[e - 1])))
            --e;
        if (e > b)
            out.push_back(cur.substr(b, e - b));
        cur.clear();
    };
    for (char c : s) {
        if (c == ';' || c == ',')
            flush();
        else
            cur.push_back(c);
    }
    flush();
    return out;
}

std::map<ge::AscendString, ge::AscendString> GlobalOptionsWithSoc(const std::string& soc_version)
{
    std::map<ge::AscendString, ge::AscendString> opts;
    if (!soc_version.empty()) {
        opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version.c_str()));
    }
    if (const char* v = std::getenv("GE_FUSION_SWITCH_FILE")) {
        opts.emplace(ge::AscendString("ge.fusionSwitchFile"), ge::AscendString(v));
    }
    if (const char* v = std::getenv("GE_JIT_COMPILE")) {
        opts.emplace(ge::AscendString("ge.jit_compile"), ge::AscendString(v));
    }
    return opts;
}

std::string DeriveOutputName(const std::string& model_json_path)
{
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

} // namespace

int main(int argc, char* argv[])
{
    // 解析: --tap "a;b;c" (单条 flag, ;/, 分隔) + 位置参数 [model_json] [weights_index] [output_name]
    std::vector<std::string> tap_outputs;
    std::vector<std::string> positional;
    bool tap_seen = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tap" || arg.rfind("--tap=", 0) == 0) {
            if (tap_seen) {
                std::cerr << "[ERROR] multiple --tap not allowed; use ONE --tap with ';' / ',' separated list,"
                          << " e.g. --tap \"mm;add_1\"" << std::endl;
                return -1;
            }
            tap_seen = true;
            std::string val;
            if (arg == "--tap") {
                if (i + 1 >= argc) {
                    std::cerr << "[ERROR] --tap requires a tensor name list (e.g. --tap \"mm;add_1\")" << std::endl;
                    return -1;
                }
                val = argv[++i];
            } else {
                val = arg.substr(6);
            }
            std::vector<std::string> parts = SplitTapList(val);
            if (parts.empty()) {
                std::cerr << "[ERROR] --tap has no valid names after splitting (e.g. --tap \"mm;add_1\")" << std::endl;
                return -1;
            }
            tap_outputs = std::move(parts);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 2) {
        std::cerr << "[ERROR] Usage: " << argv[0]
                  << " <model_json_path> <weights_index_path> [output_name] [--tap \"<name>;<name>...\"]" << std::endl;
        std::cerr << "  output_name defaults to the model_json filename stem" << std::endl;
        std::cerr << "  --tap \"n1;n2\"  tap intermediate FX output tensors as extra graph outputs"
                  << " (ONE --tap, ';' / ',' separated; single-output node uses node.name, e.g. \"mm;add_1\")"
                  << std::endl;
        return -1;
    }

    constexpr size_t kTotalStages = 3;
    const std::string model_json_path = positional[0];
    const std::string weights_index_path = positional[1];
    const std::string output_name = (positional.size() >= 3) ? positional[2] : DeriveOutputName(model_json_path);

    // [1/3] Load epair and build GE Graph
    PrintStage(1, kTotalStages, "Load epair and build GE Graph");
    std::cout << "  model_json:   " << model_json_path << std::endl;
    std::cout << "  weights_index:" << weights_index_path << std::endl;
    std::cout << "  output_name: " << output_name << std::endl;

    const char* soc_env = std::getenv("SOC_VERSION");
    const std::string soc_version = (soc_env != nullptr) ? std::string(soc_env) : std::string("KirinX90");
    std::map<ge::AscendString, ge::AscendString> init_options = GlobalOptionsWithSoc(soc_version);

    std::string weight_dir_path;
    if (const char* w = std::getenv("GE_EXTERNAL_WEIGHT"); w != nullptr && w[0] != '0') {
        std::filesystem::path weight_dir = std::filesystem::path(output_name).parent_path() / "weight";
        std::filesystem::create_directories(weight_dir);
        weight_dir_path = weight_dir.string();
        init_options.emplace(ge::AscendString("ge.externalWeightDir"), ge::AscendString(weight_dir_path.c_str()));
        std::cout << "[INFO] ge.externalWeightDir=" << weight_dir_path << std::endl;
    }
    ge::graphStatus init_status = aclgrphBuildInitialize(init_options);
    if (init_status != ge::GRAPH_SUCCESS) {
        std::cerr << "[ERROR] aclgrphBuildInitialize failed, status: " << init_status << std::endl;
        return -1;
    }
    td::EpairModelLoader loader(model_json_path.c_str(), weights_index_path.c_str(), tap_outputs);
    if (!tap_outputs.empty()) {
        std::cout << "[INFO] tap outputs (" << tap_outputs.size() << "):";
        for (const auto& t : tap_outputs)
            std::cout << " " << t;
        std::cout << std::endl;
    }
    std::cout << "[PASS] EpairModelLoader constructed" << std::endl;

    // [2/3] Extract GE Graph (also validates initialization)
    PrintStage(2, kTotalStages, "Extract GE Graph");

    ge::Graph graph("delegate_graph");
    td::Status graph_status = loader.GetGEGraph(&graph);
    if (graph_status != td::SUCCESS) {
        std::cerr << "[ERROR] GetGEGraph failed, status: " << graph_status << std::endl;
        ge::aclgrphBuildFinalize();
        std::cerr.flush();
        std::quick_exit(-1);
    }
    std::cout << "[PASS] GE Graph extracted successfully" << std::endl;

    // 可选: 保存编译前 GE 图为 .air (ge::Graph::SaveToFile, 二进制 protobuf).
    // env SAVE_AIR_FILE=<path> 触发; 若仅设 SAVE_AIR=1 则用 <output_name>.air.
    if (const char* air_env = std::getenv("SAVE_AIR_FILE"); air_env != nullptr && air_env[0] != '\0') {
        const std::string air_path = air_env;
        const ge::graphStatus save_status = graph.SaveToFile(air_path.c_str());
        if (save_status == ge::GRAPH_SUCCESS) {
            std::cout << "[PASS] graph.SaveToFile saved: " << air_path << std::endl;
        } else {
            std::cerr << "[WARN] graph.SaveToFile failed: " << save_status << std::endl;
        }
    } else if (const char* air = std::getenv("SAVE_AIR"); air != nullptr && air[0] == '1') {
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
    if (const char* v = std::getenv("GE_BUFFER_OPT")) {
        build_options.emplace(ge::AscendString("ge.bufferOptimize"), ge::AscendString(v));
    }
    if (const char* v = std::getenv("GE_OO_LEVEL")) {
        build_options.emplace(ge::AscendString("ge.oo.level"), ge::AscendString(v));
    }
    if (const char* v = std::getenv("GE_OPT_SWITCH")) {
        build_options.emplace(ge::AscendString("ge.optimizationSwitch"), ge::AscendString(v));
    }
    if (const char* v = std::getenv("GE_EXTERNAL_WEIGHT")) {
        build_options.emplace(ge::AscendString("ge.externalWeight"), ge::AscendString(v));
        std::cout << "[INFO] ge.externalWeight=" << v << " (0=内嵌/1=分文件/2=全合一)" << std::endl;
    }
    ge::graphStatus build_status = aclgrphBuildModel(graph, build_options, model);
    if (build_status != ge::GRAPH_SUCCESS) {
        std::cerr << "[ERROR] aclgrphBuildModel failed, status: " << build_status << std::endl;
        ge::aclgrphBuildFinalize();
        std::cerr.flush();
        std::quick_exit(-1);
    }
    std::cout << "[PASS] aclgrphBuildModel succeeded" << std::endl;

    // kirinx90里面调用aclgrphSaveModel流程里面自己调用的SaveToOmModel流程
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
            std::cerr.flush();
            std::quick_exit(-1);
        }
        std::cout << "[PASS] aclgrphSaveModel succeeded, " << output_name << ".om generated" << std::endl;
    }

    ge::aclgrphBuildFinalize();
    std::cout << "[PASS] aclgrphBuildFinalize succeeded" << std::endl;

    std::cout << std::endl;
    std::cout << kStageSep << std::endl;
    std::cout << " All " << kTotalStages << " stages passed" << std::endl;
    std::cout << kStageSep << std::endl;
    std::cout.flush();
    std::quick_exit(0);
}
