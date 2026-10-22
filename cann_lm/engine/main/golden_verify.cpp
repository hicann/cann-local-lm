// ═══════════════════════════════════════════════════════════════════════════
// 【临时调试代码 — 验证完成后删除】
//
// golden_verify — 模型精度验证工具。
//
// 从 golden 目录读取 input_x_tensorName 格式的二进制文件，灌入模型输入工位，
// 执行推理（ForwardRaw，跳过 InputAssembler），dump 输出为 output_x.bin。
//
// 用法:
//   golden_verify
//     --prefill-omc <path.omc>
//     --meta-json  <path.json>
//     --weight-dir-prefill <path>   (prefill 权重目录)
//     --weight-dir-decode  <path>   (decode 权重目录; 与 prefill 共用时可用 --weight-dir)
//     --golden-dir <path>
//     --output-dir <path>
//     [--mode prefill|decode]       (验证哪个图，默认 prefill)
//     [--seq-len <N>]               (prefill 默认 0 = 不修正; decode 默认 1)
//     [--decode-omc <path.omc>]     (不传则用 prefill-omc)
//     [--weight-dir <path>]         (共用权重速记: 等价于两个 weight-dir 同值)
//     [--bind-prefill]              (仅 mode=decode: 复现正式部署形态——先加载
//                                    prefill 图作权值共享源，ModelId() 传给 decode
//                                    Load 的 share_weights_model_id，对齐
//                                    pipeline_executor.cpp 的双图加载顺序)
//
// golden 文件匹配策略（按优先级）:
//   1. input_<index>_<tensorName>.bin   (索引+名称双匹配)
//   2. input_<tensorName>.bin           (仅名称匹配)
//   3. input_<index>.bin                (仅索引匹配)
//
// 输出文件:
//   output_<index>_<tensorName>.bin     (与 input 命名风格一致)
//
// 二进制格式: 裸数据，无文件头，行优先，可直接 numpy.fromfile 读取。
// ═══════════════════════════════════════════════════════════════════════════

#include "core/component/model_executor/model_executor.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/dfx_collector/dfx_collector.h"
#include "core/tensor_manager.h"
#include "infra/log.h"
#include "infra/status.h"
#include "utils/data_type_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace fs = std::filesystem;

using cann_lm::engine::DataType;
using cann_lm::engine::DfxCollector;
using cann_lm::engine::DfxConfig;
using cann_lm::engine::InputAssembler;
using cann_lm::engine::ITensorBuffer;
using cann_lm::engine::KvCachePool;
using cann_lm::engine::ModelConfig;
using cann_lm::engine::ModelExecutor;
using cann_lm::engine::Status;
using cann_lm::engine::TensorDesc;
using cann_lm::engine::TensorManager;

// ── 辅助函数 ─────────────────────────────────────────────────────────────────

static void PrintUsage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --prefill-omc <path> --meta-json <path>\n"
        "     --weight-dir-prefill <path> --weight-dir-decode <path>\n"
        "     --golden-dir <path> --output-dir <path>\n"
        "     [--mode prefill|decode] [--seq-len <N>] [--decode-omc <path>]\n"
        "     [--weight-dir <path>] [--bind-prefill]\n"
        "\n"
        "  --prefill-omc <path>          prefill 模型图位 (.omc)\n"
        "  --decode-omc  <path>          decode  模型图位（缺省 = prefill-omc）\n"
        "  --meta-json   <path>          模型元信息 JSON\n"
        "  --weight-dir-prefill <path>   prefill 权重目录\n"
        "  --weight-dir-decode  <path>   decode 权重目录\n"
        "  --weight-dir  <path>          共用权重速记（等价于两个 weight-dir 同值）\n"
        "  --golden-dir  <path>          golden 输入二进制目录\n"
        "  --output-dir  <path>          输出 dump 目录\n"
        "  --mode        prefill|decode  验证哪个图（默认 prefill）\n"
        "  --seq-len     <N>             序列长度（prefill 默认 0 = 不修正 tensorDesc；\n"
        "                                decode 默认 1）\n"
        "  --bind-prefill                仅 mode=decode: 复现正式部署形态 双图加载\n"
        "                                （先 prefill 作权值共享源，ModelId 绑定给\n"
        "                                decode 的 share_weights_model_id）\n",
        prog);
}

struct Args {
    std::string prefill_omc;
    std::string decode_omc;
    std::string meta_json;
    std::string weight_dir_prefill;   ///< prefill 权重目录（ModelArtifact.prefill_weight 同构）
    std::string weight_dir_decode;    ///< decode 权重目录（ModelArtifact.decode_weight 同构）
    std::string golden_dir;
    std::string output_dir;
    std::string mode = "prefill";     ///< prefill | decode（加载哪个图 + 用哪侧权重）
    int seq_len = -1;                 ///< -1 = 自动（prefill:0 不修正; decode:1）
    bool bind_prefill = false;        ///< mode=decode: 先加载 prefill 图作权值共享源，
                                      ///< ModelId() 绑定给 decode（复现正式部署形态）
};

static bool ParseArgs(int argc, char** argv, Args* out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "golden_verify: missing value for %s\n", name);
                return "";
            }
            return argv[++i];
        };
        if (a == "--prefill-omc")        out->prefill_omc = next("--prefill-omc");
        else if (a == "--decode-omc")    out->decode_omc  = next("--decode-omc");
        else if (a == "--meta-json")     out->meta_json   = next("--meta-json");
        else if (a == "--weight-dir-prefill") out->weight_dir_prefill = next("--weight-dir-prefill");
        else if (a == "--weight-dir-decode")  out->weight_dir_decode  = next("--weight-dir-decode");
        else if (a == "--weight-dir") {   // 共用权重速记：两侧同值（显式分传优先）
            std::string w = next("--weight-dir");
            if (out->weight_dir_prefill.empty()) out->weight_dir_prefill = w;
            if (out->weight_dir_decode.empty())  out->weight_dir_decode  = w;
        }
        else if (a == "--golden-dir")    out->golden_dir  = next("--golden-dir");
        else if (a == "--output-dir")    out->output_dir  = next("--output-dir");
        else if (a == "--mode") {
            out->mode = next("--mode");
            if (out->mode != "prefill" && out->mode != "decode") {
                fprintf(stderr, "golden_verify: invalid --mode '%s' (expect prefill|decode)\n",
                        out->mode.c_str());
                return false;
            }
        }
        else if (a == "--seq-len")       out->seq_len     = std::atoi(next("--seq-len").c_str());
        else if (a == "--bind-prefill")  out->bind_prefill = true;
        else if (a == "--help" || a == "-h") { PrintUsage(argv[0]); return false; }
        else {
            fprintf(stderr, "golden_verify: unknown argument: %s\n", a.c_str());
            PrintUsage(argv[0]);
            return false;
        }
    }
    if (out->prefill_omc.empty() || out->meta_json.empty() ||
        out->golden_dir.empty() || out->output_dir.empty()) {
        fprintf(stderr, "golden_verify: missing required arguments\n");
        PrintUsage(argv[0]);
        return false;
    }
    // 所选 mode 的权重目录必须供给（另一侧可选——单图验证用不到）
    const std::string& need_weight =
        (out->mode == "decode") ? out->weight_dir_decode : out->weight_dir_prefill;
    // --bind-prefill 仅在 mode=decode 下有意义（prefill 模式没有 decode 图可绑定）
    if (out->bind_prefill && out->mode != "decode") {
        fprintf(stderr, "golden_verify: --bind-prefill is only valid with --mode decode\n");
        PrintUsage(argv[0]);
        return false;
    }
    if (need_weight.empty()) {
        fprintf(stderr, "golden_verify: missing --weight-dir-%s (or --weight-dir for shared)\n",
                out->mode.c_str());
        PrintUsage(argv[0]);
        return false;
    }
    // bind-prefill: prefill 图作权值共享源一并加载，其权重目录也必须供给
    if (out->bind_prefill && out->weight_dir_prefill.empty()) {
        fprintf(stderr, "golden_verify: --bind-prefill requires --weight-dir-prefill "
                        "(prefill graph is loaded as weights-share source)\n");
        PrintUsage(argv[0]);
        return false;
    }
    if (out->decode_omc.empty()) out->decode_omc = out->prefill_omc;
    return true;
}

/// 文件是否存在
static bool FileExists(const std::string& path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0;
}

/// 确保目录存在
static void EnsureDir(const std::string& dir) {
    fs::create_directories(dir);
}

/// 读取整个文件到 vector<char>
static std::vector<char> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<char>(std::istreambuf_iterator<char>(f),
                             std::istreambuf_iterator<char>());
}

/// DataType 名称
static const char* DtypeName(DataType t) {
    switch (t) {
        case DataType::UINT8:    return "UINT8";
        case DataType::FLOAT32:  return "FLOAT32";
        case DataType::FLOAT16:  return "FLOAT16";
        case DataType::INT32:    return "INT32";
        case DataType::INT8:     return "INT8";
        case DataType::INT16:    return "INT16";
        case DataType::BOOL:     return "BOOL";
        case DataType::INT64:    return "INT64";
        case DataType::UINT32:   return "UINT32";
        case DataType::DOUBLE:   return "DOUBLE";
        case DataType::RESERVED: return "RESERVED";
    }
    return "unknown";
}

/// dims 向量转字符串
static std::string DimsStr(const std::vector<int32_t>& dims) {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(dims[i]);
    }
    s += "]";
    return s;
}

// ── meta_json 最小解析（仅 model_config 段）──────────────────────────────────
// 复制自 pipeline_executor.cpp 匿名命名空间的子集，只取 ModelConfig 所需字段。

static std::optional<long long> AsInt(const nlohmann::json& v) {
    if (v.is_number_integer()) return v.get<long long>();
    if (v.is_string()) {
        try { return std::stoll(v.get<std::string>()); }
        catch (...) { return std::nullopt; }
    }
    return std::nullopt;
}

/// 解析 meta_json 的 model_config 段 → ModelConfig
static bool ParseModelConfig(const std::string& meta_json_path, ModelConfig* out) {
    ModelConfig mc;

    auto bytes = ReadFile(meta_json_path);
    if (bytes.empty()) {
        fprintf(stderr, "golden_verify: cannot read meta_json: %s\n", meta_json_path.c_str());
        return false;
    }

    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(bytes);
    } catch (const std::exception& e) {
        fprintf(stderr, "golden_verify: meta_json parse error: %s\n", e.what());
        return false;
    }

    auto mit = doc.find("model_config");
    if (mit == doc.end() || !mit->is_object() || mit->empty()) {
        fprintf(stderr, "golden_verify: model_config section missing or empty\n");
        return false;
    }
    const auto& sec = *mit;

    auto get_int = [&](const char* key, int* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null()) return true;
        auto v = AsInt(*it);
        if (!v) return false;
        *dst = static_cast<int>(*v);
        return true;
    };
    auto get_str = [&](const char* key, std::string* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null()) return true;
        if (!it->is_string()) return false;
        *dst = it->get<std::string>();
        return true;
    };
    auto get_arr = [&](const char* key, std::vector<int>* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null()) return true;
        if (!it->is_array()) return false;
        for (const auto& e : *it) {
            auto n = AsInt(e);
            if (!n) return false;
            dst->push_back(static_cast<int>(*n));
        }
        return true;
    };

    if (!get_int("hidden_size",       &mc.hidden_size)) return false;
    if (!get_int("max_io_tokens",     &mc.max_io_tokens)) return false;
    if (!get_str("model_type",        &mc.model_type)) return false;
    if (!get_int("num_hidden_layers", &mc.num_hidden_layers)) return false;
    if (!get_int("pad_token_id",      &mc.pad_token_id)) return false;
    if (!get_int("kv_cache_max_len",  &mc.kv_cache_max_len)) return false;
    {
        auto it = sec.find("vocab_size");
        if (it != sec.end() && !it->is_null()) {
            auto n = AsInt(*it);
            if (!n) return false;
            mc.vocab_size = static_cast<size_t>(*n);
        }
        it = sec.find("vocab_align_size");
        if (it != sec.end() && !it->is_null()) {
            auto n = AsInt(*it);
            if (!n) return false;
            mc.vocab_align_size = static_cast<size_t>(*n);
        }
    }
    if (!get_arr("prefill_len", &mc.prefill_len)) return false;
    if (!get_arr("decode_len",  &mc.decode_len)) return false;
    if (!get_str("kv_cache_format", &mc.kv_cache_format)) return false;

    // KV 形状缺省值（与 pipeline_executor 桩默认一致）
    mc.num_attention_kv_heads  = 4;
    mc.num_attention_head_dims = 8;
    if (!get_int("num_attention_kv_heads",  &mc.num_attention_kv_heads)) return false;
    if (!get_int("num_attention_head_dims", &mc.num_attention_head_dims)) return false;
    if (!get_int("batch_num",               &mc.batch_num)) return false;
    if (!get_str("kv_cache_type",           &mc.kv_cache_type)) return false;

    // sliding_window
    if (!get_int("sliding_window", &mc.sliding_window)) return false;
    if (mc.sliding_window < 0) return false;

    // layer_types（Qwen3.5）
    if (auto it = sec.find("layer_types"); it != sec.end() && !it->is_null()) {
        if (!it->is_array()) return false;
        std::vector<std::string> types;
        for (const auto& e : *it) {
            if (!e.is_string()) return false;
            const auto& s = e.get<std::string>();
            if (s != "full_attention" && s != "linear_attention") return false;
            types.push_back(s);
        }
        if (static_cast<int>(types.size()) != mc.num_hidden_layers) return false;
        mc.layer_types = std::move(types);
    }

    // embedding per_group
    auto get_bool = [&](const char* key, bool* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null()) return true;
        if (!it->is_boolean()) return false;
        *dst = it->get<bool>();
        return true;
    };
    if (!get_bool("embedding_per_group", &mc.embedding_per_group)) return false;
    if (!get_int("embedding_group_size", &mc.embedding_group_size)) return false;

    // 容量合法性
    if (mc.kv_cache_max_len <= 0 || mc.num_hidden_layers <= 0 ||
        mc.num_attention_kv_heads <= 0 || mc.num_attention_head_dims <= 0) {
        fprintf(stderr, "golden_verify: model_config capacity invalid "
                "(kv_len=%d, layers=%d, kv_heads=%d, head_dim=%d)\n",
                mc.kv_cache_max_len, mc.num_hidden_layers,
                mc.num_attention_kv_heads, mc.num_attention_head_dims);
        return false;
    }

    *out = mc;
    return true;
}

/// kv_cache_type 字符串 → DataType
static DataType ResolveKvDtype(const std::string& type) {
    if (type.empty()) return DataType::FLOAT16;
    std::string low = type;
    for (char& c : low) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    if (low == "fp16" || low == "float16") return DataType::FLOAT16;
    if (low == "fp32" || low == "float32") return DataType::FLOAT32;
    if (low == "int8")  return DataType::INT8;
    if (low == "int16") return DataType::INT16;
    if (low == "int32") return DataType::INT32;
    return DataType::FLOAT16;
}

// ── golden 文件匹配 ──────────────────────────────────────────────────────────

/// 为输入工位查找匹配的 golden 文件路径
static std::string FindGoldenFile(const std::string& dir, std::size_t idx,
                                   const std::string& name) {
    // 策略 1: input_<idx>_<name>.bin
    std::string p1 = dir + "/input_" + std::to_string(idx) + "_" + name + ".bin";
    if (FileExists(p1)) return p1;

    // 策略 2: input_<name>.bin
    std::string p2 = dir + "/input_" + name + ".bin";
    if (FileExists(p2)) return p2;

    // 策略 3: input_<idx>.bin
    std::string p3 = dir + "/input_" + std::to_string(idx) + ".bin";
    if (FileExists(p3)) return p3;

    return "";
}

// ── 输出 dump ────────────────────────────────────────────────────────────────

/// dump 单个输出 tensor 到 output_<idx>_<name>.bin
static void DumpOutputTensor(const std::shared_ptr<ITensorBuffer>& t,
                             std::size_t idx, const std::string& out_dir) {
    if (!t || !t->GetData()) {
        fprintf(stderr, "  output[%zu]: null tensor, skipped\n", idx);
        return;
    }
    const auto& desc = t->GetTensorDesc();
    std::string name = desc.tensorName.empty() ? "unnamed" : desc.tensorName;
    std::string path = out_dir + "/output_" + std::to_string(idx) + "_" + name + ".bin";

    std::size_t bytes = t->GetSize();
    if (bytes == 0) {
        fprintf(stderr, "  output[%zu]: size=0, skipped\n", idx);
        return;
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "  output[%zu]: cannot open %s for write\n", idx, path.c_str());
        return;
    }
    std::size_t written = fwrite(t->GetData(), 1, bytes, fp);
    fclose(fp);
    if (written != bytes) {
        fprintf(stderr, "  output[%zu]: short write %s (%zu/%zu)\n",
                idx, path.c_str(), written, bytes);
        return;
    }

    fprintf(stderr, "  output[%zu] name=%-24s shape=%-20s dtype=%-8s bytes=%zu -> %s\n",
            idx, name.c_str(), DimsStr(desc.dims).c_str(),
            DtypeName(desc.dataType), bytes, path.c_str());
}

// ── 主流程 ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, &args)) return 2;

    // 按 mode 选定图位与权重路径（prefill/decode 双权重路径，与 ModelArtifact 同构）
    const bool is_decode = (args.mode == "decode");
    const std::string omc_path   = is_decode ? args.decode_omc : args.prefill_omc;
    const std::string weight_dir = is_decode ? args.weight_dir_decode : args.weight_dir_prefill;
    // seq_len 自动默认：prefill = 0（不修正 tensorDesc）；decode = 1
    const int seq_len = (args.seq_len >= 0) ? args.seq_len : (is_decode ? 1 : 0);

    // 校验文件存在
    if (!FileExists(omc_path)) {
        fprintf(stderr, "golden_verify: %s-omc not found: %s\n",
                is_decode ? "decode" : "prefill", omc_path.c_str());
        return 2;
    }
    // bind-prefill 时 prefill 图也要加载，提前校验避免半途报错
    if (args.bind_prefill && !FileExists(args.prefill_omc)) {
        fprintf(stderr, "golden_verify: prefill-omc (weights-share source) not found: %s\n",
                args.prefill_omc.c_str());
        return 2;
    }
    if (!FileExists(args.meta_json)) {
        fprintf(stderr, "golden_verify: meta-json not found: %s\n", args.meta_json.c_str());
        return 2;
    }

    EnsureDir(args.output_dir);

    // 1. 解析 meta_json → ModelConfig
    fprintf(stderr, "=== golden_verify: parsing meta_json ===\n");
    ModelConfig config;
    if (!ParseModelConfig(args.meta_json, &config)) {
        fprintf(stderr, "golden_verify: failed to parse model_config from %s\n",
                args.meta_json.c_str());
        return 1;
    }
    fprintf(stderr, "  model_config: hidden=%d layers=%d kv_len=%d kv_heads=%d head_dim=%d "
            "batch=%d kv_type=%s\n",
            config.hidden_size, config.num_hidden_layers, config.kv_cache_max_len,
            config.num_attention_kv_heads, config.num_attention_head_dims,
            config.batch_num, config.kv_cache_type.c_str());

    // 2. 解析 KV dtype
    DataType kv_dtype = ResolveKvDtype(config.kv_cache_type);

    // 3. 创建组件
    int max_batch = (config.batch_num > 0) ? config.batch_num : 1;
    auto kv_pool = std::make_shared<KvCachePool>(
        max_batch, config.kv_cache_max_len, config.num_hidden_layers,
        config.num_attention_kv_heads, config.num_attention_head_dims,
        kv_dtype, config.layer_types);

    auto assembler = std::make_shared<InputAssembler>(kv_pool, max_batch,
                                                       config.sliding_window);

    auto dfx = std::make_shared<DfxCollector>(DfxConfig{}, kv_pool);

    // 4. 创建 ModelExecutor 并加载所选图位（--mode prefill|decode）
    ModelExecutor executor(kv_pool, assembler, dfx);

    // --bind-prefill: 复现正式部署形态（pipeline_executor.cpp:760-781 双图加载顺序）。
    // 先加载 prefill 图作权值共享源，ModelId() 传给 decode Load 的
    // share_weights_model_id。绑定后 decode 仍按自身 weight_dir InitWeights、
    // 共享权值由底层同源去重——数值与 standalone 等价，但覆盖"绑定路径"本身。
    // 共享源须存活到本图前向 + dump 完成。
    std::unique_ptr<ModelExecutor> share_src;
    uint32_t share_weights_model_id = 0;
    if (args.bind_prefill) {
        // prefill 侧 seq_len 对齐正式链路 REQ-03 协商链第一优先级:
        // prefill_len[0] > 1024（工具无 options 层，跳过 default_chunk_size 分支）
        const int prefill_bind_seq_len =
            (!config.prefill_len.empty() && config.prefill_len[0] > 0)
                ? config.prefill_len[0] : 1024;
        fprintf(stderr, "=== golden_verify: --bind-prefill: loading prefill graph "
                        "as weights-share source ===\n");
        fprintf(stderr, "  prefill omc:     %s\n", args.prefill_omc.c_str());
        fprintf(stderr, "  prefill weight: %s\n", args.weight_dir_prefill.c_str());
        fprintf(stderr, "  prefill seq_len: %d (REQ-03: prefill_len[0] || 1024)\n",
                prefill_bind_seq_len);

        share_src = std::make_unique<ModelExecutor>(kv_pool, assembler, dfx);
        cann_lm::engine::ArtifactSource prefill_src;
        prefill_src.path = args.prefill_omc;
        const Status prefill_st = share_src->Load(
            prefill_src, args.weight_dir_prefill, config, prefill_bind_seq_len);
        if (!prefill_st.ok()) {
            fprintf(stderr, "golden_verify: prefill (share-source) Load failed (st=%d)\n",
                    prefill_st.code);
            return 1;
        }
        // 对齐正式链路: id==0（老 ROM 取不到 GetModelID）→ decode 跳过绑定照常加载
        share_weights_model_id = share_src->ModelId();
        fprintf(stderr, "  prefill loaded, ModelId=%u%s\n", share_weights_model_id,
                (share_weights_model_id == 0)
                    ? "  (==0: decode skips share-weights binding, same as pipeline)"
                    : "");
    }

    fprintf(stderr, "=== golden_verify: loading model (mode=%s) ===\n", args.mode.c_str());
    fprintf(stderr, "  omc: %s\n", omc_path.c_str());
    fprintf(stderr, "  weight: %s\n", weight_dir.c_str());
    fprintf(stderr, "  seq_len: %d%s\n", seq_len,
            args.bind_prefill ? " (decode side; prefill source uses own seq_len)" : "");

    cann_lm::engine::ArtifactSource omc_src;
    omc_src.path = omc_path;

    Status st = executor.Load(omc_src, weight_dir, config, seq_len, share_weights_model_id);
    if (!st.ok()) {
        fprintf(stderr, "golden_verify: model Load failed (st=%d)%s\n", st.code,
                args.bind_prefill
                    ? "  (bind-prefill: 绑定失败老 ROM 快速失败属预期行为)" : "");
        return 1;
    }
    fprintf(stderr, "  model loaded successfully (share_weights_model_id=%u)\n",
            share_weights_model_id);

    // 5. 打印输入/输出工位信息
    const auto& inputs = executor.InputTensors();
    const auto& outputs = executor.OutputTensors();
    fprintf(stderr, "=== golden_verify: model IO ===\n");
    fprintf(stderr, "  inputs: %zu\n", inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& desc = inputs[i]->GetTensorDesc();
        fprintf(stderr, "    input[%zu] name=%-24s shape=%-20s dtype=%-8s size=%zu\n",
                i, desc.tensorName.c_str(), DimsStr(desc.dims).c_str(),
                DtypeName(desc.dataType), inputs[i]->GetSize());
    }
    fprintf(stderr, "  outputs: %zu\n", outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& desc = outputs[i]->GetTensorDesc();
        fprintf(stderr, "    output[%zu] name=%-24s shape=%-20s dtype=%-8s size=%zu\n",
                i, desc.tensorName.c_str(), DimsStr(desc.dims).c_str(),
                DtypeName(desc.dataType), outputs[i]->GetSize());
    }

    // 6. 从 golden 目录读取二进制文件，填充输入工位
    fprintf(stderr, "=== golden_verify: loading golden inputs from %s ===\n",
            args.golden_dir.c_str());
    int filled = 0;
    int skipped = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& tensor = inputs[i];
        if (!tensor || !tensor->GetData()) {
            fprintf(stderr, "  input[%zu]: null tensor, skipped\n", i);
            ++skipped;
            continue;
        }
        const auto& desc = tensor->GetTensorDesc();
        std::string name = desc.tensorName.empty() ? "unnamed" : desc.tensorName;

        std::string golden_path = FindGoldenFile(args.golden_dir, i, name);
        if (golden_path.empty()) {
            fprintf(stderr, "  input[%zu] name=%-24s: no golden file found, skipped\n",
                    i, name.c_str());
            ++skipped;
            continue;
        }

        auto data = ReadFile(golden_path);
        if (data.empty()) {
            fprintf(stderr, "  input[%zu] name=%-24s: golden file empty: %s\n",
                    i, name.c_str(), golden_path.c_str());
            ++skipped;
            continue;
        }

        std::size_t tensor_bytes = tensor->GetSize();
        std::size_t file_bytes = data.size();
        std::size_t copy_bytes = std::min(tensor_bytes, file_bytes);

        if (file_bytes != tensor_bytes) {
            fprintf(stderr, "  input[%zu] name=%-24s: size mismatch (file=%zu, tensor=%zu), "
                    "copying min=%zu\n",
                    i, name.c_str(), file_bytes, tensor_bytes, copy_bytes);
        }

        std::memcpy(tensor->GetData(), data.data(), copy_bytes);
        fprintf(stderr, "  input[%zu] name=%-24s: filled %zu bytes from %s\n",
                i, name.c_str(), copy_bytes, golden_path.c_str());
        ++filled;
    }
    fprintf(stderr, "  summary: %d filled, %d skipped (total %zu inputs)\n",
            filled, skipped, inputs.size());

    if (filled == 0) {
        fprintf(stderr, "golden_verify: no golden inputs loaded, aborting\n");
        return 1;
    }

    // 7. 执行推理（ForwardRaw — 跳过 InputAssembler）
    fprintf(stderr, "=== golden_verify: running inference (ForwardRaw) ===\n");
    st = executor.ForwardRaw();
    if (!st.ok()) {
        fprintf(stderr, "golden_verify: ForwardRaw failed (st=%d)\n", st.code);
        return 1;
    }
    fprintf(stderr, "  inference completed successfully\n");

    // 8. dump 输出 tensor
    fprintf(stderr, "=== golden_verify: dumping outputs to %s ===\n",
            args.output_dir.c_str());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        DumpOutputTensor(outputs[i], i, args.output_dir);
    }

    // 卸载序对齐正式链路(pipeline_executor.cpp:786-787): decode(被验证图)先、
    // prefill 共享源后
    (void)executor.Unload();
    if (share_src) (void)share_src->Unload();

    fprintf(stderr, "=== golden_verify: done ===\n");
    return 0;
}
