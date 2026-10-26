#include "pipeline/pipeline_executor.h"

#include "infra/log.h"

#include <nlohmann/json.hpp> // D-10：仅本 .cpp（编译约束）

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>

namespace cann_lm::engine {

// ── 构造 ──────────────────────────────────────────────────────────────────────
// SR20260903001：meta_json 解析为构造加载流程的内联步骤（D-10——无独立组件）。
// 全部解析知识收敛于本文件匿名命名空间；nlohmann 仅在此 TU。

namespace {

/// 内联解析步骤产出（D-10：pipeline 私有，非跨模块契约）
struct ParseOutcome {
    ModelConfig model;
    TokenizerConfig tokenizer;
    EmbeddingConfig embedding;
    int prefill_chunk_size = 1024; // prefill_len[0] > options.default_chunk_size > 1024
    int decode_batch_size = 4;     // options.max_batch_size_override > 0 ? override : 4
    bool model_parsed = false;     // model 段 Parsed（跨段取值面：pad/vocab/hidden）
    std::vector<std::string> warnings;
    // C-1（2026-09-23，failed_but_not_processed.md §7.3，D-1 温和口径）：
    // meta_json 有供给意图（buffer/path/同名派生三通道任一命中）且关键内容坏
    // ——不可读/读空、JSON 语法非法、model_config 段缺失或解析失败。构造函数
    // 以 `omc_configured && config_fatal` 为门升级构造失败（INVALID_ARGUMENT）；
    // 无意图（stub / omc-only 工件）维持 D-16 原降级口径。
    bool config_fatal = false;
    std::string fatal_reason;
};

/// 整型容错提取：JSON number 或数字字符串（"1" 与 1 等价，REQ-02）；其余 → nullopt
std::optional<long long> AsInt(const nlohmann::json& v)
{
    if (v.is_number_integer())
        return v.get<long long>();
    if (v.is_string()) {
        try {
            return std::stoll(v.get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

/// meta_json 获取：buffer 优先（D-15）→ 显式路径 → omc 同名派生（D-17：按 path 字段，
/// 与加载优先级解耦）。nullopt = Absent。
std::optional<std::vector<char>> ResolveMetaJson(const ModelArtifact& artifact, std::string& how)
{
    auto read_file = [](const std::string& p) -> std::optional<std::vector<char>> {
        std::ifstream f(p, std::ios::binary);
        if (!f)
            return std::nullopt;
        return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    if (artifact.meta_json.data.data != nullptr && artifact.meta_json.data.size > 0) {
        if (!artifact.meta_json.path.empty()) {
            CANN_LLM_LOGW("artifact.dual_state: meta_json buffer 优先，path 忽略（D-15）");
        }
        how = "buffer";
        const auto* p = static_cast<const char*>(artifact.meta_json.data.data);
        return std::vector<char>(p, p + artifact.meta_json.data.size);
    }
    if (!artifact.meta_json.path.empty()) {
        how = "path";
        return read_file(artifact.meta_json.path);
    }
    // 同名派生（prefill 图位优先；仅基于 path 字段——D-17）
    for (const ArtifactSource* omc : {&artifact.prefill_omc, &artifact.decode_omc}) {
        if (omc->path.empty())
            continue;
        std::filesystem::path candidate(omc->path);
        candidate.replace_extension(".json");
        auto bytes = read_file(candidate.string());
        if (bytes) {
            how = "derived:" + candidate.string();
            return bytes;
        }
    }
    return std::nullopt;
}

/// 桩等效默认：KV 形状参数缺省值（与旧硬编码现状逐值一致，TC_104 回归锚点）
constexpr int kStubKvHeads = 4;
constexpr int kStubHeadDim = 8;

/// 桩默认（与变更前现状逐值一致，TC_104 回归锚点）
ModelConfig StubModelConfig()
{
    ModelConfig mc;
    mc.kv_cache_max_len = 16; // 现状 max_seq_len 桩值
    mc.num_hidden_layers = 2; // 现状 KV 池层数硬编码
    mc.num_attention_kv_heads = kStubKvHeads;
    mc.num_attention_head_dims = kStubHeadDim;
    // batch_num=0（未供给，decode_batch_size 链回退）、kv_cache_type 空（消费点回退 FLOAT16）
    return mc;
}

/// model_config 段 → ModelConfig。字段类型不符 / 容量非法 → false（段降级，TC_204）
bool ParseModelSection(const nlohmann::json& sec, ModelConfig* out)
{
    ModelConfig mc;
    auto get_int = [&](const char* key, int* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null())
            return true; // 缺省 = 默认值
        auto v = AsInt(*it);
        if (!v)
            return false;
        *dst = static_cast<int>(*v);
        return true;
    };
    auto get_str = [&](const char* key, std::string* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null())
            return true;
        if (!it->is_string())
            return false;
        *dst = it->get<std::string>();
        return true;
    };
    auto get_arr = [&](const char* key, std::vector<int>* dst) -> bool {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null())
            return true;
        if (!it->is_array())
            return false;
        for (const auto& e : *it) {
            auto n = AsInt(e);
            if (!n)
                return false;
            dst->push_back(static_cast<int>(*n));
        }
        return true;
    };
    if (!get_int("hidden_size", &mc.hidden_size))
        return false;
    if (!get_int("max_io_tokens", &mc.max_io_tokens))
        return false;
    if (!get_str("model_type", &mc.model_type))
        return false;
    if (!get_int("num_hidden_layers", &mc.num_hidden_layers))
        return false;
    if (!get_int("pad_token_id", &mc.pad_token_id))
        return false;
    if (!get_int("kv_cache_max_len", &mc.kv_cache_max_len))
        return false;
    { // vocab 系（size_t 承载）
        auto it = sec.find("vocab_size");
        if (it != sec.end() && !it->is_null()) {
            auto n = AsInt(*it);
            if (!n)
                return false;
            mc.vocab_size = static_cast<size_t>(*n);
        }
        it = sec.find("vocab_align_size");
        if (it != sec.end() && !it->is_null()) {
            auto n = AsInt(*it);
            if (!n)
                return false;
            mc.vocab_align_size = static_cast<size_t>(*n);
        }
    }
    if (!get_arr("prefill_len", &mc.prefill_len))
        return false;
    if (!get_arr("decode_len", &mc.decode_len))
        return false;
    if (!get_str("kv_cache_format", &mc.kv_cache_format))
        return false;
    // KV 形状/类型：kv_heads/head_dims 缺省 = 桩等效默认（旧硬编码现状）；batch_num
    // 缺省 = 0（未供给）；kv_cache_type 缺省 = 空（消费点回退 FLOAT16）
    mc.num_attention_kv_heads = kStubKvHeads;
    mc.num_attention_head_dims = kStubHeadDim;
    if (!get_int("num_attention_kv_heads", &mc.num_attention_kv_heads))
        return false;
    if (!get_int("num_attention_head_dims", &mc.num_attention_head_dims))
        return false;
    if (!get_int("batch_num", &mc.batch_num))
        return false;
    if (!get_str("kv_cache_type", &mc.kv_cache_type))
        return false;
    // Gemma4 滑窗宽（docs/mask_fill.md D7）：缺省 0 = 无滑窗；负值 = 配置非法 →
    // 段降级（与容量非法同口径：显式拒绝优于静默当作无滑窗产出错误模型行为）
    if (!get_int("sliding_window", &mc.sliding_window))
        return false;
    if (mc.sliding_window < 0)
        return false;
    // Qwen3.5 层型映射表（support_qwen3.5.md §4.2）：键缺失/null → 空（全
    // full_attention，零行为变化）；非数组/元素非字符串/长度≠num_hidden_layers/
    // 取值 ∉ {"full_attention","linear_attention","sliding_attention"} → 段降级
    // false（层表残缺不采信；gemma4 滑窗层 sliding_attention 亦进 KV 池、
    // 末维 = sliding_window，KvCachePool 构造消费）
    if (auto it = sec.find("layer_types"); it != sec.end() && !it->is_null()) {
        if (!it->is_array())
            return false;
        std::vector<std::string> types;
        types.reserve(it->size());
        for (const auto& e : *it) {
            if (!e.is_string())
                return false;
            const auto& s = e.get<std::string>();
            if (s != "full_attention" && s != "linear_attention" && s != "sliding_attention")
                return false;
            types.push_back(s);
        }
        if (static_cast<int>(types.size()) != mc.num_hidden_layers)
            return false;
        mc.layer_types = std::move(types);
    }
    // 容量合法性（解析层；非法 → 段降级；加载层另有直调校验，TC_106）
    if (mc.kv_cache_max_len <= 0 || mc.num_hidden_layers <= 0 || mc.num_attention_kv_heads <= 0 ||
        mc.num_attention_head_dims <= 0)
        return false;
    *out = mc;
    return true;
}

/// tokenizer_config 段 → TokenizerConfig（pad 跨段自 model 段快照）。类型不符 → false
bool ParseTokenizerSection(const nlohmann::json& sec, int pad_from_model, TokenizerConfig* out)
{
    TokenizerConfig tk;
    auto it = sec.find("tokenizer_file");
    if (it != sec.end() && !it->is_null()) {
        if (!it->is_string())
            return false;
        tk.tokenizer_file = it->get<std::string>();
    }
    auto get_int = [&](const char* key, int* dst) -> bool {
        auto i = sec.find(key);
        if (i == sec.end() || i->is_null())
            return true;
        auto v = AsInt(*i);
        if (!v)
            return false;
        *dst = static_cast<int>(*v);
        return true;
    };
    auto get_str = [&](const char* key, std::string* dst) -> bool {
        auto i = sec.find(key);
        if (i == sec.end() || i->is_null())
            return true;
        if (!i->is_string())
            return false;
        *dst = i->get<std::string>();
        return true;
    };
    if (!get_int("bos_token_id", &tk.bos_token_id))
        return false;
    if (!get_int("eos_token_id", &tk.eos_token_id))
        return false;
    if (!get_str("bos_token", &tk.bos_token))
        return false;
    if (!get_str("eos_token", &tk.eos_token))
        return false;
    // 注入开关（B2 通道来源）：bool 缺省 = false（零注入）；类型不符 → 段降级
    auto get_bool = [&](const char* key, bool* dst) -> bool {
        auto i = sec.find(key);
        if (i == sec.end() || i->is_null())
            return true;
        if (!i->is_boolean())
            return false;
        *dst = i->get<bool>();
        return true;
    };
    if (!get_bool("add_bos", &tk.add_bos))
        return false;
    if (!get_bool("add_eos", &tk.add_eos))
        return false;
    auto st = sec.find("special_tokens");
    if (st != sec.end() && !st->is_null()) {
        if (!st->is_object())
            return false;
        for (auto el = st->begin(); el != st->end(); ++el) {
            auto v = AsInt(el.value());
            if (!v)
                return false;
            tk.special_tokens[el.key()] = static_cast<int>(*v);
        }
    }
    tk.pad_token_id = pad_from_model;
    *out = tk;
    return true;
}

/// embedding_config 段 → EmbeddingConfig（vocab/hidden 跨段自 model 段；D-12 直参零中间结构）
bool ParseEmbeddingSection(const nlohmann::json& sec, size_t vocab, size_t hidden, EmbeddingConfig* out)
{
    auto get_str = [&](const char* key) -> std::optional<std::string> {
        auto it = sec.find(key);
        if (it == sec.end() || it->is_null())
            return std::string(); // 缺省 = 空串
        if (!it->is_string())
            return std::nullopt; // 类型不符 = 段降级
        return it->get<std::string>();
    };
    const auto weights = get_str("embedding_weights");
    const auto scale = get_str("embedding_dequant_scale");
    const auto type = get_str("embedding_type");
    const auto stype = get_str("embedding_scale_type");
    const auto qmode = get_str("embedding_quant_mode");
    const auto lmode = get_str("embedding_load_mode");
    // per_layer_token_identity 读取类型（gemma4，docs/gemma4_token_identity.md §4；
    // 2026-10-09 合并文件布局——Q12：数据并入主表/scale 文件，独立路径键已删除，
    // 启用由图工位判定。identity scale 类型键仅规范拼写 token_identity_scale_type
    // （Q13 时代契约拼写兜底已于 2026-10-19 移除）；缺键 →
    // FromSection 缺省 fp16，类型不符 → 段降级）
    const auto identity_type = get_str("token_identity_type");
    const auto identity_stype = get_str("token_identity_scale_type");
    if (!weights || !scale || !type || !stype || !qmode || !lmode)
        return false;
    if (!identity_type || !identity_stype)
        return false;
    return EmbeddingConfig::FromSection(*weights, *scale, *type, *stype, *qmode, *lmode, *identity_type,
                                        *identity_stype, vocab, hidden, out)
        .ok();
}

/// 总步骤：获取 → 语法检查 → 三段解析（段间隔离）→ 优先级链。
/// 恒成功（任何失败 = 桩默认 + WARN——D-16 全量口径，含缺段/Absent）。
/// C-1 例外（2026-09-23）：有供给意图且关键内容坏 → config_fatal=true 交构造
/// 门控（omc_configured 时致命）；返回值仍恒成功（解析观测点/Parsed* 不失效）。
ParseOutcome ParseConfigs(const ModelArtifact& artifact, const EngineOptions& options)
{
    ParseOutcome po;
    std::string how = "absent";
    auto bytes = ResolveMetaJson(artifact, how);
    CANN_LLM_LOGI("config.load.meta_json: source=%s", how.c_str());
    // 供给意图 = 三通道任一命中（buffer / 显式 path / 同名派生）；absent 维持原口径
    const bool meta_intent = (how != "absent");

    bool syntax_ok = false;
    nlohmann::json doc;
    if (bytes && !bytes->empty()) {
        try {
            doc = nlohmann::json::parse(*bytes);
            syntax_ok = doc.is_object();
        } catch (const std::exception&) {
            syntax_ok = false;
        }
    }
    if (!bytes || bytes->empty() || !syntax_ok) {
        const char* why = !bytes         ? (meta_intent ? "meta_json 供给路径不可读" :
                                                          "meta_json 不可得（双态空/路径缺失/派生不到）") :
                          bytes->empty() ? "meta_json 为空" :
                                           "JSON 语法非法或顶层非 object";
        po.warnings.push_back(std::string("model_config: ") + why);
        po.warnings.push_back(std::string("tokenizer_config: ") + why);
        po.warnings.push_back(std::string("embedding_config: ") + why);
        if (meta_intent) { // C-1：有意图而不可得/为空/语法非法 ⇒ fatal
            po.config_fatal = true;
            po.fatal_reason = why;
        }
        po.model = StubModelConfig();
        po.tokenizer = TokenizerConfig{};
        po.embedding = EmbeddingConfig{}; // table_path 空 = tokenId 路径
        po.prefill_chunk_size = options.default_chunk_size > 0 ? options.default_chunk_size : 1024;
        po.decode_batch_size = options.max_batch_size_override > 0 ? options.max_batch_size_override : 4;
        return po;
    }

    // model 段先行（跨段快照：pad/vocab/hidden）。
    // C-1（2026-09-23）：能走到此处 ⇒ meta_json 有供给意图且语法合法；
    // 关键段（model_config）缺失/为空/字段非法 ⇒ config_fatal——桩默认值
    // （kv_len=16、layers=2 等假配置）会在 Load/形状修正处批量爆炸，错误归因失真。
    auto mit = doc.find("model_config");
    if (mit != doc.end() && mit->is_object() && !mit->empty()) {
        po.model_parsed = ParseModelSection(*mit, &po.model);
        if (!po.model_parsed) {
            po.warnings.push_back("model_config: 字段类型非法或容量非法（kv_cache_max_len/num_hidden_layers ≤ 0）");
            po.config_fatal = true;
            po.fatal_reason = "model_config: 字段类型非法或容量非法";
        }
    } else {
        po.warnings.push_back("model_config: 段缺失或为空");
        po.config_fatal = true;
        po.fatal_reason = "model_config: 段缺失或为空";
    }
    if (!po.model_parsed)
        po.model = StubModelConfig();

    // tokenizer 段（pad 跨段：model 段 Parsed 才取值，否则 -1——F-02 段间影响）
    auto tit = doc.find("tokenizer_config");
    if (tit != doc.end() && tit->is_object() && !tit->empty()) {
        const int pad = po.model_parsed ? po.model.pad_token_id : -1;
        if (!ParseTokenizerSection(*tit, pad, &po.tokenizer)) {
            po.warnings.push_back("tokenizer_config: 字段类型非法");
            po.tokenizer = TokenizerConfig{};
        }
    } else {
        po.warnings.push_back("tokenizer_config: 段缺失或为空");
    }

    // embedding 段（vocab/hidden 跨段：model 段 Parsed 取其值，否则 0——延迟失败 D-09#2）
    auto eit = doc.find("embedding_config");
    if (eit != doc.end() && eit->is_object() && !eit->empty()) {
        const size_t vocab = po.model_parsed ? po.model.vocab_size : 0;
        const size_t hidden = po.model_parsed ? static_cast<size_t>(po.model.hidden_size) : 0;
        if (!ParseEmbeddingSection(*eit, vocab, hidden, &po.embedding)) {
            po.warnings.push_back("embedding_config: 字段类型非法或枚举未知/配置矛盾");
            po.embedding = EmbeddingConfig{}; // tokenId 路径
        }
    } else {
        po.warnings.push_back("embedding_config: 段缺失或为空");
    }

    // 优先级链（REQ-03）：prefill_len[0] > options.default_chunk_size > 1024
    po.prefill_chunk_size = (po.model_parsed && !po.model.prefill_len.empty()) ?
                                po.model.prefill_len[0] :
                                (options.default_chunk_size > 0 ? options.default_chunk_size : 1024);
    // 优先级链：model 段解析成功的 batch_num > options.max_batch_size_override > 4
    // （batch_num 是 tensor batch 维与池槽位数的同源权威——段降级时不采信）
    po.decode_batch_size = (po.model_parsed && po.model.batch_num > 0) ?
                               po.model.batch_num :
                               (options.max_batch_size_override > 0 ? options.max_batch_size_override : 4);
    return po;
}

/// tokenizer 创建（D-13/D-14：TokenizerConfig 直通零转换；算法经 options.tokenizer_type
/// 选择——ResolveTokenizerAlgorithm，取代原硬编码 BPE）
std::unique_ptr<TokenizerBase> MakeTokenizer(const TokenizerConfig& cfg, Algorithm algo)
{
    std::unique_ptr<TokenizerBase> tok;
    if (!cfg.tokenizer_file.empty()) {
        Status st = TokenizerFactory::create(algo, cfg, &tok);
        if (!st.ok()) {
            CANN_LLM_LOGW("loader.load.tokenizer: vocab load failed (st=%d, file=%s), tokenizer unset", st.code,
                          cfg.tokenizer_file.c_str());
            tok.reset(); // TC_107：置空不阻断
        }
    } else {
        // 词表未供给：直构 not-ready 实例（special 通道仍接通；现状占位行为，TC_104）。
        // 按所选算法直构，保持实例类型与配置一致。
        Status ignored = Status::SUCCESS;
        switch (algo) {
            case Algorithm::Qwen:
                tok = std::make_unique<QwenTokenizer>(cfg, &ignored);
                break;
            case Algorithm::SPM:
                tok = std::make_unique<SPMTokenizer>(cfg, &ignored);
                break;
            case Algorithm::BPE:
                tok = std::make_unique<BPETokenizer>(cfg, &ignored);
                break;
        }
    }
    return tok;
}

/// EngineOptions.tokenizer_type → Algorithm（"BPE"|"Qwen"|"SPM"）。
/// 空 = 默认 BPE（保持既有行为）；未知值 WARN 后回退 BPE（降级不阻断口径）。
Algorithm ResolveTokenizerAlgorithm(const std::string& type)
{
    if (type.empty() || type == "BPE")
        return Algorithm::BPE;
    if (type == "Qwen")
        return Algorithm::Qwen;
    if (type == "SPM")
        return Algorithm::SPM;
    CANN_LLM_LOGW("config.load.tokenizer_type: unknown '%s', fallback BPE", type.c_str());
    return Algorithm::BPE;
}

/// ModelConfig.kv_cache_type（字符串）→ DataType。
/// 词表：小写简写（"fp16"/"fp32"，仓库惯例同 embedding_type）+ DataType 枚举名
/// （"FLOAT16"），匹配不区分大小写。
/// 空 = 默认 FLOAT16（KV cache 常规）；未知值 WARN 后回退 FLOAT16（降级不阻断口径，
/// 同 ResolveTokenizerAlgorithm）。
DataType ResolveKvCacheDtype(const std::string& type)
{
    if (type.empty())
        return DataType::FLOAT16;
    std::string low = type;
    for (char& c : low) { // ASCII 小写化（dtype 名均为 ASCII）
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    static const struct {
        const char* name;
        DataType dt;
    } kDtypeMap[] = {
        {"fp16", DataType::FLOAT16},
        {"float16", DataType::FLOAT16},
        {"fp32", DataType::FLOAT32},
        {"float32", DataType::FLOAT32},
        // hiai::DataType：64 位浮点为 DOUBLE（无 FLOAT64 名）；无 UINT16/UINT64
        // （超出 hiai 词表 → 走下方 WARN + 回退 FLOAT16 的既有降级口径）。
        {"fp64", DataType::DOUBLE},
        {"float64", DataType::DOUBLE},
        {"double", DataType::DOUBLE},
        {"bool", DataType::BOOL},
        {"int8", DataType::INT8},
        {"int16", DataType::INT16},
        {"int32", DataType::INT32},
        {"int64", DataType::INT64},
        {"uint8", DataType::UINT8},
        {"uint32", DataType::UINT32},
    };
    for (const auto& e : kDtypeMap) {
        if (low == e.name)
            return e.dt;
    }
    CANN_LLM_LOGW("config.load.kv_cache_type: unknown '%s', fallback FLOAT16", type.c_str());
    return DataType::FLOAT16;
}

/// 枚举 → 字面量（仅用于调试打印）
const char* AlgorithmName(Algorithm a)
{
    switch (a) {
        case Algorithm::BPE:
            return "BPE";
        case Algorithm::Qwen:
            return "Qwen";
        case Algorithm::SPM:
            return "SPM";
    }
    return "unknown";
}
const char* DataTypeName(DataType t)
{
    switch (t) {
        case DataType::UINT8:
            return "UINT8";
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::FLOAT16:
            return "FLOAT16";
        case DataType::INT32:
            return "INT32";
        case DataType::INT8:
            return "INT8";
        case DataType::INT16:
            return "INT16";
        case DataType::BOOL:
            return "BOOL";
        case DataType::INT64:
            return "INT64";
        case DataType::UINT32:
            return "UINT32";
        case DataType::DOUBLE:
            return "DOUBLE";
        case DataType::RESERVED:
            return "RESERVED";
    }
    return "unknown";
}
const char* EmbeddingTypeName(EmbeddingType t)
{
    switch (t) {
        case EmbeddingType::Fp32:
            return "Fp32";
        case EmbeddingType::Fp16:
            return "Fp16";
        case EmbeddingType::Int8:
            return "Int8";
        case EmbeddingType::Int4:
            return "Int4";
    }
    return "unknown";
}
const char* EmbeddingScaleTypeName(EmbeddingScaleType t)
{
    switch (t) {
        case EmbeddingScaleType::None:
            return "None";
        case EmbeddingScaleType::Fp32:
            return "Fp32";
        case EmbeddingScaleType::Fp16:
            return "Fp16";
    }
    return "unknown";
}
const char* EmbeddingQuantModeName(EmbeddingQuantMode m)
{
    switch (m) {
        case EmbeddingQuantMode::PerToken:
            return "PerToken";
        case EmbeddingQuantMode::PerGroup:
            return "PerGroup";
    }
    return "unknown";
}
const char* EmbeddingLoadModeName(EmbeddingLoadMode m)
{
    switch (m) {
        case EmbeddingLoadMode::InMemory:
            return "InMemory";
        case EmbeddingLoadMode::InFlash:
            return "InFlash";
    }
    return "unknown";
}

/// ── per_layer_token_identity 工位探针（gemma4，§5.2 行宽推导/互通校验共用；只读）──
struct SlotProbe {
    bool found = false; // 工位存在（执行器已 Load 且图带该输入；host stub → false）
    size_t width = 0;   // dims[2..] 尾维连乘（{batch, seq, W_tail...}，§3）
};

/// 自执行器输入工位探取 name 工位（不修改任何状态；任何不就绪 → found=false）。
SlotProbe ProbeInputSlot(const ModelExecutor& executor, const std::string& name)
{
    const auto& idx = executor.InputNameIndex();
    auto it = idx.find(name);
    if (it == idx.end())
        return {};
    const auto& tensors = executor.InputTensors();
    if (it->second >= tensors.size() || !tensors[it->second])
        return {};

    SlotProbe p;
    p.found = true;
    const auto& dims = tensors[it->second]->GetTensorDesc().dims;
    if (dims.size() > 2) {
        size_t w = 1;
        for (size_t i = 2; i < dims.size(); ++i) {
            if (dims[i] <= 0) {
                w = 0;
                break;
            }
            w *= static_cast<size_t>(dims[i]);
        }
        p.width = w;
    }
    return p;
}

/// 完整打印 ParsedConfigs（构造观测点，M-08）
void LogParsedConfigs(const ModelConfig& model, const TokenizerConfig& tokenizer, const EmbeddingConfig& embedding,
                      Algorithm algorithm, DataType kv_cache_dtype, int prefill_chunk_size, int decode_batch_size)
{
    CANN_LLM_LOGI("=== ParsedConfigs ===");
    CANN_LLM_LOGI("  prefill_chunk_size = %d", prefill_chunk_size);
    CANN_LLM_LOGI("  decode_batch_size  = %d", decode_batch_size);
    CANN_LLM_LOGI("  algorithm          = %s", AlgorithmName(algorithm));
    CANN_LLM_LOGI("  kv_cache_dtype     = %s", DataTypeName(kv_cache_dtype));

    // ModelConfig
    CANN_LLM_LOGI("  --- ModelConfig ---");
    auto join = [](const std::vector<int>& v) -> std::string {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i)
                s += ",";
            s += std::to_string(v[i]);
        }
        s += "]";
        return s;
    };
    CANN_LLM_LOGI("    hidden_size             = %d", model.hidden_size);
    CANN_LLM_LOGI("    max_io_tokens           = %d", model.max_io_tokens);
    CANN_LLM_LOGI("    model_type              = %s", model.model_type.c_str());
    CANN_LLM_LOGI("    num_hidden_layers       = %d", model.num_hidden_layers);
    CANN_LLM_LOGI("    pad_token_id            = %d", model.pad_token_id);
    CANN_LLM_LOGI("    vocab_size              = %zu", model.vocab_size);
    CANN_LLM_LOGI("    vocab_align_size        = %zu", model.vocab_align_size);
    CANN_LLM_LOGI("    kv_cache_max_len        = %d", model.kv_cache_max_len);
    CANN_LLM_LOGI("    kv_cache_format         = %s", model.kv_cache_format.c_str());
    CANN_LLM_LOGI("    num_attention_kv_heads  = %d", model.num_attention_kv_heads);
    CANN_LLM_LOGI("    num_attention_head_dims = %d", model.num_attention_head_dims);
    CANN_LLM_LOGI("    batch_num               = %d", model.batch_num);
    CANN_LLM_LOGI("    kv_cache_type           = %s", model.kv_cache_type.c_str());
    CANN_LLM_LOGI("    sliding_window          = %d", model.sliding_window);
    { // layer_types 摘要（support_qwen3.5.md §4.2）：空打印 absent
        size_t full = 0, sliding = 0, linear = 0;
        for (const auto& t : model.layer_types) {
            if (t == "linear_attention") {
                ++linear;
            } else if (t == "sliding_attention") {
                ++sliding;
            } else {
                ++full;
            }
        }
        if (model.layer_types.empty()) {
            CANN_LLM_LOGI("    layer_types             = absent");
        } else {
            CANN_LLM_LOGI("    layer_types             = [full×%zu, sliding×%zu, linear×%zu]", full, sliding, linear);
        }
    }
    CANN_LLM_LOGI("    prefill_len             = %s", join(model.prefill_len).c_str());
    CANN_LLM_LOGI("    decode_len              = %s", join(model.decode_len).c_str());

    // TokenizerConfig
    CANN_LLM_LOGI("  --- TokenizerConfig ---");
    CANN_LLM_LOGI("    tokenizer_file     = %s", tokenizer.tokenizer_file.c_str());
    CANN_LLM_LOGI("    bos_token_id       = %d", tokenizer.bos_token_id);
    CANN_LLM_LOGI("    eos_token_id       = %d", tokenizer.eos_token_id);
    CANN_LLM_LOGI("    pad_token_id       = %d", tokenizer.pad_token_id);
    CANN_LLM_LOGI("    unk_token_id       = %d", tokenizer.unk_token_id);
    CANN_LLM_LOGI("    add_bos            = %d", tokenizer.add_bos ? 1 : 0);
    CANN_LLM_LOGI("    add_eos            = %d", tokenizer.add_eos ? 1 : 0);
    CANN_LLM_LOGI("    bos_token          = %s", tokenizer.bos_token.c_str());
    CANN_LLM_LOGI("    eos_token          = %s", tokenizer.eos_token.c_str());
    CANN_LLM_LOGI("    special_tokens_count = %zu", tokenizer.special_tokens.size());
    if (!tokenizer.special_tokens.empty()) {
        for (const auto& [k, v] : tokenizer.special_tokens) {
            CANN_LLM_LOGI("      special_token: %s -> %d", k.c_str(), v);
        }
    }

    // EmbeddingConfig
    CANN_LLM_LOGI("  --- EmbeddingConfig ---");
    CANN_LLM_LOGI("    table_path   = %s", embedding.table_path.c_str());
    CANN_LLM_LOGI("    scale_path   = %s", embedding.scale_path.c_str());
    CANN_LLM_LOGI("    table_type   = %s", EmbeddingTypeName(embedding.table_type));
    CANN_LLM_LOGI("    scale_type   = %s", EmbeddingScaleTypeName(embedding.scale_type));
    CANN_LLM_LOGI("    quant_mode   = %s", EmbeddingQuantModeName(embedding.quant_mode));
    CANN_LLM_LOGI("    group_size   = %zu", embedding.group_size);
    CANN_LLM_LOGI("    vocab_size   = %zu", embedding.vocab_size);
    CANN_LLM_LOGI("    hidden_dim   = %zu", embedding.hidden_dim);
    CANN_LLM_LOGI("    load_mode    = %s", EmbeddingLoadModeName(embedding.load_mode));
    // per_layer_token_identity（gemma4；合并文件布局 Q12——数据在主表/scale 文件
    // 后段，读取类型两项；启用与否依图工位，InitEmbedding_ 加载期观测）
    CANN_LLM_LOGI("    identity_type        = %s", EmbeddingTypeName(embedding.token_identity_type));
    CANN_LLM_LOGI("    identity_scale_type  = %s", EmbeddingScaleTypeName(embedding.token_identity_scale_type));
    CANN_LLM_LOGI("=== End ParsedConfigs ===");
}

} // namespace

// ── 内联解析步骤产物（私有嵌套结构，.cpp 内定义——D-10）────────────────────────
struct AutoregressivePipelineExecutor::ParsedConfigs {
    ModelConfig model;
    TokenizerConfig tokenizer;
    EmbeddingConfig embedding;
    Algorithm algorithm = Algorithm::BPE;        // options.tokenizer_type 解析（空 → BPE）
    DataType kv_cache_dtype = DataType::FLOAT16; // kv_cache_type 解析终值
    int prefill_chunk_size = 1024;
    int decode_batch_size = 4;
    // C-1：meta_json 有意图且关键内容坏（构造门控经 omc_configured，见 ParseOutcome）
    bool config_fatal = false;
    std::string fatal_reason;
};

AutoregressivePipelineExecutor::AutoregressivePipelineExecutor(const ModelArtifact& artifact,
                                                               const EngineOptions& options)
    : parsed_([artifact, &options]() -> std::unique_ptr<ParsedConfigs> {
          // ① 解析步骤（恒成功；降级 WARN 逐条落日志——D-16 全量口径）
          ParseOutcome po = ParseConfigs(artifact, options);
          for (const auto& w : po.warnings) {
              CANN_LLM_LOGW("config.load.section: %s", w.c_str());
          }
          auto pc = std::make_unique<ParsedConfigs>();
          pc->model = po.model;
          pc->tokenizer = po.tokenizer;
          pc->embedding = po.embedding;
          pc->prefill_chunk_size = po.prefill_chunk_size;
          pc->decode_batch_size = po.decode_batch_size;
          pc->config_fatal = po.config_fatal; // C-1：三段 WARN 照发，门控在构造
          pc->fatal_reason = po.fatal_reason;
          // options 层覆盖（EngineOptions 显式供给优先于 meta_json）：
          // tokenizer_path 非空 → 覆盖 tokenizer_file（词表供给的第二通道，优先级高）；
          // tokenizer_type → 算法选择（空 = BPE 默认，未知 = WARN 回退）。
          if (!options.tokenizer_path.empty()) {
              pc->tokenizer.tokenizer_file = options.tokenizer_path;
          }
          pc->algorithm = ResolveTokenizerAlgorithm(options.tokenizer_type);
          // kv_cache_type 解析终值（空/未知回退 FLOAT16；KvCachePool 构造消费）
          pc->kv_cache_dtype = ResolveKvCacheDtype(po.model.kv_cache_type);
          return pc;
      }()),
      // ② tokenizer 创建（词表路径驱动；失败置空不阻断，TC_107）
      tokenizer_(MakeTokenizer(parsed_->tokenizer, parsed_->algorithm)),
      prefill_executor_(/*pool=*/nullptr, /*asm_=*/nullptr, /*dfx=*/nullptr),
      decode_executor_(/*pool=*/nullptr, /*asm_=*/nullptr, /*dfx=*/nullptr),
      // ⑥ preprocessor/scheduler 接线（解析终值驱动——成员声明序保证 parsed_ 先就绪）
      preprocessor_(tokenizer_.get(), &text_embedding_, parsed_->prefill_chunk_size),
      scheduler_(&prefill_executor_, &decode_executor_, &preprocessor_, &text_embedding_, parsed_->decode_batch_size)
{
    // 调试观测点：完整打印解析后的 ParsedConfigs
    LogParsedConfigs(parsed_->model, parsed_->tokenizer, parsed_->embedding, parsed_->algorithm,
                     parsed_->kv_cache_dtype, parsed_->prefill_chunk_size, parsed_->decode_batch_size);

    // 模型图位配置态（omc path/buffer 任一非空 = 部署形态；双空 = stub 形态）。
    // 用于 C-1 门控（下）与 ④ 严格加载 / C-2 就绪断言（后）。声明一处，构造体内复用。
    const bool omc_configured = (artifact.prefill_omc.data.data != nullptr && artifact.prefill_omc.data.size > 0) ||
                                !artifact.prefill_omc.path.empty() ||
                                (artifact.decode_omc.data.data != nullptr && artifact.decode_omc.data.size > 0) ||
                                !artifact.decode_omc.path.empty();

    // C-1 修复（2026-09-23，failed_but_not_processed.md §7.3；D-1 温和口径）：
    // ParseConfigs 不再无差别「恒成功」（D-16）。meta_json 有供给意图
    // （buffer/path/同名派生）且关键内容坏——不可读/读空、语法非法、
    // model_config 段缺失或解析失败 ⇒ 部署形态下升级构造失败：桩默认值
    // （kv_len=16、layers=2 等假配置）会拿到下游批量爆炸，错误归因失真。
    // 无意图（stub 形态 / omc-only 工件）维持 D-16 原口径——TC_101/TC_204/
    // MissingModelArtifact 等既有用例行为不变。tokenizer/embedding 段维持降级：
    // 分别由 C-2 断言与 InitEmbedding_（B-10）收口。
    if (omc_configured && parsed_->config_fatal) {
        CANN_LLM_LOGE("pipeline.init: model meta_json invalid (%s), init aborted", parsed_->fatal_reason.c_str());
        init_status_ = Status::INVALID_ARGUMENT;
        return; // KV 未建、执行器未 Load、worker 未启动：无资源需要释放
    }

    // ③ KV 池对象定容（T1：tensor 显存创建后移——Cache_pool_simplify.md）：
    //    构造期仅层表/槽位记账 + HF 期望值预计算（R3 比对值，meta 权威量：
    //    层表 layer_types / 槽数 batch_num 链 / 光标容量 kv_cache_max_len /
    //    期望 heads/head_dims/sliding_window），不申请 tensor；tensor 由首个
    //    执行器 Load 按 om 图工位 desc 惰性创建（R1，创建即清零），双图一致性
    //    由 EnsureKvTensor R2 把关。kv_cache_type 降级纯观测（LogParsedConfigs
    //    照打，不再构造消费——om dtype 为唯一权威）。
    kv_pool_ = std::make_shared<KvCachePool>(parsed_->decode_batch_size, parsed_->model.kv_cache_max_len,
                                             parsed_->model.num_hidden_layers, parsed_->model.num_attention_kv_heads,
                                             parsed_->model.num_attention_head_dims, parsed_->model.layer_types,
                                             parsed_->model.sliding_window);
    assembler_ = std::make_shared<InputAssembler>(kv_pool_, parsed_->decode_batch_size, parsed_->model.sliding_window);
    (void)prefill_executor_.AttachPool(kv_pool_);
    (void)prefill_executor_.AttachAssembler(assembler_);
    (void)decode_executor_.AttachPool(kv_pool_);
    (void)decode_executor_.AttachAssembler(assembler_);
    // ③b Qwen3.5 线性状态池（support_qwen3.5.md §4.6）：layer_types 声明
    //    linear_attention 层才创建（无 linear 层 / 空 = 不创建，零行为变化）；
    //    构造期只有层清单无 tensor——状态形状以 om 图工位 desc 为权威
    //    （meta_json 无状态形状字段，§5 D1），由先 Load 的 prefill 图惰性创建
    //    （创建即整体清零）、后 Load 的 decode 图校验 dims/dtype 一致后复用。
    //    双执行器共享同一实例（同 KV 池口径；输入/输出工位绑同一 tensor，
    //    图推理期原地更新，无回绕无光标）。
    const auto linear_layers = static_cast<int>(
        std::count(parsed_->model.layer_types.begin(), parsed_->model.layer_types.end(), "linear_attention"));
    if (linear_layers > 0) {
        state_pool_ = std::make_shared<LinearStatePool>(parsed_->model.layer_types);
        (void)prefill_executor_.AttachStatePool(state_pool_);
        (void)decode_executor_.AttachStatePool(state_pool_);
    }

    // ④ 双执行器各 Load 各自图位（D-03 澄清版；先 prefill 后 decode——2026-09-23 换序：
    //    prefill 图为形状权威（与 token_identity「prefill 权威」约定统一；Qwen3.5 状态池
    //    设计同口径，由先 Load 的 prefill 图惰性创建）；KV 池 tensor 同由先 Load 的
    //    prefill 图经 EnsureKvTensor 惰性创建（T1）、后 Load 的 decode 图复验一致后复用
    //    （R2），加载顺序无功能依赖——池对象构造期前置（AttachPool 先行），tensor 无需
    //    前置）
    //    seq_len 参数：decode=1（固定），prefill=prefill_chunk_size（驱动 tensorDesc 修正）
    //    embedding 跨段字段：embed_scales 维度修正需要 quant_mode/group_size
    parsed_->model.embedding_per_group = (parsed_->embedding.quant_mode == EmbeddingQuantMode::PerGroup);
    parsed_->model.embedding_group_size = static_cast<int>(parsed_->embedding.group_size);

    // 加载失败即终止（2026-09-22，取代原 (void) 丢弃——加载失败曾只记日志，
    // 未中断初始化，半初始化引擎照常出列且到 Step 期才报 not-ready）。
    // 口径：模型图位已配置（omc path/buffer 任一非空）⇒ 严格加载，任一执行器
    // Load 失败（含权重 InitWeights，如设备端 NPU 配额耗尽）⇒ 终止初始化：
    // ①跳过 ⑤⑥（embedding 不加载、worker 不启动）；②显式卸载双执行器
    // （率先成功加载到位的 prefill 图+权重即刻释放回配额；失败执行器 Load 内
    // 已自释放 mm_，Unload 清残留工位引用）；③init_status_ 记错误码，工厂
    // 据此不交出本实例并上抛（ExecutorCreate 报错）。
    // 双图位全空 = 无模型部署形态（stub-era / 无设备 UT）：跳过加载维持原
    // 行为——生命周期 API 可用，Generate 报 executor not-ready。
    if (omc_configured) {
        Status exec_st = prefill_executor_.Load(artifact.prefill_omc, artifact.prefill_weight, parsed_->model,
                                                parsed_->prefill_chunk_size);
        if (exec_st.ok()) {
            // 权重共享绑定：prefill 加载完成后（mm Init 后 GetModelID 可用）将其
            // modelId 传给 decode 执行器——decode Load 内在 mm Init 之前填入
            // ModelInitOptions::shareWeightsModelId。绑定与 prefill/decode 权重
            // 路径是否相同无关（docs/seperate_pd_weights.md §4.2）：decode 按自身
            // decode_weight InitWeights，与 prefill 重叠的共享权值由底层同源
            // 去重复用，差异权值独立加载（复用判定由底层自理，上层无需感知）。
            // id==0（老 ROM 取不到 GetModelID）：decode 不绑定（默认 0），行为不变。
            const uint32_t prefill_model_id = prefill_executor_.ModelId();
            if (prefill_model_id == 0) {
                CANN_LLM_LOGW("pipeline.init: prefill ModelId()==0, decode skips share-weights binding");
            }
            exec_st = decode_executor_.Load(artifact.decode_omc, artifact.decode_weight, parsed_->model, 1,
                                            /*share_weights_model_id=*/prefill_model_id);
        }
        if (!exec_st.ok()) {
            CANN_LLM_LOGE("pipeline.init: executor load failed (st=%d), init aborted and executors unloaded",
                          exec_st.code);
            (void)decode_executor_.Unload();
            (void)prefill_executor_.Unload();
            init_status_ = exec_st;
            return; // 不加载 embedding、不启动 worker；构造正常完成（失败态经 InitStatus 上抛）
        }
    }

    // C-2 修复（2026-09-23，failed_but_not_processed.md §7.4）：部署形态就绪性断言。
    // MakeTokenizer 失败曾只置空不阻断（TC_107 口径）⇒「构造成功」的对象每次
    // Generate 必失败——preprocessor.cpp:23-26 对 prompt.text 无条件 encode，
    // 不存在免 tokenizer 的部署形态，失败被推迟到首个请求才暴露。部署形态下
    // 构造成功必须蕴含 tokenizer 可用（TokenizerReady() = 非 nullptr 且
    // vocab_size>0，:755-757），与 ④ 的「构造成功 ⇒ 执行器已加载」对齐。
    // 断言位于 Load 之后（D-4 决策：MissingModelArtifact 类工件在 Load 提前
    // 终止，触及不到本断言）；stub 形态（双图位全空）不断言，TC_107 语义保留。
    if (omc_configured && !TokenizerReady()) {
        CANN_LLM_LOGE("pipeline.init: deployed form requires a usable tokenizer "
                      "(vocab_size > 0), init aborted and executors unloaded");
        (void)decode_executor_.Unload();
        (void)prefill_executor_.Unload();
        init_status_ = Status::INVALID_STATE;
        return; // 不加载 embedding、不启动 worker；失败态经 InitStatus 上抛
    }

    // ⑤ embedding 表路径非空才 Load（主表失败降级 tokenId，不阻断 init——D-11；
    //    identity 工位就绪性致命化 = B-10 修复，见 InitEmbedding_ 定义处注释）
    Status embed_st = InitEmbedding_(parsed_->embedding);
    if (!embed_st.ok()) {
        CANN_LLM_LOGE("pipeline.init: embedding identity readiness check failed (st=%d), "
                      "init aborted and executors unloaded",
                      embed_st.code);
        (void)decode_executor_.Unload();
        (void)prefill_executor_.Unload();
        init_status_ = embed_st;
        return; // 不启动 worker；失败态经 InitStatus 上抛
    }
    // ⑥ sampler vocab_size 注入（parsed_->model 含 meta_json 解析的 vocab_size；
    //    host 无模型时 vocab_size=0 → Sample 返回 INVALID_STATE，pipeline 上游短路走不到）
    sampler_.Configure(parsed_->model);

    worker_ = std::thread(&AutoregressivePipelineExecutor::RunLoop, this);
}

AutoregressivePipelineExecutor::~AutoregressivePipelineExecutor()
{
    stop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(worker_mtx_);
    }
    worker_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

// ── 配置解析观测点（SR20260903001 M-08 §8：UT 载体，只读）─────────────────────

const ModelConfig& AutoregressivePipelineExecutor::ParsedModelConfig() const { return parsed_->model; }
const TokenizerConfig& AutoregressivePipelineExecutor::ParsedTokenizerConfig() const { return parsed_->tokenizer; }
const EmbeddingConfig& AutoregressivePipelineExecutor::ParsedEmbeddingConfig() const { return parsed_->embedding; }
int AutoregressivePipelineExecutor::ParsedPrefillChunkSize() const { return parsed_->prefill_chunk_size; }
int AutoregressivePipelineExecutor::ParsedDecodeBatchSize() const { return parsed_->decode_batch_size; }
DataType AutoregressivePipelineExecutor::ParsedKvCacheDtype() const { return parsed_->kv_cache_dtype; }
const KvCachePool* AutoregressivePipelineExecutor::ParsedKvPool() const { return kv_pool_.get(); }
const LinearStatePool* AutoregressivePipelineExecutor::ParsedStatePool() const { return state_pool_.get(); }
bool AutoregressivePipelineExecutor::TokenizerReady() const { return tokenizer_ && tokenizer_->vocab_size() > 0; }

Status AutoregressivePipelineExecutor::InitEmbedding_(const EmbeddingConfig& config)
{
    // SR20260903001：embedding_config 段直供（U-8 通道取代，D-12）。须在 worker 线程
    // 启动前完成（RunLoop → Scheduler::Step → FillInputs 依赖已就绪的加载态）。
    if (!config.table_path.empty()) {
        Status st = text_embedding_.Load(config); // 失败保持未加载态，不阻断 init（D-11）
        if (!st.ok()) {
            CANN_LLM_LOGW("loader.load.embedding: table load failed (st=%d, path=%s), "
                          "degraded to tokenId path",
                          st.code, config.table_path.c_str());
        }
    }

    // ── per_layer_token_identity（gemma4，docs/gemma4_token_identity.md §5.2；
    //     2026-09-22 变更：TokenId 部署形态不查表；2026-10-09 合并文件布局 Q12）──
    // identity host 查表只服务 Embedding 部署形态（主表 Loaded ⇒ Scheduler 选
    // Embedding 路径，scheduler.cpp:143/257）。TokenId 形态 identity 由图内
    // 自理：host 不加载、不落位、零行为变化；主表 Load 失败降级 tokenId 时同
    // 连带跳过（同一文件已不可信，段完整性无从校验）。
    //
    // 启用判定以图为准（B-10 裁定；Q12 后唯一信号——独立路径键已删除）：
    // prefill 图带 per_layer_token_identity 工位 ⇒ 从 embedding_weights 文件
    // 主表段之后偏移加载 identity 段（scale 段同理并入 scale 文件）。图未声明
    // 工位的情形不致命（无填充需求）；图声明了工位而供给/供给段缺失 ⇒ 构造致命。
    //   - prefill 未命中 identity 工位 ⇒ 跳过整流（「executor 未加载」与「图缺
    //     工位」不可区分，stub/host 靠此放行；prefill 权威设计不变）。
    //   - scale 工位：两图工位一致（同源导出应当一致）；prefill 缺而 decode 带 ⇒
    //     构造致命（decode 工位将无来源）；两图齐缺 ⇒ 静默不读（正常形态，
    //     非 gemma4 identity 量化部署）；prefill 带而 decode 缺 ⇒ 一次性 WARN
    //     （prefill 可正常填充，decode 无填充需求，无害；诊断提示图不对齐）。
    if (!text_embedding_.Loaded())
        return Status::SUCCESS;

    // prefill 权威：工位命中 = 启用；行宽自 dims 尾维连乘推导。
    // 缺失/未就绪 → 跳过（host stub 走此路径，构造期自然跳过，§5.2 步骤 2）
    const SlotProbe pre = ProbeInputSlot(prefill_executor_, kTokenIdentitySlotName);
    if (!pre.found) {
        CANN_LLM_LOGW("loader.load.token_identity: per_layer_token_identity slot not available on "
                      "prefill graph (executor not loaded / graph without slot), skip identity load");
        return Status::SUCCESS;
    }
    const size_t row_width = pre.width;

    // decode 交叉校验（Q9：decode 图同带工位）：两图同源导出应当一致；不一致 ⇒
    // 必有一图会被填入错宽数据 ⇒ 构造致命
    const SlotProbe dec = ProbeInputSlot(decode_executor_, kTokenIdentitySlotName);
    if (dec.found && dec.width != pre.width) {
        CANN_LLM_LOGE("loader.load.token_identity: decode row width %zu != prefill %zu "
                      "(same-source graphs must agree) — init aborted",
                      dec.width, pre.width);
        return Status::INVALID_STATE;
    }

    // scale 工位（以图定读取与否；scale 段已并入 embedding_dequant_scale 文件）。
    // 工位名 per_layer_token_identity_scale（2026-10-19 拼写契约迁移——c42 新
    // 转换器实名正拼；老转换器 …sacle 拼写工位不再探测，其图须重导出）
    const bool scale_slot_pre = ProbeInputSlot(prefill_executor_, kTokenIdentityScaleSlotName).found;
    const bool scale_slot_dec = ProbeInputSlot(decode_executor_, kTokenIdentityScaleSlotName).found;
    if (!scale_slot_pre && scale_slot_dec) {
        CANN_LLM_LOGE("loader.load.token_identity: decode graph has per_layer_token_identity_scale "
                      "slot but prefill does not (same-source graphs must agree) — decode scale "
                      "slot would have no source, init aborted");
        return Status::INVALID_STATE;
    }
    if (scale_slot_pre && !scale_slot_dec) {
        CANN_LLM_LOGW("loader.load.token_identity: prefill graph has per_layer_token_identity_scale "
                      "slot but decode does not — graphs not aligned (prefill fills, decode needs "
                      "none; harmless but worth checking graph export)");
    }
    // prefill 带 scale 工位而主表无 scale 文件（scale_type=None）⇒ scale 段无
    // 承载文件 ⇒ 构造致命（工位无法正确填充）
    if (scale_slot_pre && config.scale_type == EmbeddingScaleType::None) {
        CANN_LLM_LOGE("loader.load.token_identity: graph has per_layer_token_identity_scale slot "
                      "but embedding scale file is absent (embedding_scale_type=none) — scale "
                      "slot cannot be filled, init aborted");
        return Status::INVALID_STATE;
    }

    // 依图回填 scale 读取意图（LoadTokenIdentity 单信号：scale_type ≠ None）：
    // 图缺 scale 工位 ⇒ 不读 scale 段（静默，第 4 象限——正常形态）
    EmbeddingConfig icfg = config;
    if (!scale_slot_pre) {
        icfg.token_identity_scale_type = EmbeddingScaleType::None;
    }

    // 加载 identity 段（「填充时失败」⇒ 构造致命，B-10；主表文件内偏移读取）
    Status ist = text_embedding_.LoadTokenIdentity(icfg, row_width);
    if (!ist.ok()) {
        CANN_LLM_LOGE("loader.load.token_identity: identity load failed (st=%d, path=%s) — "
                      "identity slots cannot be filled, init aborted",
                      ist.code, config.table_path.c_str());
        return ist;
    }
    return Status::SUCCESS;
}

// ── Context 管理 ──────────────────────────────────────────────────────────────

Status AutoregressivePipelineExecutor::CreateContext(Context* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;

    ContextId id;
    id.value = next_context_id_.fetch_add(1, std::memory_order_relaxed);

    auto ctx = Context::CreateWithId(id);

    {
        std::lock_guard<std::mutex> lk(context_mtx_);
        contexts_[id] = ctx; // copy（Context 可复制，共享内部状态）
        // PromptCache R2/R7：每 Context 一套 ContextCache——池按模板池同参复制
        // 构造（tensor 不复制，T1 惰性：本 Context 首请求 prefill 期 EnsureKvTensor
        // 直建/复验，与模板池 Load 期同口径）；台账全新。R3 门控构造期一次写入：
        // linear_attention 模型禁用 feature（REQ-10）。
        ContextCache cc;
        cc.pool = std::make_shared<KvCachePool>(parsed_->decode_batch_size, parsed_->model.kv_cache_max_len,
                                                parsed_->model.num_hidden_layers, parsed_->model.num_attention_kv_heads,
                                                parsed_->model.num_attention_head_dims, parsed_->model.layer_types,
                                                parsed_->model.sliding_window);
        cc.history = std::make_shared<HistoryPromptManager>();
        cc.history->SetEnabled(!(state_pool_ && state_pool_->NumLinearLayers() > 0));
        context_caches_[id] = std::move(cc);
    }

    *out = ctx;
    return Status::SUCCESS;
}

Status AutoregressivePipelineExecutor::DestroyContext(ContextId cid)
{
    // 先取消该 Context 所有在飞请求
    (void)CancelContext(cid);

    {
        std::lock_guard<std::mutex> lk(context_mtx_);
        contexts_.erase(cid);
        context_caches_.erase(cid); // 池/台账 shared_ptr 随条目释放（REQ-7）
        active_generates_.erase(cid);
        // 活跃池归属同步清除（worker 下个请求到来时经 RebindIfNeeded_ 必然换绑，
        // 无需即时改绑）
        if (active_cache_ctx_ && *active_cache_ctx_ == cid) {
            active_cache_ctx_.reset();
        }
    }
    return Status::SUCCESS; // 幂等
}

Status AutoregressivePipelineExecutor::ConfigureContext(ContextId cid, const ContextOptions&)
{
    std::lock_guard<std::mutex> lk(context_mtx_);
    if (contexts_.find(cid) == contexts_.end())
        return Status::NOT_FOUND;
    // ContextOptions 为空占位，当前无配置项
    return Status::SUCCESS;
}

Status AutoregressivePipelineExecutor::CancelContext(ContextId cid)
{
    // 防死锁（通道②接线后必守）：Scheduler 各终态点会调用 req->rollback_cache
    // 闭包 → ResolveCache_ 需拿 context_mtx_。若本函数持 context_mtx_ 同时调
    // scheduler_.CancelContext（内部持 scheduler mtx_），则与 worker 路径
    // （scheduler Step 先持 mtx_ 终态化 → 释放后 pipeline 走 context_mtx_）
    // 构成 ABBA。故先出锁判存在性，再无锁委托 Scheduler。
    {
        std::lock_guard<std::mutex> lk(context_mtx_);
        if (contexts_.find(cid) == contexts_.end())
            return Status::NOT_FOUND;
    }
    (void)scheduler_.CancelContext(cid);
    return Status::SUCCESS;
}

// ── 推理 ──────────────────────────────────────────────────────────────────────

Status AutoregressivePipelineExecutor::Generate(const Prompt& prompt, const Context& ctx, const SamplingParams& sp,
                                                const RequestOptions& ro, Result* out)
{
    RequestHandle handle;
    Status st = GenerateAsync(prompt, ctx, sp, ro, &handle, {});
    if (!st.ok())
        return st;
    return handle.Wait(out);
}

Status AutoregressivePipelineExecutor::GenerateAsync(const Prompt& prompt, const Context& ctx, const SamplingParams& sp,
                                                     const RequestOptions& ro, RequestHandle* out,
                                                     StreamCallback callback)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    out->~RequestHandle();
    *out = RequestHandle{};

    ContextId cid = ctx.id();

    // 检查 Context 存在 + 并发检查
    {
        std::lock_guard<std::mutex> lk(context_mtx_);
        if (contexts_.find(cid) == contexts_.end()) {
            CANN_LLM_LOGW("pipeline.submit: context not found (id=%llu)", static_cast<unsigned long long>(cid.value));
            return Status::NOT_FOUND;
        }
        if (active_generates_.count(cid) > 0) {
            CANN_LLM_LOGW("pipeline.submit: concurrent generate on same context (id=%llu)",
                          static_cast<unsigned long long>(cid.value));
            return Status::INVALID_STATE;
        }
        active_generates_.insert(cid);
    }

    // RAII 清理：异常/失败路径清除 active 标记
    auto clear_active = [&]() {
        std::lock_guard<std::mutex> lk(context_mtx_);
        active_generates_.erase(cid);
    };

    // 预处理：tokenize + 比对 + 增量切块（PromptCache 数据流 design §2.2；
    // 多轮上下文组织归客户端，引擎不拼历史，remove_multi_turn.md R1）。
    // 快照原则（R10）：回退目标 = 本轮提交前状态——快照取 Match **前**的
    // 台账与池记账；终态 Failed/Cancel 回退 = Restore 快照 + RollbackTo。
    std::vector<TokenId> prompt_ids;
    HistoryPromptManager::MatchStats mstats;
    std::size_t reusable = 0;
    std::vector<TokenId> history_snapshot;
    std::size_t pool_snapshot = 0;
    {
        // 死锁修复（2026-10-16）：本块失败路径原先直接调 clear_active()
        // （其内自取 context_mtx_）→ 同线程对非递归 mutex 二次上锁 =
        // 自死锁（stub-era encode 失败路径实测挂死）。改法：锁内只记
        // 错误码，出锁后统一 clear_active + return——锁内不再打二段锁。
        Status fail_code = Status::SUCCESS;
        std::unique_lock<std::mutex> lk(context_mtx_);
        auto it = context_caches_.find(cid);
        ContextCache* cc = nullptr;
        std::size_t kv_max = 0;
        if (it == context_caches_.end()) {
            // 防御：并发 DestroyContext 窗口（active_generates_ 检查持同锁，
            // 正常序列不可达）
            fail_code = Status::NOT_FOUND;
        } else if (!it->second.pool || !it->second.history) {
            // 防御：ContextCache 不完整（不可达，构造期成对建）
            fail_code = Status::INVALID_STATE;
        } else {
            cc = &it->second;
            kv_max = static_cast<std::size_t>(cc->pool->kv_cache_max_len());
        }
        if (cc) {
            // REQ-9 容量前置校验（Match 之前、台账未动时拒绝——"不改任何
            // cache 状态"）：单轮全量输入必须能一次装入 KV 缓冲，绕回语义
            // 不支持单轮输入自身超缓冲。
            // 先 encode（比对需要 ids；切片与台账追加同帧复用，不再二次 encode）
            std::vector<std::vector<TokenId>> dummy;
            Status est = preprocessor_.PreparePrefill(prompt, &dummy,
                                                      /*reusable=*/0, &prompt_ids);
            if (!est.ok()) {
                fail_code = est; // stub-era encode 失败即此路径
            } else if (prompt_ids.size() > kv_max && kv_max > 0) {
                CANN_LLM_LOGE("promptcache.capacity: input %zu > kv_max %zu (ctx=%llu) — "
                              "rejected, cache untouched",
                              prompt_ids.size(), kv_max, static_cast<unsigned long long>(cid.value));
                fail_code = Status::INVALID_ARGUMENT;
            }
        }
        if (fail_code.code != Status::SUCCESS) {
            lk.unlock(); // 先出锁再清理（clear_active 内自取锁）
            clear_active();
            return fail_code;
        }

        // R10 提交前快照（Match 变账前）
        history_snapshot = cc->history->SnapshotTokens();
        pool_snapshot = cc->pool->TotalWritten(0);

        // 台账比对（design §3.2 规则表；R3 禁用 / 首轮恒 0）
        reusable = cc->history->MatchCachedTokens(prompt_ids, kv_max, &mstats);

        // 池记账落定（REQ-4）：命中 ⇒ lane 续原身份回退到复用边界（REQ-1
        // 验收：prefill 首 token 绝对位 == LCP ⇒ 池 total_written == reusable）；
        // miss 非首轮 ⇒ ResetLedgerOnSlot 重建（R13，lane 归零语义交由
        // scheduler AllocKvSlot 既有路径）；miss 首轮/R3 ⇒ 池侧不动。
        if (reusable > 0) {
            (void)cc->pool->ResumeSlot(0, reusable);
        } else if (reusable == 0 && mstats.history_len > 0) {
            cc->pool->ResetLedgerOnSlot(0);
        }
    }

    // REQ-11 比对埋点（单点单条；hit = reusable > 0；input=0 = 空 prompt）
    CANN_LLM_LOGI("promptcache.match: ctx=%llu hit=%d reusable=%zu history=%zu "
                  "input=%zu lcp=%zu wrapped=%d",
                  static_cast<unsigned long long>(cid.value), reusable > 0 ? 1 : 0, reusable, mstats.history_len,
                  prompt_ids.size(), mstats.lcp, mstats.wrapped ? 1 : 0);

    // 增量切块（R8/REQ-6）：仅作用 [reusable, len)；二次调用传 ids 直切
    // （tokenizer 不复发——此处 ids 为 encode 全量产物切片，纯 host 操作）
    std::vector<std::vector<TokenId>> chunks;
    Status st = preprocessor_.SliceChunks(prompt_ids, reusable, &chunks);
    if (!st.ok()) {
        clear_active();
        return st;
    }

    // 创建 RequestImpl（采样旋钮 + 请求级终止参数分层落位：
    // sp 纯采样；ro 的 max_tokens / stop 集归请求终止语义，与 tokenizer eos 无关）
    auto req = std::make_shared<RequestImpl>();
    req->sampling_params = sp;
    req->context_id = cid;
    req->callback = callback;
    req->max_tokens = ro.max_tokens;
    req->stop_token_ids = ro.stop_token_ids;
    req->stop_strings = ro.stop_strings;
    // PromptCache 请求级记账（design §4.4）：快照 + 全量 ids（REQ-2.1
    // prefill 完成点追加台账用）+ 复用长
    req->history_snapshot = std::move(history_snapshot);
    req->pool_total_written_snapshot = pool_snapshot;
    req->prompt_ids = prompt_ids;
    req->reusable_len = reusable;
    // R10 通道②：Scheduler 内部终态点（Cancel 驱逐 / ReclaimAndFill_ 清理 /
    // not-ready Fail）pipeline 不可见 req，回退闭包随 req 携带。方法：捕 raw
    // `this` + req 的 raw 指针（pipeline 与 RequestImpl 互为作用于对方寿命内——
    // pipeline 由用户持有、寿命 ≥ 一切 Handle=req shared_ptr 与 Scheduler 池
    // 引用，故 req 在调用时恒存活；不捕 shared_ptr 防自环）。幂等：回退 =
    // 恢复提交前快照，多次调用仅首次实际生效（后续 Restore 同快照 +
    // RollbackTo 被守卫拒）。Completed 场景同样注入——回退闭包只在
    // Failed/Canceled 终态被调起（RollbackCache_ 调用点皆非 Completed），
    // Completed 的成功链走 RunLoop 不经本闭包。
    RequestImpl* req_raw = req.get();
    req->rollback_cache = [this, cid, req_raw, snap_hist = req->history_snapshot,
                           snap_total = req->pool_total_written_snapshot]() {
        ContextCache cc = ResolveCache_(cid);
        if (!cc.history || !cc.pool)
            return; // Context 已销毁——随池释放，无事可回
        cc.history->Restore(snap_hist);
        if (snap_total < cc.pool->TotalWritten(0)) {
            (void)cc.pool->RollbackTo(0, snap_total);
        }
        // terminal 取 req 实时终态（调用点已 SetTerminal 完毕，必为
        // Failed/Canceled 二者之一）
        CANN_LLM_LOGI("promptcache.rollback: ctx=%llu terminal=%d history=%zu pool_total=%zu",
                      static_cast<unsigned long long>(cid.value),
                      static_cast<int>(req_raw->status.load(std::memory_order_acquire)), cc.history->Size(),
                      static_cast<std::size_t>(cc.pool->TotalWritten(0)));
    };

    // 提交到 Scheduler（Submit 仅在入参为空时失败，null 卫哨不记日志）
    RequestId req_id;
    st = scheduler_.Submit(std::move(chunks), req, &req_id);
    if (!st.ok()) {
        clear_active();
        return st;
    }

    // 唤醒 worker
    {
        std::lock_guard<std::mutex> lk(worker_mtx_);
    }
    worker_cv_.notify_one();

    *out = RequestHandle(req);
    return Status::SUCCESS;
}

// ── Worker RunLoop ────────────────────────────────────────────────────────────

void AutoregressivePipelineExecutor::RunLoop()
{
    while (!stop_.load(std::memory_order_acquire)) {
        StepOutput out;
        Status st = scheduler_.Step(&out);

        if (st.code == Status::INVALID_STATE) {
            // 无活任务 → 休眠等待唤醒
            std::unique_lock<std::mutex> lk(worker_mtx_);
            worker_cv_.wait_for(lk, std::chrono::milliseconds(100),
                                [&] { return stop_.load(std::memory_order_acquire); });
            continue;
        }

        if (!st.ok()) {
            // FAILURE：Step 已将受影响请求标记为 Failed 并清理（含 Forward 失败
            // 路径）。failure 分支不填 out.slots（Scheduler 内部已终态化）；
            // R10 台账回退由 Terminal 处理点统一承担——本 loop 尾的
            // RollbackHistoryIfNeeded_ 扫描下方.Header 注记。
            continue;
        }

        // PromptCache worker 侧接线（design §2.2 第 5-7 步）
        if (!out.slots.empty() && out.slots.front()) {
            RebindIfNeeded_(out.slots.front()->context_id);
        }

        // 逐 slot 后处理（M-07）
        for (int slot_idx = 0; slot_idx < static_cast<int>(out.slots.size()); ++slot_idx) {
            auto& req = out.slots[slot_idx];
            if (!req)
                continue;

            // 检查 cancel
            if (req->canceled.load(std::memory_order_acquire)) {
                req->SetTerminal(RequestStatus::Canceled, FinishReason::Canceled);
                RollbackHistoryIfNeeded_(*req, RequestStatus::Canceled);
                ClearActiveGenerate_(req->context_id);
                continue;
            }

            // 终态防御扫描（Step 内部失败终态化的请求会带着 Failed 出现在
            // out.slots——assembler/sampler 前的统一 R10 回退点）
            if (req->status.load(std::memory_order_acquire) == RequestStatus::Failed) {
                RollbackHistoryIfNeeded_(*req, RequestStatus::Failed);
                ClearActiveGenerate_(req->context_id);
                continue;
            }

            // 采样
            TokenId token = 0;
            if (!out.logits) {
                CANN_LLM_LOGE("postprocess.sample: logits missing after forward");
                req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                RollbackHistoryIfNeeded_(*req, RequestStatus::Failed);
                ClearActiveGenerate_(req->context_id);
                continue;
            }

            st = sampler_.Sample(*out.logits, slot_idx, req->sampling_params, req->token_ids, &token);
            if (!st.ok()) {
                CANN_LLM_LOGE("postprocess.sample: sample failed (st=%d)", st.code);
                req->SetTerminal(RequestStatus::Failed, FinishReason::Error);
                RollbackHistoryIfNeeded_(*req, RequestStatus::Failed);
                ClearActiveGenerate_(req->context_id);
                continue;
            }

            // 打印采样结果（step 为本步已生成 token 数，即新 token 的 0-based 位置）
            CANN_LLM_LOGD("debug.sampled: step=%zu slot=%d token=%d", req->token_ids.size(), slot_idx,
                          static_cast<int>(token));

            // 追加 token
            req->token_ids.push_back(token);

            // ── PromptCache 台账记账（spec REQ-2；worker 单线程）──
            // 2.1 prefill 段：首个块完成点一次性追加本轮输入增量段
            //     [reusable, |prompt_ids|)——绝对位推进与池 Advance(nodes
            //     chunk_len) 自然对齐，台账长 == total_written 不变量由此
            //     维系（多块 prefill 时中间块不记，末块一次记满）。
            //     空输入 / R3 禁用 = 无从记（prompt_ids 空或台账恒空）。
            if (out.is_prefill && !req->history_prefill_applied) {
                req->history_prefill_applied = true;
                std::lock_guard<std::mutex> lk(context_mtx_);
                auto it = context_caches_.find(req->context_id);
                if (it != context_caches_.end() && it->second.history) {
                    const std::size_t lo = std::min(req->reusable_len, req->prompt_ids.size());
                    for (std::size_t i = lo; i < req->prompt_ids.size(); ++i) {
                        it->second.history->AppendDecodedToken(req->prompt_ids[i]);
                    }
                    if (it->second.pool) {
                        // initTokenLen 初始化（R5/REQ-3 判定表）：本复用链
                        // 首个全量 prefill 完成点——total_written ≤ kv_max ⇒
                        // 锚定 = total_written；超长 prompt ⇒ 0（全环）。
                        // 只在该 Context 本链第一次落定（init==0 且无绕回
                        // 痕迹时设置；后续轮次 hit 链延续不重设）。
                        const std::size_t tw = it->second.pool->TotalWritten(0);
                        const std::size_t kv_max = static_cast<std::size_t>(it->second.pool->kv_cache_max_len());
                        const std::size_t cur_init = it->second.pool->init_token_len();
                        if (cur_init == 0 && tw > 0 && tw <= kv_max) {
                            it->second.pool->SetInitTokenLen(tw);
                        }
                    }
                }
            }
            // 2.2 decode 每步：采样输出 token 进 KV 后立即记
            {
                std::lock_guard<std::mutex> lk(context_mtx_);
                auto it = context_caches_.find(req->context_id);
                if (it != context_caches_.end() && it->second.history) {
                    it->second.history->AppendDecodedToken(token);
                }
            }

            // detokenize（skip_special_tokens 已从推理链路移除：终止字面量须保留在
            // 文本中，stop_strings 匹配才可达；剥离能力保留在 tokenizer 模块，引擎不用）
            std::string text_piece;
            if (tokenizer_) {
                DecodeOptions dec_opts;
                dec_opts.skip_special_tokens = false;
                Status dec_st = tokenizer_->decode({static_cast<int>(token)}, dec_opts, &text_piece);
                if (!dec_st.ok()) {
                    CANN_LLM_LOGW("postprocess.detokenize: decode failed (st=%d), text piece dropped", dec_st.code);
                }
            }
            req->text += text_piece;

            // stop_token 检查（请求级终止集：仅 RequestOptions 传入，与 tokenizer eos 无关）
            bool stop_hit = false;
            FinishReason finish_reason = FinishReason::MaxTokens;

            if (!req->stop_token_ids.empty()) {
                auto it = std::find(req->stop_token_ids.begin(), req->stop_token_ids.end(), token);
                if (it != req->stop_token_ids.end()) {
                    stop_hit = true;
                    finish_reason = FinishReason::StopToken;
                }
            }

            // stop_string 检查
            if (!stop_hit && !req->stop_strings.empty()) {
                for (const auto& ss : req->stop_strings) {
                    if (ss.size() <= req->text.size() &&
                        req->text.compare(req->text.size() - ss.size(), ss.size(), ss) == 0) {
                        stop_hit = true;
                        finish_reason = FinishReason::StopString;
                        // 去掉停止部分
                        req->text.resize(req->text.size() - ss.size());
                        break;
                    }
                }
            }

            // max_tokens 检查（请求级；≤0 = 默认上限）
            int max_tokens = req->max_tokens;
            if (max_tokens <= 0)
                max_tokens = 1024;

            if (!stop_hit && static_cast<int>(req->token_ids.size()) >= max_tokens) {
                stop_hit = true;
                finish_reason = FinishReason::MaxTokens;
            }

            if (stop_hit) {
                req->SetTerminal(RequestStatus::Completed, finish_reason);
                ClearActiveGenerate_(req->context_id);
            } else {
                // 流式回调
                if (req->callback) {
                    try {
                        req->callback(token, text_piece);
                    } catch (const std::exception& e) {
                        // 守护：用户回调异常不影响引擎
                        CANN_LLM_LOGW("postprocess.callback: user callback threw (%s), swallowed", e.what());
                    } catch (...) {
                        // 守护：用户回调异常不影响引擎
                        CANN_LLM_LOGW("postprocess.callback: user callback threw (unknown), swallowed");
                    }
                }
            }
        }
    }
}

void AutoregressivePipelineExecutor::ClearActiveGenerate_(ContextId cid)
{
    std::lock_guard<std::mutex> lk(context_mtx_);
    active_generates_.erase(cid);
}

// ── PromptCache pipeline 侧接线（design §4.4）────────────────────────────────

AutoregressivePipelineExecutor::ContextCache AutoregressivePipelineExecutor::ResolveCache_(ContextId cid) const
{
    std::lock_guard<std::mutex> lk(context_mtx_);
    auto it = context_caches_.find(cid);
    return it != context_caches_.end() ? it->second : ContextCache{};
}

void AutoregressivePipelineExecutor::RebindIfNeeded_(ContextId cid)
{
    // active_cache_ctx_ 仅 worker 写读（串行），无锁读安全。
    if (active_cache_ctx_ && *active_cache_ctx_ == cid)
        return;
    ContextCache cc = ResolveCache_(cid);
    if (!cc.pool)
        return; // Context 已销毁（在飞请求残留步）——保持现绑池
    // R9：双执行器 KV 族工位 O(工位数) 指针替换 + assembler 池引用同步
    // （记账推进/槽位映射以活跃池为权威）。R3 禁用模型同换绑（池同源同构，
    // 台账恒空走全量重算，零行为差异）。
    Status pst = prefill_executor_.RebindKvPool(cc.pool);
    Status dst = decode_executor_.RebindKvPool(cc.pool);
    if (!pst.ok() || !dst.ok()) {
        // 换绑失败 = 图已 Load 而 KV 工位记账缺失（时序 bug）——保持现绑池，
        // 后续 Forward 由图侧自然失败。不 crash：worker 循环需继续服务其余
        // 请求（REQ-8 终态回退照常走 ClearActiveGenerate_ 路径）。
        CANN_LLM_LOGE("promptcache.rebind: executor rebind failed (prefill=%d, decode=%d, "
                      "ctx=%llu) — keep previous binding",
                      pst.code, dst.code, static_cast<unsigned long long>(cid.value));
        return;
    }
    assembler_->RebindPool(cc.pool);
    // REQ-11 rebind 埋点（仅变化时）
    CANN_LLM_LOGI("promptcache.rebind: ctx_old=%s ctx_new=%llu",
                  active_cache_ctx_ ? std::to_string(active_cache_ctx_->value).c_str() : "(template)",
                  static_cast<unsigned long long>(cid.value));
    active_cache_ctx_ = cid;
}

void AutoregressivePipelineExecutor::RollbackHistoryIfNeeded_(const RequestImpl& req, RequestStatus terminal)
{
    // R10 失败/取消一致性：台账与池记账回退到本轮提交前快照（该轮视为
    // 未发生）。Completed 不回滚（成功链保留供下轮复用）。
    if (terminal == RequestStatus::Completed)
        return;
    ContextCache cc = ResolveCache_(req.context_id);
    if (!cc.history || !cc.pool)
        return; // Context 已销毁——随池释放，无事可回
    cc.history->Restore(req.history_snapshot);
    if (req.pool_total_written_snapshot < cc.pool->TotalWritten(0)) {
        // 仅回退（快照 < 当前 = 本轮推进过）；快照 == 当前（请求尚未
        // 写 KV 即终态）为幂等 no-op。
        (void)cc.pool->RollbackTo(0, req.pool_total_written_snapshot);
    }
    CANN_LLM_LOGI("promptcache.rollback: ctx=%llu terminal=%d history=%zu pool_total=%zu",
                  static_cast<unsigned long long>(req.context_id.value), static_cast<int>(terminal), cc.history->Size(),
                  static_cast<std::size_t>(cc.pool->TotalWritten(0)));
}

// ── PipelineExecutorFactory ─────────────────────────────────────────────────

Status PipelineExecutorFactory::create(PipelineType type, const ModelArtifact& artifact, const EngineOptions& options,
                                       std::unique_ptr<PipelineExecutorBase>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (type != PipelineType::Autoregressive)
        return Status::INVALID_ARGUMENT;
    // 加载失败即终止（2026-09-22）：构造内终止初始化并完成卸载的实例不写入
    // *out、不交出（局部 unique_ptr 析构收尾，KV 池等剩余资源随之确定性释放），
    // 错误码原样上抛（含权重 InitWeights 失败，如设备端 NPU 配额耗尽）。
    auto pipeline = std::make_unique<AutoregressivePipelineExecutor>(artifact, options);
    if (!pipeline->InitStatus().ok()) {
        return pipeline->InitStatus();
    }
    *out = std::move(pipeline);
    return Status::SUCCESS;
}

} // namespace cann_lm::engine
