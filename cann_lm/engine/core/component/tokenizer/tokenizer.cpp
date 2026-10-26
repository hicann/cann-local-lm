#include "core/component/tokenizer/tokenizer.h"

#include "infra/log.h"

#include <algorithm>
#include <unordered_map>

namespace cann_lm::engine {

// ── TokenizerBase 默认 batch_encode（REQ-05：单条失败标记不整批崩）──────────
Status TokenizerBase::batch_encode(const std::vector<std::string>& texts, const EncodeOptions& opts,
                                   std::vector<std::vector<int>>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    out->clear();
    out->reserve(texts.size());
    Status first_err = Status::SUCCESS;
    for (const auto& t : texts) {
        std::vector<int> ids;
        const Status e = encode(t, opts, &ids);
        if (!e.ok()) {
            if (first_err.ok())
                first_err = e;
            out->push_back({});
        } else {
            out->push_back(std::move(ids));
        }
    }
    // Longest 批量对齐：补 pad 至批内最长（pad id 经派生 special_token_id 取）。
    // 失败条目保持空向量、不参与 padding（外审 F-11：全 pad 行会掩盖"标记失败"）。
    if (opts.padding == EncodeOptions::Padding::Longest && !texts.empty()) {
        const std::optional<int> pad = special_token_id("pad");
        size_t max_len = 0;
        for (const auto& v : *out)
            max_len = std::max(max_len, v.size());
        if (max_len > 0) {
            if (!pad.has_value())
                return Status::INVALID_ARGUMENT; // pad 需求无 pad id
            for (auto& v : *out) {
                if (!v.empty() && v.size() < max_len)
                    v.resize(max_len, *pad);
            }
        }
    }
    return first_err;
}

// ── M-02 Registrar / TokenizerFactory ───────────────────────────────────────
// 注册表为函数局部静态（首次 register_ 时构造，C++11 magic static 线程安全）——
// 注册宏与 create 同在本翻译单元（链接丢弃约束，外审 M2：静态库中未被引用的
// 注册对象可能被链接器丢弃，同 TU 可保证 create 被引用即整 TU 被拉入）。
namespace {

std::unordered_map<Algorithm, Registrar::Ctor>& Registry()
{
    static std::unordered_map<Algorithm, Registrar::Ctor> instance;
    return instance;
}

} // namespace

void Registrar::register_(Algorithm algo, Ctor ctor)
{
    auto& reg = Registry();
    if (reg.find(algo) != reg.end()) {
        // RegistrationConflict：拒绝后者 + WARN（无返回值），保留先注册者（REQ-07 不静默）
        CANN_LLM_LOGW("tokenizer.factory.register.conflict: duplicate algorithm registration rejected");
        return;
    }
    reg.emplace(algo, ctor);
}

Status TokenizerFactory::create(Algorithm algo, const TokenizerConfig& cfg, std::unique_ptr<TokenizerBase>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    const auto& reg = Registry();
    if (reg.empty()) {
        // SIOF / 链接丢弃兜底：注册表空 → FATAL log + INVALID_ARGUMENT（不静默）
        CANN_LLM_LOGE("tokenizer.factory.create: empty registry (SIOF or linker dropped registration objects)");
        return Status::INVALID_ARGUMENT;
    }
    const auto it = reg.find(algo);
    if (it == reg.end())
        return Status::INVALID_ARGUMENT; // 未注册枚举
    Status st = Status::SUCCESS;
    std::unique_ptr<TokenizerBase> tok = it->second(cfg, &st);
    if (!tok || !st.ok())
        return st.ok() ? Status::FAILURE : st; // 加载失败：不发布实例
    *out = std::move(tok);
    return Status::SUCCESS;
}

std::vector<Algorithm> TokenizerFactory::list_registered()
{
    std::vector<Algorithm> algos;
    algos.reserve(Registry().size());
    for (const auto& [algo, _] : Registry())
        algos.push_back(algo);
    return algos;
}

// ── M-07 SPMTokenizer 占位桩（sentencepiece 库路径预留，本期不实现）─────────
// create(SPM) 恒 FAILURE（占位加载失败，预期行为非缺陷，D-04）；空 LoadParams
// 直接构造的实例 not-ready，encode/decode 返回 FAILURE（stub 时代语义）。

SPMTokenizer::SPMTokenizer(const TokenizerConfig& cfg, Status* status)
{
    (void)cfg;
    if (status)
        *status = Status::FAILURE; // 占位：.model 加载未实现
}

Status SPMTokenizer::encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids)
{
    (void)text;
    (void)opts;
    if (!out_ids)
        return Status::INVALID_ARGUMENT;
    out_ids->clear();
    return Status::FAILURE; // 占位：not-ready
}

Status SPMTokenizer::decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text)
{
    (void)ids;
    (void)opts;
    if (!out_text)
        return Status::INVALID_ARGUMENT;
    out_text->clear();
    return Status::FAILURE; // 占位：not-ready
}

size_t SPMTokenizer::vocab_size() const { return 0; }

std::optional<int> SPMTokenizer::special_token_id(std::string_view name) const
{
    (void)name;
    return std::nullopt;
}

// ── 静态自注册（与 TokenizerFactory::create 同翻译单元——链接丢弃约束）──────
REGISTER_TOKENIZER(Algorithm::BPE, BPETokenizer);
REGISTER_TOKENIZER(Algorithm::Qwen, QwenTokenizer);
REGISTER_TOKENIZER(Algorithm::SPM, SPMTokenizer);

} // namespace cann_lm::engine
