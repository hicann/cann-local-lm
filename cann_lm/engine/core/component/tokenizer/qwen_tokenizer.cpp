#include "core/component/tokenizer/tokenizer.h"
#include "core/component/tokenizer/tokenizer_detail.h"

#include <array>
#include <locale>
#include <map>
#include <regex>
#include <unordered_map>

namespace cann_lm::engine {

// ── M-04 QwenTokenizer（手写 ByteLevel BPE，对齐 Qwen3.5）───────────────────
// 流水线（用户拍板口径，D-10）：AddedToken Trie 提取（原文，先于预切分）→
// UTF-8 校验 → D-10 正则预切分（字节级）→ 字节转 Unicode 映射（ByteLevel
// byte_encoder，逐 char 转 wchar_t 组成 wtoken）→ wtoken BPE 合并 →
// subword 转回 UTF-8 查 vocab 得 id（miss→unk）→ special 注入（M-06）。
// NFC 归一化跳过（U-4 豁免）；已知豁免分叉见 DESIGN D-10 §4。

namespace {

// D-10 预切分正则（用户拍板 2026-09-01；aplha 笔误已订正为 alpha，保留
// [a-zA-Z]ü 分支）。字节级匹配——C locale 下 ASCII 分类稳定（OHOS musl 安全）。
// D-10 预切分正则（用户拍板 2026-09-01；aplha 笔误已订正为 alpha，保留
// [a-zA-Z]ü 分支）。字节级匹配。**钉死 classic locale 编译**（外审 F-07）：
// 否则 std::regex 按构造时全局 locale 分类 [:alpha:]——宿主进程切到单字节
// 扩展字符集（如 ISO-8859-1）时 0x80-0xFF 字节分类改变，切分静默分叉。
const std::regex& PretokRegex()
{
    static const std::regex re = [] {
        std::regex r;
        r.imbue(std::locale::classic()); // 先钉 locale 再编译
        r.assign(
            R"('s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]ü|[^\r\n[:alpha:][:digit:]]?[[:alpha:]]+|[[:digit:]]| ?[^\s[:alpha:][:digit:]]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");
        return r;
    }();
    return re;
}

// ByteLevel byte→unicode 映射（GPT-2 标准）：可打印区间恒等，其余映射到
// U+0100 起的 68 个码点。byte_encoder[b] = 映射码点；byte_decoder 反查。
const std::array<unsigned, 256>& ByteEncoder()
{
    static const std::array<unsigned, 256> tbl = [] {
        std::array<unsigned, 256> t{};
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool identity = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            t[static_cast<size_t>(b)] = identity ? static_cast<unsigned>(b) : static_cast<unsigned>(256 + n++);
        }
        return t;
    }();
    return tbl;
}

const std::unordered_map<unsigned, unsigned>& ByteDecoder()
{
    static const std::unordered_map<unsigned, unsigned> tbl = [] {
        std::unordered_map<unsigned, unsigned> t;
        const auto& enc = ByteEncoder();
        for (int b = 0; b < 256; ++b)
            t.emplace(enc[static_cast<size_t>(b)], static_cast<unsigned>(b));
        return t;
    }();
    return tbl;
}

} // namespace

QwenTokenizer::QwenTokenizer(const TokenizerConfig& cfg, Status* status)
{
    VocabSource src;
    src.artifact = std::filesystem::path(cfg.tokenizer_file); // D-14：词表路径入配置
    // special 通道与词表可用性解耦（SR20260903001 4.1.4）
    const Status vocab_st = vocab_.load(src);
    specials_.configure(cfg, vocab_st.ok() ? vocab_.special_strip_ids() : std::vector<int>{});
    if (!vocab_st.ok()) {
        if (status)
            *status = vocab_st;
        return;
    }
    // merges → wstring 域 rank 表（构造期一次性，用户拍板口径）
    {
        std::wstring a, b;
        for (size_t i = 0; i < vocab_.merges().size(); ++i) {
            const auto& m = vocab_.merges()[i];
            if (!tokenizer_detail::Utf8ToWide(m.first, &a) || !tokenizer_detail::Utf8ToWide(m.second, &b)) {
                if (status)
                    *status = Status::FAILURE; // merges 条目非法 UTF-8
                return;
            }
            merge_ranks_.emplace(std::make_pair(a, b), static_cast<int>(i));
        }
    }
    if (status)
        *status = Status::SUCCESS;
}

Status QwenTokenizer::encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids)
{
    if (!out_ids)
        return Status::INVALID_ARGUMENT;
    out_ids->clear();
    if (vocab_.vocab_size() == 0)
        return Status::FAILURE; // not-ready

    // UTF-8 校验（非法 → 不产生半截结果）
    if (!tokenizer_detail::ValidUtf8(text))
        return Status::INVALID_ARGUMENT;

    // ① AddedToken 提取（左最长，先于预切分，作用于原文）
    std::vector<tokenizer_detail::AddedSegment> segments;
    tokenizer_detail::ExtractAddedTokens(text, vocab_.added_tokens(), &segments);

    const auto unk_id = specials_.id("unk");

    for (const auto& seg : segments) {
        if (seg.is_added) {
            out_ids->push_back(seg.added_id);
            continue;
        }
        // ② D-10 正则预切分（字节级）
        const char* begin = seg.text.data();
        const char* end = begin + seg.text.size();
        for (auto it = std::cregex_iterator(begin, end, PretokRegex()); it != std::cregex_iterator(); ++it) {
            const std::string piece = it->str();
            if (piece.empty())
                continue;
            // ③ 字节转 Unicode 映射：逐字节 → 码点 → wchar_t wtoken
            std::vector<std::wstring> word;
            word.reserve(piece.size());
            for (const unsigned char byte : piece) {
                word.emplace_back(1, static_cast<wchar_t>(ByteEncoder()[byte]));
            }
            // ④ BPE 合并（wstring 域，按 merges 优先级）
            tokenizer_detail::BpeMerge(&word, merge_ranks_);
            // ⑤ subword 转回 UTF-8 查表取 id（miss → unk；unk 未配置则跳过）
            for (const auto& sub : word) {
                std::string utf8;
                tokenizer_detail::WideToUtf8(sub, &utf8);
                if (auto id = vocab_.token_to_id(utf8)) {
                    out_ids->push_back(*id);
                } else if (unk_id) {
                    out_ids->push_back(*unk_id);
                }
                // OOV 且无 unk：跳过（Qwen 实物 unk=null；ByteLevel 全字节入表，罕见）
            }
        }
    }

    // ⑥ special 注入/截断/padding（M-06，零注入缺省）
    if (Status st = specials_.inject(out_ids, opts); !st.ok())
        return st;
    return Status::SUCCESS;
}

Status QwenTokenizer::decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text)
{
    if (!out_text)
        return Status::INVALID_ARGUMENT;
    out_text->clear();
    if (vocab_.vocab_size() == 0)
        return Status::FAILURE; // not-ready

    std::vector<int> kept = ids;
    if (opts.skip_special_tokens) {
        if (Status st = specials_.strip(ids, &kept); !st.ok())
            return st;
    }

    // id→字面量→ByteLevel 字节还原（映射码点 → 原字节；非映射码点按原样输出）
    const auto& dec = ByteDecoder();
    std::wstring literal;
    for (const int id : kept) {
        auto lit = vocab_.id_to_token(id);
        if (!lit)
            continue; // 词表外 id：跳过（不中断）
        if (!tokenizer_detail::Utf8ToWide(*lit, &literal)) {
            *out_text += std::string(*lit); // 非法 UTF-8 字面量：原样追加
            continue;
        }
        for (const wchar_t wc : literal) {
            auto it = dec.find(static_cast<unsigned>(wc));
            if (it != dec.end()) {
                out_text->push_back(static_cast<char>(it->second));
            } else {
                std::string utf8;
                tokenizer_detail::WideToUtf8(std::wstring(1, wc), &utf8);
                *out_text += utf8;
            }
        }
    }
    return Status::SUCCESS;
}

size_t QwenTokenizer::vocab_size() const { return vocab_.vocab_size(); }

std::optional<int> QwenTokenizer::special_token_id(std::string_view name) const { return specials_.id(name); }

} // namespace cann_lm::engine
