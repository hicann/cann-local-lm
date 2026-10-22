#include "core/component/tokenizer/tokenizer.h"
#include "core/component/tokenizer/tokenizer_detail.h"

#include <array>
#include <cstdio>

namespace cann_lm::engine {

// ── M-03 BPETokenizer（手写 BPE，服务 gemma4；不依赖 sentencepiece）──────────
// 流水线（按 tokenizer.json 实物语义，v0.7）：AddedToken Trie 提取（原文，先于
// 归一化/预切分）→ UTF-8 校验 → Replace 归一（实物 normalizer：" "→"▁"）→
// Split(" ", MergedWithPrevious) 预切分（归一后无空格，等效整段单 piece）→
// BPE 合并（原生 Unicode 字面量字符串域，无 ByteLevel 字节映射）→ id 查表
// （miss→byte_fallback "<0xXX>" / unk）→ special 注入（M-06 零注入缺省）。
// 解码：字面量拼接 + ▁→" " 还原 + ByteFallback(<0xXX>→字节) 展开。

namespace {

// "<0xXX>" 字节回退字面量（byte_fallback；XX 为两位大写十六进制）
std::string ByteFallbackLiteral(unsigned char b) {
    std::array<char, 8> buf{};
    static const char* hex = "0123456789ABCDEF";
    buf[0] = '<'; buf[1] = '0'; buf[2] = 'x';
    buf[3] = hex[(b >> 4) & 0xF];
    buf[4] = hex[b & 0xF];
    buf[5] = '>';
    return std::string(buf.data(), 6);
}

bool ParseByteFallbackLiteral(std::string_view lit, unsigned char* out) {
    if (lit.size() != 6 || lit[0] != '<' || lit[1] != '0' || lit[2] != 'x' || lit[5] != '>') {
        return false;
    }
    auto hexval = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    const int hi = hexval(lit[3]);
    const int lo = hexval(lit[4]);
    if (hi < 0 || lo < 0) return false;
    *out = static_cast<unsigned char>((hi << 4) | lo);
    return true;
}

std::string ReplaceAll(std::string_view s, std::string_view from, std::string_view to) {
    if (from.empty()) return std::string(s);
    std::string out;
    out.reserve(s.size());
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t hit = s.find(from, pos);
        if (hit == std::string_view::npos) {
            out.append(s.substr(pos));
            break;
        }
        out.append(s.substr(pos, hit - pos));
        out.append(to);
        pos = hit + from.size();
    }
    return out;
}

}  // namespace

BPETokenizer::BPETokenizer(const TokenizerConfig& cfg, Status* status) {
    VocabSource src;
    src.artifact = std::filesystem::path(cfg.tokenizer_file);   // D-14：词表路径入配置
    // special 通道与词表可用性解耦（SR20260903001 4.1.4）：词表加载失败也保留注入配置，
    // 剥离集取已加载部分（失败时为空）
    const Status vocab_st = vocab_.load(src);
    specials_.configure(cfg, vocab_st.ok() ? vocab_.special_strip_ids()
                                           : std::vector<int>{});
    if (!vocab_st.ok()) {
        if (status) *status = vocab_st;
        return;
    }
    // model.type 校验：BPETokenizer 仅承载 BPE 型产物（Unigram 型 → 回议路线）
    if (vocab_.model_type() != "BPE") {
        if (status) *status = Status::FAILURE;
        return;
    }
    // merges → 字符串域 rank 表（构造期一次性）
    for (size_t i = 0; i < vocab_.merges().size(); ++i) {
        merge_ranks_.emplace(vocab_.merges()[i], static_cast<int>(i));
    }
    if (status) *status = Status::SUCCESS;
}

Status BPETokenizer::encode(std::string_view text, const EncodeOptions& opts,
                             std::vector<int>* out_ids) {
    if (!out_ids) return Status::INVALID_ARGUMENT;
    out_ids->clear();
    if (vocab_.vocab_size() == 0) return Status::FAILURE;  // not-ready

    if (!tokenizer_detail::ValidUtf8(text)) return Status::INVALID_ARGUMENT;

    // unk id：显式配置优先，缺省自产物 unk_token 推导
    std::optional<int> unk_id = specials_.id("unk");
    if (!unk_id && !vocab_.unk_token().empty()) {
        unk_id = vocab_.token_to_id(vocab_.unk_token());
    }

    // ① AddedToken 提取（左最长，先于归一化/预切分，作用于原文）
    std::vector<tokenizer_detail::AddedSegment> segments;
    tokenizer_detail::ExtractAddedTokens(text, vocab_.added_tokens(), &segments);

    for (const auto& seg : segments) {
        if (seg.is_added) {
            out_ids->push_back(seg.added_id);
            continue;
        }
        // ② Replace 归一（实物：" "→"▁"；NFKC 等归一化跳过 U-4）
        std::string normalized;
        if (vocab_.has_replace_normalizer()) {
            normalized = ReplaceAll(seg.text, vocab_.replace_normalizer().first,
                                    vocab_.replace_normalizer().second);
        } else {
            normalized.assign(seg.text);
        }
        // ③ Split(" ", MergedWithPrevious) 预切分：空格并入前段
        //   （归一后无空格 → 单 piece，与实物语义一致）
        std::vector<std::string> pieces;
        {
            std::string cur;
            for (const char c : normalized) {
                cur.push_back(c);
                if (c == ' ') {           // 归一后理论上不出现；保真实现
                    pieces.push_back(std::move(cur));
                    cur.clear();
                }
            }
            if (!cur.empty()) pieces.push_back(std::move(cur));
        }
        // ④⑤ BPE 合并（字符串域）→ id 查表（byte_fallback / unk）
        for (const auto& piece : pieces) {
            std::vector<std::string> word;
            {
                std::wstring ws;
                if (!tokenizer_detail::Utf8ToWide(piece, &ws)) {
                    return Status::FAILURE;  // 已校验过，防御
                }
                std::string ch;
                for (const wchar_t wc : ws) {
                    tokenizer_detail::WideToUtf8(std::wstring(1, wc), &ch);
                    word.push_back(ch);
                }
            }
            tokenizer_detail::BpeMerge(&word, merge_ranks_);
            for (const auto& sub : word) {
                if (auto id = vocab_.token_to_id(sub)) {
                    out_ids->push_back(*id);
                    continue;
                }
                if (vocab_.byte_fallback()) {
                    bool all_bytes_mapped = true;
                    std::vector<int> byte_ids;
                    for (const unsigned char b : sub) {
                        auto bid = vocab_.token_to_id(ByteFallbackLiteral(b));
                        if (!bid) { all_bytes_mapped = false; break; }
                        byte_ids.push_back(*bid);
                    }
                    if (all_bytes_mapped) {
                        out_ids->insert(out_ids->end(), byte_ids.begin(), byte_ids.end());
                        continue;
                    }
                }
                if (unk_id) {
                    out_ids->push_back(*unk_id);
                }
                // OOV 且无 byte_fallback/unk：跳过（不中断）
            }
        }
    }

    // ⑥ special 注入/截断/padding（M-06，零注入缺省）
    if (Status st = specials_.inject(out_ids, opts); !st.ok()) return st;
    return Status::SUCCESS;
}

Status BPETokenizer::decode(const std::vector<int>& ids, const DecodeOptions& opts,
                             std::string* out_text) {
    if (!out_text) return Status::INVALID_ARGUMENT;
    out_text->clear();
    if (vocab_.vocab_size() == 0) return Status::FAILURE;  // not-ready

    std::vector<int> kept = ids;
    if (opts.skip_special_tokens) {
        if (Status st = specials_.strip(ids, &kept); !st.ok()) return st;
    }

    // id→字面量：ByteFallback(<0xXX>) 展开为字节；其余 ▁→" " 还原后拼接
    // （decoder Sequence[Replace(▁→" "), ByteFallback, Fuse] 语义）
    const bool has_replace = vocab_.has_replace_normalizer();
    for (const int id : kept) {
        auto lit = vocab_.id_to_token(id);
        if (!lit) continue;  // 词表外 id：跳过（不中断）
        unsigned char byte = 0;
        if (ParseByteFallbackLiteral(*lit, &byte)) {
            out_text->push_back(static_cast<char>(byte));
            continue;
        }
        if (has_replace) {
            *out_text += ReplaceAll(*lit, vocab_.replace_normalizer().second,
                                    vocab_.replace_normalizer().first);
        } else {
            *out_text += std::string(*lit);
        }
    }
    return Status::SUCCESS;
}

size_t BPETokenizer::vocab_size() const { return vocab_.vocab_size(); }

std::optional<int> BPETokenizer::special_token_id(std::string_view name) const {
    return specials_.id(name);
}

}  // namespace cann_lm::engine
