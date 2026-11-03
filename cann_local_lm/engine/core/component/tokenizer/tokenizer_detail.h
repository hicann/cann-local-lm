#pragma once

// tokenizer 模块内部工具（SR20260728001 v0.7）。仅 tokenizer 实现文件包含，
// 不进公共包含路径。提供：严格 UTF-8 校验/解码、wstring↔UTF-8 转换、
// BPE 合并模板（按 merges 优先级）、AddedToken 左最长原文提取。

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cann_local_lm::engine {
namespace tokenizer_detail {

// ── 严格 UTF-8 校验：拒绝空续字节、过长编码、代理对、>U+10FFFF ──────────────
inline bool ValidUtf8(std::string_view s)
{
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const auto* end = p + s.size();
    while (p < end) {
        const unsigned char c = *p;
        if (c < 0x80) {
            ++p;
            continue;
        }
        int len = 0;
        unsigned cp = 0;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        } // 0x80-0xBF 孤立续字节 / 0xF8+
        if (p + len > end)
            return false;
        for (int i = 1; i < len; ++i) {
            const unsigned char cc = p[i];
            if ((cc & 0xC0) != 0x80)
                return false; // 非续字节
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (len == 2 && cp < 0x80)
            return false; // 过长
        if (len == 3 && cp < 0x800)
            return false; // 过长
        if (len == 4 && cp < 0x10000)
            return false; // 过长
        if (cp >= 0xD800 && cp <= 0xDFFF)
            return false; // 代理对
        if (cp > 0x10FFFF)
            return false;
        p += len;
    }
    return true;
}

// ── UTF-8 → wstring（码点级；调用方须先 ValidUtf8）─────────────────────────
inline bool Utf8ToWide(std::string_view s, std::wstring* out)
{
    out->clear();
    out->reserve(s.size());
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const auto* end = p + s.size();
    while (p < end) {
        const unsigned char c = *p;
        if (c < 0x80) {
            out->push_back(static_cast<wchar_t>(c));
            ++p;
            continue;
        }
        int len = 0;
        unsigned cp = 0;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (p + len > end)
            return false;
        for (int i = 1; i < len; ++i) {
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        out->push_back(static_cast<wchar_t>(cp));
        p += len;
    }
    return true;
}

// ── wstring → UTF-8 ─────────────────────────────────────────────────────────
inline void WideToUtf8(const std::wstring& ws, std::string* out)
{
    out->clear();
    out->reserve(ws.size());
    for (const wchar_t wc : ws) {
        const unsigned cp = static_cast<unsigned>(wc);
        if (cp < 0x80) {
            out->push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
}

// ── BPE 合并（标准算法）：按 merges 优先级反复合并相邻符号对 ────────────────
// Symbol 为 std::string（BPE 原生字面量路径）或 std::wstring（ByteLevel 映射
// 后的 wtoken 路径，用户拍板口径）。rank 缺失的对不合并。
template <typename Symbol>
void BpeMerge(std::vector<Symbol>* word, const std::map<std::pair<Symbol, Symbol>, int>& ranks)
{
    while (word->size() >= 2) {
        // 每轮找最低秩对（同秩取先出现者）
        int best_rank = -1;
        std::pair<Symbol, Symbol> best;
        for (size_t i = 0; i + 1 < word->size(); ++i) {
            auto it = ranks.find({(*word)[i], (*word)[i + 1]});
            if (it != ranks.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best = it->first;
            }
        }
        if (best_rank < 0)
            return;
        // 标准 BPE 语义（GPT-2/tokenizers 参考）：一轮合并该对的全部
        // 非重叠出现（从左到右）——逐次合并会在特定 rank 序下分叉（自审发现）
        std::vector<Symbol> out;
        out.reserve(word->size());
        size_t i = 0;
        while (i < word->size()) {
            if (i + 1 < word->size() && (*word)[i] == best.first && (*word)[i + 1] == best.second) {
                Symbol merged = (*word)[i];
                merged += (*word)[i + 1];
                out.push_back(std::move(merged));
                i += 2;
            } else {
                out.push_back((*word)[i]);
                i += 1;
            }
        }
        *word = std::move(out);
    }
}

// ── AddedToken 提取（H-2）：左最长匹配，先于预切分，作用于原文 ──────────────
// added 按 literal 长度降序尝试；输出段落序列（added id 或 原文 span）。
struct AddedSegment {
    bool is_added = false;
    int added_id = 0;
    std::string_view text; // is_added==false 时的原文片段
};

inline void ExtractAddedTokens(std::string_view text,
                               const std::vector<std::pair<std::string, int>>& added, // (literal, id)
                               std::vector<AddedSegment>* out)
{
    out->clear();
    // 长度降序索引（added 数量级 ~26，线性扫描足够）
    std::vector<size_t> order(added.size());
    for (size_t i = 0; i < added.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return added[a].first.size() > added[b].first.size(); });

    std::string_view pending; // 当前未 flush 的原文片段
    size_t pos = 0;
    const size_t n = text.size();
    auto flush = [&]() {
        if (!pending.empty()) {
            out->push_back({false, 0, pending});
            pending = std::string_view();
        }
    };
    while (pos < n) {
        bool matched = false;
        for (size_t oi = 0; oi < order.size(); ++oi) {
            const auto& [lit, id] = added[order[oi]];
            if (lit.empty() || pos + lit.size() > n)
                continue;
            if (text.compare(pos, lit.size(), lit) == 0) {
                flush();
                out->push_back({true, id, std::string_view()});
                pos += lit.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            if (pending.empty()) {
                pending = text.substr(pos, 1);
            } else {
                pending = text.substr(static_cast<size_t>(pending.data() - text.data()), pending.size() + 1);
            }
            ++pos;
        }
    }
    flush();
}

} // namespace tokenizer_detail
} // namespace cann_local_lm::engine
