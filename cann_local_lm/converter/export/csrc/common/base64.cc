#include "common/base64.h"

#include <array>

namespace td::internal {
namespace {

std::array<int8_t, 256> BuildDecodeTable()
{
    std::array<int8_t, 256> table{};
    table.fill(-1);
    for (int index = 0; index < 26; ++index) {
        table[static_cast<size_t>('A' + index)] = static_cast<int8_t>(index);
        table[static_cast<size_t>('a' + index)] = static_cast<int8_t>(26 + index);
    }
    for (int index = 0; index < 10; ++index) {
        table[static_cast<size_t>('0' + index)] = static_cast<int8_t>(52 + index);
    }
    table[static_cast<size_t>('+')] = 62;
    table[static_cast<size_t>('/')] = 63;
    return table;
}

const std::array<int8_t, 256> kDecodeTable = BuildDecodeTable();

bool IsWhitespace(const char ch)
{
    return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v';
}

} // namespace

bool DecodeBase64(const std::string& encoded, std::vector<uint8_t>& decoded)
{
    std::string compact;
    compact.reserve(encoded.size());
    for (const char ch : encoded) {
        if (!IsWhitespace(ch)) {
            compact.push_back(ch);
        }
    }

    if (compact.empty()) {
        decoded.clear();
        return true;
    }
    if (compact.size() % 4 != 0) {
        return false;
    }

    decoded.clear();
    decoded.reserve((compact.size() / 4) * 3);
    for (size_t index = 0; index < compact.size(); index += 4) {
        const char c0 = compact[index];
        const char c1 = compact[index + 1];
        const char c2 = compact[index + 2];
        const char c3 = compact[index + 3];
        if (c0 == '=' || c1 == '=') {
            return false;
        }

        const int8_t v0 = kDecodeTable[static_cast<uint8_t>(c0)];
        const int8_t v1 = kDecodeTable[static_cast<uint8_t>(c1)];
        if (v0 < 0 || v1 < 0) {
            return false;
        }

        if (c2 == '=') {
            if (c3 != '=' || index + 4 != compact.size()) {
                return false;
            }
            decoded.push_back(static_cast<uint8_t>((v0 << 2) | (v1 >> 4)));
            break;
        }

        const int8_t v2 = kDecodeTable[static_cast<uint8_t>(c2)];
        if (v2 < 0) {
            return false;
        }
        decoded.push_back(static_cast<uint8_t>((v0 << 2) | (v1 >> 4)));
        decoded.push_back(static_cast<uint8_t>(((v1 & 0x0F) << 4) | (v2 >> 2)));

        if (c3 == '=') {
            if (index + 4 != compact.size()) {
                return false;
            }
            break;
        }

        const int8_t v3 = kDecodeTable[static_cast<uint8_t>(c3)];
        if (v3 < 0) {
            return false;
        }
        decoded.push_back(static_cast<uint8_t>(((v2 & 0x03) << 6) | v3));
    }
    return true;
}

bool DecodeBase64ToString(const std::string& encoded, std::string& decoded)
{
    std::vector<uint8_t> bytes;
    if (!DecodeBase64(encoded, bytes)) {
        return false;
    }
    decoded.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
}

} // namespace td::internal
