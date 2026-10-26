// ═══════════════════════════════════════════════════════════════════════════
// DataTypeUtils 实现。纯函数，无状态、无外部依赖。
// ═══════════════════════════════════════════════════════════════════════════
#include "utils/data_type_utils.h"

#include <cstdio>
#include <cstring>

namespace cann_lm::engine {

std::size_t DataTypeElemSize(DataType t)
{
    switch (t) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::UINT8:
            return 1;
        case DataType::INT16:
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
        case DataType::DOUBLE:
            return 8;
        default:
            return 0; // RESERVED
    }
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
        default:
            return "UNKNOWN";
    }
}

uint16_t FloatToHalfBits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    uint32_t man = x & 0x007fffffu;

    if (exp == 0xffu) { // Inf / NaN（NaN 保留 quiet 位）
        return static_cast<uint16_t>(sign | 0x7c00u | (man != 0 ? 0x200u : 0u));
    }
    const int32_t e = static_cast<int32_t>(exp) - 127; // unbiased
    if (e >= 16) {                                     // 溢出 → Inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (e >= -14) { // half 规格化区间
        int32_t he = e + 15;
        uint32_t hm = man >> 13;
        const uint32_t rem = man & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) { // RNE
            ++hm;
            if (hm == 0x400u) {
                hm = 0;
                if (++he == 31)
                    return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(he) << 10) | hm);
    }
    if (e < -25) { // 下溢 → ±0
        return static_cast<uint16_t>(sign);
    }
    // half 非规格化区间：还原隐藏位后右移（RNE）
    man |= 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(-e - 1); // e=-14 → 13
    uint32_t hm = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (hm & 1u))) {
        if (++hm == 0x400u)
            return static_cast<uint16_t>(sign | 0x400u); // 进位至最小 normal
    }
    return static_cast<uint16_t>(sign | hm);
}

float HalfBitsToFloat(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (man == 0) {
            x = sign; // ±0
        } else {      // 非规格化：规格化后折算 float 指数
            uint32_t e = 0;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                ++e;
            }
            man &= 0x3ffu;
            x = sign | ((113u - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (man << 13); // Inf / NaN
    } else {
        x = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

double ReadElemAsDouble(const void* data, DataType t, std::size_t idx)
{
    if (data == nullptr)
        return 0.0;
    switch (t) {
        case DataType::FLOAT32:
            return static_cast<double>(static_cast<const float*>(data)[idx]);
        case DataType::FLOAT16:
            return static_cast<double>(HalfBitsToFloat(static_cast<const uint16_t*>(data)[idx]));
        case DataType::INT8:
            return static_cast<double>(static_cast<const int8_t*>(data)[idx]);
        case DataType::UINT8:
            return static_cast<double>(static_cast<const uint8_t*>(data)[idx]);
        case DataType::INT16:
            return static_cast<double>(static_cast<const int16_t*>(data)[idx]);
        case DataType::INT32:
            return static_cast<double>(static_cast<const int32_t*>(data)[idx]);
        case DataType::UINT32:
            return static_cast<double>(static_cast<const uint32_t*>(data)[idx]);
        case DataType::INT64:
            return static_cast<double>(static_cast<const int64_t*>(data)[idx]);
        case DataType::BOOL:
            return static_cast<const bool*>(data)[idx] ? 1.0 : 0.0;
        default:
            return 0.0;
    }
}

std::string FormatElems(const void* data, DataType t, std::size_t idx0, std::size_t n)
{
    std::string s;
    char buf[32];
    for (std::size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof(buf), i + 1 < n ? "%.6g," : "%.6g", ReadElemAsDouble(data, t, idx0 + i));
        s += buf;
    }
    return s;
}

} // namespace cann_lm::engine
