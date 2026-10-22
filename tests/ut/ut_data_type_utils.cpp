// DataTypeUtils 单元测试（utils/data_type_utils.{h,cpp}）。
// 411 行从 text_embedding.cpp 抽出为公共 util 后，行为必须逐位不变：
//   - fp16 软件转换（RNE / denormal / Inf / NaN）位级断言；
//   - DataTypeElemSize / DataTypeName 全枚举覆盖；
//   - ReadElemAsDouble / FormatElems 读出与格式（含 FLOAT16 经软件转换）。
#include "utils/data_type_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace cann_lm::engine {
namespace {

// ── fp16 位级转换：FloatToHalfBits / HalfBitsToFloat ────────────────────────

// 规格化区间：RNE 舍入（0.5 ULP 处进位到偶数）。
TEST(DataTypeUtilsTest, FloatToHalfBitsNormalRNE) {
    EXPECT_EQ(FloatToHalfBits(0.0f), 0x0000u);
    EXPECT_EQ(FloatToHalfBits(1.0f), 0x3c00u);
    EXPECT_EQ(FloatToHalfBits(-1.0f), 0xbc00u);
    EXPECT_EQ(FloatToHalfBits(2.0f), 0x4000u);
    EXPECT_EQ(FloatToHalfBits(0.5f), 0x3800u);
}

// 溢出 → ±Inf；最小规格化 / 最大有限值。
TEST(DataTypeUtilsTest, FloatToHalfBitsOverflowToInf) {
    EXPECT_EQ(FloatToHalfBits(65520.0f), 0x7c00u);    // 溢出到 Inf
    EXPECT_EQ(FloatToHalfBits(-65520.0f), 0xfc00u);
    EXPECT_EQ(FloatToHalfBits(65504.0f), 0x7bffu);    // half 最大有限值
}

// denormal 区间与下溢 → ±0。
TEST(DataTypeUtilsTest, FloatToHalfBitsDenormalAndUnderflow) {
    EXPECT_EQ(FloatToHalfBits(0.0f), 0x0000u);
    EXPECT_EQ(FloatToHalfBits(-0.0f), 0x8000u);
    // 2^-24 = half 最小 denormal（尾数 1）→ 0x0001
    EXPECT_EQ(FloatToHalfBits(5.9604645e-08f), 0x0001u);
    // 2^-25 恰在半个 ULP 处，RNE 舍入到 0
    EXPECT_EQ(FloatToHalfBits(2.9802322e-08f), 0x0000u);
    // 极小数下溢 → ±0
    EXPECT_EQ(FloatToHalfBits(1e-20f), 0x0000u);
    EXPECT_EQ(FloatToHalfBits(-1e-20f), 0x8000u);
}

// Inf / NaN 全路径（NaN 保留 quiet 位）。
TEST(DataTypeUtilsTest, FloatToHalfBitsInfNaN) {
    const float inf = 1e30f * 1e30f;
    const float ninf = -inf;
    EXPECT_EQ(FloatToHalfBits(inf), 0x7c00u);
    EXPECT_EQ(FloatToHalfBits(ninf), 0xfc00u);
    // NaN → 指数全 1 + quiet 位
    const uint32_t qnan_bits = 0x7fc00000u;
    float qnan = 0.0f;
    std::memcpy(&qnan, &qnan_bits, sizeof(qnan));
    EXPECT_EQ(FloatToHalfBits(qnan), 0x7e00u);
}

// HalfBitsToFloat：规格化 / denormal / ±0 / Inf / NaN。
TEST(DataTypeUtilsTest, HalfBitsToFloatAllRanges) {
    EXPECT_EQ(HalfBitsToFloat(0x0000u), 0.0f);
    EXPECT_EQ(HalfBitsToFloat(0x8000u), -0.0f);
    EXPECT_EQ(HalfBitsToFloat(0x3c00u), 1.0f);
    EXPECT_EQ(HalfBitsToFloat(0xbc00u), -1.0f);
    EXPECT_EQ(HalfBitsToFloat(0x7bffu), 65504.0f);
    // 最小 denormal → 2^-24
    EXPECT_EQ(HalfBitsToFloat(0x0001u), 5.9604645e-08f);
    EXPECT_TRUE(HalfBitsToFloat(0x7c00u) > 3e38f);            // +Inf
    EXPECT_TRUE(HalfBitsToFloat(0xfc00u) < -3e38f);           // -Inf
    EXPECT_TRUE(HalfBitsToFloat(0x7e00u) != HalfBitsToFloat(0x7e00u));  // NaN
}

// 往返：half 位模式 → float → half 必须逐位还原（全 65536 组合）。
TEST(DataTypeUtilsTest, HalfRoundTripExhaustive) {
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const float f = HalfBitsToFloat(static_cast<uint16_t>(bits));
        const uint16_t back = FloatToHalfBits(f);
        // NaN：只要求仍是 NaN（quiet 位归一化），其余要求逐位相等
        if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu) != 0) {
            EXPECT_EQ(back & 0x7c00u, 0x7c00u) << "bits=" << bits;
        } else {
            EXPECT_EQ(back, static_cast<uint16_t>(bits)) << "bits=" << bits;
        }
    }
}

// fp16_t：显式构造/转换，位模式与裸函数一致。
TEST(DataTypeUtilsTest, Fp16TypeBitsMatchFreeFunctions) {
    const fp16_t h(1.5f);
    EXPECT_EQ(h.bits_, FloatToHalfBits(1.5f));
    EXPECT_EQ(static_cast<float>(h), 1.5f);

    const fp16_t from_bits = [] {
        fp16_t v;
        v.bits_ = 0x3c00u;
        return v;
    }();
    EXPECT_EQ(static_cast<float>(from_bits), 1.0f);
}

// ── DataTypeElemSize / DataTypeName ─────────────────────────────────────────

TEST(DataTypeUtilsTest, ElemSizeCoversAllTypes) {
    EXPECT_EQ(DataTypeElemSize(DataType::BOOL), 1u);
    EXPECT_EQ(DataTypeElemSize(DataType::INT8), 1u);
    EXPECT_EQ(DataTypeElemSize(DataType::UINT8), 1u);
    EXPECT_EQ(DataTypeElemSize(DataType::INT16), 2u);
    EXPECT_EQ(DataTypeElemSize(DataType::FLOAT16), 2u);
    EXPECT_EQ(DataTypeElemSize(DataType::INT32), 4u);
    EXPECT_EQ(DataTypeElemSize(DataType::UINT32), 4u);
    EXPECT_EQ(DataTypeElemSize(DataType::FLOAT32), 4u);
    EXPECT_EQ(DataTypeElemSize(DataType::INT64), 8u);
    EXPECT_EQ(DataTypeElemSize(DataType::DOUBLE), 8u);
    EXPECT_EQ(DataTypeElemSize(DataType::RESERVED), 0u);
}

TEST(DataTypeUtilsTest, NameCoversAllTypes) {
    EXPECT_STREQ(DataTypeName(DataType::FLOAT32), "FLOAT32");
    EXPECT_STREQ(DataTypeName(DataType::FLOAT16), "FLOAT16");
    EXPECT_STREQ(DataTypeName(DataType::INT8), "INT8");
    EXPECT_STREQ(DataTypeName(DataType::INT32), "INT32");
    EXPECT_STREQ(DataTypeName(DataType::INT64), "INT64");
    EXPECT_STREQ(DataTypeName(DataType::UINT8), "UINT8");
    EXPECT_STREQ(DataTypeName(DataType::INT16), "INT16");
    EXPECT_STREQ(DataTypeName(DataType::BOOL), "BOOL");
    EXPECT_STREQ(DataTypeName(DataType::UINT32), "UINT32");
    EXPECT_STREQ(DataTypeName(DataType::DOUBLE), "DOUBLE");
    EXPECT_STREQ(DataTypeName(DataType::RESERVED), "UNKNOWN");
}

// ── ReadElemAsDouble ────────────────────────────────────────────────────────

TEST(DataTypeUtilsTest, ReadElemAsDoublePerType) {
    const float f32[2] = {1.5f, -2.25f};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(f32, DataType::FLOAT32, 0), 1.5);
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(f32, DataType::FLOAT32, 1), -2.25);

    const int8_t i8[2] = {-128, 127};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(i8, DataType::INT8, 0), -128.0);
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(i8, DataType::INT8, 1), 127.0);

    const uint8_t u8[1] = {255};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(u8, DataType::UINT8, 0), 255.0);

    const int16_t i16[1] = {-300};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(i16, DataType::INT16, 0), -300.0);

    const int32_t i32[1] = {-70000};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(i32, DataType::INT32, 0), -70000.0);

    const uint32_t u32[1] = {4000000000u};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(u32, DataType::UINT32, 0), 4000000000.0);

    const int64_t i64[1] = {-5000000000LL};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(i64, DataType::INT64, 0), -5000000000.0);

    const bool bl[2] = {true, false};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(bl, DataType::BOOL, 0), 1.0);
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(bl, DataType::BOOL, 1), 0.0);

    // FLOAT16 经软件转换读出
    const uint16_t h[2] = {0x3c00u, 0x3800u};
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(h, DataType::FLOAT16, 0), 1.0);
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(h, DataType::FLOAT16, 1), 0.5);

    // 未支持类型 / nullptr → 0.0（不崩）
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(f32, DataType::RESERVED, 0), 0.0);
    EXPECT_DOUBLE_EQ(ReadElemAsDouble(nullptr, DataType::FLOAT32, 0), 0.0);
}

// ── FormatElems ─────────────────────────────────────────────────────────────

TEST(DataTypeUtilsTest, FormatElemsBasic) {
    const float v[3] = {1.0f, 2.5f, -3.0f};
    EXPECT_EQ(FormatElems(v, DataType::FLOAT32, 0, 3), "1,2.5,-3");
    EXPECT_EQ(FormatElems(v, DataType::FLOAT32, 1, 2), "2.5,-3");
    EXPECT_EQ(FormatElems(v, DataType::FLOAT32, 0, 0), "");      // n=0 → 空
    EXPECT_EQ(FormatElems(v, DataType::FLOAT32, 2, 1), "-3");    // 单元素无尾逗号
}

TEST(DataTypeUtilsTest, FormatElemsHalfAndInt) {
    const uint16_t h[2] = {0x3c00u, 0x4000u};   // 1.0, 2.0
    EXPECT_EQ(FormatElems(h, DataType::FLOAT16, 0, 2), "1,2");

    const int8_t i8[3] = {1, 2, 3};
    EXPECT_EQ(FormatElems(i8, DataType::INT8, 0, 3), "1,2,3");
}

}  // namespace
}  // namespace cann_lm::engine