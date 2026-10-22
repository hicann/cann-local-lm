// ═══════════════════════════════════════════════════════════════════════════
// DataTypeUtils — dtype 相关的公共工具（engine 侧通用，非某组件私有）。
//
// 覆盖两类能力：
//   1. dtype → 元素字节数 / 简名（日志与容量计算共用）；
//   2. 软件 fp16（IEEE 754 binary16）位级转换与值读出/格式化——
//      hiai backend 迁移后宿主无 __fp16 保证：Linux x86_64 GCC(≤11) 与
//      MinGW 均缺该扩展类型；host 侧只做搬运/值展开（D-01），软件转换换
//      可移植性。舍入 RNE、denormal/Inf/NaN 全路径覆盖，与 aarch64
//      __fp16 原生通路位级一致。
//
// 命名空间：与 core/tensor_manager.h 的 engine 级别名（DataType/DataTypeSize）
// 保持一致，故直接用 cann_lm::engine。
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CANN_LM_ENGINE_UTILS_DATA_TYPE_UTILS_H
#define CANN_LM_ENGINE_UTILS_DATA_TYPE_UTILS_H

#include "types.h"   // hiai::DataType（backend/api，include 根经 cann_backend PUBLIC 传递）

#include <cstddef>
#include <cstdint>
#include <string>

namespace cann_lm::engine {

using DataType = hiai::DataType;

// ── dtype → 元素字节数 / 简名 ───────────────────────────────────────────────

/// dtype → 元素字节数（RESERVED/未知 → 0，调用方据此拒绝）。
std::size_t DataTypeElemSize(DataType t);

/// dtype → 简名（日志可读性；未知 → "UNKNOWN"）。
const char* DataTypeName(DataType t);

// ── 软件 fp16（IEEE 754 binary16）位级转换与读出 ────────────────────────────

/// float → half 位模式（RNE；denormal/Inf/NaN 全路径）。
uint16_t FloatToHalfBits(float f);

/// half 位模式 → float。
float HalfBitsToFloat(uint16_t h);

/// 低位 half 存储类型（2 字节，与 uint16_t 位模式同布局）。
/// 仅供 table/scales 缓冲区读写，不参与算术；`explicit fp16_t(float)` +
/// `explicit operator float()` 保证不会隐式参与表达式，避免半精度中间结果。
struct fp16_t {
    fp16_t() = default;
    explicit fp16_t(float f) : bits_(FloatToHalfBits(f)) {}
    explicit operator float() const { return HalfBitsToFloat(bits_); }
    uint16_t bits_ = 0;
};

// ── dtype 缓冲区读出与格式化（日志用） ──────────────────────────────────────

/// 按 dtype 读出第 idx 个元素并折算为 double（FLOAT32/FLOAT16/INT8/UINT8/
/// INT16/UINT16/INT32/UINT32/INT64/BOOL；其余 → 0.0）。
double ReadElemAsDouble(const void* data, DataType t, std::size_t idx);

/// 从 idx0 起 n 个元素格式化为 "v1,v2,..."（%.6g 精度，无尾逗号）。
std::string FormatElems(const void* data, DataType t, std::size_t idx0,
                        std::size_t n);

}  // namespace cann_lm::engine

#endif  // CANN_LM_ENGINE_UTILS_DATA_TYPE_UTILS_H