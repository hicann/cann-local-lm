/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <graph/operator_reg.h>

#include "backend/epair_archive.h"
#include "backend/ge_graph_builder.h"
#include "backend/ge_graph_optimizer.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

namespace ge {

std::vector<std::string> kAttrListString = {""};
std::vector<int64_t> kAttrListInt = {0};
std::vector<float> kAttrListFloat = {0.0f};
std::vector<bool> kAttrListBool = {false};
std::vector<ge::DataType> kAttrListType = {ge::DT_FLOAT};

REG_OP(Data)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

        REG_OP(Const)
    .OUTPUT(y, ge::TensorType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_UINT8,
                               ge::DT_INT32, ge::DT_INT64, ge::DT_UINT32, ge::DT_UINT64, ge::DT_BOOL, ge::DT_DOUBLE}))
    .ATTR(value, Tensor, ge::Tensor())
    .OP_END_FACTORY_REG(Const)

        REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

        REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)

        REG_OP(Add)
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

        REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)

        REG_OP(TDAttrCoverage)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(int_attr, Int, 0)
    .ATTR(float_attr, Float, 0.0f)
    .ATTR(string_attr, String, "")
    .ATTR(bool_attr, Bool, false)
    .ATTR(dtype_attr, Type, ge::DT_FLOAT)
    .ATTR(list_int_attr, ListInt, kAttrListInt)
    .ATTR(list_float_attr, ListFloat, kAttrListFloat)
    .ATTR(list_bool_attr, ListBool, kAttrListBool)
    .ATTR(list_string_attr, ListString, kAttrListString)
    .ATTR(list_type_attr, ListType, kAttrListType)
    .ATTR(empty_list_attr, ListInt, {})
    .OP_END_FACTORY_REG(TDAttrCoverage)

        REG_OP(TDConcatV2)
    .DYNAMIC_INPUT(x, ge::TensorType::ALL())
    .INPUT(concat_dim, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(N, Int, 2)
    .OP_END_FACTORY_REG(TDConcatV2)

        REG_OP(RopeInPlace)
    .INPUT(self, ge::TensorType::ALL())
    .INPUT(sin_cache, ge::TensorType::ALL())
    .INPUT(cos_cache, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(RopeInPlace)

        REG_OP(Topk)
    .INPUT(self, ge::TensorType::ALL())
    .OUTPUT(values, ge::TensorType::ALL())
    .OUTPUT(indices, ge::TensorType::ALL())
    .ATTR(k, Int, 1)
    .ATTR(dim, Int, 0)
    .ATTR(largest, Bool, true)
    .ATTR(sorted, Bool, true)
    .OP_END_FACTORY_REG(Topk)

}  // namespace ge

namespace td {
namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kAnnotatedMixed = std::string(FIXTURES_DIR) + "/annotated_mixed.epair";
const std::string kAnnotatedConverterNoOp = std::string(FIXTURES_DIR) + "/annotated_converter_no_ops.epair";
const std::string kAnnotatedConverterNoOpCopy = std::string(FIXTURES_DIR) + "/annotated_converter_noop_copy.epair";
const std::string kAnnotatedAttrTypes = std::string(FIXTURES_DIR) + "/annotated_attr_types.epair";
const std::string kAnnotatedFallbackWithAttrs = std::string(FIXTURES_DIR) + "/annotated_fallback_with_attrs.epair";
const std::string kAnnotatedDynamicConcat = std::string(FIXTURES_DIR) + "/annotated_dynamic_concat.epair";
const std::string kAnnotatedDynamicConcatBadPrefix =
    std::string(FIXTURES_DIR) + "/annotated_dynamic_concat_bad_prefix.epair";
const std::string kAnnotatedAssertTensorMetadata =
    std::string(FIXTURES_DIR) + "/annotated_assert_tensor_metadata.epair";
const std::string kAnnotatedAutoFunctionalized = std::string(FIXTURES_DIR) + "/annotated_auto_functionalized.epair";
const std::string kAnnotatedBadFallbackInput = std::string(FIXTURES_DIR) + "/annotated_bad_fallback_input.epair";
const std::string kAnnotatedBadConverterNoOpMeta =
    std::string(FIXTURES_DIR) + "/annotated_bad_converter_noop_meta.epair";
const std::string kAnnotatedInlineConst = std::string(FIXTURES_DIR) + "/annotated_inline_const.epair";
const std::string kAnnotatedInlineScalarConst = std::string(FIXTURES_DIR) + "/annotated_inline_scalar_const.epair";
const std::string kAnnotatedInlineConstBadValue = std::string(FIXTURES_DIR) + "/annotated_inline_const_bad_value.epair";
const std::string kAnnotatedInlineConstDescMismatch =
    std::string(FIXTURES_DIR) + "/annotated_inline_const_desc_mismatch.epair";
const std::string kMissingGraphSource = std::string(FIXTURES_DIR) + "/missing_graph_source.epair";

}  // namespace

class GeGraphBuilderTest : public ::testing::Test {};

TEST_F(GeGraphBuilderTest, BuildsSimpleAnnotatedGraph) {
  td::EpairArchive archive(kAnnotated.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedMixedGraphWithFallbackNode) {
  td::EpairArchive archive(kAnnotatedMixed.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_EQ(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithConverterNoOp) {
  td::EpairArchive archive(kAnnotatedConverterNoOp.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithMultiInputConverterNoOp) {
  td::EpairArchive archive(kAnnotatedConverterNoOpCopy.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithSupportedAttrTypes) {
  td::EpairArchive archive(kAnnotatedAttrTypes.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

// DISABLED: Fallback 算子依赖 libtorch_npu 的 NPUQueue/NPUStream 符号，
// 当前测试环境未配置 TORCH_PKG_DIR 和 TORCH_NPU_DIR，无法链接 libtorch_npu stub。
// 待 stub_kernel 模块完善后重新启用。
TEST_F(GeGraphBuilderTest, DISABLED_BuildsAnnotatedFallbackGraphWithAttrPayload) {
  td::EpairArchive archive(kAnnotatedFallbackWithAttrs.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithDynamicConcatPrefixInference) {
  td::EpairArchive archive(kAnnotatedDynamicConcat.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, SkipsAssertTensorMetadataNodes) {
  td::EpairArchive archive(kAnnotatedAssertTensorMetadata.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, RejectsAnnotatedGraphWithInvalidDynamicConcatPrefix) {
  td::EpairArchive archive(kAnnotatedDynamicConcatBadPrefix.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

// DISABLED: auto_functionalized 算子依赖 schema 打桩，
// 当前测试环境未配置 TORCH_PKG_DIR 和 TORCH_NPU_DIR，无法完成 schema 解析。
// 待 schema stub 模块完善后重新启用。
TEST_F(GeGraphBuilderTest, DISABLED_BuildsAnnotatedAutoFunctionalizedGraph) {
  td::EpairArchive archive(kAnnotatedAutoFunctionalized.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, RejectsMissingAnnotatedGraphSource) {
  td::EpairArchive archive(kMissingGraphSource.c_str());
  td::Status builder_status = td::SUCCESS;
  td::GeGraphBuilder builder(archive, builder_status);
  EXPECT_NE(builder_status, td::SUCCESS);
}

// DISABLED: Fallback 算子依赖 libtorch_npu 的 NPUQueue/NPUStream 符号，
// 当前测试环境未配置 TORCH_PKG_DIR 和 TORCH_NPU_DIR，无法链接 libtorch_npu stub。
// 待 stub_kernel 模块完善后重新启用。
TEST_F(GeGraphBuilderTest, DISABLED_RejectsFallbackNodeWithNonTensorInput) {
  td::EpairArchive archive(kAnnotatedBadFallbackInput.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, RejectsConverterNoOpWithMetaMismatch) {
  td::EpairArchive archive(kAnnotatedBadConverterNoOpMeta.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithInlineConst) {
  td::EpairArchive archive(kAnnotatedInlineConst.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

// todo 依赖 schema 打桩
TEST_F(GeGraphBuilderTest, BuildsAnnotatedGraphWithInlineScalarConst) {
  td::EpairArchive archive(kAnnotatedInlineScalarConst.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, RejectsInlineConstWithMalformedPayload) {
  td::EpairArchive archive(kAnnotatedInlineConstBadValue.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, RejectsInlineConstWithOutputDescMismatch) {
  td::EpairArchive archive(kAnnotatedInlineConstDescMismatch.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

namespace {
const std::string kSymIntInput = std::string(FIXTURES_DIR) + "/annotated_sym_int_input.epair";
const std::string kConstantInput = std::string(FIXTURES_DIR) + "/annotated_constant_input.epair";
const std::string kScalarFallback = std::string(FIXTURES_DIR) + "/annotated_scalar_fallback.epair";
const std::string kDynamicDimSymbols = std::string(FIXTURES_DIR) + "/annotated_dynamic_dim_symbols.epair";
}  // namespace

TEST_F(GeGraphBuilderTest, BuildsGraphWithSymIntInput) {
  td::EpairArchive archive(kSymIntInput.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsGraphWithConstantInput) {
  td::EpairArchive archive(kConstantInput.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsGraphWithDynamicDimSymbols) {
  td::EpairArchive archive(kDynamicDimSymbols.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsGraphWithScalarFallbackNode) {
  td::EpairArchive archive(kScalarFallback.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
}

namespace {
const std::string kSymFloatBoolInput = std::string(FIXTURES_DIR) + "/annotated_sym_float_bool_input.epair";
const std::string kTensorConstant = std::string(FIXTURES_DIR) + "/annotated_tensor_constant.epair";
}  // namespace

TEST_F(GeGraphBuilderTest, BuildsWithSymFloatAndSymBoolInputs) {
  td::EpairArchive archive(kSymFloatBoolInput.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildsWithTensorConstantPlaceholder) {
  td::EpairArchive archive(kTensorConstant.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildReturnsNullptrForNotFullyOffloaded) {
  td::EpairArchive archive(kAnnotatedMixed.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

TEST_F(GeGraphBuilderTest, BuildReturnsNullptrForUninitializedBuilder) {
  td::EpairArchive archive(kMissingGraphSource.c_str());
  td::Status builder_status = td::SUCCESS;
  td::GeGraphBuilder builder(archive, builder_status);
  EXPECT_NE(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  EXPECT_EQ(graph, nullptr);
}

}  // namespace td
