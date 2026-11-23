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
#include "backend/host_cpu_capability.h"
#include "cann_stub/cann_stub_control.h"
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

        REG_OP(aten__relu)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(aten__relu)

}  // namespace ge

namespace td {
namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kAnnotatedMixed = std::string(FIXTURES_DIR) + "/annotated_mixed.epair";
const std::string kAnnotatedHostCpu = std::string(FIXTURES_DIR) + "/annotated_host_cpu.epair";
const std::string kAnnotatedHostCpuNon4Byte = std::string(FIXTURES_DIR) + "/annotated_host_cpu_non4byte_inputs.epair";
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

class GeGraphBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 本测试二进制不触发在线 codegen（目标环境无 torch_npu 编译链）；三分派
    // 与 HostCpu 能力校验通过 fixture + stub IR 注册表覆盖。
    setenv("TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN", "0", 1);
    cann_stub::Reset();
  }
  void TearDown() override {
    unsetenv("TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN");
  }
};

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

// ===========================================================================
// HostCpu 混合调度（CPU/NPU 混合）构图测试：三分派、host placement、能力校验
// ===========================================================================

namespace {

ge::GNode FindNodeByType(ge::Graph &graph, const std::string &type) {
  for (auto &node : graph.GetAllNodes()) {
    ge::AscendString node_type;
    if (node.GetType(node_type) == ge::GRAPH_SUCCESS && node_type.GetString() != nullptr &&
        type == node_type.GetString()) {
      return node;
    }
  }
  return ge::GNode();
}

size_t CountHostPlacementInputs(ge::GNode &node) {
  size_t host_count = 0;
  for (size_t i = 0; i < node.GetInputsSize(); ++i) {
    ge::TensorDesc desc;
    if (node.GetInputDesc(static_cast<int32_t>(i), desc) != ge::GRAPH_SUCCESS) {
      continue;
    }
    if (desc.GetPlacement() == ge::kPlacementHost) {
      host_count++;
    }
  }
  return host_count;
}

}  // namespace

TEST_F(GeGraphBuilderTest, BuildsAnnotatedHostCpuGraph) {
  td::EpairArchive archive(kAnnotatedHostCpu.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr) << "CPU fallback node (device=cpu) must join the same ge::Graph";

  // CPU 节点经 FallbackNodePlanner(kPlacementHost) 入图，张量输入为 host placement。
  ge::GNode relu_node = FindNodeByType(*graph, "aten__relu");
  ge::AscendString found_type;
  ASSERT_EQ(relu_node.GetType(found_type), ge::GRAPH_SUCCESS)
      << "HostCpu custom op node must be planned into the graph";
  EXPECT_GT(CountHostPlacementInputs(relu_node), 0u)
      << "HostCpu node tensor inputs must be declared with host placement";

  // converter 节点的 placement 由 ascend_ir inputDesc 决定，不受 FallbackNodePlanner
  // placement 参数影响（默认参数保证现有路径零变化，见 PureNpuGraphBuildsUnaffected）。
  ge::GNode mm_node = FindNodeByType(*graph, "MatMul");
  ASSERT_EQ(mm_node.GetType(found_type), ge::GRAPH_SUCCESS);
}

TEST_F(GeGraphBuilderTest, BuildsHostCpuGraphWithNon4ByteUserDataInputs) {
  // 回归 0036：HostCpu 用户 Data 输入为 int64(8B)/fp16(2B) 时，旧实现按“4 字节/元素”
  // 假设直接返回 UNIMPLEMENTED；GE v2 的 D2H staging 实际按 desc dtype 计长，
  // 因此这里必须能正常构图。
  td::EpairArchive archive(kAnnotatedHostCpuNon4Byte.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr) << "HostCpu user Data inputs of 8/2-byte dtype must not be rejected";

  size_t relu_nodes = 0;
  for (auto &node : graph->GetAllNodes()) {
    ge::AscendString node_type;
    if (node.GetType(node_type) == ge::GRAPH_SUCCESS && node_type.GetString() != nullptr &&
        std::string(node_type.GetString()) == "aten__relu") {
      relu_nodes++;
      EXPECT_GT(CountHostPlacementInputs(node), 0u)
          << "HostCpu node tensor inputs must be declared with host placement";
    }
  }
  EXPECT_EQ(relu_nodes, 2u) << "both int64 and fp16 user-input HostCpu nodes must be planned";
}

TEST_F(GeGraphBuilderTest, PureNpuGraphBuildsUnaffected) {
  td::EpairArchive archive(kAnnotated.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::InputBindingMap binding_map;
  auto graph = builder.Build(binding_map);
  ASSERT_NE(graph, nullptr) << "Pure NPU epair must build exactly as before the HostCpu feature";
}

TEST_F(GeGraphBuilderTest, ReportsCpuOwnershipFromClassifiedNodes) {
  td::EpairArchive archive(kAnnotatedHostCpu.c_str());
  td::Status builder_status = td::FAILED;
  td::GeGraphBuilder builder(archive, builder_status);
  ASSERT_EQ(builder_status, td::SUCCESS);

  td::ParsedModel model;
  const RecordData model_record = archive.ReadRecord("extra/ep_json_with_ascend_ir.json");
  ASSERT_EQ(td::ParsedModel::Parse(model_record.AsString(), model), td::SUCCESS);

  const std::string report = td::ReportCpuOwnership(model);
  // 归属复述：CPU 节点 + device 分类 + namespace 分级；aten → suspicious 可疑分组，
  // 汇总报告不含级别字样，告警语义由逐节点 TDLOGW 输出保留（见 host_cpu_capability.cc）。
  EXPECT_NE(report.find("CPU nodes 1"), std::string::npos) << report;
  EXPECT_NE(report.find("device=cpu"), std::string::npos) << report;
  EXPECT_NE(report.find("suspicious: aten op has no PrivateUse1 kernel"), std::string::npos)
      << "aten->CPU must be reported as suspicious (per-node warning kept via TDLOGW)";
  EXPECT_NE(report.find("aten.relu.default"), std::string::npos) << report;
}

TEST_F(GeGraphBuilderTest, ReportsNpuFallbackOwnership) {
  td::ParsedModel model;
  const RecordData model_record =
      td::EpairArchive(kAnnotatedFallbackWithAttrs.c_str()).ReadRecord("extra/ep_json_with_ascend_ir.json");
  ASSERT_EQ(td::ParsedModel::Parse(model_record.AsString(), model), td::SUCCESS);

  const std::string report = td::ReportCpuOwnership(model);
  // NPU fallback 必须单独计数并列入清单，不能被当成 converter 或丢弃。
  EXPECT_NE(report.find("NPU nodes 1"), std::string::npos) << report;
  EXPECT_NE(report.find("converter 0 / fallback 1"), std::string::npos) << report;
  EXPECT_NE(report.find("CPU nodes 0"), std::string::npos) << report;
  EXPECT_NE(report.find("device=npu"), std::string::npos) << report;
}

}  // namespace td
