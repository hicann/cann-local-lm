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

#include <utility>

#include "backend/aten_schema_resolver.h"
#include "backend/aten_tensor_desc_mapper.h"
#include "backend/value_registry.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace {

td::RegisteredValue DummyValue() {
  return {ge::GNode(), 0};
}

}  // namespace
namespace td {
class GeGraphBuilderHelpersTest : public ::testing::Test {};

TEST_F(GeGraphBuilderHelpersTest, ParsesDispatcherTargetAndFallbackType) {
  td::AtenSchemaResolver resolver;

  td::AtenTargetInfo target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.aten.add.Tensor", "add", target), SUCCESS);
  td::ParsedNode node;
  node.name = "add";
  node.effective_schema_target = "torch.ops.aten.add.Tensor";
  td::ParsedNodeInput self_input;
  self_input.name = "self";
  self_input.kind = 1;
  self_input.input_kind = td::NodeInputKind::Tensor;
  self_input.tensor_name = "x";
  node.inputs.push_back(self_input);
  td::ParsedNodeInput other_input;
  other_input.name = "other";
  other_input.kind = 1;
  other_input.input_kind = td::NodeInputKind::Tensor;
  other_input.tensor_name = "y";
  node.inputs.push_back(other_input);
  td::AtenSchemaInfo schema;
  EXPECT_EQ(resolver.Resolve(node, schema), SUCCESS);
  EXPECT_EQ(target.namespace_name, "aten");
  EXPECT_EQ(target.op_name, "add");
  EXPECT_EQ(target.overload, "Tensor");
  EXPECT_EQ(target.dispatcher_name, "aten::add");
  ASSERT_EQ(schema.arguments.size(), 2u);
  EXPECT_EQ(schema.arguments[0].name, "self");
  EXPECT_EQ(schema.arguments[1].name, "other");
  EXPECT_EQ(resolver.MakeFallbackOpType(schema.target), "aten__add_Tensor");
}

TEST_F(GeGraphBuilderHelpersTest, BuildsInplaceAndOutFallbackType) {
  td::AtenSchemaResolver resolver;

  td::AtenTargetInfo inplace_target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.aten.relu_.default", "relu_", inplace_target), SUCCESS);
  td::AtenTargetInfo out_target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.aten.add.out", "add.out", out_target), SUCCESS);
  EXPECT_EQ(resolver.MakeFallbackOpType(inplace_target), "aten__relu_");
  EXPECT_EQ(resolver.MakeFallbackOpType(out_target), "aten__add_out");
}

TEST_F(GeGraphBuilderHelpersTest, MapsPt2DenseTensorToGeNd) {
  td::AtenTensorDescMapper mapper;
  td::ParsedTensorMeta meta;
  meta.dtype = 7;
  meta.layout = 7;
  meta.sizes = {2, 3};

  td::GeTensorDescInfo desc;
  ASSERT_EQ(mapper.MapPt2TensorMeta(meta, "x", desc), SUCCESS);
  EXPECT_EQ(desc.dtype, ge::DT_FLOAT);
  EXPECT_EQ(desc.format, ge::FORMAT_ND);
  ASSERT_EQ(desc.shape.size(), 2u);
  EXPECT_EQ(desc.shape[0], 2);
  EXPECT_EQ(desc.shape[1], 3);
}

TEST_F(GeGraphBuilderHelpersTest, ResolvesProducerAndAliasNames) {
  td::ValueRegistry registry;

  ASSERT_EQ(registry.RegisterProducerOutput("MatMul:0", DummyValue()), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("mm", "MatMul:0"), SUCCESS);
  EXPECT_NE(registry.Resolve("MatMul:0"), nullptr);
  EXPECT_NE(registry.Resolve("mm"), nullptr);
  EXPECT_NE(registry.Resolve("mm:0"), nullptr);
}

TEST_F(GeGraphBuilderHelpersTest, ResolvesChainedAliases) {
  td::ValueRegistry registry;

  ASSERT_EQ(registry.RegisterProducerOutput("Transpose:0", DummyValue()), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("permute", "Transpose:0"), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("expand", "permute"), SUCCESS);
  EXPECT_NE(registry.Resolve("expand"), nullptr);
  EXPECT_NE(registry.Resolve("expand:0"), nullptr);
}

TEST_F(GeGraphBuilderHelpersTest, ResolvesFallbackDummyMappingAlias) {
  td::ValueRegistry registry;

  ASSERT_EQ(registry.RegisterProducerOutput("RopeInPlace:0", DummyValue()), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("rope_in_place_ge_op:0", "RopeInPlace:0"), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("rope_out", "RopeInPlace:0"), SUCCESS);
  EXPECT_NE(registry.Resolve("RopeInPlace:0"), nullptr);
  EXPECT_NE(registry.Resolve("rope_in_place_ge_op:0"), nullptr);
  EXPECT_NE(registry.Resolve("rope_out"), nullptr);
}

TEST_F(GeGraphBuilderHelpersTest, ReportsAliasConflict) {
  td::ValueRegistry registry;

  ASSERT_EQ(registry.RegisterProducerOutput("Add:0", DummyValue()), SUCCESS);
  ASSERT_EQ(registry.RegisterProducerOutput("Mul:0", DummyValue()), SUCCESS);
  ASSERT_EQ(registry.RegisterAlias("out", "Add:0"), SUCCESS);
  EXPECT_NE(registry.RegisterAlias("out", "Mul:0"), SUCCESS);
}

TEST_F(GeGraphBuilderHelpersTest, MakeFallbackOpTypeWithEmptyOverload) {
  td::AtenSchemaResolver resolver;

  td::AtenTargetInfo target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.aten.relu_.default", "relu_", target), SUCCESS);
  EXPECT_EQ(target.overload, "");
  EXPECT_EQ(resolver.MakeFallbackOpType(target), "aten__relu_");
}

TEST_F(GeGraphBuilderHelpersTest, MakeFallbackOpTypeWithNonAtenNamespace) {
  td::AtenSchemaResolver resolver;

  td::AtenTargetInfo target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.my_ops.scale_add.default", "scale_add", target), SUCCESS);
  EXPECT_EQ(target.namespace_name, "my_ops");
  EXPECT_EQ(target.op_name, "scale_add");
  EXPECT_EQ(resolver.MakeFallbackOpType(target), "my_ops__scale_add");
}

TEST_F(GeGraphBuilderHelpersTest, MakeFallbackOpTypeWithEmptyNamespace) {
  td::AtenSchemaResolver resolver;
  td::AtenTargetInfo target;
  target.namespace_name = "";
  target.op_name = "custom_op";
  target.overload = "v1";

  std::string result = resolver.MakeFallbackOpType(target);
  EXPECT_EQ(result, "custom_op_v1");
}

TEST_F(GeGraphBuilderHelpersTest, ParseTargetWithCustomNamespace) {
  td::AtenSchemaResolver resolver;

  td::AtenTargetInfo target;
  EXPECT_EQ(resolver.ParseTarget("torch.ops.xllm_ascend.rope_in_place.default", "rope", target), SUCCESS);
  EXPECT_EQ(target.namespace_name, "xllm_ascend");
  EXPECT_EQ(target.op_name, "rope_in_place");
  EXPECT_EQ(target.overload, "");
  EXPECT_EQ(target.dispatcher_name, "xllm_ascend::rope_in_place");
}

TEST_F(GeGraphBuilderHelpersTest, ResolveWithScalarInputs) {
  td::AtenSchemaResolver resolver;

  td::ParsedNode node;
  node.name = "add_scalar";
  node.effective_schema_target = "torch.ops.aten.add.Scalar";
  node.schema_metadata_input_count = 0;

  td::ParsedNodeInput self_input;
  self_input.name = "self";
  self_input.kind = 1;
  self_input.input_kind = td::NodeInputKind::Tensor;
  self_input.tensor_name = "x";
  node.inputs.push_back(self_input);

  td::ParsedNodeInput other_input;
  other_input.name = "other";
  other_input.kind = 1;
  other_input.input_kind = td::NodeInputKind::ScalarInt;
  node.inputs.push_back(other_input);

  td::AtenSchemaInfo schema;
  ASSERT_NE(resolver.Resolve(node, schema), SUCCESS);
}

TEST_F(GeGraphBuilderHelpersTest, ResolveWithSymInputs) {
  td::AtenSchemaResolver resolver;

  td::ParsedNode node;
  node.name = "custom_op";
  node.effective_schema_target = "torch.ops.my_ops.custom.default";
  node.schema_metadata_input_count = 0;

  td::ParsedNodeInput tensor_input;
  tensor_input.name = "x";
  tensor_input.kind = 1;
  tensor_input.input_kind = td::NodeInputKind::Tensor;
  tensor_input.tensor_name = "x";
  node.inputs.push_back(tensor_input);

  td::ParsedNodeInput sym_input;
  sym_input.name = "batch_size";
  sym_input.kind = 1;
  sym_input.input_kind = td::NodeInputKind::SymInt;
  node.inputs.push_back(sym_input);

  td::AtenSchemaInfo schema;
  ASSERT_NE(resolver.Resolve(node, schema), SUCCESS);
}
}  // namespace td
