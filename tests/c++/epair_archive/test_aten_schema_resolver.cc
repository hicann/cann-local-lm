#include <gtest/gtest.h>

#include <string>

#include "backend/aten_schema_resolver.h"
#include "common/logging/logging.h"

namespace td {

class AtenSchemaResolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    result_ = ValidationResult{true, {}};
  }

  AtenSchemaResolver resolver_;
  ValidationResult result_;
};

TEST_F(AtenSchemaResolverTest, ParsesSimpleTarget) {
  AtenTargetInfo info = resolver_.ParseTarget(
      "torch.ops.aten.add.Tensor", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.namespace_name, "aten");
  EXPECT_EQ(info.op_name, "add");
  EXPECT_EQ(info.overload, "Tensor");
  EXPECT_EQ(info.dispatcher_name, "aten::add");
}

TEST_F(AtenSchemaResolverTest, ParsesDefaultOverloadAsEmpty) {
  AtenTargetInfo info = resolver_.ParseTarget(
      "torch.ops.aten.mm.default", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.namespace_name, "aten");
  EXPECT_EQ(info.op_name, "mm");
  EXPECT_EQ(info.overload, "");
}

TEST_F(AtenSchemaResolverTest, ParsesInplaceOp) {
  AtenTargetInfo info = resolver_.ParseTarget(
      "torch.ops.aten.add_.Tensor", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.op_name, "add_");
  EXPECT_EQ(info.overload, "Tensor");
}

TEST_F(AtenSchemaResolverTest, ParsesCustomNamespace) {
  AtenTargetInfo info = resolver_.ParseTarget(
      "torch.ops.xllm_ascend.rope_in_place.default", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.namespace_name, "xllm_ascend");
  EXPECT_EQ(info.op_name, "rope_in_place");
  EXPECT_EQ(info.overload, "");
}

TEST_F(AtenSchemaResolverTest, RejectsTargetWithoutDots) {
  AtenTargetInfo info = resolver_.ParseTarget(
      "invalid_target", result_, "test");
  EXPECT_FALSE(result_.ok);
}

TEST_F(AtenSchemaResolverTest, MakeFallbackOpType) {
  AtenTargetInfo info;
  info.namespace_name = "aten";
  info.op_name = "add";
  info.overload = "Tensor";
  EXPECT_EQ(resolver_.MakeFallbackOpType(info), "aten__add_Tensor");
}

TEST_F(AtenSchemaResolverTest, MakeFallbackOpTypeNoOverload) {
  AtenTargetInfo info;
  info.namespace_name = "aten";
  info.op_name = "mm";
  EXPECT_EQ(resolver_.MakeFallbackOpType(info), "aten__mm");
}

TEST_F(AtenSchemaResolverTest, MakeFallbackOpTypeCustomNamespace) {
  AtenTargetInfo info;
  info.namespace_name = "xllm_ascend";
  info.op_name = "rope_in_place";
  EXPECT_EQ(resolver_.MakeFallbackOpType(info), "xllm_ascend__rope_in_place");
}

TEST_F(AtenSchemaResolverTest, ResolveDetectsOutOverload) {
  ParsedNode node;
  node.effective_schema_target = "torch.ops.aten.add.out";
  node.json_node_index = 0;
  node.name = "add_out";
  node.schema_metadata_input_count = 0;

  AtenSchemaInfo info = resolver_.Resolve(node, result_);
  EXPECT_TRUE(result_.ok);
  EXPECT_TRUE(info.is_out);
  EXPECT_FALSE(info.is_inplace);
}

TEST_F(AtenSchemaResolverTest, ResolveDetectsInplaceOp) {
  ParsedNode node;
  node.effective_schema_target = "torch.ops.aten.add_.Tensor";
  node.json_node_index = 1;
  node.name = "add_inplace";
  node.schema_metadata_input_count = 0;

  ParsedNodeInput input;
  input.name = "self";
  input.input_kind = NodeInputKind::Tensor;
  input.tensor_name = "x";
  node.inputs.push_back(input);

  AtenSchemaInfo info = resolver_.Resolve(node, result_);
  EXPECT_TRUE(result_.ok);
  EXPECT_TRUE(info.is_inplace);
  EXPECT_FALSE(info.is_out);
}

}  // namespace td
