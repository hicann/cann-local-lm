#include <gtest/gtest.h>

#include <string>

#include "common/logging/logging.h"
#include "common/torch_schema_helper.h"

namespace td {

class TorchSchemaHelperTest : public ::testing::Test {
 protected:
  TorchSchemaHelper helper_;
  ValidationResult result_;
};

TEST_F(TorchSchemaHelperTest, QueryAtenAdd) {
  TorchSchemaInfo info = helper_.Query("aten::add", "Tensor", result_, "test");
  ASSERT_TRUE(result_.ok);
  EXPECT_EQ(info.namespace_name, "aten");
  EXPECT_EQ(info.op_name, "add");
  EXPECT_FALSE(info.is_inplace);
  EXPECT_FALSE(info.is_out);
  ASSERT_GE(info.arguments.size(), 2u);
  EXPECT_TRUE(info.arguments[0].is_tensor);
  EXPECT_TRUE(info.arguments[1].is_tensor);
}

TEST_F(TorchSchemaHelperTest, QueryAtenMm) {
  TorchSchemaInfo info = helper_.Query("aten::mm", "", result_, "test");
  ASSERT_TRUE(result_.ok);
  EXPECT_EQ(info.namespace_name, "aten");
  EXPECT_EQ(info.op_name, "mm");
  ASSERT_GE(info.arguments.size(), 2u);
  EXPECT_EQ(info.arguments[0].name, "self");
  EXPECT_EQ(info.arguments[1].name, "mat2");
}

TEST_F(TorchSchemaHelperTest, QueryAtenAddInplace) {
  TorchSchemaInfo info = helper_.Query("aten::add_", "Tensor", result_, "test");
  ASSERT_TRUE(result_.ok);
  EXPECT_TRUE(info.is_inplace);
}

TEST_F(TorchSchemaHelperTest, MissingSchemaReportsError) {
  TorchSchemaInfo info = helper_.Query("aten::nonexistent_op_12345", "", result_, "test");
  EXPECT_FALSE(result_.ok);
}

}  // namespace td
