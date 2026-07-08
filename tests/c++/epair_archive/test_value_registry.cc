#include <gtest/gtest.h>

#include <string>

#include "backend/value_registry.h"
#include "common/logging/logging.h"

namespace td {

class ValueRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    result_ = ValidationResult{true, {}};
  }

  ValidationResult result_;
  ValueRegistry registry_;
  ge::GNode dummy_node_;
};

TEST_F(ValueRegistryTest, RegisterAndResolvePlaceholder) {
  RegisteredValue value{dummy_node_, 0};
  ASSERT_TRUE(registry_.RegisterPlaceholderSource("input", value, result_));
  EXPECT_TRUE(result_.ok);

  EXPECT_TRUE(registry_.HasRegisteredRef("input"));
  EXPECT_TRUE(registry_.HasRegisteredRef("input:0"));

  const RegisteredValue *resolved = registry_.Resolve("input");
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->output_index, 0);
}

TEST_F(ValueRegistryTest, RegisterAndResolveProducerOutput) {
  RegisteredValue value{dummy_node_, 1};
  ASSERT_TRUE(registry_.RegisterProducerOutput("mm:0", value, result_));
  EXPECT_TRUE(result_.ok);

  EXPECT_TRUE(registry_.HasRegisteredRef("mm:0"));
  const RegisteredValue *resolved = registry_.Resolve("mm:0");
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->output_index, 1);
}

TEST_F(ValueRegistryTest, RejectsDuplicateKey) {
  RegisteredValue value{dummy_node_, 0};
  ASSERT_TRUE(registry_.RegisterProducerOutput("dup:0", value, result_));
  EXPECT_TRUE(result_.ok);

  ValidationResult dup_result{true, {}};
  EXPECT_FALSE(registry_.RegisterProducerOutput("dup:0", value, dup_result));
  EXPECT_FALSE(dup_result.ok);
}

TEST_F(ValueRegistryTest, ResolvesAlias) {
  RegisteredValue value{dummy_node_, 0};
  ASSERT_TRUE(registry_.RegisterProducerOutput("real_output:0", value, result_));
  ASSERT_TRUE(registry_.RegisterAlias("alias_name", "real_output:0", result_));
  EXPECT_TRUE(result_.ok);

  const RegisteredValue *resolved = registry_.Resolve("alias_name");
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->output_index, 0);
}

TEST_F(ValueRegistryTest, AliasWithoutPortAutoAppendsPortZero) {
  RegisteredValue value{dummy_node_, 0};
  ASSERT_TRUE(registry_.RegisterProducerOutput("target:0", value, result_));
  ASSERT_TRUE(registry_.RegisterAlias("alias_no_port", "target:0", result_));
  EXPECT_TRUE(result_.ok);

  EXPECT_TRUE(registry_.HasRegisteredRef("alias_no_port"));
  EXPECT_TRUE(registry_.HasRegisteredRef("alias_no_port:0"));
}

TEST_F(ValueRegistryTest, UnknownNameReturnsNull) {
  const RegisteredValue *resolved = registry_.Resolve("does_not_exist");
  EXPECT_EQ(resolved, nullptr);
  EXPECT_FALSE(registry_.HasRegisteredRef("does_not_exist"));
}

TEST_F(ValueRegistryTest, ChainedAliases) {
  RegisteredValue value{dummy_node_, 2};
  ASSERT_TRUE(registry_.RegisterProducerOutput("origin:0", value, result_));
  ASSERT_TRUE(registry_.RegisterAlias("mid", "origin:0", result_));
  ASSERT_TRUE(registry_.RegisterAlias("end", "mid", result_));
  EXPECT_TRUE(result_.ok);

  const RegisteredValue *resolved = registry_.Resolve("end");
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->output_index, 2);
}

}  // namespace td
