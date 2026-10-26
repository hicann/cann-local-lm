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
#include <vector>
#include "common/logging/assert_macros.h"
#include "torch_delegate/td_types.h"
#include "cann_stub_control.h"

class AssertMacrosTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cann_stub::Reset();
  }
  void TearDown() override {
    cann_stub::Reset();
  }
};

TEST_F(AssertMacrosTest, AssertNotNullSuccess) {
  auto test_func = []() -> bool {
    int value = 42;
    int *ptr = &value;
    TD_ASSERT_NOTNULL(ptr);
    return true;
  };
  EXPECT_TRUE(test_func());
}

TEST_F(AssertMacrosTest, AssertNotNullFailureReturnsFalse) {
  auto test_func = []() -> bool {
    int *ptr = nullptr;
    TD_ASSERT_NOTNULL(ptr, "Null pointer test");
    return true;
  };
  EXPECT_FALSE(test_func());
}

TEST_F(AssertMacrosTest, AssertNotNullFailureReturnsNullPtr) {
  auto test_func = []() -> int * {
    int *ptr = nullptr;
    TD_ASSERT_NOTNULL(ptr);
    return new int(42);
  };
  EXPECT_EQ(test_func(), nullptr);
}

TEST_F(AssertMacrosTest, AssertNotNullFailureReturnsNullUniquePtr) {
  auto test_func = []() -> std::unique_ptr<int> {
    int *ptr = nullptr;
    TD_ASSERT_NOTNULL(ptr);
    return std::make_unique<int>(42);
  };
  EXPECT_EQ(test_func(), nullptr);
}

TEST_F(AssertMacrosTest, AssertTrueSuccess) {
  auto test_func = []() -> bool {
    TD_ASSERT_TRUE(true);
    return true;
  };
  EXPECT_TRUE(test_func());
}

TEST_F(AssertMacrosTest, AssertTrueFailureReturnsFalse) {
  auto test_func = []() -> bool {
    TD_ASSERT_TRUE(false, "Condition test");
    return true;
  };
  EXPECT_FALSE(test_func());
}

TEST_F(AssertMacrosTest, AssertTrueFailureReturnsStatus) {
  auto test_func = []() -> td::Status {
    TD_ASSERT_TRUE(false);
    return td::SUCCESS;
  };
  td::Status status = test_func();
  EXPECT_NE(status, td::SUCCESS);
  EXPECT_EQ(status, td::INVALID_PARAM);
}

TEST_F(AssertMacrosTest, AssertTrueFailureReturnsEmptyVector) {
  auto test_func = []() -> std::vector<int> {
    TD_ASSERT_TRUE(false);
    return {1, 2, 3};
  };
  EXPECT_TRUE(test_func().empty());
}

TEST_F(AssertMacrosTest, AssertTrueFailureReturnsEmptyString) {
  auto test_func = []() -> std::string {
    TD_ASSERT_TRUE(false);
    return "success";
  };
  EXPECT_TRUE(test_func().empty());
}

TEST_F(AssertMacrosTest, AssertGenericSuccess) {
  auto test_func = []() -> bool {
    TD_ASSERT(1 > 0);
    return true;
  };
  EXPECT_TRUE(test_func());
}

TEST_F(AssertMacrosTest, AssertGenericFailureWithMessage) {
  auto test_func = []() -> bool {
    TD_ASSERT(1 < 0, "Math error: %d < %d", 1, 0);
    return true;
  };
  EXPECT_FALSE(test_func());
}

TEST_F(AssertMacrosTest, ErrorResultConvertsToBool) {
  td::ErrorResult err;
  EXPECT_FALSE(static_cast<bool>(err));
}

TEST_F(AssertMacrosTest, ErrorResultConvertsToNullptr) {
  td::ErrorResult err;
  int *ptr = err;
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(AssertMacrosTest, ErrorResultConvertsToNullUniquePtr) {
  td::ErrorResult err;
  std::unique_ptr<int> uptr = err;
  EXPECT_EQ(uptr, nullptr);
}

TEST_F(AssertMacrosTest, ErrorResultConvertsToEmptyVector) {
  td::ErrorResult err;
  std::vector<int> vec = err;
  EXPECT_TRUE(vec.empty());
}

TEST_F(AssertMacrosTest, ErrorResultConvertsToEmptyString) {
  td::ErrorResult err;
  std::string str = err;
  EXPECT_TRUE(str.empty());
}

TEST_F(AssertMacrosTest, CreateErrorMsgWithFormat) {
  std::string msg = td::CreateErrorMsg("Test %d: %s", 42, "hello");
  EXPECT_EQ(msg, "Test 42: hello");
}

TEST_F(AssertMacrosTest, CreateErrorMsgEmpty) {
  std::string msg = td::CreateErrorMsg();
  EXPECT_TRUE(msg.empty());
}
