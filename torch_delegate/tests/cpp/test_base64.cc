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

#include <cstdint>
#include <string>
#include <vector>

#include "common/base64.h"

TEST(Base64Test, DecodesPlainString) {
  std::string decoded;
  ASSERT_TRUE(td::internal::DecodeBase64ToString("SGVsbG8gV29ybGQ=", decoded));
  EXPECT_EQ(decoded, "Hello World");
}

TEST(Base64Test, DecodesBinaryBytes) {
  std::vector<uint8_t> decoded;
  ASSERT_TRUE(td::internal::DecodeBase64("QlNI", decoded));
  ASSERT_EQ(decoded.size(), 3u);
  EXPECT_EQ(decoded[0], 'B');
  EXPECT_EQ(decoded[1], 'S');
  EXPECT_EQ(decoded[2], 'H');
}

TEST(Base64Test, RejectsInvalidPayload) {
  std::string decoded;
  EXPECT_FALSE(td::internal::DecodeBase64ToString("not_base64***", decoded));
}
