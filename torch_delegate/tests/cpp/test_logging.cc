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
#include "common/logging/logging.h"

TEST(LoggingTest, InfoLogBasic) {
  TDLOGI("Test INFO log message");
}

TEST(LoggingTest, ErrorLogBasic) {
  TDLOGE("Test ERROR log message");
}

TEST(LoggingTest, WarningLogBasic) {
  TDLOGW("Test WARNING log message");
}

TEST(LoggingTest, DebugLogBasic) {
  TDLOGD("Test DEBUG log message");
}

TEST(LoggingTest, InfoLogWithFormat) {
  const char *node_name = "test_node";
  TDLOGI("Node %s created successfully", node_name);
}

TEST(LoggingTest, ErrorLogWithFormat) {
  const char *error_msg = "test_error";
  TDLOGE("Failed to create graph: %s", error_msg);
}

TEST(LoggingTest, WarningLogWithFormat) {
  const char *param_name = "deprecated_param";
  TDLOGW("Deprecated parameter: %s", param_name);
}

TEST(LoggingTest, DebugLogWithFormat) {
  const char *node_name = "test_node";
  const char *node_type = "Add";
  TDLOGD("Processing node: %s, type: %s", node_name, node_type);
}

TEST(LoggingTest, MultipleArgs) {
  int value1 = 100;
  double value2 = 3.14;
  TDLOGI("Multiple args: int=%d, double=%f", value1, value2);
}
