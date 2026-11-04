/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "executorch_npu/state.h"

#include <cassert>
#include <cstdint>

int main() {
  using executorch_npu::ReleaseStateNamespace;
  using executorch_npu::ResetState;
  using executorch_npu::SetStateInitializer;
  using executorch_npu::StateStatus;
  using executorch_npu::StateTensorData;

  const float value[] = {1.0F, 2.0F};
  const int64_t shape[] = {2};
  const StateTensorData initializer{value, sizeof(value), "float32", shape, 1};

  assert(SetStateInitializer(-1, "request", "kv", initializer) == StateStatus::kInvalidArgument);
  assert(SetStateInitializer(0, "", "kv", initializer) == StateStatus::kInvalidArgument);
  assert(SetStateInitializer(0, "request", "kv", initializer) == StateStatus::kOk);
  assert(SetStateInitializer(0, "request", "kv", initializer) == StateStatus::kInvalidState);
  assert(ResetState(0, "request", "missing", initializer) == StateStatus::kNotFound);
  assert(ReleaseStateNamespace(0, "request") == StateStatus::kOk);
  assert(ReleaseStateNamespace(0, "request") == StateStatus::kNotFound);

  const int64_t scalar_value = 0;
  const StateTensorData scalar_initializer{&scalar_value, sizeof(scalar_value), "int64", nullptr, 0};
  assert(SetStateInitializer(0, "scalar-request", "counter", scalar_initializer) == StateStatus::kOk);
  assert(ReleaseStateNamespace(0, "scalar-request") == StateStatus::kOk);
  return 0;
}
