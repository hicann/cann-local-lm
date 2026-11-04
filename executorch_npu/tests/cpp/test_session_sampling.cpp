/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "session_sampling.h"

int main() {
  assert(executorch_npu::session_sampling::ShouldStop(42, true, 42));
  assert(!executorch_npu::session_sampling::ShouldStop(42, true, 41));
  assert(!executorch_npu::session_sampling::ShouldStop(42, false, 42));

  const std::vector<float> logits = {1.0F, 2.0F, 3.0F};
  std::mt19937_64 first(1234);
  std::mt19937_64 second(1234);
  for (int index = 0; index < 10; ++index) {
    int64_t first_token = -1;
    int64_t second_token = -1;
    assert(executorch_npu::session_sampling::SampleToken(logits, 0.7F, 0, &first, &first_token));
    assert(executorch_npu::session_sampling::SampleToken(logits, 0.7F, 0, &second, &second_token));
    assert(first_token == second_token);
  }

  std::mt19937_64 generator(1);
  int64_t token = -1;
  assert(executorch_npu::session_sampling::SampleToken(logits, 1.0F, 1, &generator, &token));
  assert(token == 2);
  assert(!executorch_npu::session_sampling::SampleToken({0.0F, std::numeric_limits<float>::quiet_NaN()}, 1.0F, 0,
                                                        &generator, &token));
  assert(!executorch_npu::session_sampling::SampleToken(logits, 0.0F, 0, &generator, &token));
  return 0;
}
