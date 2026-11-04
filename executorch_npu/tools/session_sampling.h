/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace executorch_npu::session_sampling {

inline bool ShouldStop(int64_t token, bool has_eos_token, int eos_token) {
  return has_eos_token && token == eos_token;
}

inline bool SampleToken(const std::vector<float> &logits, float temperature, size_t top_k, std::mt19937_64 *generator,
                        int64_t *token) {
  if (logits.empty() || !std::isfinite(temperature) || temperature <= 0 || generator == nullptr || token == nullptr) {
    return false;
  }
  for (float value : logits) {
    if (!std::isfinite(value)) {
      return false;
    }
  }

  std::vector<size_t> indices(logits.size());
  for (size_t index = 0; index < indices.size(); ++index) {
    indices[index] = index;
  }
  if (top_k != 0 && top_k < indices.size()) {
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(), [&logits](size_t left, size_t right) {
      return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
    });
    indices.resize(top_k);
  }

  float maximum = -std::numeric_limits<float>::infinity();
  for (size_t index : indices) {
    maximum = std::max(maximum, logits[index]);
  }
  std::vector<double> probabilities;
  probabilities.reserve(indices.size());
  double total = 0;
  for (size_t index : indices) {
    const double probability = std::exp((static_cast<double>(logits[index]) - maximum) / temperature);
    probabilities.push_back(probability);
    total += probability;
  }
  if (!std::isfinite(total) || total <= 0) {
    return false;
  }

  // Build a reproducible [0, 1) value directly from mt19937_64 bits.
  const double threshold = std::ldexp(static_cast<double>((*generator)() >> 11), -53) * total;
  double cumulative = 0;
  for (size_t index = 0; index < indices.size(); ++index) {
    cumulative += probabilities[index];
    if (threshold < cumulative) {
      *token = static_cast<int64_t>(indices[index]);
      return true;
    }
  }
  *token = static_cast<int64_t>(indices.back());
  return true;
}

}  // namespace executorch_npu::session_sampling
