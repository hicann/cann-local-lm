/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reference_data.h"

#include <cmath>

namespace inplace {

ModelType ParseModelType(const std::string &name) {
  if (name == "simple") {
    return ModelType::Simple;
  }
  if (name == "chain") {
    return ModelType::Chain;
  }
  if (name == "param") {
    return ModelType::Param;
  }
  return ModelType::Unknown;
}

ModelConfig GetModelConfig(ModelType type) {
  ModelConfig config;
  config.input_shape_x = {4, 4};
  config.input_shape_y = {4, 4};

  switch (type) {
    case ModelType::Simple:
      config.name = "simple";
      config.type = ModelType::Simple;
      config.scale = 2.0f;
      config.bias = 0.0f;
      break;
    case ModelType::Chain:
      config.name = "chain";
      config.type = ModelType::Chain;
      config.scale = 2.0f;  // mul scale
      config.bias = 1.0f;   // add bias (first op)
      break;
    case ModelType::Param:
      config.name = "param";
      config.type = ModelType::Param;
      config.scale = 2.0f;
      config.bias = 1.0f;  // buffer bias value
      break;
    default:
      config.name = "unknown";
      config.type = ModelType::Unknown;
      break;
  }

  return config;
}

void ComputeReference(ModelType model_type, const std::vector<const float *> &inputs, float *output, size_t count,
                      float scale, float bias) {
  const float *x = inputs[0];

  switch (model_type) {
    case ModelType::Simple:
      // out = (x + y) * scale
      {
        const float *y = inputs[1];
        for (size_t i = 0; i < count; ++i) {
          output[i] = (x[i] + y[i]) * scale;
        }
      }
      break;

    case ModelType::Chain:
      // out = ((x + 1) * 2) / 4
      for (size_t i = 0; i < count; ++i) {
        output[i] = ((x[i] + 1.0f) * 2.0f) / 4.0f;
      }
      break;

    case ModelType::Param:
      // out = (x + bias) * scale
      for (size_t i = 0; i < count; ++i) {
        output[i] = (x[i] + bias) * scale;
      }
      break;

    default:
      // Copy input
      for (size_t i = 0; i < count; ++i) {
        output[i] = x[i];
      }
      break;
  }
}

}  // namespace inplace
