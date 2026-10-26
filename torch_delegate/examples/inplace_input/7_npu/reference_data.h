/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_INPUT_REFERENCE_DATA_H
#define INPLACE_INPUT_REFERENCE_DATA_H

#include <vector>
#include <string>

namespace inplace {

enum class ModelType {
  Unknown = 0,
  Simple = 1,
  Chain = 2,
  Param = 3,
};

struct ModelConfig {
  std::string name;
  ModelType type;
  float scale;
  float bias;
  std::vector<int64_t> input_shape_x;
  std::vector<int64_t> input_shape_y;

  std::vector<std::vector<int64_t>> GetInputShapes() const {
    if (type == ModelType::Simple) {
      return {input_shape_x, input_shape_y};
    }
    return {input_shape_x};
  }
};

ModelType ParseModelType(const std::string &name);
ModelConfig GetModelConfig(ModelType type);

/**
 * 计算参考输出
 *
 * @param model_type 模型类型
 * @param inputs 输入数据指针数组 (input_buffers.data() for each input)
 * @param output 输出数据缓冲区
 * @param count 元素数量
 * @param scale 缩放系数
 * @param bias 偏置值
 */
void ComputeReference(ModelType model_type, const std::vector<const float *> &inputs, float *output, size_t count,
                      float scale, float bias);

}  // namespace inplace

#endif  // INPLACE_INPUT_REFERENCE_DATA_H
