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

#include <cstdint>
#include <vector>
#include <string>

namespace xllm {

enum class GeDataType {
  DT_FLOAT = 0,
  DT_FLOAT16 = 1,
  DT_UINT8 = 2,
  DT_INT32 = 3,  // GE uses 3 for DT_INT32
  DT_INT64 = 9,
  DT_INT16 = 10,  // GE uses 10 for DT_INT16
  DT_UINT16 = 11,
  DT_BOOL = 12,
  DT_BFLOAT16 = 27,  // GE uses 27 for DT_BFLOAT16
};

struct TensorMeta {
  std::vector<int64_t> shape;
  GeDataType dtype;
  std::string name;
};

struct TensorData {
  TensorMeta meta;
  std::vector<uint8_t> data;
};

GeDataType StringToGeDtype(const std::string &dtype_str);
std::string GeDtypeToString(GeDataType dtype);
size_t GeDtypeSize(GeDataType dtype);

bool LoadTensorFromBin(const std::string &bin_path, TensorData &td);
void SaveTensorToBin(const std::string &bin_path, const TensorData &td);

}  // namespace xllm
