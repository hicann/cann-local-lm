/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tensor_io.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace xllm {

GeDataType StringToGeDtype(const std::string &dtype_str) {
  if (dtype_str == "DT_FLOAT" || dtype_str == "torch.float32") return GeDataType::DT_FLOAT;
  if (dtype_str == "DT_FLOAT16" || dtype_str == "torch.float16") return GeDataType::DT_FLOAT16;
  if (dtype_str == "DT_BF16" || dtype_str == "DT_BFLOAT16" || dtype_str == "torch.bfloat16")
    return GeDataType::DT_BFLOAT16;
  if (dtype_str == "DT_INT16" || dtype_str == "torch.int16") return GeDataType::DT_INT16;
  if (dtype_str == "DT_INT32" || dtype_str == "torch.int32") return GeDataType::DT_INT32;
  if (dtype_str == "DT_INT64" || dtype_str == "torch.int64") return GeDataType::DT_INT64;
  if (dtype_str == "DT_UINT8" || dtype_str == "torch.uint8") return GeDataType::DT_UINT8;
  if (dtype_str == "DT_BOOL" || dtype_str == "torch.bool") return GeDataType::DT_BOOL;
  std::cerr << "[Warning] Unknown dtype: " << dtype_str << ", defaulting to DT_FLOAT" << std::endl;
  return GeDataType::DT_FLOAT;
}

std::string GeDtypeToString(GeDataType dtype) {
  switch (dtype) {
    case GeDataType::DT_FLOAT:
      return "DT_FLOAT";
    case GeDataType::DT_FLOAT16:
      return "DT_FLOAT16";
    case GeDataType::DT_BFLOAT16:
      return "DT_BFLOAT16";
    case GeDataType::DT_INT16:
      return "DT_INT16";
    case GeDataType::DT_INT32:
      return "DT_INT32";
    case GeDataType::DT_INT64:
      return "DT_INT64";
    case GeDataType::DT_UINT8:
      return "DT_UINT8";
    case GeDataType::DT_BOOL:
      return "DT_BOOL";
    default:
      return "DT_UNKNOWN";
  }
}

size_t GeDtypeSize(GeDataType dtype) {
  switch (dtype) {
    case GeDataType::DT_FLOAT:
      return 4;
    case GeDataType::DT_FLOAT16:
      return 2;
    case GeDataType::DT_BFLOAT16:
      return 2;
    case GeDataType::DT_INT16:
      return 2;
    case GeDataType::DT_INT32:
      return 4;
    case GeDataType::DT_INT64:
      return 8;
    case GeDataType::DT_UINT8:
      return 1;
    case GeDataType::DT_BOOL:
      return 1;
    default:
      return 4;
  }
}

bool LoadTensorFromBin(const std::string &bin_path, TensorData &td) {
  std::ifstream bin_file(bin_path, std::ios::binary);
  if (!bin_file.is_open()) {
    std::cerr << "[Error] Cannot open bin file: " << bin_path << std::endl;
    return false;
  }

  uint32_t dtype_code;
  uint32_t ndim;
  bin_file.read(reinterpret_cast<char *>(&dtype_code), sizeof(uint32_t));
  bin_file.read(reinterpret_cast<char *>(&ndim), sizeof(uint32_t));

  td.meta.dtype = static_cast<GeDataType>(dtype_code);
  td.meta.shape.resize(ndim);
  bin_file.read(reinterpret_cast<char *>(td.meta.shape.data()), ndim * sizeof(int64_t));

  size_t element_count = 1;
  for (auto dim : td.meta.shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * GeDtypeSize(td.meta.dtype);

  td.data.resize(data_size);
  bin_file.read(reinterpret_cast<char *>(td.data.data()), data_size);
  bin_file.close();

  std::cout << "[Load] " << bin_path << " dtype=" << GeDtypeToString(td.meta.dtype) << ", shape=["
            << td.meta.shape.size() << " dims], size=" << data_size << " bytes" << std::endl;

  return true;
}

void SaveTensorToBin(const std::string &bin_path, const TensorData &td) {
  std::ofstream bin_file(bin_path, std::ios::binary);

  uint32_t dtype_code = static_cast<uint32_t>(td.meta.dtype);
  uint32_t ndim = static_cast<uint32_t>(td.meta.shape.size());

  bin_file.write(reinterpret_cast<const char *>(&dtype_code), sizeof(uint32_t));
  bin_file.write(reinterpret_cast<const char *>(&ndim), sizeof(uint32_t));
  bin_file.write(reinterpret_cast<const char *>(td.meta.shape.data()), ndim * sizeof(int64_t));
  bin_file.write(reinterpret_cast<const char *>(td.data.data()), td.data.size());
  bin_file.close();

  std::cout << "[Save] " << bin_path << " dtype=" << GeDtypeToString(td.meta.dtype) << ", shape=["
            << td.meta.shape.size() << " dims], size=" << td.data.size() << " bytes" << std::endl;
}

}  // namespace xllm
