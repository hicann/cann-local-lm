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

#include <filesystem>
#include <string>

#include <graph/ascend_string.h>

namespace td::internal {

inline std::string CharToString(const char *value) {
  return value == nullptr ? std::string() : std::string(value);
}

inline std::string AscendStringToString(const ge::AscendString &value) {
  const char *raw = value.GetString();
  return raw == nullptr ? std::string() : std::string(raw);
}

inline ge::AscendString StringToAscendString(const std::string &value) {
  return ge::AscendString(value.c_str());
}

inline std::filesystem::path CharToPath(const char *value) {
  return std::filesystem::path(CharToString(value));
}

inline const char *PathToChar(const std::filesystem::path &value) {
  return value.c_str();
}

}  // namespace td::internal
