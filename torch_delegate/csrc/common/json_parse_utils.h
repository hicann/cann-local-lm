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

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/model_types.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {

inline std::optional<int64_t> ParseJsonIntLike(const OrderedJson &value) {
  if (value.is_number_integer()) {
    return value.get<int64_t>();
  }
  if (value.is_string()) {
    try {
      return std::stoll(value.get<std::string>());
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

inline std::optional<double> ParseJsonFloatLike(const OrderedJson &value) {
  if (value.is_number()) {
    return value.get<double>();
  }
  if (value.is_string()) {
    try {
      return std::stod(value.get<std::string>());
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// Strict: only accepts number_integer, {as_int}, {as_expr}. Rejects bare string integers ("123").
// Used by parser/loader for EP JSON fields with well-defined integer encoding.
// For GE attr parsing that tolerates string-encoded integers, use ParseJsonIntLike directly.
inline bool TryGetIntField(const OrderedJson &value, int64_t &out) {
  if (value.is_number_integer()) {
    out = value.get<int64_t>();
    return true;
  }
  if (!value.is_object()) return false;
  if (value.contains("as_int") && value["as_int"].is_number_integer()) {
    out = value["as_int"].get<int64_t>();
    return true;
  }
  if (value.contains("as_expr")) {
    out = -1;
    return true;
  }
  return false;
}

inline Status ParseIntList(const OrderedJson &array, const std::string &context, std::vector<int64_t> &out) {
  out.clear();
  if (!array.is_array()) {
    TDLOGE("Expected JSON array, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  for (const auto &item : array) {
    int64_t value = 0;
    if (!TryGetIntField(item, value)) {
      TDLOGE("Expected integer element, context=%s", context.c_str());
      return INVALID_PARAM;
    }
    out.push_back(value);
  }
  return SUCCESS;
}

inline Status ParseTensorMeta(const OrderedJson &meta, const std::string &context, ParsedTensorMeta &out,
                              const std::string &prefix = "TENSOR") {
  out = ParsedTensorMeta{};
  if (!meta.is_object()) {
    TDLOGE("Tensor meta must be an object, prefix=%s, context=%s", prefix.c_str(), context.c_str());
    return INVALID_PARAM;
  }
  if (!meta.contains("dtype") || !meta["dtype"].is_number_integer()) {
    TDLOGE("Tensor meta missing integer dtype, prefix=%s, context=%s", prefix.c_str(), context.c_str());
    return INVALID_PARAM;
  } else {
    out.dtype = meta["dtype"].get<int64_t>();
  }
  if (!meta.contains("sizes")) {
    TDLOGE("Tensor meta missing sizes, prefix=%s, context=%s", prefix.c_str(), context.c_str());
    return INVALID_PARAM;
  } else {
    TD_CHK_STATUS(ParseIntList(meta["sizes"], context, out.sizes));
    if (prefix == "TENSOR" && meta["sizes"].is_array()) {
      out.dynamic_dim_symbols.resize(meta["sizes"].size());
      for (size_t i = 0; i < meta["sizes"].size(); ++i) {
        const auto &dim = meta["sizes"][i];
        if (dim.is_object() && dim.contains("as_expr") && dim["as_expr"].is_object()) {
          const auto &expr = dim["as_expr"];
          if (expr.contains("expr_str") && expr["expr_str"].is_string()) {
            const std::string &expr_str = expr["expr_str"].get_ref<const std::string &>();
            size_t start = expr_str.find("Symbol('");
            if (start != std::string::npos) {
              start += 8;
              size_t end = expr_str.find("'", start);
              if (end != std::string::npos) {
                out.dynamic_dim_symbols[i] = expr_str.substr(start, end - start);
                TDLOGD("ParseTensorMeta: dim %zu is dynamic, symbol=%s", i, out.dynamic_dim_symbols[i].c_str());
              }
            }
          }
        }
      }
    }
  }
  if (meta.contains("requires_grad") && meta["requires_grad"].is_boolean()) {
    out.requires_grad = meta["requires_grad"].get<bool>();
  }
  if (meta.contains("device") && meta["device"].is_object()) {
    const auto &device = meta["device"];
    if (device.contains("type") && device["type"].is_string()) {
      out.device.type = device["type"].get<std::string>();
    }
    if (device.contains("index")) {
      if (device["index"].is_number_integer()) {
        out.device.index = device["index"].get<int64_t>();
      }
    }
  }
  if (meta.contains("strides")) {
    TD_CHK_STATUS(ParseIntList(meta["strides"], context, out.strides));
  }
  if (meta.contains("storage_offset")) {
    int64_t offset = 0;
    if (!TryGetIntField(meta["storage_offset"], offset)) {
      TDLOGE("storage_offset must be integer, prefix=%s, context=%s", prefix.c_str(), context.c_str());
      return INVALID_PARAM;
    } else {
      out.storage_offset = offset;
    }
  }
  if (meta.contains("layout") && meta["layout"].is_number_integer()) {
    out.layout = meta["layout"].get<int64_t>();
  }
  return SUCCESS;
}

inline size_t Product(const std::vector<int64_t> &dims) {
  size_t product = 1;
  for (const int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    product *= static_cast<size_t>(dim);
  }
  return product;
}

inline std::string ToCamelCase(const std::string &value) {
  std::string result;
  bool capitalize = true;
  for (const char c : value) {
    if (c == '_' || c == '.') {
      capitalize = true;
      continue;
    }
    if (capitalize) {
      result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      capitalize = false;
    } else {
      result += c;
    }
  }
  return result;
}

}  // namespace td
