/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parse_value_store.h"

#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {

Status ParseTensorValues(const OrderedJson &obj, std::unordered_map<std::string, ParsedTensorMeta> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, meta] : obj.items()) {
    ParsedTensorMeta tensor_meta;
    const Status tm_status = ParseTensorMeta(meta, name, tensor_meta);
    if (tm_status != SUCCESS) {
      TDLOGE("ParseModelJson: ParseTensorMeta failed for '%s', status=%lld", name.c_str(),
             static_cast<long long>(tm_status));
      return tm_status;
    }
    out[name] = tensor_meta;
  }
  return SUCCESS;
}

Status ParseSymIntValues(const OrderedJson &obj, std::unordered_map<std::string, int64_t> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_number_integer()) {
      out[name] = value.get<int64_t>();
    } else if (value.is_object() && value.contains("as_int")) {
      out[name] = value["as_int"].get<int64_t>();
    } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
      const auto &expr = value["as_expr"];
      if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_int")) {
        out[name] = expr["hint"]["as_int"].get<int64_t>();
      }
    }
  }
  return SUCCESS;
}

Status ParseSymFloatValues(const OrderedJson &obj, std::unordered_map<std::string, double> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_number()) {
      out[name] = value.get<double>();
    } else if (value.is_object() && value.contains("as_float")) {
      out[name] = value["as_float"].get<double>();
    } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
      const auto &expr = value["as_expr"];
      if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_float")) {
        out[name] = expr["hint"]["as_float"].get<double>();
      }
    }
  }
  return SUCCESS;
}

Status ParseSymBoolValues(const OrderedJson &obj, std::unordered_map<std::string, bool> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_boolean()) {
      out[name] = value.get<bool>();
    } else if (value.is_object() && value.contains("as_bool")) {
      out[name] = value["as_bool"].get<bool>();
    } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
      const auto &expr = value["as_expr"];
      if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_bool")) {
        out[name] = expr["hint"]["as_bool"].get<bool>();
      }
    }
  }
  return SUCCESS;
}

Status ParseIntValues(const OrderedJson &obj, std::unordered_map<std::string, int64_t> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_number_integer()) {
      out[name] = value.get<int64_t>();
    } else if (value.is_object() && value.contains("as_int")) {
      out[name] = value["as_int"].get<int64_t>();
    }
  }
  return SUCCESS;
}

Status ParseFloatValues(const OrderedJson &obj, std::unordered_map<std::string, double> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_number()) {
      out[name] = value.get<double>();
    } else if (value.is_object() && value.contains("as_float")) {
      out[name] = value["as_float"].get<double>();
    }
  }
  return SUCCESS;
}

Status ParseBoolValues(const OrderedJson &obj, std::unordered_map<std::string, bool> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_boolean()) {
      out[name] = value.get<bool>();
    } else if (value.is_object() && value.contains("as_bool")) {
      out[name] = value["as_bool"].get<bool>();
    }
  }
  return SUCCESS;
}

Status ParseStringValues(const OrderedJson &obj, std::unordered_map<std::string, std::string> &out) {
  if (!obj.is_object()) {
    return SUCCESS;
  }
  for (const auto &[name, value] : obj.items()) {
    if (value.is_string()) {
      out[name] = value.get<std::string>();
    } else if (value.is_object() && value.contains("as_string")) {
      out[name] = value["as_string"].get<std::string>();
    }
  }
  return SUCCESS;
}

}  // namespace td
