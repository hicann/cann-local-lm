/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/parse_ascend_ir.h"

#include <string>

#include <nlohmann/json.hpp>

#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {
namespace {

Status ParseOpDesc(const OrderedJson &desc, const std::string &context, ParsedAscendIrOpDesc &parsed) {
  parsed = ParsedAscendIrOpDesc{};
  parsed.raw_json = desc;
  if (!desc.is_object()) {
    TDLOGE("Desc must be object, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  if (desc.contains("name") && desc["name"].is_string()) {
    parsed.name = desc["name"].get<std::string>();
  }
  if (desc.contains("dtype") && desc["dtype"].is_string()) {
    parsed.dtype = desc["dtype"].get<std::string>();
  }
  if (desc.contains("layout") && desc["layout"].is_string()) {
    parsed.layout = desc["layout"].get<std::string>();
  }
  if (desc.contains("deviceType") && desc["deviceType"].is_string()) {
    parsed.device_type = desc["deviceType"].get<std::string>();
  }
  if (desc.contains("shape") && desc["shape"].is_object() && desc["shape"].contains("dim")) {
    const auto &dim = desc["shape"]["dim"];
    if (dim.is_array()) {
      for (const auto &item : dim) {
        if (item.is_string()) {
          parsed.shape.push_back(std::stoll(item.get<std::string>()));
        } else if (item.is_number_integer()) {
          parsed.shape.push_back(item.get<int64_t>());
        }
      }
    }
  }
  if (desc.contains("attr")) {
    parsed.attr = desc["attr"];
  }
  return SUCCESS;
}

}  // namespace

Status ParseAscendIrImpl(const OrderedJson &ascend_ir, const std::string &context, ParsedAscendIr &parsed) {
  if (ascend_ir.is_null()) {
    return INVALID_PARAM;
  }
  if (!ascend_ir.is_object()) {
    TDLOGE("ParseAscendIr failed: ascend_ir must be object, context=%s", context.c_str());
    return INVALID_PARAM;
  }

  parsed = ParsedAscendIr{};
  parsed.raw_json = ascend_ir;
  if (ascend_ir.contains("source") && ascend_ir["source"].is_string()) {
    parsed.source = ascend_ir["source"].get<std::string>();
    TDLOGD("ParseAscendIr: source=%s, context=%s", parsed.source.c_str(), context.c_str());
  }
  if (!ascend_ir.contains("ops") || !ascend_ir["ops"].is_array()) {
    TDLOGE("ParseAscendIr failed: missing ops array, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  TDLOGD("ParseAscendIr: ops_count=%zu, context=%s", ascend_ir["ops"].size(), context.c_str());
  for (const auto &op : ascend_ir["ops"]) {
    if (!op.is_object()) {
      TDLOGE("ParseAscendIr failed: op must be object, context=%s", context.c_str());
      return INVALID_PARAM;
    }
    ParsedAscendIrOp parsed_op;
    parsed_op.raw_json = op;
    if (op.contains("name") && op["name"].is_string()) {
      parsed_op.name = op["name"].get<std::string>();
    }
    if (op.contains("type") && op["type"].is_string()) {
      parsed_op.type = op["type"].get<std::string>();
    }
    const bool has_buildable_ir_fields = op.contains("name") || op.contains("type") || op.contains("input") ||
                                         op.contains("inputDesc") || op.contains("outputDesc");
    if (has_buildable_ir_fields && (parsed_op.name.empty() || parsed_op.type.empty())) {
      TDLOGE("ParseAscendIr failed: op missing name/type, context=%s", context.c_str());
      return INVALID_PARAM;
    }
    if (op.contains("input") && op["input"].is_array()) {
      for (const auto &input : op["input"]) {
        if (input.is_string()) {
          parsed_op.inputs.push_back(input.get<std::string>());
        }
      }
    }
    if (op.contains("attr")) {
      parsed_op.attrs = op["attr"];
    }
    if (op.contains("inputDesc") && op["inputDesc"].is_array()) {
      for (const auto &desc : op["inputDesc"]) {
        ParsedAscendIrOpDesc parsed_desc;
        TD_CHK_STATUS(ParseOpDesc(desc, context, parsed_desc));
        parsed_op.input_descs.push_back(std::move(parsed_desc));
      }
    }
    if (op.contains("outputDesc") && op["outputDesc"].is_array()) {
      for (const auto &desc : op["outputDesc"]) {
        ParsedAscendIrOpDesc parsed_desc;
        TD_CHK_STATUS(ParseOpDesc(desc, context, parsed_desc));
        parsed_op.output_descs.push_back(std::move(parsed_desc));
      }
    }
    parsed.ops.push_back(std::move(parsed_op));
  }
  if (ascend_ir.contains("mapping") && ascend_ir["mapping"].is_object()) {
    const auto &mapping = ascend_ir["mapping"];
    if (mapping.contains("inputs") && mapping["inputs"].is_object()) {
      for (const auto &[fx_tensor_name, ge_tensor_name] : mapping["inputs"].items()) {
        if (ge_tensor_name.is_string()) {
          parsed.input_mappings.emplace(fx_tensor_name, ge_tensor_name.get<std::string>());
        }
      }
    }
    if (mapping.contains("outputs") && mapping["outputs"].is_object()) {
      for (const auto &[fx_tensor_name, ge_tensor_name] : mapping["outputs"].items()) {
        if (ge_tensor_name.is_string()) {
          parsed.output_mappings.emplace(fx_tensor_name, ge_tensor_name.get<std::string>());
        }
      }
    } else if (mapping.contains("outputs") && mapping["outputs"].is_array()) {
      size_t index = 0;
      for (const auto &output : mapping["outputs"]) {
        if (output.is_string()) {
          parsed.output_mappings.emplace(std::to_string(index++), output.get<std::string>());
        }
      }
    }
  }
  return SUCCESS;
}

}  // namespace td
