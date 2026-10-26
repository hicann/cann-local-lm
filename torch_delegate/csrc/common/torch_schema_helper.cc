/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/torch_schema_helper.h"

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <torch/torch.h>

#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {
namespace {

bool IsTensorType(const c10::TypePtr &type) {
  if (type->isSubtypeOf(c10::TensorType::get())) {
    return true;
  }
  const auto *optional_type = type->castRaw<c10::OptionalType>();
  return optional_type != nullptr && optional_type->getElementType()->isSubtypeOf(c10::TensorType::get());
}

bool IsTensorListType(const c10::TypePtr &type) {
  if (type->isSubtypeOf(c10::ListType::ofTensors())) {
    return true;
  }
  const auto *optional_type = type->castRaw<c10::OptionalType>();
  return optional_type != nullptr && IsTensorListType(optional_type->getElementType());
}

bool IsOptionalTensorType(const c10::TypePtr &type) {
  const auto *optional_type = type->castRaw<c10::OptionalType>();
  return optional_type != nullptr && optional_type->getElementType()->isSubtypeOf(c10::TensorType::get());
}

TorchSchemaValueInfo BuildValueInfo(const c10::Argument &arg, size_t fallback_index) {
  TorchSchemaValueInfo info;
  info.name = arg.name().empty() ? ("output_" + std::to_string(fallback_index)) : arg.name();
  info.is_tensor = IsTensorType(arg.type());
  info.is_optional_tensor = IsOptionalTensorType(arg.type());
  info.is_tensor_list = IsTensorListType(arg.type());
  info.is_kwarg_only = arg.kwarg_only();
  info.has_default = arg.default_value().has_value();
  info.is_out = arg.is_out();
  if (arg.alias_info() != nullptr) {
    info.has_alias = true;
    info.is_alias_write = arg.alias_info()->isWrite();
  }
  return info;
}

}  // namespace

Status TorchSchemaHelper::Query(const std::string &dispatcher_name, const std::string &overload,
                                const std::string &context, TorchSchemaInfo &out) const {
  out = TorchSchemaInfo{};
  out.full_op_name = dispatcher_name;
  out.overload = overload;

  const size_t separator = dispatcher_name.find("::");
  if (separator == std::string::npos) {
    TDLOGE("Dispatcher target missing namespace separator, context=%s, target=%s", context.c_str(),
           dispatcher_name.c_str());
    return INVALID_PARAM;
  }
  out.namespace_name = dispatcher_name.substr(0, separator);
  out.op_name = dispatcher_name.substr(separator + 2);

  const c10::OperatorName op_name(dispatcher_name, overload);
  const auto handle = torch::Dispatcher::singleton().findSchema(op_name);
  if (!handle.has_value()) {
    TDLOGE("Dispatcher schema not found, context=%s, target=%s.%s", context.c_str(), dispatcher_name.c_str(),
           overload.c_str());
    return FAILED;
  }

  const auto &schema = handle->schema();
  out.is_out = overload == "out";
  if (!schema.arguments().empty()) {
    const auto &first_arg = schema.arguments().front();
    if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
      out.is_inplace = true;
    }
  }

  out.arguments.reserve(schema.arguments().size());
  for (size_t index = 0; index < schema.arguments().size(); ++index) {
    TorchSchemaValueInfo value = BuildValueInfo(schema.arguments()[index], index);
    if (value.is_alias_write) {
      out.is_inplace = true;
    }
    if (value.is_tensor_list) {
      out.has_tensor_list_input = true;
    }
    out.arguments.push_back(std::move(value));
  }

  out.returns.reserve(schema.returns().size());
  for (size_t index = 0; index < schema.returns().size(); ++index) {
    out.returns.push_back(BuildValueInfo(schema.returns()[index], index));
  }
  return SUCCESS;
}

}  // namespace td
