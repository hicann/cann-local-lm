#include "common/torch_schema_helper.h"

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <torch/torch.h>

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

TorchSchemaInfo TorchSchemaHelper::Query(const std::string &dispatcher_name, const std::string &overload,
                                         ValidationResult &result, const std::string &context) const {
  TorchSchemaInfo info;
  info.full_op_name = dispatcher_name;
  info.overload = overload;

  const size_t separator = dispatcher_name.find("::");
  if (separator == std::string::npos) {
    AddError(result, "INVALID_ATEN_TARGET", "Dispatcher target missing namespace separator", context);
    return info;
  }
  info.namespace_name = dispatcher_name.substr(0, separator);
  info.op_name = dispatcher_name.substr(separator + 2);

  const c10::OperatorName op_name(dispatcher_name, overload);
  const auto handle = torch::Dispatcher::singleton().findSchema(op_name);
  if (!handle.has_value()) {
    AddError(result, "MISSING_ATEN_SCHEMA", "Dispatcher schema not found",
             context + ":" + dispatcher_name + "." + overload);
    return info;
  }

  const auto &schema = handle->schema();
  info.is_out = overload == "out";
  if (!schema.arguments().empty()) {
    const auto &first_arg = schema.arguments().front();
    if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
      info.is_inplace = true;
    }
  }

  info.arguments.reserve(schema.arguments().size());
  for (size_t index = 0; index < schema.arguments().size(); ++index) {
    TorchSchemaValueInfo value = BuildValueInfo(schema.arguments()[index], index);
    if (value.is_alias_write) {
      info.is_inplace = true;
    }
    if (value.is_tensor_list) {
      info.has_tensor_list_input = true;
    }
    info.arguments.push_back(std::move(value));
  }

  info.returns.reserve(schema.returns().size());
  for (size_t index = 0; index < schema.returns().size(); ++index) {
    info.returns.push_back(BuildValueInfo(schema.returns()[index], index));
  }
  return info;
}

}  // namespace td
