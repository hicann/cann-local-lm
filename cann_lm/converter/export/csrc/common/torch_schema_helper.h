#pragma once

#include <string>
#include <vector>

#include "common/logging/logging.h"

namespace td {

struct TorchSchemaValueInfo {
  std::string name;
  bool is_tensor = false;
  bool is_optional_tensor = false;
  bool is_tensor_list = false;
  bool is_kwarg_only = false;
  bool has_default = false;
  bool is_out = false;
  bool has_alias = false;
  bool is_alias_write = false;
};

struct TorchSchemaInfo {
  std::string full_op_name;
  std::string overload;
  std::string namespace_name;
  std::string op_name;
  std::vector<TorchSchemaValueInfo> arguments;
  std::vector<TorchSchemaValueInfo> returns;
  bool is_inplace = false;
  bool is_out = false;
  bool has_tensor_list_input = false;
};

class TorchSchemaHelper {
 public:
  TorchSchemaInfo Query(const std::string &dispatcher_name, const std::string &overload, ValidationResult &result,
                        const std::string &context) const;
};

}  // namespace td
