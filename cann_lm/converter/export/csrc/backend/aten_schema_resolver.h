#pragma once

#include <string>
#include <vector>

#include "backend/epair_model_parser.h"
#include "common/logging/logging.h"

namespace td {

struct AtenTargetInfo {
  std::string target;
  std::string namespace_name;
  std::string op_name;
  std::string overload;
  std::string dispatcher_name;
};

struct AtenSchemaArgInfo {
  std::string name;
  bool is_tensor = false;
  bool is_optional_tensor = false;
  bool is_tensor_list = false;
  bool is_out = false;
  bool has_default = false;
};

struct AtenSchemaInfo {
  AtenTargetInfo target;
  std::vector<AtenSchemaArgInfo> arguments;
  bool is_inplace = false;
  bool is_out = false;
};

class AtenSchemaResolver {
 public:
  AtenTargetInfo ParseTarget(const std::string &target, ValidationResult &result, const std::string &context) const;

  AtenSchemaInfo Resolve(const ParsedNode &node, ValidationResult &result) const;

  std::string MakeFallbackOpType(const AtenTargetInfo &info) const;
};

}  // namespace td
