#pragma once

#include <string>
#include <unordered_map>

#include <graph/gnode.h>

#include "common/logging/logging.h"

namespace td {

struct RegisteredValue {
  ge::GNode producer;
  int32_t output_index = 0;
};

class ValueRegistry {
 public:
  bool RegisterPlaceholderSource(const std::string &placeholder_name, const RegisteredValue &value,
                                 ValidationResult &result);

  bool RegisterProducerOutput(const std::string &output_ref, const RegisteredValue &value, ValidationResult &result);

  bool RegisterAlias(const std::string &alias, const std::string &target_ref, ValidationResult &result);

  const RegisteredValue *Resolve(const std::string &name) const;
  bool HasRegisteredRef(const std::string &name) const;

 private:
  bool RegisterInternal(const std::string &key, const RegisteredValue &value, ValidationResult &result);

  bool BindAlias(const std::string &alias, const std::string &target_ref, ValidationResult &result);

  std::unordered_map<std::string, RegisteredValue> values_;
  std::unordered_map<std::string, std::string> aliases_;
};

}  // namespace td
