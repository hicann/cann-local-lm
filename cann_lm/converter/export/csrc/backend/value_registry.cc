#include "backend/value_registry.h"

#include <unordered_set>

#include "common/logging/logging.h"

namespace td {
namespace {

bool HasPortSuffix(const std::string &name) {
  return name.find(':') != std::string::npos;
}

std::string ResolveAliasTarget(const std::unordered_map<std::string, RegisteredValue> &values,
                               const std::unordered_map<std::string, std::string> &aliases, const std::string &ref) {
  std::string current = ref;
  std::unordered_set<std::string> visited;
  while (values.find(current) == values.end()) {
    const auto alias_it = aliases.find(current);
    if (alias_it == aliases.end() || !visited.emplace(current).second) {
      break;
    }
    current = alias_it->second;
  }
  return current;
}

}  // namespace

bool ValueRegistry::RegisterPlaceholderSource(const std::string &placeholder_name, const RegisteredValue &value,
                                              ValidationResult &result) {
  return RegisterInternal(placeholder_name + ":0", value, result) &&
         BindAlias(placeholder_name, placeholder_name + ":0", result);
}

bool ValueRegistry::RegisterProducerOutput(const std::string &output_ref, const RegisteredValue &value,
                                           ValidationResult &result) {
  return RegisterInternal(output_ref, value, result);
}

bool ValueRegistry::RegisterAlias(const std::string &alias, const std::string &target_ref, ValidationResult &result) {
  if (!BindAlias(alias, target_ref, result)) {
    return false;
  }
  if (!HasPortSuffix(alias)) {
    return BindAlias(alias + ":0", target_ref, result);
  }
  return true;
}

const RegisteredValue *ValueRegistry::Resolve(const std::string &name) const {
  const std::string resolved = ResolveAliasTarget(values_, aliases_, name);
  const auto value_it = values_.find(resolved);
  if (value_it != values_.end()) {
    return &value_it->second;
  }
  return nullptr;
}

bool ValueRegistry::HasRegisteredRef(const std::string &name) const {
  return Resolve(name) != nullptr;
}

bool ValueRegistry::RegisterInternal(const std::string &key, const RegisteredValue &value, ValidationResult &result) {
  const auto [it, inserted] = values_.emplace(key, value);
  if (!inserted) {
    AddError(result, "DUPLICATE_VALUE_REF", "Duplicate value registration", key);
    return false;
  }
  return true;
}

bool ValueRegistry::BindAlias(const std::string &alias, const std::string &target_ref, ValidationResult &result) {
  if (values_.find(alias) != values_.end() && alias != target_ref) {
    AddError(result, "ALIAS_CONFLICT", "Alias conflicts with existing producer ref", alias);
    return false;
  }
  const std::string canonical_target = ResolveAliasTarget(values_, aliases_, target_ref);
  const auto [it, inserted] = aliases_.emplace(alias, canonical_target);
  if (!inserted && it->second != canonical_target) {
    AddError(result, "ALIAS_CONFLICT", "Alias already bound to another producer", alias);
    return false;
  }
  return true;
}

}  // namespace td
