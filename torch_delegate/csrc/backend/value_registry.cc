/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/value_registry.h"

#include <unordered_set>

#include "common/logging/assert_macros.h"
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

Status ValueRegistry::RegisterPlaceholderSource(const std::string &placeholder_name, const RegisteredValue &value) {
  TD_CHK_STATUS(RegisterInternal(placeholder_name + ":0", value));
  TD_CHK_STATUS(BindAlias(placeholder_name, placeholder_name + ":0"));
  return SUCCESS;
}

Status ValueRegistry::RegisterProducerOutput(const std::string &output_ref, const RegisteredValue &value) {
  return RegisterInternal(output_ref, value);
}

Status ValueRegistry::RegisterAlias(const std::string &alias, const std::string &target_ref) {
  TD_CHK_STATUS(BindAlias(alias, target_ref));
  if (!HasPortSuffix(alias)) {
    TD_CHK_STATUS(BindAlias(alias + ":0", target_ref));
  }
  return SUCCESS;
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

Status ValueRegistry::RegisterInternal(const std::string &key, const RegisteredValue &value) {
  const auto [it, inserted] = values_.emplace(key, value);
  if (!inserted) {
    TDLOGE("Duplicate value registration, key=%s", key.c_str());
    return INVALID_PARAM;
  }
  return SUCCESS;
}

Status ValueRegistry::BindAlias(const std::string &alias, const std::string &target_ref) {
  if (values_.find(alias) != values_.end() && alias != target_ref) {
    TDLOGE("Alias conflicts with existing producer ref, alias=%s, target_ref=%s", alias.c_str(), target_ref.c_str());
    return INVALID_PARAM;
  }
  const std::string canonical_target = ResolveAliasTarget(values_, aliases_, target_ref);
  const auto [it, inserted] = aliases_.emplace(alias, canonical_target);
  if (!inserted && it->second != canonical_target) {
    TDLOGE("Alias already bound to another producer, alias=%s, target_ref=%s", alias.c_str(), target_ref.c_str());
    return INVALID_PARAM;
  }
  return SUCCESS;
}

}  // namespace td
