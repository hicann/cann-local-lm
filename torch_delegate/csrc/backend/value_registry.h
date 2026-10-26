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

#include <string>
#include <unordered_map>

#include <graph/gnode.h>

#include "torch_delegate/td_types.h"

namespace td {

struct RegisteredValue {
  ge::GNode producer;
  int32_t output_index = 0;
};

class ValueRegistry {
 public:
  Status RegisterPlaceholderSource(const std::string &placeholder_name, const RegisteredValue &value);

  Status RegisterProducerOutput(const std::string &output_ref, const RegisteredValue &value);

  Status RegisterAlias(const std::string &alias, const std::string &target_ref);

  const RegisteredValue *Resolve(const std::string &name) const;
  bool HasRegisteredRef(const std::string &name) const;

 private:
  Status RegisterInternal(const std::string &key, const RegisteredValue &value);

  Status BindAlias(const std::string &alias, const std::string &target_ref);

  std::unordered_map<std::string, RegisteredValue> values_;
  std::unordered_map<std::string, std::string> aliases_;
};

}  // namespace td
