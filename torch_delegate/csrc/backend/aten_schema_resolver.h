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
#include <vector>

#include "backend/model_types.h"
#include "torch_delegate/td_types.h"

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
  Status ParseTarget(const std::string &target, const std::string &context, AtenTargetInfo &out) const;

  Status Resolve(const ParsedNode &node, AtenSchemaInfo &out) const;

  std::string MakeFallbackOpType(const AtenTargetInfo &info) const;
};

}  // namespace td
