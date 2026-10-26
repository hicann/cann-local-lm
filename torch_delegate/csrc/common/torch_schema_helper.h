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

#include "torch_delegate/td_types.h"

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
  Status Query(const std::string &dispatcher_name, const std::string &overload, const std::string &context,
               TorchSchemaInfo &out) const;
};

}  // namespace td
