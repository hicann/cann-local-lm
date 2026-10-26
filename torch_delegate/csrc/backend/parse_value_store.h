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

#include "backend/model_types.h"
#include "torch_delegate/td_types.h"

namespace td {

Status ParseTensorValues(const OrderedJson &obj, std::unordered_map<std::string, ParsedTensorMeta> &out);
Status ParseSymIntValues(const OrderedJson &obj, std::unordered_map<std::string, int64_t> &out);
Status ParseSymFloatValues(const OrderedJson &obj, std::unordered_map<std::string, double> &out);
Status ParseSymBoolValues(const OrderedJson &obj, std::unordered_map<std::string, bool> &out);
Status ParseIntValues(const OrderedJson &obj, std::unordered_map<std::string, int64_t> &out);
Status ParseFloatValues(const OrderedJson &obj, std::unordered_map<std::string, double> &out);
Status ParseBoolValues(const OrderedJson &obj, std::unordered_map<std::string, bool> &out);
Status ParseStringValues(const OrderedJson &obj, std::unordered_map<std::string, std::string> &out);

}  // namespace td
