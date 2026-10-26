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

std::string SelectNodeStem(const ParsedNode &parsed);
std::string BuildNodeName(const ParsedNode &parsed);
std::string BuildNodeJsonContext(const ParsedNode &node);
bool IsFallbackNode(const ParsedNode &node);
Status BuildFallbackOpType(const ParsedNode &node, std::optional<std::string> &out);
std::string BuildUniqueFallbackNodeName(const std::string &op_type, std::unordered_map<std::string, size_t> &counters);

}  // namespace td
