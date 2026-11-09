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

#include "backend/model_types.h"
#include "backend/parsed_model.h"

namespace td {

// C++ 加载期 CPU/NPU 归属复述：与导出期的 Device 归属报告对应，逐节点重述
// 执行 Device 归属结论。CPU 节点按 target 的 namespace 分级——aten 算子缺
// NPU kernel 属可疑（提示 torch_npu 覆盖缺口），自定义 namespace 属预期；
// 仅告警不报错。返回报告文本（供单测断言）。
std::string ReportCpuOwnership(const ParsedModel &model);

}  // namespace td
