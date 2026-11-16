/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/host_cpu_capability.h"

#include <sstream>

#include "backend/graph_build_utils.h"
#include "common/logging/logging.h"
#include "common/op_target_utils.h"

namespace td {

std::string ReportCpuOwnership(const ParsedModel &model) {
  std::vector<std::string> npu_fallback_nodes;
  std::vector<std::string> aten_cpu_nodes;
  std::vector<std::string> custom_cpu_nodes;
  size_t npu_converter_count = 0;
  for (const ParsedNode &node : model.Nodes()) {
    if (IsConverterNode(node)) {
      npu_converter_count++;
      continue;
    }
    if (!node.ascend_ir.has_value()) {
      continue;
    }
    // 归属推断：converter 节点固定 NPU；fallback 节点按 ascend_ir.device
    // 分类，CPU 节点再按 target namespace 分级（aten 属可疑，自定义属预期）。
    const bool is_cpu = node.ascend_ir->device == "cpu";
    std::ostringstream entry;
    entry << node.name << " (" << node.effective_schema_target << ") device=" << (is_cpu ? "cpu" : "npu");
    if (!is_cpu) {
      npu_fallback_nodes.push_back(entry.str());
      continue;
    }
    const std::string name_space = ParseOpTarget(node.effective_schema_target).namespace_name;
    if (name_space == "aten") {
      aten_cpu_nodes.push_back(entry.str());
    } else {
      custom_cpu_nodes.push_back(entry.str());
    }
  }

  const size_t npu_count = npu_converter_count + npu_fallback_nodes.size();
  const size_t cpu_count = aten_cpu_nodes.size() + custom_cpu_nodes.size();
  std::ostringstream summary;
  summary << "[torch_delegate] Load-time device ownership: NPU nodes " << npu_count << " (converter "
          << npu_converter_count << " / fallback " << npu_fallback_nodes.size() << "), CPU nodes " << cpu_count;
  std::ostringstream report;
  report << summary.str() << "\n";
  for (const std::string &entry : npu_fallback_nodes) {
    report << "  fallback callback to PrivateUse1: " << entry << "\n";
  }
  for (const std::string &entry : aten_cpu_nodes) {
    report << "  suspicious: aten op has no PrivateUse1 kernel: " << entry << "\n";
  }
  for (const std::string &entry : custom_cpu_nodes) {
    report << "  expected: custom namespace with CPU-only impl: " << entry << "\n";
  }
  const std::string report_text = report.str();

  // 日志级别由归属分组决定，不依赖文案匹配：aten→CPU 为可疑告警，其余为信息。
  TDLOG_CHUNKED(ACL_INFO, "HostCpu ownership: summary", summary.str());
  for (const std::string &entry : npu_fallback_nodes) {
    TDLOG_CHUNKED(ACL_INFO, "HostCpu ownership: NPU fallback op", entry);
  }
  for (const std::string &entry : aten_cpu_nodes) {
    TDLOG_CHUNKED(ACL_WARNING, "HostCpu ownership: suspicious aten op on CPU", entry);
  }
  for (const std::string &entry : custom_cpu_nodes) {
    TDLOG_CHUNKED(ACL_INFO, "HostCpu ownership: custom namespace op on CPU", entry);
  }
  return report_text;
}

}  // namespace td
