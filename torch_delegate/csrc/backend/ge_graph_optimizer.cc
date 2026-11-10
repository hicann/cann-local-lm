/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/ge_graph_optimizer.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <graph/attr_value.h>
#include <graph/gnode.h>
#include <graph/named_io_node_builder.h>
#include <graph/operator.h>
#include <graph/tensor.h>

#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"

namespace td {
namespace {

constexpr char kOpTypeData[] = "Data";
constexpr char kOpTypePack[] = "Pack";
constexpr char kAttrIndex[] = "index";
constexpr char kAttrAxis[] = "axis";
constexpr char kAttrInputsAllSym[] = "_inputs_all_sym";

Status GetNodeName(const ge::GNode &node, std::string &name) {
  ge::AscendString ascend_name;
  if (node.GetName(ascend_name) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  name = (ascend_name.GetString() != nullptr) ? ascend_name.GetString() : "";
  return SUCCESS;
}

Status GetNodeType(const ge::GNode &node, std::string &type) {
  ge::AscendString ascend_type;
  if (node.GetType(ascend_type) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  type = (ascend_type.GetString() != nullptr) ? ascend_type.GetString() : "";
  return SUCCESS;
}

Status GetDataNodeIndex(const ge::GNode &node, int64_t &index) {
  if (node.GetAttr(ge::AscendString(kAttrIndex), index) != ge::GRAPH_SUCCESS) {
    std::string name;
    (void)GetNodeName(node, name);
    TDLOGE("GetDataNodeIndex failed: node=%s has no index attr", name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool IsCpuDataNode(const ge::GNode &node) {
  ge::TensorDesc desc;
  if (node.GetInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
    return false;
  }
  return desc.GetPlacement() == ge::kPlacementHost;
}

bool IsSymPackNode(ge::GNode &node) {
  std::string type;
  if (GetNodeType(node, type) != SUCCESS || type != kOpTypePack) {
    return false;
  }
  bool all_sym = false;
  if (!node.HasAttr(ge::AscendString(kAttrInputsAllSym))) {
    return false;
  }
  if (node.GetAttr(ge::AscendString(kAttrInputsAllSym), all_sym) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (!all_sym) {
    return false;
  }
  int64_t axis = 0;
  if (node.GetAttr(ge::AscendString(kAttrAxis), axis) != ge::GRAPH_SUCCESS || axis != 0) {
    std::string name;
    (void)GetNodeName(node, name);
    TDLOGE("IsSymPackNode: sym pack axis must be 0, node=%s, axis=%ld", name.c_str(), static_cast<long>(axis));
    return false;
  }
  return true;
}

int64_t CountDataNodes(const ge::Graph &graph) {
  int64_t count = 0;
  for (const auto &node : graph.GetAllNodes()) {
    std::string type;
    if (GetNodeType(node, type) == SUCCESS && type == kOpTypeData) {
      ++count;
    }
  }
  return count;
}

std::string MakePackInputKey(const std::vector<std::string> &input_names) {
  std::string key;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i > 0) {
      key += ",";
    }
    key += input_names[i];
  }
  return key;
}

Status GetPackInputDataNames(const ge::GNode &pack_node, std::vector<std::string> &input_names) {
  for (size_t i = 0; i < pack_node.GetInputsSize(); ++i) {
    auto [src_node, src_port] = pack_node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
    if (src_node == nullptr) {
      std::string name;
      (void)GetNodeName(pack_node, name);
      TDLOGE("GetPackInputDataNames failed: pack node=%s input %zu is null", name.c_str(), i);
      return FAILED;
    }
    std::string src_type;
    TD_CHK_STATUS(GetNodeType(*src_node, src_type));
    if (src_type != kOpTypeData) {
      std::string name;
      (void)GetNodeName(pack_node, name);
      TDLOGE("GetPackInputDataNames failed: pack node=%s input %zu is not Data (got %s)", name.c_str(), i,
             src_type.c_str());
      return INVALID_PARAM;
    }
    std::string src_name;
    TD_CHK_STATUS(GetNodeName(*src_node, src_name));
    input_names.push_back(src_name);
  }
  return SUCCESS;
}

struct MergedDataInfo {
  ge::GNode data_node;
  std::vector<int64_t> source_indices;
};

Status CreateMergedDataNode(ge::Graph &graph, int64_t index, size_t input_count, ge::GNode &out_node) {
  ge::TensorDesc desc(ge::Shape({static_cast<int64_t>(input_count)}), ge::FORMAT_ND, ge::DT_INT64);
  desc.SetPlacement(ge::kPlacementHost);

  ge::AttrValue index_attr;
  index_attr.SetAttrValue(index);

  std::vector<std::pair<std::string, ge::TensorDesc>> data_inputs = {{"x", desc}};
  std::vector<std::pair<std::string, ge::TensorDesc>> data_outputs = {{"y", desc}};
  std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{kAttrIndex, index_attr}};

  ge::NamedIoNodeBuilder builder(graph);
  builder.Type(kOpTypeData).Name(("SymPack_merged_" + std::to_string(index)).c_str());
  for (const auto &input : data_inputs) {
    builder.AddInput(input.first.c_str(), input.second);
  }
  for (const auto &output : data_outputs) {
    builder.AddOutput(output.first.c_str(), output.second);
  }
  for (const auto &attr : attrs) {
    builder.Attr(attr.first.c_str(), attr.second);
  }

  ge::AscendString error_message;
  auto node_ptr = builder.Build(error_message);
  if (node_ptr == nullptr) {
    TDLOGE("CreateMergedDataNode failed: builder error=%s",
           error_message.GetString() == nullptr ? "unknown" : error_message.GetString());
    return FAILED;
  }
  out_node = *node_ptr;
  return SUCCESS;
}

Status ReconnectPackConsumers(ge::Graph &graph, ge::GNode &pack_node, ge::GNode &merged_data) {
  auto consumers = pack_node.GetOutDataNodesAndPortIndexs(0);
  for (auto &[consumer_ptr, port] : consumers) {
    if (consumer_ptr == nullptr) {
      continue;
    }
    ge::GNode consumer = *consumer_ptr;
    if (graph.RemoveEdge(pack_node, 0, consumer, port) != ge::GRAPH_SUCCESS) {
      std::string name;
      (void)GetNodeName(pack_node, name);
      TDLOGE("ReconnectPackConsumers failed: RemoveEdge pack=%s port=%d", name.c_str(), port);
      return FAILED;
    }
    if (graph.AddDataEdge(merged_data, 0, consumer, port) != ge::GRAPH_SUCCESS) {
      std::string name;
      (void)GetNodeName(pack_node, name);
      TDLOGE("ReconnectPackConsumers failed: AddDataEdge merged->consumer port=%d, pack=%s", port, name.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

}  // namespace

Status GeGraphOptimizer::SymPackMergePass(ge::Graph &graph, InputBindingMap &binding_map) {
  TDLOGI("SymPackMergePass enter");

  std::unordered_map<std::string, int64_t> host_data_indices;
  std::vector<ge::GNode> sym_pack_nodes;

  for (auto &node : graph.GetAllNodes()) {
    std::string type;
    TD_CHK_STATUS(GetNodeType(node, type));
    if (type == kOpTypeData) {
      if (IsCpuDataNode(node)) {
        std::string name;
        TD_CHK_STATUS(GetNodeName(node, name));
        int64_t idx = 0;
        TD_CHK_STATUS(GetDataNodeIndex(node, idx));
        host_data_indices[name] = idx;
        TDLOGD("SymPackMergePass: host data=%s index=%ld", name.c_str(), static_cast<long>(idx));
      }
    } else if (type == kOpTypePack) {
      if (IsSymPackNode(node)) {
        sym_pack_nodes.push_back(node);
      }
    }
  }

  if (sym_pack_nodes.empty()) {
    TDLOGI("SymPackMergePass exit: no sym pack nodes found (no-op)");
    return SUCCESS;
  }

  TDLOGI("SymPackMergePass: found %zu sym pack nodes, %zu host data nodes", sym_pack_nodes.size(),
         host_data_indices.size());

  int64_t next_data_index = CountDataNodes(graph);
  std::vector<MergedDataInfo> merged_list;
  std::unordered_map<std::string, ge::GNode> pack_to_merged;

  for (auto &pack_node : sym_pack_nodes) {
    std::vector<std::string> input_names;
    TD_CHK_STATUS(GetPackInputDataNames(pack_node, input_names));

    std::string pack_name;
    (void)GetNodeName(pack_node, pack_name);

    std::vector<int64_t> source_indices;
    source_indices.reserve(input_names.size());
    for (const auto &name : input_names) {
      auto it = host_data_indices.find(name);
      if (it == host_data_indices.end()) {
        TDLOGE("SymPackMergePass failed: pack=%s input=%s is not a host data node", pack_name.c_str(), name.c_str());
        return INVALID_PARAM;
      }
      source_indices.push_back(it->second);
    }

    std::string input_key = MakePackInputKey(input_names);
    ge::GNode merged_node;
    bool reused = false;

    for (const auto &info : merged_list) {
      if (info.source_indices == source_indices) {
        merged_node = info.data_node;
        reused = true;
        TDLOGD("SymPackMergePass: reuse merged data for pack=%s", pack_name.c_str());
        break;
      }
    }

    if (!reused) {
      TD_CHK_STATUS(CreateMergedDataNode(graph, next_data_index, input_names.size(), merged_node));
      merged_list.push_back({merged_node, source_indices});
      ++next_data_index;
      TDLOGI("SymPackMergePass: created merged data index=%ld for pack=%s source_indices_size=%zu",
             static_cast<long>(next_data_index - 1), pack_name.c_str(), source_indices.size());
    }

    TD_CHK_STATUS(ReconnectPackConsumers(graph, pack_node, merged_node));

    if (graph.RemoveNode(pack_node) != ge::GRAPH_SUCCESS) {
      TDLOGE("SymPackMergePass failed: RemoveNode pack=%s", pack_name.c_str());
      return FAILED;
    }
    TDLOGD("SymPackMergePass: removed pack node=%s", pack_name.c_str());
  }

  binding_map.clear();
  binding_map.resize(next_data_index);
  for (const auto &info : merged_list) {
    int64_t idx = 0;
    TD_CHK_STATUS(GetDataNodeIndex(info.data_node, idx));
    if (idx >= 0 && static_cast<size_t>(idx) < binding_map.size()) {
      binding_map[idx].source_user_indices = info.source_indices;
    }
  }

  TDLOGI("SymPackMergePass exit: success, merged_data_count=%zu, total_data_count=%ld", merged_list.size(),
         static_cast<long>(next_data_index));
  return SUCCESS;
}

Status GeGraphOptimizer::RemoveDeadDataAndReorderPass(ge::Graph &graph, InputBindingMap &binding_map) {
  TDLOGI("RemoveDeadDataAndReorderPass enter");

  struct DataNodeInfo {
    ge::GNode node;
    std::string name;
    int64_t old_index = 0;
    int64_t refcount = 0;
  };

  std::vector<DataNodeInfo> data_nodes;
  std::unordered_map<std::string, size_t> name_to_data_idx;

  for (const auto &node : graph.GetAllNodes()) {
    std::string type;
    TD_CHK_STATUS(GetNodeType(node, type));
    if (type == kOpTypeData) {
      std::string name;
      TD_CHK_STATUS(GetNodeName(node, name));
      int64_t idx = 0;
      TD_CHK_STATUS(GetDataNodeIndex(node, idx));
      name_to_data_idx[name] = data_nodes.size();
      data_nodes.push_back({node, name, idx, 0});
    }
  }

  for (const auto &node : graph.GetAllNodes()) {
    std::string type;
    TD_CHK_STATUS(GetNodeType(node, type));
    if (type == kOpTypeData) {
      continue;
    }
    for (size_t i = 0; i < node.GetInputsSize(); ++i) {
      auto [src_node, src_port] = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
      if (src_node == nullptr) {
        continue;
      }
      std::string src_name;
      TD_CHK_STATUS(GetNodeName(*src_node, src_name));
      auto it = name_to_data_idx.find(src_name);
      if (it != name_to_data_idx.end()) {
        data_nodes[it->second].refcount++;
      }
    }
  }

  int64_t new_index = 0;
  InputBindingMap new_binding_map;

  for (auto &info : data_nodes) {
    if (info.refcount == 0) {
      if (graph.RemoveNode(info.node) != ge::GRAPH_SUCCESS) {
        TDLOGE("RemoveDeadDataAndReorderPass failed: RemoveNode dead data=%s", info.name.c_str());
        return FAILED;
      }
      TDLOGD("RemoveDeadDataAndReorderPass: removed dead data=%s old_index=%ld", info.name.c_str(),
             static_cast<long>(info.old_index));
      continue;
    }

    if (info.node.SetAttr(ge::AscendString(kAttrIndex), new_index) != ge::GRAPH_SUCCESS) {
      TDLOGE("RemoveDeadDataAndReorderPass failed: SetAttr index for data=%s", info.name.c_str());
      return FAILED;
    }

    if (info.old_index >= 0 && static_cast<size_t>(info.old_index) < binding_map.size() &&
        !binding_map[info.old_index].source_user_indices.empty()) {
      new_binding_map.push_back(binding_map[info.old_index]);
    } else {
      new_binding_map.push_back({{info.old_index}});
    }
    TDLOGD("RemoveDeadDataAndReorderPass: data=%s old_index=%ld -> new_index=%ld refcount=%ld", info.name.c_str(),
           static_cast<long>(info.old_index), static_cast<long>(new_index), static_cast<long>(info.refcount));
    ++new_index;
  }

  binding_map = std::move(new_binding_map);

  TDLOGI("RemoveDeadDataAndReorderPass exit: success, surviving_data_count=%ld", static_cast<long>(new_index));
  return SUCCESS;
}

Status GeGraphOptimizer::Optimize(ge::Graph &graph, InputBindingMap &binding_map) {
  TDLOGI("GeGraphOptimizer::Optimize enter");
  binding_map.clear();

  TD_CHK_STATUS(SymPackMergePass(graph, binding_map));

  if (binding_map.empty()) {
    TDLOGI("GeGraphOptimizer::Optimize exit: no sym pack optimization (no-op)");
    return SUCCESS;
  }

  TD_CHK_STATUS(RemoveDeadDataAndReorderPass(graph, binding_map));

  TDLOGI("GeGraphOptimizer::Optimize exit: success, binding_map_size=%zu", binding_map.size());
  return SUCCESS;
}

}  // namespace td
