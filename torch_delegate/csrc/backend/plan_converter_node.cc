/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/plan_converter_node.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <graph/attr_value.h>
#include <graph/gnode.h>
#include <graph/tensor.h>

#include "backend/attr_parser.h"
#include "backend/ge_weight_loader.h"
#include "backend/graph_build_utils.h"
#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"

namespace td {

ConverterNodePlanner::ConverterNodePlanner(GraphBuildContext &ctx, const GraphPlanningDeps &deps)
    : ctx_(ctx), deps_(deps) {}

Status ConverterNodePlanner::ProcessConstOp(const ParsedNode &node, const ParsedAscendIrOp &op) {
  TD_ASSERT_NOTNULL(ctx_.graph, "ProcessConstOp failed: graph is null");
  TDLOGI("Restore Const op in ascend_ir: name[%s]", op.name.c_str());
  if (!op.attrs.is_object() || !op.attrs.contains("value")) {
    TDLOGE("%s",
           MakeConverterNodeError(node, &op, "inline const missing value attr", NodeErrorContext("", op.name)).c_str());
    return INVALID_PARAM;
  }
  TensorHolder holder;
  const Status restore_status =
      GeWeightLoader::RestoreInlineConstTensor(op.attrs["value"], deps_.tensor_desc_mapper, op.name, holder);
  if (restore_status != SUCCESS) {
    TDLOGE("%s",
           MakeConverterNodeError(node, &op, "inline const restore error", NodeErrorContext("", op.name)).c_str());
    return restore_status;
  }
  const Status validate_status = ValidateConstOutputDesc(op, holder, deps_.tensor_desc_mapper);
  if (validate_status != SUCCESS) {
    TDLOGE("%s",
           MakeConverterNodeError(node, &op, "inline const output desc validation error", NodeErrorContext("", op.name))
               .c_str());
    return validate_status;
  }
  RegisteredValue const_value;
  TD_CHK_STATUS(CreateConstNode(*ctx_.graph, {op.name, holder}, deps_.tensor_desc_mapper, const_value));
  TD_CHK_STATUS(ctx_.registry.RegisterProducerOutput(op.name + ":0", const_value));
  return SUCCESS;
}

Status ConverterNodePlanner::ProcessBuildableOp(const ParsedNode &node, const ParsedAscendIrOp &op) {
  TD_ASSERT_NOTNULL(ctx_.graph, "ProcessBuildableOp failed: graph is null");
  std::vector<std::pair<std::string, ge::TensorDesc>> inputs;
  std::vector<PlannedEdgeBinding> bindings;
  inputs.reserve(op.inputs.size());
  bindings.reserve(op.inputs.size());
  for (size_t input_index = 0; input_index < op.inputs.size(); ++input_index) {
    const ParsedAscendIrOpDesc *desc = input_index < op.input_descs.size() ? &op.input_descs[input_index] : nullptr;
    if (desc == nullptr || desc->name.empty()) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "missing converter inputDesc name",
                                          NodeErrorContext("", "", op.inputs[input_index]))
                       .c_str());
      return INVALID_PARAM;
    }
    if (op.inputs[input_index].empty()) {
      if (IsOptionalUnfed(*desc)) {
        continue;
      }
      TDLOGE("%s", MakeConverterNodeError(node, &op, "converter input tensor ref is empty",
                                          NodeErrorContext("", desc->name, ""))
                       .c_str());
      return INVALID_PARAM;
    }
    td::GeTensorDescInfo desc_info;
    if (deps_.tensor_desc_mapper.MapAscendIrDesc(desc, nullptr, op.name + ":" + desc->name, desc_info) != SUCCESS) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "converter input desc mapping error",
                                          NodeErrorContext("", desc->name, op.inputs[input_index]))
                       .c_str());
      return INVALID_PARAM;
    }
    inputs.push_back({desc->name, MakeTensorDesc(desc_info)});
    bindings.push_back({ResolveBoundaryInputSourceRef(node, op.inputs[input_index]), desc->name, -1});
  }
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs;
  outputs.reserve(op.output_descs.size());
  for (const auto &desc : op.output_descs) {
    if (desc.name.empty()) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "missing converter outputDesc name").c_str());
      return INVALID_PARAM;
    }
    td::GeTensorDescInfo desc_info;
    if (deps_.tensor_desc_mapper.MapAscendIrDesc(&desc, nullptr, op.name + ":" + desc.name, desc_info) != SUCCESS) {
      TDLOGE("%s",
             MakeConverterNodeError(node, &op, "converter output desc mapping error", NodeErrorContext("", desc.name))
                 .c_str());
      return INVALID_PARAM;
    }
    outputs.push_back({desc.name, MakeTensorDesc(desc_info)});
  }
  std::vector<std::pair<std::string, ge::AttrValue>> attrs;
  if (op.attrs.is_object()) {
    attrs.reserve(op.attrs.size());
    for (const auto &[attr_name, attr_value_json] : op.attrs.items()) {
      bool ok = true;
      auto parsed = ParseAttrValue(attr_value_json, op.name + ":" + attr_name, ok);
      if (!ok) {
        TDLOGE("%s",
               MakeConverterNodeError(node, &op, "converter attr parse error", NodeErrorContext("", "", attr_name))
                   .c_str());
        return INVALID_PARAM;
      }
      if (parsed.has_value()) {
        attrs.push_back({attr_name, parsed.value()});
      }
    }
  }
  std::unique_ptr<ge::GNode> node_ptr;
  if (BuildRecoverableNode(*ctx_.graph, {op.type, op.name, inputs, outputs, attrs, op.name}, node_ptr) != SUCCESS) {
    TDLOGE("%s", MakeConverterNodeError(node, &op, "recoverable build error").c_str());
    return FAILED;
  }
  ge::GNode ge_node = *node_ptr;
  for (auto &binding : bindings) {
    if (LookupInputIndex(ge_node, binding.input_name, binding.dst_index, op.name) != SUCCESS) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "lookup input index error",
                                          NodeErrorContext("", binding.input_name, binding.src_ref, binding.dst_index))
                       .c_str());
      return INVALID_PARAM;
    }
    ctx_.deferred_edges.push_back({binding.src_ref, ge_node, binding.dst_index});
  }
  for (const auto &output : outputs) {
    int32_t output_index = 0;
    if (LookupOutputIndex(ge_node, output.first, output_index, op.name) != SUCCESS) {
      TDLOGE(
          "%s",
          MakeConverterNodeError(node, &op, "lookup output index error", NodeErrorContext("", output.first)).c_str());
      return INVALID_PARAM;
    }
    const std::string output_ref = op.name + ":" + std::to_string(output_index);
    if (ctx_.registry.RegisterProducerOutput(output_ref, {ge_node, output_index}) != SUCCESS) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "register output error",
                                          NodeErrorContext("", output.first, output_ref, output_index))
                       .c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ConverterNodePlanner::RegisterOutputMappings(const ParsedNode &node) {
  const ParsedAscendIr &ascend_ir = node.ascend_ir.value();
  const std::vector<std::string> node_output_names = CollectNodeOutputTensorNames(node);
  for (size_t index = 0; index < node_output_names.size(); ++index) {
    const std::string &fx_tensor_name = node_output_names[index];
    auto output_it = ascend_ir.output_mappings.find(fx_tensor_name);
    if (output_it == ascend_ir.output_mappings.end()) {
      output_it = ascend_ir.output_mappings.find(std::to_string(index));
    }
    if (output_it == ascend_ir.output_mappings.end()) {
      TDLOGE("%s",
             MakeConverterNodeError(node, nullptr, "mapping.outputs is incomplete", NodeErrorContext(fx_tensor_name))
                 .c_str());
      return INVALID_PARAM;
    }
    ctx_.deferred_outputs.push_back({fx_tensor_name, output_it->second});
  }
  return SUCCESS;
}

Status ConverterNodePlanner::Plan(const ParsedNode &node) {
  const ParsedAscendIr &ascend_ir = node.ascend_ir.value();
  for (const auto &op : ascend_ir.ops) {
    if (op.name.empty() || op.type.empty()) {
      continue;
    }
    if (IsConstOp(op)) {
      TD_CHK_STATUS(ProcessConstOp(node, op));
    } else {
      TD_CHK_STATUS(ProcessBuildableOp(node, op));
    }
  }
  TD_CHK_STATUS(RegisterOutputMappings(node));
  return SUCCESS;
}

Status ConverterNodePlanner::PlanNoOpWithMappings(const ParsedNode &node) {
  const std::vector<std::string> node_output_names = CollectNodeOutputTensorNames(node);
  for (const auto &output_name : node_output_names) {
    auto output_it = node.ascend_ir->output_mappings.find(output_name);
    if (output_it == node.ascend_ir->output_mappings.end()) {
      TDLOGE("%s", MakeConverterNodeError(node, nullptr, "NoOp converter mapping.outputs is incomplete",
                                          NodeErrorContext(output_name))
                       .c_str());
      return INVALID_PARAM;
    }
    ctx_.deferred_outputs.push_back({output_name, output_it->second});
  }
  return SUCCESS;
}

Status ConverterNodePlanner::PlanNoOpPassthrough(const ParsedNode &node) {
  static const std::unordered_map<std::string, std::string> kPassthroughInputMap = {
      {"torch.ops.aten.expand.default", "self"},
      {"torch.ops.aten.slice.Tensor", "self"},
      {"torch.ops.aten.copy.default", "src"},
  };
  auto it = kPassthroughInputMap.find(node.target);
  TD_ASSERT_TRUE(
      it != kPassthroughInputMap.end(), "%s",
      MakeConverterNodeError(node, nullptr, "NoOp converter passthrough unsupported for this operator").c_str());

  const std::string &passthrough_input_name = it->second;
  const ParsedNodeInput *passthrough_input = nullptr;
  for (const auto &input : node.inputs) {
    if (input.name == passthrough_input_name) {
      passthrough_input = &input;
      break;
    }
  }
  TD_ASSERT_NOTNULL(passthrough_input, "%s",
                    MakeConverterNodeError(node, nullptr, "NoOp converter passthrough input not found",
                                           NodeErrorContext("", passthrough_input_name))
                        .c_str());
  TD_ASSERT_TRUE(passthrough_input->input_kind == NodeInputKind::Tensor && !passthrough_input->tensor_name.empty(),
                 "%s",
                 MakeConverterNodeError(node, nullptr, "NoOp converter passthrough input must be a tensor",
                                        NodeErrorContext("", passthrough_input->name))
                     .c_str());

  TD_ASSERT_TRUE(node.outputs.size() == 1 && node.outputs[0].is_tensor, "%s",
                 MakeConverterNodeError(node, nullptr, "NoOp converter expects exactly one tensor output").c_str());

  const std::string &output_name = node.outputs[0].tensor_name;
  const ParsedTensorMeta *input_meta = deps_.model.FindTensorMeta(passthrough_input->tensor_name);
  const ParsedTensorMeta *output_meta = deps_.model.FindTensorMeta(output_name);
  TD_ASSERT_NOTNULL(input_meta, "%s",
                    MakeConverterNodeError(node, nullptr, "NoOp converter input tensor meta missing",
                                           NodeErrorContext(output_name, passthrough_input->tensor_name))
                        .c_str());
  TD_ASSERT_NOTNULL(output_meta, "%s",
                    MakeConverterNodeError(node, nullptr, "NoOp converter output tensor meta missing",
                                           NodeErrorContext(output_name, passthrough_input->tensor_name))
                        .c_str());
  TD_ASSERT_TRUE(IsSameTensorMeta(*input_meta, *output_meta), "%s",
                 MakeConverterNodeError(node, nullptr, "NoOp converter input/output tensor meta mismatch",
                                        NodeErrorContext(output_name, passthrough_input->tensor_name))
                     .c_str());

  ctx_.deferred_outputs.push_back({output_name, passthrough_input->tensor_name});
  return SUCCESS;
}

/* 处理 NoOp 场景：
 * torchair 对 expand/copy/slice 等算子做 symsize 优化：当输入和输出的符号形状相同、
 * 无需创建 broadcast 情况下，converter 直接返回源 tensor，
 * 不产生任何 GE op。此时 ascend_ir.ops 为 []，对应节点为 NoOp
 * 当前方案：
 * 1. output_mappings 非空 → PlanNoOpWithMappings：直接使用 mapping 中的 GE ref 做透传
 * 2. output_mappings 为空 → PlanNoOpPassthrough：白名单确定透传源 input name，
 *    再通过 tensor meta 校验后做 FX tensor alias 透传
 * */
Status ConverterNodePlanner::PlanNoOp(const ParsedNode &node) {
  if (node.ascend_ir.has_value() && !node.ascend_ir->output_mappings.empty()) {
    return PlanNoOpWithMappings(node);
  }
  if (!node.ascend_ir.has_value() || !node.ascend_ir->input_mappings.empty()) {
    TDLOGE(
        "%s",
        MakeConverterNodeError(node, nullptr, "NoOp converter with non-empty mapping.inputs is unsupported").c_str());
    return UNIMPLEMENTED;
  }
  return PlanNoOpPassthrough(node);
}

}  // namespace td
