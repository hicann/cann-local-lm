/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/plan_fallback_node.h"

#include <string>
#include <utility>
#include <vector>

#include <graph/attr_value.h>
#include <graph/gnode.h>
#include <graph/tensor.h>

#include "backend/aten_schema_resolver.h"
#include "backend/attr_parser.h"
#include "backend/ge_weight_loader.h"
#include "backend/graph_build_utils.h"
#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"
#include "common/pt2_type_constants.h"

namespace td {

FallbackNodePlanner::FallbackNodePlanner(GraphBuildContext &ctx, const GraphPlanningDeps &deps,
                                         ge::Placement tensor_placement)
    : ctx_(ctx), deps_(deps), tensor_placement_(tensor_placement) {}

Status FallbackNodePlanner::CollectFallbackAttrs(const ParsedNode &node, OrderedJson &out) {
  if (!node.ascend_ir.has_value()) {
    out = OrderedJson::object();
    return SUCCESS;
  }
  OrderedJson attrs = OrderedJson::object();
  for (const auto &op : node.ascend_ir->ops) {
    if (!op.attrs.is_object()) {
      continue;
    }
    for (const auto &[name, value] : op.attrs.items()) {
      if (attrs.contains(name)) {
        TDLOGE("Fallback attribute appears multiple times, context=%s:%s", context_.c_str(), name.c_str());
        return INVALID_PARAM;
      }
      attrs[name] = value;
    }
  }
  out = std::move(attrs);
  return SUCCESS;
}

Status FallbackNodePlanner::BuildScalarConstHolder(const ParsedNodeInput &input, TensorHolder &out) {
  TensorHolder holder;
  holder.format = pt2::kLayoutStrided;
  switch (input.input_kind) {
    case NodeInputKind::ScalarInt: {
      holder.datatype = pt2::kDtypeLong;
      holder.value = std::vector<int64_t>{input.raw_arg["as_int"].get<int64_t>()};
      break;
    }
    case NodeInputKind::ScalarFloat: {
      holder.datatype = pt2::kDtypeFloat;
      holder.value = std::vector<float>{static_cast<float>(input.raw_arg["as_float"].get<double>())};
      break;
    }
    case NodeInputKind::ScalarBool: {
      holder.datatype = pt2::kDtypeBool;
      holder.value = std::vector<uint8_t>{static_cast<uint8_t>(input.raw_arg["as_bool"].get<bool>() ? 1 : 0)};
      break;
    }
    case NodeInputKind::ScalarListInt: {
      holder.datatype = pt2::kDtypeLong;
      std::vector<int64_t> vals;
      for (const auto &v : input.raw_arg["as_ints"]) {
        vals.push_back(v.get<int64_t>());
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      out = std::move(holder);
      return SUCCESS;
    }
    case NodeInputKind::ScalarListFloat: {
      holder.datatype = pt2::kDtypeFloat;
      std::vector<float> vals;
      for (const auto &v : input.raw_arg["as_floats"]) {
        vals.push_back(static_cast<float>(v.get<double>()));
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      out = std::move(holder);
      return SUCCESS;
    }
    case NodeInputKind::ScalarListBool: {
      holder.datatype = pt2::kDtypeBool;
      std::vector<uint8_t> vals;
      for (const auto &v : input.raw_arg["as_bools"]) {
        vals.push_back(v.get<bool>() ? 1 : 0);
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      out = std::move(holder);
      return SUCCESS;
    }
    case NodeInputKind::Enum: {
      holder.datatype = pt2::kDtypeLong;
      holder.value = std::vector<int64_t>{input.raw_arg["as_scalar_type"].get<int64_t>()};
      break;
    }
    case NodeInputKind::ScalarString: {
      holder.datatype = pt2::kDtypeByte;
      std::string val = input.raw_arg["as_string"].get<std::string>();
      holder.dims = {static_cast<int64_t>(val.size())};
      holder.value = std::vector<uint8_t>(val.begin(), val.end());
      out = std::move(holder);
      return SUCCESS;
    }
    default:
      TDLOGE("Unsupported scalar input type, context=%s, input=%s, kind=%s", context_.c_str(), input.name.c_str(),
             NodeInputKindToString(input.input_kind));
      return UNIMPLEMENTED;
  }
  out = std::move(holder);
  return SUCCESS;
}

Status FallbackNodePlanner::ResolveIrDef(const ParsedNode &node) {
  context_ = NodeDebugContext(node);
  if (node.outputs.empty()) {
    TDLOGE("%s", MakeFallbackNodeError(node, "", "no outputs").c_str());
    return INVALID_PARAM;
  }
  td::AtenTargetInfo target;
  if (deps_.schema_resolver.ParseTarget(node.effective_schema_target, context_, target) != SUCCESS) {
    TDLOGE("%s", MakeFallbackNodeError(node, "", "target parse error").c_str());
    return INVALID_PARAM;
  }
  op_type_ = deps_.schema_resolver.MakeFallbackOpType(target);
  ir_def_ = ctx_.ir_cache.Resolve(op_type_, context_);
  if (ir_def_ == nullptr) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type_, "registered IR query error").c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FallbackNodePlanner::RegisterScalarInputs(const ParsedNode &node) {
  TD_ASSERT_NOTNULL(ctx_.graph, "RegisterScalarInputs failed: graph is null");
  for (const auto &input : node.inputs) {
    if (!IsScalarInput(input)) continue;
    std::string const_name = node.name + "_const_" + input.name;
    TensorHolder holder;
    if (BuildScalarConstHolder(input, holder) != SUCCESS) {
      TDLOGE("%s",
             MakeFallbackNodeError(node, op_type_, "scalar const build error", NodeErrorContext(input.name)).c_str());
      return INVALID_PARAM;
    }
    RegisteredValue const_value;
    if (CreateConstNode(*ctx_.graph, {const_name, holder, ge::kPlacementHost}, deps_.tensor_desc_mapper, const_value) !=
        SUCCESS) {
      TDLOGE("%s",
             MakeFallbackNodeError(node, op_type_, "scalar const node creation error", NodeErrorContext(input.name))
                 .c_str());
      return FAILED;
    }
    if (ctx_.registry.RegisterProducerOutput(const_name, const_value) != SUCCESS) {
      TDLOGE("%s",
             MakeFallbackNodeError(node, op_type_, "register scalar const output error", NodeErrorContext(input.name))
                 .c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FallbackNodePlanner::BuildInputDefs(const ParsedNode &node) {
  for (const auto &input : node.inputs) {
    if (input.input_kind == NodeInputKind::Tensor || input.input_kind == NodeInputKind::TensorList ||
        IsScalarInput(input) || IsSymInput(input) || input.input_kind == NodeInputKind::None) {
      continue;
    }
    TDLOGE("Unsupported fallback input: name=%s, input_kind=%s", input.name.c_str(),
           NodeInputKindToString(input.input_kind));
    return UNIMPLEMENTED;
  }
  // None inputs (e.g. Optional[Tensor] = None) still occupy an IR index slot
  // because OPTIONAL_INPUT is declared in REG_OP. Skip creation but advance
  // ir_index so subsequent inputs map to the correct IR position.
  if (node.inputs.size() > ir_def_->input_names.size()) {
    TDLOGE("Fallback JSON inputs exceed registered IR input count, context=%s", context_.c_str());
    return INVALID_PARAM;
  }
  size_t ir_index = 0;
  for (const auto &input : node.inputs) {
    if (input.input_kind == NodeInputKind::None) {
      ir_index++;
      continue;
    }
    const std::string &ir_name = ir_def_->input_names[ir_index++];
    if (input.input_kind == NodeInputKind::Tensor) {
      const ParsedTensorMeta *input_meta = deps_.model.FindTensorMeta(input.tensor_name);
      if (input_meta == nullptr) {
        TDLOGE("Fallback input tensor meta missing for single tensor input, context=%s, tensor=%s", context_.c_str(),
               input.tensor_name.c_str());
        return INVALID_PARAM;
      }
      // HostCpu 用户 Data 输入的 D2H staging 由 GE 按 desc 实际 dtype 计算长度
      // （CalcTensorSizeFromStorage → CalcAlignedSizeByShape → GetSizeInBytes），
      // 非 4 字节 dtype（int64/fp16/bf16 等）无需再做 fail-fast。
      td::GeTensorDescInfo desc_info;
      TD_CHK_STATUS(deps_.tensor_desc_mapper.MapPt2TensorMeta(*input_meta, context_ + ":" + ir_name, desc_info));
      ge::TensorDesc desc = MakeTensorDesc(desc_info);
      desc.SetPlacement(tensor_placement_);
      inputs_.push_back({ir_name, desc});
      bindings_.push_back({input.tensor_name, ir_name, -1});
    } else if (input.input_kind == NodeInputKind::TensorList) {
      for (size_t ti = 0; ti < input.tensor_names.size(); ++ti) {
        const std::string &tn = input.tensor_names[ti];
        const ParsedTensorMeta *meta = deps_.model.FindTensorMeta(tn);
        if (meta == nullptr) {
          TDLOGE("Fallback input tensor meta missing, context=%s, tensor=%s", context_.c_str(), tn.c_str());
          return INVALID_PARAM;
        }
        td::GeTensorDescInfo di;
        TD_CHK_STATUS(deps_.tensor_desc_mapper.MapPt2TensorMeta(*meta, context_ + ":" + ir_name, di));
        const std::string inst = ir_name + std::to_string(ti);
        ge::TensorDesc desc = MakeTensorDesc(di);
        desc.SetPlacement(tensor_placement_);
        inputs_.push_back({inst, desc});
        bindings_.push_back({tn, inst, -1});
      }
    } else if (IsScalarInput(input)) {
      ge::DataType dtype = GetScalarDtype(input);
      std::vector<int64_t> shape_dims;
      if (input.input_kind == NodeInputKind::ScalarListInt && input.raw_arg.contains("as_ints") &&
          input.raw_arg["as_ints"].is_array()) {
        shape_dims = {static_cast<int64_t>(input.raw_arg["as_ints"].size())};
      } else if (input.input_kind == NodeInputKind::ScalarListFloat && input.raw_arg.contains("as_floats") &&
                 input.raw_arg["as_floats"].is_array()) {
        shape_dims = {static_cast<int64_t>(input.raw_arg["as_floats"].size())};
      } else if (input.input_kind == NodeInputKind::ScalarListBool && input.raw_arg.contains("as_bools") &&
                 input.raw_arg["as_bools"].is_array()) {
        shape_dims = {static_cast<int64_t>(input.raw_arg["as_bools"].size())};
      }
      inputs_.push_back({ir_name, ge::TensorDesc(ge::Shape(shape_dims), ge::FORMAT_ND, dtype)});
      bindings_.push_back({node.name + "_const_" + input.name, ir_name, -1});
    } else if (IsSymInput(input)) {
      ge::DataType dtype = ge::DT_INT64;
      if (input.input_kind == NodeInputKind::SymFloat || input.input_kind == NodeInputKind::SymListFloat) {
        dtype = ge::DT_FLOAT;
      } else if (input.input_kind == NodeInputKind::SymBool || input.input_kind == NodeInputKind::SymListBool) {
        dtype = ge::DT_BOOL;
      }
      std::vector<int64_t> shape_dims;
      if (input.input_kind == NodeInputKind::SymListInt || input.input_kind == NodeInputKind::SymListFloat ||
          input.input_kind == NodeInputKind::SymListBool) {
        shape_dims = {static_cast<int64_t>(input.tensor_names.size())};
      }
      inputs_.push_back({ir_name, ge::TensorDesc(ge::Shape(shape_dims), ge::FORMAT_ND, dtype)});
      if (input.input_kind == NodeInputKind::SymListInt || input.input_kind == NodeInputKind::SymListFloat ||
          input.input_kind == NodeInputKind::SymListBool) {
        for (size_t ti = 0; ti < input.tensor_names.size(); ++ti) {
          bindings_.push_back({input.tensor_names[ti], ir_name + std::to_string(ti), -1});
        }
      } else {
        bindings_.push_back({input.tensor_name, ir_name, -1});
      }
    }
  }
  return SUCCESS;
}

Status FallbackNodePlanner::BuildOutputDefs(const ParsedNode &node) {
  if (node.outputs.size() != ir_def_->output_names.size()) {
    TDLOGE("Fallback JSON outputs do not match registered IR output count, context=%s", context_.c_str());
    return INVALID_PARAM;
  }
  for (size_t oi = 0; oi < node.outputs.size(); ++oi) {
    const ParsedNodeOutput &output = node.outputs[oi];
    const std::string &ir_name = ir_def_->output_names[oi];
    if (!output.is_tensor && !output.is_tensor_list) {
      TDLOGE("Fallback node only supports tensor/tensor-list outputs, context=%s:%s", context_.c_str(),
             ir_name.c_str());
      return UNIMPLEMENTED;
    }
    const std::vector<std::string> tensor_names = CollectOutputTensorNames(output);
    for (size_t ti = 0; ti < tensor_names.size(); ++ti) {
      const std::string &tn = tensor_names[ti];
      const ParsedTensorMeta *meta = deps_.model.FindTensorMeta(tn);
      if (meta == nullptr) {
        TDLOGE("Fallback output tensor meta missing, tensor=%s", tn.c_str());
        return INVALID_PARAM;
      }
      td::GeTensorDescInfo di;
      TD_CHK_STATUS(deps_.tensor_desc_mapper.MapPt2TensorMeta(*meta, context_ + ":" + ir_name, di));
      const std::string inst = output.is_tensor ? ir_name : ir_name + std::to_string(ti);
      outputs_.push_back({inst, MakeTensorDesc(di)});
    }
  }
  return SUCCESS;
}

Status FallbackNodePlanner::BuildAttrs(const ParsedNode &node) {
  OrderedJson attrs_json;
  TD_CHK_STATUS(CollectFallbackAttrs(node, attrs_json));
  if (!attrs_json.is_object()) {
    TDLOGE("Fallback attrs must be object, context=%s", context_.c_str());
    return INVALID_PARAM;
  }
  attrs_.reserve(attrs_json.size());
  for (const auto &[name, value] : attrs_json.items()) {
    bool ok = true;
    auto parsed = ParseAttrValue(value, context_ + ":" + name, ok);
    if (!ok) {
      return INVALID_PARAM;
    }
    if (parsed.has_value()) {
      attrs_.push_back({name, parsed.value()});
    }
  }

  return SUCCESS;
}

Status FallbackNodePlanner::BuildAndRegisterNode(const ParsedNode &node) {
  TD_ASSERT_NOTNULL(ctx_.graph, "BuildAndRegisterNode failed: graph is null");
  std::unique_ptr<ge::GNode> node_ptr;
  if (BuildRecoverableNode(*ctx_.graph, {op_type_, node.name, inputs_, outputs_, attrs_, context_}, node_ptr) !=
      SUCCESS) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type_, "recoverable build error").c_str());
    return FAILED;
  }
  ge::GNode ge_node = *node_ptr;
  for (auto &binding : bindings_) {
    if (LookupInputIndex(ge_node, binding.input_name, binding.dst_index, context_) != SUCCESS) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type_, "lookup input index error",
                                         NodeErrorContext("", binding.input_name, binding.src_ref, binding.dst_index))
                       .c_str());
      return INVALID_PARAM;
    }
  }
  for (const auto &binding : bindings_) {
    ctx_.deferred_edges.push_back({binding.src_ref, ge_node, binding.dst_index});
  }
  const std::vector<std::string> output_tensor_names = CollectNodeOutputTensorNames(node);
  for (size_t index = 0; index < output_tensor_names.size(); ++index) {
    int32_t output_index = 0;
    if (LookupOutputIndex(ge_node, outputs_[index].first, output_index, context_) != SUCCESS) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type_, "lookup output index error",
                                         NodeErrorContext(output_tensor_names[index], outputs_[index].first))
                       .c_str());
      return INVALID_PARAM;
    }
    const std::string output_ref = node.name + ":" + std::to_string(output_index);
    if (ctx_.registry.RegisterProducerOutput(output_ref, {ge_node, output_index}) != SUCCESS) {
      TDLOGE("%s", MakeFallbackNodeError(
                       node, op_type_, "register output error",
                       NodeErrorContext(output_tensor_names[index], outputs_[index].first, output_ref, output_index))
                       .c_str());
      return FAILED;
    }
    ctx_.deferred_outputs.push_back({output_tensor_names[index], output_ref});
    if (node.ascend_ir.has_value()) {
      auto it = node.ascend_ir->output_mappings.find(output_tensor_names[index]);
      if (it != node.ascend_ir->output_mappings.end() && !it->second.empty()) {
        ctx_.deferred_outputs.push_back({it->second, output_ref});
      }
    }
  }
  return SUCCESS;
}

Status FallbackNodePlanner::Plan(const ParsedNode &node) {
  TD_CHK_STATUS(ResolveIrDef(node));
  TD_CHK_STATUS(RegisterScalarInputs(node));
  TD_CHK_STATUS(BuildInputDefs(node));
  TD_CHK_STATUS(BuildOutputDefs(node));
  TD_CHK_STATUS(BuildAttrs(node));
  TD_CHK_STATUS(BuildAndRegisterNode(node));
  return SUCCESS;
}

}  // namespace td
