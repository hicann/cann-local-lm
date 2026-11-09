/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fallback_codegen/fallback_codegen.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <cstdio>
#include <array>
#include <set>
#include <tuple>
#include <unistd.h>
#include <dlfcn.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include "common/logging/logging.h"
#include "common/op_target_utils.h"
#include "common/pt2_type_constants.h"

namespace torch_delegate::fallback {

using json = nlohmann::json;

namespace {

bool IsTensorType(c10::TypePtr type) {
  if (type->isSubtypeOf(c10::TensorType::get())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type && opt_type->getElementType()->isSubtypeOf(c10::TensorType::get())) return true;
  return false;
}

bool IsScalarType(c10::TypePtr type) {
  if (type->str() == "Scalar") return true;
  if (type->isSubtypeOf(c10::IntType::get())) return true;
  if (type->isSubtypeOf(c10::FloatType::get())) return true;
  if (type->isSubtypeOf(c10::BoolType::get())) return true;
  if (type->isSubtypeOf(c10::StringType::get())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type) return IsScalarType(opt_type->getElementType());
  return false;
}

bool IsOptionalType(c10::TypePtr type) {
  return type->castRaw<c10::OptionalType>() != nullptr;
}

c10::TypePtr UnwrapOptional(c10::TypePtr type) {
  if (auto opt = type->castRaw<c10::OptionalType>()) {
    return opt->getElementType();
  }
  return type;
}

bool IsIntType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::IntType::get());
}

bool IsFloatType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::FloatType::get());
}

bool IsBoolType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::BoolType::get());
}

bool IsStringType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::StringType::get());
}

bool IsScalarPolyType(c10::TypePtr type) {
  return UnwrapOptional(type)->str() == "Scalar";
}

bool IsIntListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::IntType::get());
  }
  return false;
}

bool IsFloatListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::FloatType::get());
  }
  return false;
}

bool IsBoolListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::BoolType::get());
  }
  return false;
}

bool IsScalarListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->str() == "Scalar";
  }
  return false;
}

bool IsEnumType(c10::TypePtr type) {
  const std::string &s = UnwrapOptional(type)->str();
  return s == "ScalarType" || s == "Layout" || s == "Device" || s == "MemoryFormat";
}

bool IsTensorListType(c10::TypePtr type) {
  if (type->isSubtypeOf(c10::ListType::ofTensors())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type) return IsTensorListType(opt_type->getElementType());
  return false;
}

bool IsOptional(InputKind kind) {
  return kind == InputKind::OPTIONAL_TENSOR || kind == InputKind::INT_OPT || kind == InputKind::FLOAT_OPT ||
         kind == InputKind::BOOL_OPT || kind == InputKind::STRING_OPT || kind == InputKind::INT_LIST_OPT ||
         kind == InputKind::FLOAT_LIST_OPT || kind == InputKind::BOOL_LIST_OPT || kind == InputKind::SCALAR_OPT ||
         kind == InputKind::SCALAR_TYPE_OPT;
}

bool IsValueDependency(InputKind kind) {
  return kind == InputKind::INT || kind == InputKind::INT_OPT || kind == InputKind::FLOAT ||
         kind == InputKind::FLOAT_OPT || kind == InputKind::BOOL || kind == InputKind::BOOL_OPT ||
         kind == InputKind::STRING || kind == InputKind::STRING_OPT || kind == InputKind::INT_LIST ||
         kind == InputKind::INT_LIST_OPT || kind == InputKind::FLOAT_LIST || kind == InputKind::FLOAT_LIST_OPT ||
         kind == InputKind::BOOL_LIST || kind == InputKind::BOOL_LIST_OPT || kind == InputKind::SCALAR ||
         kind == InputKind::SCALAR_OPT || kind == InputKind::SCALAR_LIST || kind == InputKind::SCALAR_TYPE ||
         kind == InputKind::SCALAR_TYPE_OPT;
}

bool HasDefaultValue(const c10::Argument &arg) {
  return arg.default_value().has_value();
}

std::string ToCamelCase(const std::string &snake_case) {
  std::string result;
  bool capitalize_next = true;
  for (char c : snake_case) {
    if (c == '_') {
      capitalize_next = true;
    } else {
      if (capitalize_next) {
        result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        capitalize_next = false;
      } else {
        result += c;
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// HostCpu 形状推导数据源（ShapeInferOp 生成）
// ---------------------------------------------------------------------------

// 从 arg/output JSON 值中收集张量名（覆盖 tensor/tensor_list/optional 变体）。
void CollectArgTensorNames(const json &value, std::vector<std::string> &out) {
  if (!value.is_object()) {
    return;
  }
  if (value.contains("as_tensor") && value["as_tensor"].is_object() && value["as_tensor"].contains("name") &&
      value["as_tensor"]["name"].is_string()) {
    out.push_back(value["as_tensor"]["name"].get<std::string>());
    return;
  }
  if (value.contains("as_tensors") && value["as_tensors"].is_array()) {
    for (const auto &t : value["as_tensors"]) {
      if (t.is_object() && t.contains("name") && t["name"].is_string()) {
        out.push_back(t["name"].get<std::string>());
      }
    }
    return;
  }
  if (value.contains("as_optional_tensor") && value["as_optional_tensor"].is_object()) {
    CollectArgTensorNames(value["as_optional_tensor"], out);
    return;
  }
  if (value.contains("as_optional_tensors") && value["as_optional_tensors"].is_array()) {
    for (const auto &t : value["as_optional_tensors"]) {
      CollectArgTensorNames(t, out);
    }
    return;
  }
}

// 匹配 "Symbol('<name>')"（torch <= 2.8）或 torch >= 2.9 实际序列化格式
// "Symbol('<name>', positive=True, integer=True)"（name 仅限 [A-Za-z0-9_]+，
// assumptions 参数为 bool 标记），用于区分纯符号维与复合表达式维（如
// "Symbol('s0') + 1"、"2*Symbol('s0')"）。assumptions 只描述符号取值域，
// 不改变"该维是动态符号"的语义，因此携带参数仍视为纯符号；复合表达式既不能
// 内联为静态维，也不能按符号透传，必须显式解析失败，禁止静默降级。
bool ParsePureSymbolExpr(const std::string &expr, std::string &symbol) {
  static const std::string kPrefix = "Symbol('";
  if (expr.size() <= kPrefix.size() || expr.compare(0, kPrefix.size(), kPrefix) != 0) {
    return false;
  }
  const size_t name_start = kPrefix.size();
  const size_t quote = expr.find('\'', name_start);
  if (quote == std::string::npos || quote == name_start) {
    return false;
  }
  symbol = expr.substr(name_start, quote - name_start);
  for (char c : symbol) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }
  // 符号名之后仅允许两种收尾："')"（纯符号）或 "',<assumptions>)"
  // （torch >= 2.9），assumptions 内不允许再出现引号（排除嵌套表达式）。
  const std::string tail = expr.substr(quote);
  if (tail == "')") {
    return true;
  }
  return tail.size() > 3 && tail.compare(0, 2, "',") == 0 && tail.back() == ')' &&
         tail.find('\'', 2) == std::string::npos;
}

// 解析 EP JSON tensor_meta 为 ShapeMeta；动态维符号从 as_expr.expr_str 提取，
// 仅接受纯符号表达式全匹配（复合表达式解析失败，交由调用方显式报错/降权处理）。
bool ParseShapeMetaJson(const json &meta, ShapeMeta &out) {
  if (!meta.is_object() || !meta.contains("dtype") || !meta["dtype"].is_number_integer() || !meta.contains("sizes") ||
      !meta["sizes"].is_array()) {
    return false;
  }
  out.dtype = meta["dtype"].get<int64_t>();
  out.sizes.clear();
  out.dynamic_symbols.assign(meta["sizes"].size(), "");
  for (size_t i = 0; i < meta["sizes"].size(); ++i) {
    const auto &dim = meta["sizes"][i];
    if (dim.is_number_integer()) {
      out.sizes.push_back(dim.get<int64_t>());
    } else if (dim.is_object() && dim.contains("as_int") && dim["as_int"].is_number_integer()) {
      out.sizes.push_back(dim["as_int"].get<int64_t>());
    } else if (dim.is_object() && dim.contains("as_expr") && dim["as_expr"].is_object() &&
               dim["as_expr"].contains("expr_str") && dim["as_expr"]["expr_str"].is_string()) {
      const std::string expr_str = dim["as_expr"]["expr_str"].get<std::string>();
      std::string symbol;
      if (!ParsePureSymbolExpr(expr_str, symbol)) {
        // 复合 SymInt 表达式（如 "Symbol('s0') + 1"）或纯算术 repr：按符号透传会
        // 静默生成错误 InferShape，按 -1 内联会生成非法静态维，必须显式解析失败。
        return false;
      }
      out.sizes.push_back(-1);
      out.dynamic_symbols[i] = symbol;
    } else {
      return false;
    }
  }
  return true;
}

// 输出结构签名：静态维直接内联取值，动态维按"解析后的透传来源 (input_meta_index, dim)"
// 比较，而非原始符号名。跨实例符号重命名（如 s0 vs s1，结构等价）不再误报；
// 同符号不同解析路径（如 s0 ← in0.dim1 vs s0 ← in0.dim0）能被正确拒绝。
// 解析规则与 GenerateShapeInferMethods 的 symbol_map 构建一致（首个同符号输入维生效，
// input_metas 顺序即 schema tensor 参数顺序）。
std::string OutputStructureSignature(const OpTarget &target) {
  std::ostringstream oss;
  for (const auto &meta : target.output_metas) {
    oss << meta.dtype << "[";
    for (size_t i = 0; i < meta.sizes.size(); ++i) {
      if (i > 0) oss << ",";
      const auto &sym = meta.dynamic_symbols[i];
      if (sym.empty()) {
        oss << meta.sizes[i];
        continue;
      }
      bool resolved = false;
      for (size_t k = 0; k < target.input_metas.size() && !resolved; ++k) {
        const auto &in_meta = target.input_metas[k];
        for (size_t d = 0; d < in_meta.dynamic_symbols.size(); ++d) {
          if (in_meta.dynamic_symbols[d] == sym) {
            oss << "in" << k << "d" << d;
            resolved = true;
            break;
          }
        }
      }
      if (!resolved) {
        // 不可解析符号：代表实例在生成期显式报错；签名组件保持可比较。
        oss << "unres:" << sym;
      }
    }
    oss << "];";
  }
  return oss.str();
}

}  // namespace

void FallbackCodegen::CollectShapeInfo(const json &node, const json *tensor_values, OpTarget &target) {
  target.has_shape_info = false;
  target.input_metas.clear();
  target.output_metas.clear();
  target.input_arg_tensor_counts.clear();
  if (tensor_values == nullptr) {
    // 简化格式（如单测 fixture）无 tensor_values：此处不失败，延后到生成期严格报错
    // （GenerateGeOpClass 要求 HostCpu 算子必须携带 shape info）。
    TDLOGW(
        "CollectShapeInfo: CPU op '%s' has no tensor_values in model JSON; "
        "ShapeInferOp generation will fail if this target reaches codegen",
        target.target.c_str());
    return;
  }
  // 返回值区分三种情形，便于给出精确报错：
  // kOk 成功；kMissing tensor_values 中无此名字（如常量特化）；
  // kInvalid meta 存在但不可解析（复合 SymInt 表达式等，禁止静默降级）。
  enum class MetaLookup { kOk, kMissing, kInvalid };
  auto lookup = [&](const std::string &name, ShapeMeta &meta) {
    auto it = tensor_values->find(name);
    if (it == tensor_values->end()) {
      return MetaLookup::kMissing;
    }
    return ParseShapeMetaJson(*it, meta) ? MetaLookup::kOk : MetaLookup::kInvalid;
  };

  // 收集输入条目（每个条目对应一个 schema 参数，含 0..N 个张量名）。
  // auto_functionalized_v2 节点的 inputs[0] 是被包裹算子占位（0 张量），
  // _all_bases / *_base_index 是 functionalization 记录字段，均不是内层
  // schema 参数；且原始顺序（sin/cos 在 self 之前）与内层算子 schema 顺序
  // （self 排第一）不一致。这里与 parse_functionalized.cc 的运行时归一化
  // 保持一致：跳过占位条目，_all_bases 中被 *_base_index 选中的基础张量
  // （内层 self 类参数）排最前（顺序即 *_base_index 的出现顺序），其余参数
  // 保持原顺序。InferShape 的 IR 输入序号按内层 schema 展开，若两侧顺序
  // 不一致，动态维透传会绑错输入。
  const json &inputs = node.value("inputs", json::array());
  const bool is_functionalized = node.value("target", "") == "torch.ops.higher_order.auto_functionalized_v2";
  std::vector<std::string> all_bases;
  std::vector<int64_t> base_indices;
  std::vector<std::vector<std::string>> ordered_entries;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    std::vector<std::string> names;
    CollectArgTensorNames(input.value("arg", json::object()), names);
    if (is_functionalized) {
      const std::string input_name = input.value("name", "");
      const auto &arg = input.contains("arg") && input["arg"].is_object() ? input["arg"] : json::object();
      if (i == 0 && arg.contains("as_operator")) {
        continue;  // 被包裹算子占位，不参与内层 schema 对齐
      }
      if (input_name == "_all_bases") {
        all_bases = std::move(names);
        continue;
      }
      static constexpr std::string_view kBaseIndexSuffix = "_base_index";
      if (input_name.size() >= kBaseIndexSuffix.size() &&
          input_name.compare(input_name.size() - kBaseIndexSuffix.size(), kBaseIndexSuffix.size(), kBaseIndexSuffix) ==
              0) {
        if (!arg.contains("as_int") || !arg["as_int"].is_number_integer()) {
          throw FallbackCodegenError("functionalized node wrapping '" + target.target + "' has non-integer input '" +
                                     input_name + "'; cannot align inputs with the wrapped op schema");
        }
        base_indices.push_back(arg["as_int"].get<int64_t>());
        continue;
      }
    }
    ordered_entries.push_back(std::move(names));
  }
  if (is_functionalized) {
    std::vector<std::vector<std::string>> base_entries;
    base_entries.reserve(base_indices.size());
    for (const int64_t base_index : base_indices) {
      if (base_index < 0 || static_cast<size_t>(base_index) >= all_bases.size()) {
        throw FallbackCodegenError("functionalized node wrapping '" + target.target + "' has _base_index " +
                                   std::to_string(base_index) + " out of range of _all_bases (size " +
                                   std::to_string(all_bases.size()) + ")");
      }
      base_entries.push_back({all_bases[static_cast<size_t>(base_index)]});
    }
    ordered_entries.insert(ordered_entries.begin(), std::make_move_iterator(base_entries.begin()),
                           std::make_move_iterator(base_entries.end()));
  }

  for (const auto &names : ordered_entries) {
    target.input_arg_tensor_counts.push_back(names.size());
    for (const auto &name : names) {
      ShapeMeta meta;
      MetaLookup st = lookup(name, meta);
      if (st == MetaLookup::kInvalid) {
        // 输入 meta 不可解析（复合表达式）：保留空位（不参与符号透传匹配），
        // 引用其符号的输出维会在生成期显式报错。
        TDLOGW(
            "CollectShapeInfo: CPU op '%s' input tensor '%s' has an unparseable tensor_meta "
            "(compound SymInt expr is not supported); its dims won't participate in symbol passthrough",
            target.target.c_str(), name.c_str());
        meta.dtype = -1;
      } else if (st == MetaLookup::kMissing) {
        // 输入 meta 缺失（如常量特化）：保留空位，不参与符号透传匹配。
        meta.dtype = -1;
      }
      target.input_metas.push_back(std::move(meta));
    }
  }
  for (const auto &output : node.value("outputs", json::array())) {
    std::vector<std::string> names;
    CollectArgTensorNames(output, names);
    for (const auto &name : names) {
      ShapeMeta meta;
      MetaLookup st = lookup(name, meta);
      if (st == MetaLookup::kInvalid) {
        throw FallbackCodegenError("CPU op '" + target.target + "' output tensor '" + name +
                                   "' has an unparseable tensor_meta: only static dims and pure Symbol('x') "
                                   "expressions are supported; compound SymInt expressions are data-dependent "
                                   "shapes; cannot generate ShapeInferOp");
      }
      if (st == MetaLookup::kMissing) {
        throw FallbackCodegenError("CPU op '" + target.target + "' output tensor '" + name +
                                   "' has no tensor_meta; cannot generate ShapeInferOp");
      }
      target.output_metas.push_back(std::move(meta));
    }
  }
  if (target.output_metas.empty()) {
    throw FallbackCodegenError("CPU op '" + target.target +
                               "' has no tensor outputs; "
                               "HostCpu backend only supports tensor outputs");
  }
  target.has_shape_info = true;
}

std::string FallbackCodegen::GenerateGeOpName(const std::string &op_name, const std::string &overload) {
  bool has_trailing_underscore = !op_name.empty() && op_name.back() == '_';
  std::string base_op_name = has_trailing_underscore ? op_name.substr(0, op_name.size() - 1) : op_name;

  std::string ge_op_name = ToCamelCase(base_op_name);

  if (!overload.empty()) {
    ge_op_name += ToCamelCase(overload);
  } else if (has_trailing_underscore) {
    ge_op_name += "Inplace";
  }

  return ge_op_name;
}

std::string FallbackCodegen::GenerateGeOpIdentifier(const std::string &full_op_name, const std::string &overload) {
  std::string id = full_op_name;
  size_t pos = id.find("::");
  if (pos != std::string::npos) {
    id.replace(pos, 2, "__");
  }
  if (!overload.empty()) {
    id += "_" + overload;
  }
  return id;
}

std::pair<std::string, std::string> FallbackCodegen::SplitNamespaceOp(const std::string &full_op) {
  // Dispatcher format: namespace::op_name (e.g., "aten::matmul")
  size_t colon_pos = full_op.find("::");
  if (colon_pos == std::string::npos) {
    return {"", full_op};
  }
  return {full_op.substr(0, colon_pos), full_op.substr(colon_pos + 2)};
}

OpTarget FallbackCodegen::ParseTargetFromJson(const json &node) {
  OpTarget target;
  target.target = node["target"].get<std::string>();

  // 读取执行 Device 归属标注；缺失按 "npu" 处理（老 epair 兼容）。
  // 未知取值直接报错，不静默跳过，避免按错误 Device 生成算子。
  if (node.contains("ascend_ir") && node["ascend_ir"].is_object() && node["ascend_ir"].contains("device")) {
    if (!node["ascend_ir"]["device"].is_string()) {
      throw FallbackCodegenError("node '" + target.target + "' ascend_ir.device must be a string");
    }
    target.device = node["ascend_ir"]["device"].get<std::string>();
    if (target.device != "npu" && target.device != "cpu") {
      throw FallbackCodegenError("node '" + target.target + "' has unknown ascend_ir.device '" + target.device +
                                 "', expect 'npu' or 'cpu'");
    }
  }

  // Handle Higher-Order Operators (HOPs)
  // auto_functionalized and auto_functionalized_v2 wrap mutable operators
  // The real operator name is in inputs[0].arg.as_operator
  // Example JSON structure:
  //   {
  //     "target": "torch.ops.higher_order.auto_functionalized_v2",
  //     "inputs": [
  //       {"name": "", "arg": {"as_operator": "torch.ops.xllm_ascend.rope_in_place.default"}, "kind": 1},
  //       ...
  //     ]
  //   }
  if (target.target == "torch.ops.higher_order.auto_functionalized" ||
      target.target == "torch.ops.higher_order.auto_functionalized_v2") {
    TDLOGD("Detected HOP: %s, extracting real operator", target.target.c_str());

    if (node.contains("inputs") && node["inputs"].is_array() && node["inputs"].size() > 0) {
      const auto &first_input = node["inputs"][0];
      if (first_input.contains("arg") && first_input["arg"].is_object() && first_input["arg"].contains("as_operator") &&
          first_input["arg"]["as_operator"].is_string()) {
        // Replace target with real operator name from as_operator
        target.target = first_input["arg"]["as_operator"].get<std::string>();
        // 节点 inputs 含算子占位与 functionalization 记录字段，原始顺序与内层
        // 算子 schema 不一致；CollectShapeInfo 会按运行时归一化顺序重建输入表。
        TDLOGD("HOP unwrapped: real operator = %s", target.target.c_str());
      } else {
        TDLOGE("auto_functionalized node missing 'as_operator' in first input");
        throw FallbackCodegenError(
            "auto_functionalized node missing 'as_operator' in first input. "
            "Expected: inputs[0].arg.as_operator = \"torch.ops.xxx.op_name.default\"");
      }
    } else {
      TDLOGE("auto_functionalized node has no inputs or invalid inputs structure");
      throw FallbackCodegenError("auto_functionalized node has no inputs or invalid inputs structure");
    }
  }

  // FX graph 格式：torch.ops.<namespace>.<op_name>.<overload>，拆分规则见
  // common/op_target_utils.h（"default" overload 映射为空串）。
  const td::OpTargetParts parts = td::ParseOpTarget(target.target);
  target.namespace_name = parts.namespace_name;
  target.op_name = parts.op_name;
  target.overload = parts.overload;

  return target;
}

std::vector<OpTarget> FallbackCodegen::ParseTargetsFromJson(const json &model) {
  std::vector<OpTarget> targets;

  // Try two possible paths for nodes:
  // 1. Direct "nodes" field (simplified format for test fixtures)
  // 2. ExportedProgram format: "graph_module.graph.nodes" (standard EP JSON)
  json nodes;
  if (model.contains("nodes") && model["nodes"].is_array()) {
    nodes = model["nodes"];
    TDLOGD("Found nodes in direct 'nodes' field (simplified format)");
  } else if (model.contains("graph_module") && model["graph_module"].is_object() &&
             model["graph_module"].contains("graph") && model["graph_module"]["graph"].is_object() &&
             model["graph_module"]["graph"].contains("nodes") && model["graph_module"]["graph"]["nodes"].is_array()) {
    nodes = model["graph_module"]["graph"]["nodes"];
    TDLOGD("Found nodes in graph_module.graph.nodes (ExportedProgram format)");
  } else {
    TDLOGW("No nodes found in JSON, neither 'nodes' nor 'graph_module.graph.nodes' exists");
    return targets;
  }

  TDLOGD("Processing %zu nodes from JSON", nodes.size());
  size_t fallback_count = 0;

  // tensor_values 查找表：HostCpu 节点的 InferShape 数据源（EP JSON tensor_meta）。
  const json *tensor_values = nullptr;
  if (model.contains("graph_module") && model["graph_module"].is_object() && model["graph_module"].contains("graph") &&
      model["graph_module"]["graph"].is_object() && model["graph_module"]["graph"].contains("tensor_values") &&
      model["graph_module"]["graph"]["tensor_values"].is_object()) {
    tensor_values = &model["graph_module"]["graph"]["tensor_values"];
  } else if (model.contains("tensor_values") && model["tensor_values"].is_object()) {
    tensor_values = &model["tensor_values"];
  }

  for (const auto &node : nodes) {
    bool needs_fallback = false;
    if (node.contains("ascend_ir") && node["ascend_ir"].is_object()) {
      const auto &ascend_ir = node["ascend_ir"];
      if (!ascend_ir.contains("source") || (ascend_ir.contains("source") && ascend_ir["source"].is_string() &&
                                            ascend_ir["source"].get<std::string>() == "")) {
        std::string node_target = node.value("target", "");
        if (node_target == "torch.ops.aten._assert_tensor_metadata.default") {
          TDLOGD("Skipping _assert_tensor_metadata node for fallback codegen");
        } else {
          needs_fallback = true;
          fallback_count++;
        }
      }
    }

    if (needs_fallback) {
      OpTarget target = ParseTargetFromJson(node);
      if (target.device == "cpu") {
        CollectShapeInfo(node, tensor_values, target);
      }
      targets.push_back(std::move(target));
    }
  }

  TDLOGI("ParseTargetsFromJson: total_nodes=%zu, needs_fallback=%zu", nodes.size(), fallback_count);

  std::set<std::tuple<std::string, std::string, std::string, std::string>> seen_ops;
  std::map<std::string, std::string> cpu_shape_signatures;
  std::vector<OpTarget> unique_targets;
  unique_targets.reserve(targets.size());

  for (const auto &target : targets) {
    auto key = std::make_tuple(target.namespace_name, target.op_name, target.overload, target.device);
    if (!seen_ops.insert(key).second) {
      if (target.device == "cpu") {
        // 同一 CPU 算子多实例：InferShape 用代表性节点内联生成，输出结构不一致时
        // 内联值必然出错，直接报错。
        std::string signature = OutputStructureSignature(target);
        auto dedup_key = target.namespace_name + "::" + target.op_name + "." + target.overload;
        auto [it, inserted] = cpu_shape_signatures.emplace(dedup_key, signature);
        if (!inserted && it->second != signature) {
          throw FallbackCodegenError(
              "CPU op '" + target.namespace_name + "::" + target.op_name + "' (overload '" + target.overload +
              "') appears with inconsistent output structures: '" + it->second + "' vs '" + signature +
              "'. ShapeInferOp inlines the representative instance's meta; split the model or unify shapes.");
        }
      }
      TDLOGD("Skipping duplicate op: namespace=%s, op_name=%s, overload=%s, device=%s", target.namespace_name.c_str(),
             target.op_name.c_str(), target.overload.empty() ? "(default)" : target.overload.c_str(),
             target.device.c_str());
      continue;
    }
    if (target.device == "cpu") {
      cpu_shape_signatures.emplace(target.namespace_name + "::" + target.op_name + "." + target.overload,
                                   OutputStructureSignature(target));
    }
    unique_targets.push_back(target);
  }

  // 同一算子（含 overload）不允许同时出现 npu 与 cpu 两种归属：GE 注册表按算子类型
  // 唯一，无法为同一类型同时注册 kDevice 与 kHostCPU 两个后端实现。
  std::map<std::tuple<std::string, std::string, std::string>, std::string> device_by_op;
  for (const auto &target : unique_targets) {
    auto key = std::make_tuple(target.namespace_name, target.op_name, target.overload);
    auto [it, inserted] = device_by_op.emplace(key, target.device);
    if (!inserted && it->second != target.device) {
      std::string full_op = target.namespace_name + "::" + target.op_name;
      throw FallbackCodegenError(
          "op '" + full_op + "' (overload '" + target.overload + "') appears with inconsistent device ownership: '" +
          it->second + "' and '" + target.device +
          "'. GE registers one backend per op type; unify the ownership of this op or use distinct overloads.");
    }
  }

  TDLOGI("ParseTargetsFromJson: deduplicated %zu -> %zu unique ops", targets.size(), unique_targets.size());

  return unique_targets;
}

std::vector<OpTarget> FallbackCodegen::ParseModelJson(const std::string &json_path) {
  std::ifstream f(json_path);
  if (!f) {
    throw FallbackCodegenError("cannot read model.json: " + json_path);
  }

  json model = json::parse(f);
  return ParseTargetsFromJson(model);
}

std::vector<OpTarget> FallbackCodegen::ParseModelJsonFromString(const std::string &model_json) {
  try {
    json model = json::parse(model_json);
    return ParseTargetsFromJson(model);
  } catch (const json::parse_error &e) {
    throw FallbackCodegenError("cannot parse model.json content: " + std::string(e.what()));
  }
}

CodegenResult FallbackCodegen::GenerateFromModelJson(const std::string &model_json) {
  TDLOGI("FallbackCodegen::GenerateFromModelJson() enter");

  std::vector<OpTarget> targets;
  try {
    targets = ParseModelJsonFromString(model_json);
    TDLOGD("Parsed %zu targets from model_json", targets.size());
  } catch (const FallbackCodegenError &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromModelJson() parse failed: %s", what ? what : "(null)");
    CodegenResult result;
    result.success = false;
    result.skipped = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
    return result;
  }

  if (targets.empty()) {
    TDLOGI("FallbackCodegen::GenerateFromModelJson() skipped: no nodes need fallback");
    CodegenResult result;
    result.success = true;
    result.skipped = true;
    result.error_message = "Skipped: no nodes need fallback codegen (all nodes have valid ascend_ir.source)";
    return result;
  }

  TDLOGI("FallbackCodegen::GenerateFromModelJson() %zu nodes need fallback codegen", targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    TDLOGD("  target[%zu]: namespace=%s, op_name=%s, overload=%s", i, targets[i].namespace_name.c_str(),
           targets[i].op_name.c_str(), targets[i].overload.empty() ? "(default)" : targets[i].overload.c_str());
  }

  const char *torch_pkg_dir_env = std::getenv("TORCH_PKG_DIR");
  const char *torch_npu_dir_env = std::getenv("TORCH_NPU_DIR");
  const char *output_dir_env = std::getenv("TORCH_DELEGATE_OUTPUT_DIR");

  std::string torch_pkg_dir = torch_pkg_dir_env ? torch_pkg_dir_env : "";
  std::string torch_npu_dir = torch_npu_dir_env ? torch_npu_dir_env : "";
  std::string output_dir = output_dir_env ? output_dir_env : "./output_dir";

  TDLOGI("FallbackCodegen config: output_dir=%s, torch_pkg_dir=%s, torch_npu_dir=%s", output_dir.c_str(),
         torch_pkg_dir.empty() ? "(not set)" : torch_pkg_dir.c_str(),
         torch_npu_dir.empty() ? "(not set)" : torch_npu_dir.c_str());

  if (torch_pkg_dir.empty() || torch_npu_dir.empty()) {
    TDLOGE("FallbackCodegen::GenerateFromModelJson() failed: TORCH_PKG_DIR or TORCH_NPU_DIR not set");
    CodegenResult result;
    result.success = false;
    result.skipped = false;
    result.error_message = "TORCH_PKG_DIR and TORCH_NPU_DIR must be set when fallback codegen is needed";
    return result;
  }

  TDLOGI("FallbackCodegen::GenerateFromModelJson() starting codegen");

  CodegenResult result = GenerateFromTargets(targets, torch_pkg_dir, torch_npu_dir, output_dir);
  result.skipped = false;

  if (result.success) {
    TDLOGI("FallbackCodegen::GenerateFromModelJson() exit: success, generated %zu ops",
           result.generated_op_names.size());
  } else {
    TDLOGE("FallbackCodegen::GenerateFromModelJson() exit: failed, error=%s", result.error_message.c_str());
  }

  return result;
}

SchemaInfo FallbackCodegen::GetSchemaInfo(const std::string &full_op_name, const std::string &overload) {
  SchemaInfo info;
  info.full_op_name = full_op_name;
  info.overload = overload;

  auto [ns, op] = SplitNamespaceOp(full_op_name);
  info.namespace_name = ns;
  info.op_name = op;

  c10::OperatorName op_name(full_op_name, overload);
  auto handle = torch::Dispatcher::singleton().findSchema(op_name);
  if (!handle.has_value()) {
    TDLOGE("Schema not found for '%s' with overload '%s'", full_op_name.c_str(),
           overload.empty() ? "(default)" : overload.c_str());
    throw FallbackCodegenError("schema not found for '" + full_op_name + "' with overload '" + overload + "'");
  }

  TDLOGD("Schema found for '%s', args=%zu, returns=%zu", full_op_name.c_str(), handle->schema().arguments().size(),
         handle->schema().returns().size());

  const auto &schema = handle->schema();

  // Determine inplace/out semantics from schema alias_info
  // inplace: first argument has alias_info()->isWrite() (e.g., relu_, copy_)
  // out: overload == "out" or argument.is_out() == true
  info.is_inplace = false;
  info.is_out = (overload == "out");

  if (!schema.arguments().empty()) {
    const auto &first_arg = schema.arguments().at(0);
    if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
      // First argument is written to -> inplace semantics
      info.is_inplace = true;
    }
  }

  int fixed_count = 0;
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto &arg = schema.arguments().at(i);
    SchemaArgInfo arg_info;
    arg_info.name = arg.name();
    arg_info.is_kwarg_only = arg.kwarg_only();
    arg_info.has_default = HasDefaultValue(arg);
    arg_info.is_out = arg.is_out();

    // Extract default value if present
    if (arg_info.has_default) {
      try {
        const auto &default_val = arg.default_value();
        if (!default_val.has_value()) {
          arg_info.default_value = nullptr;
        } else if (default_val->isNone()) {
          arg_info.default_value = nullptr;
        } else if (default_val->isInt()) {
          arg_info.default_value = default_val->toInt();
        } else if (default_val->isDouble()) {
          arg_info.default_value = default_val->toDouble();
        } else if (default_val->isBool()) {
          arg_info.default_value = default_val->toBool();
        } else if (default_val->isString()) {
          arg_info.default_value = default_val->toStringRef();
        } else if (default_val->isIntList()) {
          arg_info.default_value = default_val->toIntVector();
        } else if (default_val->isDoubleList()) {
          arg_info.default_value = default_val->toDoubleVector();
        } else {
          // For other types (Tensor, bool list, etc.), we can't easily extract the default value
          // Mark as having default but leave default_value as null
          arg_info.default_value = nullptr;
        }
      } catch (const std::exception &e) {
        // If we can't extract the default value, just mark it as having one
        arg_info.default_value = nullptr;
      }
    }

    // Parse alias_info for arguments
    auto alias_info = arg.alias_info();
    if (alias_info) {
      arg_info.has_alias = true;
      arg_info.is_alias_write = alias_info->isWrite();
      if (alias_info->isWrite()) {
        info.is_inplace = true;  // Mark as inplace if any argument has write alias
      }
    }

    auto type = arg.type();

    bool is_opt = IsOptionalType(type);
    if (IsTensorType(type)) {
      arg_info.type_category = "tensor";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::OPTIONAL_TENSOR : InputKind::TENSOR;
      arg_info.ge_tensor_type = "TensorType::ALL()";
      if (!arg_info.has_default && !arg_info.is_out) {
        fixed_count++;
      }
    } else if (IsTensorListType(type)) {
      arg_info.type_category = "tensor_list";
      arg_info.input_kind = InputKind::TENSOR_LIST;
      arg_info.is_variadic = true;
      arg_info.fixed_index = fixed_count;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsScalarPolyType(type)) {
      arg_info.type_category = "scalar";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::SCALAR_OPT : InputKind::SCALAR;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsIntType(type)) {
      arg_info.type_category = "int";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::INT_OPT : InputKind::INT;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else if (IsFloatType(type)) {
      arg_info.type_category = "float";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::FLOAT_OPT : InputKind::FLOAT;
      arg_info.ge_tensor_type = "FloatTensorType()";
    } else if (IsBoolType(type)) {
      arg_info.type_category = "bool";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::BOOL_OPT : InputKind::BOOL;
      arg_info.ge_tensor_type = "BoolTensorType()";
    } else if (IsIntListType(type)) {
      arg_info.type_category = "int_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::INT_LIST_OPT : InputKind::INT_LIST;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else if (IsFloatListType(type)) {
      arg_info.type_category = "float_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::FLOAT_LIST_OPT : InputKind::FLOAT_LIST;
      arg_info.ge_tensor_type = "FloatTensorType()";
    } else if (IsBoolListType(type)) {
      arg_info.type_category = "bool_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::BOOL_LIST_OPT : InputKind::BOOL_LIST;
      arg_info.ge_tensor_type = "BoolTensorType()";
    } else if (IsScalarListType(type)) {
      arg_info.type_category = "scalar_list";
      arg_info.input_kind = InputKind::SCALAR_LIST;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsStringType(type)) {
      arg_info.type_category = "string";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::STRING_OPT : InputKind::STRING;
      arg_info.ge_tensor_type = "StringTensorType()";
    } else if (IsEnumType(type)) {
      arg_info.type_category = "enum";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::SCALAR_TYPE_OPT : InputKind::SCALAR_TYPE;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else {
      throw FallbackCodegenError("unsupported argument type '" + type->str() + "' for argument '" + arg_info.name +
                                 "' in schema");
    }

    info.arguments.push_back(arg_info);
  }

  info.num_fixed_inputs = fixed_count;

  for (const auto &arg : info.arguments) {
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      info.has_variadic_input = true;
      break;
    }
  }

  if (schema.is_vararg()) {
    info.has_variadic_input = true;
  }

  for (size_t i = 0; i < schema.returns().size(); ++i) {
    const auto &ret = schema.returns().at(i);
    SchemaArgInfo ret_info;
    ret_info.name = ret.name().empty() ? ("output_" + std::to_string(i)) : std::string(ret.name());

    // Parse alias_info for returns
    auto alias_info = ret.alias_info();
    if (alias_info) {
      ret_info.has_alias = true;
      ret_info.is_alias_write = alias_info->isWrite();
    }

    auto type = ret.type();
    if (IsTensorType(type)) {
      ret_info.type_category = "tensor";
    } else if (IsTensorListType(type)) {
      ret_info.type_category = "tensor_list";
      ret_info.is_variadic = true;
    } else {
      throw FallbackCodegenError("unsupported return type '" + type->str() + "'");
    }

    info.returns.push_back(ret_info);
  }

  return info;
}

std::string FallbackCodegen::GenerateHelperFunctions(bool need_hostcpu) {
  std::ostringstream os;

  os << "at::ScalarType MapGeDtypeToTorch(ge::DataType dtype) {\n";
  os << "    switch (dtype) {\n";
  os << "        case ge::DT_FLOAT: return at::kFloat;\n";
  os << "        case ge::DT_FLOAT16: return at::kHalf;\n";
  os << "        case ge::DT_BF16: return at::kBFloat16;\n";
  os << "        case ge::DT_DOUBLE: return at::kDouble;\n";
  os << "        case ge::DT_INT8: return at::kChar;\n";
  os << "        case ge::DT_INT16: return at::kShort;\n";
  os << "        case ge::DT_INT32: return at::kInt;\n";
  os << "        case ge::DT_INT64: return at::kLong;\n";
  os << "        case ge::DT_UINT8: return at::kByte;\n";
  os << "        case ge::DT_UINT16: return at::kUInt16;\n";
  os << "        case ge::DT_UINT32: return at::kUInt32;\n";
  os << "        case ge::DT_UINT64: return at::kUInt64;\n";
  os << "        case ge::DT_BOOL: return at::kBool;\n";
  os << "        case ge::DT_COMPLEX64: return at::kComplexFloat;\n";
  os << "        case ge::DT_COMPLEX128: return at::kComplexDouble;\n";
  os << "        case ge::DT_QINT8: return at::kQInt8;\n";
  os << "        case ge::DT_QUINT8: return at::kQUInt8;\n";
  os << "        case ge::DT_QINT32: return at::kQInt32;\n";
  os << "        case ge::DT_HIFLOAT8: return at::kFloat8_e5m2;\n";
  os << "        case ge::DT_FLOAT8_E5M2: return at::kFloat8_e5m2;\n";
  os << "        case ge::DT_FLOAT8_E4M3FN: return at::kFloat8_e4m3fn;\n";
  os << "        case ge::DT_FLOAT8_E8M0: return at::kFloat8_e8m0fnu;\n";
  os << "        default:\n";
  os << "            TORCH_CHECK(false, \"Unsupported GE DataType: \", static_cast<int>(dtype));\n";
  os << "            return at::kFloat; // unreachable\n";
  os << "    }\n";
  os << "}\n\n";

  os << "ge::DataType MapTorchDtypeToGe(at::ScalarType dtype) {\n";
  os << "    switch (dtype) {\n";
  os << "        case at::kFloat: return ge::DT_FLOAT;\n";
  os << "        case at::kHalf: return ge::DT_FLOAT16;\n";
  os << "        case at::kBFloat16: return ge::DT_BF16;\n";
  os << "        case at::kDouble: return ge::DT_DOUBLE;\n";
  os << "        case at::kChar: return ge::DT_INT8;\n";
  os << "        case at::kShort: return ge::DT_INT16;\n";
  os << "        case at::kInt: return ge::DT_INT32;\n";
  os << "        case at::kLong: return ge::DT_INT64;\n";
  os << "        case at::kByte: return ge::DT_UINT8;\n";
  os << "        case at::kUInt16: return ge::DT_UINT16;\n";
  os << "        case at::kUInt32: return ge::DT_UINT32;\n";
  os << "        case at::kUInt64: return ge::DT_UINT64;\n";
  os << "        case at::kBool: return ge::DT_BOOL;\n";
  os << "        case at::kComplexFloat: return ge::DT_COMPLEX64;\n";
  os << "        case at::kComplexDouble: return ge::DT_COMPLEX128;\n";
  os << "        case at::kQInt8: return ge::DT_QINT8;\n";
  os << "        case at::kQUInt8: return ge::DT_QUINT8;\n";
  os << "        case at::kQInt32: return ge::DT_QINT32;\n";
  os << "        case at::kFloat8_e5m2: return ge::DT_FLOAT8_E5M2;\n";
  os << "        case at::kFloat8_e4m3fn: return ge::DT_FLOAT8_E4M3FN;\n";
  os << "        case at::kFloat8_e8m0fnu: return ge::DT_FLOAT8_E8M0;\n";
  os << "        default:\n";
  os << "            TORCH_CHECK(false, \"Unsupported PyTorch ScalarType: \", static_cast<int>(dtype));\n";
  os << "            return ge::DT_FLOAT; // unreachable\n";
  os << "    }\n";
  os << "}\n\n";

  os << "at::Tensor CreateNpuTensor(const gert::Tensor* ge_tensor) {\n";
  os << "    std::vector<int64_t> sizes;\n";
  os << "    for (size_t i = 0; i < ge_tensor->GetShape().GetShape().GetDimNum(); i++) {\n";
  os << "        sizes.push_back(ge_tensor->GetShape().GetShape().GetDim(i));\n";
  os << "    }\n";
  os << "    at::ScalarType torch_dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());\n";
  os << "    auto options = at::TensorOptions().dtype(torch_dtype).device(at::kPrivateUse1);\n";
  os << "    return at_npu::native::from_blob(const_cast<void*>(ge_tensor->GetAddr()), sizes, options);\n";
  os << "}\n\n";

  if (need_hostcpu) {
    // HostCpu 变体：host 指针零拷贝包装，避免 D2H 拷贝。
    os << "at::Tensor CreateCpuTensor(const gert::Tensor* ge_tensor) {\n";
    os << "    std::vector<int64_t> sizes;\n";
    os << "    for (size_t i = 0; i < ge_tensor->GetShape().GetShape().GetDimNum(); i++) {\n";
    os << "        sizes.push_back(ge_tensor->GetShape().GetShape().GetDim(i));\n";
    os << "    }\n";
    os << "    at::ScalarType torch_dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());\n";
    os << "    auto options = at::TensorOptions().dtype(torch_dtype).device(at::kCPU);\n";
    os << "    return at::from_blob(const_cast<void*>(ge_tensor->GetAddr()), sizes, options);\n";
    os << "}\n\n";
  }

  // ReadScalarInput: device 张量走 D2H 拷贝；host 张量直接借用 GetAddr()，不做深拷贝。
  // 背景：GE 对 custom op（REG_AUTO_MAPPING_OP 路径）会把 Const 节点数据放到 device 内存，
  //       kPlacementHost 设置与 HostInputs 注册对 custom op 均不生效
  //       （custom_node_converter.cc 硬编码 kOnDeviceHbm）。当 placement 实际为 host 时，
  //       无需 D2H，直接借用地址即可。placement 须精确为 device 或 host，其他一律报错。
  os << "// ScalarInputBuf: device 路径持有 D2H 拷贝缓冲；host 路径直接借用 tensor 地址（无深拷贝）\n";
  os << "struct ScalarInputBuf {\n";
  os << "    std::vector<uint8_t> owned;\n";
  os << "    const uint8_t* view = nullptr;\n";
  os << "    size_t len = 0;\n";
  os << "    const uint8_t* data() const { return view != nullptr ? view : owned.data(); }\n";
  os << "    size_t size() const { return len; }\n";
  os << "};\n\n";
  os << "// ReadScalarInput: device 张量 D2H 拷贝；host 张量直接借用 GetAddr()（不深拷贝）；其他报错\n";
  os << "ScalarInputBuf ReadScalarInput(const gert::Tensor* tensor) {\n";
  os << "    size_t byte_size = tensor->GetSize();\n";
  os << "    auto placement = tensor->GetPlacement();\n";
  os << "    if (gert::TensorPlacementUtils::IsOnHost(placement)) {\n";
  os << "        return {{}, static_cast<const uint8_t*>(tensor->GetAddr()), byte_size};\n";
  os << "    }\n";
  os << "    if (gert::TensorPlacementUtils::IsOnDevice(placement)) {\n";
  os << "        std::vector<uint8_t> buf(byte_size);\n";
  os << "        aclrtMemcpy(buf.data(), byte_size, tensor->GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);\n";
  os << "        return {std::move(buf), nullptr, byte_size};\n";
  os << "    }\n";
  os << "    TORCH_CHECK(false, \"ReadScalarInput: unsupported tensor placement: \", static_cast<int>(placement));\n";
  os << "    return {};  // unreachable\n";
  os << "}\n\n";

  return os.str();
}

// 算子原型（REG_OP 宏链）两类后端完全一致，仅绑定宏不同（见 GenerateGeOpClass）。
std::string FallbackCodegen::GenerateGeOpReg(const SchemaInfo &schema) {
  std::ostringstream os;

  std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);

  os << "REG_OP(" << ge_op_name << ")\n";

  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      os << "    .DYNAMIC_INPUT(" << arg.name << ", TensorType::ALL())\n";
    } else if (arg.type_category == "tensor") {
      if (arg.input_kind == InputKind::OPTIONAL_TENSOR) {
        os << "    .OPTIONAL_INPUT(" << arg.name << ", TensorType::ALL())\n";
      } else {
        os << "    .INPUT(" << arg.name << ", TensorType::ALL())\n";
      }
    } else if (IsValueDependency(arg.input_kind)) {
      if (IsOptional(arg.input_kind)) {
        os << "    .OPTIONAL_INPUT(" << arg.name << ", " << arg.ge_tensor_type << ")\n";
      } else {
        os << "    .INPUT(" << arg.name << ", " << arg.ge_tensor_type << ")\n";
      }
    }
  }

  // Generate outputs based on schema semantics:
  // 1. First, outputs from returns (non-inplace outputs)
  // 2. Then, inplace outputs (from arguments with write alias)

  // Step 1: Process returns - outputs from schema returns
  for (size_t output_idx = 0; output_idx < schema.returns.size(); ++output_idx) {
    const auto &ret = schema.returns[output_idx];
    if (ret.is_variadic) {
      os << "    .DYNAMIC_OUTPUT(" << ret.name << ", TensorType::ALL())\n";
    } else {
      // For inplace ops, returns without alias are new tensors
      // For non-inplace ops, all returns are outputs
      if (schema.is_inplace && ret.has_alias && ret.is_alias_write) {
        // Return has write alias -> this is the inplace output
        // Skip here, will be handled in Step 2
        continue;
      }
      os << "    .OUTPUT(" << ret.name << ", TensorType::ALL())\n";
    }
  }

  // Step 2: For inplace ops, add outputs for arguments with write alias
  if (schema.is_inplace) {
    // Find all arguments with write alias (inplace arguments)
    for (size_t arg_idx = 0; arg_idx < schema.arguments.size(); ++arg_idx) {
      const auto &arg = schema.arguments[arg_idx];
      if (arg.has_alias && arg.is_alias_write) {
        // This argument is written inplace -> need an output
        os << "    .OUTPUT(" << arg.name << ", TensorType::ALL())\n";
      }
    }

    // Special case: if schema returns void but has inplace arg
    if (schema.returns.empty()) {
      // Already handled above (all inplace args get outputs)
    }
  }

  os << "    .ATTR(op_name, String, \"" << schema.full_op_name << "\")\n";
  os << "    .ATTR(overload, String, \"" << schema.overload << "\")\n";
  os << "    .ATTR(input_kinds, ListInt, {";
  bool first_kind = true;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (!first_kind) os << ", ";
    first_kind = false;
    os << static_cast<int>(arg.input_kind);
  }
  os << "})\n";
  // Non-tensor InputKind base value: the first enum value after all tensor types.
  // GE reads this attribute to determine whether an input_kind value represents a non-tensor
  // type (>= base) that should be passed via host memory. Falls back to 3 in GE when absent.
  os << "    .ATTR(_custom_op_non_tensor_kind_base, Int, " << static_cast<int>(InputKind::INT) << ")\n";

  // Count actual outputs: non-alias-write returns + inplace mutation outputs
  // Alias-write returns are skipped in REG_OP Step 1 (handled as mutation in Step 2),
  // so they must not be double-counted here.
  size_t num_outputs = 0;
  for (const auto &ret : schema.returns) {
    if (!(schema.is_inplace && ret.has_alias && ret.is_alias_write)) {
      num_outputs++;
    }
  }
  if (schema.is_inplace) {
    for (const auto &arg : schema.arguments) {
      if (arg.has_alias && arg.is_alias_write) {
        num_outputs++;
      }
    }
  }
  os << "    .ATTR(num_outputs, Int, " << num_outputs << ")\n";
  os << "    .ATTR(is_inplace, Bool, " << (schema.is_inplace ? "true" : "false") << ")\n";
  os << "    .OP_END_FACTORY_REG(" << ge_op_name << ");\n";

  std::vector<int> dep_indices;
  int input_idx = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (IsValueDependency(arg.input_kind)) {
      dep_indices.push_back(input_idx);
    }
    input_idx++;
  }
  os << "IMPL_OP(" << ge_op_name << ")\n";
  if (!dep_indices.empty()) {
    os << "    .InputsDataDependency({";
    for (size_t i = 0; i < dep_indices.size(); ++i) {
      if (i > 0) os << ", ";
      os << dep_indices[i];
    }
    os << "});\n";
  } else {
    os << "    ;\n";
  }

  return os.str();
}

// Helper: generate shape computation + MallocOutputTensor + write-back for one output tensor.
// Assumes an `at::Tensor result` variable is in scope.
// NPU 变体：aclrtMemcpyAsync D2D 写回 device 输出内存；
// HostCpu 变体：非连续结果先 .contiguous()，直接 memcpy 写 host 输出内存。
static void EmitOutputTensorWrite(std::ostringstream &os, const std::string &out_idx_var, bool is_host_cpu) {
  os << "            auto result_sizes = result.sizes();\n";
  os << "            gert::StorageShape storage_shape;\n";
  os << "            gert::Shape& origin_shape = storage_shape.MutableOriginShape();\n";
  os << "            origin_shape.SetDimNum(result_sizes.size());\n";
  os << "            for (size_t d = 0; d < result_sizes.size(); ++d) origin_shape.SetDim(d, result_sizes[d]);\n";
  os << "            gert::Shape& storage_shape_ref = storage_shape.MutableStorageShape();\n";
  os << "            storage_shape_ref.SetDimNum(result_sizes.size());\n";
  os << "            for (size_t d = 0; d < result_sizes.size(); ++d) storage_shape_ref.SetDim(d, result_sizes[d]);\n";
  os << "            ge::DataType data_type = MapTorchDtypeToGe(result.scalar_type());\n";
  os << "            gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});\n";
  os << "            size_t tensor_numel = 1; for (size_t d = 0; d < result_sizes.size(); ++d) { tensor_numel *= "
        "static_cast<size_t>(result_sizes[d]); }\n";
  os << "            size_t tensor_bytes = tensor_numel * result.element_size();\n";
  os << "            gert::Tensor* ge_out = ctx->MallocOutputTensor(" << out_idx_var
     << ", storage_shape, format, data_type);\n";
  if (is_host_cpu) {
    os << "            at::Tensor contiguous_result = result.contiguous();\n";
    os << "            if (contiguous_result.data_ptr() != ge_out->GetAddr()) {\n";
    os << "                std::memcpy(ge_out->GetAddr(), contiguous_result.data_ptr(), tensor_bytes);\n";
    os << "            }\n";
  } else {
    // The flat memcpy below only describes a contiguous buffer (data_ptr + numel).
    // Views like transpose/slice/unfold share the source storage with a different
    // stride; materialize them first or the copy keeps the raw storage order.
    os << "            if (!result.is_contiguous()) { result = result.contiguous(); }\n";
    os << "            aclrtMemcpyAsync(const_cast<void*>(ge_out->GetAddr()), tensor_bytes,\n";
    os << "                result.data_ptr(), tensor_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ge_stream);\n";
  }
}

// Helper function to generate code for adding a default value or None
static void EmitDefaultOrNone(std::ostringstream &os, const SchemaArgInfo &arg) {
  if (arg.has_default && !arg.default_value.is_null()) {
    // Use the actual default value
    if (arg.input_kind == InputKind::INT || arg.input_kind == InputKind::INT_OPT) {
      os << "            args.emplace_back(static_cast<int64_t>(" << arg.default_value.get<int64_t>() << "));\n";
    } else if (arg.input_kind == InputKind::FLOAT || arg.input_kind == InputKind::FLOAT_OPT) {
      os << "            args.emplace_back(static_cast<double>(" << arg.default_value.get<double>() << "));\n";
    } else if (arg.input_kind == InputKind::BOOL || arg.input_kind == InputKind::BOOL_OPT) {
      os << "            args.emplace_back(" << (arg.default_value.get<bool>() ? "true" : "false") << ");\n";
    } else if (arg.input_kind == InputKind::INT_LIST || arg.input_kind == InputKind::INT_LIST_OPT) {
      os << "            args.emplace_back(std::vector<int64_t>{";
      const auto &list = arg.default_value.get<std::vector<int64_t>>();
      for (size_t i = 0; i < list.size(); ++i) {
        if (i > 0) os << ", ";
        os << list[i];
      }
      os << "});\n";
    } else if (arg.input_kind == InputKind::STRING || arg.input_kind == InputKind::STRING_OPT) {
      os << "            args.emplace_back(std::string(\"" << arg.default_value.get<std::string>() << "\"));\n";
    } else {
      // For other types, use None
      os << "            args.emplace_back(c10::IValue());\n";
    }
  } else {
    // No default value or couldn't extract it, use None
    os << "            args.emplace_back(c10::IValue());\n";
  }
}

std::string FallbackCodegen::GenerateExecuteMethod(const SchemaInfo &schema, bool is_host_cpu) {
  std::ostringstream os;

  // NPU 变体：EagerOpExecutionContext + device stream；
  // HostCpu 变体：HostCpuOpExecutionContext，host 执行，无 stream guard。
  if (is_host_cpu) {
    os << "    ge::graphStatus Execute(gert::HostCpuOpExecutionContext* ctx) override {\n";
    // DEBUG 级节点名日志：走 CANN aclAppLog，
    // 与 TorchDelegate TDLOG 同一通道，部署侧可在 slog 中按 DEBUG 级检索。
    os << "        aclAppLog(ACL_DEBUG, \"HostCpuExecute\", \"custom_op.cpp\", __LINE__,\n";
    os << "            \"op=" << schema.full_op_name << " (" << schema.overload << ") execute\");\n";
    os << "        try {\n";
  } else {
    os << "    ge::graphStatus Execute(gert::EagerOpExecutionContext* ctx) override {\n";
  }
  const char *wrap_fn = is_host_cpu ? "CreateCpuTensor" : "CreateNpuTensor";

  // Prelude: declare ge_input_<ir_index> for each non-variadic, non-out argument.
  // Use IR-indexed APIs (GetRequiredInputTensor/GetOptionalInputTensor) so that
  // indices align with REG_OP declarations regardless of DYNAMIC_INPUT expansion
  // or unconnected OPTIONAL_INPUT.
  // Prelude: declare ge_input_<ir_index> for each non-variadic, non-out argument.
  // Use IR-indexed APIs (GetRequiredInputTensor/GetOptionalInputTensor) instead of
  // flat-indexed GetInputTensor, so that indices align with REG_OP declarations
  // regardless of DYNAMIC_INPUT expansion or unconnected OPTIONAL_INPUT.
  // Reference: eager_op_execution_context.h — GetInputTensor uses flat runtime
  // index (DYNAMIC_INPUT occupies N slots, unconnected OPTIONAL occupies 0),
  // while GetRequiredInputTensor/GetOptionalInputTensor use IR declaration index
  // (each declaration occupies exactly 1 slot).
  int ir_index = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) {
      ir_index++;
      continue;
    }
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      ir_index++;
      continue;
    }
    if (IsOptional(arg.input_kind)) {
      os << "        const gert::Tensor* ge_input_" << ir_index << " = ctx->GetOptionalInputTensor(" << ir_index
         << ");\n";
    } else {
      os << "        const gert::Tensor* ge_input_" << ir_index << " = ctx->GetRequiredInputTensor(" << ir_index
         << ");\n";
    }
    ir_index++;
  }

  if (!is_host_cpu) {
    os << "        aclrtStream ge_stream = static_cast<aclrtStream>(ctx->GetStream());\n";
  }

  os << "\n";

  if (schema.is_out) {
    int num_outputs = static_cast<int>(schema.returns.size());
    os << "        // out version: get output tensor descriptions from GE InferShape\n";
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      if (schema.returns[output_idx].is_variadic) {
        os << "        // output " << output_idx << " is variadic, skip pre-allocation\n";
        continue;
      }
      os << "        const gert::Tensor* output_desc_" << output_idx << " = ctx->GetOutputTensor(" << output_idx
         << ");\n";
      os << "        gert::StorageShape out_shape_" << output_idx << " = output_desc_" << output_idx
         << "->GetShape();\n";
      os << "        ge::DataType out_dtype_" << output_idx << " = output_desc_" << output_idx << "->GetDataType();\n";
      os << "        gert::StorageFormat out_format_" << output_idx << " = output_desc_" << output_idx
         << "->GetFormat();\n";
      os << "        gert::Tensor* ge_output_" << output_idx << " = ctx->MallocOutputTensor(" << output_idx
         << ", out_shape_" << output_idx << ", out_format_" << output_idx << ", out_dtype_" << output_idx << ");\n";
    }
    os << "\n";
  }

  os << "        std::vector<c10::IValue> args;\n";

  // Body: iterate arguments by IR index. For DYNAMIC_INPUT, read N from attrs
  // and use GetDynamicInputTensor(ir_index, i). For other kinds, use ge_input_<ir_index>
  // declared in the prelude.
  ir_index = 0;
  int out_arg_output_idx = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) {
      int output_idx = out_arg_output_idx;
      if (output_idx < static_cast<int>(schema.returns.size()) && !schema.returns[output_idx].is_variadic) {
        os << "        args.emplace_back(" << wrap_fn << "(ge_output_" << output_idx << "));\n";
      } else {
        os << "        args.emplace_back(c10::IValue()); // variadic output placeholder\n";
      }
      out_arg_output_idx++;
      ir_index++;
      continue;
    }

    // DYNAMIC_INPUT: query instance count from GetIrInputInstanceInfo (not from
    // N attr), so multiple DYNAMIC_INPUTs are naturally supported. Pack all
    // instances into c10::List<at::Tensor> for callBoxed compatibility.
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      os << "        {\n";
      os << "            const auto* ins_info = ctx->GetIrInputInstanceInfo(" << ir_index << ");\n";
      os << "            size_t dyn_n = (ins_info != nullptr) ? ins_info->GetInstanceNum() : 0;\n";
      os << "            std::vector<at::Tensor> tensor_list;\n";
      os << "            tensor_list.reserve(dyn_n);\n";
      os << "            for (size_t i = 0; i < dyn_n; ++i) {\n";
      os << "                const gert::Tensor* t = ctx->GetDynamicInputTensor(" << ir_index << ", i);\n";
      os << "                tensor_list.emplace_back(" << wrap_fn << "(t));\n";
      os << "            }\n";
      os << "            args.emplace_back(c10::List<at::Tensor>(std::move(tensor_list)));\n";
      os << "        }\n";
      ir_index++;
    } else if (arg.type_category == "tensor") {
      if (IsOptional(arg.input_kind)) {
        os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
        os << "            args.emplace_back(" << wrap_fn << "(ge_input_" << ir_index << "));\n";
        os << "        } else {\n";
        EmitDefaultOrNone(os, arg);
        os << "        }\n";
      } else {
        os << "        args.emplace_back(" << wrap_fn << "(ge_input_" << ir_index << "));\n";
      }
      ir_index++;
    } else if (arg.input_kind == InputKind::INT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(static_cast<int64_t>(*reinterpret_cast<const int64_t*>(buf_" << ir_index
         << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::INT_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(c10::optional<int64_t>(static_cast<int64_t>(*reinterpret_cast<const "
            "int64_t*>(buf_"
         << ir_index << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::FLOAT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(static_cast<double>(*reinterpret_cast<const float*>(buf_" << ir_index
         << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::FLOAT_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(c10::optional<double>(static_cast<double>(*reinterpret_cast<const "
            "float*>(buf_"
         << ir_index << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::BOOL) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(static_cast<bool>(*reinterpret_cast<const uint8_t*>(buf_" << ir_index
         << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::BOOL_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(c10::optional<bool>(static_cast<bool>(*reinterpret_cast<const uint8_t*>(buf_"
         << ir_index << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::INT_LIST || arg.input_kind == InputKind::INT_LIST_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            std::vector<int64_t> list_" << ir_index << ";\n";
      os << "            size_t count_" << ir_index << " = buf_" << ir_index << ".size() / sizeof(int64_t);\n";
      os << "            const int64_t* data_" << ir_index << " = reinterpret_cast<const int64_t*>(buf_" << ir_index
         << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << ir_index << "; ++i) list_" << ir_index << ".push_back(data_"
         << ir_index << "[i]);\n";
      os << "            args.emplace_back(std::move(list_" << ir_index << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::FLOAT_LIST || arg.input_kind == InputKind::FLOAT_LIST_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            std::vector<double> list_" << ir_index << ";\n";
      os << "            size_t count_" << ir_index << " = buf_" << ir_index << ".size() / sizeof(float);\n";
      os << "            const float* data_" << ir_index << " = reinterpret_cast<const float*>(buf_" << ir_index
         << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << ir_index << "; ++i) list_" << ir_index
         << ".push_back(static_cast<double>(data_" << ir_index << "[i]));\n";
      os << "            args.emplace_back(std::move(list_" << ir_index << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::BOOL_LIST || arg.input_kind == InputKind::BOOL_LIST_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            std::vector<bool> list_" << ir_index << ";\n";
      os << "            size_t count_" << ir_index << " = buf_" << ir_index << ".size() / sizeof(uint8_t);\n";
      os << "            const uint8_t* data_" << ir_index << " = reinterpret_cast<const uint8_t*>(buf_" << ir_index
         << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << ir_index << "; ++i) list_" << ir_index
         << ".push_back(static_cast<bool>(data_" << ir_index << "[i]));\n";
      os << "            args.emplace_back(std::move(list_" << ir_index << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::SCALAR || arg.input_kind == InputKind::SCALAR_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            if (ge_input_" << ir_index << "->GetDataType() == ge::DT_INT64) {\n";
      os << "                args.emplace_back(c10::Scalar(*reinterpret_cast<const int64_t*>(buf_" << ir_index
         << ".data())));\n";
      os << "            } else {\n";
      os << "                args.emplace_back(c10::Scalar(static_cast<double>(*reinterpret_cast<const float*>(buf_"
         << ir_index << ".data()))));\n";
      os << "            }\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::SCALAR_LIST) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            std::vector<c10::Scalar> scalar_list_" << ir_index << ";\n";
      os << "            if (ge_input_" << ir_index << "->GetDataType() == ge::DT_INT64) {\n";
      os << "                size_t count_" << ir_index << " = buf_" << ir_index << ".size() / sizeof(int64_t);\n";
      os << "                const int64_t* data_" << ir_index << " = reinterpret_cast<const int64_t*>(buf_" << ir_index
         << ".data());\n";
      os << "                for (size_t i = 0; i < count_" << ir_index << "; ++i) scalar_list_" << ir_index
         << ".emplace_back(data_" << ir_index << "[i]);\n";
      os << "            } else {\n";
      os << "                size_t count_" << ir_index << " = buf_" << ir_index << ".size() / sizeof(float);\n";
      os << "                const float* data_" << ir_index << " = reinterpret_cast<const float*>(buf_" << ir_index
         << ".data());\n";
      os << "                for (size_t i = 0; i < count_" << ir_index << "; ++i) scalar_list_" << ir_index
         << ".emplace_back(static_cast<double>(data_" << ir_index << "[i]));\n";
      os << "            }\n";
      os << "            args.emplace_back(std::move(scalar_list_" << ir_index << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::STRING || arg.input_kind == InputKind::STRING_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            std::string str_" << ir_index << "(reinterpret_cast<const char*>(buf_" << ir_index
         << ".data()), buf_" << ir_index << ".size());\n";
      os << "            args.emplace_back(std::move(str_" << ir_index << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else if (arg.input_kind == InputKind::SCALAR_TYPE || arg.input_kind == InputKind::SCALAR_TYPE_OPT) {
      os << "        if (ge_input_" << ir_index << " != nullptr) {\n";
      os << "            auto buf_" << ir_index << " = ReadScalarInput(ge_input_" << ir_index << ");\n";
      os << "            args.emplace_back(static_cast<at::ScalarType>(*reinterpret_cast<const int64_t*>(buf_"
         << ir_index << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      ir_index++;
    } else {
      os << "        args.emplace_back(c10::IValue()); // unsupported kind " << static_cast<int>(arg.input_kind)
         << "\n";
      ir_index++;
    }
  }
  os << "\n";

  if (!is_host_cpu) {
    os << "        int32_t device_id = 0;\n";
    os << "        aclrtGetDevice(&device_id);\n";
    os << "        c10_npu::NPUStream npu_stream = c10_npu::getStreamFromExternal(ge_stream, device_id);\n";
    os << "        c10_npu::NPUStreamGuard guard(npu_stream);\n\n";
  }

  std::string dispatcher_overload = schema.overload.empty() ? "" : schema.overload;
  os << "        static const auto op_handle = torch::Dispatcher::singleton()\n";
  os << "            .findSchemaOrThrow(\"" << schema.full_op_name << "\", \"" << dispatcher_overload << "\");\n\n";

  // For inplace operations, save tensor references before callBoxed
  // because callBoxed will clear the args vector
  if (schema.is_inplace) {
    os << "        // Save tensor references before callBoxed (it will clear args)\n";
    for (size_t arg_idx = 0; arg_idx < schema.arguments.size(); ++arg_idx) {
      const auto &arg = schema.arguments[arg_idx];
      if (arg.has_alias && arg.is_alias_write) {
        os << "        at::Tensor saved_tensor_" << arg_idx << " = args[" << arg_idx << "].toTensor();\n";
      }
    }
    os << "\n";
  }

  os << "        op_handle.callBoxed(args);\n";
  if (!is_host_cpu) {
    // TASK_QUEUE_ENABLE=1 时 callBoxed 内部走异步入队，kernel 延迟到消费者线程提交。
    // 后续 aclrtMemcpyAsync 直接提交到 ge_stream，可能与未提交的 kernel 乱序。
    // stream(true) 清空 task queue，确保 kernel 已提交到 ge_stream 后再处理输出。
    // 注意：torch_npu 需配合修改 NPUStream::stream() 让外部流也 drain default repo。
    os << "        (void)npu_stream.stream(true);\n";
  }

  if (schema.is_inplace) {
    // Inplace operations with returns:
    // - Return value outputs use MallocOutputTensor (independent output tensors)
    // - Mutation outputs use MakeOutputRefInput (reference back to modified input)
    //
    // Output ordering: return value outputs first [0..R-1], then mutation outputs [R..R+M-1].
    // Use flat_out_idx to track flat output index across OUTPUT and DYNAMIC_OUTPUT.
    std::vector<size_t> return_ir_indices;
    for (size_t i = 0; i < schema.returns.size(); ++i) {
      if (!(schema.is_inplace && schema.returns[i].has_alias && schema.returns[i].is_alias_write)) {
        return_ir_indices.push_back(i);
      }
    }

    os << "        size_t flat_out_idx = 0;\n";

    // Return value outputs: write callBoxed results via MallocOutputTensor
    if (!return_ir_indices.empty()) {
      os << "        const size_t num_return_outputs = " << return_ir_indices.size() << ";\n";
      os << "        TORCH_CHECK(args.size() >= num_return_outputs, \"Expected at least num_return_outputs "
            "results\");\n";
      for (size_t ret_pos = 0; ret_pos < return_ir_indices.size(); ++ret_pos) {
        size_t out_ir_idx = return_ir_indices[ret_pos];
        const auto &ret = schema.returns[out_ir_idx];
        os << "        {\n";
        os << "            size_t result_idx = args.size() - num_return_outputs + " << ret_pos << ";\n";
        if (ret.is_variadic) {
          os << "            auto result_list = args[result_idx].toList<at::Tensor>();\n";
          os << "            const auto* out_ins = ctx->GetIrOutputInstanceInfo(" << out_ir_idx << ");\n";
          os << "            size_t out_n = (out_ins != nullptr) ? out_ins->GetInstanceNum() : result_list.size();\n";
          os << "            TORCH_CHECK(result_list.size() == out_n, \"DYNAMIC_OUTPUT instance count mismatch\");\n";
          os << "            for (size_t i = 0; i < out_n; ++i) {\n";
          os << "                at::Tensor result = result_list[static_cast<int64_t>(i)];\n";
          EmitOutputTensorWrite(os, "flat_out_idx", is_host_cpu);
          os << "                flat_out_idx++;\n";
          os << "            }\n";
        } else {
          os << "            at::Tensor result = args[result_idx].toTensor();\n";
          EmitOutputTensorWrite(os, "flat_out_idx", is_host_cpu);
          os << "            flat_out_idx++;\n";
        }
        os << "        }\n";
      }
    }

    // Mutation outputs: write data back to input, then make output reference it
    for (size_t arg_idx = 0; arg_idx < schema.arguments.size(); ++arg_idx) {
      const auto &arg = schema.arguments[arg_idx];
      if (arg.has_alias && arg.is_alias_write) {
        os << "        {\n";
        os << "            at::Tensor& result = saved_tensor_" << arg_idx << ";\n";
        os << "            auto result_sizes = result.sizes();\n";
        os << "            gert::StorageShape storage_shape;\n";
        os << "            gert::Shape& origin_shape = storage_shape.MutableOriginShape();\n";
        os << "            origin_shape.SetDimNum(result_sizes.size());\n";
        os << "            for (size_t d = 0; d < result_sizes.size(); ++d) origin_shape.SetDim(d, result_sizes[d]);\n";
        os << "            gert::Shape& storage_shape_ref = storage_shape.MutableStorageShape();\n";
        os << "            storage_shape_ref.SetDimNum(result_sizes.size());\n";
        os << "            for (size_t d = 0; d < result_sizes.size(); ++d) storage_shape_ref.SetDim(d, "
              "result_sizes[d]);\n";
        os << "            ge::DataType data_type = MapTorchDtypeToGe(result.scalar_type());\n";
        os << "            gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});\n";
        os << "            size_t tensor_numel = 1; for (size_t d = 0; d < result_sizes.size(); ++d) { tensor_numel *= "
              "static_cast<size_t>(result_sizes[d]); }\n";
        os << "            size_t tensor_bytes = tensor_numel * result.element_size();\n";
        os << "            const gert::Tensor* input_tensor = ctx->GetRequiredInputTensor(" << arg_idx << ");\n";
        os << "            void* src = result.data_ptr();\n";
        os << "            void* dst = const_cast<void*>(input_tensor->GetAddr());\n";
        os << "            if (src != dst) {\n";
        if (is_host_cpu) {
          os << "                std::memcpy(dst, src, tensor_bytes);\n";
        } else {
          os << "                aclrtMemcpyAsync(dst, input_tensor->GetSize(),\n";
          os << "                    src, tensor_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ge_stream);\n";
        }
        os << "            }\n";
        os << "            ctx->MakeOutputRefInput(flat_out_idx, " << arg_idx << ");\n";
        os << "            flat_out_idx++;\n";
        os << "        }\n";
      }
    }
  } else {
    // Non-inplace operations: all returns are output tensors.
    // Use flat_out_idx to track the flat output index across regular OUTPUT
    // and DYNAMIC_OUTPUT declarations (DYNAMIC_OUTPUT expands to N instances).
    size_t num_returns = schema.returns.size();
    os << "        const size_t num_returns = " << num_returns << ";\n";
    os << "        TORCH_CHECK(args.size() >= num_returns, \"Expected at least num_returns results\");\n";
    os << "        size_t flat_out_idx = 0;\n";
    for (size_t out_ir_idx = 0; out_ir_idx < num_returns; ++out_ir_idx) {
      const auto &ret = schema.returns[out_ir_idx];
      os << "        {\n";
      os << "            size_t result_idx = args.size() - num_returns + " << out_ir_idx << ";\n";
      if (ret.is_variadic) {
        os << "            auto result_list = args[result_idx].toList<at::Tensor>();\n";
        os << "            const auto* out_ins = ctx->GetIrOutputInstanceInfo(" << out_ir_idx << ");\n";
        os << "            size_t out_n = (out_ins != nullptr) ? out_ins->GetInstanceNum() : result_list.size();\n";
        os << "            TORCH_CHECK(result_list.size() == out_n, \"DYNAMIC_OUTPUT instance count mismatch\");\n";
        os << "            for (size_t i = 0; i < out_n; ++i) {\n";
        os << "                at::Tensor result = result_list[static_cast<int64_t>(i)];\n";
        EmitOutputTensorWrite(os, "flat_out_idx", is_host_cpu);
        os << "                flat_out_idx++;\n";
        os << "            }\n";
      } else {
        os << "            at::Tensor result = args[result_idx].toTensor();\n";
        EmitOutputTensorWrite(os, "flat_out_idx", is_host_cpu);
        os << "            flat_out_idx++;\n";
      }
      os << "        }\n";
    }
  }

  if (is_host_cpu) {
    // HostCpu 变体：host kernel 异常不外抛，打 ERROR 日志（含算子名）后返回
    // GRAPH_FAILED 交由 GE 执行栈报错。
    os << "        } catch (const std::exception &e) {\n";
    os << "            aclAppLog(ACL_ERROR, \"HostCpuExecute\", __FILE__, __LINE__,\n";
    os << "                \"op=" << schema.full_op_name << " (" << schema.overload
       << ") execute failed: %s\", e.what());\n";
    os << "            return ge::GRAPH_FAILED;\n";
    os << "        } catch (...) {\n";
    os << "            aclAppLog(ACL_ERROR, \"HostCpuExecute\", __FILE__, __LINE__,\n";
    os << "                \"op=" << schema.full_op_name << " (" << schema.overload
       << ") execute failed: unknown exception\");\n";
    os << "            return ge::GRAPH_FAILED;\n";
    os << "        }\n";
    os << "        return ge::GRAPH_SUCCESS;\n";
    os << "    }\n";
    return os.str();
  }

  os << "\n        return ge::GRAPH_SUCCESS;\n";
  os << "    }\n";

  return os.str();
}

// PT2 serde dtype 枚举值 → GE dtype 符号名（InferDataType 内联用）。
// 编号直接复用 td::pt2 的 PT2 serde 常量（1-based，Float=7），未知取值在生成期报错。
static std::string Pt2DtypeToGeName(int64_t dtype) {
  switch (dtype) {
    case td::pt2::kDtypeByte:
      return "ge::DT_UINT8";  // kByte
    case td::pt2::kDtypeChar:
      return "ge::DT_INT8";  // kChar
    case td::pt2::kDtypeShort:
      return "ge::DT_INT16";  // kShort
    case td::pt2::kDtypeInt:
      return "ge::DT_INT32";  // kInt
    case td::pt2::kDtypeLong:
      return "ge::DT_INT64";  // kLong
    case td::pt2::kDtypeHalf:
      return "ge::DT_FLOAT16";  // kHalf
    case td::pt2::kDtypeFloat:
      return "ge::DT_FLOAT";  // kFloat
    case td::pt2::kDtypeDouble:
      return "ge::DT_DOUBLE";  // kDouble
    case td::pt2::kDtypeBool:
      return "ge::DT_BOOL";  // kBool
    case td::pt2::kDtypeBFloat16:
      return "ge::DT_BF16";  // kBFloat16
    case td::pt2::kDtypeUInt16:
      return "ge::DT_UINT16";  // kUInt16
    default:
      throw FallbackCodegenError("unsupported pt2 dtype " + std::to_string(dtype) +
                                 " for HostCpu InferDataType generation; supported constants: kDtypeByte/kDtypeChar/"
                                 "kDtypeShort/kDtypeInt/kDtypeLong/kDtypeHalf/kDtypeFloat/kDtypeDouble/kDtypeBool/"
                                 "kDtypeBFloat16/kDtypeUInt16 (kDtypeUInt32/kDtypeUInt64 and complex dtypes are "
                                 "not supported yet)");
  }
}

std::string FallbackCodegen::GenerateShapeInferMethods(const SchemaInfo &schema, const OpTarget &target) {
  // HostCpu 后端必须提供 shape/dtype 推导：静态维内联常量；动态维从符号
  // 匹配的输入 shape 透传；data-dependent（如 nonzero）无法静态推导，报错。
  std::ostringstream os;

  // 输入 IR 声明与 input_metas 对齐：收集后的每个 inputs 条目恰好对应一个非
  // out 的 schema 参数（标量条目的张量数为 0；auto_functionalized_v2 节点已在
  // CollectShapeInfo 中剔除算子占位并按运行时归一化顺序重排），因此
  // count_cursor 必须逐参数推进 —— 只在 tensor 参数上推进会与
  // input_arg_tensor_counts 的条目基准错位（如 gather(Tensor, int, Tensor)
  // 中第二个张量会读到标量条目）。
  // 每个 IR 声明占 1 个序号（与 Execute 取参的 IR 序号规则一致）。
  struct InputSource {
    size_t ir_index;
    size_t meta_index;
    size_t list_index;  // 变长列表（DYNAMIC_INPUT）内的相对序号，GetDynamicInputShape 第二参
    bool optional;
    bool dynamic;
  };
  std::vector<InputSource> sources;
  {
    size_t meta_cursor = 0;
    size_t count_cursor = 0;
    size_t ir_index = 0;
    for (const auto &arg : schema.arguments) {
      if (arg.is_out) {
        ir_index++;
        continue;
      }
      size_t count = 1;
      if (count_cursor < target.input_arg_tensor_counts.size()) {
        count = target.input_arg_tensor_counts[count_cursor];
      }
      count_cursor++;
      const bool is_variadic_list = arg.is_variadic && arg.type_category == "tensor_list";
      if (arg.type_category == "tensor" || is_variadic_list) {
        for (size_t k = 0; k < count && meta_cursor + k < target.input_metas.size(); ++k) {
          sources.push_back({ir_index, meta_cursor + k, k, IsOptional(arg.input_kind), is_variadic_list});
        }
        meta_cursor += count;
      }
      ir_index++;
    }
  }

  // 输出动态维符号 → 首个同符号输入维（透传来源）。
  struct DimSource {
    size_t ir_index;
    size_t dim;
    const InputSource *src;
  };
  std::map<std::string, DimSource> symbol_map;
  for (const auto &s : sources) {
    const auto &meta = target.input_metas[s.meta_index];
    for (size_t d = 0; d < meta.dynamic_symbols.size(); ++d) {
      if (!meta.dynamic_symbols[d].empty()) {
        symbol_map.emplace(meta.dynamic_symbols[d], DimSource{s.ir_index, d, &s});
      }
    }
  }

  os << "    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override {\n";
  for (size_t o = 0; o < target.output_metas.size(); ++o) {
    const auto &meta = target.output_metas[o];
    os << "        {\n";
    os << "            gert::Shape* out_shape = ctx->GetOutputShape(" << o << ");\n";
    os << "            out_shape->SetDimNum(" << meta.sizes.size() << ");\n";
    for (size_t d = 0; d < meta.sizes.size(); ++d) {
      const auto &sym = meta.dynamic_symbols[d];
      if (sym.empty()) {
        os << "            out_shape->SetDim(" << d << ", " << meta.sizes[d] << ");\n";
        continue;
      }
      auto it = symbol_map.find(sym);
      if (it == symbol_map.end()) {
        throw FallbackCodegenError("CPU op '" + schema.full_op_name + "' (overload '" + schema.overload +
                                   "') output dim " + std::to_string(d) + " depends on symbol '" + sym +
                                   "' not derivable from inputs "
                                   "(data-dependent shape); move this op out of the exported graph");
      }
      const auto &src = it->second;
      if (src.src->dynamic) {
        // 相对序号取符号实际所在变长张量的位置，不能硬编码 0（第 2+ 个变长
        // 张量的动态维会读错输入）。
        os << "            out_shape->SetDim(" << d << ", ctx->GetDynamicInputShape(" << src.ir_index << ", "
           << src.src->list_index << ")->GetDim(" << src.dim << "));\n";
      } else if (src.src->optional) {
        os << "            out_shape->SetDim(" << d << ", ctx->GetOptionalInputShape(" << src.ir_index << ")->GetDim("
           << src.dim << "));\n";
      } else {
        os << "            out_shape->SetDim(" << d << ", ctx->GetRequiredInputShape(" << src.ir_index << ")->GetDim("
           << src.dim << "));\n";
      }
    }
    os << "        }\n";
  }
  os << "        return ge::GRAPH_SUCCESS;\n";
  os << "    }\n";

  os << "    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override {\n";
  for (size_t o = 0; o < target.output_metas.size(); ++o) {
    os << "        ctx->SetOutputDataType(" << o << ", " << Pt2DtypeToGeName(target.output_metas[o].dtype) << ");\n";
  }
  os << "        return ge::GRAPH_SUCCESS;\n";
  os << "    }\n";

  return os.str();
}

std::string FallbackCodegen::GenerateGeOpClass(const SchemaInfo &schema, bool is_host_cpu,
                                               const OpTarget *shape_source) {
  std::ostringstream os;

  std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);

  os << "namespace ge {\n\n";
  if (is_host_cpu) {
    os << "class " << ge_op_name << " final : public HostCpuExecuteOp, public ShapeInferOp {\n";
    os << "public:\n";
    os << GenerateExecuteMethod(schema, is_host_cpu);
    if (shape_source == nullptr || !shape_source->has_shape_info) {
      throw FallbackCodegenError("HostCpu op '" + schema.full_op_name +
                                 "' has no shape info; "
                                 "ShapeInferOp implementation is mandatory for HostCpu backend");
    }
    os << GenerateShapeInferMethods(schema, *shape_source);
    os << "};\n\n";
    os << GenerateGeOpReg(schema) << "\n";
    os << "REG_OP_BACKEND(" << ge_op_name << ", \"" << ge_op_name << "\", ge::OpBackend::kHostCPU);\n";
    os << "} // namespace ge\n\n";
    return os.str();
  }

  os << "class " << ge_op_name << " : public EagerExecuteOp {\n";
  os << "public:\n";
  os << GenerateExecuteMethod(schema, is_host_cpu);
  os << "};\n\n";

  os << GenerateGeOpReg(schema) << "\n";
  os << "REG_AUTO_MAPPING_OP(" << ge_op_name << ");\n";
  os << "} // namespace ge\n\n";

  return os.str();
}

std::string FallbackCodegen::EmitCustomOpCpp(const std::vector<SchemaInfo> &schemas,
                                             const std::vector<OpTarget> &targets, const std::string &gcc_cmd) {
  std::ostringstream os;
  const size_t hostcpu_count =
      std::count_if(targets.begin(), targets.end(), [](const OpTarget &t) { return t.device == "cpu"; });
  TDLOGI("EmitCustomOpCpp: schemas=%zu, hostcpu_ops=%zu, npu_ops=%zu", schemas.size(), hostcpu_count,
         targets.size() - hostcpu_count);

  os << "// AUTO-GENERATED by FallbackCodegen\n";

  if (!gcc_cmd.empty()) {
    os << "//\n";
    os << "// Online compilation command (g++):\n";
    os << "//   " << gcc_cmd << "\n";
    os << "//\n";
    os << "// If g++ compilation fails, try CMake method:\n";
    os << "//   cmake -B build && cmake --build build -j4\n";
    os << "//\n";
  }

  os << "#include <torch/torch.h>\n";
  os << "#include <cstring>\n";
  os << "#include <torch_npu/csrc/aten/common/from_blob.h>\n";
  os << "#include <torch_npu/csrc/core/npu/NPUStream.h>\n";
  os << "#include <torch_npu/csrc/core/npu/NPUGuard.h>\n";
  os << "#include <acl/acl.h>\n";
  if (hostcpu_count > 0) {
    // aclAppLog：HostCpu 算子 DEBUG/ERROR 日志通道（与 TDLOG 同源）。
    os << "#include <acl/acl_base_rt.h>\n";
  }
  os << "#include <graph/custom_op.h>\n";
  os << "#include <graph/types.h>\n";
  os << "#include <graph/operator_reg.h>\n";
  os << "#include <register/op_impl_registry.h>\n\n";

  os << "using namespace ge;\n\n";
  os << "namespace {\n\n";
  os << GenerateHelperFunctions(hostcpu_count > 0);
  os << "}\n\n";

  for (size_t i = 0; i < schemas.size(); ++i) {
    const bool is_host_cpu = i < targets.size() && targets[i].device == "cpu";
    const OpTarget *shape_source = is_host_cpu ? &targets[i] : nullptr;
    os << GenerateGeOpClass(schemas[i], is_host_cpu, shape_source);
  }

  return os.str();
}

std::string FallbackCodegen::EmitCMakeLists(const std::string &output_dir, const std::string &torch_pkg_dir,
                                            const std::string &torch_npu_dir) {
  (void)output_dir;  // Reserved for future use
  std::ostringstream os;

  os << "# AUTO-GENERATED by FallbackCodegen\n";
  os << "#\n";
  os << "# 编译方法：\n";
  os << "#   1. 设置环境变量（推荐）：\n";
  os << "#      source ~/Ascend/ascend-toolkit/set_env.sh  # 设置 ASCEND_HOME_PATH\n";
  os << "#      export TORCH_PKG_DIR=/path/to/site-packages/torch\n";
  os << "#      export TORCH_NPU_DIR=/path/to/libtorch_npu\n";
  os << "#      cmake -B build && cmake --build build -j4\n";
  os << "#\n";
  os << "#   2. 直接指定 cmake 变量：\n";
  os << "#      cmake -B build -DTORCH_PKG_DIR=/path/to/site-packages/torch\n";
  os << "#                 -DASCEND_HOME_PATH=/path/to/cann\n";
  os << "#                 -DTORCH_NPU_DIR=/path/to/libtorch_npu\n";
  os << "#      cmake --build build -j4\n";
  os << "#\n";
  os << "#   3. 离线编译（拷贝到另一环境）：\n";
  os << "#      拷贝 custom_op.cpp 和 CMakeLists.txt 到目标机器\n";
  os << "#      按上述方法设置环境变量或 cmake 参数后编译\n";
  os << "#\n\n";

  os << "cmake_minimum_required(VERSION 3.16)\n";
  os << "project(cust_opapi LANGUAGES CXX)\n\n";

  os << "set(CMAKE_CXX_STANDARD 17)\n";
  os << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -fPIC\")\n\n";

  os << "# ---------------------------------------------------------------------------\n";
  os << "# 依赖路径配置（支持环境变量或 cmake 变量）\n";
  os << "# ---------------------------------------------------------------------------\n";

  os << "# Torch: TORCH_PKG_DIR 指向 Python torch 包目录，内部拼接 TorchConfig.cmake 路径\n";
  if (!torch_pkg_dir.empty() && torch_pkg_dir != "<placeholder>") {
    os << "set(TORCH_PKG_DIR \"\" CACHE PATH \"Path to Python torch package directory\")\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
    os << "    if(DEFINED ENV{TORCH_PKG_DIR})\n";
    os << "        set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})\n";
    os << "    endif()\n";
    os << "endif()\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
    os << "    set(TORCH_PKG_DIR \"" << torch_pkg_dir << "\")  # fallback: codegen 时传入的路径\n";
    os << "endif()\n";
  } else {
    os << "set(TORCH_PKG_DIR \"\" CACHE PATH \"Path to Python torch package directory\")\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\" AND DEFINED ENV{TORCH_PKG_DIR})\n";
    os << "    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})\n";
    os << "endif()\n";
  }
  os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
  os << "    message(FATAL_ERROR \"TORCH_PKG_DIR not set. Please set: export "
        "TORCH_PKG_DIR=/path/to/site-packages/torch\")\n";
  os << "endif()\n";
  os << "set(TORCH_CMAKE_DIR \"${TORCH_PKG_DIR}/share/cmake/Torch\")\n";
  os << "find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_DIR})\n\n";

  os << "# TORCH_NPU_DIR: torch_npu 库目录（包含 lib/libtorch_npu.so）\n";
  os << "set(TORCH_NPU_DIR \"\" CACHE PATH \"torch_npu library directory\")\n";
  if (!torch_npu_dir.empty()) {
    os << "if(TORCH_NPU_DIR STREQUAL \"\")\n";
    os << "    if(DEFINED ENV{TORCH_NPU_DIR})\n";
    os << "        set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})\n";
    os << "    else()\n";
    os << "        set(TORCH_NPU_DIR \"" << torch_npu_dir << "\")  # fallback\n";
    os << "    endif()\n";
    os << "endif()\n";
  } else {
    os << "if(TORCH_NPU_DIR STREQUAL \"\" AND DEFINED ENV{TORCH_NPU_DIR})\n";
    os << "    set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})\n";
    os << "endif()\n";
  }
  os << "\n";

  os << "# ASCEND_HOME_PATH: CANN 安装路径（从环境变量获取）\n";
  os << "set(ASCEND_HOME_PATH \"\" CACHE PATH \"CANN installation root\")\n";
  os << "if(ASCEND_HOME_PATH STREQUAL \"\" AND DEFINED ENV{ASCEND_HOME_PATH})\n";
  os << "    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})\n";
  os << "endif()\n";
  os << "\n";

  os << "# ---------------------------------------------------------------------------\n";
  os << "# 编译配置\n";
  os << "# ---------------------------------------------------------------------------\n";

  os << "include_directories(${ASCEND_HOME_PATH}/include)\n";
  os << "include_directories(${ASCEND_HOME_PATH}/include/graph)\n";
  os << "include_directories(${ASCEND_HOME_PATH}/include/register)\n\n";

  os << "add_library(cust_opapi SHARED custom_op.cpp)\n\n";

  os << "# Derive pytorch source root from TORCH_NPU_DIR (parent of libtorch_npu/)\n";
  os << "# Required for NPUQueue.h -> torch_npu/csrc/logging/LogContext.h\n";
  os << "get_filename_component(PYTORCH_SRC_DIR \"${TORCH_NPU_DIR}/..\" ABSOLUTE)\n\n";
  os << "target_include_directories(cust_opapi PRIVATE\n";
  os << "    ${TORCH_INCLUDE_DIRS}\n";
  os << "    ${TORCH_NPU_DIR}/include\n";
  os << "    ${ASCEND_HOME_PATH}/include\n";
  os << "    ${ASCEND_HOME_PATH}/include/graph\n";
  os << "    ${PYTORCH_SRC_DIR}\n";
  os << ")\n\n";

  os << "target_link_libraries(cust_opapi PRIVATE\n";
  os << "    ${TORCH_LIBRARIES}\n";
  os << "    -Wl,--no-as-needed\n";
  os << "    torch_npu\n";
  os << "    ascendcl\n";
  os << "    -Wl,--as-needed\n";
  os << ")\n\n";

  os << "target_link_directories(cust_opapi PRIVATE\n";
  os << "    ${TORCH_NPU_DIR}/lib\n";
  os << "    ${ASCEND_HOME_PATH}/lib64\n";
  os << ")\n\n";

  os << "set_target_properties(cust_opapi PROPERTIES\n";
  os << "    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}\n";
  os << ")\n";

  return os.str();
}

std::string FallbackCodegen::BuildGccCommandFromPaths(const std::string &torch_pkg_dir,
                                                      const std::string &torch_npu_dir, const std::string &cpp_path,
                                                      const std::string &so_path) {
  std::ostringstream cmd;

  cmd << "g++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -g";

  std::string torch_base = torch_pkg_dir;

  std::string cann_root;
  const char *env = std::getenv("ASCEND_HOME_PATH");
  if (env && strlen(env) > 0) {
    cann_root = env;
  } else {
    TDLOGE("ASCEND_HOME_PATH not set or empty, cannot build g++ command");
    throw FallbackCodegenError("ASCEND_HOME_PATH must be set for online compilation");
  }

  cmd << " -I" << torch_base << "/include";
  cmd << " -I" << torch_base << "/include/torch/csrc/api/include";
  cmd << " -I" << cann_root << "/include";
  cmd << " -I" << cann_root << "/include/graph";
  cmd << " -I" << cann_root << "/include/register";
  cmd << " -I" << torch_npu_dir << "/include";
  cmd << " -I" << (std::filesystem::path(torch_npu_dir) / "..").lexically_normal().string();

  const std::filesystem::path torch_libs = std::filesystem::path(torch_base).parent_path() / "torch.libs";
  cmd << " -L" << torch_base << "/lib";
  if (std::filesystem::exists(torch_libs)) {
    cmd << " -L" << torch_libs.string();
  }
  cmd << " -L" << torch_npu_dir << "/lib";
  cmd << " -L" << cann_root << "/lib64";

  cmd << " -Wl,-rpath," << torch_base << "/lib";
  if (std::filesystem::exists(torch_libs)) {
    cmd << " -Wl,-rpath," << torch_libs.string();
    cmd << " -Wl,-rpath-link," << torch_libs.string();
  }
  cmd << " -Wl,-rpath," << torch_npu_dir << "/lib";
  cmd << " -Wl,-rpath," << cann_root << "/lib64";

  cmd << " -Wl,--no-as-needed";
  cmd << " -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl";
  cmd << " -Wl,--as-needed";

  cmd << " -o " << so_path << " " << cpp_path;

  return cmd.str();
}

void LogLongError(const char *prefix, const std::string &text) {
  constexpr size_t kMaxChunkSize = 800;
  size_t total = text.size();
  if (total == 0) {
    TDLOGE("%s: (empty)", prefix);
    return;
  }
  size_t chunk_idx = 0;
  for (size_t pos = 0; pos < total; pos += kMaxChunkSize) {
    size_t end = std::min(pos + kMaxChunkSize, total);
    TDLOGE("%s [%zu/%zu]: %s", prefix, chunk_idx + 1, (total + kMaxChunkSize - 1) / kMaxChunkSize,
           text.substr(pos, end - pos).c_str());
    chunk_idx++;
  }
}

void FallbackCodegen::CompileWithGcc(const std::string &gcc_cmd) {
  std::array<char, 4096> buf{};
  std::string output;

  std::string cmd_with_stderr = gcc_cmd + " 2>&1";
  FILE *pipe = popen(cmd_with_stderr.c_str(), "r");
  if (!pipe) {
    throw FallbackCodegenError("popen failed for g++ command");
  }
  while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
    output += buf.data();
  }
  int rc = pclose(pipe);
  if (rc != 0) {
    throw FallbackCodegenError("g++ compilation failed: " + output);
  }
}

CodegenResult FallbackCodegen::GenerateFromTargets(const std::vector<OpTarget> &targets,
                                                   const std::string &torch_pkg_dir, const std::string &torch_npu_dir,
                                                   const std::string &output_dir) {
  TDLOGI("FallbackCodegen::GenerateFromTargets() enter, targets=%zu", targets.size());

  CodegenResult result;

  try {
    TDLOGD("Fetching schemas for %zu targets", targets.size());
    std::vector<SchemaInfo> schemas;
    for (const auto &target : targets) {
      std::string full_op = target.namespace_name + "::" + target.op_name;
      TDLOGD("  Getting schema: %s, overload=%s", full_op.c_str(),
             target.overload.empty() ? "(default)" : target.overload.c_str());
      schemas.push_back(GetSchemaInfo(full_op, target.overload));
      if (target.device == "cpu") {
        // CPU dispatch key 预检：部署环境缺 libtorch CPU
        // kernel 时在 codegen 阶段失败，而非运行期 callBoxed 才暴露。
        c10::OperatorName op_name(full_op, target.overload);
        auto handle = torch::Dispatcher::singleton().findSchema(op_name);
        if (!handle.has_value() || !handle->hasComputedKernelForDispatchKey(c10::DispatchKey::CPU)) {
          throw FallbackCodegenError(
              "op '" + full_op + "' (overload '" + target.overload +
              "') has no CPU kernel registered in this environment; HostCpu fallback would fail at runtime. "
              "Register a CPU implementation (torch.library / TORCH_LIBRARY_IMPL) or move the op out of the graph.");
        }
      }
    }

    TDLOGD("Generating custom_op.cpp content");
    std::string cmake_content = EmitCMakeLists(output_dir, torch_pkg_dir, torch_npu_dir);

    for (const auto &schema : schemas) {
      std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);
      result.generated_op_names.push_back(ge_op_name);
    }

    TDLOGI("Online mode: starting compilation");

    const char *keep_env = std::getenv("TORCH_DELEGATE_KEEP_ARTIFACTS");
    bool keep_artifacts = keep_env != nullptr && std::string(keep_env) == "1";
    TDLOGI("FallbackCodegen config: TORCH_DELEGATE_KEEP_ARTIFACTS=%s, keep_artifacts=%s",
           keep_env ? keep_env : "(not set)", keep_artifacts ? "true" : "false");

    std::string work_dir;
    bool cleanup_after_load = true;

    {
      char temp_dir[] = "/tmp/fbcg_XXXXXX";
      if (mkdtemp(temp_dir) == nullptr) {
        TDLOGE("mkdtemp failed: errno=%d", errno);
        throw FallbackCodegenError("mkdtemp failed");
      }
      work_dir = temp_dir;
      cleanup_after_load = !keep_artifacts;
      TDLOGI("Online mode: using temp_dir=%s, keep=%s", work_dir.c_str(), keep_artifacts ? "true" : "false");
    }

    if (!std::filesystem::exists(work_dir)) {
      TDLOGI("Creating working directory: %s", work_dir.c_str());
      std::filesystem::create_directories(work_dir);
    }

    const char *ascend_home = std::getenv("ASCEND_HOME_PATH");
    TDLOGI("Environment check: ASCEND_HOME_PATH=%s", ascend_home ? ascend_home : "(not set)");

    if (ascend_home && strlen(ascend_home) > 0) {
      std::string acl_header = std::string(ascend_home) + "/include/acl/acl.h";
      std::string acl_lib = std::string(ascend_home) + "/lib64/libascendcl.so";
      bool acl_header_exists = std::filesystem::exists(acl_header);
      bool acl_lib_exists = std::filesystem::exists(acl_lib);
      TDLOGI("CANN paths: acl.h=%s (%s), libascendcl.so=%s (%s)", acl_header.c_str(),
             acl_header_exists ? "exists" : "missing", acl_lib.c_str(), acl_lib_exists ? "exists" : "missing");
    }

    std::string cpp_path = work_dir + "/custom_op.cpp";
    std::string cmake_path = work_dir + "/CMakeLists.txt";
    std::string so_path = work_dir + "/libcust_opapi.so";

    std::string gcc_cmd = BuildGccCommandFromPaths(torch_pkg_dir, torch_npu_dir, cpp_path, so_path);
    TDLOGI("Generated g++ command (%zu chars)", gcc_cmd.size());

    std::string cpp_content_with_cmd = EmitCustomOpCpp(schemas, targets, gcc_cmd);
    TDLOGI("Generated custom_op.cpp size: %zu bytes", cpp_content_with_cmd.size());

    {
      std::ofstream f(cpp_path);
      if (!f) {
        TDLOGE("Failed to write custom_op.cpp: %s (errno=%d)", cpp_path.c_str(), errno);
        throw FallbackCodegenError("cannot write custom_op.cpp");
      }
      f << cpp_content_with_cmd;
    }
    TDLOGI("Written custom_op.cpp to: %s", cpp_path.c_str());

    if (keep_artifacts) {
      {
        std::ofstream f(cmake_path);
        if (!f) {
          TDLOGE("Failed to write CMakeLists.txt: %s (errno=%d)", cmake_path.c_str(), errno);
          throw FallbackCodegenError("cannot write CMakeLists.txt");
        }
        f << cmake_content;
      }
      TDLOGI("Written CMakeLists.txt to: %s", cmake_path.c_str());
    }

    TDLOGI("Starting g++ compilation...");
    try {
      CompileWithGcc(gcc_cmd);
      TDLOGI("Compilation succeeded");
    } catch (const FallbackCodegenError &e) {
      LogLongError("Compilation FAILED, command was", gcc_cmd);
      LogLongError("Compilation error", e.what());
      TDLOGE("Generated file location: %s", cpp_path.c_str());
      TDLOGE("To manually compile, copy the g++ command above and fix the error");
      throw;
    }

    TDLOGI("Loading shared library: %s", so_path.c_str());
    result.so_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!result.so_handle) {
      const char *dl_error = dlerror();
      TDLOGE("dlopen FAILED: %s", dl_error ? dl_error : "(unknown error)");
      TDLOGE("Shared library path: %s", so_path.c_str());
      TDLOGE("Library file exists: %s", std::filesystem::exists(so_path) ? "yes" : "no");
      if (std::filesystem::exists(so_path)) {
        TDLOGE("Library file size: %zu bytes", std::filesystem::file_size(so_path));
      }
      throw FallbackCodegenError("dlopen failed: " + std::string(dl_error ? dl_error : "unknown error"));
    }
    TDLOGI("Loaded shared library: %s, handle=%p", so_path.c_str(), result.so_handle);

    if (cleanup_after_load) {
      TDLOGI("Cleaning up temporary files: %s", work_dir.c_str());
      unlink(cpp_path.c_str());
      unlink(so_path.c_str());
      rmdir(work_dir.c_str());
      TDLOGI("Cleanup completed");
    } else {
      std::string model_name =
          schemas.empty() ? "unknown" : GenerateGeOpIdentifier(schemas[0].full_op_name, schemas[0].overload);
      std::string preserve_dir = output_dir + "/" + model_name;
      if (!std::filesystem::exists(preserve_dir)) {
        std::filesystem::create_directories(preserve_dir);
      }
      std::error_code ec;
      std::filesystem::copy_file(cpp_path, preserve_dir + "/custom_op.cpp",
                                 std::filesystem::copy_options::overwrite_existing, ec);
      std::filesystem::copy_file(cmake_path, preserve_dir + "/CMakeLists.txt",
                                 std::filesystem::copy_options::overwrite_existing, ec);
      std::filesystem::copy_file(so_path, preserve_dir + "/libcust_opapi.so",
                                 std::filesystem::copy_options::overwrite_existing, ec);
      TDLOGI("Artifacts copied to: %s", preserve_dir.c_str());
      result.generated_cpp_path = preserve_dir + "/custom_op.cpp";
      result.generated_cmake_path = preserve_dir + "/CMakeLists.txt";
      unlink(cpp_path.c_str());
      unlink(cmake_path.c_str());
      unlink(so_path.c_str());
      rmdir(work_dir.c_str());
    }

    result.success = true;
    TDLOGI("FallbackCodegen::GenerateFromTargets() exit: success, generated %zu ops", result.generated_op_names.size());

  } catch (const FallbackCodegenError &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromTargets() failed: %s", what ? what : "(null)");
    result.success = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
  } catch (const std::exception &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromTargets() failed: %s", what ? what : "(null)");
    result.success = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
  }

  return result;
}

}  // namespace torch_delegate::fallback
