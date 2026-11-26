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

#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <optional>

#include <nlohmann/json.hpp>

namespace torch_delegate::fallback {

// InputKind 枚举：标识参数的类型
enum class InputKind {
  TENSOR,           // Tensor
  OPTIONAL_TENSOR,  // Optional[Tensor]
  TENSOR_LIST,      // List[Tensor]
  INT,              // int
  INT_OPT,          // Optional[int]
  FLOAT,            // float
  FLOAT_OPT,        // Optional[float]
  BOOL,             // bool
  BOOL_OPT,         // Optional[bool]
  STRING,           // str
  STRING_OPT,       // Optional[str]
  INT_LIST,         // List[int]
  INT_LIST_OPT,     // Optional[List[int]]
  FLOAT_LIST,       // List[float]
  FLOAT_LIST_OPT,   // Optional[List[float]]
  BOOL_LIST,        // List[bool]
  BOOL_LIST_OPT,    // Optional[List[bool]]
  SCALAR,           // Scalar (多态类型)
  SCALAR_OPT,       // Optional[Scalar]
  SCALAR_LIST,      // List[Scalar]
  SCALAR_TYPE,      // ScalarType
  SCALAR_TYPE_OPT,  // Optional[ScalarType]
};

struct SchemaArgInfo {
  std::string name;
  std::string type_category;                 // "tensor", "scalar", "tensor_list"
  InputKind input_kind = InputKind::TENSOR;  // 精确的类型标识
  std::string ge_tensor_type;                // GE tensor type (e.g., "TensorType::ALL()", "IntTensorType()")
  bool is_variadic = false;
  bool has_default = false;
  nlohmann::json default_value;  // 默认值（如果有的话）
  bool is_kwarg_only = false;
  bool is_out = false;       // marks out argument (for out versions)
  bool is_optional = false;  // schema 类型是否为 Optional[T]（与“有默认值”无关）
  int fixed_index = -1;      // for variadic args, index of first variadic

  // Alias information (for returns)
  bool has_alias = false;       // whether this arg/return has alias_info
  bool is_alias_write = false;  // whether this is a write alias (e.g., Tensor(a!))
};

struct SchemaInfo {
  std::string full_op_name;  // namespace::op_name
  std::string overload;
  std::string op_name;  // just op_name
  std::string namespace_name;
  std::vector<SchemaArgInfo> arguments;
  std::vector<SchemaArgInfo> returns;
  bool is_inplace = false;
  bool is_out = false;
  bool has_variadic_input = false;
  int num_fixed_inputs = 0;
};

struct OpTarget {
  std::string target;  // full target from model.json
  std::string namespace_name;
  std::string op_name;
  std::string overload;
  // 执行 Device 归属："npu"（缺省）| "cpu"，来自节点 ascend_ir.device。
  std::string device = "npu";
};

struct CodegenResult {
  bool success = false;
  bool skipped = false;
  std::string error_message;
  std::string generated_cpp_path;
  std::string generated_cmake_path;
  void *so_handle = nullptr;
  std::vector<std::string> generated_op_names;
};

class FallbackCodegenError : public std::runtime_error {
 public:
  explicit FallbackCodegenError(const std::string &msg) : std::runtime_error("FallbackCodegen: " + msg) {}
};

class FallbackCodegen {
 public:
  static std::vector<OpTarget> ParseModelJson(const std::string &json_path);

  static std::vector<OpTarget> ParseModelJsonFromString(const std::string &model_json);

  static CodegenResult GenerateFromModelJson(const std::string &model_json);

  static CodegenResult GenerateFromTargets(const std::vector<OpTarget> &targets, const std::string &torch_dir,
                                           const std::string &torch_npu_dir,
                                           const std::string &output_dir = "./output_dir");

  static std::string GenerateGeOpName(const std::string &op_name, const std::string &overload);

  // 生成 GE 算子标识符（用于 REG_OP 注册）
  // 格式：namespace__op_overload（例如：aten__add_Scalar）
  static std::string GenerateGeOpIdentifier(const std::string &full_op_name, const std::string &overload);

 private:
  static std::vector<OpTarget> ParseTargetsFromJson(const nlohmann::json &model);

  static OpTarget ParseTargetFromJson(const nlohmann::json &node);
  static std::pair<std::string, std::string> SplitNamespaceOp(const std::string &full_op);

  static SchemaInfo GetSchemaInfo(const std::string &full_op_name, const std::string &overload);

  static std::string GenerateGeOpClass(const SchemaInfo &schema, bool is_host_cpu);
  static std::string GenerateGeOpReg(const SchemaInfo &schema);
  static std::string GenerateExecuteMethod(const SchemaInfo &schema, bool is_host_cpu);
  static std::string GenerateHelperFunctions(bool need_hostcpu);

  static std::string EmitCustomOpCpp(const std::vector<SchemaInfo> &schemas, const std::vector<OpTarget> &targets,
                                     const std::string &gcc_cmd = "");
  static std::string EmitCMakeLists(const std::string &output_dir, const std::string &torch_dir,
                                    const std::string &torch_npu_dir);

  static std::string BuildGccCommandFromPaths(const std::string &torch_dir, const std::string &torch_npu_dir,
                                              const std::string &cpp_path, const std::string &so_path);
  static void CompileWithGcc(const std::string &gcc_cmd);
};

}  // namespace torch_delegate::fallback
