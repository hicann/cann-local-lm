/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <dlfcn.h>

#define private public
#include "fallback_codegen/fallback_codegen.h"
#undef private

using namespace torch_delegate::fallback;

class FallbackCodegenTest : public ::testing::Test {
 protected:
  void TearDown() override {
    unsetenv("TORCH_PKG_DIR");
    unsetenv("TORCH_NPU_DIR");
    unsetenv("TORCH_DELEGATE_OUTPUT_DIR");
  }
};

const std::string kModelJsonMixedOverloads = R"({
  "nodes": [
    {
      "name": "add_tensor",
      "target": "torch.ops.aten.add.Tensor",
      "inputs": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "source": "converter",
        "ops": [{"name": "Add", "attr": []}]
      }
    },
    {
      "name": "add_scalar",
      "target": "torch.ops.aten.add.Scalar",
      "inputs": [{"name": "x", "type": "tensor"}, {"name": "scalar", "type": "scalar"}],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    },
    {
      "name": "mul_tensor",
      "target": "torch.ops.aten.mul.Tensor",
      "inputs": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "source": "",
        "ops": [{"attr": []}]
      }
    },
    {
      "name": "relu_default",
      "target": "torch.ops.aten.relu.default",
      "inputs": [{"name": "x", "type": "tensor"}],
      "outputs": [{"name": "y", "type": "tensor"}],
      "ascend_ir": {
        "source": "converter",
        "ops": [{"name": "Relu", "attr": []}]
      }
    }
  ]
})";

const std::string kModelJsonAllConverted = R"({
  "nodes": [
    {
      "name": "add_1",
      "target": "torch.ops.aten.add.Tensor",
      "ascend_ir": {
        "source": "converter",
        "ops": [{"name": "Add"}]
      }
    },
    {
      "name": "mul_1",
      "target": "torch.ops.aten.mul.Tensor",
      "ascend_ir": {
        "source": "converter",
        "ops": [{"name": "Mul"}]
      }
    }
  ]
})";

const std::string kModelJsonNoAscendIr = R"({
  "nodes": [
    {
      "name": "custom_op_1",
      "target": "torch.ops.my_ns.custom_op.default",
      "inputs": [{"name": "x", "type": "tensor"}],
      "outputs": [{"name": "y", "type": "tensor"}]
    },
    {
      "name": "custom_op_2",
      "target": "torch.ops.my_ns.custom_op.overload1",
      "inputs": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
      "outputs": [{"name": "z", "type": "tensor"}]
    }
  ]
})";

const std::string kModelJsonAutoFunctionalized = R"({
  "nodes": [
    {
      "name": "rope_in_place_node",
      "target": "torch.ops.higher_order.auto_functionalized_v2",
      "inputs": [
        {
          "name": "",
          "arg": {
            "as_operator": "torch.ops.xllm_ascend.rope_in_place.default"
          },
          "kind": 1
        },
        {
          "name": "sin_cache",
          "arg": {
            "as_tensor": {
              "name": "sin_cache"
            }
          },
          "kind": 2
        },
        {
          "name": "cos_cache",
          "arg": {
            "as_tensor": {
              "name": "cos_cache"
            }
          },
          "kind": 2
        }
      ],
      "outputs": [
        {
          "as_none": true
        },
        {
          "as_tensor": {
            "name": "getitem_1"
          }
        }
      ],
      "ascend_ir": {
        "ops": [
          {
            "attr": {
              "_inference_rule": {
                "s": "ewogICJzaGFwZSI6IHsKICAgICJpbnB1dHMiOiBbXSwKICAgICJvdXRwdXRzIjogW10KICB9LAogICJkdHlwZSI6IFtdCn0="
              }
            }
          }
        ]
      }
    }
  ]
})";

const std::string kModelJsonExportedProgramFormat = R"({
  "graph_module": {
    "graph": {
      "nodes": [
        {
          "target": "torch.ops.xllm_ascend.rope_in_place.default",
          "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "hidden_states"}}}
          ],
          "outputs": [
            {"as_tensor": {"name": "rope_in_place"}}
          ],
          "ascend_ir": {
            "ops": [
              {
                "attr": {
                  "_inference_rule": {"s": "base64_encoded_content"}
                }
              }
            ]
          }
        },
        {
          "target": "torch.ops.aten.add.Tensor",
          "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "a"}}}],
          "outputs": [{"as_tensor": {"name": "b"}}],
          "ascend_ir": {
            "source": "converter",
            "ops": [{"name": "Add"}]
          }
        }
      ]
    }
  }
})";

const std::string kModelJsonMixedSources = R"({
  "nodes": [
    {
      "name": "converted_node",
      "target": "torch.ops.aten.add.Tensor",
      "ascend_ir": {
        "source": "converter",
        "ops": [{"name": "Add"}]
      }
    },
    {
      "name": "mock_node_no_field",
      "target": "torch.ops.my_ns.op1.default",
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    },
    {
      "name": "mock_node_empty_source",
      "target": "torch.ops.my_ns.op2.default",
      "ascend_ir": {
        "source": "",
        "ops": [{"attr": []}]
      }
    },
    {
      "name": "no_ascend_ir_node",
      "target": "torch.ops.my_ns.op3.default",
      "inputs": [{"name": "x", "type": "tensor"}]
    }
  ]
})";

TEST_F(FallbackCodegenTest, ParseMixedOverloadsFilterBySource) {
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonMixedOverloads);

  ASSERT_EQ(targets.size(), 2u);

  EXPECT_EQ(targets[0].target, "torch.ops.aten.add.Scalar");
  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "add");
  EXPECT_EQ(targets[0].overload, "Scalar");

  EXPECT_EQ(targets[1].target, "torch.ops.aten.mul.Tensor");
  EXPECT_EQ(targets[1].namespace_name, "aten");
  EXPECT_EQ(targets[1].op_name, "mul");
  EXPECT_EQ(targets[1].overload, "Tensor");
}

TEST_F(FallbackCodegenTest, ParseAllConvertedNodesSkipped) {
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonAllConverted);

  EXPECT_EQ(targets.size(), 0u);
}

TEST_F(FallbackCodegenTest, ParseNodesWithoutAscendIrField) {
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonNoAscendIr);

  EXPECT_EQ(targets.size(), 0u);
}

TEST_F(FallbackCodegenTest, ParseMixedSourcesField) {
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonMixedSources);

  ASSERT_EQ(targets.size(), 2u);

  EXPECT_EQ(targets[0].target, "torch.ops.my_ns.op1.default");
  EXPECT_EQ(targets[1].target, "torch.ops.my_ns.op2.default");
}

TEST_F(FallbackCodegenTest, ParseAutoFunctionalizedV2Node) {
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonAutoFunctionalized);

  ASSERT_EQ(targets.size(), 1u);

  // After unwrapping auto_functionalized_v2, target should be the real operator
  EXPECT_EQ(targets[0].target, "torch.ops.xllm_ascend.rope_in_place.default");
  EXPECT_EQ(targets[0].namespace_name, "xllm_ascend");
  EXPECT_EQ(targets[0].op_name, "rope_in_place");
  EXPECT_EQ(targets[0].overload, "");
}

TEST_F(FallbackCodegenTest, ParseExportedProgramJsonFormat) {
  // Test parsing ExportedProgram JSON format (graph_module.graph.nodes)
  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonExportedProgramFormat);

  // Should parse 1 node (rope_in_place has no source field, needs fallback)
  // add.Tensor has source="converter", not included
  ASSERT_EQ(targets.size(), 1u);

  EXPECT_EQ(targets[0].target, "torch.ops.xllm_ascend.rope_in_place.default");
  EXPECT_EQ(targets[0].namespace_name, "xllm_ascend");
  EXPECT_EQ(targets[0].op_name, "rope_in_place");
  EXPECT_EQ(targets[0].overload, "");
}

TEST_F(FallbackCodegenTest, GenerateFromModelJsonAllConvertedSkipped) {
  CodegenResult result = FallbackCodegen::GenerateFromModelJson(kModelJsonAllConverted);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.skipped);
  EXPECT_FALSE(result.error_message.empty());
  EXPECT_EQ(result.generated_op_names.size(), 0u);
}

TEST_F(FallbackCodegenTest, GenerateFromModelJsonMissingEnvVars) {
  unsetenv("TORCH_PKG_DIR");
  unsetenv("TORCH_NPU_DIR");

  CodegenResult result = FallbackCodegen::GenerateFromModelJson(kModelJsonMixedSources);

  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.skipped);
  EXPECT_FALSE(result.error_message.empty());
  EXPECT_TRUE(result.error_message.find("TORCH_PKG_DIR") != std::string::npos);
  EXPECT_TRUE(result.error_message.find("TORCH_NPU_DIR") != std::string::npos);
}

TEST_F(FallbackCodegenTest, GenerateFromModelJsonInvalidJson) {
  std::string invalid_json = "{ invalid json content }";

  EXPECT_THROW({ FallbackCodegen::ParseModelJsonFromString(invalid_json); }, FallbackCodegenError);
}

TEST_F(FallbackCodegenTest, ParseModelJsonFromStringEmptyJson) {
  std::string empty_nodes_json = R"({"nodes": []})";

  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(empty_nodes_json);

  EXPECT_EQ(targets.size(), 0u);
}

TEST_F(FallbackCodegenTest, ParseTargetFormatVariations) {
  std::string json_with_formats = R"({
  "nodes": [
    {
      "target": "torch.ops.aten.topk.default",
      "ascend_ir": {"ops": []}
    },
    {
      "target": "torch.ops.aten.add.Tensor",
      "ascend_ir": {"ops": []}
    },
    {
      "target": "torch.ops.custom_ns.my_op.default",
      "ascend_ir": {"ops": []}
    }
  ]
})";

  std::vector<OpTarget> targets = FallbackCodegen::ParseModelJsonFromString(json_with_formats);

  ASSERT_EQ(targets.size(), 3u);

  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "topk");
  EXPECT_EQ(targets[0].overload, "");

  EXPECT_EQ(targets[1].namespace_name, "aten");
  EXPECT_EQ(targets[1].op_name, "add");
  EXPECT_EQ(targets[1].overload, "Tensor");

  EXPECT_EQ(targets[2].namespace_name, "custom_ns");
  EXPECT_EQ(targets[2].op_name, "my_op");
  EXPECT_EQ(targets[2].overload, "");
}

// DISABLED: 在线编译用例依赖真实 torch/torch_npu 安装（g++ 实时编译并 dlopen 产物 .so），
// stub + ASAN 的 UT 环境无法满足，默认禁用，不随 build_test.sh 执行。
// 手工运行：先 export TORCH_PKG_DIR / TORCH_NPU_DIR（指向真实安装），再单独过滤运行本用例
// （fixture 的 TearDown 会 unset 这两个变量，勿与其他用例同批运行）：
//   ./test_fallback_codegen --gtest_also_run_disabled_tests
//       --gtest_filter=FallbackCodegenTest.DISABLED_GenerateFromModelJsonOnlineCompilation
TEST_F(FallbackCodegenTest, DISABLED_GenerateFromModelJsonOnlineCompilation) {
  const char *torch_pkg_dir = std::getenv("TORCH_PKG_DIR");
  const char *torch_npu_dir = std::getenv("TORCH_NPU_DIR");
  if (torch_pkg_dir == nullptr || torch_npu_dir == nullptr) {
    GTEST_SKIP() << "TORCH_PKG_DIR / TORCH_NPU_DIR not set, online compilation requires real torch/torch_npu";
  }
  setenv("TORCH_DELEGATE_OUTPUT_DIR", "/tmp/test_fallback_output", 1);

  CodegenResult result = FallbackCodegen::GenerateFromModelJson(kModelJsonMixedSources);

  if (result.success) {
    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.generated_op_names.size(), 3u);
    EXPECT_NE(result.so_handle, nullptr);

    if (result.so_handle) {
      dlclose(result.so_handle);
    }
  } else {
    std::cerr << "[WARNING] Online compilation test skipped: " << result.error_message << std::endl;
  }
}

TEST_F(FallbackCodegenTest, CodegenResultFieldsValidation) {
  CodegenResult result;
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.skipped);
  EXPECT_TRUE(result.error_message.empty());
  EXPECT_TRUE(result.generated_cpp_path.empty());
  EXPECT_TRUE(result.generated_cmake_path.empty());
  EXPECT_EQ(result.so_handle, nullptr);
  EXPECT_EQ(result.generated_op_names.size(), 0u);
}

TEST_F(FallbackCodegenTest, GenerateGeOpIdentifierFormat) {
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("aten::add", "Tensor"), "aten__add_Tensor");
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("aten::relu_", ""), "aten__relu_");
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("aten::add", "out"), "aten__add_out");
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("my_ops::scale_add", ""), "my_ops__scale_add");
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("aten::add", ""), "aten__add");
  EXPECT_EQ(FallbackCodegen::GenerateGeOpIdentifier("aten::mm", "default"), "aten__mm_default");
}

TEST_F(FallbackCodegenTest, ParseTargetFromJsonExtractsTargetInfo) {
  const std::string json = R"({
      "nodes": [
        {
          "name": "scale_add",
          "target": "torch.ops.my_ops.scale_add.default",
          "inputs": [
            {"name": "x", "arg": {"as_tensor": {"name": "x"}}},
            {"name": "y", "arg": {"as_tensor": {"name": "y"}}},
            {"name": "scale", "arg": {"as_float": 0.5}},
            {"name": "shift", "arg": {"as_int": 2}}
          ],
          "outputs": [{"name": "z", "type": "tensor"}],
          "ascend_ir": {
            "ops": [{"attr": []}]
          }
        }
      ]
    })";

  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "my_ops");
  EXPECT_EQ(targets[0].op_name, "scale_add");
}

TEST_F(FallbackCodegenTest, ParseTargetFromJsonWithNoneInputs) {
  const std::string json = R"({
      "nodes": [
        {
          "name": "op_with_none",
          "target": "torch.ops.aten.add.Tensor",
          "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
            {"name": "other", "arg": {"as_none": {}}},
            {"name": "alpha", "arg": {"as_int": 1}}
          ],
          "outputs": [{"name": "z", "type": "tensor"}],
          "ascend_ir": {
            "ops": [{"attr": []}]
          }
        }
      ]
    })";

  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "add");
  EXPECT_EQ(targets[0].overload, "Tensor");
}

const std::string kModelJsonWithScalarInputs = R"({
  "nodes": [
    {
      "name": "add_scalar",
      "target": "torch.ops.aten.add.Scalar",
      "inputs": [
        {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "other", "arg": {"as_int": 5}},
        {"name": "alpha", "arg": {"as_float": 1.0}}
      ],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    }
  ]
})";

TEST_F(FallbackCodegenTest, ParseModelJsonFromStringWithScalarInputs) {
  auto targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonWithScalarInputs);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "add");
  EXPECT_EQ(targets[0].overload, "Scalar");
}

const std::string kModelJsonWithSymInputs = R"({
  "nodes": [
    {
      "name": "custom_op",
      "target": "torch.ops.my_ops.custom.default",
      "inputs": [
        {"name": "x", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "batch_size", "arg": {"as_sym_int": {"as_name": "batch_size"}}},
        {"name": "threshold", "arg": {"as_sym_float": {"as_name": "threshold"}}},
        {"name": "flag", "arg": {"as_sym_bool": {"as_name": "flag"}}}
      ],
      "outputs": [{"name": "y", "type": "tensor"}],
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    }
  ]
})";

TEST_F(FallbackCodegenTest, ParseModelJsonFromStringWithSymInputs) {
  auto targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonWithSymInputs);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "my_ops");
  EXPECT_EQ(targets[0].op_name, "custom");
}

const std::string kModelJsonMixedScalarAndTensor = R"({
  "nodes": [
    {
      "name": "add_scalar",
      "target": "torch.ops.aten.add.Scalar",
      "inputs": [
        {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "other", "arg": {"as_int": 10}}
      ],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    }
  ]
})";

TEST_F(FallbackCodegenTest, ParseMixedScalarAndTensorInputs) {
  auto targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonMixedScalarAndTensor);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "add");
  EXPECT_EQ(targets[0].overload, "Scalar");
}

const std::string kModelJsonEmptyInputs = R"({
  "nodes": [
    {
      "name": "no_input_op",
      "target": "torch.ops.aten.zeros.default",
      "inputs": [],
      "outputs": [{"name": "z", "type": "tensor"}],
      "ascend_ir": {
        "ops": [{"attr": []}]
      }
    }
  ]
})";

TEST_F(FallbackCodegenTest, ParseEmptyInputsArray) {
  auto targets = FallbackCodegen::ParseModelJsonFromString(kModelJsonEmptyInputs);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].namespace_name, "aten");
  EXPECT_EQ(targets[0].op_name, "zeros");
}

// ===========================================================================
// Dynamic Input (Tensor[]) Codegen Tests
// Schema: cat(tensors: Tensor[], dim: int) -> Tensor
// IR layout: [0] DYNAMIC_INPUT(tensors), [1] INPUT(dim)
// ===========================================================================

static SchemaInfo MakeDynamicInputSchema() {
  SchemaInfo schema;
  schema.full_op_name = "aten::cat";
  schema.overload = "";
  schema.op_name = "cat";
  schema.namespace_name = "aten";
  schema.is_inplace = false;
  schema.is_out = false;
  schema.has_variadic_input = true;
  schema.num_fixed_inputs = 1;

  SchemaArgInfo tensors_arg;
  tensors_arg.name = "tensors";
  tensors_arg.type_category = "tensor_list";
  tensors_arg.input_kind = InputKind::TENSOR_LIST;
  tensors_arg.is_variadic = true;
  tensors_arg.ge_tensor_type = "TensorType::ALL()";
  schema.arguments.push_back(tensors_arg);

  SchemaArgInfo dim_arg;
  dim_arg.name = "dim";
  dim_arg.type_category = "int";
  dim_arg.input_kind = InputKind::INT;
  dim_arg.ge_tensor_type = "IntTensorType()";
  schema.arguments.push_back(dim_arg);

  SchemaArgInfo output;
  output.name = "result";
  output.type_category = "tensor";
  schema.returns.push_back(output);

  return schema;
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_DynamicInput_UsesGetDynamicInputTensor) {
  SchemaInfo schema = MakeDynamicInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  std::cerr << "===== Generated Execute (Dynamic Input) =====\n" << code << "\n";

  // Should use GetDynamicInputTensor(ir_index=0, relative_index=i)
  EXPECT_NE(code.find("GetDynamicInputTensor(0"), std::string::npos)
      << "Must use GetDynamicInputTensor(ir_idx, rel_idx) for DYNAMIC_INPUT";
  // Should NOT use flat-indexed GetInputTensor for the dynamic input
  EXPECT_EQ(code.find("GetInputTensor(0 + i)"), std::string::npos)
      << "Must not use flat-indexed GetInputTensor for dynamic input";
  // Should pack tensors into c10::List<at::Tensor> for callBoxed
  EXPECT_NE(code.find("c10::List<at::Tensor>"), std::string::npos)
      << "Must pack dynamic input tensors into c10::List<at::Tensor>";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_DynamicInput_NReadFromInstanceInfo) {
  SchemaInfo schema = MakeDynamicInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  // N (instance count) must be read from GetIrInputInstanceInfo, not from attrs
  EXPECT_NE(code.find("GetIrInputInstanceInfo"), std::string::npos)
      << "Must use GetIrInputInstanceInfo to get DYNAMIC_INPUT instance count";
  EXPECT_NE(code.find("GetInstanceNum"), std::string::npos) << "Must call GetInstanceNum() on the instance info";
  // Should not use bare N as loop bound without definition
  EXPECT_EQ(code.find("static_cast<size_t>(N)"), std::string::npos) << "N must not be used as undefined variable";
  // Should not read from attrs for N
  EXPECT_EQ(code.find("GetAttrs"), std::string::npos) << "Must not use GetAttrs for DYNAMIC_INPUT count";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_OutputWrite_MaterializesNonContiguousResult) {
  SchemaInfo schema = MakeDynamicInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  // Output write must materialize non-contiguous views before the flat memcpy,
  // otherwise data_ptr() + numel copies the raw storage order (silent corruption)
  EXPECT_NE(code.find("if (!result.is_contiguous()) { result = result.contiguous(); }"), std::string::npos)
      << "Must normalize non-contiguous results before aclrtMemcpyAsync";

  // StorageFormat must be initialized (uninitialized format is UB metadata)
  EXPECT_NE(code.find("gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {})"), std::string::npos)
      << "Must initialize StorageFormat instead of leaving it default-constructed";
  EXPECT_EQ(code.find("gert::StorageFormat format;"), std::string::npos) << "Must not emit uninitialized StorageFormat";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_DynamicInput_DimUsesCorrectIrIndex) {
  SchemaInfo schema = MakeDynamicInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  // dim is at IR index 1 (after DYNAMIC_INPUT at IR 0)
  // Should use GetRequiredInputTensor(1) or similar IR-indexed API
  EXPECT_NE(code.find("GetRequiredInputTensor(1)"), std::string::npos)
      << "dim (IR index 1) must use GetRequiredInputTensor(1)";
}

TEST_F(FallbackCodegenTest, GenerateGeOpReg_DynamicInput_RegOpCorrect) {
  SchemaInfo schema = MakeDynamicInputSchema();
  std::string reg = FallbackCodegen::GenerateGeOpReg(schema);

  std::cerr << "===== Generated REG_OP (Dynamic Input) =====\n" << reg << "\n";

  // REG_OP declarations are correct
  EXPECT_NE(reg.find("DYNAMIC_INPUT(tensors"), std::string::npos);
  // N is no longer declared as REQUIRED_ATTR — instance count comes from GetIrInputInstanceInfo
  EXPECT_EQ(reg.find("REQUIRED_ATTR(N, Int)"), std::string::npos)
      << "Must not declare REQUIRED_ATTR(N) — use GetIrInputInstanceInfo instead";
  // InputsDataDependency uses IR index 1 for dim (correct, not flat N)
  EXPECT_NE(reg.find("InputsDataDependency({1})"), std::string::npos);
}

// ===========================================================================
// Optional Input Codegen Tests
// Schema: addmm(input: Tensor, mat1: Tensor, mat2: Tensor, bias: Optional[Tensor], beta: Scalar) -> Tensor
// IR layout: [0] INPUT(input), [1] INPUT(mat1), [2] INPUT(mat2),
//            [3] OPTIONAL_INPUT(bias), [4] INPUT(beta)
// ===========================================================================

static SchemaInfo MakeOptionalInputSchema() {
  SchemaInfo schema;
  schema.full_op_name = "aten::addmm";
  schema.overload = "default";
  schema.op_name = "addmm";
  schema.namespace_name = "aten";
  schema.is_inplace = false;
  schema.is_out = false;
  schema.has_variadic_input = false;
  schema.num_fixed_inputs = 3;

  auto make_tensor = [](const std::string &name, InputKind kind) {
    SchemaArgInfo arg;
    arg.name = name;
    arg.type_category = "tensor";
    arg.input_kind = kind;
    arg.ge_tensor_type = "TensorType::ALL()";
    return arg;
  };

  schema.arguments.push_back(make_tensor("input", InputKind::TENSOR));
  schema.arguments.push_back(make_tensor("mat1", InputKind::TENSOR));
  schema.arguments.push_back(make_tensor("mat2", InputKind::TENSOR));
  schema.arguments.push_back(make_tensor("bias", InputKind::OPTIONAL_TENSOR));

  SchemaArgInfo beta_arg;
  beta_arg.name = "beta";
  beta_arg.type_category = "scalar";
  beta_arg.input_kind = InputKind::SCALAR;
  beta_arg.ge_tensor_type = "TensorType::ALL()";
  schema.arguments.push_back(beta_arg);

  SchemaArgInfo output;
  output.name = "result";
  output.type_category = "tensor";
  schema.returns.push_back(output);

  return schema;
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_OptionalInput_UsesGetOptionalInputTensor) {
  SchemaInfo schema = MakeOptionalInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  std::cerr << "===== Generated Execute (Optional Input) =====\n" << code << "\n";

  // bias is OPTIONAL_INPUT at IR index 3, must use GetOptionalInputTensor(3)
  EXPECT_NE(code.find("GetOptionalInputTensor(3)"), std::string::npos)
      << "bias (OPTIONAL at IR 3) must use GetOptionalInputTensor(3)";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_OptionalInput_RequiredTensorsUseCorrectApi) {
  SchemaInfo schema = MakeOptionalInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  // input/mat1/mat2 are REQUIRED at IR 0/1/2, must use GetRequiredInputTensor
  EXPECT_NE(code.find("GetRequiredInputTensor(0)"), std::string::npos)
      << "input (IR 0) must use GetRequiredInputTensor(0)";
  EXPECT_NE(code.find("GetRequiredInputTensor(1)"), std::string::npos)
      << "mat1 (IR 1) must use GetRequiredInputTensor(1)";
  EXPECT_NE(code.find("GetRequiredInputTensor(2)"), std::string::npos)
      << "mat2 (IR 2) must use GetRequiredInputTensor(2)";
  // beta is SCALAR at IR 4, also a required input
  EXPECT_NE(code.find("GetRequiredInputTensor(4)"), std::string::npos)
      << "beta (IR 4) must use GetRequiredInputTensor(4)";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_OptionalInput_NullCheckEffective) {
  SchemaInfo schema = MakeOptionalInputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  // The nullptr check should be on GetOptionalInputTensor result (effective)
  // Should not use flat-indexed GetInputTensor for optional input
  EXPECT_EQ(code.find("GetInputTensor(3)"), std::string::npos)
      << "Must not use flat-indexed GetInputTensor for optional input";
}

TEST_F(FallbackCodegenTest, GenerateGeOpReg_OptionalInput_RegOpCorrect) {
  SchemaInfo schema = MakeOptionalInputSchema();
  std::string reg = FallbackCodegen::GenerateGeOpReg(schema);

  std::cerr << "===== Generated REG_OP (Optional Input) =====\n" << reg << "\n";

  // REG_OP declaration is correct: OPTIONAL_INPUT declared
  EXPECT_NE(reg.find("OPTIONAL_INPUT(bias"), std::string::npos);
  // bias is at IR index 3, not a data dependency (not in InputsDataDependency)
  // beta is at IR index 4, is a data dependency (SCALAR -> IsValueDependency=true)
  EXPECT_NE(reg.find("InputsDataDependency({4})"), std::string::npos);
}

// ===========================================================================
// Mixed Dynamic + Optional Input Codegen Test
// Schema: foo(tensors: Tensor[], bias: Optional[Tensor], dim: int) -> Tensor
// IR layout: [0] DYNAMIC_INPUT(tensors), [1] OPTIONAL_INPUT(bias), [2] INPUT(dim)
// Exercises both fixes simultaneously
// ===========================================================================

static SchemaInfo MakeMixedDynamicOptionalSchema() {
  SchemaInfo schema;
  schema.full_op_name = "aten::foo";
  schema.overload = "default";
  schema.op_name = "foo";
  schema.namespace_name = "aten";
  schema.is_inplace = false;
  schema.is_out = false;
  schema.has_variadic_input = true;
  schema.num_fixed_inputs = 1;

  SchemaArgInfo tensors_arg;
  tensors_arg.name = "tensors";
  tensors_arg.type_category = "tensor_list";
  tensors_arg.input_kind = InputKind::TENSOR_LIST;
  tensors_arg.is_variadic = true;
  tensors_arg.ge_tensor_type = "TensorType::ALL()";
  schema.arguments.push_back(tensors_arg);

  SchemaArgInfo bias_arg;
  bias_arg.name = "bias";
  bias_arg.type_category = "tensor";
  bias_arg.input_kind = InputKind::OPTIONAL_TENSOR;
  bias_arg.ge_tensor_type = "TensorType::ALL()";
  schema.arguments.push_back(bias_arg);

  SchemaArgInfo dim_arg;
  dim_arg.name = "dim";
  dim_arg.type_category = "int";
  dim_arg.input_kind = InputKind::INT;
  dim_arg.ge_tensor_type = "IntTensorType()";
  schema.arguments.push_back(dim_arg);

  SchemaArgInfo output;
  output.name = "result";
  output.type_category = "tensor";
  schema.returns.push_back(output);

  return schema;
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_MixedDynamicAndOptional) {
  SchemaInfo schema = MakeMixedDynamicOptionalSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  std::cerr << "===== Generated Execute (Mixed Dynamic + Optional) =====\n" << code << "\n";

  // 1. DYNAMIC_INPUT(tensors) at IR 0: must use GetDynamicInputTensor(0, i)
  EXPECT_NE(code.find("GetDynamicInputTensor(0"), std::string::npos)
      << "tensors (DYNAMIC at IR 0) must use GetDynamicInputTensor(0, i)";
  // 2. OPTIONAL_INPUT(bias) at IR 1: must use GetOptionalInputTensor(1)
  EXPECT_NE(code.find("GetOptionalInputTensor(1)"), std::string::npos)
      << "bias (OPTIONAL at IR 1) must use GetOptionalInputTensor(1)";
  // 3. INPUT(dim) at IR 2: must use GetRequiredInputTensor(2)
  EXPECT_NE(code.find("GetRequiredInputTensor(2)"), std::string::npos)
      << "dim (REQUIRED at IR 2) must use GetRequiredInputTensor(2)";
  // 4. N (instance count) must be read from GetIrInputInstanceInfo
  EXPECT_NE(code.find("GetIrInputInstanceInfo"), std::string::npos)
      << "Must use GetIrInputInstanceInfo for DYNAMIC_INPUT count";
  // 5. Must not use flat-indexed GetInputTensor for dynamic/optional
  EXPECT_EQ(code.find("GetInputTensor(0 + i)"), std::string::npos)
      << "Must not use flat-indexed GetInputTensor for dynamic input";
}

// ===========================================================================
// ReadScalarInput placement-aware codegen tests
// 验证：device 张量走 D2H 拷贝；host 张量直接借用 GetAddr()（不深拷贝）
// ===========================================================================
TEST_F(FallbackCodegenTest, GenerateHelperFunctions_ReadScalarInput_HasPlacementBranch) {
  std::string code = FallbackCodegen::GenerateHelperFunctions(false);

  std::cerr << "===== Generated Helper Functions (ReadScalarInput) =====\n" << code << "\n";

  // 1. 必须定义 ScalarInputBuf 视图类型（drop-in 替换 vector<uint8_t>）
  EXPECT_NE(code.find("struct ScalarInputBuf"), std::string::npos) << "Must define ScalarInputBuf struct";
  // 视图类型必须提供 data() 与 size() 方法（与 vector<uint8_t> 接口一致）
  EXPECT_NE(code.find("const uint8_t* data() const"), std::string::npos) << "ScalarInputBuf must expose data()";
  EXPECT_NE(code.find("size_t size() const"), std::string::npos) << "ScalarInputBuf must expose size()";

  // 2. ReadScalarInput 返回类型必须改为 ScalarInputBuf
  EXPECT_NE(code.find("ScalarInputBuf ReadScalarInput("), std::string::npos)
      << "ReadScalarInput must return ScalarInputBuf";

  // 3. host 分支：用 IsOnHost 判断 placement，直接借用 GetAddr()，不调用 aclrtMemcpy
  EXPECT_NE(code.find("TensorPlacementUtils::IsOnHost"), std::string::npos) << "Must branch on IsOnHost(placement)";
  EXPECT_NE(code.find("tensor->GetPlacement()"), std::string::npos) << "Must query tensor placement";

  // 4. device 分支：用 IsOnDevice 精确判断，保留 ACL_MEMCPY_DEVICE_TO_HOST（D2H 拷贝）
  EXPECT_NE(code.find("TensorPlacementUtils::IsOnDevice"), std::string::npos) << "Must branch on IsOnDevice(placement)";
  EXPECT_NE(code.find("ACL_MEMCPY_DEVICE_TO_HOST"), std::string::npos)
      << "Device path must still use ACL_MEMCPY_DEVICE_TO_HOST";

  // 5. 其他 placement 一律报错（严格原则）
  EXPECT_NE(code.find("TORCH_CHECK(false, \"ReadScalarInput: unsupported tensor placement"), std::string::npos)
      << "Unknown placement must trigger TORCH_CHECK error";
}

TEST_F(FallbackCodegenTest, GenerateHelperFunctions_ReadScalarInput_HostPathNoMemcpy) {
  std::string code = FallbackCodegen::GenerateHelperFunctions(false);

  // host 借用分支：直接返回 GetAddr() 指针，不调用 aclrtMemcpy
  // 验证策略：整个 ReadScalarInput 函数中 aclrtMemcpy 仅出现一次（device 路径），
  //          且 host 借用返回语句存在（含 GetAddr() 借用）
  size_t fn_start = code.find("ReadScalarInput(const gert::Tensor*");
  ASSERT_NE(fn_start, std::string::npos) << "ReadScalarInput function must exist";
  size_t fn_end = code.find("\n}\n", fn_start);
  ASSERT_NE(fn_end, std::string::npos) << "ReadScalarInput function must close";
  std::string fn_body = code.substr(fn_start, fn_end - fn_start);

  // aclrtMemcpy 仅出现一次（device 路径）
  size_t memcpy_count = 0;
  size_t search_pos = 0;
  while ((search_pos = fn_body.find("aclrtMemcpy", search_pos)) != std::string::npos) {
    memcpy_count++;
    search_pos += std::string("aclrtMemcpy").length();
  }
  EXPECT_EQ(memcpy_count, 1u) << "aclrtMemcpy must appear exactly once (device path only)";

  // host 借用返回语句存在：return {{}, ... tensor->GetAddr() ...}
  EXPECT_NE(fn_body.find("return {{}, static_cast<const uint8_t*>(tensor->GetAddr())"), std::string::npos)
      << "Host path must borrow tensor->GetAddr() directly";

  // device 路径返回语句存在：return {std::move(buf), nullptr, byte_size};
  EXPECT_NE(fn_body.find("return {std::move(buf), nullptr, byte_size}"), std::string::npos)
      << "Device path must return owned buffer";
}

// ===========================================================================
// Dynamic Output (-> Tensor[]) Codegen Tests
// Schema: my_split(Tensor input, int n) -> Tensor[]
// IR layout: [0] INPUT(input), [1] INPUT(n)
// Output: [0] DYNAMIC_OUTPUT(outputs)
// ===========================================================================

static SchemaInfo MakeDynamicOutputSchema() {
  SchemaInfo schema;
  schema.full_op_name = "my_ops::my_split";
  schema.overload = "";
  schema.op_name = "my_split";
  schema.namespace_name = "my_ops";
  schema.is_inplace = false;
  schema.is_out = false;
  schema.has_variadic_input = false;
  schema.num_fixed_inputs = 2;

  SchemaArgInfo input_arg;
  input_arg.name = "input";
  input_arg.type_category = "tensor";
  input_arg.input_kind = InputKind::TENSOR;
  input_arg.ge_tensor_type = "TensorType::ALL()";
  schema.arguments.push_back(input_arg);

  SchemaArgInfo n_arg;
  n_arg.name = "n";
  n_arg.type_category = "int";
  n_arg.input_kind = InputKind::INT;
  n_arg.ge_tensor_type = "IntTensorType()";
  schema.arguments.push_back(n_arg);

  SchemaArgInfo output;
  output.name = "outputs";
  output.type_category = "tensor_list";
  output.is_variadic = true;
  schema.returns.push_back(output);

  return schema;
}

TEST_F(FallbackCodegenTest, GenerateGeOpReg_DynamicOutput_HasDynamicOutputDecl) {
  SchemaInfo schema = MakeDynamicOutputSchema();
  std::string code = FallbackCodegen::GenerateGeOpReg(schema);

  std::cerr << "===== Generated REG_OP (Dynamic Output) =====\n" << code << "\n";

  EXPECT_NE(code.find(".DYNAMIC_OUTPUT(outputs, TensorType::ALL())"), std::string::npos)
      << "Must declare DYNAMIC_OUTPUT for Tensor[] return";
  EXPECT_EQ(code.find(".OUTPUT(outputs,"), std::string::npos) << "Must not use OUTPUT for variadic return";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_DynamicOutput_UsesToListAndInstanceInfo) {
  SchemaInfo schema = MakeDynamicOutputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  std::cerr << "===== Generated Execute (Dynamic Output) =====\n" << code << "\n";

  // Must extract c10::List<at::Tensor> from callBoxed result
  EXPECT_NE(code.find("toList<at::Tensor>()"), std::string::npos)
      << "Must use toList<at::Tensor>() for Tensor[] return";
  // Must NOT use toTensor() for the variadic return
  EXPECT_EQ(code.find("args[result_idx].toTensor()"), std::string::npos)
      << "Must not use toTensor() for variadic return";
  // Must query GetIrOutputInstanceInfo for instance count
  EXPECT_NE(code.find("GetIrOutputInstanceInfo(0)"), std::string::npos)
      << "Must use GetIrOutputInstanceInfo for DYNAMIC_OUTPUT count";
  // Must use flat_out_idx for MallocOutputTensor
  EXPECT_NE(code.find("MallocOutputTensor(flat_out_idx"), std::string::npos)
      << "Must use flat_out_idx for MallocOutputTensor";
  // Must have TORCH_CHECK for instance count mismatch
  EXPECT_NE(code.find("DYNAMIC_OUTPUT instance count mismatch"), std::string::npos)
      << "Must have TORCH_CHECK for instance count mismatch";
}

// ===========================================================================
// Mixed Output (-> (Tensor, Tensor[])) Codegen Test
// Schema: my_split_and_copy(Tensor input, int n) -> (Tensor, Tensor[])
// Output: [0] OUTPUT(copy), [1] DYNAMIC_OUTPUT(chunks)
// ===========================================================================

static SchemaInfo MakeMixedOutputSchema() {
  SchemaInfo schema = MakeDynamicOutputSchema();
  schema.full_op_name = "my_ops::my_split_and_copy";
  schema.op_name = "my_split_and_copy";

  // First return: regular Tensor
  SchemaArgInfo copy_ret;
  copy_ret.name = "copy";
  copy_ret.type_category = "tensor";
  schema.returns.clear();
  schema.returns.push_back(copy_ret);

  // Second return: Tensor[]
  SchemaArgInfo chunks_ret;
  chunks_ret.name = "chunks";
  chunks_ret.type_category = "tensor_list";
  chunks_ret.is_variadic = true;
  schema.returns.push_back(chunks_ret);

  return schema;
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_MixedOutput_FlatOutIdxIncrement) {
  SchemaInfo schema = MakeMixedOutputSchema();
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, false);

  std::cerr << "===== Generated Execute (Mixed Output) =====\n" << code << "\n";

  // First return (Tensor): should use toTensor() and flat_out_idx
  EXPECT_NE(code.find("args[result_idx].toTensor()"), std::string::npos) << "First return (Tensor) must use toTensor()";
  // Second return (Tensor[]): should use toList<at::Tensor>()
  EXPECT_NE(code.find("toList<at::Tensor>()"), std::string::npos)
      << "Second return (Tensor[]) must use toList<at::Tensor>()";
  // Must query GetIrOutputInstanceInfo for the DYNAMIC_OUTPUT (IR index 1)
  EXPECT_NE(code.find("GetIrOutputInstanceInfo(1)"), std::string::npos)
      << "Must use GetIrOutputInstanceInfo(1) for DYNAMIC_OUTPUT at IR index 1";
  // flat_out_idx must be declared and incremented
  EXPECT_NE(code.find("size_t flat_out_idx = 0;"), std::string::npos) << "Must declare flat_out_idx";
  // Both paths must increment flat_out_idx
  size_t incr_count = 0;
  size_t pos = 0;
  while ((pos = code.find("flat_out_idx++", pos)) != std::string::npos) {
    incr_count++;
    pos += 15;
  }
  EXPECT_GE(incr_count, 2u) << "flat_out_idx++ must appear in both output paths";
}

// ===========================================================================
// HostCpu 变体（CPU/NPU 混合调度）codegen 测试
// 验证：HostCpuExecuteOp/ShapeInferOp 基类、REG_OP_BACKEND 绑定、CreateCpuTensor、
//       无 stream guard / aclrtMemcpyAsync、InferShape 静态内联与动态透传
// ===========================================================================

static OpTarget MakeHostCpuTarget(const std::string &full_op, const std::string &overload) {
  OpTarget target;
  target.target = "torch.ops." + full_op;
  size_t pos = full_op.find("::");
  target.namespace_name = full_op.substr(0, pos);
  target.op_name = full_op.substr(pos + 2);
  target.overload = overload;
  target.device = "cpu";
  return target;
}

TEST_F(FallbackCodegenTest, ParseTargetFromJson_DeviceFieldDefaultsToNpu) {
  const std::string json = R"({
    "nodes": [
      {
        "name": "mul_tensor",
        "target": "torch.ops.aten.mul.Tensor",
        "ascend_ir": {"source": "", "ops": [{"attr": []}]}
      }
    ]
  })";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].device, "npu") << "Missing device field must default to npu (manifest v1 compat)";
}

TEST_F(FallbackCodegenTest, ParseTargetFromJson_DeviceFieldCpu) {
  const std::string json = R"({
    "nodes": [
      {
        "name": "mul_tensor",
        "target": "torch.ops.aten.mul.Tensor",
        "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]}
      }
    ]
  })";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].device, "cpu");
}

TEST_F(FallbackCodegenTest, ParseTargetFromJson_UnknownDeviceThrows) {
  const std::string json = R"({
    "nodes": [
      {
        "name": "mul_tensor",
        "target": "torch.ops.aten.mul.Tensor",
        "ascend_ir": {"source": "", "device": "gpu", "ops": [{"attr": []}]}
      }
    ]
  })";
  EXPECT_THROW({ FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError);
}

TEST_F(FallbackCodegenTest, ParseTargets_InconsistentDeviceForSameOpThrows) {
  const std::string json = R"({
    "nodes": [
      {
        "name": "relu_npu",
        "target": "torch.ops.aten.relu.default",
        "ascend_ir": {"source": "", "device": "npu", "ops": [{"attr": []}]}
      },
      {
        "name": "relu_cpu",
        "target": "torch.ops.aten.relu.default",
        "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]}
      }
    ]
  })";
  EXPECT_THROW(
      { FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError)
      << "Same op with npu+cpu ownership must be rejected (one GE backend per op type)";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_HostCpu_ContextAndTensorWrap) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::relu", "");
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, true);

  std::cerr << "===== Generated Execute (HostCpu) =====\n" << code << "\n";

  // 替换 2：上下文类型为 HostCpuOpExecutionContext
  EXPECT_NE(code.find("Execute(gert::HostCpuOpExecutionContext* ctx)"), std::string::npos)
      << "HostCpu variant must use HostCpuOpExecutionContext";
  EXPECT_EQ(code.find("EagerOpExecutionContext"), std::string::npos) << "Must not use EagerOpExecutionContext";

  // 替换 3：host 指针零拷贝包装
  EXPECT_NE(code.find("CreateCpuTensor("), std::string::npos) << "Must wrap host memory via CreateCpuTensor";
  EXPECT_EQ(code.find("CreateNpuTensor("), std::string::npos) << "Must not wrap device memory in HostCpu variant";

  // 消除项：无 stream guard、无 device 拷贝、无 task-queue drain
  EXPECT_EQ(code.find("NPUStreamGuard"), std::string::npos) << "Host execution must not use NPUStreamGuard";
  EXPECT_EQ(code.find("getStreamFromExternal"), std::string::npos) << "Host execution must not touch device streams";
  EXPECT_EQ(code.find("aclrtMemcpyAsync"), std::string::npos) << "HostCpu output write-back must be plain memcpy";
  EXPECT_EQ(code.find("npu_stream.stream(true)"), std::string::npos) << "No task-queue drain on host path";
  EXPECT_EQ(code.find("ctx->GetStream()"), std::string::npos) << "HostCpu context has no stream";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_HostCpu_ScalarReadsHostMemory) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::add", "Scalar");
  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, true);

  // 标量输入：复用 ReadScalarInput（其 host 分支直接借用 GetAddr() 视图，无 D2H 拷贝）
  EXPECT_NE(code.find("ReadScalarInput(ge_input_"), std::string::npos) << "Scalar path must read via ReadScalarInput";
  EXPECT_EQ(code.find("ACL_MEMCPY_DEVICE_TO_HOST"), std::string::npos) << "Host path must not D2H copy";
  EXPECT_EQ(code.find("aclrtMemcpy"), std::string::npos) << "Host scalar path must not call aclrtMemcpy";
}

TEST_F(FallbackCodegenTest, GenerateExecuteMethod_NonOptionalDefaultsUsePlainTypes) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::_addmm_activation", "");
  bool found_use_gelu = false;
  bool found_beta = false;
  for (const auto &arg : schema.arguments) {
    if (arg.name == "use_gelu") {
      found_use_gelu = true;
      EXPECT_TRUE(arg.has_default);
      EXPECT_FALSE(arg.is_optional) << "plain bool must not be treated as Optional[bool]";
    }
    if (arg.name == "beta") {
      found_beta = true;
      EXPECT_TRUE(arg.has_default);
      EXPECT_FALSE(arg.is_optional);
    }
  }
  EXPECT_TRUE(found_use_gelu);
  EXPECT_TRUE(found_beta);

  std::string code = FallbackCodegen::GenerateExecuteMethod(schema, true);
  EXPECT_EQ(code.find("c10::optional<bool>"), std::string::npos) << "Non-Optional bool schema arg must pass plain bool";
  EXPECT_NE(code.find("args.emplace_back(c10::Scalar(static_cast<int64_t>(1)));"), std::string::npos)
      << "Scalar default must materialize as c10::Scalar instead of None";
}

TEST_F(FallbackCodegenTest, SchemaOptionalFlagTracksSchemaType) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::native_dropout", "");
  bool found_train = false;
  for (const auto &arg : schema.arguments) {
    if (arg.name == "train") {
      found_train = true;
      EXPECT_TRUE(arg.is_optional) << "bool? train must keep Optional semantics";
    }
  }
  EXPECT_TRUE(found_train);
}

TEST_F(FallbackCodegenTest, GenerateGeOpClass_HostCpu_RegOpBackendBinding) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::relu", "");
  OpTarget target = MakeHostCpuTarget("aten::relu", "");
  ShapeMeta out_meta;
  out_meta.dtype = 7;  // pt2 serde kFloat=7（1-based，与 torch 0-based ScalarType Float=6 区分）
  out_meta.sizes = {2, 3};
  out_meta.dynamic_symbols.assign(2, "");
  target.output_metas.push_back(out_meta);
  ShapeMeta in_meta;
  in_meta.dtype = 7;
  in_meta.sizes = {2, 3};
  in_meta.dynamic_symbols.assign(2, "");
  target.input_metas.push_back(in_meta);
  target.input_arg_tensor_counts.push_back(1);
  target.has_shape_info = true;

  std::string cls = FallbackCodegen::GenerateGeOpClass(schema, true, &target);

  std::cerr << "===== Generated Class (HostCpu) =====\n" << cls << "\n";

  // 替换 1：基类为 HostCpuExecuteOp + ShapeInferOp
  EXPECT_NE(cls.find("class aten__relu final : public HostCpuExecuteOp, public ShapeInferOp"), std::string::npos)
      << "HostCpu class must inherit HostCpuExecuteOp and ShapeInferOp";
  // 绑定宏：REG_OP_BACKEND(..., ge::OpBackend::kHostCPU)
  EXPECT_NE(cls.find("REG_OP_BACKEND(aten__relu, \"aten__relu\", ge::OpBackend::kHostCPU);"), std::string::npos)
      << "HostCpu variant must register via REG_OP_BACKEND with kHostCPU";
  EXPECT_EQ(cls.find("REG_AUTO_MAPPING_OP"), std::string::npos) << "Must not use REG_AUTO_MAPPING_OP for HostCpu";
  // ShapeInferOp 两个接口必须实现
  EXPECT_NE(cls.find("InferShape(gert::InferShapeContext* ctx) override"), std::string::npos);
  EXPECT_NE(cls.find("InferDataType(gert::InferDataTypeContext* ctx) override"), std::string::npos);
  // DEBUG 级节点名日志 + 异常路径 ERROR 日志并返回 GRAPH_FAILED
  EXPECT_NE(cls.find("aclAppLog(ACL_DEBUG, \"HostCpuExecute\""), std::string::npos)
      << "HostCpu Execute must emit a DEBUG log with the op name";
  EXPECT_NE(cls.find("aclAppLog(ACL_ERROR, \"HostCpuExecute\""), std::string::npos)
      << "HostCpu Execute must log ERROR on the exception path";
  EXPECT_NE(cls.find("return ge::GRAPH_FAILED;"), std::string::npos)
      << "HostCpu Execute must return GRAPH_FAILED on exceptions";
  EXPECT_NE(cls.find("try {"), std::string::npos);
  // 静态 shape 内联
  EXPECT_NE(cls.find("out_shape->SetDim(0, 2);"), std::string::npos) << "Static dims must be inlined";
  EXPECT_NE(cls.find("out_shape->SetDim(1, 3);"), std::string::npos) << "Static dims must be inlined";
  // dtype 内联（pt2 serde kFloat=7 → ge::DT_FLOAT；注意与 torch 0-based ScalarType Float=6 区分）
  EXPECT_NE(cls.find("ctx->SetOutputDataType(0, ge::DT_FLOAT);"), std::string::npos);
}

TEST_F(FallbackCodegenTest, GenerateGeOpClass_HostCpu_MissingShapeInfoThrows) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::relu", "");
  OpTarget target = MakeHostCpuTarget("aten::relu", "");
  target.has_shape_info = false;
  EXPECT_THROW(
      { FallbackCodegen::GenerateGeOpClass(schema, true, &target); }, FallbackCodegenError)
      << "ShapeInferOp is mandatory for HostCpu backend";
}

TEST_F(FallbackCodegenTest, GenerateShapeInfer_DynamicDimPassthroughFromInput) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::relu", "");
  OpTarget target = MakeHostCpuTarget("aten::relu", "");
  ShapeMeta in_meta;
  in_meta.dtype = 7;
  in_meta.sizes = {2, 3};
  in_meta.dynamic_symbols = {"", "s0"};
  target.input_metas.push_back(in_meta);
  ShapeMeta out_meta;
  out_meta.dtype = 7;
  out_meta.sizes = {2, -1};
  out_meta.dynamic_symbols = {"", "s0"};
  target.output_metas.push_back(out_meta);
  target.input_arg_tensor_counts.push_back(1);
  target.has_shape_info = true;

  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, target);

  std::cerr << "===== Generated InferShape (dynamic passthrough) =====\n" << code << "\n";

  // 静态维内联 + 动态维从输入 0 的 dim 1 透传
  EXPECT_NE(code.find("out_shape->SetDim(0, 2);"), std::string::npos);
  EXPECT_NE(code.find("ctx->GetRequiredInputShape(0)->GetDim(1)"), std::string::npos)
      << "Dynamic dim must passthrough from the matching input dim";
}

TEST_F(FallbackCodegenTest, GenerateShapeInfer_DataDependentDimThrows) {
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("aten::relu", "");
  OpTarget target = MakeHostCpuTarget("aten::relu", "");
  ShapeMeta in_meta;
  in_meta.dtype = 7;
  in_meta.sizes = {2};
  in_meta.dynamic_symbols = {""};
  target.input_metas.push_back(in_meta);
  ShapeMeta out_meta;
  out_meta.dtype = 7;
  out_meta.sizes = {-1};
  out_meta.dynamic_symbols = {"s9"};
  target.output_metas.push_back(out_meta);
  target.input_arg_tensor_counts.push_back(1);
  target.has_shape_info = true;

  EXPECT_THROW(
      { FallbackCodegen::GenerateShapeInferMethods(schema, target); }, FallbackCodegenError)
      << "Data-dependent output dim must fail codegen with a clear error";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuShapeConflictThrows) {
  const std::string json = R"({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "relu_1",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "outputs": [{"as_tensor": {"name": "relu_1_out"}}]
          },
          {
            "name": "relu_2",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "outputs": [{"as_tensor": {"name": "relu_2_out"}}]
          }
        ],
        "tensor_values": {
          "relu_1_out": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 3}]},
          "relu_2_out": {"dtype": 7, "sizes": [{"as_int": 5}, {"as_int": 5}]}
        }
      }
    }
  })";
  EXPECT_THROW(
      { FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError)
      << "CPU op instances with inconsistent static shapes must be rejected";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuShapeConsistentDedup) {
  const std::string json = R"({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "relu_1",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "outputs": [{"as_tensor": {"name": "relu_1_out"}}]
          },
          {
            "name": "relu_2",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "outputs": [{"as_tensor": {"name": "relu_2_out"}}]
          }
        ],
        "tensor_values": {
          "relu_1_out": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 3}]},
          "relu_2_out": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 3}]}
        }
      }
    }
  })";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u) << "Consistent CPU instances dedup to one generated op";
  EXPECT_EQ(targets[0].device, "cpu");
  ASSERT_EQ(targets[0].output_metas.size(), 1u);
  EXPECT_EQ(targets[0].output_metas[0].sizes, (std::vector<int64_t>{3, 3}));
}

TEST_F(FallbackCodegenTest, GenerateHelperFunctions_HostCpu_HasCreateCpuTensor) {
  std::string code = FallbackCodegen::GenerateHelperFunctions(true);
  EXPECT_NE(code.find("at::Tensor CreateCpuTensor(const gert::Tensor* ge_tensor)"), std::string::npos)
      << "HostCpu helper must define CreateCpuTensor";
  EXPECT_NE(code.find("device(at::kCPU)"), std::string::npos) << "CreateCpuTensor must wrap at::kCPU memory";
  // CreateCpuTensor 段内不得引用 PrivateUse1（CreateNpuTensor 作为 NPU 变体助手仍会生成）
  size_t cpu_fn_start = code.find("at::Tensor CreateCpuTensor(");
  size_t cpu_fn_end = code.find("\n}\n", cpu_fn_start);
  ASSERT_NE(cpu_fn_end, std::string::npos);
  EXPECT_EQ(code.find("kPrivateUse1", cpu_fn_start), std::string::npos)
      << "CreateCpuTensor body must not wrap PrivateUse1 memory";
}

// ===========================================================================
// codegen 阶段 CPU dispatch key 预检
// ===========================================================================

#include <torch/library.h>

// 只注册 schema、不注册任何 kernel 的算子：findSchema 成功但无 CPU kernel。
TORCH_LIBRARY(td_precheck, m) {
  m.def("no_cpu_kernel_op(Tensor a) -> Tensor");
}

TEST_F(FallbackCodegenTest, GenerateFromTargets_CpuKernelPrecheckFailsEarly) {
  const std::string json = R"({
    "nodes": [
      {
        "name": "bad_cpu_op",
        "target": "torch.ops.td_precheck.no_cpu_kernel_op.default",
        "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]}
      }
    ]
  })";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0].device, "cpu");

  // 预检在临时目录/g++ 之前发生：无论编译环境如何都应立刻失败且带算子名
  // （GenerateFromTargets 捕获异常后通过 result.error_message 返回）。
  const CodegenResult result =
      FallbackCodegen::GenerateFromTargets(targets, "/nonexistent/torch", "/nonexistent/torch_npu", "/tmp");
  EXPECT_FALSE(result.success) << "CPU dispatch key pre-check must fail for kernel-less op";
  EXPECT_NE(result.error_message.find("no CPU kernel registered"), std::string::npos) << result.error_message;
  EXPECT_NE(result.error_message.find("td_precheck::no_cpu_kernel_op"), std::string::npos) << result.error_message;
}

TEST_F(FallbackCodegenTest, GenerateFromTargets_NpuTargetSkipsCpuPrecheck) {
  // NPU 目标不做 CPU kernel 预检（无 CPU kernel 的算子可走 PrivateUse1 回调）。
  const std::string json = R"({
    "nodes": [
      {
        "name": "bad_cpu_op",
        "target": "torch.ops.td_precheck.no_cpu_kernel_op.default",
        "ascend_ir": {"source": "", "device": "npu", "ops": [{"attr": []}]}
      }
    ]
  })";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  // 编译环境不存在必然失败，但失败原因不应是 CPU kernel 预检。
  const CodegenResult result =
      FallbackCodegen::GenerateFromTargets(targets, "/nonexistent/torch", "/nonexistent/torch_npu", "/tmp");
  EXPECT_FALSE(result.success) << "expected compile-env failure";
  EXPECT_EQ(result.error_message.find("no CPU kernel registered"), std::string::npos)
      << "NPU targets must not run the CPU kernel pre-check: " << result.error_message;
}

// ===========================================================================
// 评审回归：复合 SymInt 严格解析 / count 游标对齐 / 变长列表相对序号 /
// 结构签名去重
// ===========================================================================

// 仅注册 schema 的测试算子：覆盖“标量参数夹在张量之间 / 变长列表 / 双张量 /
// functionalized 内层三张量”签名。
TORCH_LIBRARY(td_shapeinfer, m) {
  m.def("scalar_mid(Tensor a, int k, Tensor b) -> Tensor");
  m.def("list_second(Tensor[] xs, int k) -> Tensor");
  m.def("pair(Tensor a, Tensor b) -> Tensor");
  m.def("trio(Tensor self, Tensor sin_cache, Tensor cos_cache) -> Tensor");
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuCompoundSymIntExprOutputThrows) {
  // 输出维为复合 SymInt 表达式（s0+1）：按 s0 透传会静默生成错误 InferShape，
  // 按 -1 内联会生成非法静态维，必须显式报错。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "pair_1",
            "target": "torch.ops.td_shapeinfer.pair.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "z": {"dtype": 7, "sizes": [{"as_int": 4}]},
          "p_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0') + 1"}}]}
        }
      }
    }
  })json";
  EXPECT_THROW(
      { FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError)
      << "Compound SymInt expr on output dim must fail codegen (data-dependent shape)";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuSymbolWithAssumptionsAccepted) {
  // torch >= 2.9 实际序列化的动态维为 Symbol('s77', positive=True, integer=True)：
  // assumptions 只描述符号取值域，携带参数仍是纯符号维，必须与 Symbol('s0')
  // 一样按符号透传；只有复合表达式才报错。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "pair_1",
            "target": "torch.ops.td_shapeinfer.pair.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s77', positive=True, integer=True)"}}]},
          "z": {"dtype": 7, "sizes": [{"as_int": 4}]},
          "p_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s77', positive=True, integer=True)"}}]}
        }
      }
    }
  })json";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  ASSERT_EQ(targets[0].input_metas.size(), 2u);
  ASSERT_EQ(targets[0].input_metas[0].dynamic_symbols.size(), 1u);
  EXPECT_EQ(targets[0].input_metas[0].dynamic_symbols[0], "s77");

  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("td_shapeinfer::pair", "");
  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, targets[0]);
  // 输出维 s77 从输入 a（IR 序号 0）透传，而不是报“不可推导”错误。
  EXPECT_NE(code.find("ctx->GetRequiredInputShape(0)->GetDim(0)"), std::string::npos) << code;
  EXPECT_EQ(code.find("not derivable from inputs"), std::string::npos) << code;
}

TEST_F(FallbackCodegenTest, ParseTargets_HopNodeReordersFunctionalizedInputs) {
  // auto_functionalized_v2 节点的原始 inputs 顺序为
  // [op, sin_cache, cos_cache, _input_base_index, _all_bases]，而内层算子
  // trio 的 schema 顺序为 (self, sin_cache, cos_cache)。CollectShapeInfo 必须按
  // parse_functionalized.cc 的运行时归一化重排（self 类基础张量排最前），
  // 否则输出维 s0 会绑到 IR 序号 2（cos_cache）而非 0（self）。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "hop_trio",
            "target": "torch.ops.higher_order.auto_functionalized_v2",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "op", "arg": {"as_operator": "torch.ops.td_shapeinfer.trio.default"}},
              {"name": "sin_cache", "arg": {"as_tensor": {"name": "sin"}}},
              {"name": "cos_cache", "arg": {"as_tensor": {"name": "cos"}}},
              {"name": "_input_base_index", "arg": {"as_int": 0}},
              {"name": "_all_bases", "arg": {"as_tensors": [{"name": "hidden_states"}]}}
            ],
            "outputs": [{"as_tensor": {"name": "trio_out"}}]
          }
        ],
        "tensor_values": {
          "hidden_states": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "sin": {"dtype": 7, "sizes": [{"as_int": 8}]},
          "cos": {"dtype": 7, "sizes": [{"as_int": 8}]},
          "trio_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]}
        }
      }
    }
  })json";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  ASSERT_EQ(targets[0].target, "torch.ops.td_shapeinfer.trio.default");
  // 归一化后的输入 meta 顺序：(self=hidden_states, sin, cos)。
  ASSERT_EQ(targets[0].input_metas.size(), 3u);
  ASSERT_EQ(targets[0].input_arg_tensor_counts.size(), 3u);
  ASSERT_EQ(targets[0].input_metas[0].dynamic_symbols.size(), 1u);
  EXPECT_EQ(targets[0].input_metas[0].dynamic_symbols[0], "s0") << "self must be the first input after reordering";

  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("td_shapeinfer::trio", "");
  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, targets[0]);
  EXPECT_NE(code.find("ctx->GetRequiredInputShape(0)->GetDim(0)"), std::string::npos)
      << "symbol s0 must passthrough from the reordered self input (IR index 0)";
  EXPECT_EQ(code.find("ctx->GetRequiredInputShape(2)->GetDim(0)"), std::string::npos)
      << "must not bind s0 to cos_cache (raw input order would do that)";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuArithmeticExprOutputThrows) {
  // expr_str 为纯算术 repr（不含 Symbol(' 子串）：旧实现会把 -1 当静态维内联，
  // 必须显式报错。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "pair_1",
            "target": "torch.ops.td_shapeinfer.pair.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "z": {"dtype": 7, "sizes": [{"as_int": 4}]},
          "p_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "2*Symbol('s0')"}}]}
        }
      }
    }
  })json";
  EXPECT_THROW(
      { FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError)
      << "Arithmetic (non pure-symbol) SymInt expr must fail codegen";
}

TEST_F(FallbackCodegenTest, GenerateShapeInfer_ScalarBetweenTensors_AlignsCounts) {
  // schema: scalar_mid(Tensor a, int k, Tensor b)。input_arg_tensor_counts 按
  // 节点 inputs 建（[1,0,1]，标量条目为 0），count_cursor 必须逐参数推进；
  // 只在 tensor 参数上推进会令 b 读到标量条目 count=0，meta 永不绑定。
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("td_shapeinfer::scalar_mid", "");
  OpTarget target = MakeHostCpuTarget("td_shapeinfer::scalar_mid", "");
  ShapeMeta meta_a;
  meta_a.dtype = 7;
  meta_a.sizes = {2};
  meta_a.dynamic_symbols = {""};
  ShapeMeta meta_b;
  meta_b.dtype = 7;
  meta_b.sizes = {-1};
  meta_b.dynamic_symbols = {"s0"};
  target.input_metas = {meta_a, meta_b};
  ShapeMeta out_meta;
  out_meta.dtype = 7;
  out_meta.sizes = {-1};
  out_meta.dynamic_symbols = {"s0"};
  target.output_metas = {out_meta};
  target.input_arg_tensor_counts = {1, 0, 1};
  target.has_shape_info = true;

  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, target);
  // b 的 IR 序号为 2（a=0、k=1、b=2），其 dim0 符号 s0 应透传自 b 自身
  EXPECT_NE(code.find("ctx->GetRequiredInputShape(2)->GetDim(0)"), std::string::npos)
      << "tensor arg after a scalar must bind its own meta (cursor advances per schema arg)";
}

TEST_F(FallbackCodegenTest, GenerateShapeInfer_DynamicList_SecondTensorPassthrough) {
  // 符号来自变长列表第 2 个张量：GetDynamicInputShape 第二参必须是相对序号 1，
  // 不能硬编码 0（那会读第 1 个张量的 shape）。
  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("td_shapeinfer::list_second", "");
  OpTarget target = MakeHostCpuTarget("td_shapeinfer::list_second", "");
  ShapeMeta t1;
  t1.dtype = 7;
  t1.sizes = {3};
  t1.dynamic_symbols = {""};
  ShapeMeta t2;
  t2.dtype = 7;
  t2.sizes = {-1};
  t2.dynamic_symbols = {"s0"};
  target.input_metas = {t1, t2};
  ShapeMeta out_meta;
  out_meta.dtype = 7;
  out_meta.sizes = {-1};
  out_meta.dynamic_symbols = {"s0"};
  target.output_metas = {out_meta};
  target.input_arg_tensor_counts = {2, 0};
  target.has_shape_info = true;

  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, target);
  EXPECT_NE(code.find("ctx->GetDynamicInputShape(0, 1)->GetDim(0)"), std::string::npos)
      << "symbol from the 2nd variadic tensor must read relative index 1";
  EXPECT_EQ(code.find("GetDynamicInputShape(0, 0)"), std::string::npos) << "must not hardcode relative_index=0";
}

TEST_F(FallbackCodegenTest, ParseTargets_HopNode_ShapeInfoSkipsOperatorEntry) {
  // auto_functionalized_v2 节点 inputs[0] 是 as_operator 占位（0 张量），
  // CollectShapeInfo 收集输入时剔除占位，内层算子的张量参数从收集表 0 起
  // 与 schema 参数对齐。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "hop_pair",
            "target": "torch.ops.higher_order.auto_functionalized_v2",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "", "arg": {"as_operator": "torch.ops.td_shapeinfer.pair.default"}},
              {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_int": 2}]},
          "z": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "p_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]}
        }
      }
    }
  })json";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u);
  ASSERT_EQ(targets[0].input_metas.size(), 2u) << "as_operator placeholder must not appear in the input table";

  SchemaInfo schema = FallbackCodegen::GetSchemaInfo("td_shapeinfer::pair", "");
  std::string code = FallbackCodegen::GenerateShapeInferMethods(schema, targets[0]);
  // 输出符号 s0 来自 z（b 参数，IR 序号 1）：剔除占位后 b 绑定自身 meta；
  // 若未剔除，b 会错绑 a 的 meta（无符号）→ 误报数据依赖维。
  EXPECT_NE(code.find("ctx->GetRequiredInputShape(1)->GetDim(0)"), std::string::npos)
      << "inner-op tensor args must align with the input table after dropping the as_operator entry";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuSymbolRenameSameStructureDedupOk) {
  // 两个 relu 实例输出分别为 [s0] 与 [s1]，结构完全等价（均透传 in0.dim0）。
  // 按符号名比较会误报 "inconsistent output shapes"（拒合法模型）；
  // 结构签名比较应放行并去重。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "relu_1",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}}],
            "outputs": [{"as_tensor": {"name": "r1_out"}}]
          },
          {
            "name": "relu_2",
            "target": "torch.ops.aten.relu.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "z"}}}],
            "outputs": [{"as_tensor": {"name": "r2_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "r1_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "z": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s1')"}}]},
          "r2_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s1')"}}]}
        }
      }
    }
  })json";
  auto targets = FallbackCodegen::ParseModelJsonFromString(json);
  ASSERT_EQ(targets.size(), 1u) << "structurally equivalent instances (renamed symbols) must dedup";
  EXPECT_EQ(targets[0].output_metas[0].dynamic_symbols, std::vector<std::string>({"s0"}))
      << "representative instance keeps its own symbols";
}

TEST_F(FallbackCodegenTest, ParseTargets_CpuSameSymbolDifferentSourceThrows) {
  // 两实例共享符号名 s0 但解析路径不同（out ← self.dim0 vs out ← other.dim0）。
  // 按符号名比较签名相同（都是 ?s0）→ 静默按代表实例透传，实例 2 运行时取错维；
  // 结构签名必须拒绝。
  const std::string json = R"json({
    "graph_module": {
      "graph": {
        "nodes": [
          {
            "name": "pair_1",
            "target": "torch.ops.td_shapeinfer.pair.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p1_out"}}]
          },
          {
            "name": "pair_2",
            "target": "torch.ops.td_shapeinfer.pair.default",
            "ascend_ir": {"source": "", "device": "cpu", "ops": [{"attr": []}]},
            "inputs": [
              {"name": "a", "arg": {"as_tensor": {"name": "x2"}}},
              {"name": "b", "arg": {"as_tensor": {"name": "z2"}}}
            ],
            "outputs": [{"as_tensor": {"name": "p2_out"}}]
          }
        ],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "z": {"dtype": 7, "sizes": [{"as_int": 4}]},
          "p1_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "x2": {"dtype": 7, "sizes": [{"as_int": 4}]},
          "z2": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]},
          "p2_out": {"dtype": 7, "sizes": [{"as_expr": {"expr_str": "Symbol('s0')"}}]}
        }
      }
    }
  })json";
  EXPECT_THROW(
      { FallbackCodegen::ParseModelJsonFromString(json); }, FallbackCodegenError)
      << "same symbol name but different passthrough source must be rejected";
}
