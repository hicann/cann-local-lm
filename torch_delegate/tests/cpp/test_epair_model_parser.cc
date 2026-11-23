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

#include <string>

#include "backend/parsed_model.h"
#include "backend/parse_ascend_ir.h"
#include "common/abi_compat_utils.h"
#include "common/epair_archive_constants.h"
#include "common/logging/logging.h"
#include "backend/epair_archive.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kMissingGraphSource = std::string(FIXTURES_DIR) + "/missing_graph_source.epair";
const std::string kGraphSource = td::epair_spec::kAnnotatedModelJson;

}  // namespace

namespace td {
class EpairModelParserTest : public ::testing::Test {};

TEST_F(EpairModelParserTest, ParsesAnnotatedFixture) {
  EpairArchive archive(kAnnotated.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  EXPECT_EQ(model.SchemaVersion().major, 8);
  EXPECT_EQ(model.SchemaVersion().minor, 7);
  ASSERT_EQ(model.GraphInputs().size(), 3u);
  ASSERT_EQ(model.GraphOutputs().size(), 1u);
  ASSERT_EQ(model.Nodes().size(), 2u);
  EXPECT_EQ(model.LogicalNameOf("p_weight"), "weight");
  EXPECT_EQ(model.PlaceholderKindOf("b_bias"), td::PlaceholderKind::Buffer);
  ASSERT_NE(model.FindTensorMeta("mm"), nullptr);
  const td::ParsedNode *mm_node = model.FindNodeByOutput("mm");
  ASSERT_NE(mm_node, nullptr);
  EXPECT_EQ(mm_node->json_node_index, 0u);
  EXPECT_EQ(mm_node->name, "node_0_aten_mm_default");
  ASSERT_NE(model.FindNodeByName(mm_node->name), nullptr);
  ASSERT_TRUE(mm_node->ascend_ir.has_value());
  ASSERT_EQ(mm_node->fx_input_tensor_names.size(), 2u);
  ASSERT_EQ(mm_node->fx_output_tensor_names.size(), 1u);
  ASSERT_TRUE(mm_node->ascend_ir->input_mappings.empty());
  ASSERT_EQ(mm_node->ascend_ir->output_mappings.size(), 1u);
  EXPECT_EQ(mm_node->ascend_ir->output_mappings.at("mm"), "MatMul:0");
  ASSERT_EQ(mm_node->ascend_ir->boundary_outputs.size(), 1u);
  EXPECT_EQ(mm_node->ascend_ir->boundary_outputs[0].fx_tensor_name, "mm");
  EXPECT_EQ(mm_node->ascend_ir->boundary_outputs[0].ge_tensor_name, "MatMul:0");
  EXPECT_EQ(mm_node->ascend_ir->ops[0].type, "MatMul");
}

TEST_F(EpairModelParserTest, MissingGraphSourceDetectedByArchive) {
  EpairArchive archive(kMissingGraphSource.c_str());
  EXPECT_FALSE(archive.HasRecord(kGraphSource.c_str()));
}

TEST_F(EpairModelParserTest, IgnoresAsNoneOutputForAutoFunctionalizedNode) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [
            {"as_tensor": {"name": "hidden_states"}},
            {"as_tensor": {"name": "sin_cache"}},
            {"as_tensor": {"name": "cos_cache"}}
          ],
          "outputs": [{"as_tensor": {"name": "updated_x"}}],
          "nodes": [{
            "target": "torch.ops.higher_order.auto_functionalized_v2",
            "inputs": [
              {"name": "op", "arg": {"as_operator": "torch.ops.xllm_ascend.rope_in_place.default"}, "kind": 1},
              {"name": "sin_cache", "arg": {"as_tensor": {"name": "sin_cache"}}, "kind": 2},
              {"name": "cos_cache", "arg": {"as_tensor": {"name": "cos_cache"}}, "kind": 2},
              {"name": "_self_base_index", "arg": {"as_int": 0}, "kind": 2},
              {"name": "_all_bases", "arg": {"as_tensors": [{"name": "hidden_states"}]}, "kind": 2}
            ],
            "outputs": [
              {"as_none": true},
              {"as_tensor": {"name": "updated_x"}}
            ],
            "ascend_ir": {"source": "", "ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "hidden_states": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "sin_cache": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "cos_cache": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "updated_x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [
            {"user_input": {"arg": {"as_tensor": {"name": "hidden_states"}}}},
            {"user_input": {"arg": {"as_tensor": {"name": "sin_cache"}}}},
            {"user_input": {"arg": {"as_tensor": {"name": "cos_cache"}}}}
          ],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "updated_x"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 1u);
  EXPECT_EQ(model.Nodes()[0].effective_schema_target, "torch.ops.xllm_ascend.rope_in_place.default");
  EXPECT_EQ(model.Nodes()[0].name, "RopeInPlace");
  EXPECT_TRUE(model.Nodes()[0].functionalize.is_functionalized);
  EXPECT_EQ(model.Nodes()[0].functionalize.wrapped_target, "torch.ops.xllm_ascend.rope_in_place.default");
  ASSERT_EQ(model.Nodes()[0].inputs.size(), 3u);
  EXPECT_EQ(model.Nodes()[0].inputs[0].name, "self");
  EXPECT_EQ(model.Nodes()[0].inputs[0].tensor_name, "hidden_states");
  EXPECT_EQ(model.Nodes()[0].inputs[1].tensor_name, "sin_cache");
  EXPECT_EQ(model.Nodes()[0].inputs[2].tensor_name, "cos_cache");
  ASSERT_EQ(model.Nodes()[0].outputs.size(), 1u);
  EXPECT_EQ(model.Nodes()[0].outputs[0].tensor_name, "updated_x");
  ASSERT_EQ(model.Nodes()[0].functionalize.runtime_input_tensor_names.size(), 3u);
  EXPECT_EQ(model.Nodes()[0].functionalize.runtime_output_tensor_names[0], "updated_x");
}

TEST_F(EpairModelParserTest, FunctionalizedBaseInputUsesSchemaArgName) {
  // 回归 0012：inplace 基础张量必须按 schema 参数名（这里是 a）命名，
  // 而不是固定的 self；否则按名对齐 IR 槽位时匹配不到 a 而构图失败。
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [{"as_tensor": {"name": "x"}}],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "nodes": [{
            "target": "torch.ops.higher_order.auto_functionalized_v2",
            "inputs": [
              {"name": "op", "arg": {"as_operator": "torch.ops.my_ops.inplace_scale_out.default"}, "kind": 1},
              {"name": "scale", "arg": {"as_float": 2.0}, "kind": 2},
              {"name": "_a_base_index", "arg": {"as_int": 0}, "kind": 2},
              {"name": "_all_bases", "arg": {"as_tensors": [{"name": "x"}]}, "kind": 2}
            ],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"source": "", "ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [4, 10], "requires_grad": false, "strides": [10, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [4, 10], "requires_grad": false, "strides": [10, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 1u);
  ASSERT_EQ(model.Nodes()[0].inputs.size(), 2u);
  EXPECT_EQ(model.Nodes()[0].inputs[0].name, "a");
  EXPECT_EQ(model.Nodes()[0].inputs[0].tensor_name, "x");
  EXPECT_EQ(model.Nodes()[0].inputs[1].name, "scale");
}

TEST_F(EpairModelParserTest, RejectsAutoFunctionalizedV1Node) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [{"as_tensor": {"name": "x"}}],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "nodes": [{
            "target": "torch.ops.higher_order.auto_functionalized",
            "inputs": [
              {"name": "op", "arg": {"as_operator": "torch.ops.xllm_ascend.rope_in_place.default"}, "kind": 1},
              {"name": "_all_bases", "arg": {"as_tensors": [{"name": "x"}]}, "kind": 2},
              {"name": "_input_base_index", "arg": {"as_int": 0}, "kind": 2}
            ],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"source": "", "ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  EXPECT_NE(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
}

TEST_F(EpairModelParserTest, ParsesTensorListFallbackOutputs) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [
            {"as_tensor": {"name": "x0"}},
            {"as_tensor": {"name": "x1"}}
          ],
          "outputs": [
            {"as_tensor": {"name": "y0"}},
            {"as_tensor": {"name": "y1"}}
          ],
          "nodes": [{
            "target": "torch.ops.my_ns.list_identity.default",
            "inputs": [
              {"name": "x", "arg": {"as_tensors": [{"name": "x0"}, {"name": "x1"}]}, "kind": 1}
            ],
            "outputs": [
              {"as_tensors": [{"name": "y0"}, {"name": "y1"}]}
            ],
            "ascend_ir": {"source": "", "ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x0": {"dtype": 0, "sizes": [2, 2], "requires_grad": false, "strides": [2, 1], "storage_offset": 0, "layout": 7},
            "x1": {"dtype": 0, "sizes": [2, 2], "requires_grad": false, "strides": [2, 1], "storage_offset": 0, "layout": 7},
            "y0": {"dtype": 0, "sizes": [2, 2], "requires_grad": false, "strides": [2, 1], "storage_offset": 0, "layout": 7},
            "y1": {"dtype": 0, "sizes": [2, 2], "requires_grad": false, "strides": [2, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [
            {"user_input": {"arg": {"as_tensor": {"name": "x0"}}}},
            {"user_input": {"arg": {"as_tensor": {"name": "x1"}}}}
          ],
          "output_specs": [
            {"user_output": {"arg": {"as_tensor": {"name": "y0"}}}},
            {"user_output": {"arg": {"as_tensor": {"name": "y1"}}}}
          ]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 1u);
  ASSERT_EQ(model.Nodes()[0].outputs.size(), 1u);
  EXPECT_TRUE(model.Nodes()[0].outputs[0].is_tensor_list);
  ASSERT_EQ(model.Nodes()[0].outputs[0].tensor_names.size(), 2u);
  EXPECT_EQ(model.Nodes()[0].outputs[0].tensor_names[0], "y0");
  EXPECT_EQ(model.Nodes()[0].outputs[0].tensor_names[1], "y1");
  ASSERT_NE(model.FindNodeByOutput("y0"), nullptr);
  ASSERT_NE(model.FindNodeByOutput("y1"), nullptr);
}

TEST_F(EpairModelParserTest, PreservesConverterNoOpAndLegacyFallbackSource) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [{"as_tensor": {"name": "x"}}],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "nodes": [
            {
              "target": "torch.ops.aten.expand.default",
              "inputs": [
                {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                {"name": "size", "arg": {"as_ints": [3, 3]}, "kind": 1}
              ],
              "outputs": [{"as_tensor": {"name": "expand"}}],
              "ascend_ir": {"source": "converter", "ops": [], "mapping": null}
            },
            {
              "target": "torch.ops.aten.relu.default",
              "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "expand"}}, "kind": 1}],
              "outputs": [{"as_tensor": {"name": "y"}}],
              "ascend_ir": {"ops": [{"attr": {}}]}
            }
          ],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "expand": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 2u);
  ASSERT_TRUE(model.Nodes()[0].ascend_ir.has_value());
  EXPECT_EQ(model.Nodes()[0].ascend_ir->source, "converter");
  EXPECT_TRUE(model.Nodes()[0].source_info.is_converter);
  EXPECT_TRUE(model.Nodes()[0].ascend_ir->ops.empty());
  ASSERT_TRUE(model.Nodes()[1].ascend_ir.has_value());
  EXPECT_TRUE(model.Nodes()[1].ascend_ir->source.empty());
  EXPECT_TRUE(model.Nodes()[1].source_info.is_fallback);
}

TEST_F(EpairModelParserTest, AssignsStableFallbackGeNamesAndPreservesDummyMappings) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [
            {"as_tensor": {"name": "hidden_states"}},
            {"as_tensor": {"name": "sin_cache"}},
            {"as_tensor": {"name": "cos_cache"}}
          ],
          "outputs": [{"as_tensor": {"name": "view_1"}}],
          "nodes": [
            {
              "target": "torch.ops.xllm_ascend.rope_in_place.default",
              "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "hidden_states"}}, "kind": 1},
                {"name": "sin_cache", "arg": {"as_tensor": {"name": "sin_cache"}}, "kind": 1},
                {"name": "cos_cache", "arg": {"as_tensor": {"name": "cos_cache"}}, "kind": 1}
              ],
              "outputs": [{"as_tensor": {"name": "rope_out_0"}}],
              "ascend_ir": {
                 "ops": [{"attr": {}}],
                 "mapping": {"inputs": {}, "outputs": {"rope_out_0": "rope_in_place_ge_op:0"}}
               }
            },
            {
              "target": "torch.ops.aten.reshape.default",
              "inputs": [
                {"name": "self", "arg": {"as_tensor": {"name": "rope_out_0"}}, "kind": 1},
                {"name": "shape", "arg": {"as_ints": [256, 128]}, "kind": 1}
              ],
              "outputs": [{"as_tensor": {"name": "view_0"}}],
              "ascend_ir": {
                "source": "converter",
                 "ops": [{
                   "name": "Reshape",
                   "type": "Reshape",
                   "input": ["rope_out_0"]
                 }],
                 "mapping": {"inputs": {"rope_out_0": "rope_in_place_ge_op:0"}, "outputs": {"view_0": "Reshape:0"}}
               }
            },
            {
              "target": "torch.ops.xllm_ascend.rope_in_place.default",
              "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "view_0"}}, "kind": 1},
                {"name": "sin_cache", "arg": {"as_tensor": {"name": "sin_cache"}}, "kind": 1},
                {"name": "cos_cache", "arg": {"as_tensor": {"name": "cos_cache"}}, "kind": 1}
              ],
              "outputs": [{"as_tensor": {"name": "rope_out_1"}}],
              "ascend_ir": {
                 "ops": [{"attr": {}}],
                 "mapping": {"inputs": {}, "outputs": {"rope_out_1": "rope_in_place_ge_op_1:0"}}
               }
            },
            {
              "target": "torch.ops.aten.reshape.default",
              "inputs": [
                {"name": "self", "arg": {"as_tensor": {"name": "rope_out_1"}}, "kind": 1},
                {"name": "shape", "arg": {"as_ints": [128, 256]}, "kind": 1}
              ],
              "outputs": [{"as_tensor": {"name": "view_1"}}],
              "ascend_ir": {
                "source": "converter",
                 "ops": [{
                   "name": "Reshape_1",
                   "type": "Reshape",
                   "input": ["rope_out_1"]
                 }],
                 "mapping": {"inputs": {"rope_out_1": "rope_in_place_ge_op_1:0"}, "outputs": {"view_1": "Reshape_1:0"}}
               }
            }
          ],
          "tensor_values": {
            "hidden_states": {"dtype": 0, "sizes": [128, 2, 128], "requires_grad": false, "strides": [256, 128, 1], "storage_offset": 0, "layout": 7},
            "sin_cache": {"dtype": 0, "sizes": [128, 2, 128], "requires_grad": false, "strides": [256, 128, 1], "storage_offset": 0, "layout": 7},
            "cos_cache": {"dtype": 0, "sizes": [128, 2, 128], "requires_grad": false, "strides": [256, 128, 1], "storage_offset": 0, "layout": 7},
            "rope_out_0": {"dtype": 0, "sizes": [128, 2, 128], "requires_grad": false, "strides": [256, 128, 1], "storage_offset": 0, "layout": 7},
            "view_0": {"dtype": 0, "sizes": [256, 128], "requires_grad": false, "strides": [128, 1], "storage_offset": 0, "layout": 7},
            "rope_out_1": {"dtype": 0, "sizes": [256, 128], "requires_grad": false, "strides": [128, 1], "storage_offset": 0, "layout": 7},
            "view_1": {"dtype": 0, "sizes": [128, 256], "requires_grad": false, "strides": [256, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [
            {"user_input": {"arg": {"as_tensor": {"name": "hidden_states"}}}},
            {"user_input": {"arg": {"as_tensor": {"name": "sin_cache"}}}},
            {"user_input": {"arg": {"as_tensor": {"name": "cos_cache"}}}}
          ],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "view_1"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 4u);
  EXPECT_EQ(model.Nodes()[0].name, "RopeInPlace");
  EXPECT_EQ(model.Nodes()[2].name, "RopeInPlace_1");
  ASSERT_TRUE(model.Nodes()[0].ascend_ir.has_value());
  ASSERT_TRUE(model.Nodes()[2].ascend_ir.has_value());
  ASSERT_EQ(model.Nodes()[0].ascend_ir->output_mappings.size(), 1u);
  ASSERT_EQ(model.Nodes()[2].ascend_ir->output_mappings.size(), 1u);
  ASSERT_EQ(model.Nodes()[0].ascend_ir->boundary_outputs.size(), 1u);
  ASSERT_EQ(model.Nodes()[1].ascend_ir->boundary_inputs.size(), 1u);
  EXPECT_EQ(model.Nodes()[0].ascend_ir->output_mappings.at("rope_out_0"), "rope_in_place_ge_op:0");
  EXPECT_EQ(model.Nodes()[2].ascend_ir->output_mappings.at("rope_out_1"), "rope_in_place_ge_op_1:0");
  EXPECT_EQ(model.Nodes()[1].ascend_ir->boundary_inputs[0].schema_input_name, "self");
  EXPECT_EQ(model.Nodes()[1].ascend_ir->boundary_inputs[0].fx_tensor_name, "rope_out_0");
  EXPECT_EQ(model.Nodes()[1].ascend_ir->boundary_inputs[0].ge_tensor_name, "rope_in_place_ge_op:0");
  ASSERT_TRUE(model.Nodes()[1].ascend_ir.has_value());
  ASSERT_EQ(model.Nodes()[1].ascend_ir->ops.size(), 1u);
  ASSERT_EQ(model.Nodes()[3].ascend_ir->ops.size(), 1u);
  EXPECT_EQ(model.Nodes()[1].ascend_ir->ops[0].inputs[0], "rope_out_0");
  EXPECT_EQ(model.Nodes()[3].ascend_ir->ops[0].inputs[0], "rope_out_1");
  ASSERT_NE(model.FindNodeByName("RopeInPlace"), nullptr);
  ASSERT_NE(model.FindNodeByName("RopeInPlace_1"), nullptr);
}

namespace {
const std::string kSymIntInput = std::string(FIXTURES_DIR) + "/annotated_sym_int_input.epair";
const std::string kSymFloatBoolInput = std::string(FIXTURES_DIR) + "/annotated_sym_float_bool_input.epair";
const std::string kConstantInput = std::string(FIXTURES_DIR) + "/annotated_constant_input.epair";
const std::string kDynamicDimSymbols = std::string(FIXTURES_DIR) + "/annotated_dynamic_dim_symbols.epair";
const std::string kScalarValueMaps = std::string(FIXTURES_DIR) + "/annotated_scalar_value_maps.epair";
const std::string kScalarFallback = std::string(FIXTURES_DIR) + "/annotated_scalar_fallback.epair";
}  // namespace

TEST_F(EpairModelParserTest, ParsesSymIntUserInput) {
  EpairArchive archive(kSymIntInput.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_GE(model.InputSpecs().size(), 2u);

  bool found_sym_int = false;
  for (const auto &spec : model.InputSpecs()) {
    if (spec.kind == td::PlaceholderKind::UserInput && spec.input_kind == td::NodeInputKind::SymInt) {
      found_sym_int = true;
      EXPECT_EQ(spec.placeholder_name, "batch_size");
    }
  }
  EXPECT_TRUE(found_sym_int);
}

TEST_F(EpairModelParserTest, ParsesSymFloatAndSymBoolUserInputs) {
  EpairArchive archive(kSymFloatBoolInput.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  bool found_sym_float = false;
  bool found_sym_bool = false;
  for (const auto &spec : model.InputSpecs()) {
    if (spec.kind == td::PlaceholderKind::UserInput) {
      if (spec.input_kind == td::NodeInputKind::SymFloat) {
        found_sym_float = true;
        EXPECT_EQ(spec.placeholder_name, "threshold");
      } else if (spec.input_kind == td::NodeInputKind::SymBool) {
        found_sym_bool = true;
        EXPECT_EQ(spec.placeholder_name, "flag");
      }
    }
  }
  EXPECT_TRUE(found_sym_float);
  EXPECT_TRUE(found_sym_bool);
}

TEST_F(EpairModelParserTest, ParsesConstantInputSpec) {
  EpairArchive archive(kConstantInput.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  bool found_constant_input = false;
  for (const auto &spec : model.InputSpecs()) {
    if (spec.kind == td::PlaceholderKind::ConstantInput) {
      found_constant_input = true;
      EXPECT_EQ(spec.placeholder_name, "scale_factor");
      EXPECT_EQ(spec.input_kind, td::NodeInputKind::ScalarInt);
    }
  }
  EXPECT_TRUE(found_constant_input);
}

TEST_F(EpairModelParserTest, ParsesRangeConstraints) {
  EpairArchive archive(kSymIntInput.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  const auto *range = model.FindRangeConstraint("batch_size");
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->first, 1);
  EXPECT_EQ(range->second, 8);
}

TEST_F(EpairModelParserTest, ParsesRangeConstraintsWithNullBounds) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [{"as_tensor": {"name": "x"}}],
        "outputs": [{"as_tensor": {"name": "x"}}],
        "nodes": [],
        "tensor_values": {
          "x": {
            "dtype": 7,
            "sizes": [{"as_expr": {"expr_str": "Symbol('s0')", "hint": {"as_int": 4}}}, 3],
            "strides": [3, 1],
            "storage_offset": 0,
            "layout": 7,
            "requires_grad": false
          }
        }
      },
      "signature": {
        "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
        "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]
      }
    },
    "range_constraints": {
      "s0": {"min_val": null, "max_val": null},
      "s1": {"min_val": null, "max_val": 16},
      "s2": {"min_val": 2, "max_val": null}
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  {
    const auto *range = model.FindRangeConstraint("s0");
    ASSERT_NE(range, nullptr);
    EXPECT_EQ(range->first, 0);
    EXPECT_EQ(range->second, -1);
  }
  {
    const auto *range = model.FindRangeConstraint("s1");
    ASSERT_NE(range, nullptr);
    EXPECT_EQ(range->first, 0);
    EXPECT_EQ(range->second, 16);
  }
  {
    const auto *range = model.FindRangeConstraint("s2");
    ASSERT_NE(range, nullptr);
    EXPECT_EQ(range->first, 2);
    EXPECT_EQ(range->second, -1);
  }
}

TEST_F(EpairModelParserTest, ParsesDynamicDimSymbols) {
  EpairArchive archive(kDynamicDimSymbols.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  const auto *meta = model.FindTensorMeta("x");
  ASSERT_NE(meta, nullptr);
  EXPECT_FALSE(meta->dynamic_dim_symbols.empty());
  if (!meta->dynamic_dim_symbols.empty()) {
    EXPECT_EQ(meta->dynamic_dim_symbols[0], "s0");
  }
}

TEST_F(EpairModelParserTest, ParsesScalarValueMaps) {
  EpairArchive archive(kScalarValueMaps.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  EXPECT_NE(model.FindInt("count"), std::nullopt);
  EXPECT_EQ(model.FindInt("count").value(), 10);
  EXPECT_NE(model.FindFloat("scale"), std::nullopt);
  EXPECT_DOUBLE_EQ(model.FindFloat("scale").value(), 0.5);
  EXPECT_NE(model.FindBool("enabled"), std::nullopt);
  EXPECT_TRUE(model.FindBool("enabled").value());
  EXPECT_NE(model.FindString("mode"), std::nullopt);
  EXPECT_EQ(model.FindString("mode").value(), "train");
}

TEST_F(EpairModelParserTest, ParsesScalarFallbackNodeInputs) {
  EpairArchive archive(kScalarFallback.c_str());
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.Nodes().size(), 1u);

  const auto &node = model.Nodes()[0];
  ASSERT_GE(node.inputs.size(), 2u);

  bool found_scalar_int = false;
  for (const auto &input : node.inputs) {
    if (input.input_kind == td::NodeInputKind::ScalarInt) {
      found_scalar_int = true;
      EXPECT_EQ(input.name, "other");
    }
  }
  EXPECT_TRUE(found_scalar_int);
}

TEST_F(EpairModelParserTest, ParsesValueRefsWithSymTypes) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [{"as_sym_int": {"as_name": "batch_size"}}],
        "outputs": [{"as_sym_int": {"as_name": "batch_size"}}],
        "nodes": [],
        "tensor_values": {}
      },
      "signature": {
        "input_specs": [
          {"user_input": {"arg": {"as_sym_int": {"as_name": "batch_size"}}, "value_type": "sym_int"}}
        ],
        "output_specs": [
          {"user_output": {"arg": {"as_sym_int": {"as_name": "batch_size"}}}}
        ]
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_GE(model.GraphInputs().size(), 1u);
  EXPECT_EQ(model.GraphInputs()[0].name, "batch_size");
}

TEST_F(EpairModelParserTest, ParsesValueRefsWithScalarTypes) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [
          {"as_int": 42},
          {"as_float": 3.14},
          {"as_bool": true},
          {"as_string": "hello"}
        ],
        "outputs": [],
        "nodes": [],
        "tensor_values": {}
      },
      "signature": {
        "input_specs": [
          {"constant_input": {"name": "int_arg", "value": {"as_int": 42}}},
          {"constant_input": {"name": "float_arg", "value": {"as_float": 3.14}}},
          {"constant_input": {"name": "bool_arg", "value": {"as_bool": true}}},
          {"constant_input": {"name": "string_arg", "value": {"as_string": "hello"}}}
        ],
        "output_specs": []
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
  ASSERT_EQ(model.GraphInputs().size(), 4u);
  EXPECT_EQ(model.GraphInputs()[0].name, "int_arg");
  EXPECT_EQ(model.GraphInputs()[1].name, "float_arg");
  EXPECT_EQ(model.GraphInputs()[2].name, "bool_arg");
  EXPECT_EQ(model.GraphInputs()[3].name, "string_arg");
}

TEST_F(EpairModelParserTest, ParsesTensorMetaWithAsExpr) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [{"as_tensor": {"name": "x"}}],
        "outputs": [{"as_tensor": {"name": "x"}}],
        "nodes": [],
        "tensor_values": {
          "x": {
            "dtype": 7,
            "sizes": [{"as_expr": {"expr_str": "Symbol('s0')", "hint": {"as_int": 4}}}, 3],
            "strides": [3, 1],
            "storage_offset": 0,
            "layout": 7,
            "requires_grad": false
          }
        }
      },
      "signature": {
        "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
        "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_json, model), td::SUCCESS);

  const auto *meta = model.FindTensorMeta("x");
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->sizes[0], -1);
  EXPECT_EQ(meta->sizes[1], 3);
  EXPECT_FALSE(meta->dynamic_dim_symbols.empty());
  if (!meta->dynamic_dim_symbols.empty()) {
    EXPECT_EQ(meta->dynamic_dim_symbols[0], "s0");
  }
}

TEST_F(EpairModelParserTest, RejectsNoneConstantInput) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [
            {"as_tensor": {"name": "x"}},
            {"as_none": true}
          ],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "nodes": [{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [
            {"user_input": {"arg": {"as_tensor": {"name": "x"}}}},
            {"constant_input": {"name": "bias", "value": {"as_none": true}}}
          ],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  EXPECT_NE(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
}

TEST_F(EpairModelParserTest, RejectsNoneGraphInput) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [
            {"as_tensor": {"name": "x"}},
            {"as_none": true}
          ],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "nodes": [{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [
            {"user_input": {"arg": {"as_tensor": {"name": "x"}}}},
            {"constant_input": {"name": "bias", "value": {"as_int": 0}}}
          ],
          "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  EXPECT_NE(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
}

TEST_F(EpairModelParserTest, RejectsNoneOutputSpec) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [{"as_tensor": {"name": "x"}}],
          "outputs": [
            {"as_tensor": {"name": "y"}},
            {"as_none": true}
          ],
          "nodes": [{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
          "output_specs": [
            {"user_output": {"arg": {"as_tensor": {"name": "y"}}}},
            {"user_output": {"arg": {"as_none": true}}}
          ]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  EXPECT_NE(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
}

TEST_F(EpairModelParserTest, RejectsNoneGraphOutput) {
  const std::string model_json = R"JSON({
      "graph_module": {
        "graph": {
          "inputs": [{"as_tensor": {"name": "x"}}],
          "outputs": [
            {"as_tensor": {"name": "y"}},
            {"as_none": true}
          ],
          "nodes": [{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "ascend_ir": {"ops": [{"attr": {}}]}
          }],
          "tensor_values": {
            "x": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7},
            "y": {"dtype": 0, "sizes": [3, 3], "requires_grad": false, "strides": [3, 1], "storage_offset": 0, "layout": 7}
          }
        },
        "signature": {
          "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
          "output_specs": [
            {"user_output": {"arg": {"as_tensor": {"name": "y"}}}},
            {"user_output": {"arg": {"as_int": 0}}}
          ]
        }
      },
      "schema_version": {"major": 8, "minor": 7}
    })JSON";

  td::ParsedModel model;
  EXPECT_NE(td::ParsedModel::Parse(model_json, model), td::SUCCESS);
}
// ===========================================================================
// ascend_ir.device 解析（未知取值严格报错）
// ===========================================================================

TEST_F(EpairModelParserTest, ParsesAscendIrDeviceCpu) {
  const auto ir = OrderedJson::parse(R"({"source": "", "device": "cpu", "ops": []})");
  td::ParsedAscendIr parsed;
  ASSERT_EQ(td::ParseAscendIrImpl(ir, "test", parsed), td::SUCCESS);
  EXPECT_EQ(parsed.device, "cpu");
}

TEST_F(EpairModelParserTest, ParsesAscendIrDeviceMissingDefaultsToNpu) {
  const auto ir = OrderedJson::parse(R"({"source": "", "ops": []})");
  td::ParsedAscendIr parsed;
  ASSERT_EQ(td::ParseAscendIrImpl(ir, "test", parsed), td::SUCCESS);
  EXPECT_EQ(parsed.device, "npu") << "Missing device field must default to npu (manifest v1 compat)";
}

TEST_F(EpairModelParserTest, ParsesAscendIrDeviceNpuExplicit) {
  const auto ir = OrderedJson::parse(R"({"source": "", "device": "npu", "ops": []})");
  td::ParsedAscendIr parsed;
  ASSERT_EQ(td::ParseAscendIrImpl(ir, "test", parsed), td::SUCCESS);
  EXPECT_EQ(parsed.device, "npu");
}

TEST_F(EpairModelParserTest, RejectsAscendIrUnknownDevice) {
  const auto ir = OrderedJson::parse(R"({"source": "", "device": "gpu", "ops": []})");
  td::ParsedAscendIr parsed;
  EXPECT_EQ(td::ParseAscendIrImpl(ir, "test", parsed), td::INVALID_PARAM) << "Unknown device value must be rejected";
}

TEST_F(EpairModelParserTest, RejectsAscendIrNonStringDevice) {
  const auto ir = OrderedJson::parse(R"({"source": "", "device": 1, "ops": []})");
  td::ParsedAscendIr parsed;
  EXPECT_EQ(td::ParseAscendIrImpl(ir, "test", parsed), td::INVALID_PARAM);
}

TEST_F(EpairModelParserTest, ParsesHostCpuFixtureNode) {
  const std::string kHostCpu = std::string(FIXTURES_DIR) + "/annotated_host_cpu.epair";
  td::EpairArchive archive(kHostCpu.c_str());
  ASSERT_TRUE(archive.HasRecord(kGraphSource.c_str()));
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  td::ParsedModel model;
  ASSERT_EQ(td::ParsedModel::Parse(model_record.AsString(), model), td::SUCCESS);
  const td::ParsedNode *relu_node = model.FindNodeByName("Relu");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_TRUE(relu_node->ascend_ir.has_value());
  EXPECT_EQ(relu_node->ascend_ir->device, "cpu");
  EXPECT_EQ(relu_node->ascend_ir->source, "");
}
}  // namespace td
