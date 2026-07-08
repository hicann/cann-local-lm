#include <gtest/gtest.h>

#include <string>

#include "backend/epair_model_parser.h"
#include "common/abi_compat_utils.h"
#include "common/epair_archive_constants.h"
#include "common/logging/logging.h"
#include "backend/epair_archive.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
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
  ValidationResult result{true, {}};
  const RecordData model_record = archive.ReadRecord(kGraphSource.c_str());
  const std::string model_json = model_record.AsString();
  const td::OrderedJson root = td::EpairModelParser::ParseRootJson(model_json, result);
  const td::ParsedModel parsed_model = td::EpairModelParser::ParseModelJson(root, result);
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);

  EXPECT_TRUE(result.ok);
  ASSERT_EQ(parsed_model.nodes.size(), 2u);
  EXPECT_EQ(parser.SchemaVersionValue().major, 8);
  EXPECT_EQ(parser.SchemaVersionValue().minor, 7);
  ASSERT_EQ(parser.GraphInputs().size(), 3u);
  ASSERT_EQ(parser.GraphOutputs().size(), 1u);
  ASSERT_EQ(parser.Nodes().size(), 2u);
  EXPECT_EQ(parser.LogicalNameOfPlaceholder("p_weight"), "weight");
  EXPECT_EQ(parser.PlaceholderKindOf("b_bias"), td::PlaceholderKind::Buffer);
  ASSERT_NE(parser.FindTensorValue("mm"), nullptr);
}

TEST_F(EpairModelParserTest, MissingGraphSourceDetectedByArchive) {
  EpairArchive archive(kMissingGraphSource.c_str());
  EXPECT_FALSE(archive.HasRecord(kGraphSource.c_str()));
}

TEST_F(EpairModelParserTest, ParsesSymIntUserInput) {
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

  ValidationResult result{true, {}};
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);

  ASSERT_TRUE(result.ok);
  ASSERT_GE(parser.GraphInputs().size(), 1u);
  EXPECT_EQ(parser.GraphInputs()[0].name, "batch_size");
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
        "outputs": [{"as_int": 42}],
        "nodes": [],
        "tensor_values": {}
      },
      "signature": {
        "input_specs": [],
        "output_specs": []
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  ValidationResult result{true, {}};
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);

  ASSERT_TRUE(result.ok);
  ASSERT_EQ(parser.GraphInputs().size(), 4u);
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

  ValidationResult result{true, {}};
  td::EpairModelParser::Parse(model_json, result);

  ASSERT_FALSE(result.ok);
  ASSERT_FALSE(result.issues.empty());
  EXPECT_EQ(result.issues[0].code, "UNSUPPORTED_FUNCTIONALIZED_V1");
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

  ValidationResult result{true, {}};
  const td::OrderedJson root = td::EpairModelParser::ParseRootJson(model_json, result);
  const td::ParsedModel parsed_model = td::EpairModelParser::ParseModelJson(root, result);

  ASSERT_TRUE(result.ok);

  auto it = parsed_model.tensor_values.find("x");
  ASSERT_NE(it, parsed_model.tensor_values.end());
  EXPECT_EQ(it->second.sizes[0], -1);
  EXPECT_EQ(it->second.sizes[1], 3);
}

TEST_F(EpairModelParserTest, ParsesScalarValueMaps) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [{"as_tensor": {"name": "x"}}],
        "outputs": [{"as_tensor": {"name": "x"}}],
        "nodes": [],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [3], "strides": [1], "storage_offset": 0, "layout": 7, "requires_grad": false}
        },
        "int_values": {"count": 10},
        "float_values": {"scale": 0.5},
        "bool_values": {"enabled": true},
        "string_values": {"mode": "train"}
      },
      "signature": {
        "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
        "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  ValidationResult result{true, {}};
  const td::OrderedJson root = td::EpairModelParser::ParseRootJson(model_json, result);
  const td::ParsedModel parsed_model = td::EpairModelParser::ParseModelJson(root, result);

  ASSERT_TRUE(result.ok);

  EXPECT_FALSE(parsed_model.int_values.empty());
  EXPECT_FALSE(parsed_model.float_values.empty());
  EXPECT_FALSE(parsed_model.bool_values.empty());
  EXPECT_FALSE(parsed_model.string_values.empty());

  auto int_it = parsed_model.int_values.find("count");
  if (int_it != parsed_model.int_values.end()) {
    EXPECT_EQ(int_it->second, 10);
  }

  auto float_it = parsed_model.float_values.find("scale");
  if (float_it != parsed_model.float_values.end()) {
    EXPECT_DOUBLE_EQ(float_it->second, 0.5);
  }

  auto bool_it = parsed_model.bool_values.find("enabled");
  if (bool_it != parsed_model.bool_values.end()) {
    EXPECT_TRUE(bool_it->second);
  }

  auto string_it = parsed_model.string_values.find("mode");
  if (string_it != parsed_model.string_values.end()) {
    EXPECT_EQ(string_it->second, "train");
  }
}

TEST_F(EpairModelParserTest, ParsesScalarFallbackNodeInputs) {
  const std::string model_json = R"JSON({
    "graph_module": {
      "graph": {
        "inputs": [{"as_tensor": {"name": "x"}}],
        "outputs": [{"as_tensor": {"name": "y"}}],
        "nodes": [{
          "target": "torch.ops.aten.add.Tensor",
          "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "other", "arg": {"as_int": 1}, "kind": 1}
          ],
          "outputs": [{"as_tensor": {"name": "y"}}],
          "ascend_ir": {"source": "", "ops": [{"attr": {}}]}
        }],
        "tensor_values": {
          "x": {"dtype": 7, "sizes": [3], "strides": [1], "storage_offset": 0, "layout": 7, "requires_grad": false},
          "y": {"dtype": 7, "sizes": [3], "strides": [1], "storage_offset": 0, "layout": 7, "requires_grad": false}
        }
      },
      "signature": {
        "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
        "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}]
      }
    },
    "schema_version": {"major": 8, "minor": 7}
  })JSON";

  ValidationResult result{true, {}};
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);

  ASSERT_TRUE(result.ok);
  ASSERT_EQ(parser.Nodes().size(), 1u);

  const auto &node = parser.Nodes()[0];
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

}  // namespace td
