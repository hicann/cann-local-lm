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

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "torch_delegate/epair_model_loader.h"
#include "backend/epair_archive.h"
#include "common/epair_archive_constants.h"
#include "ge_stub/ge_stub_control.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kMissingGraphSource = std::string(FIXTURES_DIR) + "/missing_graph_source.epair";
const std::string kSymIntInput = std::string(FIXTURES_DIR) + "/annotated_sym_int_input.epair";

// 构造指定 placement 的 int64 标量 tensor；host 数据须由调用方保证生命周期。
gert::Tensor MakeInt64ScalarTensor(int64_t &host_value, void *fake_device_addr, gert::TensorPlacement placement) {
  gert::Tensor tensor;
  tensor.SetDataType(ge::DT_INT64);
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  void *addr = (placement == gert::kOnHost) ? static_cast<void *>(&host_value) : fake_device_addr;
  gert::TensorData tensor_data(addr, nullptr, sizeof(int64_t), placement);
  tensor.SetData(std::move(tensor_data));
  return tensor;
}

gert::Tensor MakeFloatDeviceTensor(void *fake_device_addr) {
  gert::Tensor tensor;
  tensor.SetDataType(ge::DT_FLOAT);
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  gert::TensorData tensor_data(fake_device_addr, nullptr, sizeof(float), gert::kOnDeviceHbm);
  tensor.SetData(std::move(tensor_data));
  return tensor;
}

}  // namespace

namespace td {

class EpairModelLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stub::GeApiStub::Instance().Reset();
  }

  void TearDown() override {
    stub::GeApiStub::Instance().Reset();
  }
};

TEST_F(EpairModelLoaderTest, RunBeforeLoadRejected) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModelWithStreamAsync(nullptr, inputs, outputs), FAILED);
}

TEST_F(EpairModelLoaderTest, ConstructorFailurePropagatesToCompileAndLoad) {
  EpairModelLoader loader(kMissingGraphSource.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  EXPECT_EQ(loader.CompileAndLoad(options, nullptr), FAILED);
}

TEST_F(EpairModelLoaderTest, CompileSucceedsForAnnotatedModel) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  EXPECT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  EXPECT_EQ(stub::GeApiStub::Instance().RegisteredGraphCount(), 1u);
}

TEST_F(EpairModelLoaderTest, CompileAndLoadSucceedsWithNullStream) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  EXPECT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
}

TEST_F(EpairModelLoaderTest, CompileAndLoadIsIdempotent) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  EXPECT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  EXPECT_EQ(stub::GeApiStub::Instance().RegisteredGraphCount(), 1u);
}

TEST_F(EpairModelLoaderTest, RunSucceedsAfterCompileAndLoad) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModelWithStreamAsync(nullptr, inputs, outputs), SUCCESS);
}

TEST_F(EpairModelLoaderTest, RunModelBeforeLoadRejected) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), FAILED);
}

TEST_F(EpairModelLoaderTest, RunModelSucceedsAfterCompileAndLoad) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), SUCCESS);
  EXPECT_FALSE(outputs.empty());
}

TEST_F(EpairModelLoaderTest, RunModelPropagatesGeSessionRunFailure) {
  stub::GeApiStub::Instance().SetRunGraphStatus(ge::FAILED);
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), FAILED);
}

TEST_F(EpairModelLoaderTest, RunModelRejectedAfterAsyncRun) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  ASSERT_EQ(loader.RunModelWithStreamAsync(nullptr, inputs, outputs), SUCCESS);
  EXPECT_EQ(loader.RunModel(inputs, outputs), FAILED);
}

TEST_F(EpairModelLoaderTest, RunModelWithStreamAsyncRejectedAfterSyncRun) {
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  ASSERT_EQ(loader.RunModel(inputs, outputs), SUCCESS);
  EXPECT_EQ(loader.RunModelWithStreamAsync(nullptr, inputs, outputs), FAILED);
}

TEST_F(EpairModelLoaderTest, CompilePropagatesGeSessionAddGraphFailure) {
  stub::GeApiStub::Instance().SetAddGraphStatus(ge::FAILED);
  EpairModelLoader loader(kAnnotated.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  EXPECT_EQ(loader.CompileAndLoad(options, nullptr), FAILED);
}

// fixture 用户输入: [x(Tensor, device), batch_size(SymInt, host)]，sym 输入传 device tensor 应被拒绝
TEST_F(EpairModelLoaderTest, RunModelRejectsDeviceSymIntInput) {
  EpairModelLoader loader(kSymIntInput.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  int64_t host_value = 4;
  std::vector<gert::Tensor> inputs;
  inputs.push_back(MakeFloatDeviceTensor(reinterpret_cast<void *>(0x1000)));
  inputs.push_back(MakeInt64ScalarTensor(host_value, reinterpret_cast<void *>(0x2000), gert::kOnDeviceHbm));
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), INVALID_PARAM);
}

TEST_F(EpairModelLoaderTest, RunModelAcceptsHostSymIntInput) {
  EpairModelLoader loader(kSymIntInput.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  int64_t host_value = 4;
  std::vector<gert::Tensor> inputs;
  inputs.push_back(MakeFloatDeviceTensor(reinterpret_cast<void *>(0x1000)));
  inputs.push_back(MakeInt64ScalarTensor(host_value, nullptr, gert::kOnHost));
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), SUCCESS);
}

TEST_F(EpairModelLoaderTest, RunModelWithStreamAsyncRejectsDeviceSymIntInput) {
  EpairModelLoader loader(kSymIntInput.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  int64_t host_value = 4;
  std::vector<gert::Tensor> inputs;
  inputs.push_back(MakeFloatDeviceTensor(reinterpret_cast<void *>(0x1000)));
  inputs.push_back(MakeInt64ScalarTensor(host_value, reinterpret_cast<void *>(0x2000), gert::kOnDeviceHbm));
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModelWithStreamAsync(nullptr, inputs, outputs), INVALID_PARAM);
}

TEST_F(EpairModelLoaderTest, RunModelRejectsMissingSymIntInput) {
  EpairModelLoader loader(kSymIntInput.c_str());
  std::map<ge::AscendString, ge::AscendString> options;
  ASSERT_EQ(loader.CompileAndLoad(options, nullptr), SUCCESS);
  std::vector<gert::Tensor> inputs;
  inputs.push_back(MakeFloatDeviceTensor(reinterpret_cast<void *>(0x1000)));
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(loader.RunModel(inputs, outputs), INVALID_PARAM);
}

TEST_F(EpairModelLoaderTest, GetGEGraphReturnsValidGraph) {
  EpairModelLoader loader(kAnnotated.c_str());
  ge::Graph graph;
  EXPECT_EQ(loader.GetGEGraph(&graph), SUCCESS);
  const auto nodes = graph.GetDirectNode();
  const std::vector<std::pair<std::string, std::string>> expect_nodes = {
      {"p_weight", "Const"}, {"b_bias", "Const"}, {"x", "Data"},
      {"MatMul", "MatMul"},  {"Add", "Add"},      {"Node_Output", "NetOutput"},
  };
  EXPECT_EQ(nodes.size(), expect_nodes.size());
  for (size_t idx = 0; idx < nodes.size(); ++idx) {
    const auto &node = nodes[idx];
    ge::AscendString name;
    ge::AscendString type;
    node.GetName(name);
    node.GetType(type);
    EXPECT_EQ(expect_nodes[idx].first, name.GetString());
    EXPECT_EQ(expect_nodes[idx].second, type.GetString());
  }
}

// ===========================================================================
// manifest 内容一致性（cann_version/cpu_nodes 字段自 v2 起写入，仅作诊断信息）
// ===========================================================================

namespace {
const std::string kHostCpuFixture = std::string(FIXTURES_DIR) + "/annotated_host_cpu.epair";
}  // namespace

TEST_F(EpairModelLoaderTest, ManifestFixtureConsistent) {
  // annotated_host_cpu 的 manifest 记录导出环境信息（cann_version/cpu_nodes），
  // 归档内容自洽即可加载，版本号不参与加载期门禁。
  td::EpairArchive archive(kHostCpuFixture.c_str());
  ASSERT_TRUE(archive.HasRecord(td::epair_spec::kAnnotatedManifestJson));
  const RecordData manifest_record = archive.ReadRecord(td::epair_spec::kAnnotatedManifestJson);
  const auto manifest = nlohmann::json::parse(manifest_record.AsString());
  ASSERT_EQ(manifest["cann_version"].get<std::string>(), "9.2.0");
  ASSERT_EQ(manifest["cpu_nodes"].get<int>(), 1);
}
}  // namespace td
