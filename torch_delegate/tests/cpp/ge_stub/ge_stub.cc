/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ge/ge_api_v2.h>

#include "ge_stub/ge_stub_control.h"

namespace ge {
namespace {

size_t InferOutputCount(const Graph &graph) {
  size_t output_count = 0;
  for (auto &node : graph.GetDirectNode()) {
    AscendString type;
    (void)node.GetType(type);
    if (type == "NetOutput") {
      output_count += node.GetInputsSize();
    }
  }
  return output_count == 0 ? 1 : output_count;
}

}  // namespace

Status GEInitializeV2(const std::map<AscendString, AscendString> &options) {
  (void)options;
  return SUCCESS;
}

Status GEFinalizeV2() {
  return SUCCESS;
}

AscendString GEGetErrorMsgV3() {
  return AscendString("[STUB] ge_runner_v2 error");
}

AscendString GEGetWarningMsgV3() {
  return AscendString("[STUB] ge_runner_v2 warning");
}

GeSession::GeSession(const std::map<AscendString, AscendString> &options) {
  (void)options;
  std::lock_guard<std::mutex> lock(td::stub::GeApiStub::Instance().Mutex());
  td::stub::GeApiStub::Instance().SessionGraphs().emplace(
      this, std::unordered_map<uint32_t, td::stub::GeApiStub::GraphSpec>{});
}

GeSession::~GeSession() {
  std::lock_guard<std::mutex> lock(td::stub::GeApiStub::Instance().Mutex());
  td::stub::GeApiStub::Instance().SessionGraphs().erase(this);
}

Status GeSession::AddGraph(uint32_t graph_id, const Graph &graph) {
  return AddGraph(graph_id, graph, {});
}

Status GeSession::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options) {
  auto &stub = td::stub::GeApiStub::Instance();
  if (stub.Behavior().add_graph_status != ge::SUCCESS) {
    return stub.Behavior().add_graph_status;
  }
  (void)options;
  std::lock_guard<std::mutex> lock(stub.Mutex());
  auto &graphs = stub.SessionGraphs()[this];
  td::stub::GeApiStub::GraphSpec spec;
  spec.output_count = InferOutputCount(graph);
  graphs[graph_id] = spec;
  return SUCCESS;
}

Status GeSession::CompileGraph(uint32_t graph_id) {
  auto &stub = td::stub::GeApiStub::Instance();
  if (stub.Behavior().compile_status != ge::SUCCESS) {
    return stub.Behavior().compile_status;
  }
  std::lock_guard<std::mutex> lock(stub.Mutex());
  td::stub::GeApiStub::GraphSpec *spec = stub.FindGraphSpec(this, graph_id);
  if (spec == nullptr) {
    return FAILED;
  }
  spec->compiled = true;
  return SUCCESS;
}

Status GeSession::LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
                            void *stream) const {
  auto &stub = td::stub::GeApiStub::Instance();
  if (stub.Behavior().load_status != ge::SUCCESS) {
    return stub.Behavior().load_status;
  }
  (void)options;
  (void)stream;
  std::lock_guard<std::mutex> lock(stub.Mutex());
  td::stub::GeApiStub::GraphSpec *spec = stub.FindGraphSpec(this, graph_id);
  if (spec == nullptr || !spec->compiled) {
    return FAILED;
  }
  spec->loaded = true;
  return SUCCESS;
}

Status GeSession::RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<gert::Tensor> &inputs,
                                          std::vector<gert::Tensor> &outputs) {
  auto &stub = td::stub::GeApiStub::Instance();
  if (stub.Behavior().run_status != ge::SUCCESS) {
    return stub.Behavior().run_status;
  }
  (void)stream;
  (void)inputs;
  std::lock_guard<std::mutex> lock(stub.Mutex());
  td::stub::GeApiStub::GraphSpec *spec = stub.FindGraphSpec(this, graph_id);
  if (spec == nullptr || !spec->loaded) {
    return FAILED;
  }
  if (outputs.empty()) {
    outputs.resize(spec->output_count);
  }
  return SUCCESS;
}

Status GeSession::RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
                           std::vector<gert::Tensor> &outputs) {
  auto &stub = td::stub::GeApiStub::Instance();
  if (stub.Behavior().run_status != ge::SUCCESS) {
    return stub.Behavior().run_status;
  }
  (void)inputs;
  std::lock_guard<std::mutex> lock(stub.Mutex());
  td::stub::GeApiStub::GraphSpec *spec = stub.FindGraphSpec(this, graph_id);
  if (spec == nullptr || !spec->loaded) {
    return FAILED;
  }
  if (outputs.empty()) {
    outputs.resize(spec->output_count);
  }
  return SUCCESS;
}

uint64_t GeSession::GetSessionId() const {
  std::lock_guard<std::mutex> lock(td::stub::GeApiStub::Instance().Mutex());
  return 0;
}

}  // namespace ge

namespace td {
namespace stub {

GeApiStub &GeApiStub::Instance() {
  static GeApiStub instance;
  return instance;
}

std::mutex &GeApiStub::Mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const ge::GeSession *, std::unordered_map<uint32_t, GeApiStub::GraphSpec>> &
GeApiStub::SessionGraphs() {
  static std::unordered_map<const ge::GeSession *, std::unordered_map<uint32_t, GeApiStub::GraphSpec>> graphs;
  return graphs;
}

GeApiStub::StubBehavior &GeApiStub::Behavior() {
  static StubBehavior behavior;
  return behavior;
}

GeApiStub::GraphSpec *GeApiStub::FindGraphSpec(const ge::GeSession *session, uint32_t graph_id) {
  auto session_it = SessionGraphs().find(session);
  if (session_it == SessionGraphs().end()) {
    return nullptr;
  }
  auto graph_it = session_it->second.find(graph_id);
  return graph_it == session_it->second.end() ? nullptr : &graph_it->second;
}

void GeApiStub::Reset() {
  std::lock_guard<std::mutex> lock(Mutex());
  SessionGraphs().clear();
  Behavior() = {};
}

void GeApiStub::SetAddGraphStatus(ge::Status status) {
  Behavior().add_graph_status = status;
}

void GeApiStub::SetCompileGraphStatus(ge::Status status) {
  Behavior().compile_status = status;
}

void GeApiStub::SetLoadGraphStatus(ge::Status status) {
  Behavior().load_status = status;
}

void GeApiStub::SetRunGraphStatus(ge::Status status) {
  Behavior().run_status = status;
}

size_t GeApiStub::RegisteredGraphCount() {
  std::lock_guard<std::mutex> lock(Mutex());
  size_t graph_count = 0;
  for (const auto &session_graphs : SessionGraphs()) {
    graph_count += session_graphs.second.size();
  }
  return graph_count;
}

bool GeApiStub::IsGraphCompiled(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(Mutex());
  for (const auto &session_graphs : SessionGraphs()) {
    const auto it = session_graphs.second.find(graph_id);
    if (it != session_graphs.second.end()) {
      return it->second.compiled;
    }
  }
  return false;
}

bool GeApiStub::IsGraphLoaded(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(Mutex());
  for (const auto &session_graphs : SessionGraphs()) {
    const auto it = session_graphs.second.find(graph_id);
    if (it != session_graphs.second.end()) {
      return it->second.loaded;
    }
  }
  return false;
}

}  // namespace stub
}  // namespace td

extern "C" {

bool IsIrRepSupport(const char *rep) {
  (void)rep;
  return true;
}

ge::Status GetRegisteredIrDefV2(const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
                                std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
                                std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs) {
  inputs.clear();
  outputs.clear();
  attrs.clear();
  if (op_type == nullptr) {
    return ge::FAILED;
  }

  const std::string op(op_type);
  if (op == "Add") {
    inputs.push_back({"x1", "Tensor"});
    inputs.push_back({"x2", "Tensor"});
    outputs.push_back({"y", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }
  if (op == "Relu" || op == "ReluInplace") {
    inputs.push_back({"x", "Tensor"});
    outputs.push_back({"y", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }
  if (op == "aten__relu") {
    // HostCpu fallback 算子（aten::relu.default 的 GE 算子类型），供三分派构图测试。
    inputs.push_back({"x", "Tensor"});
    outputs.push_back({"y", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }
  if (op == "Topk") {
    inputs.push_back({"self", "Tensor"});
    outputs.push_back({"values", "Tensor"});
    outputs.push_back({"indices", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }
  if (op == "RopeInPlace") {
    inputs.push_back({"self", "Tensor"});
    inputs.push_back({"sin_cache", "Tensor"});
    inputs.push_back({"cos_cache", "Tensor"});
    outputs.push_back({"y", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }
  if (op == "BroadcastTo") {
    inputs.push_back({"x", "Tensor"});
    inputs.push_back({"shape", "Tensor"});
    outputs.push_back({"y", "Tensor"});
    return ge::GRAPH_SUCCESS;
  }

  return ge::FAILED;
}

}  // extern "C"
