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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <ge/ge_api_v2.h>

/// Test-only control interface for the GE API stub.
/// Include this header only from test files, never from production code.
namespace td {
namespace stub {

class GeApiStub {
 public:
  static GeApiStub &Instance();

  void Reset();
  void SetAddGraphStatus(ge::Status status);
  void SetCompileGraphStatus(ge::Status status);
  void SetLoadGraphStatus(ge::Status status);
  void SetRunGraphStatus(ge::Status status);
  size_t RegisteredGraphCount();
  bool IsGraphCompiled(uint32_t graph_id);
  bool IsGraphLoaded(uint32_t graph_id);

 private:
  GeApiStub() = default;
  ~GeApiStub() = default;
  GeApiStub(const GeApiStub &) = delete;
  GeApiStub &operator=(const GeApiStub &) = delete;

  struct GraphSpec {
    bool compiled = false;
    bool loaded = false;
    size_t output_count = 0;
  };

  struct StubBehavior {
    ge::Status add_graph_status = ge::SUCCESS;
    ge::Status compile_status = ge::SUCCESS;
    ge::Status load_status = ge::SUCCESS;
    ge::Status run_status = ge::SUCCESS;
  };

  friend class ::ge::GeSession;

  std::mutex &Mutex();
  std::unordered_map<const ge::GeSession *, std::unordered_map<uint32_t, GraphSpec>> &SessionGraphs();
  StubBehavior &Behavior();
  GraphSpec *FindGraphSpec(const ge::GeSession *session, uint32_t graph_id);
};

}  // namespace stub
}  // namespace td
