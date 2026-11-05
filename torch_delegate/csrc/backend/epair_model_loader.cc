/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "torch_delegate/epair_model_loader.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ge/ge_api_v2.h>
#include <exe_graph/runtime/tensor.h>

#include "backend/epair_archive.h"
#include "backend/ge_graph_builder.h"
#include "backend/ge_graph_optimizer.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"

namespace td {
namespace {

Status ToTdStatus(const ge::Status status) {
  return status == ge::SUCCESS ? SUCCESS : FAILED;
}

std::atomic<uint32_t> g_next_graph_id{0};

int64_t ReadInt64Scalar(const gert::Tensor &tensor) {
  const void *addr = tensor.GetAddr();
  if (addr == nullptr) {
    TDLOGE("ReadInt64Scalar failed: tensor addr is null");
    return 0;
  }
  return *static_cast<const int64_t *>(addr);
}

gert::Tensor MakeInt64HostTensor(const std::vector<int64_t> &values) {
  gert::StorageShape shape;
  shape.MutableOriginShape().AppendDim(static_cast<int64_t>(values.size()));
  shape.MutableStorageShape().AppendDim(static_cast<int64_t>(values.size()));

  gert::Tensor tensor;
  tensor.GetShape() = shape;
  tensor.SetDataType(ge::DT_INT64);
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});

  const size_t byte_size = values.size() * sizeof(int64_t);
  gert::TensorData tensor_data(const_cast<int64_t *>(values.data()), nullptr, byte_size, gert::kOnHost);
  tensor.SetData(std::move(tensor_data));
  return tensor;
}

gert::Tensor CloneTensorView(const gert::Tensor &src) {
  gert::Tensor dst;
  dst.GetShape() = src.GetShape();
  dst.SetDataType(src.GetDataType());
  dst.MutableFormat() = src.GetFormat();
  dst.MutableTensorData().ShareFrom(src.GetTensorData());
  return dst;
}

Status BuildGeInputs(const std::vector<gert::Tensor> &user_inputs, const InputBindingMap &binding_map,
                     std::vector<gert::Tensor> &ge_inputs, std::vector<std::vector<int64_t>> &packed_storage) {
  if (binding_map.empty()) {
    return SUCCESS;
  }

  ge_inputs.reserve(binding_map.size());
  for (const auto &binding : binding_map) {
    if (binding.source_user_indices.size() == 1) {
      int64_t idx = binding.source_user_indices[0];
      if (idx < 0 || static_cast<size_t>(idx) >= user_inputs.size()) {
        TDLOGE("BuildGeInputs failed: user input index %ld out of range (size=%zu)", static_cast<long>(idx),
               user_inputs.size());
        return INVALID_PARAM;
      }
      ge_inputs.push_back(CloneTensorView(user_inputs[idx]));
    } else {
      packed_storage.emplace_back();
      auto &values = packed_storage.back();
      values.reserve(binding.source_user_indices.size());
      for (int64_t idx : binding.source_user_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= user_inputs.size()) {
          TDLOGE("BuildGeInputs failed: user input index %ld out of range (size=%zu)", static_cast<long>(idx),
                 user_inputs.size());
          return INVALID_PARAM;
        }
        values.push_back(ReadInt64Scalar(user_inputs[idx]));
      }
      ge_inputs.push_back(MakeInt64HostTensor(values));
    }
  }
  return SUCCESS;
}

// sym 输入（SymInt/SymFloat/SymBool）在图中为 host placement 的 Data 节点，
// 用户传入的对应 tensor 必须位于 Host 侧；device tensor 会导致 host 侧误读 device 指针。
Status ValidateSymInputPlacements(const std::vector<gert::Tensor> &inputs,
                                  const std::vector<int64_t> &sym_user_indices) {
  for (int64_t index : sym_user_indices) {
    if (index < 0 || static_cast<size_t>(index) >= inputs.size()) {
      TDLOGE("ValidateSymInputPlacements failed: sym input index %ld out of range (user input count=%zu)",
             static_cast<long>(index), inputs.size());
      return INVALID_PARAM;
    }
    const gert::TensorPlacement placement = inputs[static_cast<size_t>(index)].GetTensorData().GetPlacement();
    if (!gert::TensorPlacementUtils::IsOnHost(placement)) {
      TDLOGE("ValidateSymInputPlacements failed: sym input index %ld requires a host tensor, actual placement=%d",
             static_cast<long>(index), static_cast<int>(placement));
      return INVALID_PARAM;
    }
  }
  return SUCCESS;
}

}  // namespace

class EpairModelLoader::Impl {
 public:
  explicit Impl(const char *archive_path) {
    Initialize(archive_path);
  }

  Status Compile(const std::map<ge::AscendString, ge::AscendString> &options) {
    if (init_status_ != SUCCESS) {
      TDLOGE("Compile failed: EpairModelLoader is not initialized, status=%s", td::StatusName(init_status_));
      return init_status_;
    }
    if (compiled_flag_) {
      TDLOGW("Graph is already compiled, graph_id=%u, session_id=%zu", graph_id_, session_id_);
      return SUCCESS;
    }
    session_ = MakeShared<ge::GeSession>(options);
    if (session_ == nullptr) {
      TDLOGE("Compile failed: GeSession creation failed");
      return FAILED;
    }
    session_id_ = session_->GetSessionId();

    ge::Status ge_status = session_->AddGraph(graph_id_, *graph_, options);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("Compile failed: AddGraph error, return ge_status=%u, graph_id=%u, session_id=%zu", ge_status, graph_id_,
             session_id_);
      return ToTdStatus(ge_status);
    }

    ge_status = session_->CompileGraph(graph_id_);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("Compile failed: CompileGraph return ge_status=%u, graph_id=%u, session_id=%zu", ge_status, graph_id_,
             session_id_);
      return ToTdStatus(ge_status);
    }

    compiled_flag_ = true;
    return SUCCESS;
  }

  Status Load(const std::map<ge::AscendString, ge::AscendString> &options, void *stream) {
    if (init_status_ != SUCCESS) {
      return init_status_;
    }
    if (loaded_flag_) {
      TDLOGW("Graph is already loaded, graph_id=%u, session_id=%zu", graph_id_, session_id_);
      return SUCCESS;
    }
    if (!compiled_flag_ || session_ == nullptr) {
      TDLOGE("Load failed: model is not compiled (compiled_flag=%s) or GeSession is null",
             compiled_flag_ ? "true" : "false");
      return FAILED;
    }

    const ge::Status ge_status = session_->LoadGraph(graph_id_, options, stream);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("Load failed: LoadGraph return ge_status=%u, graph_id=%u, session_id=%zu", ge_status, graph_id_,
             session_id_);
      return ToTdStatus(ge_status);
    }

    loaded_flag_ = true;
    return SUCCESS;
  }

  Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream) {
    const Status compile_status = Compile(options);
    if (compile_status != SUCCESS) {
      return compile_status;
    }
    return Load(options, stream);
  }

  Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs) {
    if (init_status_ != SUCCESS) {
      return init_status_;
    }
    if (!loaded_flag_ || session_ == nullptr) {
      TDLOGE("RunModelWithStreamAsync failed: model is not loaded");
      return FAILED;
    }
    if (run_mode_ == RunMode::kSync) {
      TDLOGE(
          "RunModelWithStreamAsync failed: RunModel has already been called on this instance; RunModel and "
          "RunModelWithStreamAsync are mutually exclusive and cannot be mixed on the same model instance");
      return FAILED;
    }

    TD_CHK_STATUS(ValidateSymInputPlacements(inputs, sym_user_indices_));

    if (input_binding_map_.empty()) {
      const ge::Status ge_status = session_->RunGraphWithStreamAsync(graph_id_, stream, inputs, outputs);
      if (ge_status != ge::SUCCESS) {
        TDLOGE("RunModelWithStreamAsync failed: RunGraphWithStreamAsync error, return ge_status=%u, graph_id=%u",
               ge_status, graph_id_);
        return ToTdStatus(ge_status);
      }
      run_mode_ = RunMode::kAsync;
      return SUCCESS;
    }

    std::vector<gert::Tensor> ge_inputs;
    std::vector<std::vector<int64_t>> packed_storage;
    TD_CHK_STATUS(BuildGeInputs(inputs, input_binding_map_, ge_inputs, packed_storage));

    const ge::Status ge_status = session_->RunGraphWithStreamAsync(graph_id_, stream, ge_inputs, outputs);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("RunModelWithStreamAsync failed: RunGraphWithStreamAsync error, return ge_status=%u, graph_id=%u",
             ge_status, graph_id_);
      return ToTdStatus(ge_status);
    }
    run_mode_ = RunMode::kAsync;
    return SUCCESS;
  }

  Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
    if (init_status_ != SUCCESS) {
      return init_status_;
    }
    if (!loaded_flag_ || session_ == nullptr) {
      TDLOGE("RunModel failed: model is not loaded");
      return FAILED;
    }
    if (run_mode_ == RunMode::kAsync) {
      TDLOGE(
          "RunModel failed: RunModelWithStreamAsync has already been called on this instance; RunModel and "
          "RunModelWithStreamAsync are mutually exclusive and cannot be mixed on the same model instance");
      return FAILED;
    }

    TD_CHK_STATUS(ValidateSymInputPlacements(inputs, sym_user_indices_));

    if (input_binding_map_.empty()) {
      const ge::Status ge_status = session_->RunGraph(graph_id_, inputs, outputs);
      if (ge_status != ge::SUCCESS) {
        TDLOGE("RunModel failed: RunGraph error, return ge_status=%u, graph_id=%u", ge_status, graph_id_);
        return ToTdStatus(ge_status);
      }
      run_mode_ = RunMode::kSync;
      return SUCCESS;
    }

    std::vector<gert::Tensor> ge_inputs;
    std::vector<std::vector<int64_t>> packed_storage;
    TD_CHK_STATUS(BuildGeInputs(inputs, input_binding_map_, ge_inputs, packed_storage));

    const ge::Status ge_status = session_->RunGraph(graph_id_, ge_inputs, outputs);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("RunModel failed: RunGraph error, return ge_status=%u, graph_id=%u", ge_status, graph_id_);
      return ToTdStatus(ge_status);
    }
    run_mode_ = RunMode::kSync;
    return SUCCESS;
  }

  Status GetGEGraph(ge::Graph *out_graph) {
    TD_ASSERT_NOTNULL(out_graph, "GetGEGraph requires non-null out_graph");
    if (init_status_ != SUCCESS) {
      TDLOGE("GetGEGraph failed: init_status=%s", td::StatusName(init_status_));
      return init_status_;
    }
    if (graph_ == nullptr) {
      TDLOGE("GetGEGraph failed: graph is null");
      return FAILED;
    }
    if (out_graph->CopyFrom(*graph_) != ge::GRAPH_SUCCESS) {
      TDLOGE("GetGEGraph failed: copy error");
      return FAILED;
    }
    TDLOGI("GetGEGraph success: graph_id=%u", graph_id_);
    return SUCCESS;
  }

  Status GetInitStatus() const {
    return init_status_;
  }

 private:
  // 同步（RunModel）与异步（RunModelWithStreamAsync）执行接口互斥，同一实例不可混用。
  // GE 层 GeSession 亦有执行模式检查（CheckRunMode），此处提前拦截并给出明确报错，便于定位。
  enum class RunMode {
    kIdle,  // 尚未执行过任何接口
    kSync,  // 已通过 RunModel 成功执行
    kAsync  // 已通过 RunModelWithStreamAsync 成功执行
  };

  void Initialize(const char *archive_path) {
    TDLOGI("Initialize enter: path=%s", archive_path ? archive_path : "(null)");

    if (archive_path == nullptr) {
      TDLOGE("Initialize failed: path is null");
      init_status_ = INVALID_PARAM;
      return;
    }

    TDLOGI("Initialize step: creating archive, path=%s", archive_path);
    const auto archive = EpairArchive::TryCreate(archive_path);
    if (archive == nullptr) {
      TDLOGE("Initialize failed: archive creation error, path=%s", archive_path);
      init_status_ = FAILED;
      return;
    }

    TDLOGI("Initialize step: building graph");
    Status builder_status = FAILED;
    GeGraphBuilder builder(*archive, builder_status);
    if (builder_status != SUCCESS) {
      TDLOGE("Initialize failed: builder init failed, status=%s", td::StatusName(builder_status));
      init_status_ = FAILED;
      return;
    }

    graph_ = builder.Build(input_binding_map_, &sym_user_indices_);
    if (graph_ == nullptr) {
      TDLOGE("Initialize failed: graph build failed");
      init_status_ = FAILED;
      return;
    }

    graph_id_ = g_next_graph_id.fetch_add(1, std::memory_order_relaxed);
    init_status_ = SUCCESS;
    TDLOGI("Initialize exit: success, graph_id=%u", graph_id_);
  }

  Status init_status_ = FAILED;
  uint32_t graph_id_ = 0;
  uint64_t session_id_ = 0;
  bool compiled_flag_ = false;
  bool loaded_flag_ = false;
  RunMode run_mode_ = RunMode::kIdle;
  InputBindingMap input_binding_map_;
  std::vector<int64_t> sym_user_indices_;
  std::shared_ptr<ge::Graph> graph_;
  std::shared_ptr<ge::GeSession> session_;
};

EpairModelLoader::EpairModelLoader(const char *archive_path) : impl_(MakeShared<Impl>(archive_path)) {
  if (impl_ == nullptr) {
    TDLOGE("EpairModelLoader construction failed: Impl allocation failed");
    return;
  }
  // Check if initialization succeeded
  if (impl_->GetInitStatus() != SUCCESS) {
    TDLOGE("EpairModelLoader construction failed: Initialize failed with status=%s",
           td::StatusName(impl_->GetInitStatus()));
  }
}

EpairModelLoader::~EpairModelLoader() = default;

Status EpairModelLoader::CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream) {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->CompileAndLoad(options, stream);
}

Status EpairModelLoader::RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                                                 std::vector<gert::Tensor> &outputs) {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->RunModelWithStreamAsync(stream, inputs, outputs);
}

Status EpairModelLoader::RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->RunModel(inputs, outputs);
}

Status EpairModelLoader::GetGEGraph(ge::Graph *graph) const {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->GetGEGraph(graph);
}

}  // namespace td
