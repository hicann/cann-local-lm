#include "torch_delegate/epair_model_loader.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ge/ge_api_v2.h>

#include "backend/epair_archive.h"
#include "backend/ge_graph_builder.h"
#include "common/logging/assert_macros.h"
#include "common/memory_utils.h"

namespace td {
namespace {

Status ToTdStatus(const ge::Status status) {
  return status == ge::SUCCESS ? SUCCESS : FAILED;
}

std::atomic<uint32_t> g_next_graph_id{0};

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
    const ge::Status ge_status = session_->RunGraphWithStreamAsync(graph_id_, stream, inputs, outputs);
    if (ge_status != ge::SUCCESS) {
      TDLOGE("RunModelWithStreamAsync failed: RunGraphWithStreamAsync error, return ge_status=%u, graph_id=%u", ge_status, graph_id_);
      return ToTdStatus(ge_status);
    }
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
  void Initialize(const char *archive_path) {
    TDLOGI("Initialize enter: path=%s", archive_path ? archive_path : "(null)");

    if (archive_path == nullptr) {
      TDLOGE("Initialize failed: path is null");
      init_status_ = INVALID_PARAM;
      return;
    }

    TDLOGI("Initialize step: creating archive");
    const auto archive = EpairArchive::TryCreate(archive_path);
    if (archive == nullptr) {
      TDLOGE("Initialize failed: archive creation error, path=%s", archive_path);
      init_status_ = FAILED;
      return;
    }
    TDLOGI("Initialize step: archive created, path=%s", archive->Path());

    ValidationResult result{true, {}};
    TDLOGI("Initialize step: building graph");
    GeGraphBuilder builder(*archive, result);
    if (!result.ok) {
      TDLOGE("Initialize failed: builder init failed, issues=%zu", result.issues.size());
      init_status_ = FAILED;
      return;
    }
    TDLOGI("Initialize step: builder initialized");

    graph_ = builder.Build(result);
    if (!result.ok || graph_ == nullptr) {
      TDLOGE("Initialize failed: graph build failed, issues=%zu", result.issues.size());
      init_status_ = FAILED;
      return;
    }
    TDLOGI("Initialize step: graph built");

    graph_id_ = g_next_graph_id.fetch_add(1, std::memory_order_relaxed);
    init_status_ = SUCCESS;
    TDLOGI("Initialize exit: success, graph_id=%u", graph_id_);
  }

  Status init_status_ = FAILED;
  uint32_t graph_id_ = 0;
  uint64_t session_id_ = 0;
  bool compiled_flag_ = false;
  bool loaded_flag_ = false;
  std::unique_ptr<ge::Graph> graph_;
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

Status EpairModelLoader::GetGEGraph(ge::Graph *graph) const {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->GetGEGraph(graph);
}

}  // namespace td
