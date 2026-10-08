#include "torch_delegate/epair_model_loader.h"

#include <atomic>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "backend/ge_graph_builder.h"
#include "common/logging/assert_macros.h"
#include "common/memory_utils.h"

namespace td {
namespace {

std::atomic<uint32_t> g_next_graph_id{0};

}  // namespace

class EpairModelLoader::Impl {
 public:
  Impl(const char *model_json_path, const char *weights_index_path,
       const std::vector<std::string> &tap_outputs) {
    Initialize(model_json_path, weights_index_path, tap_outputs);
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
  void Initialize(const char *model_json_path, const char *weights_index_path,
                  const std::vector<std::string> &tap_outputs) {
    TDLOGI("Initialize enter: model_json=%s, weights_index=%s, tap_outputs=%zu",
           model_json_path ? model_json_path : "(null)",
           weights_index_path ? weights_index_path : "(null)", tap_outputs.size());

    if (model_json_path == nullptr || weights_index_path == nullptr) {
      TDLOGE("Initialize failed: one or more paths are null");
      init_status_ = INVALID_PARAM;
      return;
    }

    // Read model JSON
    std::ifstream file(model_json_path);
    if (!file) {
      TDLOGE("Initialize failed: cannot open %s", model_json_path);
      init_status_ = FAILED;
      return;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string model_json = ss.str();

    ValidationResult result{true, {}};
    // CLI-014 B: infer graph_name from model_json_path filename (prefill/decode/single) for v2 weight_info
    std::string graph_name = "single";
    if (model_json_path) {
      std::string path(model_json_path);
      if (path.find("prefill") != std::string::npos) graph_name = "prefill";
      else if (path.find("decode") != std::string::npos) graph_name = "decode";
    }
    GeGraphBuilder builder(model_json, weights_index_path, result, graph_name);
    if (!result.ok) {
      TDLOGE("Initialize failed: builder init failed, issues=%zu", result.issues.size());
      init_status_ = FAILED;
      return;
    }
    if (!tap_outputs.empty()) {
      builder.SetTapOutputs(tap_outputs);  // tap 须在 Build() 前设, FinalizeGraph 解析
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
  std::unique_ptr<ge::Graph> graph_;
};

EpairModelLoader::EpairModelLoader(const char *model_json_path, const char *weights_index_path)
    : EpairModelLoader(model_json_path, weights_index_path, {}) {}

EpairModelLoader::EpairModelLoader(const char *model_json_path, const char *weights_index_path,
                                   const std::vector<std::string> &tap_outputs)
    : impl_(MakeShared<Impl>(model_json_path, weights_index_path, tap_outputs)) {
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

Status EpairModelLoader::GetGEGraph(ge::Graph *graph) const {
  TD_ASSERT_NOTNULL(impl_, "EpairModelLoader construction failed: Impl allocation failed");
  return impl_->GetGEGraph(graph);
}

}  // namespace td
