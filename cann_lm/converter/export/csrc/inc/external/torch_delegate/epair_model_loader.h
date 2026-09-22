#pragma once

#include <memory>
#include <string>
#include <vector>

#include <graph/graph.h>

#include "torch_delegate/td_types.h"

namespace td {

/**
 * @brief EP 模型加载器，负责从模型 JSON 和权重索引文件构建 GE 计算图。
 *
 * EpairModelLoader 封装了模型 JSON 解析、权重加载、GE Graph 构建的完整流程。
 * 权重 bin 文件路径从 weight_info.json 索引中自动推导（相对于索引文件目录）。
 * 该类使用 PIMPL 模式隔离实现细节，确保对外 ABI 稳定。
 *
 * 典型使用流程：
 * @code
 *   td::EpairModelLoader loader(
 *       "/path/to/ep_json_with_ascend_ir.json",
 *       "/path/to/weight_info.json");
 *   ge::Graph graph;
 *   loader.GetGEGraph(&graph);
 * @endcode
 */
class EpairModelLoader {
 public:
  /**
   * @brief 从模型 JSON 文件和权重索引文件构造模型加载器。
   * @param model_json_path ep_json_with_ascend_ir.json 文件路径，必须非空。
   * @param weights_index_path weight_info.json 文件路径，必须非空。
   *   权重 bin 文件 (weights_0.bin, weights_1.bin, ...) 路径从此索引推导。
   * @note 构造过程会解析模型 JSON 和权重，构建 GE Graph，但不执行编译。
   */
  EpairModelLoader(const char *model_json_path, const char *weights_index_path);

  /**
   * @brief 同上，但额外指定要 tap 到 graph 输出的中间 FX tensor 名列表。
   * @param tap_outputs 要暴露为 graph 额外输出的 FX 输出 tensor 名（单输出 node 用 node.name，
   *   如 "mm"/"add_1"；多输出 node 用 "name_0"）。FinalizeGraph 经 ValueRegistry alias
   *   解析到对应 GE GNode，append 到 graph outputs。保活中间值供 OM 逐算子比对。
   */
  EpairModelLoader(const char *model_json_path, const char *weights_index_path,
                   const std::vector<std::string> &tap_outputs);

  ~EpairModelLoader();

  EpairModelLoader(const EpairModelLoader &) = delete;
  EpairModelLoader &operator=(const EpairModelLoader &) = delete;

  /**
   * @brief 获取内部 GE Graph 的副本。
   * @param graph 输出参数，用于存储 GE Graph 的副本，必须非空。
   * @return 成功返回 Status::SUCCESS，失败返回对应错误码。
   * @pre 构造函数必须成功执行（模型初始化成功）。
   * @note 该方法通过拷贝返回内部 Graph 副本，可单独编译和执行。
   */
  Status GetGEGraph(ge::Graph *graph) const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace td
