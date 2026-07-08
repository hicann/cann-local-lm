#pragma once

#include <map>
#include <memory>
#include <vector>

#include <exe_graph/runtime/tensor.h>
#include <graph/ascend_string.h>
#include <graph/graph.h>

#include "torch_delegate/td_types.h"

namespace td {

/**
 * @brief EP 模型加载器，负责从 .epair 归档文件构建并执行 GE 计算图。
 *
 * EpairModelLoader 封装了 epair 归档解析、GE Graph 构建、编译和执行的完整流程。
 * 该类使用 PIMPL 模式隔离实现细节，确保对外 ABI 稳定。
 *
 * 典型使用流程：
 * @code
 *   td::EpairModelLoader loader("/path/to/model.epair");
 *   loader.CompileAndLoad(options, stream);
 *   loader.RunModelWithStreamAsync(stream, inputs, outputs);
 * @endcode
 */
class EpairModelLoader {
 public:
  /**
   * @brief 从 .epair 归档文件构造模型加载器。
   * @param archive_path epair 归档文件的绝对路径，必须非空。
   * @note 构造过程会解析归档并构建 GE Graph，但不执行编译。
   */
  explicit EpairModelLoader(const char *archive_path);

  ~EpairModelLoader();

  EpairModelLoader(const EpairModelLoader &) = delete;
  EpairModelLoader &operator=(const EpairModelLoader &) = delete;

  /**
   * @brief 编译并加载模型到指定流。
   * @param options 编译选项。
   * @param stream ACL 流指针，用于异步执行。
   * @return 成功返回 Status::SUCCESS，失败返回对应错误码。
   * @note 该方法组合了 Compile 和 Load 两个步骤。
   */
  Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream);

  /**
   * @brief 在指定流上异步运行模型。
   * @param stream ACL 流指针，必须与 CompileAndLoad 时传入的流一致。
   * @param inputs 输入张量列表, 数据位于 Device 侧。
   * @param outputs 输出张量列表，用于存储计算结果，数据位于 Device 侧。
   * @return 成功返回 Status::SUCCESS，失败返回对应错误码。
   * @pre 必须先调用 CompileAndLoad 成功。
   */
  Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs);

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
