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
   * @param stream ACL 流指针，指定图在哪个Stream上运行。
   * @param inputs 输入张量列表, tensor 输入数据位于 Device 侧；SymInt/SymFloat/SymBool
   *               输入必须位于 Host 侧（否则返回 INVALID_PARAM）。
   * @param outputs 输出张量列表，用于存储计算结果，数据位于 Device 侧。
   * @return 成功返回 Status::SUCCESS，失败返回对应错误码。
   * @pre 必须先调用 CompileAndLoad 成功。
   * @note 本接口与 RunModel 互斥，同一模型不可混用：若同一实例已调用过
   *       RunModel，本接口将直接返回错误并打印明确日志提示互斥。
   */
  Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs);

  /**
   * @brief 同步运行模型，阻塞直到图执行完成后再返回。
   * @param inputs 输入张量列表，tensor 输入数据可位于 Host 侧或 Device 侧；
   *               SymInt/SymFloat/SymBool 输入必须位于 Host 侧（否则返回 INVALID_PARAM）。
   * @param outputs 输出张量列表，调用方无需分配内存，执行完成后由 GE 自动分配
   *                Host 内存赋值返回。
   * @return 成功返回 Status::SUCCESS，失败返回对应错误码。
   * @pre 必须先调用 CompileAndLoad 成功。
   * @note 与 RunModelWithStreamAsync 不同，本接口返回时输出已可读取，
   *       无需再调用 aclrtSynchronizeStream 同步；本接口与 RunModelWithStreamAsync
   *       互斥，同一模型不可混用：若同一实例已调用过 RunModelWithStreamAsync，
   *       本接口将直接返回错误并打印明确日志提示互斥。内部封装 GE 同步接口
   *       RunGraph；同一实例不支持多线程并发调用。
   */
  Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

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
