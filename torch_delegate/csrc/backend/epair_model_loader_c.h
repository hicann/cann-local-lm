/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file epair_model_loader_c.h
 * @brief torch_delegate backend 内部 C ABI（非对外接口，仅限 Python ctypes 绑定使用，
 *        不随 whl 发布，不承诺兼容性）。
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief EpairModelLoader 的 C ABI 不透明句柄，内部为 td::EpairModelLoader 实例。
 *
 * 由 TdEpairModelLoaderCreate 创建、TdEpairModelLoaderDestroy 销毁；
 * 生命周期由调用方独占管理，不允许多个调用方并发操作同一句柄。
 */
typedef void TdEpairModelLoaderHandle;

/**
 * @brief 创建 EpairModelLoader 并完成 .epair 归档解析与 GE 图构建（不编译）。
 * @param archive_path epair 归档文件路径，必须非空。
 * @param handle 输出参数，成功时输出不透明句柄，失败时置为 NULL，必须非空。
 * @return 状态码：0=SUCCESS；1=FAILED；2=INVALID_PARAM；3=INTERNAL_ERROR。
 * @note 失败时可通过 TdEpairModelLoaderLastError() 获取线程局部错误串。
 */
int64_t TdEpairModelLoaderCreate(const char *archive_path, TdEpairModelLoaderHandle **handle);

/**
 * @brief 将句柄内部 GE 图拷贝至调用方提供的 ge::Graph（独立副本）。
 * @param handle TdEpairModelLoaderCreate 输出的句柄，必须非空。
 * @param graph 调用方创建的 ge::Graph 对象指针（ge::Graph*），用于接收图副本，必须非空。
 * @return 状态码：0=SUCCESS；1=FAILED；2=INVALID_PARAM；3=INTERNAL_ERROR。
 * @pre 构造成功（初始化成功）且句柄未销毁。
 * @note 失败时可通过 TdEpairModelLoaderLastError() 获取线程局部错误串。
 */
int64_t TdEpairModelLoaderGetGraph(TdEpairModelLoaderHandle *handle, void *graph /* ge::Graph* */);

/**
 * @brief 销毁句柄并释放内部资源。
 * @param handle 待销毁句柄；NULL 直接返回成功。
 * @note 同一非空句柄只能销毁一次，重复销毁为未定义行为（double free）。
 * @return 状态码：0=SUCCESS；3=INTERNAL_ERROR（析构异常兜底）。
 */
int64_t TdEpairModelLoaderDestroy(TdEpairModelLoaderHandle *handle);

/**
 * @brief 返回线程局部最近一次错误串（含阶段、路径与状态码名）。
 * @return const char* 错误串；无错误时为空串。返回指针由实现持有，调用方不得释放，
 *         内容在下一次本线程出错前有效。
 */
const char *TdEpairModelLoaderLastError(void);

#ifdef __cplusplus
}
#endif
