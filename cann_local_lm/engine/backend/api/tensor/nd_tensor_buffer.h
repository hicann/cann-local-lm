/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_TENSOR_ND_TENSOR_BUFFER_H
#define FRAMEWORK_TENSOR_ND_TENSOR_BUFFER_H

#include <utility>
#include <vector>
#include <memory>

#include "buffer.h"
#include "types.h"

// 未使用：全仓零引用（上游 opaque C 类型保真预留）。
#ifdef __cplusplus
extern "C" {
#endif

struct HIAI_NDTensorBuffer;

#ifdef __cplusplus
}
#endif

namespace hiai {
class INDTensorBuffer : public IBuffer {
public:
    virtual const NDTensorDesc& GetTensorDesc() const = 0;

    virtual void SetCacheStatus(uint8_t status) = 0;

    virtual uint8_t GetCacheStatus() = 0;
};

std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc);
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc, const void* data,
                                                      size_t dataSize);
std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferNoCopy(const NDTensorDesc& tensorDesc, const void* data,
                                                            size_t dataSize);
// 未使用：零调用方（NativeHandle 共享内存通路预留，定义在 nd_tensor_buffer_factory.cpp）。
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc, const NativeHandle& handle);
std::shared_ptr<INDTensorBuffer> CreateKVCacheBuffer(const NDTensorDesc& tensorDesc, size_t kvCacheNum,
                                                     size_t tileSize);
uint32_t ResizeKVCacheBuffer(const std::shared_ptr<INDTensorBuffer>& tensor, size_t size);
} // namespace hiai
#endif // FRAMEWORK_BUFFER_ND_TENSOR_BUFFER_H
