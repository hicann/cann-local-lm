#include "ndk_create_itf.h"
#include "infra/log.h"

namespace hiai {
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc)
{
    return CreateNDTensorBufferFromNDK(tensorDesc);
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc, const void* data, size_t dataSize)
{
    return CreateNDTensorBufferFromNDK(tensorDesc, data, dataSize);
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferNoCopy(
    const NDTensorDesc& tensorDesc, const void* data, size_t dataSize)
{
    (void)tensorDesc;
    (void)data;
    (void)dataSize;
    CANN_LLM_LOGE("Not Support.");
    return nullptr;
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc, const NativeHandle& handle)
{
    return CreateNDTensorBufferFromNDK(tensorDesc, handle);
}

std::shared_ptr<INDTensorBuffer> CreateKVCacheBuffer(const NDTensorDesc& tensorDesc,
    size_t kvCacheNum, size_t tileSize)
{
    return CreateKVCacheBufferFromNDK(tensorDesc, kvCacheNum, tileSize);
}

Status ResizeKVCacheBuffer(const std::shared_ptr<INDTensorBuffer> &tensor, size_t size)
{
    return ResizeKVCacheBufferFromNDK(tensor, size);
}
} // namespace hiai
