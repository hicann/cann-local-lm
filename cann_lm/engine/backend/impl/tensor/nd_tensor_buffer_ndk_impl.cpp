/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */
#include "nd_tensor_buffer_ndk_impl.h"

#include <functional>
#include "infra/assertion.h"
#include "infra/securestl.h"
#include "securec.h"
#include "infra/log.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_tensor.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_align.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_nncore.h"
namespace hiai {

NDTensorBufferNDKImpl::NDTensorBufferNDKImpl(NN_Tensor* nnTensor) : nnTensor_(nnTensor)
{
}

NDTensorBufferNDKImpl::NDTensorBufferNDKImpl(NN_Tensor* nnTensor, const NDTensorDesc& desc)
    : desc_(desc), nnTensor_(nnTensor)
{
}

NDTensorBufferNDKImpl::~NDTensorBufferNDKImpl()
{
    if (nnTensor_ != nullptr) {
        HIAI_NDK_NNTensor_Destroy(&nnTensor_);
    }
}

NN_Tensor* NDTensorBufferNDKImpl::GetNNTensor()
{
    return nnTensor_;
}

void* NDTensorBufferNDKImpl::GetData()
{
    if (nnTensor_ == nullptr) {
        return nullptr;
    }

    return HIAI_NDK_NNTensor_GetDataBuffer(nnTensor_);
}

void NDTensorBufferNDKImpl::SetCacheStatus(uint8_t status)
{
    HIAI_NDK_HiAITensor_SetCacheStatus(nnTensor_, status);
}

uint8_t NDTensorBufferNDKImpl::GetCacheStatus()
{
    uint8_t cacheStatus = 0;
    HIAI_EXPECT_TRUE_R(HIAI_NDK_HiAITensor_GetCacheStatus(nnTensor_, &cacheStatus) == SUCCESS, 0);
    return cacheStatus;
}

size_t NDTensorBufferNDKImpl::GetSize() const
{
    if (nnTensor_ == nullptr) {
        return 0;
    }

    size_t size = 0;
    HIAI_EXPECT_TRUE_R(HIAI_NDK_NNTensor_GetSize(nnTensor_, &size) == SUCCESS, 0);
    return size;
}

namespace {
std::shared_ptr<NN_TensorDesc> Convert2NNTensorDesc(const NDTensorDesc& tensorDesc)
{
    if (tensorDesc.dims.empty()) {
        return nullptr;
    }

    std::shared_ptr<NN_TensorDesc> nnTensorDesc(HIAI_NDK_NNTensorDesc_Create(),
        [](NN_TensorDesc* p) {
            HIAI_NDK_NNTensorDesc_Destroy(&p);
        });
    HIAI_EXPECT_NOT_NULL_R(nnTensorDesc, nullptr);

    HIAI_EXPECT_EXEC_R(HIAI_NDK_NNTensorDesc_SetShape(nnTensorDesc.get(), tensorDesc.dims.data(),
        tensorDesc.dims.size()), nullptr);

    HIAI_EXPECT_TRUE_R(tensorDesc.dataType <= DataType::UINT32, nullptr);
    OH_NN_DataType dataType = HIAIAlign::ConvertHIAIDataTypeToNN(static_cast<HIAI_DataType>(tensorDesc.dataType));
    HIAI_EXPECT_EXEC_R(HIAI_NDK_NNTensorDesc_SetDataType(nnTensorDesc.get(), dataType), nullptr);

    HIAI_EXPECT_TRUE_R(tensorDesc.format != Format::RESERVED, nullptr);
    OH_NN_Format format = HIAIAlign::ConvertHIAIFormatToNN(static_cast<HIAI_Format>(tensorDesc.format));
    HIAI_EXPECT_EXEC_R(HIAI_NDK_NNTensorDesc_SetFormat(nnTensorDesc.get(), format), nullptr);

    return nnTensorDesc;
}
}

std::shared_ptr<INDTensorBuffer> CreateKVCacheBufferFromNDK(const NDTensorDesc& tensorDesc,
    size_t kvCacheNum, size_t tileSize)
{
    auto nnTensorDesc = Convert2NNTensorDesc(tensorDesc);
    HIAI_EXPECT_NOT_NULL_R(nnTensorDesc, nullptr);
    NN_Tensor* nnTensor = HIAI_NDK_HiAITensor_CreateKVCache(nnTensorDesc.get(), kvCacheNum, tileSize);
    HIAI_EXPECT_NOT_NULL_R(nnTensor, nullptr);
    return make_shared_nothrow<NDTensorBufferNDKImpl>(nnTensor, tensorDesc);
}

Status ResizeKVCacheBufferFromNDK(const std::shared_ptr<INDTensorBuffer> &tensor, size_t size)
{
    HIAI_EXPECT_NOT_NULL(tensor);
    auto tensorImpl = std::dynamic_pointer_cast<NDTensorBufferNDKImpl>(tensor);
    HIAI_EXPECT_NOT_NULL(tensorImpl);
    return HIAI_NDK_HiAITensor_ResizeKVCache(tensorImpl->GetNNTensor(), size);
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc)
{
    auto nnTensorDesc = Convert2NNTensorDesc(tensorDesc);
    HIAI_EXPECT_NOT_NULL_R(nnTensorDesc, nullptr);

    NN_Tensor* nnTensor = HIAI_NDK_NNTensor_Create(nnTensorDesc.get());
    HIAI_EXPECT_NOT_NULL_R(nnTensor, nullptr);

    return make_shared_nothrow<NDTensorBufferNDKImpl>(nnTensor, tensorDesc);
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc,
    const void* data, size_t dataSize)
{
    auto nnTensorDesc = Convert2NNTensorDesc(tensorDesc);
    HIAI_EXPECT_NOT_NULL_R(nnTensorDesc, nullptr);

    NN_Tensor* nnTensor = HIAI_NDK_NNTensor_Create(nnTensorDesc.get());
    HIAI_EXPECT_NOT_NULL_R(nnTensor, nullptr);

    size_t tensorSize = 0;
    Status status = HIAI_NDK_NNTensor_GetSize(nnTensor, &tensorSize);
    if (status != SUCCESS || tensorSize != dataSize) {
        CANN_LLM_LOGE("mismatch buffer size.");
        HIAI_NDK_NNTensor_Destroy(&nnTensor);
        return nullptr;
    }
    void* tensorData = HIAI_NDK_NNTensor_GetDataBuffer(nnTensor);
    if (memcpy_s(tensorData, tensorSize, data, tensorSize) != 0) {
        CANN_LLM_LOGE("memcpy buffer failed.");
        HIAI_NDK_NNTensor_Destroy(&nnTensor);
        return nullptr;
    }

    return make_shared_nothrow<NDTensorBufferNDKImpl>(nnTensor, tensorDesc);
}

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc,
    const NativeHandle& handle)
{
    auto nnTensorDesc = Convert2NNTensorDesc(tensorDesc);
    HIAI_EXPECT_NOT_NULL_R(nnTensorDesc, nullptr);

    HIAI_EXPECT_TRUE_R(handle.size > 0 && handle.offset >= 0, nullptr);
    size_t size = handle.size;
    size_t offset = handle.offset;
    NN_Tensor* nnTensor = HIAI_NDK_NNTensor_CreateWithFd(nnTensorDesc.get(), handle.fd, size, offset);
    HIAI_EXPECT_NOT_NULL_R(nnTensor, nullptr);

    return make_shared_nothrow<NDTensorBufferNDKImpl>(nnTensor, tensorDesc);
}
} // namespace hiai