/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_BUFFER_ND_TENSOR_BUFFER_NDK_IMPL_H
#define FRAMEWORK_BUFFER_ND_TENSOR_BUFFER_NDK_IMPL_H
#include "tensor/nd_tensor_buffer.h"
#include "neural_network_runtime/neural_network_core.h"

namespace hiai {

class NDTensorBufferNDKImpl : public INDTensorBuffer {
public:
    explicit NDTensorBufferNDKImpl(NN_Tensor* nnTensor);
    NDTensorBufferNDKImpl(NN_Tensor* nnTensor, const NDTensorDesc& desc);
    ~NDTensorBufferNDKImpl() override;

    NN_Tensor* GetNNTensor();

    void* GetData() override;

    size_t GetSize() const override;

    const NDTensorDesc& GetTensorDesc() const override { return desc_; }

    void SetCacheStatus(uint8_t status) override;

    uint8_t GetCacheStatus() override;

#ifdef AI_SUPPORT_DECODER_OFFLOAD
    int32_t GetFd() { return -1; }
    void* GetImpl() { return nullptr; }
#endif

private:
    NDTensorBufferNDKImpl(const NDTensorBufferNDKImpl&) = delete;
    NDTensorBufferNDKImpl& operator=(const NDTensorBufferNDKImpl&) = delete;

private:
    NDTensorDesc desc_;
    NN_Tensor* nnTensor_{nullptr};
};
} // namespace hiai

#endif
