/**
 * backend impl 层内部派发声明：NDK 通路工厂入口（Create*FromNDK）。
 * 受众仅限各域中性工厂 TU（*_factory.cpp）；上层一律经 api/ 中性工厂调用，
 * 不得直接 include 本头。ACL 通路落地时在此追加 Create*FromACL 声明，
 * 由各域 factory TU 按 DetectAvailableDevice 分派。
 */
#ifndef CANN_BACKEND_IMPL_NDK_CREATE_ITF_H
#define CANN_BACKEND_IMPL_NDK_CREATE_ITF_H

#include "types.h"
#include "model/built_model.h"
#include "model_manager/model_manager.h"
#include "tensor/nd_tensor_buffer.h"

namespace hiai {
std::shared_ptr<IBuiltModel> CreateBuiltModelFromNDK();

std::shared_ptr<IModelManager> CreateModelManagerFromNDK();

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc);

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc, const void* data,
                                                             size_t dataSize);

std::shared_ptr<INDTensorBuffer> CreateNDTensorBufferFromNDK(const NDTensorDesc& tensorDesc,
                                                             const NativeHandle& handle);

std::shared_ptr<INDTensorBuffer> CreateKVCacheBufferFromNDK(const NDTensorDesc& tensorDesc, size_t kvCacheNum,
                                                            size_t tileSize);

Status ResizeKVCacheBufferFromNDK(const std::shared_ptr<INDTensorBuffer>& tensor, size_t size);
} // namespace hiai
#endif // CANN_BACKEND_IMPL_NDK_CREATE_ITF_H
