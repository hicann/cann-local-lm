/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */
#include "built_model_ndk_impl.h"

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_align.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_nncore.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_options.h"
#include "model_manager/executor_container.h"
// inc
#include "infra/securestl.h"
#include "types.h"
// src/framework/inc
#include "infra/assertion.h"
#include "file_util.h"
#include "infra/log.h"
// src/framework
#include "built_model/customdata_util.h"

#include <functional>
#include "securec.h"

namespace hiai {
std::atomic<uint64_t> BuiltModelNDKImpl::idGenerator_ {0};

BuiltModelNDKImpl::BuiltModelNDKImpl()
{
    builtModelId_ = idGenerator_++;
}

BuiltModelNDKImpl::BuiltModelNDKImpl(std::shared_ptr<OH_NNCompilation> builtModel,
    std::shared_ptr<IBuffer> modelBuffer)
    : nnCompilation_(std::move(builtModel)), modelBuffer_(modelBuffer)
{
    builtModelId_ = idGenerator_++;
}

BuiltModelNDKImpl::~BuiltModelNDKImpl()
{
}

std::shared_ptr<OH_NNCompilation> BuiltModelNDKImpl::GetNNCompilation()
{
    return nnCompilation_;
}

namespace {
std::string GetModelNamefromModel(const std::shared_ptr<IBuffer> buffer)
{
    if (buffer->GetSize() < sizeof(ModelFileHeader) || buffer->GetData() == nullptr) {
        CANN_LLM_LOGW("size of buffer < size of modelheader");
        return "";
    }

    std::string namePrefix = "default_";
    ModelFileHeader header = *reinterpret_cast<const ModelFileHeader*>(buffer->GetData());
    if (header.magic != MODEL_FILE_MAGIC_NUM) {
        return "";
    }
    int32_t nameLen = static_cast<int32_t>(strlen(reinterpret_cast<const char*>(header.name)));
    std::string modelName = namePrefix +
            std::string(reinterpret_cast<const char*>(header.name), std::min(nameLen, 32));
    return modelName;
}

Status NNTensorDescToNDTensorDesc(const NN_TensorDesc* nntensorDesc, NDTensorDesc &ndTensorDesc)
{
    int32_t* shape = nullptr;
    size_t shapeLength = 0;
    Status retCode = HIAI_NDK_NNTensorDesc_GetShape(nntensorDesc, &shape, &shapeLength);
    HIAI_EXPECT_TRUE(retCode == SUCCESS);

    OH_NN_DataType dataType;
    retCode = HIAI_NDK_NNTensorDesc_GetDataType(nntensorDesc, &dataType);
    HIAI_EXPECT_TRUE(retCode == SUCCESS);

    OH_NN_Format format;
    retCode = HIAI_NDK_NNTensorDesc_GetFormat(nntensorDesc, &format);
    HIAI_EXPECT_TRUE(retCode == SUCCESS);

    const char* tensorName = nullptr;
    retCode = HIAI_NDK_NNTensorDesc_GetName(nntensorDesc, &tensorName);
    HIAI_EXPECT_TRUE(retCode == SUCCESS);

    for (size_t i = 0; i < shapeLength; i++) {
        ndTensorDesc.dims.push_back(shape[i]);
    }
    ndTensorDesc.dataType = static_cast<DataType>(HIAIAlign::ConvertNNDataTypeToHIAI(dataType));
    ndTensorDesc.format = static_cast<Format>(HIAIAlign::ConvertNNFormatToHIAI(format));
    ndTensorDesc.tensorName = tensorName == nullptr ? "" : std::string(tensorName);
    return SUCCESS;
}
}

// buffer需要用户自己释放内存
Status BuiltModelNDKImpl::RestoreFromBuffer(const std::shared_ptr<IBuffer>& buffer)
{
    HIAI_EXPECT_NOT_NULL(buffer);
    HIAI_EXPECT_NOT_NULL(buffer->GetData());
    HIAI_EXPECT_TRUE(buffer->GetSize() > 0);
    HIAI_EXPECT_TRUE(nnCompilation_ == nullptr);

    const std::shared_ptr<IBuffer> outBuffer = CustomDataUtil::GetModelData(buffer, customModelData_);
    HIAI_EXPECT_NOT_NULL(outBuffer);
    modelBuffer_ = outBuffer;

    OH_NNCompilation* nnCompilation = HIAI_NDK_NNCompilation_ConstructWithOfflineModelBuffer(
        reinterpret_cast<void*>(reinterpret_cast<char*>(outBuffer->GetData())), outBuffer->GetSize());
    HIAI_EXPECT_NOT_NULL(nnCompilation);

    if (HIAI_NDK_NNCompilation_Build(nnCompilation) != SUCCESS) {
        HIAI_NDK_NNCompilation_Destroy(&nnCompilation);
        return FAILURE;
    }

    nnCompilation_.reset(nnCompilation,
        [](OH_NNCompilation* p) {
            HIAI_NDK_NNCompilation_Destroy(&p);
        });
    HIAI_EXPECT_NOT_NULL(nnCompilation_);

    modelName_ = GetModelNamefromModel(modelBuffer_);
    return SUCCESS;
}

Status BuiltModelNDKImpl::RestoreFromFile(const char* file)
{
    HIAI_EXPECT_TRUE(file != nullptr && std::string(file) != "");
    HIAI_EXPECT_TRUE(nnCompilation_ == nullptr);

    if (CustomDataUtil::HasCustomData(file)) {
        buffer_ = FileUtil::LoadToBuffer(file);
        HIAI_EXPECT_NOT_NULL(buffer_);

        std::shared_ptr<hiai::IBuffer> buffer =
            CreateLocalBuffer(static_cast<void*>(buffer_->MutableData()), buffer_->GetSize(), false);

        return RestoreFromBuffer(buffer);
    }

    modelPath_ = std::string(file);
    OH_NNCompilation* compilation = HIAI_NDK_NNCompilation_ConstructForCache();
    HIAI_EXPECT_NOT_NULL(compilation);
    HIAI_NDK_NNCompilation_SetCache(compilation, modelPath_.c_str(), 0);
    if (HIAI_NDK_NNCompilation_Build(compilation) != SUCCESS) {
        HIAI_NDK_NNCompilation_Destroy(&compilation);
        CANN_LLM_LOGE("HIAI_NDK_NNCompilation_Build failed");
        return FAILURE;
    }

    nnCompilation_.reset(compilation,
        [](OH_NNCompilation* p) {
            HIAI_NDK_NNCompilation_Destroy(&p);
        });
    HIAI_EXPECT_NOT_NULL(nnCompilation_);
    return SUCCESS;
}

std::shared_ptr<OH_NNExecutor> BuiltModelNDKImpl::GetExecutor() const
{
    if (nnCompilation_ == nullptr) {
        CANN_LLM_LOGE("please restore or build first.");
        return nullptr;
    }

    OH_NNCompilation* compilation = reinterpret_cast<OH_NNCompilation*>(nnCompilation_.get());
    ExecutorContainer* container = ExecutorContainer::GetInstance();
    HIAI_EXPECT_NOT_NULL_R(container, nullptr);
    auto nnExecutor = container->GetExecutor(builtModelId_);
    if (nnExecutor == nullptr) {
        nnExecutor = std::shared_ptr<OH_NNExecutor>(HIAI_NDK_NNExecutor_Construct(compilation),
            [](OH_NNExecutor* p) { HIAI_NDK_NNExecutor_Destroy(&p); });
        HIAI_EXPECT_NOT_NULL_R(nnExecutor, nullptr);
        container->AddExecutor(builtModelId_, nnExecutor);
    }
    return nnExecutor;
}

std::vector<NDTensorDesc> BuiltModelNDKImpl::GetInputTensorDescs() const
{
    std::vector<NDTensorDesc> ndTensorDescs;
    auto nnExecutor = GetExecutor();
    HIAI_EXPECT_NOT_NULL_R(nnExecutor, ndTensorDescs);

    size_t inputCount = 0;
    Status retCode = HIAI_NDK_NNExecutor_GetInputCount(nnExecutor.get(), &inputCount);
    if (retCode != SUCCESS || inputCount == 0) {
        CANN_LLM_LOGE("OH_NNExecutor_GetInputCount failed.");
        return ndTensorDescs;
    }

    for (size_t i = 0; i < inputCount; ++i) {
        NN_TensorDesc* nnTensorDesc = HIAI_NDK_NNExecutor_CreateInputTensorDesc(nnExecutor.get(), i);
        if (nnTensorDesc == nullptr) {
            CANN_LLM_LOGE("OH_NNExecutor_CreateInputTensorDesc failed.");
            break;
        }
        NDTensorDesc ndTensorDesc;
        auto status = NNTensorDescToNDTensorDesc(nnTensorDesc, ndTensorDesc);
        HIAI_NDK_NNTensorDesc_Destroy(&nnTensorDesc);
        if (status != SUCCESS) {
            CANN_LLM_LOGE("NNTensorDescToNDTensorDesc failed.");
            break;
        }
        ndTensorDescs.push_back(ndTensorDesc);
    }
    if (ndTensorDescs.size() != inputCount) {
        CANN_LLM_LOGE("ndTensorDescs size mismatch.");
        ndTensorDescs.clear();
        return ndTensorDescs;
    }
    return ndTensorDescs;
}

std::vector<NDTensorDesc> BuiltModelNDKImpl::GetOutputTensorDescs() const
{
    std::vector<NDTensorDesc> ndTensorDescs;
    auto nnExecutor = GetExecutor();
    HIAI_EXPECT_NOT_NULL_R(nnExecutor, ndTensorDescs);

    size_t outputCount = 0;
    Status retCode = HIAI_NDK_NNExecutor_GetOutputCount(nnExecutor.get(), &outputCount);
    if (retCode != SUCCESS || outputCount == 0) {
        CANN_LLM_LOGE("OH_NNExecutor_GetOutputCount failed.");
        return ndTensorDescs;
    }

    for (size_t i = 0; i < outputCount; ++i) {
        NN_TensorDesc* nnTensorDesc = HIAI_NDK_NNExecutor_CreateOutputTensorDesc(nnExecutor.get(), i);
        if (nnTensorDesc == nullptr) {
            CANN_LLM_LOGE("OH_NNExecutor_CreateOutputTensorDesc failed.");
            break;
        }
        NDTensorDesc ndTensorDesc;
        auto status = NNTensorDescToNDTensorDesc(nnTensorDesc, ndTensorDesc);
        HIAI_NDK_NNTensorDesc_Destroy(&nnTensorDesc);
        if (status != SUCCESS) {
            CANN_LLM_LOGE("NNTensorDescToNDTensorDesc failed.");
            break;
        }
        ndTensorDescs.push_back(ndTensorDesc);
    }
    if (ndTensorDescs.size() != outputCount) {
        CANN_LLM_LOGE("ndTensorDescs size mismatch.");
        ndTensorDescs.clear();
        return ndTensorDescs;
    }
    return ndTensorDescs;
}

void BuiltModelNDKImpl::SetName(const std::string& name)
{
    if (nnCompilation_ == nullptr) {
        CANN_LLM_LOGE("please restore or build first.");
        return;
    }
    modelName_ = name;
}

void BuiltModelNDKImpl::SetCustomData(const CustomModelData& customModelData)
{
    customModelData_ = customModelData;
}

const CustomModelData& BuiltModelNDKImpl::GetCustomData()
{
    return customModelData_;
}

std::shared_ptr<IBuiltModel> CreateBuiltModelFromNDK()
{
    return make_shared_nothrow<BuiltModelNDKImpl>();
}
}
