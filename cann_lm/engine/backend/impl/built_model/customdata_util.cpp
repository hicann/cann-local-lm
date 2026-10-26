/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#include "customdata_util.h"

#include "securec.h"
#include "infra/log.h"
#include "base_buffer.h"
#include "file_util.h"
// src/framework/inc
#include "infra/assertion.h"
#include "types.h"

namespace hiai {
bool CheckOffsetValid(const std::shared_ptr<IBuffer>& buffer, size_t offset) { return offset <= buffer->GetSize(); }

std::shared_ptr<IBuffer> SplitCustomData(const std::shared_ptr<IBuffer>& buffer, CustomModelData& customModelData)
{
    size_t modelDataOffset = strlen(CUST_DATA_TAG) + sizeof(int32_t);

    int32_t customDataTypeLen;
    HIAI_EXPECT_TRUE_R(CheckOffsetValid(buffer, modelDataOffset + sizeof(int32_t)), nullptr);

    if (memcpy_s(&customDataTypeLen, sizeof(int32_t),
                 reinterpret_cast<void*>(reinterpret_cast<char*>(buffer->GetData()) + modelDataOffset),
                 sizeof(int32_t)) != 0) {
        CANN_LLM_LOGE("memcpy data failed.");
        return nullptr;
    }

    modelDataOffset += sizeof(int32_t);

    HIAI_EXPECT_TRUE_R(customDataTypeLen > 0, nullptr);
    HIAI_EXPECT_TRUE_R(CheckOffsetValid(buffer, modelDataOffset + customDataTypeLen), nullptr);

    std::string type((reinterpret_cast<char*>(buffer->GetData()) + modelDataOffset), customDataTypeLen);
    customModelData.type = type;

    modelDataOffset += static_cast<size_t>(static_cast<uint32_t>(customDataTypeLen));

    int32_t customDataValueLen;
    if (!CheckOffsetValid(buffer, modelDataOffset + sizeof(int32_t))) {
        return nullptr;
    }
    if (memcpy_s(&customDataValueLen, sizeof(int32_t),
                 reinterpret_cast<void*>(reinterpret_cast<char*>(buffer->GetData()) + modelDataOffset),
                 sizeof(int32_t)) != 0) {
        CANN_LLM_LOGE("memcpy data failed.");
        return nullptr;
    }
    modelDataOffset += sizeof(int32_t);

    HIAI_EXPECT_TRUE_R(customDataValueLen > 0, nullptr);
    HIAI_EXPECT_TRUE_R(
        CheckOffsetValid(buffer, modelDataOffset + static_cast<size_t>(static_cast<uint32_t>(customDataValueLen))),
        nullptr);

    std::string value((reinterpret_cast<char*>(buffer->GetData()) + modelDataOffset), customDataValueLen);
    customModelData.value = value;

    modelDataOffset += static_cast<size_t>(static_cast<uint32_t>(customDataValueLen));

    size_t modelDataSize = buffer->GetSize() - modelDataOffset;
    HIAI_EXPECT_TRUE_R(modelDataSize > 0, nullptr);

    std::shared_ptr<IBuffer> outBuffer = CreateLocalBuffer(modelDataSize);
    HIAI_EXPECT_NOT_NULL_R(outBuffer, nullptr);

    if (memcpy_s(outBuffer->GetData(), outBuffer->GetSize(),
                 reinterpret_cast<void*>(reinterpret_cast<char*>(buffer->GetData()) + modelDataOffset),
                 modelDataSize) != 0) {
        CANN_LLM_LOGE("memcpy data failed.");
        return nullptr;
    }
    return outBuffer;
}

std::shared_ptr<IBuffer> CustomDataUtil::GetModelData(const std::shared_ptr<IBuffer>& buffer,
                                                      CustomModelData& customModelData)
{
    HIAI_EXPECT_TRUE_R(buffer->GetSize() > strlen(CUST_DATA_TAG), nullptr);
    HIAI_EXPECT_NOT_NULL_R(buffer->GetData(), nullptr);

    if (strncmp(static_cast<const char*>(buffer->GetData()), static_cast<const char*>(CUST_DATA_TAG),
                strlen(CUST_DATA_TAG)) == 0) {
        return SplitCustomData(buffer, customModelData);
    }

    return buffer;
}

bool CustomDataUtil::HasCustomData(const char* file)
{
    std::shared_ptr<BaseBuffer> buffer = FileUtil::LoadToBuffer(file, strlen(CUST_DATA_TAG));
    HIAI_EXPECT_TRUE_R(buffer != nullptr && buffer->GetSize() == strlen(CUST_DATA_TAG), false);

    if (strncmp(reinterpret_cast<const char*>(buffer->MutableData()), static_cast<const char*>(CUST_DATA_TAG),
                strlen(CUST_DATA_TAG)) != 0) {
        return false;
    }

    return true;
}
} // namespace hiai
