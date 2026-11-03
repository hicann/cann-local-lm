/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
 */

#include "hiai_ndk_align.h"

#include <map>

namespace hiai {

HIAI_DataType HIAIAlign::ConvertNNDataTypeToHIAI(OH_NN_DataType dataType)
{
    static std::map<OH_NN_DataType, HIAI_DataType> nnToHiaiDataType = {
        {OH_NN_UNKNOWN, HIAI_DATATYPE_UINT8}, // 默认值
        {OH_NN_BOOL, HIAI_DATATYPE_BOOL},       {OH_NN_INT8, HIAI_DATATYPE_INT8},
        {OH_NN_INT16, HIAI_DATATYPE_INT16},     {OH_NN_INT32, HIAI_DATATYPE_INT32},
        {OH_NN_INT64, HIAI_DATATYPE_INT64},     {OH_NN_UINT8, HIAI_DATATYPE_UINT8},
        {OH_NN_UINT32, HIAI_DATATYPE_UINT32},   {OH_NN_FLOAT16, HIAI_DATATYPE_FLOAT16},
        {OH_NN_FLOAT32, HIAI_DATATYPE_FLOAT32},
    };

    auto iter = nnToHiaiDataType.find(dataType);
    if (iter == nnToHiaiDataType.end()) {
        return HIAI_DATATYPE_NOT_SUPPORTED;
    }
    return iter->second;
}

OH_NN_DataType HIAIAlign::ConvertHIAIDataTypeToNN(HIAI_DataType dataType)
{
    static std::map<HIAI_DataType, OH_NN_DataType> hiaiToNNDataType = {
        {HIAI_DATATYPE_BOOL, OH_NN_BOOL},       {HIAI_DATATYPE_INT8, OH_NN_INT8},
        {HIAI_DATATYPE_INT16, OH_NN_INT16},     {HIAI_DATATYPE_INT32, OH_NN_INT32},
        {HIAI_DATATYPE_INT64, OH_NN_INT64},     {HIAI_DATATYPE_UINT8, OH_NN_UINT8},
        {HIAI_DATATYPE_UINT32, OH_NN_UINT32},   {HIAI_DATATYPE_FLOAT16, OH_NN_FLOAT16},
        {HIAI_DATATYPE_FLOAT32, OH_NN_FLOAT32},
    };

    auto iter = hiaiToNNDataType.find(dataType);
    if (iter == hiaiToNNDataType.end()) {
        return OH_NN_UNKNOWN;
    }
    return iter->second;
}

HIAI_Format HIAIAlign::ConvertNNFormatToHIAI(OH_NN_Format format)
{
    static std::map<OH_NN_Format, HIAI_Format> nnToHiaiFormat = {
        {OH_NN_FORMAT_NONE, HIAI_FORMAT_NCHW}, // 默认值
        {OH_NN_FORMAT_NCHW, HIAI_FORMAT_NCHW},
        {OH_NN_FORMAT_NHWC, HIAI_FORMAT_NHWC},
        {OH_NN_FORMAT_ND, HIAI_FORMAT_ND},
    };

    auto iter = nnToHiaiFormat.find(format);
    if (iter == nnToHiaiFormat.end()) {
        return HIAI_FORMAT_NOT_SUPPORT;
    }
    return iter->second;
}

OH_NN_Format HIAIAlign::ConvertHIAIFormatToNN(HIAI_Format format)
{
    static std::map<HIAI_Format, OH_NN_Format> hiaiToNNFormat = {
        {HIAI_FORMAT_NCHW, OH_NN_FORMAT_NCHW},
        {HIAI_FORMAT_NHWC, OH_NN_FORMAT_NHWC},
        {HIAI_FORMAT_ND, OH_NN_FORMAT_ND},
    };

    auto iter = hiaiToNNFormat.find(format);
    if (iter == hiaiToNNFormat.end()) {
        return OH_NN_FORMAT_NONE;
    }
    return iter->second;
}
} // namespace hiai
