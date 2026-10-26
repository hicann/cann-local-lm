/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_ALIGN_H
#define HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_ALIGN_H

#include "neural_network_runtime/neural_network_runtime_type.h"

namespace hiai {
typedef enum {
    HIAI_DATATYPE_UINT8 = 0,
    HIAI_DATATYPE_FLOAT32 = 1,
    HIAI_DATATYPE_FLOAT16 = 2,
    HIAI_DATATYPE_INT32 = 3,
    HIAI_DATATYPE_INT8 = 4,
    HIAI_DATATYPE_INT16 = 5,
    HIAI_DATATYPE_BOOL = 6,
    HIAI_DATATYPE_INT64 = 7,
    HIAI_DATATYPE_UINT32 = 8,
    HIAI_DATATYPE_NOT_SUPPORTED = 255,
} HIAI_DataType;

typedef enum {
    HIAI_FORMAT_NCHW = 0,
    HIAI_FORMAT_NHWC = 1,
    HIAI_FORMAT_ND = 2,
    HIAI_FORMAT_NOT_SUPPORT = 255,
} HIAI_Format;

class HIAIAlign {
public:
    static HIAI_DataType ConvertNNDataTypeToHIAI(OH_NN_DataType dataType);
    static OH_NN_DataType ConvertHIAIDataTypeToNN(HIAI_DataType dataType);

    static HIAI_Format ConvertNNFormatToHIAI(OH_NN_Format format);
    static OH_NN_Format ConvertHIAIFormatToNN(HIAI_Format format);
};
} // namespace hiai

#endif
