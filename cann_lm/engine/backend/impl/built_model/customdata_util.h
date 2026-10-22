/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_MODEL_MANAGER_CUSTOMDATA_UTIL_H
#define FRAMEWORK_MODEL_MANAGER_CUSTOMDATA_UTIL_H

// api/framework
#include "model/built_model.h"

namespace hiai {
const char *const CUST_DATA_TAG = "CUST";
class CustomDataUtil {
public:
    static std::shared_ptr<IBuffer> GetModelData(
        const std::shared_ptr<IBuffer>& buffer, CustomModelData& customModelData);

    static bool HasCustomData(const char* file);
};
} // namespace hiai
#endif // FRAMEWORK_MODEL_MANAGER_CUSTOMDATA_UTIL_H
