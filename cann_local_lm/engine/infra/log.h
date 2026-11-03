/* Copyright (C) 2020-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef UTILS_LOG_H
#define UTILS_LOG_H

#include <string.h>
#ifdef __OHOS__
#include <hilog/log.h>
#else
#include <stdio.h>
#endif

#define REAL_FILE_NAME strrchr(__FILE__, '/')

#ifdef __OHOS__

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0010
#define LOG_TAG "CANN_LLM_OPEN"

#define CANN_LLM_LOGD(fmt, ...) \
    OH_LOG_DEBUG(LOG_APP, "%s %s(%d)::" #fmt, (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGI(fmt, ...) \
    OH_LOG_INFO(LOG_APP, "%s %s(%d)::" #fmt, (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGW(fmt, ...) \
    OH_LOG_WARN(LOG_APP, "%s %s(%d)::" #fmt, (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGE(fmt, ...) \
    OH_LOG_ERROR(LOG_APP, "%s %s(%d)::" #fmt, (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#else
#define CANN_LLM_LOGD(fmt, ...) \
    fprintf(stderr, "[ DEBUG ]: %s %s(%d)::" #fmt "\n", (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGI(fmt, ...) \
    fprintf(stderr, "[ INFO  ]: %s %s(%d)::" #fmt "\n", (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGW(fmt, ...) \
    fprintf(stderr, "[WARNING]: %s %s(%d)::" #fmt "\n", (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define CANN_LLM_LOGE(fmt, ...) \
    fprintf(stderr, "[ ERROR ]: %s %s(%d)::" #fmt "\n", (REAL_FILE_NAME + 1), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#endif

#endif
