/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */

#ifndef INFRA_ASSERTION_H
#define INFRA_ASSERTION_H

#include "types.h"
#include "infra/log.h"

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_NOT_NULL(ptr)                     \
    do {                                              \
        if ((ptr) == nullptr) {                       \
            CANN_LLM_LOGE(#ptr "null, return FAIL."); \
            return hiai::FAILURE;                     \
        }                                             \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_NOT_NULL_VOID(ptr)                \
    do {                                              \
        if ((ptr) == nullptr) {                       \
            CANN_LLM_LOGW(#ptr "null, return FAIL."); \
            return;                                   \
        }                                             \
    } while (0)

#define HIAI_EXPECT_NOT_NULL_R(ptr, ret)              \
    do {                                              \
        if ((ptr) == nullptr) {                       \
            CANN_LLM_LOGE(#ptr "null, return FAIL."); \
            return ret;                               \
        }                                             \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_TRUE(expr)                                 \
    do {                                                       \
        if (!(expr)) {                                         \
            CANN_LLM_LOGE("%s is false, return FAIL.", #expr); \
            return hiai::FAILURE;                              \
        }                                                      \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_TRUE_WITH_LOG(expr, format, ...) \
    do {                                             \
        if (!(expr)) {                               \
            CANN_LLM_LOGE(format, ##__VA_ARGS__);    \
            return hiai::FAILURE;                    \
        }                                            \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_TRUE_R(expr, ret)                              \
    do {                                                           \
        if (!(expr)) {                                             \
            CANN_LLM_LOGE("%s is false, return %s.", #expr, #ret); \
            return ret;                                            \
        }                                                          \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_TRUE_R_NOLOG(expr, ret) \
    do {                                    \
        if (!(expr)) {                      \
            return ret;                     \
        }                                   \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_TRUE_VOID(expr)                       \
    do {                                                  \
        if (!(expr)) {                                    \
            CANN_LLM_LOGW("%s is false, return.", #expr); \
            return;                                       \
        }                                                 \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_EXEC(func)         \
    do {                               \
        hiai::Status status = func;    \
        if (status != hiai::SUCCESS) { \
            return status;             \
        }                              \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_EXEC_R(func, ret)  \
    do {                               \
        hiai::Status status = func;    \
        if (status != hiai::SUCCESS) { \
            return ret;                \
        }                              \
    } while (0)

/////////////////////////////////////////////////////////////////
#define HIAI_EXPECT_EXEC_VOID(func)    \
    do {                               \
        hiai::Status status = func;    \
        if (status != hiai::SUCCESS) { \
            return;                    \
        }                              \
    } while (0)

#endif // INFRA_ASSERTION_H
