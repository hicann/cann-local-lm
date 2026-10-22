/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/utils/assert.h
 * @brief Assertion / error-check macros used throughout the shim.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/utils/assert.h
 */
#pragma once
#include "log.h"

// ===== Null pointer checks =====
#define ACL_EXPECT_NOT_NULL(ptr) \
    do { if ((ptr) == nullptr) { ACL_LOGE(#ptr " is null"); return ACL_ERROR_FAILURE; } } while(0)

#define ACL_EXPECT_NOT_NULL_VOID(ptr) \
    do { if ((ptr) == nullptr) { ACL_LOGE(#ptr " is null"); return; } } while(0)

#define ACL_EXPECT_NOT_NULL_R(ptr, ret) \
    do { if ((ptr) == nullptr) { ACL_LOGE(#ptr " is null"); return ret; } } while(0)

// ===== Boolean expression checks =====
#define ACL_EXPECT_TRUE(expr) \
    do { if (!(expr)) { ACL_LOGE(#expr " false"); return ACL_ERROR_FAILURE; } } while(0)

#define ACL_EXPECT_TRUE_VOID(expr) \
    do { if (!(expr)) { ACL_LOGE(#expr " false"); return; } } while(0)

#define ACL_EXPECT_TRUE_WITH_LOG(expr, ...) \
    do { if (!(expr)) { ACL_LOGE(__VA_ARGS__); return ACL_ERROR_FAILURE; } } while(0)

#define ACL_EXPECT_TRUE_R(expr, ret) \
    do { if (!(expr)) { ACL_LOGE(#expr " false"); return ret; } } while(0)

#define ACL_EXPECT_TRUE_R_WITH_LOG(expr, ret, ...) \
    do { if (!(expr)) { ACL_LOGE(__VA_ARGS__); return ret; } } while(0)

// ===== Execution result checks (for aclError-returning functions) =====
#define ACL_EXPECT_EXEC(func) \
    do { aclError status = func; if (status != ACL_SUCCESS) { ACL_LOGE(#func " failed"); return status; } } while(0)

#define ACL_EXPECT_EXEC_R(func, ret) \
    do { aclError status = func; if (status != ACL_SUCCESS) { ACL_LOGE(#func " failed"); return ret; } } while(0)

#define ACL_EXPECT_EXEC_VOID(func) \
    do { aclError status = func; if (status != ACL_SUCCESS) { ACL_LOGE(#func " failed"); return; } } while(0)
