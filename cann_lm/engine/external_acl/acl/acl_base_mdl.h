/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * (C3 fill — was an empty placeholder "install package to override".)
 *
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef INC_EXTERNAL_ACL_ACL_BASE_MDL_H_
#define INC_EXTERNAL_ACL_ACL_BASE_MDL_H_

/* C3: the vendored acl_base_rt.h aclDataType enum uses ACL_FLOAT / ACL_DOUBLE
   (older names). Newer ACL + the backend's ndk_type_utils.h use ACL_FLOAT32 /
   ACL_FLOAT64. Alias them, guarded so real ACL enum values (if the SDK overrides
   this file) take precedence. Included transitively via acl_base.h, so consumers
   of acl_base.h (e.g. ndk_type_utils.h) see these aliases. */

#ifndef ACL_FLOAT32
#define ACL_FLOAT32 ACL_FLOAT
#endif

#ifndef ACL_FLOAT64
#define ACL_FLOAT64 ACL_DOUBLE
#endif

#endif  // INC_EXTERNAL_ACL_ACL_BASE_MDL_H_
