# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include(TorchDelegateCannCommon)

if(TARGET es_math)
    set(es_math_FOUND TRUE)
    return()
endif()

torch_delegate_cann_library_search_paths(_td_library_paths)

find_library(_ES_MATH_LIBRARY
    NAMES es_math
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(es_math
    FOUND_VAR es_math_FOUND
    REQUIRED_VARS _ES_MATH_LIBRARY
)

if(es_math_FOUND)
    message(STATUS "[es_math] library: ${_ES_MATH_LIBRARY}")

    add_library(es_math SHARED IMPORTED)
    set_target_properties(es_math PROPERTIES
        IMPORTED_LOCATION "${_ES_MATH_LIBRARY}"
    )
endif()
