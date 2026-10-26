# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include(TorchDelegateCannCommon)

if(TARGET ascendcl AND TARGET ascendcl_headers)
    set(ascendcl_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_ASCENDCL_INCLUDE_DIR
    NAMES acl/acl_rt.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_ASCENDCL_LIBRARY
    NAMES ascendcl
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ascendcl
    FOUND_VAR ascendcl_FOUND
    REQUIRED_VARS _ASCENDCL_INCLUDE_DIR _ASCENDCL_LIBRARY
)

if(ascendcl_FOUND)
    message(STATUS "[ascendcl] library: ${_ASCENDCL_LIBRARY}")

    add_library(ascendcl_headers INTERFACE IMPORTED)
    set_target_properties(ascendcl_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ASCENDCL_INCLUDE_DIR}"
    )

    add_library(ascendcl SHARED IMPORTED)
    set_target_properties(ascendcl PROPERTIES
        INTERFACE_LINK_LIBRARIES "ascendcl_headers"
        IMPORTED_LOCATION "${_ASCENDCL_LIBRARY}"
    )
endif()
