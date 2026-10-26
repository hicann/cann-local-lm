# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include(TorchDelegateCannCommon)

if(TARGET ascend_hal)
    set(ascend_hal_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_ASCEND_HAL_INCLUDE_DIR
    NAMES driver/ascend_hal.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_ASCEND_HAL_LIBRARY
    NAMES ascend_hal
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ascend_hal
    FOUND_VAR ascend_hal_FOUND
    REQUIRED_VARS _ASCEND_HAL_INCLUDE_DIR _ASCEND_HAL_LIBRARY
)

if(ascend_hal_FOUND)
    message(STATUS "[ascend_hal] library: ${_ASCEND_HAL_LIBRARY}")

    if(NOT TORCH_DELEGATE_CONFIG_MODE)
        # Development mode: create a stub symlink to isolate from the real CANN
        # library, so that test builds can control which library is loaded at
        # runtime (e.g., the CANN stub for unit tests).
        set(_ASCEND_HAL_STUB_DIR "${CMAKE_BINARY_DIR}/stub_lib")
        file(MAKE_DIRECTORY "${_ASCEND_HAL_STUB_DIR}")

        get_filename_component(_ASCEND_HAL_REALPATH "${_ASCEND_HAL_LIBRARY}" REALPATH)
        set(_ASCEND_HAL_STUB_LIBRARY "${_ASCEND_HAL_STUB_DIR}/libascend_hal.so")

        if(EXISTS "${_ASCEND_HAL_STUB_LIBRARY}")
            file(REMOVE "${_ASCEND_HAL_STUB_LIBRARY}")
        endif()

        file(CREATE_LINK
            "${_ASCEND_HAL_REALPATH}"
            "${_ASCEND_HAL_STUB_LIBRARY}"
            COPY_ON_ERROR
        )

        message(STATUS "[ascend_hal] staged library: ${_ASCEND_HAL_STUB_LIBRARY}")
        set(_ASCEND_HAL_TARGET_LOCATION "${_ASCEND_HAL_STUB_LIBRARY}")
    else()
        # Consumer mode (via TorchDelegateConfig.cmake): point directly to the
        # real CANN library. No stub symlink is needed, and we must NOT pollute
        # the consumer's build directory with development-only artifacts.
        set(_ASCEND_HAL_TARGET_LOCATION "${_ASCEND_HAL_LIBRARY}")
    endif()

    add_library(ascend_hal SHARED IMPORTED)
    set_target_properties(ascend_hal PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ASCEND_HAL_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_ASCEND_HAL_TARGET_LOCATION}"
    )
endif()
