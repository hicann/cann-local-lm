# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# CANN 安装路径配置
#
# 优先级（高 → 低）：
#   1. -DASCEND_INSTALL_PATH=<path>   CMake 缓存变量（CLion / 手动 cmake 使用）
#   2. $ENV{ASCEND_HOME_PATH}         环境变量（source CANN 后自动设置）
# ---------------------------------------------------------------------------
set(ASCEND_INSTALL_PATH "" CACHE PATH "CANN installation root directory (fallback: ASCEND_HOME_PATH env)")

if(ASCEND_INSTALL_PATH STREQUAL "")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_INSTALL_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        message(FATAL_ERROR
            "ASCEND_INSTALL_PATH not set.\n"
            "Please specify:\n"
            "  cmake -DASCEND_INSTALL_PATH=/path/to/cann\n"
            "Or source the Ascend environment:\n"
            "  source /usr/local/Ascend/cann/set_env.sh"
        )
    endif()
endif()

if(NOT IS_DIRECTORY "${ASCEND_INSTALL_PATH}")
    message(FATAL_ERROR
        "ASCEND_INSTALL_PATH directory does not exist: ${ASCEND_INSTALL_PATH}"
    )
endif()

function(torch_delegate_cann_include_search_paths out_var)
    set(_paths
        "${ASCEND_INSTALL_PATH}/include"
    )

    list(REMOVE_DUPLICATES _paths)
    set(${out_var} "${_paths}" PARENT_SCOPE)
endfunction()

function(torch_delegate_cann_library_search_paths out_var)
    set(_paths
        "${ASCEND_INSTALL_PATH}/lib64"
        "${ASCEND_INSTALL_PATH}/devlib"
    )

    list(REMOVE_DUPLICATES _paths)
    set(${out_var} "${_paths}" PARENT_SCOPE)
endfunction()

function(torch_delegate_cann_runtime_dirs out_var)
    torch_delegate_cann_library_search_paths(_paths)
    set(${out_var} "${_paths}" PARENT_SCOPE)
endfunction()
