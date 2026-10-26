# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Findlibtorch_cpu.cmake - Find libtorch library
#
# This module finds libtorch from multiple sources:
#   1. Torch_DIR CMake variable (highest priority; TorchConfig dir or torch package/root dir)
#   2. TORCH_PKG_DIR environment variable (Python torch package dir)
#   3. Python torch package auto-detection
#   4. CANN_3RD_LIB_PATH/libtorch (fallback)
#
# Usage:
#   find_package(libtorch_cpu REQUIRED)
#
# Variables:
#   Torch_DIR           - Explicit TorchConfig dir or torch package/root dir (highest priority)
#   TORCH_PKG_DIR (env) - Environment variable pointing to Python torch package dir
#   CANN_3RD_LIB_PATH    - Third-party libraries installation directory (fallback)
#   libtorch_cpu_FOUND  - True if found
# -----------------------------------------------------------------------------------------------------------

if(TARGET libtorch_cpu::libtorch_cpu)
    set(libtorch_cpu_FOUND TRUE)
    return()
endif()

function(_libtorch_cpu_resolve_config_dir _input_dir _output_var)
    if(EXISTS "${_input_dir}/TorchConfig.cmake")
        set(${_output_var} "${_input_dir}" PARENT_SCOPE)
    elseif(EXISTS "${_input_dir}/share/cmake/Torch/TorchConfig.cmake")
        set(${_output_var} "${_input_dir}/share/cmake/Torch" PARENT_SCOPE)
    else()
        set(${_output_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(_libtorch_cpu_detect_from_python _output_var)
    if(DEFINED ENV{VIRTUAL_ENV})
        set(Python3_FIND_STRATEGY LOCATION)
        message(STATUS "[libtorch_cpu] VIRTUAL_ENV detected: $ENV{VIRTUAL_ENV}")
    endif()

    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_Interpreter_FOUND)
        message(STATUS "[libtorch_cpu] Python3 interpreter not found")
        set(${_output_var} "" PARENT_SCOPE)
        return()
    endif()
    message(STATUS "[libtorch_cpu] Python3: ${Python3_EXECUTABLE} (${Python3_VERSION})")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'share' / 'cmake' / 'Torch').as_posix())"
        RESULT_VARIABLE _libtorch_cpu_python_result
        OUTPUT_VARIABLE _libtorch_cpu_python_torch_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(_libtorch_cpu_python_result EQUAL 0 AND
       EXISTS "${_libtorch_cpu_python_torch_dir}/TorchConfig.cmake")
        set(${_output_var} "${_libtorch_cpu_python_torch_dir}" PARENT_SCOPE)
    else()
        message(STATUS "[libtorch_cpu] torch not importable from ${Python3_EXECUTABLE}")
        set(${_output_var} "" PARENT_SCOPE)
    endif()
endfunction()

set(_LIBTORCH_CPU_CANDIDATES)

if(Torch_DIR)
    _libtorch_cpu_resolve_config_dir("${Torch_DIR}" _LIBTORCH_CPU_TORCH_DIR_FROM_CMAKE)
    if(_LIBTORCH_CPU_TORCH_DIR_FROM_CMAKE)
        message(STATUS "[libtorch_cpu] Candidate from CMake Torch_DIR: ${_LIBTORCH_CPU_TORCH_DIR_FROM_CMAKE}")
        list(APPEND _LIBTORCH_CPU_CANDIDATES "${_LIBTORCH_CPU_TORCH_DIR_FROM_CMAKE}")
    else()
        message(STATUS "[libtorch_cpu] Ignoring invalid CMake Torch_DIR: ${Torch_DIR}")
    endif()
endif()

if(DEFINED ENV{TORCH_PKG_DIR})
    _libtorch_cpu_resolve_config_dir("$ENV{TORCH_PKG_DIR}" _LIBTORCH_CPU_TORCH_DIR_FROM_ENV)
    if(_LIBTORCH_CPU_TORCH_DIR_FROM_ENV)
        message(STATUS "[libtorch_cpu] Candidate from environment TORCH_PKG_DIR: ${_LIBTORCH_CPU_TORCH_DIR_FROM_ENV}")
        list(APPEND _LIBTORCH_CPU_CANDIDATES "${_LIBTORCH_CPU_TORCH_DIR_FROM_ENV}")
    else()
        message(STATUS "[libtorch_cpu] Ignoring invalid environment TORCH_PKG_DIR: $ENV{TORCH_PKG_DIR}")
    endif()
endif()

_libtorch_cpu_detect_from_python(_LIBTORCH_CPU_TORCH_DIR_FROM_PYTHON)
if(_LIBTORCH_CPU_TORCH_DIR_FROM_PYTHON)
    message(STATUS "[libtorch_cpu] Candidate auto-detected from Python torch: ${_LIBTORCH_CPU_TORCH_DIR_FROM_PYTHON}")
    list(APPEND _LIBTORCH_CPU_CANDIDATES "${_LIBTORCH_CPU_TORCH_DIR_FROM_PYTHON}")
endif()

if(CANN_3RD_LIB_PATH)
    _libtorch_cpu_resolve_config_dir("${CANN_3RD_LIB_PATH}/libtorch" _LIBTORCH_CPU_TORCH_DIR_FROM_THIRD_PARTY)
    if(_LIBTORCH_CPU_TORCH_DIR_FROM_THIRD_PARTY)
        message(STATUS "[libtorch_cpu] Candidate from CANN_3RD_LIB_PATH: ${_LIBTORCH_CPU_TORCH_DIR_FROM_THIRD_PARTY}")
        list(APPEND _LIBTORCH_CPU_CANDIDATES "${_LIBTORCH_CPU_TORCH_DIR_FROM_THIRD_PARTY}")
    endif()
endif()

list(REMOVE_DUPLICATES _LIBTORCH_CPU_CANDIDATES)

set(_LIBTORCH_CPU_SELECTED_CONFIG_DIR "")

foreach(_LIBTORCH_CPU_CANDIDATE IN LISTS _LIBTORCH_CPU_CANDIDATES)
    message(STATUS "[libtorch_cpu] Probing Torch config: ${_LIBTORCH_CPU_CANDIDATE}")
    unset(Torch_FOUND)
    unset(TORCH_LIBRARIES)
    unset(TORCH_INCLUDE_DIRS)
    find_package(Torch CONFIG QUIET PATHS "${_LIBTORCH_CPU_CANDIDATE}" NO_DEFAULT_PATH)
    if(Torch_FOUND)
        set(_LIBTORCH_CPU_SELECTED_CONFIG_DIR "${_LIBTORCH_CPU_CANDIDATE}")
        break()
    endif()
endforeach()

if(_LIBTORCH_CPU_SELECTED_CONFIG_DIR)
    set(Torch_DIR "${_LIBTORCH_CPU_SELECTED_CONFIG_DIR}" CACHE PATH "libtorch CMake config directory (TorchConfig.cmake)" FORCE)
    message(STATUS "[libtorch_cpu] Using Torch config: ${_LIBTORCH_CPU_SELECTED_CONFIG_DIR}")

    get_filename_component(TORCH_INSTALL_PREFIX "${_LIBTORCH_CPU_SELECTED_CONFIG_DIR}/../../.." ABSOLUTE)
    set(TORCH_LIB_PATH "${TORCH_INSTALL_PREFIX}/lib")
    get_filename_component(_TORCH_PARENT_DIR "${TORCH_INSTALL_PREFIX}" DIRECTORY)
    # Python torch wheels are not laid out consistently across architectures.
    # On x86_64, libtorch and its transitive shared libraries are typically all
    # placed under torch/lib. Some aarch64 wheels split part of the dependency
    # closure into a sibling torch.libs directory instead.
    set(TORCH_EXTERNAL_LIBS_PATH "${_TORCH_PARENT_DIR}/torch.libs")
    set(LIBTORCH_CPU_LINK_HINT_DIRS "")

    if(EXISTS "${TORCH_LIB_PATH}")
        list(APPEND LIBTORCH_CPU_LINK_HINT_DIRS "${TORCH_LIB_PATH}")
        message(STATUS "[libtorch_cpu] Found torch/lib: ${TORCH_LIB_PATH}")
    endif()

    if(EXISTS "${TORCH_EXTERNAL_LIBS_PATH}")
        list(APPEND LIBTORCH_CPU_LINK_HINT_DIRS "${TORCH_EXTERNAL_LIBS_PATH}")
        message(STATUS "[libtorch_cpu] Found torch.libs: ${TORCH_EXTERNAL_LIBS_PATH}")
    else()
        message(STATUS "[libtorch_cpu] torch.libs not present; only torch/lib will be used for link hints")
    endif()

    add_library(libtorch_cpu::libtorch_cpu INTERFACE IMPORTED)
    set_target_properties(libtorch_cpu::libtorch_cpu PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # The final application link step must resolve DT_NEEDED entries from
        # libtorch-linked objects immediately. Runtime-only settings such as
        # LD_LIBRARY_PATH do not help the linker at this stage, so we provide
        # both torch/lib and, when present, torch.libs via -rpath-link.
        # This is especially important for aarch64 Python wheels, where some
        # libtorch dependencies live only in torch.libs.
        foreach(_LIBTORCH_CPU_LINK_HINT_DIR IN LISTS LIBTORCH_CPU_LINK_HINT_DIRS)
            message(STATUS "[libtorch_cpu] Adding linker hint: -Wl,-rpath-link,${_LIBTORCH_CPU_LINK_HINT_DIR}")
            set_property(TARGET libtorch_cpu::libtorch_cpu APPEND PROPERTY
                INTERFACE_LINK_OPTIONS "LINKER:-rpath-link,${_LIBTORCH_CPU_LINK_HINT_DIR}"
            )
        endforeach()
    endif()

    set(LIBTORCH_CPU_RUNTIME_DIRS "${LIBTORCH_CPU_LINK_HINT_DIRS}" CACHE INTERNAL "libtorch runtime directories")
    set(libtorch_cpu_FOUND TRUE)
    return()
endif()

# Not found
set(libtorch_cpu_FOUND FALSE)

if(libtorch_cpu_FIND_REQUIRED)
    message(FATAL_ERROR
        "[libtorch_cpu] libtorch not found.\n"
        "[libtorch_cpu] Options:\n"
        "[libtorch_cpu]   1. Set TORCH_PKG_DIR environment variable: export TORCH_PKG_DIR=/path/to/site-packages/torch\n"
        "[libtorch_cpu]   2. Specify Torch_DIR CMake variable: -DTorch_DIR=/path/to/torch/share/cmake/Torch\n"
        "[libtorch_cpu]   3. Install Python torch and let this module auto-detect it\n"
        "[libtorch_cpu]   4. Build third-party libraries: ./build_third_party.sh (for nlohmann_json and gtest)\n\n"
        "[libtorch_cpu] Note: environment TORCH_PKG_DIR should point to the Python torch package directory\n"
        "[libtorch_cpu]       For Python whl package: <venv>/lib/python3.X/site-packages/torch\n"
        "[libtorch_cpu]       -DTorch_DIR still accepts <torch-root>/share/cmake/Torch for compatibility"
    )
endif()
