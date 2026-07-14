cmake_minimum_required(VERSION 3.18)
project(export_llm_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(EXAMPLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(PROJECT_ROOT "${EXAMPLE_DIR}/../.." ABSOLUTE)

# Backend library
set(CANN_LM_CSRC_DIR "${PROJECT_ROOT}/cann_lm/converter/export/csrc")
set(CANN_LM_BACKEND_DIR "${CANN_LM_CSRC_DIR}/build")

if(NOT EXISTS "${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so")
    message(FATAL_ERROR
        "Backend library not found at ${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so\n"
        "  Run: cd ${CANN_LM_CSRC_DIR} && bash build_csrc.sh"
    )
endif()

add_library(torch_delegate_backend SHARED IMPORTED)
set_target_properties(torch_delegate_backend PROPERTIES
    IMPORTED_LOCATION "${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CANN_LM_CSRC_DIR}/inc/external"
)

# CANN
set(CANN_ROOT "" CACHE PATH "Path to CANN installation (ASCEND_HOME_PATH)")
if(CANN_ROOT STREQUAL "" AND DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
endif()
if(CANN_ROOT STREQUAL "")
    message(FATAL_ERROR "CANN_ROOT not set. Source CANN env first.")
endif()

# Torch
set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
if(TORCH_PKG_DIR STREQUAL "" AND DEFINED ENV{TORCH_PKG_DIR})
    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})
endif()
if(TORCH_PKG_DIR STREQUAL "")
    message(FATAL_ERROR "TORCH_PKG_DIR not set.")
endif()

set(TORCH_LIB_PATH "${TORCH_PKG_DIR}/lib")
get_filename_component(_TORCH_PARENT "${TORCH_PKG_DIR}" DIRECTORY)

set(TORCH_LINK_DIRS "${TORCH_LIB_PATH}")
set(_TORCH_EXTERNAL_LIBS "${_TORCH_PARENT}/torch.libs")
if(EXISTS "${_TORCH_EXTERNAL_LIBS}")
    list(APPEND TORCH_LINK_DIRS "${_TORCH_EXTERNAL_LIBS}")
endif()

# Executable
add_executable(export_llm_model main.cpp)

target_include_directories(export_llm_model PRIVATE
    ${CANN_ROOT}/include
)

target_link_directories(export_llm_model PRIVATE
    ${CANN_ROOT}/lib64
    ${TORCH_LINK_DIRS}
)

foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
    target_link_options(export_llm_model PRIVATE
        "LINKER:-rpath-link,${_torch_link_dir}"
    )
endforeach()

target_link_options(export_llm_model PRIVATE
    "LINKER:-rpath-link,${CANN_ROOT}/lib64"
)

target_link_options(export_llm_model PRIVATE "LINKER:--copy-dt-needed-entries")

target_link_libraries(export_llm_model PRIVATE
    torch_delegate_backend
    ge_runner_v2
    ge_compiler
    acl_rt
)
