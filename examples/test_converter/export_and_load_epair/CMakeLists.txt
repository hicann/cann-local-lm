cmake_minimum_required(VERSION 3.18)
project(export_and_load_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(EXAMPLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(PROJECT_ROOT "${EXAMPLE_DIR}/../../.." ABSOLUTE)

set(CANN_LM_CSRC_DIR "${PROJECT_ROOT}/cann_lm/converter/export/csrc")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT}/cmake/modules")
include(DelegateCommonFlags)

# ---------------------------------------------------------------------------
# Backend: in-tree target (顶层 add_subdirectory 已建) 或 fallback 预编译 .so
# ---------------------------------------------------------------------------
if(NOT TARGET torch_delegate_backend)
    set(CANN_LM_BACKEND_DIR "${PROJECT_ROOT}/build")
    if(NOT EXISTS "${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so")
        message(FATAL_ERROR
            "Backend not found at ${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so\n"
            "  顶层统一构建: bash scripts/cann_lm_compile/build_cpp.sh\n"
            "  bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend")
    endif()
    add_library(torch_delegate_backend SHARED IMPORTED)
    set_target_properties(torch_delegate_backend PROPERTIES
        IMPORTED_LOCATION "${CANN_LM_BACKEND_DIR}/libtorch_delegate_backend.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CANN_LM_CSRC_DIR}/inc/external"
    )
endif()

# ---------------------------------------------------------------------------
# CANN (ASCEND_HOME_PATH → aarch64-linux 子目录, CANN 9.2.0 布局)
# ---------------------------------------------------------------------------
set(CANN_ROOT "" CACHE PATH "Path to CANN installation (ASCEND_HOME_PATH)")
if(CANN_ROOT STREQUAL "" AND DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
endif()
if(CANN_ROOT STREQUAL "")
    message(FATAL_ERROR "CANN_ROOT not set. Source CANN env first (source scripts/set_env_compile.sh).")
endif()
set(CANN_ARCH_DIR "${CANN_ROOT}/aarch64-linux")

# ---------------------------------------------------------------------------
# Torch (for linker to resolve DT_NEEDED of libtorch_delegate_backend.so)
# ---------------------------------------------------------------------------
set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
if(TORCH_PKG_DIR STREQUAL "" AND DEFINED ENV{TORCH_PKG_DIR})
    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})
endif()
if(TORCH_PKG_DIR STREQUAL "")
    message(FATAL_ERROR "TORCH_PKG_DIR not set. Export TORCH_PKG_DIR=<venv>/lib/python3.11/site-packages/torch")
endif()

set(TORCH_LIB_PATH "${TORCH_PKG_DIR}/lib")
get_filename_component(_TORCH_PARENT "${TORCH_PKG_DIR}" DIRECTORY)

set(TORCH_LINK_DIRS "${TORCH_LIB_PATH}")
set(_TORCH_EXTERNAL_LIBS "${_TORCH_PARENT}/torch.libs")
if(EXISTS "${_TORCH_EXTERNAL_LIBS}")
    list(APPEND TORCH_LINK_DIRS "${_TORCH_EXTERNAL_LIBS}")
    message(STATUS "[Torch] also using torch.libs: ${_TORCH_EXTERNAL_LIBS}")
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(export_and_load_example main.cpp)
delegate_apply_common_flags(export_and_load_example)

target_include_directories(export_and_load_example PRIVATE
    ${CANN_ARCH_DIR}/include
)

target_link_directories(export_and_load_example PRIVATE
    ${CANN_ARCH_DIR}/lib64
    ${TORCH_LINK_DIRS}
)

foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
    target_link_options(export_and_load_example PRIVATE
        "LINKER:-rpath-link,${_torch_link_dir}"
    )
endforeach()

target_link_options(export_and_load_example PRIVATE
    "LINKER:-rpath-link,${CANN_ARCH_DIR}/lib64"
    "LINKER:--copy-dt-needed-entries"
)

target_link_libraries(export_and_load_example PRIVATE
    torch_delegate_backend
    ge_runner_v2
    ge_compiler
    acl_rt
)
