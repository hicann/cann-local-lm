# -----------------------------------------------------------------------------------------------------------
# Findlibtorch_npu.cmake - Find or build libtorch_npu
#
# This module provides libtorch_npu_build target.
# libtorch_npu is the C++ library for Ascend NPU support, built from
# Ascend Extension for PyTorch source.
#
# It follows the same pattern as FindGTest.cmake / Findnlohmann_json.cmake:
#   - Check if already built → skip
#   - Otherwise → git clone + python3 build_libtorch_npu.py
# -----------------------------------------------------------------------------------------------------------

if(TARGET libtorch_npu_build)
    return()
endif()

include(ExternalProject)

set(LIBTORCH_NPU_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch_npu")

if(EXISTS "${LIBTORCH_NPU_DIR}/lib" AND EXISTS "${LIBTORCH_NPU_DIR}/include")
    file(GLOB _libtorch_npu_sos "${LIBTORCH_NPU_DIR}/lib/libtorch_npu*.so")
    if(_libtorch_npu_sos)
        message(STATUS "[libtorch_npu] Already built at ${LIBTORCH_NPU_DIR}")
        add_custom_target(libtorch_npu_build)
        return()
    endif()
endif()

if(NOT BUILD_LIBTORCH_NPU)
    message(STATUS "[libtorch_npu] Build disabled (BUILD_LIBTORCH_NPU=OFF)")
    add_custom_target(libtorch_npu_build)
    return()
endif()

message(STATUS "[libtorch_npu] Not found, will build from source.")
message(STATUS "[libtorch_npu] Source: ${LIBTORCH_NPU_SRC_URL}")
message(STATUS "[libtorch_npu] Branch: ${LIBTORCH_NPU_BRANCH}")

set(PYTORCH_SRC_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/../pytorch_src")

ExternalProject_Add(libtorch_npu_build
    GIT_REPOSITORY "${LIBTORCH_NPU_SRC_URL}"
    GIT_TAG "${LIBTORCH_NPU_BRANCH}"
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SOURCE_DIR "${PYTORCH_SRC_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND python3 "${PYTORCH_SRC_DIR}/build_libtorch_npu.py"
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PYTORCH_SRC_DIR}/libtorch_npu"
        "${LIBTORCH_NPU_DIR}"
    EXCLUDE_FROM_ALL TRUE
)
