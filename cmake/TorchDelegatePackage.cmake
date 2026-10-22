# -----------------------------------------------------------------------------------------------------------
# TorchDelegatePackage.cmake - Package TorchDelegate for distribution
#
# This module provides packaging functionality for TorchDelegate.
# It creates a zip package containing:
#   - inc/torch_delegate/ (external headers)
#   - lib/ (shared libraries and CMake config package)
#
# Usage:
#   include(TorchDelegatePackage)
#   torch_delegate_package(
#       VERSION "0.1.0"
#       BUILD_TYPE "debug"
#       OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output/packages"
#   )
# -----------------------------------------------------------------------------------------------------------

function(torch_delegate_package)
    set(options "")
    set(oneValueArgs VERSION BUILD_TYPE OUTPUT_DIR)
    set(multiValueArgs "")
    
    cmake_parse_arguments(PACKAGE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Required arguments
    if(NOT PACKAGE_VERSION)
        message(FATAL_ERROR "torch_delegate_package: VERSION is required")
    endif()
    if(NOT PACKAGE_BUILD_TYPE)
        set(PACKAGE_BUILD_TYPE "debug")
    endif()
    if(NOT PACKAGE_OUTPUT_DIR)
        set(PACKAGE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output/packages")
    endif()
    
    # Detect platform
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(PACKAGE_PLATFORM "linux-x86_64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(PACKAGE_PLATFORM "linux-aarch64")
        else()
            set(PACKAGE_PLATFORM "linux-${CMAKE_SYSTEM_PROCESSOR}")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(PACKAGE_PLATFORM "macos-${CMAKE_SYSTEM_PROCESSOR}")
    else()
        set(PACKAGE_PLATFORM "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    
    # Package name
    set(PACKAGE_NAME "torch_delegate-${PACKAGE_VERSION}-${PACKAGE_PLATFORM}-${PACKAGE_BUILD_TYPE}")
    
    # Create package directory
    set(PACKAGE_DIR "${CMAKE_BINARY_DIR}/package/${PACKAGE_NAME}")
    file(MAKE_DIRECTORY "${PACKAGE_DIR}")
    
    # Copy installed files to package directory
    message(STATUS "[Package] Creating package: ${PACKAGE_NAME}")
    message(STATUS "[Package] Package directory: ${PACKAGE_DIR}")
    
    # Copy headers
    file(MAKE_DIRECTORY "${PACKAGE_DIR}/inc")
    file(COPY "${CMAKE_BINARY_DIR}/install/inc/torch_delegate" 
        DESTINATION "${PACKAGE_DIR}/inc"
    )
    
    # Copy delegate libraries and CMake config package only.
    # Do not bundle transitive CANN libraries such as libmetadef.so.
    file(MAKE_DIRECTORY "${PACKAGE_DIR}/lib")
    file(COPY "${CMAKE_BINARY_DIR}/install/lib/libtorch_delegate_backend.so"
        DESTINATION "${PACKAGE_DIR}/lib"
    )
    file(MAKE_DIRECTORY "${PACKAGE_DIR}/lib/cmake")
    file(COPY "${CMAKE_BINARY_DIR}/install/lib/cmake/TorchDelegate"
        DESTINATION "${PACKAGE_DIR}/lib/cmake"
    )
    
    # Create zip package
    set(PACKAGE_ZIP "${PACKAGE_OUTPUT_DIR}/${PACKAGE_NAME}.zip")
    file(MAKE_DIRECTORY "${PACKAGE_OUTPUT_DIR}")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar "cf" "${PACKAGE_ZIP}" --format=zip "${PACKAGE_NAME}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/package"
        RESULT_VARIABLE PACKAGE_RESULT
    )
    
    if(PACKAGE_RESULT EQUAL 0)
        message(STATUS "[Package] Package created: ${PACKAGE_ZIP}")
    else()
        message(FATAL_ERROR "[Package] Failed to create package: ${PACKAGE_ZIP}")
    endif()
endfunction()
