# -----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for building third-party libraries
#
# This CMake script builds all third-party dependencies:
#   - nlohmann_json 3.11.3 - gitcode mirror
#   - GoogleTest 1.14.0 - gitcode mirror
#
# libtorch is now obtained from Python environment (venv) or user-specified Torch_DIR.
#
# Usage:
#   cmake -B build -S . -DCMAKE_THIRD_PARTY_LIB_DIR=/path/to/install
#   cmake --build build
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(third_party_builder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default third-party installation directory
if(NOT CMAKE_THIRD_PARTY_LIB_DIR)
    set(CMAKE_THIRD_PARTY_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../output/third_party")
endif()

message(STATUS "Third-party installation directory: ${CMAKE_THIRD_PARTY_LIB_DIR}")

# Create directory if not exists
if(NOT EXISTS "${CMAKE_THIRD_PARTY_LIB_DIR}")
    file(MAKE_DIRECTORY "${CMAKE_THIRD_PARTY_LIB_DIR}")
endif()

# Add modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Build all third-party libraries (libtorch is now from Python environment)
include(Findnlohmann_json)
include(FindGTest)
include(Findmakeself)
include(Findlibtorch_npu)

# Summary
message(STATUS "")
message(STATUS "Third-party libraries configuration:")
message(STATUS "  Installation directory: ${CMAKE_THIRD_PARTY_LIB_DIR}")
message(STATUS "  nlohmann_json: ${CMAKE_THIRD_PARTY_LIB_DIR}/json")
message(STATUS "  gtest:         ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest")
message(STATUS "  makeself:      ${CMAKE_THIRD_PARTY_LIB_DIR}/makeself")
message(STATUS "  libtorch_npu:  ${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch_npu")
message(STATUS "")
message(STATUS "Note: libtorch is obtained from Python environment (venv) or Torch_DIR")
