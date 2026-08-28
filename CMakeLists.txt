cmake_minimum_required(VERSION 3.18)
project(cann_lm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(CANN_LM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# 本仓 cmake/modules 自包含: Find*.cmake + DelegateCommonFlags + TorchDelegateCannCommon
# (不再依赖 sibling torch-delegate-harmony/cmake 的同名模块)
list(PREPEND CMAKE_MODULE_PATH "${CANN_LM_ROOT}/cmake/modules")
include(DelegateCommonFlags)
include(TorchDelegateCannCommon)
torch_delegate_cann_include_search_paths(CANN_INCLUDE_DIRS)
torch_delegate_cann_library_search_paths(CANN_LIB_DIRS)

# ── 构件开关 (默认全开; 纯 backend 用 -DBUILD_COMPILE_TOOL=OFF -DBUILD_TESTS=OFF) ──
# cann_lm_compile (epair→OM 工具) 由 csrc/CMakeLists 的 BUILD_COMPILE_TOOL 构建 (并入 csrc)
option(BUILD_BACKEND       "Build torch_delegate_backend (csrc) lib"     ON)
option(BUILD_COMPILE_TOOL  "Build cann_lm_compile (epair→OM) tool"       ON)
option(BUILD_TESTS         "Build C++ tests (tests/c++/epair_archive)"   ON)

# ── backend lib + cann_lm_compile 工具 (均由 csrc/CMakeLists 提供, 不依赖 torch) ──
if(BUILD_BACKEND OR BUILD_COMPILE_TOOL)
    add_subdirectory(cann_lm/converter/export/csrc)
endif()

# ── C++ tests ──
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/c++/epair_archive)
endif()
