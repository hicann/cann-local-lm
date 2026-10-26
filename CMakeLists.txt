cmake_minimum_required(VERSION 3.18)
project(CANN_LM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 顶层启用 CTest：engine/ 与 converter/ 子目录内的 enable_testing() 只作用于子目录层级，
# 顶层注册一次以让 `ctest --test-dir build` 发现全部测试。
enable_testing()

# ──────────────────────────────────────────────────────────────
# 构件开关 (默认: engine 构建为主, converter 需 torch+CANN 环境按需开启)
#   cmake -B build -DBUILD_CONVERTER=ON   # 同时构建 converter/csrc + tests
# ──────────────────────────────────────────────────────────────

# engine: C++ 推理引擎 (cann_lm/engine, fork 主线, 不依赖 torch)
option(BUILD_ENGINE "Build cann_lm C++ engine (cann_lm/engine)" ON)

# converter: epair→OM 工具 + backend lib (cann_lm/converter/export/csrc)
#   依赖 cmake/modules 的 DelegateCommonFlags/TorchDelegateCannCommon + torch + CANN
option(BUILD_CONVERTER "Build cann_lm converter/export/csrc + tests (needs torch+CANN)" OFF)

if(BUILD_ENGINE)
    add_subdirectory(cann_lm/engine)
endif()

if(BUILD_CONVERTER)
    get_filename_component(CANN_LM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

    # 本仓 cmake/modules 自包含: Find*.cmake + DelegateCommonFlags + TorchDelegateCannCommon
    # (不再依赖 sibling torch-delegate-harmony/cmake 的同名模块)
    list(PREPEND CMAKE_MODULE_PATH "${CANN_LM_ROOT}/cmake/modules")
    include(DelegateCommonFlags)
    include(TorchDelegateCannCommon)
    torch_delegate_cann_include_search_paths(CANN_INCLUDE_DIRS)
    torch_delegate_cann_library_search_paths(CANN_LIB_DIRS)

    # ── backend lib + cann_lm_compile 工具 (均由 csrc/CMakeLists 提供, 不依赖 torch) ──
    add_subdirectory(cann_lm/converter/export/csrc)

    # ── C++ tests (tests/cpp/epair_archive; converter C++ UT, 需 CANN 工具链, 无 NPU 设备) ──
    add_subdirectory(tests/cpp/epair_archive)
endif()
