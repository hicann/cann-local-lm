cmake_minimum_required(VERSION 3.18)

# ── 镜像环境三方库直连 ──
# 镜像 /home/jenkins/opensource 下三方库已解压好: lib_cache/{json,gtest} + makeself.
# 直接复用本仓 cmake/third_party/modules 的 Find 模块 (扁平布局兜底: find_path/
# find_library 建 IMPORTED target), 不走 cmake-master 的 add_cann_third_party 机制.
# 启用: -DMIRROR_THIRD_PARTY=ON -DCANN_3RD_LIB_PATH=<镜像三方库根 (opensource)>
option(MIRROR_THIRD_PARTY "Use mirror lib_cache third-party directly" OFF)

project(CANN_LM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ── 镜像环境三方库直连 (project 后) ──
# 把本仓 modules Find 模块前置到 CMAKE_MODULE_PATH, 使 find_package(nlohmann_json)
# / find_package(GTest) 走 module 模式直接命中 lib_cache 扁平布局 (include + lib).
# 非镜像 (OFF): 行为不变, 仍走 -Dnlohmann_json_DIR / -DGTest_DIR CONFIG 包路径.
if(MIRROR_THIRD_PARTY)
    list(PREPEND CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/modules")
    if(CANN_3RD_LIB_PATH AND NOT "${CANN_3RD_LIB_PATH}" IN_LIST CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${CANN_3RD_LIB_PATH}")
    endif()
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

# engine UT: tests/cpp/engine (host 端 GTest + stub NDK)
# build_test.sh 经 -DBUILD_TESTS=ON 开启; 须在 add_subdirectory(cann_lm/engine)
# 之后——tests CMakeLists 要求 cann_lm_engine target 已定义
option(BUILD_TESTS "Build engine C++ unit tests (tests/cpp/engine)" OFF)

if(BUILD_ENGINE)
    add_subdirectory(cann_lm/engine)
endif()

if(BUILD_ENGINE AND BUILD_TESTS)
    add_subdirectory(tests/cpp/engine)
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
