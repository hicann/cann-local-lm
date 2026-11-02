# examples/test_engine/CMakeLists.txt — smoke + bbit（独立或子目录构建）
#
# 两种模式:
#   1. 子目录模式（ohos/CMakeLists.txt add_subdirectory）: cann_lm_engine target 已定义
#   2. 独立模式（build.sh --smoke）: 通过 CANN_LM_ENGINE_LIB 找预编 .so
#
# 独立构建用法:
#   cmake -S examples/test_engine -B build_smoke \
#       -DCMAKE_TOOLCHAIN_FILE=<ohos.toolchain.cmake> \
#       -DCANN_LM_ENGINE_LIB=<build_ohos>/libcann_lm_engine.so \
#       -DCANN_LM_ENGINE_INCLUDE=<repo>/cann_lm/engine
#   cmake --build build_smoke -j$(nproc)

cmake_minimum_required(VERSION 3.18)

if(NOT TARGET cann_lm_engine)
    # 独立模式：找预编 .so + include
    project(test_engine_smoke C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if(NOT CANN_LM_ENGINE_LIB)
        message(FATAL_ERROR "CANN_LM_ENGINE_LIB not set (point to libcann_lm_engine.so)")
    endif()
    if(NOT CANN_LM_ENGINE_INCLUDE)
        message(FATAL_ERROR "CANN_LM_ENGINE_INCLUDE not set (point to cann_lm/engine/)")
    endif()

    add_library(cann_lm_engine SHARED IMPORTED)
    set_target_properties(cann_lm_engine PROPERTIES IMPORTED_LOCATION "${CANN_LM_ENGINE_LIB}")
    set(ENGINE_INCLUDE_DIR ${CANN_LM_ENGINE_INCLUDE})
    # backend include dirs（subdirectory 模式下由 cann_backend PUBLIC 传播）
    set(ENGINE_BACKEND_INCLUDES
        ${CANN_LM_ENGINE_INCLUDE}/backend/api
        ${CANN_LM_ENGINE_INCLUDE}/backend
        ${CANN_LM_ENGINE_INCLUDE}/external_cann
        ${CANN_LM_ENGINE_INCLUDE}/hiai_acl/src
    )
else()
    # 子目录模式：target 已定义
    set(ENGINE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cann_lm/engine)
    set(ENGINE_BACKEND_INCLUDES "")
endif()

# nlohmann/json: find_package (Findnlohmann_json.cmake → output/third_party/json)
# bbit/golden_verify 解析 testcase/meta_json 需要; 独立模式自寻,
# 子目录模式复用父级 (ohos/CMakeLists.txt) 已建的 nlohmann_json target
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/third_party/modules")
set(CANN_3RD_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../output/third_party" CACHE PATH "Third-party libraries directory")
find_package(nlohmann_json REQUIRED)

# ── smoke ──
add_executable(api_smoke smoke/api_smoke.cpp)
target_link_libraries(api_smoke PRIVATE cann_lm_engine)
target_include_directories(api_smoke PRIVATE ${ENGINE_INCLUDE_DIR} ${ENGINE_BACKEND_INCLUDES})
target_compile_options(api_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(c_api_smoke smoke/c_api_smoke.c)
target_link_libraries(c_api_smoke PRIVATE cann_lm_engine)
target_include_directories(c_api_smoke PRIVATE ${ENGINE_INCLUDE_DIR} ${ENGINE_BACKEND_INCLUDES})
target_compile_options(c_api_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(golden_verify smoke/golden_verify.cpp)
target_link_libraries(golden_verify PRIVATE cann_lm_engine nlohmann_json::nlohmann_json)
target_include_directories(golden_verify PRIVATE ${ENGINE_INCLUDE_DIR} ${ENGINE_BACKEND_INCLUDES})
target_compile_options(golden_verify PRIVATE -Wall -Wextra -Wpedantic)

# ── bbit: 黑盒集成测试 ──
add_executable(bbit bbit/bbit.cpp)
target_link_libraries(bbit PRIVATE cann_lm_engine nlohmann_json::nlohmann_json)
target_include_directories(bbit PRIVATE ${ENGINE_INCLUDE_DIR} ${ENGINE_BACKEND_INCLUDES})
target_compile_options(bbit PRIVATE -Wall -Wextra -Wpedantic)
