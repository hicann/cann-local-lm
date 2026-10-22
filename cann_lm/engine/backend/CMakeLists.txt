# cann_backend 总构建入口：cann_backend_support → hiai_ndk → cann_backend → backend_smoke
# （support 层已拍平折叠进本文件）
# 引擎根 CMakeLists 经 add_subdirectory(backend EXCLUDE_FROM_ALL) 接入；亦可独立配置。
cmake_minimum_required(VERSION 3.16)
project(cann_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(API_FW   ${CMAKE_CURRENT_SOURCE_DIR}/api)
set(EXT_CANN ${CMAKE_CURRENT_SOURCE_DIR}/../external_cann)

# ── 支撑层（buffer/file_util，原 support/ 子目录拍平）──
add_library(cann_backend_support STATIC
    buffer/base_buffer.cpp
    file_util/file_util.cpp
    file_util/file_util_impl.cpp
)
target_include_directories(cann_backend_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/file_util
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer
)
target_include_directories(cann_backend_support SYSTEM PUBLIC
    ${API_FW} ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# securec：host 环境无系统 securec.h 时编入 shim（shim 实体在 engine/infra/securec/）；有则不编（避免遮蔽系统实现）
include(CheckIncludeFile)
check_include_file("securec.h" HAVE_SYSTEM_SECUREC)
if(NOT HAVE_SYSTEM_SECUREC)
    target_sources(cann_backend_support PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../infra/securec/securec.cpp)
    target_include_directories(cann_backend_support SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../infra/securec)
endif()

# ── NDK dlopen 转发层 ──
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../hiai_ndk hiai_ndk_build)

# ── 移植实现 ──
# force-include ohos_compat.h（与 hiai_ascendcl 同机制）：剥离 CANNKit 头的
# Clang `__availability__` 属性，GCC 词法层才能解析 external_cann/ 头。
set(OHOS_COMPAT_OPT -include ${CMAKE_CURRENT_SOURCE_DIR}/../hiai_acl/src/ohos_compat.h)

set(BACKEND_IMPL_SRC
    impl/model_manager/model_manager_factory.cpp
    impl/model_manager/model_manager_ndk_impl.cpp
    impl/model_manager/executor_container.cpp
    impl/built_model/built_model_factory.cpp
    impl/built_model/built_model_ndk_impl.cpp
    impl/built_model/customdata_util.cpp
    impl/tensor/nd_tensor_buffer_factory.cpp
    impl/tensor/nd_tensor_buffer_ndk_impl.cpp
)
add_library(cann_backend STATIC ${BACKEND_IMPL_SRC})
target_compile_options(cann_backend PRIVATE ${OHOS_COMPAT_OPT})
target_include_directories(cann_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/file_util
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer
)
target_include_directories(cann_backend SYSTEM PUBLIC
    ${API_FW} ${EXT_CANN} ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_definitions(cann_backend PUBLIC)
target_link_libraries(cann_backend PUBLIC hiai_ndk cann_backend_support)

# ── smoke ──
add_executable(backend_smoke smoke/backend_smoke.cpp)
target_link_libraries(backend_smoke PRIVATE cann_backend)
target_compile_options(backend_smoke PRIVATE -Wall -Wextra -Wpedantic ${OHOS_COMPAT_OPT})
enable_testing()
add_test(NAME backend_smoke_graceful COMMAND backend_smoke)
