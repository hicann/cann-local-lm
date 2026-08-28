# ============================================================================
# ohos-aarch64.cmake — OHOS (鸿蒙) aarch64 交叉编译 toolchain [预留 stub]
#
# 状态: 不可用 (注释化模板占位). 待 OHOS NDK 路径 / 目标产物形态确定后补全.
# 用法 (待激活):
#   cmake -S <src> -B build -DCMAKE_TOOLCHAIN_FILE=<本路径> \
#         -DOHOS_NDK_ROOT=/path/to/ohos/ndk
# ============================================================================

# ---- 激活时取消注释并按实际 OHOS NDK 路径调整 ----

# set(CMAKE_SYSTEM_NAME      OHOS)
# set(CMAKE_SYSTEM_PROCESSOR aarch64)
#
# set(OHOS_NDK_ROOT "" CACHE PATH "Path to OHOS NDK")
# if(OHOS_NDK_ROOT STREQUAL "")
#     message(FATAL_ERROR "OHOS_NDK_ROOT not set")
# endif()
#
# set(_TOOLCHAIN_PREFIX aarch64-linux-ohos)
# set(CMAKE_C_COMPILER   "${OHOS_NDK_ROOT}/bin/${_TOOLCHAIN_PREFIX}-clang")
# set(CMAKE_CXX_COMPILER "${OHOS_NDK_ROOT}/bin/${_TOOLCHAIN_PREFIX}-clang++")
# set(CMAKE_SYSROOT      "${OHOS_NDK_ROOT}/sysroot")
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
#
# # 鸿蒙目标产物形态 (so/om/hap) 确定后, 在此补充链接选项与产物规则
