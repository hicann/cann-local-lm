# CANN-LM engine OHOS 交叉编译工具链(自包含版)
#
# 本机 aarch64-linux-glibc,目标 aarch64-linux-ohos(musl),同 CPU 架构。
# 双路线(编译器自动二选一,见下方"编译器"段):
#   A) 完整 SDK(华为 command-line-tools 的 native 子树):llvm/bin/clang 是 ARM-native,
#      本机可直接运行,且自带 builtins/crt resource dir(无需手动补 /usr/lib/clang/18)。
#   B) 精简 SDK(OpenHarmony 镜像提取,无 llvm/bin):回退本机 clang-18 + 手动补 builtins。
# flag 设置照搬官方 native/build/cmake/ohos.toolchain.cmake;libc++ 头/库显式指定
# (libcxx-ohos 头 + -L llvm/lib/aarch64-linux-ohos),两路线通用。
#
# 用法(由 build_ohos.sh 调用):
#   cmake -G "Unix Makefiles" \
#     -DCMAKE_TOOLCHAIN_FILE=<this> \
#     -DOHOS_SDK_NATIVE=<.../native 绝对路径> \
#     -DOHOS_ARCH=arm64-v8a <source-dir>

cmake_minimum_required(VERSION 3.16)

if(DEFINED _CANNLM_OHOS_TOOLCHAIN_DEFINED)
  return()
endif()
set(_CANNLM_OHOS_TOOLCHAIN_DEFINED true)

set(CMAKE_SYSTEM_NAME      OHOS)
set(CMAKE_SYSTEM_VERSION   1)

# ── 路径 ──────────────────────────────────────────────────────────
set(OHOS_SDK_NATIVE "" CACHE PATH "Path to extracted OHOS SDK native/ dir (sysroot/ llvm/ ...)")
if(NOT OHOS_SDK_NATIVE OR NOT IS_DIRECTORY "${OHOS_SDK_NATIVE}")
  message(FATAL_ERROR "OHOS_SDK_NATIVE not set/invalid: '${OHOS_SDK_NATIVE}'")
endif()
if(NOT IS_DIRECTORY "${OHOS_SDK_NATIVE}/sysroot")
  message(FATAL_ERROR "no sysroot under OHOS_SDK_NATIVE: ${OHOS_SDK_NATIVE}/sysroot")
endif()

# ── arch / 目标三元组 ───────────────────────────────────────────
set(OHOS_ARCH "arm64-v8a" CACHE STRING "OHOS target arch")
if(OHOS_ARCH STREQUAL arm64-v8a)
  set(_OHOS_TRIPLE aarch64-linux-ohos)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
else()
  message(FATAL_ERROR "unsupported OHOS_ARCH=${OHOS_ARCH} (only arm64-v8a wired)")
endif()

# ── 编译器:优先 SDK 自带 clang(ARM-native,自带 builtins/crt resource dir) ──
# 无则回退本机 clang-18(精简 SDK 路线,需手动补 builtins 到 /usr/lib/clang/18)
if(EXISTS "${OHOS_SDK_NATIVE}/llvm/bin/clang" AND EXISTS "${OHOS_SDK_NATIVE}/llvm/bin/clang++")
  set(_OHOS_CC  "${OHOS_SDK_NATIVE}/llvm/bin/clang")
  set(_OHOS_CXX "${OHOS_SDK_NATIVE}/llvm/bin/clang++")
  set(_OHOS_TOOLCHAIN_SRC "SDK clang(ARM-native)")
else()
  find_program(CLANG_18_BIN clang-18)
  find_program(CLANG_18_CXX_BIN clang++-18)
  if(NOT CLANG_18_BIN OR NOT CLANG_18_CXX_BIN)
    message(FATAL_ERROR "no compiler: SDK 无 ${OHOS_SDK_NATIVE}/llvm/bin/clang,且本机无 clang-18 —— 用完整 SDK 或 apt install clang-18")
  endif()
  set(_OHOS_CC  ${CLANG_18_BIN})
  set(_OHOS_CXX ${CLANG_18_CXX_BIN})
  set(_OHOS_TOOLCHAIN_SRC "本机 clang-18(精简 SDK 路线)")
endif()
set(CMAKE_C_COMPILER   ${_OHOS_CC})
set(CMAKE_CXX_COMPILER ${_OHOS_CXX})
# 让 clang 去 SDK 的 llvm 下找 builtins/crt(等价官方 EXTERNAL_TOOLCHAIN;
# SDK 自带 clang-15 时其 resource dir 在 llvm/lib/clang/15.0.4/,本段冗余但无害)
set(CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN   "${OHOS_SDK_NATIVE}/llvm")
set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "${OHOS_SDK_NATIVE}/llvm")
set(CMAKE_ASM_COMPILER_EXTERNAL_TOOLCHAIN "${OHOS_SDK_NATIVE}/llvm")
set(CMAKE_C_COMPILER_TARGET   ${_OHOS_TRIPLE})
set(CMAKE_CXX_COMPILER_TARGET ${_OHOS_TRIPLE})
set(CMAKE_ASM_COMPILER_TARGET ${_OHOS_TRIPLE})

# 透传给 try_compile(否则 ABI 检测时 OHOS_SDK_NATIVE 为空)
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  OHOS_SDK_NATIVE OHOS_ARCH _OHOS_TRIPLE)

# ── sysroot / 库搜索 ────────────────────────────────────────────
set(CMAKE_SYSROOT "${OHOS_SDK_NATIVE}/sysroot")
set(CMAKE_LIBRARY_ARCHITECTURE "${_OHOS_TRIPLE}")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "/usr/lib/${_OHOS_TRIPLE}")
set(CMAKE_FIND_ROOT_PATH "${OHOS_SDK_NATIVE}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ── 编译 flags(照搬官方) ──────────────────────────────────────
set(_OHOS_C_FLAGS
  -fdata-sections -ffunction-sections -funwind-tables
  -fstack-protector-strong -no-canonical-prefixes -fno-addrsig
  -Wa,--noexecstack -Wformat -Werror=format-security)
string(REPLACE ";" " " _OHOS_C_FLAGS "${_OHOS_C_FLAGS}")

# 用 *_FLAGS_INIT 而非 "CACHE ... FORCE" 自引用:CMake 仅在缓存项不存在(首次配置)时
# 用 _INIT 初始化 CMAKE_C_FLAGS;重配时保留缓存值、不再注入 → 避免每次 configure
# 叠加一份 flag(build_ohos.sh 复用 build 目录、每次全配置,自引用 + FORCE 会线性累积)。
set(CMAKE_C_FLAGS_INIT   "${_OHOS_C_FLAGS} -D__MUSL__")
set(CMAKE_ASM_FLAGS_INIT "${_OHOS_C_FLAGS} -D__MUSL__")

# C++ 用 OHOS NDK 的 libc++ 头(非宿主 clang 自带的,避免 ABI 不匹配);
# libc++ 路径并入 CMAKE_CXX_FLAGS_INIT(同一 _INIT 变量累加,仍是首次注入)
set(CMAKE_CXX_FLAGS_INIT "${_OHOS_C_FLAGS} -D__MUSL__ -stdlib=libc++ -nostdinc++ -isystem ${OHOS_SDK_NATIVE}/llvm/include/libcxx-ohos/include/c++/v1")

set(CMAKE_C_FLAGS_DEBUG   "-O0 -g -fno-limit-debug-info" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-limit-debug-info" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE   "-O2 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)

# ── 链接 flags(照搬官方) ────────────────────────────────────────
set(_OHOS_LINK
  "-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos;--rtlib=compiler-rt;-fuse-ld=lld;-Wl,--build-id=sha1;-Wl,--warn-shared-textrel;-Wl,--fatal-warnings;-lunwind;-Wl,--no-undefined;-Qunused-arguments;-Wl,-z,noexecstack")
string(REPLACE ";" " " _OHOS_LINK "${_OHOS_LINK}")
set(CMAKE_SHARED_LINKER_FLAGS "${_OHOS_LINK}" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS  "${_OHOS_LINK}" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS     "${_OHOS_LINK} -Wl,--gc-sections" CACHE STRING "" FORCE)

set(CMAKE_C_STANDARD_LIBRARIES_INIT   "-lm")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "-lm")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# ── ar/ranlib(归档静态库用,如 gtest) ────────────────────────────
# 优先 SDK 自带 llvm-ar(ARM-native,与 SDK clang 配套);次选本机 llvm-ar-18。
# (旧精简 SDK 无 llvm/bin/llvm-ar——当时注释"x86_64 宿主二进制跑不了"已不适用:
#  完整 SDK 的 llvm/bin/llvm-ar 是 ARM-native,本机可直接运行)
if(EXISTS "${OHOS_SDK_NATIVE}/llvm/bin/llvm-ar")
  set(CMAKE_AR "${OHOS_SDK_NATIVE}/llvm/bin/llvm-ar" CACHE FILEPATH "" FORCE)
else()
  find_program(LLVM_AR_18 llvm-ar-18)
  if(LLVM_AR_18)
    set(CMAKE_AR "${LLVM_AR_18}" CACHE FILEPATH "" FORCE)
  else()
    message(FATAL_ERROR "no llvm-ar: SDK 无 ${OHOS_SDK_NATIVE}/llvm/bin/llvm-ar 且本机无 llvm-ar-18")
  endif()
endif()
if(EXISTS "${OHOS_SDK_NATIVE}/llvm/bin/llvm-ranlib")
  set(CMAKE_RANLIB "${OHOS_SDK_NATIVE}/llvm/bin/llvm-ranlib" CACHE FILEPATH "" FORCE)
else()
  find_program(LLVM_RANLIB_18 llvm-ranlib-18)
  if(LLVM_RANLIB_18)
    set(CMAKE_RANLIB "${LLVM_RANLIB_18}" CACHE FILEPATH "" FORCE)
  else()
    message(FATAL_ERROR "no llvm-ranlib: SDK 无且本机无 llvm-ranlib-18")
  endif()
endif()

add_compile_definitions(__OHOS__)
message(STATUS "[CANN-LM OHOS toolchain] triple=${_OHOS_TRIPLE} sysroot=${CMAKE_SYSROOT} cc=${_OHOS_CC} (${_OHOS_TOOLCHAIN_SRC})")
