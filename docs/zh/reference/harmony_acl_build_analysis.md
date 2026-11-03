# harmony_acl 构建体系分析(参考文档)

简体中文 | [English](../../en/reference/harmony_acl_build_analysis.md)

> 来源项目:`/mnt/workspace/cann-recipes-harmony-infer/harmony_acl`
> 分析日期:2026-08-24
> 用途:作为 CANN-LM `engine/` 移植到鸿蒙 PC(编译为 .so / 可执行 UT)的参考范本。
> 本文档自包含——未来会话冷启动加载即可理解,无需回看源项目。

---

## 1. harmony_acl 是什么

华为 `cann-recipes-harmony-infer` 仓库下的子项目。提供与昇腾平台统一的 **ACL(AscendCL)接口**,使开发者基于同一套接口在**鸿蒙(OHOS)端侧**部署模型推理。核心产物:

- `libhiai_ascendcl.so` —— 封装 ACL 接口的动态库
- `ascendcl_ut` —— GTest 单元测试可执行文件
- 同时支持 **OHOS / Linux / Windows** 三平台编译(同一份 `CMakeLists.txt` 用分支切平台)

这正好是 CANN-LM engine 想要的形态:把 engine 编成 .so + 可执行 UT,且能在鸿蒙 PC 跑。

---

## 2. 目录结构

```
harmony_acl/
├── build.sh                 # 主构建入口(OHOS 交叉编译 + 依赖头文件下载)
├── build_ut.sh / build_ut.bat # Windows/MinGW UT 构建(独立于 build.sh)
├── ohos.toolchain.cmake     # OHOS 交叉编译工具链文件(核心)
├── README.md                # 前置条件 + 编译/运行说明
├── scripts/
│   ├── CMakeLists.txt       # 真正的构建逻辑(三平台分支)
│   └── CMakeLists.txt.back  # 历史版本(对照可看演进)
├── src/                     # 库源码
│   ├── acl_*.cpp/.h         # ACL 接口实现
│   ├── ascend_string.cpp
│   ├── ndk/                 # 鸿蒙 NDK 层:动态加载、hiai_* 代理
│   └── utils/               # log.h / assert.h / securestl.h 等工具
├── tests/cpp/engine/                # GTest 测试
│   ├── main.cpp             # 测试 main
│   ├── aippParam_ut.cpp
│   ├── compiler_ut.cpp
│   └── ndk/                 # NDK 测试桩 + 头文件
├── include/                 # (构建时生成)ACL/graph 头文件落地处
├── depends/                 # (构建时填充)下载的依赖头文件包
└── build_ohos/              # (产物)libhiai_ascendcl.so / ascendcl_ut / libc++_shared.so
```

要点:`include/` 和 `depends/` 是 **build.sh 运行时生成/填充**的,不在仓库里静态存在。

---

## 3. OHOS 交叉编译工具链(`ohos.toolchain.cmake`)

这是移植到鸿蒙的**最关键文件**。33 行,逐项解读:

```cmake
set(CMAKE_SYSTEM_NAME OHOS)            # 目标系统=OpenHarmony(非 Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)    # 目标 CPU=ARM64

# 强制目标三元组,切断对宿主(Windows/Linux)库的依赖
set(CMAKE_C_COMPILER_TARGET "aarch64-linux-ohos")
set(CMAKE_CXX_COMPILER_TARGET "aarch64-linux-ohos")

set(OHOS_SDK "<SDK路径>" CACHE PATH "OpenHarmony SDK path")
set(CMAKE_C_COMPILER   "${OHOS_SDK}/native/llvm/bin/clang")     # 鸿蒙 SDK 自带 clang
set(CMAKE_CXX_COMPILER "${OHOS_SDK}/native/llvm/bin/clang++")
set(CMAKE_SYSROOT "${OHOS_SDK}/native/sysroot")                 # 系统根(头+基础库)

set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-ohos")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "/usr/lib/aarch64-linux-ohos")

# find 模式:只在 sysroot 里找库/头,绝不用宿主的
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
list(APPEND CMAKE_FIND_ROOT_PATH "${OHOS_SDK}/native")

# 强制用鸿蒙 SDK 自带的 lld 链接器
set(CMAKE_EXE_LINKER_FLAGS   "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

# 鸿蒙必要的编译宏
add_compile_definitions(__OHOS__ __MUSL__=1)
```

**关键启示**:
1. 用 **SDK 自带的 clang/clang++**(非系统 gcc),目标三元组 `aarch64-linux-ohos`。
2. **sysroot** 指向 SDK 的 `native/sysroot`,所有头/库只从 sysroot 找(`ONLY` 模式),避免污染宿主库。
3. 鸿蒙用 **musl libc**(故 `__MUSL__=1`),与 Linux glibc 不同。
4. 链接器强制 **lld**。
5. `__OHOS__` 宏是源码里 `#ifdef __OHOS__` 平台分支的开关——源码据此切鸿蒙路径。

> 注:`build.sh` 里实际用的是 **SDK 官方 toolchain** `${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake`(通过 `-D CMAKE_TOOLCHAIN_FILE=...`),仓库根的 `ohos.toolchain.cmake` 是一份手写简化/对照版。两者二选一,官方版更全。

---

## 4. `build.sh` 主构建流程(155 行)

### 4.1 依赖头文件自动下载(`auto_download_header_with_git` + `deal_after_download`)

用 git sparse-checkout 从 gitcode 拉取 CANN 三个仓的头文件:

| 仓 | sparse-checkout 路径 | 落地到 |
|----|---------------------|--------|
| `cann/runtime` | `include/external/acl` | `include/acl/` |
| `cann/ge` | `inc` | `include/acl/` + `include/graph/`(挑选特定 .h) |
| `cann/metadef` | `inc` | `include/graph/`(挑选特定 .h) |
| `js/json` (nlohmann) | 全量 | `depends/nlohmann/` |

若 git 失败,README 给出手动下载 zip 放 `depends/` 的回退路径。脚本同时识别 zip 包优先解压、否则用 clone 结果。

**启示**:CANN-LM 若依赖 CANN 头文件,可复用这套 sparse-checkout 自动拉取机制,不必把头文件入库。

### 4.2 cmake 配置与编译

```bash
CMD_LINE_TOOLS_DIR=M:\\harmony-pc\\command-line-tools    # ← 唯一需改的环境变量
OHOS_SDK=${CMD_LINE_TOOLS_DIR}/sdk/default/openharmony/
OHOS_INC_DIR=${OHOS_SDK}/native/sysroot/usr/include/
HMS_INC_DIR=${CMD_LINE_TOOLS_DIR}/sdk/default/hms/native/sysroot/usr/include/

cmake -G "Ninja" \
  -D OHOS_INC_DIR=${OHOS_INC_DIR} \
  -D HMS_INC_DIR=${HMS_INC_DIR} \
  -D OHOS_STL=c++_shared \
  -D OHOS_ARCH=arm64-v8a \
  -D OHOS_PLATFORM=OHOS \
  -D CMAKE_TOOLCHAIN_FILE=${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake \
  ../scripts
cmake --build . -j
```

**启示**:
- `CMD_LINE_TOOLS_DIR` 是华为 **command-line-tools** (native 开发环境)的解压目录,构建前必须改这一行。
- 传给官方 toolchain 的标准变量:`OHOS_STL`(c++_shared/static)、`OHOS_ARCH`(arm64-v8a)、`OHOS_PLATFORM=OHOS`。
- 用 **Ninja** 生成器。
- 产物落在 `build/` 下(脚本里 `mkdir build && cd build`)。注意 README 说 UT 在 `build/ut`,而 build.sh 里写的是 `build/` —— 存在文档与脚本不一致的小 bug。

---

## 5. `scripts/CMakeLists.txt` 三平台构建逻辑(246 行,核心)

结构:`add_library(hiai_ascendcl SHARED ...)` 生成主 .so,然后按平台分支处理 UT 与桩库。

### 5.1 主库(所有平台共用)

```cmake
add_library(hiai_ascendcl SHARED ${ACL_SRC})   # 源码 = src/*.cpp + src/ndk/*.cpp
if(GNU OR Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3")   # 注意:此行原在分支前,Windows 报错(见§7)
```

### 5.2 OHOS 分支(`OHOS_PLATFORM STREQUAL "OHOS"`)

```cmake
# RPATH 设为 $ORIGIN,让 .so 在自身目录找依赖
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

find_library(HILOG_LIB NAMES hilog_ndk.z)            # 鸿蒙日志库(必需)
target_link_libraries(hiai_ascendcl PUBLIC ${HILOG_LIB})

# 鸿蒙系统库 neural_network_core(IMPORTED)
find_library(NNCORE_LIB NAMES neural_network_core)
add_library(neural_network_core SHARED IMPORTED)
set_target_properties(neural_network_core PROPERTIES IMPORTED_LOCATION "${NNCORE_LIB}")

# HMS 侧 libhiai_foundation.so(IMPORTED,路径由 -DHMS_LIB_DIR 传入)
add_library(hiai_foundation SHARED IMPORTED)
set_target_properties(hiai_foundation PROPERTIES IMPORTED_LOCATION "${HMS_LIB_DIR}/libhiai_foundation.so")

# GTest:本地源码 FetchContent(scripts/googletest-1.14.0/)
FetchContent_Declare(googletest SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest-1.14.0)
FetchContent_MakeAvailable(googletest)

# UT 可执行
add_executable(ascendcl_ut ${UT_TEST_SRC})
target_link_libraries(ascendcl_ut PUBLIC hiai_ascendcl neural_network_core hiai_foundation gtest gtest_main ${HILOG_LIB})

# ★ 鸿蒙自动签名(POST_BUILD):binary-sign-tool.jar 对 .so / ut / libc++_shared.so 自签名
```

**鸿蒙特有要点**:
- 链接 `hilog_ndk.z`(鸿蒙日志)。
- 系统库 `neural_network_core` 通过 `find_library` 在 sysroot 找。
- 第三方 `.so`(`libhiai_foundation.so`)用 **IMPORTED target** + 路径变量传入。
- GTest 用**本地源码** FetchContent(鸿蒙环境联网受限,不能 GIT_REPOSITORY)。
- **必须签名**:`binary-sign-tool.jar` + java,对 `ascendcl_ut`、`libhiai_ascendcl.so`、`libc++_shared.so` 做 selfSign。产物名加 `_signed` 后缀。无签名无法在鸿蒙设备运行。
- 拷贝 `libc++_shared.so` 到产物目录(运行时 C++ 运行库)。

### 5.3 Linux 分支(`UNIX AND NOT APPLE`)

无鸿蒙系统库,改为**自建桩库**模拟 `hiai_foundation` / `neural_network_core`(用 `tests/cpp/` 下的桩源码编译成 .so),再链 UT。GTest 同样本地源码 FetchContent。用于主机端跑 UT。

### 5.4 Windows 分支(`WIN32`)

- `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON`(DLL 默认不导出符号,必须开)。
- `CMAKE_SHARED_LIBRARY_PREFIX ""`(去 MinGW 的 lib 前缀)。
- MSVC:`_CRT_SECURE_NO_WARNINGS NOMINAX UNICODE _UNICODE` + `/utf-8`。
- GTest 用 `GIT_REPOSITORY`(Windows 联网 OK),tag `release-1.12.1`。
- 同样自建桩库。

### 5.5 UT 构建(`build_ut.sh` / `build_ut.bat`)

**独立于 `build.sh`**,专为 Windows 主机端跑 UT:
- `build_ut.bat` 调 MSYS2 UCRT64 环境,执行 `build_ut.sh`。
- `build_ut.sh` 用 **MinGW**(`-G "MinGW Makefiles"`,gcc/g++),输出到 `build_ut/`,只走 CMakeLists 的 WIN32 分支。
- 与 OHOS 的关系:OHOS UT 由 `build.sh` 走 OHOS 分支产出(签名后在鸿蒙跑);Windows UT 由 `build_ut.*` 走 WIN32 分支产出(主机跑)。

---

## 6. 鸿蒙 PC 执行环境要求(运行时)

从 README + CMakeLists 提炼,在鸿蒙 PC 上跑产物需要:

| 项 | 说明 |
|----|------|
| **签名** | `.so` 与可执行文件必须用 `binary-sign-tool.jar -selfSign` 签名,否则系统拒绝加载 |
| **libc++_shared.so** | 鸿蒙 C++ 运行库,需拷贝到产物目录(已用 `$ORIGIN` RPATH 就近找) |
| **LD_LIBRARY_PATH** | README 里 `export LD_LIBRARY_PATH=$(pwd)` —— 让可执行找到同目录 .so |
| **系统库路径** | `hilog_ndk.z`、`neural_network_core` 等在鸿蒙系统 `/system/lib64/ndk/` 下,设备自带 |
| **动态加载** | `src/ndk/ndk_proxy.cpp` 在运行时 dlopen `libhiai_foundation.so` 等,路径按 `__OHOS__` 宏切到 `/system/lib64/ndk/` |

---

## 7. 跨平台适配的经验坑(从 `windows_ut_adaptation_spec.md` 提炼)

这份 117 行 spec 记录了把 UT 适配到 Windows 时踩的坑,对 CANN-LM 未来跨鸿蒙/Linux/Windows 同样适用。归类:

### P0 编译阻断
- **POSIX 专有头文件**:`<dirent.h>` 在非 POSIX 平台不存在,未用就删。
- **编译器专有标志**:`-O0 -g3` 是 GCC/Clang 专有,MSVC 报错。要 `if(GNU OR Clang)` 包住,放在平台分支**之后**。
- **硬编码平台后缀**:`.so` vs `.dll`——动态加载处用 `#ifdef __OHOS__ / _WIN32 / else` 三分支。
- **路径分隔符逻辑**:`find("/") == npos || find("\\") == npos` 应是 `&&`(逻辑 bug)。

### P1 链接/运行时失败
- **Windows DLL 符号导出**:`CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` 必须在 WIN32 分支开头。
- **MSVC 定义覆盖所有 target**:不是只给 UT,库 target 也要(`strncpy` 等触发 C4996)。

### P2 警告/潜在
- 重复宏定义(MSVC C4005)。
- `float32_t`(C++23 `<stdfloat>`)兼容性差,改 `float`。

**通用规律**:源码里所有**平台相关**行为都用 `__OHOS__` / `_WIN32` / `else(Linux)` 三段宏隔离;CMake 里用 `OHOS_PLATFORM`/`UNIX`/`WIN32` 三分支隔离。这是跨鸿蒙+主机编译的核心模式。

---

## 8. 对 CANN-LM engine 的可借鉴点(预告 PLAN)

对照本项目 `cann_local_lm/engine/CMakeLists.txt`(现状:单一 Linux/GCC、STATIC 库、`find_package(GTest REQUIRED)` 主机装、无平台分支):

| harmony_acl 做法 | CANN-LM 现状 | 借鉴方向 |
|------------------|-------------|---------|
| `ohos.toolchain.cmake` + `-DCMAKE_TOOLCHAIN_FILE` 交叉编译 | 无 toolchain,纯本机构化 | 引入 OHOS toolchain 文件 |
| `add_library(... SHARED ...)` 产 .so | `add_library(cann_lm_engine STATIC ...)` | 加 OHOS 分支产 SHARED .so |
| 三平台分支(OHOS/Linux/Windows) | 单分支 | 加 `if(OHOS_PLATFORM)/elseif(UNIX)/elseif(WIN32)` |
| GTest 本地源码 FetchContent(鸿蒙联网受限) | `find_package(GTest REQUIRED)` 主机装 | OHOS 分支改本地 FetchContent |
| 鸿蒙自动签名 + 拷 libc++_shared.so | 无 | OHOS 产物需签名脚本 |
| `$ORIGIN` RPATH | 无 | 设 RPATH 便于就近找 .so |
| `hilog_ndk.z` / `__OHOS__` / `__MUSL__` | 无平台宏 | 源码需审 POSIX 依赖、加平台宏 |
| 依赖头文件 sparse-checkout 拉取 | 无 CANN 头依赖(目前) | 若日后接 CANN runtime,复用机制 |

**最大差距**:CANN-LM engine 现在是 **STATIC 库 + 主机 GCC + 主机 GTest**,完全没有交叉编译/平台分支/签名链。要上鸿蒙 PC,需新增一套 OHOS toolchain 配置 + CMake 平台分支 + 签名脚本,并审计源码里的 POSIX 专有依赖。详见后续 PLAN 文档的兼容性矩阵。

---

## 9. 一页速查

```
工具链:  OHOS SDK native/llvm/clang, aarch64-linux-ohos, sysroot, lld
编译器宏: __OHOS__ __MUSL__=1
STL:     c++_shared(需拷 libc++_shared.so 到产物目录)
主库:    add_library(hiai_ascendcl SHARED ...)
UT:      add_executable(ascendcl_ut ...), 链 gtest + 主库 + IMPORTED 系统so
GTest:   本地源码 FetchContent(ohos/linux); GIT_REPOSITORY(windows)
签名:    binary-sign-tool.jar -selfSign 1, 产物 *_signed
运行:    export LD_LIBRARY_PATH=$(pwd); ./ascendcl_ut
平台隔离: 源码 __OHOS__/_WIN32/else; CMake OHOS_PLATFORM/UNIX/WIN32
坑:      POSIX 头(dirent)、GCC 标志条件化、.so/.dll 后缀、Windows 符号导出
```
