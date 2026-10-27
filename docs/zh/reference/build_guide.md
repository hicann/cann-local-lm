# CANN-LM 编译与验证指南

简体中文 | [English](../../en/reference/build_guide.md)

> 覆盖两种环境:**Linux 本机构建**(开发/CI/主机 UT)与 **鸿蒙(OHOS)交叉编译**(产 `libcann_lm_engine.so` 供鸿蒙 PC 运行)。
> 配套文档:`harmony_acl_build_analysis.md`(参考项目分析)、`cann_lm_harmony_port_plan.md`(移植路径与差距)。
> 维护日期:2026-08-24。

---

## 0. 一句话

engine 是纯标准 C++17、零 POSIX 依赖,可在 Linux 本机直接构建跑 UT,也可交叉编译为鸿蒙 musl 目标的 `.so` + 8 个可执行(UT/bbit/smoke)并自签名。**两套构建相互独立**——鸿蒙构建是独立旁路(`cann_lm/engine/ohos/` + `scripts/build_ohos.sh` + `scripts/sign_ohos.sh` + `scripts/package_ohos.sh`),不改动现有 Linux 构建文件,Linux 构建零风险。鸿蒙一条命令到位:`./scripts/build_ohos.sh`(编译→自动签名)。

---

## 1. 涉及的文件及作用

| 文件 | 作用 | 所属环境 |
|------|------|---------|
| `CMakeLists.txt`(仓库根) | 顶层 CMake,设 C++17,`add_subdirectory(cann_lm/engine)` | 两者 |
| `cann_lm/engine/CMakeLists.txt` | **Linux 构建真身**:定义 `cann_lm_engine`(STATIC)、6 个 UT/smoke/bbit target、GTest(`find_package`)、`gtest_discover_tests` | Linux |
| `cann_lm/engine/**/*.cpp/.h` | engine 源码(18 个 .cpp),两套构建共用 | 两者 |
| `cann_lm/engine/ohos/ohos.toolchain.cmake` | **鸿蒙交叉编译工具链**(自包含):本机 clang-18 + OHOS sysroot,flag 照搬官方 toolchain;ar/ranlib 用本机 llvm-18 | 鸿蒙 |
| `cann_lm/engine/ohos/CMakeLists.txt` | **鸿蒙构建**:同一批源码编 SHARED `libcann_lm_engine.so` + **8 个可执行**(UT/bbit/smoke);GTest FetchContent 本地源码;静态 `add_test` | 鸿蒙 |
| `cann_lm/engine/ohos/binary-sign-tool.jar` | **鸿蒙二进制签名工具**(jar,架构无关);自签名 `-selfSign 1`,注入 `.codesign` 段 | 鸿蒙 |
| `scripts/build_ohos.sh` | **鸿蒙构建+签名入口**:校验 `OHOS_SDK_NATIVE`→cmake build→自动调 `sign_ohos.sh` | 鸿蒙 |
| `scripts/sign_ohos.sh` | **签名脚本**:对 build_ohos/ 下 .so ×2 + 8 可执行各产 `*_signed` | 鸿蒙 |
| `scripts/package_ohos.sh` | **打包脚本**:从 build_ohos/ 签名产物生成 `engine_ohos.run`(自解压部署包) | 鸿蒙 |
| `tools/`(含 `binary-sign-tool.jar`) | vendored 签名工具(12MB,jar,**gitignore 不入库**;见 tools/README.md) | 鸿蒙 |
| `build_ohos/` | **构建产物**(gitignore):交叉编译签名 .so + UT + run.sh | 鸿蒙 |
| `/mnt/workspace/gtest-1.14.0.tar.gz` | GTest 源码(github v1.14.0,867K),OHOS CMake FetchContent 本地引用 | 鸿蒙 |
| `tests/`(cann_lm/engine/../../tests) | GTest UT 源码 + bbit 黑盒 | 两者 |
| `.project/memory/gotchas.md` | OHOS 交叉编译六个必踩坑的根因与解法 | 鸿蒙(参考) |
| `docs/reference/harmony_acl_build_analysis.md` | 参考项目 harmony_acl 构建体系全解 | 鸿蒙(参考) |
| `docs/reference/cann_lm_harmony_port_plan.md` | 兼容性矩阵 + 移植路径 + 差距清单 | 鸿蒙(参考) |

### 为什么鸿蒙用独立 CMake 而不改现有那个?

现有 `cann_lm/engine/CMakeLists.txt` 是单一 Linux 分支 + STATIC 库 + 主机 `find_package(GTest)`。要加鸿蒙需引入 `OHOS_PLATFORM` 分支、SHARED 库、签名、GTest 本地源码等,改动面大且鸿蒙分支在本机无法编译验证。**首版用独立 `ohos/CMakeLists.txt` 旁路**,源码清单目前是过渡期复制(后续稳定后可抽 `engine_src.cmake` 供两者共享,消除复制)。这样 Linux 构建文件一字未动,回归零风险。

---

## 2. Linux 环境构建与验证

### 2.1 前置
```bash
sudo apt-get install build-essential cmake libgtest-dev
# cmake ≥ 3.10;本机已有 cmake 3.20
```

### 2.2 构建
```bash
cd /mnt/workspace/CANN-LM
BUILD=/tmp/cannlm_build
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
cmake /mnt/workspace/CANN-LM -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j"$(nproc)"
```
产物全部在 `$BUILD/cann_lm/engine/` 下(因为 engine 的 CMakeLists 定义 target 在该子目录)。

### 2.3 产物
| 文件 | 作用 |
|------|------|
| `libcann_lm_engine.a` | STATIC 库(被各 target 链接) |
| `model_executor_ut` | 核心组件 UT(kv_cache/input_assembler/model_executor/dfx) |
| `ut_c_api` | C API UT |
| `ut_engine_executor` | engine_executor UT |
| `ut_tokenizer` | tokenizer UT |
| `ut_scheduler` | scheduler UT |
| `bbit` | 黑盒集成(`--check` + 完整流程) |
| `api_smoke` / `c_api_smoke` | C++/C API 冒烟程序 |

### 2.4 验证
```bash
cd "$BUILD/cann_lm/engine"

# 全部单元测试
ctest --output-on-failure
# 期望:100% tests passed, 0 failed (1 disabled: TC003 stub 时代故禁)

# 直接跑某个 UT 看输出
./ut_tokenizer
./ut_c_api

# 黑盒集成
./bbit --check
./bbit
```
> **注意**:`ctest` 必须在 `cann_lm/engine/` 子目录下执行。顶层 `CMakeLists.txt` 未调 `enable_testing()`,故顶层目录 `ctest` 会报 "No tests found"——这是项目现状,非 bug。

### 2.5 当前基线(2026-08-24 实测,pull 后)
- 67 tests passed / 0 failed / 1 disabled(原 63;新 `test_tensor_manager` 加 4 个)
- 编译器:本机 gcc/g++(aarch64-linux-gnu)
- GTest:主机 `libgtest-dev` 包
> 同步要点:Linux `CMakeLists.txt` 是源码清单真源;OHOS `cann_lm/engine/ohos/CMakeLists.txt` 是手抄镜像,pull 后若 Linux 新增 `.cpp`/UT 文件,需同步补进 OHOS 清单(已对 `stub_built_model.cpp`/`tensor_manager.cpp`/`test_tensor_manager.cpp` 做过)。

---

## 3. 鸿蒙(OHOS)环境构建与验证

产 **`libcann_lm_engine.so`**(aarch64-linux-ohos / musl)。

### 3.1 路线选择(为何不用官方 command-line-tools)
官方华为 command-line-tools 只发 **x86_64-linux** 二进制;若本机是 x86_64 可直接用。**本机为 aarch64-linux-glibc**,与目标 aarch64-linux-ohos 同 CPU 架构、仅 libc 不同,故采用更优路线:

> **本机 `clang-18` + 从开源镜像提取的 OHOS SDK sysroot**(clang-18 原生运行,无需 qemu,也无需跑 x86_64 NDK clang)。

SDK 取自 OpenHarmony 开源镜像(直接 curl,非 JS 渲染的华为 consumer 页):
```
https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz
```

### 3.2 前置准备(一次性)

```bash
# 1) 编译器
sudo apt-get install clang-18

# 2) 下载并提取 SDK(只取需要的 82M,非全量 2.5G)
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# 外层 tar 内是嵌套 zip,提取 linux native 那个
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# 解出所需:sysroot + libc++ 头/库 + builtins + 官方 toolchain 参考
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native

# 3) 把 builtins/crt 装进 clang-18 的 resource dir(解坑#1,见 §5)
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

### 3.3 构建

```bash
cd /mnt/workspace/CANN-LM
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
./scripts/build_ohos.sh
```
`scripts/build_ohos.sh` 做的事:校验 `OHOS_SDK_NATIVE` → `cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=... -DOHOS_SDK_NATIVE=... cann_lm/engine/ohos` → `cmake --build` → 自动调 `scripts/sign_ohos.sh` 签名。产物:2 个 .so + 8 个可执行 + 各自 `*_signed`。

### 3.4 产物(在 `build_ohos/`)

| 文件 | 作用 |
|------|------|
| `libcann_lm_engine.so` | 鸿蒙 SHARED 库(ELF AArch64,NEEDED=`libc.so`(musl)+`libc++_shared.so`) |
| `libc++_shared.so` | 鸿蒙 C++ 运行库,随 .so 分发(构建自动从 sysroot 拷贝) |
| `model_executor_ut` | 核心组件 UT(kv_cache/input_assembler/model_executor/dfx) |
| `ut_c_api` / `ut_engine_executor` / `ut_tokenizer` / `ut_scheduler` | 各组件 UT |
| `bbit` | 黑盒集成(`--check` + 完整流程) |
| `api_smoke` / `c_api_smoke` | C++/C API 冒烟程序 |
| 上述每个的 `*_signed` / `*_signed.so` | 签名版(`build_ohos.sh` 自动产,见 §3.6) |

### 3.5 验证产物(确认是 OHOS musl 目标,非 glibc)

```bash
cd /mnt/workspace/CANN-LM/build_ohos
readelf -d libcann_lm_engine.so | grep NEEDED
# 期望:NEEDED libc++_shared.so, libc.so   ← libc.so 而非 libc.so.6 = musl/OHOS
readelf -h libcann_lm_engine.so | grep Machine
# 期望:Machine: AArch64
```
> 在 Linux 本机**无法直接运行**该 .so(它链接 musl 的 `libc.so`,本机是 glibc)。运行验证必须在鸿蒙设备上(见 §3.6)。

### 3.6 签名(本机自动完成)

`scripts/build_ohos.sh` 在 build 成功后自动调 `scripts/sign_ohos.sh`,用 `tools/binary-sign-tool.jar` 对每个产物自签名(照 harmony_acl: `-selfSign 1`)。jar 与架构无关,本机 java 21 直接跑,**无需 cert/keystore**(selfSign 模式用 descriptor digest 作签名,注入 ELF 的 `.codesign` 段)。

```bash
# 单独重跑签名(不重新编译):
./scripts/sign_ohos.sh                    # 默认对 build_ohos/ 签名
./scripts/sign_ohos.sh <dir> <jar路径>    # 指定
```
签名产出同名 `*_signed`(可执行)与 `*_signed.so`(.so)。

### 3.7 设备运行验证(需鸿蒙 PC / 设备)

**先打包部署包**(`build_ohos/` 签名产物 → `engine_ohos.run`):
```bash
./scripts/package_ohos.sh     # 打成自解压 .run 包
```
`engine_ohos.run` 是自包含部署包(自解压)。鸿蒙 PC 上解压后 `./run.sh` 一键跑全部。

本机无链接设备时的约定流程:**打包 engine_ohos.run → 推到设备 → 解压 → `./run.sh`**。

```bash
export LD_LIBRARY_PATH=$(pwd)
./model_executor_ut          # 跑 UT
./bbit --check               # 黑盒
```
> 设备端需开发者模式/允许自签名二进制(selfSign 仅供开发调试,非正式发布签名)。

### 3.8 当前状态(2026-08-24)
- ✅ `.so` + `libc++_shared.so` 交叉编译通过(ELF AArch64,musl 目标)
- ✅ **8 个可执行交叉编译通过**(UT/bbit/smoke;GTest 本地源码 FetchContent,静态 add_test)
- ✅ **全部产物签名完成**(`*_signed`,本机 java 21 + binary-sign-tool.jar -selfSign 1)
- ⏳ 鸿蒙设备实跑验证(待设备;git 提交后设备 pull)

---

## 4. 两套构建对照

| 维度 | Linux 本机 | 鸿蒙 OHOS 交叉 |
|------|-----------|---------------|
| 入口 | `cmake <repo> && cmake --build` | `./scripts/build_ohos.sh` |
| CMake 真身 | `cann_lm/engine/CMakeLists.txt` | `cann_lm/engine/ohos/CMakeLists.txt` |
| 工具链 | 系统 gcc/g++ | `cann_lm/engine/ohos/ohos.toolchain.cmake`(clang-18 + sysroot) |
| 目标 triple | aarch64-linux-gnu(本机) | aarch64-linux-ohos(musl) |
| 库类型 | STATIC(`.a`) | SHARED(`.so`) |
| GTest | 主机 `find_package(GTest)` | 本地源码 FetchContent(github v1.14.0);静态 `add_test`(交叉编译禁用 `gtest_discover_tests`) |
| 产物目录 | `$BUILD/cann_lm/engine/` | `build_ohos/` |
| 可运行环境 | 本机 | 仅鸿蒙设备(签名后) |
| 是否改动现有 CMake | — | **否**(旁路,零风险) |

---

## 5. 鸿蒙交叉编译三个必踩坑(详见 `.project/memory/gotchas.md`)

| # | 症状 | 根因 | 解法 |
|---|------|------|------|
| 1 | `libclang_rt.builtins.a` / `crtbeginS.o` 找不到 | clang-18 只在自己 resource dir(`/usr/lib/clang/18/lib/aarch64-linux-ohos/`)找 | 把 NDK 的 `libclang_rt.builtins.a` + `clang_rt.crtbegin.o`/`crtend.o` 拷进去 |
| 2 | `-lunwind` / `-lc++` 找不到 | 这些静态库在 `native/llvm/lib/aarch64-linux-ohos/`,不在 sysroot 标准路径 | 链接 flags 加 `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos` |
| 3 | `<__config_site>` not found | 用错 libc++ 头集:通用 `llvm/include/c++/v1/` 缺生成头 | 改用 `llvm/include/libcxx-ohos/include/c++/v1/`(带 `__config_site` 的 OHOS 配置版) |

---

## 6. 环境变量速查

| 变量 | 含义 | 默认/示例 |
|------|------|----------|
| `OHOS_SDK_NATIVE` | 提取出的 OHOS SDK `native/` 绝对路径(含 sysroot/llvm) | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | `scripts/build_ohos.sh` 输出目录(可选) | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | 构建类型 | Linux:Debug;OHOS:Release |
