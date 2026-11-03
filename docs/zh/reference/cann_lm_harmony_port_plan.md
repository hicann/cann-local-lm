# CANN-LM engine 鸿蒙 PC 移植路径与差距清单(PLAN)

简体中文 | [English](../../en/reference/cann_lm_harmony_port_plan.md)

> 配套参考:`docs/reference/harmony_acl_build_analysis.md`(harmony_acl 构建分析)
> 编制日期:2026-08-24
> 目标:把 `cann_local_lm/engine/` 在鸿蒙 PC 上编译为 **.so** 或 **可执行 UT** 并能运行。
> 本文档为 Phase 3 PLAN 产出(base.DESIGN + chip-adaptation.COMPATIBILITY-MAP)。

---

## 1. 现状一句话

engine 是**纯标准 C++17**、**零 POSIX 专有依赖**、**零外部库依赖**(仅内部源码 + GTest)、当前用**单一 Linux/GCC + STATIC 库 + 主机 `find_package(GTest)`** 构建。**可移植性极高**——移植工作 ~90% 在构建系统侧,源码侧几乎不动。

---

## 2. 兼容性矩阵(COMPATIBILITY-MAP:源平台 Linux/glibc → 目标 鸿蒙/musl)

逐项对照 harmony_acl 范本扫出的结果:

| 维度 | 源平台(现 Linux 主机) | 目标平台(鸿蒙 PC / OHOS) | 兼容? | 处理 |
|------|---------------------|------------------------|------|------|
| 编译器 | 系统 gcc/g++ | OHOS SDK 自带 clang(`aarch64-linux-ohos`) | 否(不同编译器) | 引入 toolchain 文件 |
| 目标三元组 | 本机 x86_64-linux-gnu | `aarch64-linux-ohos` | 否 | toolchain 强制设定 |
| libc | glibc | **musl** | 部分差异 | `__MUSL__=1`,审 musl 缺失 API |
| ABI/架构 | x86_64 | arm64-v8a | 否 | 交叉编译 |
| STL | libstdc++ | **libc++_shared** | 不同 | `OHOS_STL=c++_shared`,拷 `libc++_shared.so` |
| `<vector>/<memory>/<algorithm>/<atomic>/<mutex>/<thread>/<chrono>/<random>/<cmath>` | ✓ | ✓(标准库) | ✓ | 无需改 |
| `<filesystem>`(dfx_collector 落盘) | ✓ | ✓(clang/musl 支持) | ✓(注意链接) | C++17 已开,鸿蒙 clang 通常 header-only 通过;若链接报错加 `-lc++filesystem` 类似处理(实测确认) |
| POSIX 头(`dirent`/`dlfcn`/`pthread`/`unistd`/`sys/*`) | — | **无 POSIX 专有头** | ✓ | engine 源码零此类 include,**无坑** |
| POSIX API(realpath/dlopen/opendir/...) | — | 无 | ✓ | 零使用 |
| 平台宏 | 无 | 需 `__OHOS__`/`__MUSL__=1` | 需新增 | 源码目前无平台分支——若日后接 CANN runtime/日志再加 |
| 链接器 | ld | **lld**(toolchain 强制) | 否 | toolchain 设 `-fuse-ld=lld` |
| GTest | `find_package(GTest REQUIRED)` 主机装 | 主机端不可用,鸿蒙联网受限 | 否 | OHOS 分支改**本地源码 FetchContent** |
| 库类型 | `STATIC` | 需 `SHARED` 产 .so | 改 | OHOS 分支 `SHARED` |
| 签名 | 无 | **必须 selfSign** | 否 | 加 `binary-sign-tool.jar` POST_BUILD |
| RPATH | 无 | `$ORIGIN` 就近找 .so | 需新增 | 设 `CMAKE_INSTALL_RPATH $ORIGIN` |
| 外部系统库 | 无 | hilog(若用鸿蒙日志) | 视需求 | 目前 engine 无日志库依赖,可暂不链 |
| 动态加载第三方 .so | 无 | (harmony_acl 有,本项目暂无) | N/A | 本项目无 dlopen,暂不涉及 |

**矩阵结论**:**无 P0 编译阻断**。源码零 POSIX 依赖是最大利好——不像 harmony_acl 还要处理 `dirent`/`realpath`/`.so vs .dll`。唯一需要审的是 `<filesystem>` 在鸿蒙 clang/musl 下的链接(大概率没问题,实测一跑便知)。

---

## 3. .so vs 可执行 UT —— 决策

你的目的:"编译成一个 so,或者编译出可以执行的 ut 文件"。两者不冲突,harmony_acl 是**同时产出**。建议:

| 产物 | 价值 | 难度 | 建议 |
|------|------|------|------|
| **libcanngin_engine.so**(SHARED 库) | 供鸿蒙应用/其他模块动态加载 engine 能力 | 低(改 STATIC→SHARED + 签名) | ✅ 产出 |
| **可执行 UT**(各 *_ut + bbit) | 在鸿蒙 PC 上直接验证 engine 各组件逻辑 | 中(UT 需链 GTest,鸿蒙 GTest 要本地源码) | ✅ 产出(先跑通比 .so 更能验证正确性) |

**推荐**:**两者并列产出**,但**优先级 = 先 UT 可执行**。理由:UT 能在鸿蒙 PC 直接跑出 GREEN,是"真的能在鸿蒙编译并执行"的最强证据;.so 只是产物,自身不证明逻辑对。先 UT 后 .so。

> 与 harmony_acl 一致:它的 `build.sh` OHOS 分支同时生成 `libhiai_ascendcl.so` + `ascendcl_ut`。

---

## 4. 行动路径(分阶段,对应 Locked Step 的 PORT/COMPILE)

### 前置条件(环境准备,你需提供)
1. **华为 command-line-tools**(linux x64 版):下载解压,记录其目录路径(对应 harmony_acl 的 `CMD_LINE_TOOLS_DIR`)。下载页:https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos
2. **OHOS SDK**:随 command-line-tools 带,含 `native/llvm/clang`、`native/sysroot`、`native/build/cmake/ohos.toolchain.cmake`。
3. **Java**(签名用):`binary-sign-tool.jar` 需 java 运行。
4. **Ninja**(`cmake -G Ninja`)。
5. **GTest 源码**:下 `googletest-1.14.0` 源码 zip 放本地(OHOS 联网受限,不能用 GIT_REPOSITORY)。
6. 宿主机 ubuntu 24.04+ + build-essential(跑主机 UT 对照用)。

### Step A — 引入 OHOS toolchain(不改源码)
- 在 `cann_local_lm/engine/` 旁新增 `scripts/ohos.toolchain.cmake`(参照 harmony_acl,或直接用 SDK 官方那个)。
- 新增 `scripts/build_ohos.sh`:设 `CMD_LINE_TOOLS_DIR`/`OHOS_SDK` 等变量 → `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=... -DOHOS_PLATFORM=OHOS -DOHOS_ARCH=arm64-v8a -DOHOS_STL=c++_shared ..` → `cmake --build`。

### Step B — CMake 加 OHOS 分支(改 `cann_local_lm/engine/CMakeLists.txt`)
现状是单分支。改为:
```cmake
if(OHOS_PLATFORM STREQUAL "OHOS")
  # SHARED 库 + $ORIGIN RPATH + GTest 本地 FetchContent + 签名 POST_BUILD + 拷 libc++_shared.so
  add_library(cann_lm_engine SHARED ${ENGINE_SRC})
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
  FetchContent_Declare(googletest SOURCE_DIR ${CMAKE_SOURCE_DIR}/../third_party/googletest-1.14.0)
  FetchContent_MakeAvailable(googletest)
  # 各 *_ut target 改链 gtest target(去掉 find_package(GTest REQUIRED) 在此分支)
  # 签名 custom command
elseif(UNIX AND NOT APPLE)
  # 现有 Linux 分支(STATIC + find_package GTest)原样保留
elseif(WIN32)
  # 可选,后续
endif()
```
- 把现有 `find_package(GTest REQUIRED)` 用 `if(NOT OHOS_PLATFORM...)` 包住(OHOS 分支用 FetchContent)。
- `gtest_discover_tests` 在交叉编译下可能有问题(cross-execution),OHOS 分支考虑改用 `ctest` 注册命令而非 discover(实测确认)。

### Step C — 审 `<filesystem>`(唯一源码侧风险点)
- `dfx_collector.h` 用 `std::filesystem`。OHOS clang/musl 通常 header-only 通过;若链接报 undefined reference,加 target link 选项。**不动源码**,只调构建。

### Step D — 签名 + 产物部署
- 对 `cann_lm_engine.so`、各 UT 可执行、`libc++_shared.so` 跑 `binary-sign-tool.jar -selfSign`。
- 产物集中在 `build_ohos/`。

### Step E — 鸿蒙 PC 运行验证
- 推送到鸿蒙 PC,`export LD_LIBRARY_PATH=$(pwd)`,跑 `*_ut` → 期望与主机端同样的 GREEN(对照 PROGRESS.md 里 8/20 的 ctest 全绿基线)。
- 跑 bbit(黑盒集成)。

### Step F — REVIEW(thorough 多轮)
- +1 自审 + +2 外部审查,直至零新发现或 5 轮。

---

## 5. 风险与待实测项

| 风险 | 说明 | 缓解 |
|------|------|------|
| `gtest_discover_tests` 交叉编译失效 | 它要在主机跑编译出的测试二进制做发现,交叉目标二进制跑不了 | OHOS 分支改 `add_test(NAME ... COMMAND <target>)` 静态注册 |
| `<filesystem>` 链接 | 鸿蒙 musl/clang 下偶需额外链接 | 实测;若报错按 clang 提示加 link |
| 签名工具路径/Java 版本 | harmony_acl 里硬编码了 Windows路径 | 改成可配置变量 + linux 适配 |
| OHOS SDK 版本差异 | `ohos.toolchain.cmake` 变量名随版本变 | 优先用 SDK 自带那个,不自己写 |
| UT 在鸿蒙设备能否 exec | 鸿蒙对未签名/未授权二进制有限制 | 必须 selfSign + 落到可执行目录 |

---

## 6. 退出标准(对齐 active_workflow)

- [x] 兼容性矩阵已记录(本节 §2)
- [x] .so vs UT 方案已决策(§3:并列,先 UT)
- [ ] OHOS toolchain + build 脚本就位(Step A)
- [ ] CMake OHOS 分支接通,主机端仍能原样构建(Step B,无回归)
- [ ] 鸿蒙 PC 上 UT 跑通 GREEN(Step E)
- [ ] `cann_lm_engine.so` 产出并签名可加载(Step D/E)
- [ ] +1 +2 审查通过(Step F)
