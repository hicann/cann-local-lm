# CANN-LM 编译与验证指南

> 覆盖两种环境:**Linux 本机构建**(开发/CI/主机 UT)与 **鸿蒙(OHOS)交叉编译**(产 `libcann_lm_engine.so` 供鸿蒙 PC 运行)。
> 配套文档:`harmony_acl_build_analysis.md`(参考项目分析)、`cann_lm_harmony_port_plan.md`(移植路径与差距)。
> 维护日期:2026-10-15。

---

## 0. 一句话

engine 是纯标准 C++17、零 POSIX 依赖,可在 Linux 本机直接构建跑 UT,也可交叉编译为鸿蒙 musl 目标的 `.so` + 9 个可执行(UT/bbit/smoke)并自签名。**两套构建相互独立**——鸿蒙构建是独立旁路(`cann_lm/engine/ohos/` + `scripts/cann_lm_engine/` 下的 `build_ohos.sh`/`sign_ohos.sh`/`package_ohos.sh`/`build_package_ohos.sh`),不改动现有 Linux 构建文件,Linux 构建零风险。鸿蒙一条命令到位:`./scripts/cann_lm_engine/build_ohos.sh`(编译→自动签名),或一键 `./scripts/cann_lm_engine/build_package_ohos.sh`(编译→签名→打包 `engine_ohos.run`)。

---

## 1. 涉及的文件及作用

| 文件 | 作用 | 所属环境 |
|------|------|---------|
| `CMakeLists.txt`(仓库根) | 顶层 CMake,设 C++17,`add_subdirectory(cann_lm/engine)` | 两者 |
| `cann_lm/engine/CMakeLists.txt` | **Linux 构建真身**:定义 `cann_lm_engine`(STATIC)+ `cann_backend`(子项目)、**13 个 UT + bbit + 2 smoke(+1 临时 golden_verify)** target、GTest(`find_package`)、`gtest_discover_tests` | Linux |
| `cann_lm/engine/**/*.cpp/.h` | engine 源码(18 个 .cpp),两套构建共用 | 两者 |
| `cann_lm/engine/ohos/ohos.toolchain.cmake` | **鸿蒙交叉编译工具链**(自包含):本机 clang-18 + OHOS sysroot,flag 照搬官方 toolchain;ar/ranlib 用本机 llvm-18 | 鸿蒙 |
| `cann_lm/engine/ohos/CMakeLists.txt` | **鸿蒙构建**:同一批源码编 SHARED `libcann_lm_engine.so` + **9 个可执行**(UT/bbit/smoke);GTest FetchContent 本地源码;静态 `add_test` | 鸿蒙 |
| `tools/binary-sign-tool.jar` | **鸿蒙二进制签名工具**(jar,架构无关);自签名 `-selfSign 1`,注入 `.codesign` 段;放 `tools/`,**gitignore 不入库**(见 tools/README.md) | 鸿蒙 |
| `scripts/cann_lm_engine/build_ohos.sh` | **鸿蒙构建+签名入口**:校验 `OHOS_SDK_NATIVE`(精简 SDK)+ 本机 `clang-18` → `cmake -G Unix Makefiles` build → 11 产物校验 + libc++ 兜底 → 自动调 `sign_ohos.sh` | 鸿蒙 |
| `scripts/cann_lm_engine/sign_ohos.sh` | **签名脚本**:对 build_ohos/ 下 .so ×2 + **9 可执行**各产 `*_signed` | 鸿蒙 |
| `scripts/cann_lm_engine/package_ohos.sh` | **打包脚本**:从 build_ohos/ 签名产物生成 `engine_ohos.run`(11 产物去 `_signed` 后缀、重命名 .so、生成 `run.sh` 跑 UT+bbit+smoke) | 鸿蒙 |
| `scripts/cann_lm_engine/build_package_ohos.sh` | **一键脚本**:预检 java+jar → `build_ohos.sh`(编译+签名) → `package_ohos.sh`(打包) → `engine_ohos.run`;支持 `--clean` 全量重建 | 鸿蒙 |
| `build_ohos/` | **构建产物**(gitignore):11 签名产物 + run.sh | 鸿蒙 |
| `/mnt/workspace/gtest-1.14.0.tar.gz` | GTest 源码(github v1.14.0,867K),OHOS CMake FetchContent 本地引用(CMake 变量 `GTEST_TAR` 默认此路径) | 鸿蒙 |
| `tests/` | GTest UT 源码 + bbit 黑盒 | 两者 |
| `docs/build_reference/harmony_acl_build_analysis.md` | 参考项目 harmony_acl 构建体系全解 | 鸿蒙(参考) |
| `docs/build_reference/cann_lm_harmony_port_plan.md` | 兼容性矩阵 + 移植路径 + 差距清单 | 鸿蒙(参考) |

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
全部落在 `$BUILD/cann_lm/engine/` 下。target 清单(对齐 `cann_lm/engine/CMakeLists.txt`):

| 文件 | 作用 |
|------|------|
| `libcann_lm_engine.a` | STATIC 库(被各 target 链接);经 `target_link_libraries PUBLIC cann_backend` 拉入 backend 子项目 |
| **UT(GTest_FOUND 守卫,`gtest_discover_tests` 注册)** | |
| `model_executor_ut` | 核心组件 UT,聚合 4 个测试文件:`test_kv_cache_pool`/`test_model_executor`/`test_dfx_collector`/`test_tensor_manager` |
| `ut_input_assembler` | InputAssembler UT(含 gemma4 双 bool mask 用例,见 `docs/mask_fill.md`) |
| `ut_linear_state_pool` | Qwen3.5 线性状态池 UT(`support_qwen3.5.md` §4.4) |
| `ut_engine_executor` | engine_executor UT |
| `ut_tokenizer` | tokenizer UT(读 `tests/fixtures/tokenizer/`) |
| `ut_c_api` | C API UT |
| `ut_scheduler` | scheduler UT |
| `ut_sampler` | sampler UT |
| `ut_request_handle` | request_handle UT |
| `ut_backend_shareweights` | 权重共享 UT(shim 快速失败 + 默认 0 不绑定契约) |
| `ut_share_weights_executor` | 权重共享 executor 层 UT(Load share_weights_model_id 透传) |
| `ut_data_type_utils` | data_type_utils UT |
| `ut_text_embedding` | TextEmbedding UT(SR20260828001) |
| `tokenizer_realdata` | tokenizer 实物 UT(**条件构建**:仅 `tmp/qwen`+`tmp/gemma` 的 `tokenizer.json` 在场才注册) |
| **黑盒 / 冒烟** | |
| `bbit` | 黑盒集成(`--check` NULL 防御 + `--config` 完整流程;含 `bbit_host_flow` 用例) |
| `api_smoke` / `c_api_smoke` | C++ / C API 冒烟程序 |
| `golden_verify` | 【临时调试】模型精度验证工具(验证完成后删除) |

> 另有 `stub_ndk`(host 回归用的 external_ndk 打桩 .so,位于 `tests/cpp/` 下)与 `cann_backend` 子项目自带 demo/test,均 `EXCLUDE_FROM_ALL`——不进默认构建,需显式 `--target` 触发。
> OHOS 交叉编译(`ohos/CMakeLists.txt`)只取上述的**一个子集**编 9 个可执行(6 UT + bbit + 2 smoke),见 §3.4。

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
> ⚠ 该 67 项基线测于 2026-08-24,此后 §2.3 的若干 UT 目标(`ut_text_embedding`/`ut_data_type_utils`/`ut_sampler`/`ut_request_handle`/`ut_backend_shareweights`/`ut_share_weights_executor`/`ut_linear_state_pool`)陆续加入,实际用例数已高于 67——以本机 `ctest` 实跑为准。
> 同步要点:Linux `CMakeLists.txt` 是源码清单真源;OHOS `cann_lm/engine/ohos/CMakeLists.txt` 是手抄镜像,pull 后若 Linux 新增 `.cpp`/UT 文件,需同步补进 OHOS 清单(已对 `stub_built_model.cpp`/`tensor_manager.cpp`/`test_tensor_manager.cpp` 做过)。

---

## 3. 鸿蒙(OHOS)交叉编译(详见分平台手册)

产 **`libcann_lm_engine.so`**(aarch64-linux-ohos / musl)+ `libc++_shared.so` + 9 个可执行,均自签名。路线:本机 `clang-18` + 精简 OHOS SDK(无 `llvm/bin`/`build-tools`,SDK 仅贡献 sysroot/libc++/builtins/工具链文件)+ `cmake -G "Unix Makefiles"` + `make`(非 Ninja)。本机 aarch64-glibc 与目标 aarch64-ohos 同 CPU 架构、仅 libc 不同,clang-18 原生运行无需 qemu。

> **详细步骤(依赖清单 / SDK 提取 / 获取代码 / 编译命令 / 常见问题 / 提交)见分平台手册**:
> - Linux 主机 → [`build_linux_guide.md`](build_linux_guide.md)
> - Windows(WSL) → [`build_win_guide.md`](build_win_guide.md)
>
> 一条命令(两平台通用):`bash scripts/cann_lm_engine/build_ohos.sh`(编译+签名),或一键 `bash scripts/cann_lm_engine/build_package_ohos.sh [--clean]`(编译+签名+打包 `engine_ohos.run`)。三坑见 §5,环境变量见 §6。

### 3.1 产物速查(平铺 `build_ohos/`,共 11 个 + 各 `*_signed`)

| 文件 | 作用 |
|------|------|
| `libcann_lm_engine.so` | 鸿蒙 SHARED 库(ELF AArch64,NEEDED=`libc.so`(musl)+`libc++_shared.so`) |
| `libc++_shared.so` | C++ 运行库(构建自动从 sysroot 拷贝;缺失时兜底) |
| `model_executor_ut` / `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding` | 各组件 UT |
| `bbit` | 黑盒集成(`--check` + 完整流程) |
| `api_smoke` / `c_api_smoke` | C++/C API 冒烟 |
| 上述每个的 `*_signed` / `*_signed.so` | 签名版(`build_ohos.sh` 自动产) |

> 产物验证(readelf 确认 musl 目标)、单独重签、打包 `engine_ohos.run`(`run.sh` 跑 6 UT + bbit + 2 smoke)、设备运行流程,均见分平台手册。本机 glibc **无法直接运行**这些产物(链接 musl 的 `libc.so`),运行验证须在鸿蒙设备(开发者模式,允许自签名二进制)。

### 3.2 当前状态(2026-10-15 实测)
- ✅ `.so` + `libc++_shared.so` 交叉编译通过(ELF AArch64,musl)
- ✅ **9 可执行交叉编译通过**(6 UT + bbit + 2 smoke;GTest FetchContent 本地源码,静态 add_test)
- ✅ **11 产物全部签名完成**(本机 java + `tools/binary-sign-tool.jar -selfSign 1`)
- ✅ **打包通过**(`engine_ohos.run` 含全 11 产物 + run.sh);一键 `build_package_ohos.sh` 验证通过
- ✅ `model_executor_ut` / `ut_text_embedding` 已恢复(backend hiai 迁移完成)
- ⏳ 鸿蒙设备实跑验证(待设备;打包 engine_ohos.run 推设备)

---

## 4. 两套构建对照

| 维度 | Linux 本机 | 鸿蒙 OHOS 交叉 |
|------|-----------|---------------|
| 入口 | `cmake <repo> && cmake --build` | `./scripts/cann_lm_engine/build_ohos.sh` |
| CMake 真身 | `cann_lm/engine/CMakeLists.txt` | `cann_lm/engine/ohos/CMakeLists.txt` |
| 工具链 | 系统 gcc/g++ | `cann_lm/engine/ohos/ohos.toolchain.cmake`(clang-18 + sysroot) |
| 目标 triple | aarch64-linux-gnu(本机) | aarch64-linux-ohos(musl) |
| 库类型 | STATIC(`.a`) | SHARED(`.so`) |
| GTest | 主机 `find_package(GTest)` | 本地源码 FetchContent(github v1.14.0);静态 `add_test`(交叉编译禁用 `gtest_discover_tests`) |
| 产物目录 | `$BUILD/cann_lm/engine/` | `build_ohos/` |
| 可运行环境 | 本机 | 仅鸿蒙设备(签名后) |
| 是否改动现有 CMake | — | **否**(旁路,零风险) |

---

## 5. 鸿蒙交叉编译三个必踩坑

> 历史踩坑记录原存 `.project/memory/gotchas.md`(内部记忆,已随 `.project/` 一并 gitignore,不入库);根因与解法摘录于下表,build_ohos.sh / ohos.toolchain.cmake 已固化对应处理。

| # | 症状 | 根因 | 解法 |
|---|------|------|------|
| 1 | `libclang_rt.builtins.a` / `crtbeginS.o` 找不到 | clang-18 只在自己 resource dir(`/usr/lib/clang/18/lib/aarch64-linux-ohos/`)找 | 把 NDK 的 `libclang_rt.builtins.a` + `clang_rt.crtbegin.o`/`crtend.o` 拷进去 |
| 2 | `-lunwind` / `-lc++` 找不到 | 这些静态库在 `native/llvm/lib/aarch64-linux-ohos/`,不在 sysroot 标准路径 | 链接 flags 加 `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos` |
| 3 | `<__config_site>` not found | 用错 libc++ 头集:通用 `llvm/include/c++/v1/` 缺生成头 | 改用 `llvm/include/libcxx-ohos/include/c++/v1/`(带 `__config_site` 的 OHOS 配置版) |

---

## 6. 环境变量速查

| 变量 | 含义 | 默认/示例 |
|------|------|----------|
| `OHOS_SDK_NATIVE` | 提取出的 OHOS SDK `native/` 绝对路径(含 sysroot + `llvm/lib`/`include` + toolchain;**无** `llvm/bin`、**无** `build-tools`) | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | `scripts/cann_lm_engine/build_ohos.sh` 输出目录(可选) | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | 构建类型 | Linux:Debug;OHOS:Release |
| `GTEST_TAR` | GTest 源码 tarball 路径(OHOS CMake FetchContent 本地引用) | `/mnt/workspace/gtest-1.14.0.tar.gz` |
