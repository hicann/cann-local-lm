简体中文 | [English](../../../en/cann_lm_engine/build_reference/build_linux_guide.md)

# Linux 编译指南 — cann_llm_open（OHOS 交叉编译）

> 目标：在 Linux(aarch64) 上交叉编译引擎 + bbit + 全部 UT，产出签名版可部署件。
> 双 SDK 路线（`ohos.toolchain.cmake` 内编译器自动二选一，`build_ohos.sh` 自动探测 SDK）：
> - **路线A（推荐）**：华为 command-line-tools 的 **native 子树**——`llvm/bin/clang` 是 ARM-native，本机可直接运行，**自带 builtins/crt resource dir**（无需补 `/usr/lib/clang/18`），无需本机 clang-18。
> - **路线B（精简）**：OpenHarmony 镜像提取的 SDK（无 `llvm/bin`）——回退**本机 clang-18** + 手动补 builtins。
> 两路线产物一致（OHOS musl/AArch64），构建器均用本机 make（完整 SDK 的 build-tools 仅含 cmake，无 ninja）。

## 1. 依赖清单

| # | 依赖 | 获取方式 | 用途 |
|---|------|---------|------|
| 1 | clang-18 / clang++-18（**仅路线B**） | `apt install clang-18 llvm-18` | 本机交叉编译器（精简 SDK 无 `llvm/bin/clang` 时回退）。**路线A 用 SDK 自带 clang-15，无需本机 clang-18** |
| 2 | cmake ≥ 3.16 | `apt install cmake`（本机已有 3.20） | 构建配置 |
| 3 | make | `apt install build-essential` 自带 | 构建执行（`-G "Unix Makefiles"`） |
| 4 | JDK | `apt install default-jdk`（实测 java-21 可用） | 签名工具 `binary-sign-tool.jar` 需要 java |
| 5 | OHOS SDK（native 子树，精简版） | 见 §2 | sysroot + libc++ 头/库 + builtins + `ohos.toolchain.cmake` |
| 6 | googletest v1.14.0 tarball | 见 §3 | OHOS CMake FetchContent 本地引用（交叉编译不能用主机 `find_package(GTest)`） |
| 7 | binary-sign-tool.jar | 放 `tools/`（12MB，`.gitignore` 不入库；见 `tools/README.md`） | 自签名 |

> **路线B（精简 SDK）**：无 `llvm/bin`（无 clang）、`build-tools` 仅 cmake 无 ninja——编译器用本机 clang-18，SDK 贡献 sysroot/libc++/builtins/工具链文件。本机 aarch64-linux-glibc 与目标同 CPU 架构、仅 libc 不同，clang-18 原生运行无需 qemu。
> **路线A（完整 SDK，推荐）**：`llvm/bin/clang` 是 ARM-native 本机直接运行，自带 builtins resource dir（无需补 `/usr/lib/clang/18`），`build-tools` 仅 cmake（仍用本机 make）。两路线产物一致，由 `ohos.toolchain.cmake` 内编译器自动二选一。

## 2. 获取 OHOS SDK

### 2.1 路线A：完整 SDK（推荐）

从华为开发者官网下载 **Command Line Tools for Linux(ARM)**：
https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos

下载 zip（如 `commandline-tools-linux-arm64-26.0.0.851.zip`），只解压 native 子树：

```bash
mkdir -p /mnt/workspace/ohos_sdk/cli_root
unzip -oq commandline-tools-linux-arm64-*.zip \
  'command-line-tools/sdk/default/openharmony/native/*' \
  -d /mnt/workspace/ohos_sdk/cli_root
# 移到固定路径（build_ohos.sh 自动探测此路径）
mv /mnt/workspace/ohos_sdk/cli_root/command-line-tools/sdk/default/openharmony/native \
   /mnt/workspace/ohos_sdk/cli_native
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/cli_native
```

完整 SDK 的 native 子树（**含 `llvm/bin/clang`，ARM-native**）：
```
$OHOS_SDK_NATIVE/  (= .../cli_native)
  ├── llvm/bin/clang(+clang-15,llvm-ar,ld.lld,...)   ← ARM-native,本机直接运行
  ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← builtins/crt 自带(坑#1 自动解)
  ├── llvm/lib/aarch64-linux-ohos/                   ← libc++_shared.so/libunwind.a/libc++abi.a
  ├── llvm/include/libcxx-ohos/include/c++/v1/       ← OHOS libc++ 头(带 __config_site)
  ├── sysroot/usr/lib/aarch64-linux-ohos/            ← 完整 musl sysroot
  └── build/cmake/ohos.toolchain.cmake               ← 官方工具链(参考,实际用仓内版)
```

验证：
```bash
$OHOS_SDK_NATIVE/llvm/bin/clang --version   # clang 15.0.4, Target: aarch64-unknown-linux-gnu
```
> 路线A 无需本机 clang-18，无需补 builtins（坑#1/2/3 全自动解）。直接进 §4 编译。

### 2.2 路线B：精简 SDK（OpenHarmony 镜像，无 llvm/bin）

SDK 取自 OpenHarmony 开源镜像（直接 curl，非 JS 渲染的华为 consumer 页）：

```bash
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# 外层 tar 内是嵌套 zip，提取 linux native 那个
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# 只解需要的 82M（非全量 2.5G）：sysroot + libc++ 头/库 + builtins + 官方 toolchain 参考
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
```

解压后目录结构（注意：**没有 `llvm/bin`、没有 `build-tools`**）：
```
$OHOS_SDK_NATIVE/
  ├── sysroot/                       ← 目标系统头文件/库（musl）
  ├── llvm/lib/aarch64-linux-ohos/   ← libc++_shared.so / libunwind.a / libc++.a …
  ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← libclang_rt.builtins.a / crt
  ├── llvm/include/libcxx-ohos/include/c++/v1/       ← OHOS 专配 libc++ 头（带 __config_site）
  └── build/cmake/ohos.toolchain.cmake              ← 官方工具链文件（参考，实际用仓内版）
```

### 验证 SDK

```bash
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
[ -d "$OHOS_SDK_NATIVE/sysroot" ] && echo "sysroot ok" || echo "sysroot 缺失"
ls "$OHOS_SDK_NATIVE/llvm/lib/aarch64-linux-ohos/" | head    # 应有 libc++_shared.so 等
clang-18 --version    # 本机编译器，不是 SDK 的
```

### 首次环境（仅路线B）：把 builtins/crt 装进 clang-18 resource dir（解坑#1，见 §6）

> 路线A 用 SDK 自带 clang-15，其 resource dir `llvm/lib/clang/15.0.4/` 自带 builtins/crt，**跳过此步**。

```bash
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

## 3. 获取代码与 GTest tarball

```bash
git clone ssh://szv-y.codehub.huawei.com:2222/h00651112/cann_llm_open.git
cd cann_llm_open
```

- 写权限需找仓 owner 开通（否则报 CH.00905401）。
- clone 到**纯英文路径**（中文路径在 Windows 必炸，Linux 下也建议英文）。

GTest tarball（OHOS 交叉编译用）：
```bash
# 下载 googletest v1.14.0（867K），放到 OHOS CMakeLists 默认路径
curl -L -o /mnt/workspace/gtest-1.14.0.tar.gz \
  https://github.com/google/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz
```
> `cann_lm/engine/ohos/CMakeLists.txt` 用 FetchContent 引此 tarball（CMake 变量 `GTEST_TAR`，默认 `/mnt/workspace/gtest-1.14.0.tar.gz`）。缺失会 `FATAL_ERROR` 并提示下载地址。无需 `find_package(GTest)`（交叉编译下主机 GTest 不可用）。

## 4. 编译

```bash
cd /mnt/workspace/CANN-LM
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native   # 或写入 ~/.bashrc 持久化

# 方式一：只编译 + 签名
bash scripts/cann_lm_engine/build_ohos.sh

# 方式二：一键编译 + 签名 + 打包部署目录
bash scripts/cann_lm_engine/build_package_ohos.sh
# 加 --clean 自动清除：bash scripts/cann_lm_engine/build_package_ohos.sh --clean
```

### 编译流程说明（`build_ohos.sh` 内部）

1. **SDK 定位**：`OHOS_SDK_NATIVE` 环境变量；空则自动探测（`cli_native` 完整SDK > `ohos_native/native` 精简SDK），校验 `sysroot` 存在。
2. **编译器校验**：SDK 自带 `${OHOS_SDK_NATIVE}/llvm/bin/clang` 优先（路线A）；无则查本机 `clang-18`（路线B）。两皆无才报错。
3. **CMake configure**：`cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=cann_lm/engine/ohos/ohos.toolchain.cmake -DOHOS_SDK_NATIVE=... -DOHOS_PLATFORM=OHOS -DCMAKE_BUILD_TYPE=Release -B build_ohos -S cann_lm/engine/ohos`。
4. **Make 编译**：`cmake --build build_ohos -j$(nproc)`（本机 make，非 ninja）。
5. **产物校验**：11 个产物逐一 `[ok]`/`[缺]`（见 §5）；`libc++_shared.so` 缺失则从 SDK find 兜底拷贝。
6. **签名**：自动调 `scripts/cann_lm_engine/sign_ohos.sh`。

### 绿判据

输出末尾出现 `=== build_ohos.sh 完成 ===`（方式一）或 `=== build_package_ohos.sh 完成 ===`（方式二），且 make 阶段无 error。

### 一键脚本（`build_package_ohos.sh`）

精简编排层，只做：`--clean`（删 `build_ohos` 全量重建）→ 预检 java + jar → 调 `build_ohos.sh`（编译+签名）→ 拷签名 .so 进 wheel → `pip wheel`（产 `cann_lm_engine-*+signed*.whl`）→ 收尾 `ls build_out/cann_lm_engine-*+signed*.whl`。SDK/clang-18/GTest/签名全部由 `build_ohos.sh` 内部处理。

## 5. 产物

产物平铺在 `build_ohos/`（默认；`BUILD_DIR` 环境变量可覆盖）：

| 文件 | 说明 |
|------|------|
| `libcann_lm_engine.so` | 引擎 SHARED 库（ELF AArch64，NEEDED=`libc.so`(musl)+`libc++_shared.so`） |
| `libc++_shared.so` | 鸿蒙 C++ 运行库，构建自动从 sysroot 拷贝 |
| `model_executor_ut` | 核心组件 UT（kv_cache/input_assembler/model_executor/dfx/tensor_manager） |
| `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding` | 各组件 UT |
| `bbit` | 黑盒集成（`--check` + 完整流程） |
| `api_smoke` / `c_api_smoke` | C++/C API 冒烟程序 |
| 上述每个的 `*_signed` / `*_signed.so` | 签名版（`build_ohos.sh` 自动产） |

共 **2 个 .so + 9 个可执行**，签名后各一份 `*_signed`。

### 确认是 OHOS musl 目标（非 glibc）

```bash
cd build_ohos
readelf -d libcann_lm_engine.so | grep NEEDED
# 期望:NEEDED libc++_shared.so, libc.so   ← libc.so 而非 libc.so.6 = musl/OHOS
readelf -h libcann_lm_engine.so | grep Machine
# 期望:Machine: AArch64
```
> 在 Linux 本机**无法直接运行**这些产物（它们链接 musl 的 `libc.so`，本机是 glibc）。运行验证必须在鸿蒙设备上。

### 产签名 wheel（`build_package_ohos.sh`）

`build_package_ohos.sh` 在编译+签名后自动将签名 `.so` 拷入 `cann_lm/lib/` 并 `pip wheel` 产出 `build_out/cann_lm_engine-*+signed*.whl`（内嵌 `.so` + `libc++_shared.so`）。

设备端：`pip install cann_lm_engine-*+signed*.whl` 后 `cllm generate --config testcase.json`（详见 [快速入门](../../QUICKSTART.md)）。C++ 测试（bbit/smoke）需单独 `bash build.sh --smoke` 构建，推到设备后 `./bbit --config testcase.json`。

## 6. 鸿蒙交叉编译三个必踩坑（首次环境准备）

> 路线A（完整 SDK + 自带 clang-15）：三坑全自动解（clang-15 用自己 resource dir 找 builtins；libc++/libcxx-ohos 头库路径两路线一致，toolchain 已配）。下表主要针对路线B（本机 clang-18）首次搭环境。

| # | 症状 | 根因 | 解法 | 路线A |
|---|------|------|------|------|
| 1 | `libclang_rt.builtins.a` / `crtbegin.o` 找不到 | clang-18 只在自己 resource dir(`/usr/lib/clang/18/...`)找 | 把 NDK 的 builtins/crt 拷进 `/usr/lib/clang/18/lib/aarch64-linux-ohos/`（见 §2.2 末） | **自动解**（clang-15 resource dir 自带） |
| 2 | `-lunwind` / `-lc++` 找不到 | 静态库在 `llvm/lib/aarch64-linux-ohos/`，不在 sysroot 标准路径 | 链接 flags 加 `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos`（`ohos.toolchain.cmake` 已加） | 同（路径一致） |
| 3 | `<__config_site>` not found | 用错 libc++ 头集：通用 `llvm/include/c++/v1/` 缺生成头 | 改用 `llvm/include/libcxx-ohos/include/c++/v1/`（带 `__config_site` 的 OHOS 配置版，toolchain 已配） | 同（路径一致） |

> 这三坑在 `ohos.toolchain.cmake` 与 SDK 提取阶段已处理；日常编译不会再遇，仅路线B首次搭环境时参考。

## 7. 常见问题

| # | 症状 | 原因 / 解决 |
|---|------|------------|
| 1 | `ERROR: OHOS_SDK_NATIVE 无效(无 sysroot 子目录)` | SDK 路径不对，确认 `${OHOS_SDK_NATIVE}/sysroot` 存在；`export OHOS_SDK_NATIVE=<native 目录绝对路径>` |
| 2 | `ERROR: SDK 无 llvm/bin/clang 且本机无 clang-18` | 两路线都没满足：路线A 把完整 SDK 放到 `cli_native`（自带 clang）；路线B `apt install clang-18 llvm-18` |
| 3 | CMake `FATAL_ERROR: GTest source tarball not found: /mnt/workspace/gtest-1.14.0.tar.gz` | OHOS 用 FetchContent 本地源码，不 `find_package(GTest)`。下载 googletest v1.14.0 到该默认路径，或 `-DGTEST_TAR=<你的 tarball 路径>` 覆盖 |
| 4 | 签名步骤报 `java 不在 PATH` | `apt install default-jdk`；确认 `java -version` 有输出 |
| 5 | 改了代码但产物 md5 没变 | 增量态用了陈货。`rm -rf build_ohos` 或 `build_package_ohos.sh --clean` 后重跑 |
| 6 | 推送被拒（master 保护分支） | master 禁直推。走功能分支 + MR：`git push origin HEAD:refs/heads/oxt/<功能名>` |
| 7 | `--gcc-toolchain=... argument unused` 大量警告 | OHOS 工具链在本机的已知行为，**无害**，不影响产物 |

## 8. 环境变量速查

| 变量 | 含义 | 默认/示例 |
|------|------|----------|
| `OHOS_SDK_NATIVE` | OHOS SDK `native/` 绝对路径（含 sysroot/llvm） | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | `build_ohos.sh` 输出目录 | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | 构建类型 | OHOS:Release |
| `GTEST_TAR` | GTest tarball 路径（CMake 变量） | `/mnt/workspace/gtest-1.14.0.tar.gz` |

## 9. 当前状态（2026-10-20 实测）

- ✅ **双 SDK 路线均验证通过**：
  - 路线A（完整 command-line-tools SDK + 自带 clang-15 ARM-native）：11 产物全绿，**坑#1 自动解**（无需补 `/usr/lib/clang/18`，无需本机 clang-18）
  - 路线B（精简 OpenHarmony 镜像 SDK + 本机 clang-18 + 补 builtins）：11 产物全绿
- ✅ `libcann_lm_engine.so` + `libc++_shared.so` 交叉编译通过（ELF AArch64，musl 目标；`readelf -d` NEEDED `libc.so` 非 `libc.so.6`）
- ✅ **9 个可执行交叉编译通过**（6 UT + bbit + 2 smoke；GTest 本地源码 FetchContent，静态 `add_test`）
- ✅ **全部产物签名完成**（`*_signed`，本机 java-21 + binary-sign-tool.jar `-selfSign 1`）
- ✅ **签名 wheel 产出**（`cann_lm_engine-*+signed*.whl` 内嵌 .so + libc++_shared.so）
- ✅ `model_executor_ut` / `ut_text_embedding` 已恢复（backend hiai 迁移完成）
- ⏳ ut_sampler/ut_request_handle/ut_data_type_utils 等旧 backend API UT 未随 hiai 迁移，暂未注册（源码重写属独立工作项）
- ⏳ 鸿蒙设备实跑验证（待设备；签名 wheel 推设备 pip install）

## 10. 提交代码

```bash
# 提交信息格式
git commit -m "DTS:8888888888888 Description:<一句话英文>
Details: <补充细节>
Change-Id: I<40位hex>"

# 生成 Change-Id（无 uuidgen 时）
python3 -c "import uuid; print('I' + uuid.uuid4().hex)"
```

推送到功能分支（master 是保护分支）：
```bash
git push origin HEAD:refs/heads/oxt/<你的功能名>
```
