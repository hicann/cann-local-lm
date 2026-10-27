简体中文 | [English](../../../en/cann_lm_engine/build_reference/build_win_guide.md)

# Windows 编译指南（WSL 内 OHOS 交叉编译）

> 目标：在 Windows PC 上通过 WSL（Windows Subsystem for Linux）用本机 clang-18 + 精简 OHOS SDK 交叉编译引擎 + bbit，产出签名版可部署件。
> 维护日期：2026-10-15。配套：`build_linux_guide.md`（同一流程的 Linux 本机版）、`build_guide.md`（总览）。

> **关键前提**：当前编译脚本（`scripts/cann_lm_engine/*.sh`）是**纯 Linux 逻辑**（本机 `clang-18` + `cmake -G "Unix Makefiles"` + `make`），**没有 Windows 原生 clang.exe/MinGW/`.exe` 分支**（旧的 clang.exe/shim 逻辑已删除）。因此 Windows 上必须通过 **WSL** 跑，与 Linux 指南完全一致的流程。**不要**在 Git Bash 原生跑（Git Bash 无 clang-18/make）。

---

## 1. 依赖清单（WSL 内）

| # | 依赖 | 获取方式 | 用途 |
|---|------|---------|------|
| 1 | WSL（Ubuntu 发行版） | Windows 10/11 自带（`wsl --install -d Ubuntu`）或 Microsoft Store | Linux 运行宿主（脚本依赖 clang-18/make/bash） |
| 2 | clang-18 / clang++-18 | WSL 内 `apt install clang-18 llvm-18` | 交叉编译器（`ohos.toolchain.cmake` 内 `find_program(clang-18)`） |
| 3 | cmake ≥ 3.16 + make | WSL 内 `apt install cmake build-essential` | 构建器（`cmake -G "Unix Makefiles"` + `make -j$(nproc)`） |
| 4 | JDK（1.8+，17/21 实测可用） | WSL 内 `apt install default-jdk` | 签名工具 `binary-sign-tool.jar` 需 java |
| 5 | OHOS SDK（精简 native 子树） | 见 §2 | sysroot + llvm/lib（builtins/libc++）+ 官方 ohos.toolchain.cmake |
| 6 | GTest 源码 tarball | 下载 github googletest v1.14.0 | OHOS CMakeLists FetchContent 本地引用（默认 `/mnt/workspace/gtest-1.14.0.tar.gz`） |
| 7 | binary-sign-tool.jar | `tools/binary-sign-tool.jar`（不入库，见 tools/README.md） | 产物自签名 |
| 8 | 代码仓 | CodeHub clone（见 §3） | 引擎 + bbit 源码 |

> 精简 SDK **无 `llvm/bin`（无自带 clang）、无 `build-tools`（无自带 cmake/ninja）**——编译器用本机 clang-18，构建器用系统 cmake + make，**不依赖 SDK 自带工具链二进制**。

---

## 2. 获取 OHOS SDK（WSL 内提取精简版）

本流程用的不是 DevEco Studio 装的完整 Windows SDK（只含 `.exe`，**不能**用于 WSL+clang-18 流程），而是从 OpenHarmony 开源镜像提取的**精简 Linux SDK**：只取 sysroot + llvm/lib（builtins/libc++ 静态库与头）+ 官方 `ohos.toolchain.cmake`。

```bash
# WSL 内
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# 外层 tar 内是嵌套 zip，提取 linux native 那个
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# 解出所需：sysroot + libc++ 头/库 + builtins + 官方 toolchain 参考
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
```

> 提取后目录结构（注意**无 `llvm/bin`、无 `build-tools`**）：
> ```
> $OHOS_SDK_NATIVE/
>   ├── sysroot/                                   ← 目标系统头/库（musl libc）
>   ├── build/cmake/ohos.toolchain.cmake           ← 官方工具链文件（参考）
>   ├── llvm/lib/aarch64-linux-ohos/               ← libc++/unwind 静态库
>   ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← builtins/crt
>   └── llvm/include/libcxx-ohos/include/c++/v1/    ← libc++ 头（带 __config_site）
> ```

### 一次性补丁：把 builtins/crt 装进 clang-18 的 resource dir

clang-18 只在自己 resource dir（`/usr/lib/clang/18/lib/aarch64-linux-ohos/`）找 `libclang_rt.builtins.a` / crtbegin/end，需把 SDK 的拷过去（详见 §5 坑⑥）：
```bash
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

### 验证 SDK
```bash
[ -d "$OHOS_SDK_NATIVE/sysroot" ] && echo "SDK 就绪: $OHOS_SDK_NATIVE"
clang-18 --version     # 本机 clang-18（非 SDK 自带）
cmake --version        # 本机 cmake
```

---

## 3. 获取代码

```bash
# WSL 内，clone 到 WSL 文件系统（不要放 /mnt/c/，见 §5 坑②）
git clone ssh://szv-y.codehub.huawei.com:2222/h00651112/cann_llm_open.git
cd cann_llm_open
```
- 写权限需找仓 owner 开通（否则报 CH.00905401）。
- clone 到**纯英文路径**（中文路径编译必炸，见 §5 坑③）。

---

## 4. 编译

脚本已从顶层 `scripts/` 迁到 `scripts/cann_lm_engine/`。

```bash
# WSL 内，每次新开 shell 设 SDK 路径（或写入 ~/.bashrc 持久化）
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native

cd <repo>

# 方式一：只编译 + 签名（产物平铺在 build_ohos/）
bash scripts/cann_lm_engine/build_ohos.sh

# 方式二：一键编译 + 签名 + 打包部署目录
bash scripts/cann_lm_engine/build_package_ohos.sh
# 加 --clean 自动清除 build_ohos 全量重建：
bash scripts/cann_lm_engine/build_package_ohos.sh --clean
```

### 绿判据
输出末尾出现 `=== build_ohos.sh 完成 ===`（方式一）或 `一键完成: engine_ohos.run`（方式二），且 **make 阶段无 error**。

### 编译流程说明（build_ohos.sh 内部 5 步）
1. **SDK 校验**：取 `OHOS_SDK_NATIVE`（默认 `/mnt/workspace/ohos_sdk/ohos_native/native`），校验 `sysroot` 子目录存在；检查本机 `clang-18`/`clang++-18`（`command -v`，缺失提示 `apt install clang-18 llvm-18`）。
2. **CMake configure**：`cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=<repo>/cann_lm/engine/ohos/ohos.toolchain.cmake -DOHOS_SDK_NATIVE=... -DOHOS_PLATFORM=OHOS -DCMAKE_BUILD_TYPE=Release -B build_ohos -S cann_lm/engine/ohos`（工具链内 `find_program(clang-18)`，**不**用 SDK 自带 clang）。
3. **make 编译**：`cmake --build build_ohos -j$(nproc)`（Unix Makefiles，**非 ninja**）。
4. **产物校验**：校验 11 个产物（见下表）齐备；`libc++_shared.so` 缺失则从 SDK 兜底拷贝。
5. **签名**：自动调 `sign_ohos.sh`，对 2 .so + 9 可执行逐个 `java -jar binary-sign-tool.jar sign ... -selfSign 1`。

### 产物（平铺在 `build_ohos/`，共 11 个 + 各自 `*_signed`）

| 文件 | 说明 |
|------|------|
| `libcann_lm_engine.so`（+`_signed.so`） | 引擎 SHARED 库（aarch64-linux-ohos / musl） |
| `libc++_shared.so`（+`_signed.so`） | C++ 运行库（构建自动从 sysroot 拷贝） |
| `model_executor_ut`（+`_signed`） | 核心组件 UT（kv_cache/input_assembler/model_executor/dfx/tensor_manager） |
| `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding`（各 +`_signed`） | 各组件 UT |
| `bbit`（+`_signed`） | 黑盒集成（`--check` + 完整流程） |
| `api_smoke` / `c_api_smoke`（各 +`_signed`） | C++/C API 冒烟程序 |

### 打包（方式二自动执行，或单独跑）
`package_ohos.sh` 从 `build_ohos/` 签名产物生成 `engine_ohos.run`：11 产物去 `_signed` 后缀、恢复可执行位、生成 `run.sh`（跑 6 UT + bbit + 2 smoke，自动 chmod + `LD_LIBRARY_PATH`）+ README.md。鸿蒙 PC 上解压后 `./run.sh` 一键跑全部。

---

## 5. 常见问题（Windows / WSL 特有）

| # | 症状 | 原因 / 解决 |
|---|------|------------|
| 1 | 在 Git Bash 原生跑报 `clang-18: command not found` / `make: not found` | 脚本是 Linux 逻辑，**必须用 WSL**，不要在 Git Bash 跑。`wsl` 进入 Ubuntu 后再执行 |
| 2 | 仓库放 `/mnt/c/...` 编译慢 / 可执行位丢失 | 跨文件系统（Windows ↔ WSL）I/O 慢且权限位易丢。clone 到 WSL 文件系统（`~/CANN-LM`，对应 `\\wsl$\<distro>\home\<user>\CANN-LM`） |
| 3 | 路径含中文 → 编译/链接失败 | WSL 内也避免中文路径，用纯英文 |
| 4 | 改了代码但产物没变 | 增量态用陈货。`rm -rf build_ohos` 或 `bash scripts/cann_lm_engine/build_package_ohos.sh --clean` 后重跑 |
| 5 | `ERROR: 未找到 clang-18/clang++-18` | WSL 内 `apt install clang-18 llvm-18`（ohos.toolchain.cmake 依赖本机 clang-18） |
| 6 | `libclang_rt.builtins.a` / `crtbeginS.o` 找不到 | 没做 §2 的 builtins 补丁。把 SDK 的 builtins/crt 拷进 `/usr/lib/clang/18/lib/aarch64-linux-ohos/` |
| 7 | `ERROR: OHOS_SDK_NATIVE 无效` | SDK 路径不对，确认 `sysroot/` 子目录存在，`export OHOS_SDK_NATIVE=<native 绝对路径>` |
| 8 | 签名报 `java 不在 PATH` | WSL 内 `apt install default-jdk`，确认 `java -version` 有输出 |
| 9 | `GTest source tarball not found` | 缺 `/mnt/workspace/gtest-1.14.0.tar.gz`。下载 github googletest v1.14.0 放该路径，或 `cmake -DGTEST_TAR=<路径>` |
| 10 | 大量 `argument unused: '--gcc-toolchain=...'` 警告 | OHOS 工具链在 Linux 上的已知行为，**无害**，不影响产物 |
| 11 | DevEco Studio 装的 SDK 用不了 | DevEco Windows SDK 只含 `.exe`，本流程需精简 Linux SDK（含 sysroot + llvm/lib，见 §2） |
| 12 | 推送被拒 | master 保护分支，走功能分支 + MR（见 §6） |

---

## 6. 提交代码

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
