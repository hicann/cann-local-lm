# cann-device-lm构建指南

简体中文 | [English](../en/build_guide.md)

> 本文档聚焦 **cann-device-lm构建**：构建前置（Part 1）→ 一键构建+安装+使用（Part 2）→ 速查/FAQ（Part 3）。环境依赖说明见 [env_dependencies.md](env_dependencies.md)，端到端用例见 [快速入门](QUICKSTART.md)。
>
> 以下命令除特别说明外，均在 **cann-device-lm仓根**执行（`cd /path/to/cann-device-lm`）。CANN安装路径以 `$HOME/Ascend` 为例（可按实际调整，后续所有 `source` 命令路径需同步替换）。

---

# Part 1: 构建前置

> 对齐 [env_dependencies.md](env_dependencies.md) Part 1「构建依赖」。运行依赖（P1 venv / P2 torch_npu）见env_dependencies Part 2。

## 系统级基础工具

| 工具 | 版本要求 | 备注 |
|------|---------|------|
| CMake | >= 3.18 | C++ 构建 |
| Python | >= 3.10 | whl打包（pip） |
| gcc / g++ | >= 8.0且 < 14.0 | `-Wall -Wextra -Werror` 零警告 |
| make | GNU Make | 随 `build-essential` |
| patchelf | | `.run` 包设 .so rpath（缺失跳过） |
| git-lfs | | 拉取HF模型权重 |
| pip | | whl打包 |
| **java (JRE)** | 任意 | 仅 `--pkg` [3/4] OHOS签名需要 |

一键安装（Ubuntu/Debian）：

```bash
apt install -y cmake build-essential python3 python3-venv python3-pip patchelf git-lfs default-jre
```

## CANN

CANN是**外部预置**（用户先source其 `set_env.sh`），本仓不构建。

| 组件 | 版本 | 获取 |
|------|------|------|
| CANN toolkit | >= 9.3.0 | [master下载](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) |
| CANN ops | 按SOC | Ascend910 → `Ascend910-ops`；kirinx90 → `ops-nn`/`ops-math`/`ops-transformer` |

安装（以下值为示例，按实际版本/SOC/架构调整）：

```bash
# 以 CANN 9.3.0 / aarch64 / Ascend910 / 安装到 ~/Ascend 为例
install_path=$HOME/Ascend
cann_version=9.3.0
arch=aarch64
soc_name=Ascend910

# 1. toolkit
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 2. ops 算子包
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 3. 建符号链接（GE 通过此路径找 platform_config）
ln -s ${install_path}/cann-${cann_version} ${install_path}/cann
```

安装后验证：

```bash
source $HOME/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH   # 期望: .../cann-9.3.0/cann-9.3.0
```

> 后续所有 `source $HOME/Ascend/cann/set_env.sh` 均指向此路径。如安装到其他路径，替换 `$HOME/Ascend` 为实际路径。

## OHOS交叉编译套件（仅 `build.sh --pkg` [3/4]需要）

> 仅 `build.sh --pkg` 的engine交叉编译阶段需要。日常 `scripts/build.sh cpp/wheel` 不需要。

### 下载

command-line-tools需人工浏览器下载（API返回403需登录）：

1. 访问 [华为开发者 - Command Line Tools](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)
2. 登录华为开发者账号
3. 下载 **Command Line Tools for Linux(ARM)**（如 `commandline-tools-linux-arm64-26.0.0.851.zip`）

### 内容

zip解压后包含：

| 目录 | 内容 |
|------|------|
| `command-line-tools/sdk/default/openharmony/native/llvm/bin/clang` | OHOS专用编译器（clang 15.0.4，ARM-native，自带builtins/crt） |
| `command-line-tools/sdk/default/openharmony/native/sysroot/` | OHOS musl sysroot（鸿蒙libc头文件/库） |
| `command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar` | 二进制签名工具（鸿蒙要求 .so签名） |
| `command-line-tools/sdk/default/openharmony/native/build/cmake/ohos.toolchain.cmake` | OHOS官方CMake工具链 |

### 安装与配置

```bash
cd /path/to/cann-device-lm

# 1. 解压 zip
unzip -oq commandline-tools-linux-arm64-*.zip -d /your/ohos-sdk-path

# 2. 设环境变量（指向 native 子目录）
export OHOS_SDK_NATIVE=/your/ohos-sdk-path/command-line-tools/sdk/default/openharmony/native

# 3. 拷签名工具到仓内 tools/（build.sh 硬检查 tools/binary-sign-tool.jar）
mkdir -p tools/
cp /your/ohos-sdk-path/command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar tools/

# 4. 安装 java（签名步骤需要）
apt install -y default-jre
```

### 验证

```bash
$OHOS_SDK_NATIVE/llvm/bin/clang --version   # 期望: clang 15.0.4
ls $OHOS_SDK_NATIVE/sysroot                 # 期望: 目录存在
ls tools/binary-sign-tool.jar                # 期望: 文件存在
```
---

## 运行时前置：P1 converter venv 与 P2 系统python

> ⚠️ **仅converter运行时需要（导出epair），`--pkg` 构建不需要。** 如只需产 `.run` 分发物，跳过此节直接到 Part 2。

### P1 与 P2 环境说明

cann-device-lm 的运行时分为两套互相隔离的环境：

| 环境 | 用途 | torch | torch_npu | 位置 | 说明 |
|------|------|-------|-----------|------|------|
| **P1 converter venv** | 导出epair + 量化Step B消费 | 2.9.0+cpu | **不装**（stub路径） | `deps/torch-delegate/.venv` | converter设备无关，CPU-only机器也能跑 |
| **P2 系统python** | NPU出字 + 量化Step A（dopt GPTQ） | 2.9.0 | 2.9.0.post8 | 系统python | 需NPU设备，装真torch_npu |

> **P1/P2隔离根因**：converter设计为可脱离NPU运行（CPU-only机器也能导出），为此对torchair与torch_npu **解耦**——torchair GE转换不依赖torch_npu。P1刻意不装torch_npu（走stub路径），P2装真torch_npu跑NPU；两者虽同为torch 2.9.0但venv与系统python隔离，依赖不同、环境不同，不能共用。

> **⚠️ 如果当前环境已存在 torch_npu（如系统python已装 torch_npu 2.9.0.post8）**：
> 强烈建议为 torch_delegate、torchair、cann-lm-converter 创建一套**独立 venv**（即 P1 converter venv）。直接在装了 torch_npu 的 python 环境中安装 converter 会导致 torchair 代理覆盖、`_LazyTorchair` 冲突、FX→GE 转换失败等问题。P1 venv 与 P2 系统python 隔离是 converter 的设计约束，不可省略。

### P1 converter venv 安装

| 组件 | 来源 |
|------|------|
| torch-delegate | 仓内 `torch_delegate/`（含 `build.sh`，自动创建 `.venv` + torch 2.9.0+cpu） |
| torchair | 见下方两种方式任选其一 |

```bash
source $HOME/Ascend/cann/set_env.sh
cd /path/to/cann-device-lm

# 1. torch-delegate：仓内源码构建（build.sh 自动创建 .venv [torch 2.9.0+cpu] + 编译 whl）
cp -a torch_delegate deps/torch-delegate
cd deps/torch-delegate && ./build.sh && .venv/bin/pip install --no-deps output/package/torch_delegate-*.whl

# 2. torchair（二选一）：
```

**方式 A：从源码编译**（`.so` 链 CANN stubs，converter 无需装 torch_npu）

```bash
# .venv/bin/pip 即上一步 torch-delegate build.sh 创建的 P1 venv
cd /path/to/cann-device-lm
git clone https://gitcode.com/Ascend/torchair.git deps/torchair && \
  cd deps/torchair && TARGET_PYTHON_PATH="$(realpath ../torch-delegate/.venv/bin/python)" bash build.sh -c && \
  ../torch-delegate/.venv/bin/pip install --no-deps output/torchair-*.whl
```

**方式 B：从 torch_npu whl 提取**（`.so` 来自 torch_npu 官方构建，须匹配 torch 版本）

```bash
VENV=deps/torch-delegate/.venv
PY_VER=$($VENV/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')

# 下载 torch_npu whl（不安装），解压取出 torchair
pip download torch_npu==2.9.0.post9 \
  --index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/ \
  --no-deps -d /tmp/torch_npu_whl
unzip -o /tmp/torch_npu_whl/torch_npu-*.whl 'torch_npu/dynamo/torchair/*' -d /tmp/torch_npu_extracted

# 拷到 P1 venv site-packages，目录名改为 torchair
cp -r /tmp/torch_npu_extracted/torch_npu/dynamo/torchair \
  $VENV/lib/$PY_VER/site-packages/torchair
```

> 两种方式产出的 torchair 含相同的 `.so`（`_torchair.so`/`_npu_graph_executor.so` 等），差异在编译环境：方式 A 用 CANN stubs 编译，方式 B 用 torch_npu 官方构建。converter venv 不装 torch_npu，方式 A 是推荐路径。

### 校验

```bash
bash scripts/build.sh check
```

---

# Part 2: 构建与部署

## 一键构建

```bash
cd /path/to/cann-device-lm
source $HOME/Ascend/cann/set_env.sh
export OHOS_SDK_NATIVE=<解压目录>/command-line-tools/sdk/default/openharmony/native

bash build.sh --pkg
```

> `--pkg` 自动下载+构建三方库（nlohmann_json / GTest / makeself，从gitcode镜像），首次运行下载，后续幂等跳过。

产物 `dist/` 下两个 .run：`cann_lm_compile_converter-*.run`（Ascend910编译机）+ `cann_lm_engine_ohos-*.run`（鸿蒙设备）。

## 安装

```bash
# Ascend910 编译机
bash dist/cann_lm_compile_converter-0.1.0-aarch64.run --extract=/opt/cann-device-lm

# 鸿蒙设备
bash dist/cann_lm_engine_ohos-0.1.0-aarch64.run --extract=/data/local/tmp/cann_lm_engine
```

## 使用

```bash
# Ascend910 编译 omc（安装机需先装 CANN + driver）
source $HOME/Ascend/cann/set_env.sh    # CANN（提供 lib64/driver/opp）
source /opt/cann-device-lm/set_env.sh         # cann-device-lm
/opt/cann-device-lm/cann_lm_compile <ep_json> <weight_info> [output_name]

# 鸿蒙推理
cd /data/local/tmp/cann_lm_engine && ./run.sh
```

> 端到端用例见 [快速入门](QUICKSTART.md)。

---

# Part 3: 速查

## 版本配置（`scripts/build.conf`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `CANN_VERSION` | `9.3.0` | CANN版本（>= 9.3.0） |
| `TORCH_VERSION` | `2.9.0` | P1 venv torch版本 |
| `SOC_VERSION` | `Ascend910_9362` | omc编译目标SOC |
| `CONVERTER_PYTHON` | `${TORCH_DELEGATE_HOME}/.venv/bin/python` | P1 venv python |
| `DOPT_PYTHON` | _(自动探测)_ | P2系统python（量化NPU用） |

> 任一变量已在环境中设置则不覆盖。切换版本/路径改 `scripts/build.conf` 或export覆盖。

## scripts/build.sh subcommand flags

> 根 `build.sh` 只有 `--pkg`（全量构建+打包）和 `-u`（UT编译+运行）。以下为 `scripts/build.sh` 的subcommand（日常开发用）。

| subcommand | flags | 说明 |
|-----------|-------|------|
| `cpp` | `--debug`/`--release`(默认) / `--asan` / `--no-tests` / `--no-compile-tool` / `--target <name>` | 构建 `libtorch_delegate_backend.so` + `cann_lm_compile` |
| `tests` | `--no-asan` / `--skip-run` | ASAN构建 + ctest |
| `third_party` | `--third_party_path=PATH` / `--jobs=N` | 自建nlohmann_json + GTest + makeself |
| `deps` | （无） | 装cann_lm wheels进P1 venv |
| `wheel` | （无） | cann_lm + cann_lm_engine whl |
| `package` | （透传） | 打 `cann_lm_compile-*.run` |
| `all` | （透传） | deps → third_party → cpp → wheel |

## 常见问题

### Q1: ASCEND_HOME_PATH未设置

```bash
source $HOME/Ascend/cann/set_env.sh
```

### Q2: torchair / torch_delegate未安装

见「运行时前置：P1 converter venv」节，按顺序装torch-delegate（产 `.venv`）→ torchair，然后 `bash scripts/build.sh deps && bash scripts/build.sh check`。

### Q3: omc编译退出码

`cann_lm_compile` 使用 `std::quick_exit(0)` 跳过 GE 库静态析构，正常退出码 0。

### Q4: CANN版本未找到

检查本地是否有CANN run包，没有则从 [master下载](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)。安装后 `ln -s cann-9.3.0 cann`。

### Q5: 换机器后DOPT_PYTHON要改吗

不用。`_env_common.sh` 的 `_resolve_dopt_python` 自动探测能 `import torch_npu` 的python。

### Q6: `$DOPT_PYTHON` 报 `No module named 'torch_npu'` / `_ctypes` undefined symbol

`LD_LIBRARY_PATH` 不含python `<prefix>/lib` → 加载系统旧libpython。`_env_common.sh` 已自动处理；手动跑时：

```bash
export LD_LIBRARY_PATH=$(dirname $(dirname $DOPT_PYTHON))/lib:${LD_LIBRARY_PATH:-}
```

### Q7: torchaudio版本不匹配 `undefined symbol: torch_library_impl`

torchaudio须与torch同版本。修复：

```bash
pip install torchaudio==2.9.0 --force-reinstall --no-deps --user   # 匹配 torch 2.9.0
```

> 详见 [env_dependencies.md](env_dependencies.md) §2.2。

### Q8: OHOS交叉编译大量 `clang++: warning: argument unused`

OHOS工具链cmake传递的冗余flag，**无害**，不影响编译结果。

## 相关文件

- `build.sh` — 根构建入口（`--pkg` 全量构建 + `-u` UT）
- `scripts/build.sh` — 日常开发构建入口（subcommand调度）
- `scripts/build.conf` — 版本号 + 路径默认值
- `scripts/set_env.sh` — converter+compile统一env
- `scripts/cann_lm_compile/build_cpp.sh` — C++ 构建
- `scripts/cann_lm_compile/package_run.sh` — 打 `.run` 包
- [env_dependencies.md](env_dependencies.md) — 环境依赖说明
- [快速入门](QUICKSTART.md) — 端到端用例
