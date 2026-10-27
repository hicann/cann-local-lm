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

> GTest由 `--pkg` 自动从gitcode下载到 `output/third_party/gtest/`，无需手动设置。

---

## 运行时前置：P1 converter venv

> ⚠️ **此节仅converter运行时需要（导出epair），`--pkg` 构建不需要**。`build_converter` fallback系统pip打包（`--no-deps` 纯源码），`build_cpp` 零torch引用，`build_engine` 用OHOS clang。如只需产 `.run` 分发物，可跳过此节直接到Part 2。
>
> P1 venv属**运行依赖**，见 [env_dependencies.md](env_dependencies.md) §2.1。cann-device-lm **不本仓构建** torchair / torch-delegate，它们是前置条件。

| 组件 | 作用 | git源 |
|------|------|--------|
| torch-delegate | `torch_delegate` whl + `.venv`（torch 2.9.0+cpu） | 已合入cann-device-lm仓 `torch_delegate/` 目录（`gitcode.com/cann/cann-device-lm`） |
| torchair | `torchair` whl（FX→GE IR转换） | `gitcode.com/Ascend/torchair.git` |

### 方式一：从源码构建安装

> 完整依赖列表见 [`scripts/requirements-p1.txt`](../../scripts/requirements-p1.txt)。

```bash
source $HOME/Ascend/cann/set_env.sh

# 1. torch-delegate（产 .venv + torch_delegate whl）
cd /path/to/cann-device-lm
git clone https://gitcode.com/cann/cann-device-lm.git deps/torch-delegate  # 或直接用 cann-device-lm 仓内 torch_delegate/ 目录
cd deps/torch-delegate && ./build.sh    # build.sh 自动创建 .venv（torch 2.9.0+cpu）+ 编译 whl
.venv/bin/pip install --force-reinstall --no-deps output/package/torch_delegate-*.whl

# 2. torchair（TARGET_PYTHON 指向 P1 venv, ABI 匹配）
cd /path/to/cann-device-lm
deps/torch-delegate/.venv/bin/pip install wheel numpy protobuf
git clone https://gitcode.com/Ascend/torchair.git deps/torchair
cd deps/torchair
TARGET_PYTHON_PATH="$(realpath ../torch-delegate/.venv/bin/python)" bash build.sh -c
../torch-delegate/.venv/bin/pip install --force-reinstall --no-deps output/torchair-*.whl
```

### 方式二：用whl包安装

已有预编whl（方式一产出或发布包附带），不clone torch-delegate（无自动 `.venv`），需手动建venv：

```bash
python3 -m venv /path/to/venv          # 手动创建 venv（方式一由 torch-delegate build.sh 自动创建）
source /path/to/venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy protobuf
pip install torch_delegate-0.1.0-py3-none-any.whl torchair-0.1-py3-none-any.whl
export CONVERTER_PYTHON=/path/to/venv/bin/python
```

### 校验前置就绪

```bash
source $HOME/Ascend/cann/set_env.sh
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

### Q3: omc编译退出非零但omc已产出

GE teardown触发glibc "corrupted size vs. prev size" → Aborted，但omc在此前已产出。以omc文件是否存在判定成功（`run_compile.sh` 已用 `|| true` 处理）。

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
