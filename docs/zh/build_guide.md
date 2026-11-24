# cann-local-lm构建指南

简体中文 | [English](../en/build_guide.md)

> 本文档聚焦 **cann-local-lm构建**：构建前置（Part 1）→ 一键构建+安装+使用（Part 2）→ 速查/FAQ（Part 3）。环境依赖说明见 [env_dependencies.md](env_dependencies.md)，端到端用例见 [快速入门](QUICKSTART.md)。
>
> 以下命令除特别说明外，均在 **cann-local-lm仓根**执行（`cd /path/to/cann-local-lm`）。CANN安装路径以 `$HOME/Ascend` 为例（可按实际调整，后续所有 `source` 命令路径需同步替换）。

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

CANN是**外部预置**（用户先source其 `set_env.sh`），本仓不构建,当前只支持x86版本。

| 组件 | 版本 | 获取 |
|------|------|------|
| CANN toolkit mobile station | >= 9.3.0 | [cann-mobile-station](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/Ascend-cann-toolkit_9.3.0_linux-x86_64-mobile-station.run) |
| CANN ops | KirinX90 | [KirinX90](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/Ascend-cann-kirinx90-ops_9.3.0_linux-x86_64.run) |

安装（以下值为示例，按实际版本/SOC/架构调整）：

```bash
# 以 CANN 9.3.0 / x86_64 / KirinX90 / 安装到 ~/Ascend 为例
install_path=$HOME/Ascend
cann_version=9.3.0
arch=x86_64
soc_name=kirinx90

# 1. toolkit（包名带 -mobile-station 后缀）
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run --install --install-path=${install_path}

# 2. ops 算子包
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 3. 建符号链接（GE 通过此路径找 platform_config）
ln -s ${install_path}/cann-${cann_version} ${install_path}/cann
```

安装后验证：

```bash
source $HOME/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH   # 期望: .../cann-9.3.0（单层，指向 toolkit 安装根）
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
cd /path/to/cann-local-lm

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
$OHOS_SDK_NATIVE/llvm/bin/clang --version   # 期望: 文件存在
ls $OHOS_SDK_NATIVE/sysroot                 # 期望: 目录存在
ls tools/binary-sign-tool.jar                # 期望: 文件存在
```
---

## 运行时前置：P1 converter venv 与 P2 系统python

> ⚠️ **仅converter运行时需要（导出epair），`--pkg` 构建不需要。** 如只需产 `.run` 分发物，跳过此节直接到 Part 2。

### P1 与 P2 环境说明

cann-local-lm 的运行时分为两套互相隔离的环境：

| 环境 | 用途 | torch | torch_npu | 位置 | 说明 |
|------|------|-------|-----------|------|------|
| **P1 converter venv** | 导出epair + 量化Step B消费 | 2.9.0+cpu | **不装**（stub路径） | `deps/torch-delegate/.venv` | converter设备无关，CPU-only机器也能跑 |
| **P2 系统python** | NPU出字 + 量化Step A（dopt GPTQ） | 2.9.0 | 2.9.0.post8 | 系统python | 需NPU设备，装真torch_npu |

> **P1/P2隔离根因**：converter设计为可脱离NPU运行（CPU-only机器也能导出），为此对torchair与torch_npu **解耦**——torchair GE转换不依赖torch_npu。P1刻意不装torch_npu（走stub路径），P2装真torch_npu跑NPU；两者虽同为torch 2.9.0但venv与系统python隔离，依赖不同、环境不同，不能共用。

> **⚠️ 如果当前环境已存在 torch_npu（如系统python已装 torch_npu 2.9.0.post8）**：
> 强烈建议为 torch_delegate、torchair、cllm-converter 创建一套**独立 venv**（即 P1 converter venv）。直接在装了 torch_npu 的 python 环境中安装 converter 会导致 torchair 代理覆盖、`_LazyTorchair` 冲突、FX→GE 转换失败等问题。P1 venv 与 P2 系统python 隔离是 converter 的设计约束，不可省略。

### P1 converter venv 安装

| 组件 | 来源 |
|------|------|
| torch-delegate | 仓内 `torch_delegate/`（含 `build.sh`，自动创建 `.venv` + torch 2.9.0+cpu） |
| torchair | 见下方两种方式任选其一 |

```bash
source $HOME/Ascend/cann/set_env.sh
cd /path/to/cann-local-lm

# 1. torch-delegate：仓内源码构建（build.sh 自动创建 .venv [torch 2.9.0+cpu] + 编译 whl）
mkdir -p deps && cp -a torch_delegate deps/torch-delegate
cd deps/torch-delegate && ./build.sh && .venv/bin/pip install --no-deps output/package/torch_delegate-*.whl

# 2. torchair（二选一）：
```

**方式 A：从源码编译**（converter 无需装 torch_npu）

```bash
# .venv/bin/pip 即上一步 torch-delegate build.sh 创建的 P1 venv
cd /path/to/cann-local-lm
git clone https://gitcode.com/Ascend/torchair.git deps/torchair
cd deps/torchair

# ⚠️ 两个变量都要设，且不能用 realpath：
#   1. realpath 会把 .venv/bin/python symlink 解析成系统 python 路径，丢失 venv 激活 → 找到系统 torch 而非 venv torch。
#      用 $(cd ../torch-delegate && pwd)/.venv/bin/python 取绝对路径但保留 symlink。
#   2. torchair build.sh main() 在 TARGET_PYTHON_PATH 为空时覆盖 PYTHON_BIN_PATH=$(which python3)（build.sh:136-139），
#      故必须同时设 TARGET_PYTHON_PATH 阻止覆盖。
#   另：清掉 stale tools/TORCH_VERSION 缓存（存在则 configure 跳过重新检测，configure:13）。
rm -f tools/TORCH_VERSION tools/TORCH_INSTALLED_PATH tools/PYTHON_BIN_PATH tools/COMPILE_FLAGS
rm -rf build/CMakeCache.txt build/CMakeFiles
VENV_PY="$(cd ../torch-delegate && pwd)/.venv/bin/python"
PYTHON_BIN_PATH="$VENV_PY" TARGET_PYTHON_PATH="$VENV_PY" bash build.sh -c

# 装 whl（--no-deps 避免 pip 拉入 torch_npu；torchair 运行时依赖需手动补）
../torch-delegate/.venv/bin/pip install --no-deps output/torchair-*.whl
../torch-delegate/.venv/bin/pip install packaging cloudpickle
```

> **验证 configure 选对 torch**：构建后 `cat tools/TORCH_VERSION` 应为 `20900`（torch 2.9.0）。若为 `20701`（torch 2.7.1）说明命中了系统 python 的 torch，`_torchair.so` ABI 不匹配，import 时报 `undefined symbol: _ZN2at14RecordFunction6beforeEPKcl`——需清除 `tools/TORCH_VERSION` 缓存后用正确的 `PYTHON_BIN_PATH` 重新构建。

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

> 两种方式产出的 torchair 含相同的 `.so`（`_torchair.so`/`_npu_graph_executor.so` 等），差异在编译环境：方式 A 从源码编译（链本机 CANN），方式 B 用 torch_npu 官方构建。converter venv 不装 torch_npu。
>
> **方式 A 补依赖**：`--no-deps` 跳过了 torchair 声明的运行时依赖（`protobuf`/`decorator`/`psutil`/`scipy` 通常已由 CANN site-packages 或 torch 自动拉入，但 `packaging`/`cloudpickle` 需手动补——`torchair/__init__.py` 首行即 `from packaging import version`，缺失会导致 `ModuleNotFoundError`）。若不愿逐个补，也可 `pip install -r scripts/requirements-p1.txt` 一次性装齐。

### 校验

```bash
# 需先 source $HOME/Ascend/cann/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 deps/torch-delegate/.venv/bin/python -c "import torchair; print('OK', torchair.__file__)"
# 预期: OK /mnt/workspace/.../deps/torch-delegate/.venv/lib/python3.12/site-packages/torchair/__init__.py
```

> 此处仅验证 torchair import 是否成功（捕获 ABI 不匹配 / 缺依赖等问题）。完整基底校验（含 cann_local_lm / third_party）见 Part 2 末尾「[环境校验](#环境校验)」。

---

# Part 2: 构建与部署

## 一键构建

```bash
cd /path/to/cann-local-lm
source $HOME/Ascend/cann/set_env.sh
export OHOS_SDK_NATIVE=<解压目录>/command-line-tools/sdk/default/openharmony/native

bash build.sh --pkg
```

> `--pkg` 自动下载+构建三方库（nlohmann_json / GTest / makeself，从gitcode镜像），首次运行下载，后续幂等跳过。

产物落 `build_out/`（由 `build.sh` 的 `DIST` 变量控制，默认 `${REPO}/build_out`，可环境变量覆盖）：
- `cann_local_lm_converter-*.run`（Ascend910 编译机，**必备**）—— 含 `cann_lm_compile` + `libtorch_delegate_backend.so` + `libcustom_pass_plugin.so` + `op_cpu/` + `op_pass/` + cann_lm whl + set_env.sh
- `cann_local_lm_engine-*.whl`（engine python ctypes 绑定 whl，纯 python 不含 .so）
- `deploy_ohos/`（engine OHOS 交叉编译产物：signed `libcann_lm_engine.so` + UT/bbit/smoke，**需 OHOS SDK**；缺 SDK 时该步跳过，不影响 910 OM 编译流程）

> converter whl 已打进 `.run` 的 `wheels/` 目录，打包后从 `build_out/` 移除。

## 安装

```bash
# Ascend910 编译机
bash build_out/cann_local_lm_converter-0.1.0-aarch64.run --extract=/opt/cann-local-lm

# 鸿蒙设备（engine 签名 wheel，--pkg 或 build_package_ohos.sh 产出）
pip3 install build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl
```

## 使用

```bash
# Ascend910 编译 omc（安装机需先装 CANN + driver）
source $HOME/Ascend/cann/set_env.sh    # CANN（提供 lib64/driver/opp）
source /opt/cann-local-lm/set_env.sh         # cann-local-lm
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> [output_name]

# 鸿蒙推理（完整部署流程见 [快速入门](QUICKSTART.md)）
# cllm generate --config testcase.json --prompt "你好"
# cllm chat --config testcase.json
```

> **权值分离**：非 Kirin SOC 下 `/opt/cann-local-lm/set_env.sh` 不设 `GE_EXTERNAL_WEIGHT`，编译出的 OM **内嵌权重**（如 gemma4 5B 约 4.5G/段）。若需更小的 OM + 外置权值文件，编译前 `export GE_EXTERNAL_WEIGHT=2`，产物为 `<out>.om`（数十 M）+ 同目录 `weight/`（外置权值，运行时加载）。Kirin 系 SOC 自动设 `=2`，见下节。

> ⚠️ **Kirin 平台强制要求**：模型权重超过 **2 GB** 时**必须**使用外置权重（`GE_EXTERNAL_WEIGHT=2`），即产物为 `<out>.omc` + 同目录 `weight/`，运行时两者一起加载。内嵌模式（`GE_EXTERNAL_WEIGHT=0`）在 Kirin 平台仅适用于 2 GB 以内的模型；超过该规模请勿设 `=0`。Kirin 系 SOC 下 `set_env.sh` 已自动设 `=2`，正常无需干预——只有在手工 `export GE_EXTERNAL_WEIGHT=0` 时才需要确认模型规模。

## 编译环境变量（GE / mobile）

`set_env.sh` 会按 `SOC_VERSION` 自动设置下列变量：**Kirin 系（`KirinX90` / `Kirin9030` / …）自动进入 mobile 编译链**，其余 SOC 不设 mobile 相关项。一般无需手工设置，仅在需要覆盖默认值时手动 `export`。

| 变量 | 默认 | 作用 | 何时需要改 |
|------|------|------|-----------|
| `SOC_VERSION` | 包内 `KirinX90`；独立 `cann_lm_compile` 包 `Ascend910_9362` | 目标 SoC，传给 GE 的 `ge.socVersion`。**Kirin 系 → mobile 编译链**（`.omc` + `MobileModelHelper` 路径）；其余 → 通用 `.om` 路径 | 换目标芯片时。这是唯一的模式开关，设成非 Kirin 值即关闭 mobile 分支 |
| `GE_FUSION_SWITCH_FILE` | 包内 `fusion_autoub_off.cfg` | GE 图融合开关配置。默认关闭 auto-UB 融合 | 需要开融合调优时指向自定义 cfg；调试融合问题时临时 `unset` 用 GE 默认 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0` | 阻止 torch 自动加载 npu backend | 不要改。P1 converter venv 未装 torch_npu，置 1 会 import 失败 |
| `GE_EXTERNAL_WEIGHT` | 包内：mobile 下 `2`，非 mobile 不设；开发态 `set_env.sh`：恒为 `2` | `2` = 权值外置（`<out>.omc` + 同目录 `weight/`，运行时加载）；`0` = 权值内嵌进 omc | **Kirin 平台模型权重 > 2 GB 时必须为 `2`**（内嵌仅适用 ≤ 2 GB）。非 Kirin 平台需要单文件分发、不介意体积时可设 `0` |
| `GE_MOBILE_WEIGHT_SHARE_OPS` | `MatMulV3,QuantBatchMatmulV4,RmsNorm` | mobile 共享权重算子白名单（GE `[Mobile]` 路径读取）。列出的算子若输入来自同一 `FileConstant`，只存一份权重 | 出现权重重复导致产物过大时，把对应算子名加进来；排查共享权重问题时置空回退 GE 内置默认 |
| `ASCEND_CUSTOM_OPP_PATH` | mobile 下 `<安装目录>/op_cpu` | GE python 自定义算子插件目录，注册 `Bitcast` / `TransData` 的 HostCpu kernel（int4 布局常量折叠） | 不要改。缺失会导致 int4 量化图编译失败（`TransData` 无 kernel） |
| `ASCEND_GE_PY_PASS_PATH` | mobile 下 `<安装目录>/op_pass` | GE python pass 插件目录，注册 `QbmX2TransDataPass`（`QuantBatchMatmulV4` 的 x2 int4 `ND→FRACTAL_Z`） | 不要改。缺失会报 `FRACTAL_Z infershape requires x2 4D shape` |

典型用法：

```bash
# 量化 mobile 编译（Kirin）：只需设 SOC，其余自动
source $HOME/Ascend/cann/set_env.sh
export SOC_VERSION=KirinX90
source /opt/cann-local-lm/set_env.sh
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> <out>   # → <out>.omc + weight/

# 通用 Ascend910 编译：设非 Kirin SOC 即关闭 mobile 分支
export SOC_VERSION=Ascend910_9362
source /opt/cann-local-lm/set_env.sh
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> <out>   # → <out>.om

# 内嵌权重（单文件分发；Kirin 平台仅限模型 ≤ 2 GB）
export GE_EXTERNAL_WEIGHT=0        # source set_env.sh 前后设均可（默认值只在未设时生效）
```

> 开发态（源码树，未打 `.run` 包）用法完全一致，只是把 `source /opt/cann-local-lm/set_env.sh` 换成 `source scripts/cann_lm_compile/set_env.sh`，插件目录自动指向源码树 `cann_local_lm/converter/export/{op_cpu,op_pass}`。

> 端到端用例见 [快速入门](QUICKSTART.md)。

## 环境校验

完成上述构建（`--pkg` 或 `deps` + `cpp`）后，运行基底校验确认所有组件就绪：

```bash
bash scripts/build.sh check
```

预期输出（全部 ✓）：

```
  ✓ CANN ...
  ✓ P1 venv ...
  ✓ torch (2.9.0+cpu)
  ✓ torchair
  ✓ torch_delegate
  ✓ cann_local_lm (converter)
  ✓ third_party json
```

> 若 `cann_local_lm (converter)` 显示 ✗：运行 `bash scripts/build.sh deps` 装入 P1 venv（仅装 converter，不含 engine）。
> 若 `third_party` 显示 ✗：运行 `bash build.sh third_party`（或 `--pkg` 已自动构建）。
> 若 `torchair` 显示 ✗：见 [Q2](#q2-torchair--torch_delegate未安装)。

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
| `wheel` | （无） | cann_local_lm + cann_lm_engine whl |
| `package` | （透传） | 打 `cann_lm_compile-*.run` |
| `all` | （透传） | deps → third_party → cpp → wheel |

## 常见问题

### Q1: ASCEND_HOME_PATH未设置

```bash
source $HOME/Ascend/cann/set_env.sh
```

### Q2: torchair / torch_delegate未安装

见「运行时前置：P1 converter venv」节，按顺序装torch-delegate（产 `.venv`）→ torchair，然后 `bash scripts/build.sh deps && bash scripts/build.sh check`。

若 torchair 已装但 `check` 仍报 `✗ torchair`，通常是 `import torchair` 失败（check 用 `2>/dev/null` 静默吞错）。手动复现定位：

```bash
source $HOME/Ascend/cann/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 deps/torch-delegate/.venv/bin/python -c "import torchair"
```

常见原因：
- **`ModuleNotFoundError: No module named 'packaging'`**：方式 A 用 `--no-deps` 装的 torchair 缺运行时依赖，`pip install packaging cloudpickle` 补齐。
- **`ImportError: undefined symbol: _ZN2at14RecordFunction6beforeEPKcl`**：torchair 编译时用了错误版本的 torch（如系统 python 的 torch 2.7.1，而非 venv 的 2.9.0）。检查 `cat deps/torchair/tools/TORCH_VERSION`——应为 `20900`，若为 `20701` 则需清 `tools/TORCH_VERSION` 缓存 + 用 `PYTHON_BIN_PATH`（非 `TARGET_PYTHON_PATH`）重新构建，详见方式 A 注释。

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


### Q9: kirin平台模型调试工具

[model_run_tool](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/CANN-Kit-tools-7.0.1.0.run) 详细使用说明请./model_run_tool -h 。

### Q10: prefill、decode权重共享说明
 `GE_EXTERNAL_WEIGHT` 、`GE_MOBILE_WEIGHT_SHARE_OPS` 两个参数必须配合使用。其中 `GE_EXTERNAL_WEIGHT` ，kirin平台只支持 `GE_EXTERNAL_WEIGHT`=0、2；`GE_MOBILE_WEIGHT_SHARE_OPS` 参数仅kirin支持。且该功能仅PC 7.0 B115SPX及以后版本支持，其余版本仅做兼容性适配。

### Q11 量化场景必须设置自定义融合算子
`ASCEND_CUSTOM_OPP_PATH`、`ASCEND_GE_PY_PASS_PATH` 当前配套版本的量化算子仅支持`FRACTAL_Z`格式，权重转换和重排需在CPU侧去实现。


## 相关文件

- `build.sh` — 根构建入口（`--pkg` 全量构建 + `-u` UT）
- `scripts/build.sh` — 日常开发构建入口（subcommand调度）
- `scripts/build.conf` — 版本号 + 路径默认值
- `scripts/set_env.sh` — converter+compile统一env
- `scripts/cann_lm_compile/build_cpp.sh` — C++ 构建
- `scripts/cann_lm_compile/package_run.sh` — 打 `.run` 包
- [env_dependencies.md](env_dependencies.md) — 环境依赖说明
- [快速入门](QUICKSTART.md) — 端到端用例
