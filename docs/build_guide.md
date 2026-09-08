# CANN-LM 构建指南

本文档描述 CANN-LM 的端到端构建流程，包括环境准备、三套运行环境（① 导出 / ② OM 编译 / ③ NPU 验证）、构建命令、安装包使用和常见问题。

CANN-LM 将 HuggingFace 模型导出为 epair 格式，经 C++ 后端加载编译为 OM，最终在 Ascend NPU 上推理。构建链跨 Python（converter/engine whl）与 C++（backend .so + OM 编译工具 + 测试）两侧，依赖上游组件 torchair 与 torch-delegate。

## 前置条件

| 工具 | 版本要求 | 备注 |
|------|---------|------|
| CMake | >= 3.18 | C++ 统一构建入口 |
| Python | >= 3.10 | 使用 `dict[str, Any]` 等内置泛型语法 |
| gcc / g++ | >= 8.0 且 < 14.0 | `-Wall -Wextra -Werror` 零警告 |
| make | GNU Make | 随 `build-essential` 安装 |
| patchelf | | 打 `.run` 包时设置 .so rpath（缺失则跳过，靠 `set_env.sh` 的 `LD_LIBRARY_PATH`） |
| makeself | >= 2.5.0 | 打 `.run` 自解压包；由 torch-delegate 三方库构建产出 |
| git-lfs | | `pull_models.sh` 拉取 HF 模型权重 |
| torch (CPU) | 2.9.0+cpu | 由上游 torch-delegate `.venv` 提供 |
| CANN | 9.2.0（master） | 见下方安装说明 |
| libasan | 与 gcc 版本匹配 | ASAN 运行时，`build_cpp.sh --asan` 使用；包名随 gcc 主版本变化（gcc 13 对应 `libasan8`） |

一键安装依赖（Ubuntu/Debian）：

```bash
# 基础构建依赖 + ASAN 运行时 + patchelf + makeself 前置（libasan8 包名随 gcc 主版本变化）
apt install -y cmake build-essential python3 python3-venv python3-pip libasan8 patchelf git-lfs
```

## 安装 CANN 包

### 场景1：体验 master 版本能力或基于 master 版本进行开发

请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

1. 安装 CANN Toolkit 开发套件包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```

    - `${cann_version}`：表示 CANN 包版本号。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径。root 用户默认安装在 `/usr/local/Ascend` 目录，普通用户默认安装在 `${HOME}/Ascend` 目录。

2. 安装 CANN ops 算子包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```

    其中 `${soc_name}` 表示 NPU 型号名称（如 `Ascend910`）。

3. 建立版本符号链接，使 GE 能找到 platform_config：

    ```bash
    ln -s ${install_path}/cann-${cann_version} ${install_path}/cann
    ```

### 场景2：体验已发布版本能力或基于已发布版本进行开发

如果您想体验**官网正式发布的 CANN 包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本 CANN 软件包（仅支持 CANN 9.1.0 及后续版本）进行安装。

### 场景3：本机已有 CANN 安装包

若 `/mnt/workspace/Ascend_tar` 下有 CANN run 包，直接用其安装；缺失时从[master 下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)下载。

### 驱动与固件（可选，仅运行样例依赖）

驱动与固件的下载和安装操作请参考《CANN软件安装指南》中"准备软件包"和"安装NPU驱动和固件"章节。驱动与固件是运行样例依赖，若仅编译环境（①②），可以不安装。

## 安装上游组件

CANN-LM 依赖两个上游组件（路径默认值见 `scripts/build.conf`，可由环境变量 `TORCHAIR_ROOT` / `TORCH_DELEGATE_HOME` 覆盖）：

| 组件 | 源码路径 | 作用 |
|------|---------|------|
| torchair | `${TORCHAIR_ROOT}`（默认 `/mnt/workspace/torchair`） | torchair GE 转换（FX→GE），编译产出 stubs 供 converter 在 CPU 上跑 |
| torch-delegate | `${TORCH_DELEGATE_HOME}`（默认 `/mnt/workspace/torch-delegate-harmony`） | 提供 `.venv`（torch 2.9.0+cpu）+ `libtorch_delegate_backend.so` + 三方库（nlohmann_json/GTest/makeself）+ epair 序列化 |

**构建顺序**：先 torch-delegate（产出 `.venv` + `third_party`），再 torchair（构建时指定 P1 venv python 为 target，stubs 与 venv 的 torch ABI 匹配）。也可直接用 CANN-LM 顶层 `bash scripts/build.sh deps` 一键串装（幂等，已装则跳），下文展开手动构建细节。

> **关键设计**：torchair 须以 `NO_ASCEND_SDK=1` 源码编译，生成的 `build/stubs/`（`libtorch_npu.so`/`libge_runner.so`/`libhccl_stub.so` 等）供 converter 在无 NPU 设备的 CPU 上运行 GE 转换；converter venv **不装 torch_npu**（走 stub 路径，见 [CLI 设计](cann_lm_converter/cli/cli_design.md)「converter 与 torch_npu 的关系」）；torch-delegate 的 `.venv` 提供 CPU torch + torch cmake 配置（C++ 构建链期引用，但 backend .so 不链 libtorch）。

### 安装 torch-delegate

torch-delegate 的一站式 `build.sh` 自动完成：环境检查 → 创建 `.venv`（torch 2.9.0+cpu）→ 编译三方库（nlohmann_json/GTest/makeself）→ 编译 C++ delegate 库 → 构建 Python whl → 打包 `.run`。

```bash
source /home/developer/Ascend/cann/set_env.sh   # 前置：设 ASCEND_HOME_PATH
cd /mnt/workspace/torch-delegate-harmony

./build.sh                  # 默认 release，自动创建 .venv
# ./build.sh --debug        # debug 构建
# ./build.sh --pre_check    # 仅校验环境，不构建
```

构建完成后产物：

| 产物 | 路径 | CANN-LM 用途 |
|------|------|--------------|
| P1 venv（torch 2.9.0+cpu） | `.venv/` | ① converter 运行环境 + C++ 构建链期 torch cmake 来源 |
| 三方库 | `output/third_party/{json,gtest,makeself}/` | C++ 测试 + `.run` 打包 |
| Python whl | `output/package/torch_delegate-*.whl` | 装进 P1（`torch_delegate` 包，epair 序列化） |
| `.run` 安装包 | `output/package/torch_delegate-*.run` | 独立部署（可选） |

将 torch_delegate whl 装进 P1 venv（CANN-LM ① env 用）：

```bash
.venv/bin/pip install --force-reinstall --no-deps output/package/torch_delegate-*.whl
```

`build.sh` 参数：

| 参数 | 说明 |
|------|------|
| `--debug` / `--release` | 构建类型（默认 release，互斥） |
| `--skip_third_party` | 跳过三方库编译（已有 `output/third_party` 时用） |
| `--with_libtorch_npu` | 编译 libtorch_npu（默认 OFF，converter 不需要） |
| `--system_python` | 使用当前 Python 环境，不创建 `.venv`（适用于 conda/CI） |
| `--cann_3rd_lib_path=PATH` | 指定三方库路径 |
| `--output_dir=PATH` | 产物输出路径（默认 `output/package`） |
| `--pre_check` | 仅校验环境并退出 |
| `--help` | 帮助信息 |

> 更多细节见 torch-delegate 仓库的 `docs/build.md`（`.run` 安装包使用、C++ 项目接入、ABI 兼容性等）。

### 安装 torchair

torchair 须以 `NO_ASCEND_SDK=1` 编译（不依赖真实 NPU SDK），生成 CPU 可用的 whl + stubs。构建时通过 `TARGET_PYTHON_PATH` 指定目标 Python（须指向 torch-delegate 的 P1 venv python，使 stubs 与 venv 的 torch ABI 匹配）。

```bash
source /home/developer/Ascend/cann/set_env.sh   # 前置：torchair cmake 需 CANN 头
cd /mnt/workspace/torchair

# configure（检测 Python/torch 环境；首次或环境变更时跑）
TARGET_PYTHON_PATH=/mnt/workspace/torch-delegate-harmony/.venv/bin/python \
  NO_ASCEND_SDK=1 bash ./configure

# 编译（CI 模式，NO_ASCEND_SDK=1；-c 仅编译不安装）
TARGET_PYTHON_PATH=/mnt/workspace/torch-delegate-harmony/.venv/bin/python \
  bash build.sh -c
```

构建完成后产物：

| 产物 | 路径 | CANN-LM 用途 |
|------|------|--------------|
| Python whl | `output/torchair-0.1-py3-none-any.whl` | 装进 P1（`torchair` 包，GE 转换） |
| stubs | `build/stubs/*.so` | ① converter env 的 `LD_LIBRARY_PATH` 前置（`libtorch_npu.so`/`libge_runner.so`/`libhccl_stub.so` 等，给 GE 引擎提供 C 符号） |

将 torchair whl 装进 P1 venv：

```bash
/mnt/workspace/torch-delegate-harmony/.venv/bin/pip install --force-reinstall --no-deps \
  output/torchair-0.1-py3-none-any.whl
```

也可用 `bash build.sh -i` 一步完成编译 + 安装（安装到 `TARGET_PYTHON_PATH` 对应环境）。`build.sh` 参数：

| 参数 | 说明 |
|------|------|
| `-c` | CI 模式编译（须配合 `NO_ASCEND_SDK=1`） |
| `-i` | 编译 + 安装 |
| `-u` | 运行 C++ 单元测试（需真实 NPU 或 stubs env） |
| `-s` | 运行系统测试 |
| `-j[n]` | 编译线程数（默认 8） |
| `-v` | verbose |
| `-g PREFIX` | 指定 GCC 工具链前缀（交叉编译） |
| `-h` | 帮助 |

> **stubs 必须用 torchair 编译产出的** `build/stubs`（ABI 匹配）。不要用真实 CANN lib64 替换——会 `initialize_graph_engine → aclrtSetDevice → 107001`（无设备）。只有"torchair + 其自带 stubs"这套组合在 CPU 上可用。

### 一键串装（推荐）

以上两步可由 CANN-LM 顶层脚本一键完成（幂等，已装则跳过）：

```bash
cd /mnt/workspace/project/CANN-LM_5883
bash scripts/build.sh deps
```

`deps` 自动按顺序：构建 torchair（`TARGET_PYTHON_PATH=P1`）→ 安装 whl → 构建 torch-delegate → 安装 whl → 构建 cann_lm/engine whl → 安装。完成后 ① export env 就绪。

## 快速开始

### 1. 设置 CANN 环境

根据 CANN 安装路径执行对应的 setup 脚本（由 `scripts/_env_common.sh` 自动 source）：

```bash
# 普通用户默认路径（本机）
source /home/developer/Ascend/cann/set_env.sh

# root 用户默认路径
source /usr/local/Ascend/cann/set_env.sh

# 自定义安装路径（根据实际安装路径替换）
source ${install_path}/cann/set_env.sh
```

设置完成后，`ASCEND_HOME_PATH` 环境变量会自动指向 CANN 安装目录。路径默认值集中在 `scripts/build.conf`，可由 `CANN_SET_ENV` 环境变量覆盖。

### 2. 理解三套运行环境

CANN-LM 有三套互不叠加的 runtime env 基底（公共部分 `scripts/_env_common.sh` 由各脚本 source）：

```bash
source scripts/set_env_export.sh        # ① 导出 epair + 量化 (P1 CPU venv + torchair stubs, 不装 torch_npu)
source scripts/set_env_cann.sh          # ② OM 编译 (真 CANN, 无 python/torch; 带 SOC/GE_FUSION/GE_EXTERNAL flag)
source scripts/set_env_cann.sh --npu    # ③ NPU verify-runtime (真 CANN + DEVICE=npu; dopt GPTQ / NPU 出字 / OM 验证)
```

| env | LD_LIBRARY_PATH | python/torch | 用途 |
|-----|-----------------|--------------|------|
| ① export | stubs 在前，CANN lib64 在后 | P1 venv（torch 2.9.0+cpu，无 torch_npu） | `torch.export` + torchair GE 转换（CPU 上跑，不依赖 NPU 设备） |
| ② cann（OM 编译） | 真 CANN lib64 在前，无 stubs | 无 python/torch（C++ 编译不需） | `build_cpp.sh` + `cann_lm_compile`（epair→OM） |
| ③ cann --npu | 真 CANN lib64 在前，无 stubs | 系统 Python（自带 torch_npu） | dopt GPTQ 量化 / NPU 出字 / OM 运行验证 |

> **关键**：① 与 ③ 互斥——stubs-first LD 与装 torch_npu 不能共存。① 用 torchair 自带 stubs（ABI 匹配）给 GE 引擎提供 C 符号，避免真 NPU `aclrtSetDevice → 107001`（无设备）。**不要**用真实 CANN lib64 替换 stubs。

`build.sh` 各 subcommand 会自动 source 对应 env，无需手动 source。

### 3. 一键端到端构建

```bash
bash scripts/build.sh all
```

`all` 串联 `deps → cpp → wheel → om`，完成从上游依赖到 OM 产物的全流程。

### 4. 校验基底（不构建）

```bash
bash scripts/build.sh check
```

校验 CANN / P1 venv（torch/torchair/torch_delegate/cann_lm）/ stubs / P2 python（torch_npu）/ third_party 是否就绪，输出各项 ✓/✗ 状态。

## 构建命令

### build.sh — 顶层端到端构建入口

| subcommand | env | 说明 |
|-----------|-----|------|
| `check` | ② | 校验基底（CANN / P1 / P2 / stubs / third_party），不构建 |
| `deps` | ② | L1：构建 torchair + torch-delegate + cann_lm/engine wheel 装进 P1（① 就绪，幂等：已装则跳） |
| `cpp [flags]` | ② | delegate `build_cpp.sh`（`libtorch_delegate_backend.so` + `cann_lm_compile` + tests） |
| `wheel` | 无 | delegate `build_wheel.sh`（P1 venv pip 纯打包，不需 CANN） |
| `om [args]` | ② | delegate `run_compile.sh`（epair → OM，自带 source） |
| `package` | ② | delegate `package_run.sh`（打 `.run` 自解压包） |
| `all` | ② | deps → cpp → wheel → om |
| `-h\|--help\|help` | — | 帮助信息 |

env 映射规则：`check`/`deps`/`cpp`/`om`/`package` → ② cann（构建/编译要真 CANN，非 stubs）；`wheel` 不需 env（P1 venv pip 纯打包）。① export 仅供日常 `source` 跑 converter。

示例：

```bash
# 校验基底
bash scripts/build.sh check

# 安装上游依赖（幂等）
bash scripts/build.sh deps

# C++ 构建（默认 release，全树）
bash scripts/build.sh cpp

# C++ 构建（ASAN，开发期强制）
bash scripts/build.sh cpp --asan

# 打 Python wheel
bash scripts/build.sh wheel

# OM 编译（默认 epair）
bash scripts/build.sh om

# 打 .run 安装包
bash scripts/build.sh package

# 一键全流程
bash scripts/build.sh all
```

### build_cpp.sh — C++ 统一构建

`scripts/cann_lm_compile/build_cpp.sh` 是 C++ 统一构建入口（顶层 `CMakeLists.txt`，单 build 树）。

| 参数 | 说明 |
|------|------|
| `--debug` | 构建 Debug 版本（默认 Release） |
| `--asan` | 启用 AddressSanitizer（ASAN flags 含 `-static-libasan`，ctest 裸跑无需 LD_PRELOAD） |
| `--no-tests` | 不构建 C++ 测试 |
| `--no-compile-tool` / `--no-om` | 不构建 `cann_lm_compile` 工具（`--no-om` 为别名） |
| `--no-backend` | 不构建 `libtorch_delegate_backend.so` |
| `--target <name>` | 仅构建指定 target（如 `torch_delegate_backend`） |
| `-h\|--help` | 帮助信息 |

产物（默认全开）：

| 产物 | 来源 | 用途 |
|------|------|------|
| `build/libtorch_delegate_backend.so` | `cann_lm/converter/export/csrc` | backend 库（epair 加载 → GE Graph） |
| `build/cann_lm_compile` | `csrc/tools/compile_main.cc` | epair → OM 编译工具 |
| `build/test_*` | `tests/c++/epair_archive` | C++ 测试（GTest） |

前置：`source scripts/set_env_cann.sh`（设 `ASCEND_HOME_PATH` / `LD_LIBRARY_PATH` / `SOC` / `GE_*`）。csrc 零 torch 引用，`libtorch_delegate_backend.so` 不链 libtorch，仅需 CANN 头 + 三方库（nlohmann_json + GTest，由 torch-delegate `output/third_party` 提供）。

示例：

```bash
# 全量 release
bash scripts/cann_lm_compile/build_cpp.sh

# ASAN（开发期强制）
bash scripts/cann_lm_compile/build_cpp.sh --asan

# 仅 backend lib
bash scripts/cann_lm_compile/build_cpp.sh --no-tests --no-compile-tool

# 仅指定 target
bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend
```

运行 C++ 测试：

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build
```

### build_wheel.sh — Python whl 打包

两个独立 distribution（cann_lm 是 PEP 420 namespace 包）：

| 脚本 | 产物 | 内容 |
|------|------|------|
| `scripts/cann_lm_converter/build_wheel.sh` | `dist/cann_lm-*.whl` | converter + models + dopt_lm |
| `scripts/cann_lm_engine/build_wheel.sh` | `dist/cann_lm_engine-*.whl` | cann_lm.engine（api/backend/core/infra/pipeline） |

用 P1 venv（`TORCH_DELEGATE_HOME/.venv`）的 pip 打包，`--no-deps`（torchair/torch-delegate 不在 PyPI，需源码/whl 预装）。由 `build.sh wheel` 串联调用。

### run_compile.sh — epair→OM 一键编译

`examples/test_converter/export_llm_model/run_compile.sh` 自动 source ② env + 构建 + 运行 + 存日志。

| 参数 | 说明 |
|------|------|
| `<epair_dir>` | 指定 epair 目录（自动找 `ep_json` + `weight_info`） |
| `<model_json> <weights_index> [output_name]` | 显式路径 |
| `--no-build` | 跳过构建（用已有 `build/cann_lm_compile`） |
| `--keep-om` / `--delete-om` | 保留（默认）/删除 OM 产物 |
| `--output <name>` | OM 输出名（默认 `build/om_output/<epair名>`） |
| `--log <path>` | 日志路径（默认 `/tmp/opencode/cann_lm_compile_run.log`） |

产物：`<output>.om` + （权值分离时）`<output 目录>/weight/*.bin`。

> **注意**：`cann_lm_compile` 退出时 GE 静态注册器 teardown 可能触发 glibc "corrupted size vs. prev size" → Aborted（非零退出），但 OM 在此前已正常产出，脚本以 OM 文件存在判定成功。

### package_run.sh — 打 .run 自解压包

`scripts/cann_lm_compile/package_run.sh` 流程：build（`--no-tests`）→ stage（binary + .so + cfg + set_env）→ patchelf rpath → makeself。

产物：`dist/cann_lm_compile-<ver>-<arch>.run`，包内容含 `cann_lm_compile` / `lib/libtorch_delegate_backend.so` / `fusion_autoub_off.cfg` / `set_env.sh`。安装机仅需 CANN（不依赖 torch/venv）。

## 版本配置

所有版本号与路径默认值集中在 `scripts/build.conf`：

```bash
# 版本（信息性，供 build.sh 对齐/校验）
TORCH_DELEGATE_VERSION=0.1.0
TORCH_VERSION=2.9.0
TORCHAIR_VERSION=0.1
CANN_VERSION=9.2.0

# 路径默认值（可被环境变量覆盖）
PROJECT_ROOT=/mnt/workspace/project/CANN-LM_5883
TORCHAIR_ROOT=/mnt/workspace/torchair
TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
CANN_SET_ENV=/home/developer/Ascend/cann/set_env.sh
CONVERTER_PYTHON=${TORCH_DELEGATE_HOME}/.venv/bin/python   # P1 venv
DOPT_PYTHON=/opt/buildtools/Python-3.11.4/bin/python3      # P2 系统 Python
STUBS_DIR=${TORCHAIR_ROOT}/build/stubs
SOC_VERSION=Ascend910_9362
```

环境变量优先：任一变量若已在环境中设置，此处不覆盖（用 `${VAR:-default}`）。切换版本/路径只需修改此文件或在调用前 export 对应环境变量。

## .run 安装包使用

### 安装

```bash
# 指定安装路径
bash dist/cann_lm_compile-0.1.0-<arch>.run --target /opt/cann-lm

# 静默安装
bash dist/cann_lm_compile-0.1.0-<arch>.run --target /opt/cann-lm --quiet
```

### 安装目录结构

```
/opt/cann-lm/
├── cann_lm_compile              ← epair→OM 编译工具
├── lib/
│   └── libtorch_delegate_backend.so   ← backend 库（rpath=$ORIGIN，可重定位）
├── fusion_autoub_off.cfg        ← GE fusion 开关配置（关 AUB fusion）
└── set_env.sh                   ← 运行环境设置脚本
```

### 设置环境并运行

```bash
# 前置：安装机需先 source CANN set_env.sh（提供 CANN lib64 / driver / opp）
source /home/developer/Ascend/cann/set_env.sh

# source 安装包的 set_env.sh（设 LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE + SOC_VERSION）
source /opt/cann-lm/set_env.sh

# 运行编译
/opt/cann-lm/cann_lm_compile <ep_json> <weight_info> [output_name]
```

## 输出目录结构

```
build/                                    ← C++ 构建产物（顶层统一 build 树）
├── libtorch_delegate_backend.so          ← backend 库
├── cann_lm_compile                      ← epair→OM 编译工具
├── test_*                                ← C++ 测试
└── om_output/                           ← OM 编译产物
    └── <epair_name>/
        ├── <name>.om                     ← 编译后 OM 模型
        └── weight/*.bin                  ← 权值分离时外置权值（GE_EXTERNAL_WEIGHT=2）

dist/                                     ← 发布产物
├── cann_lm-*.whl                         ← converter + models + dopt_lm
├── cann_lm_engine-*.whl                  ← cann_lm.engine
└── cann_lm_compile-<ver>-<arch>.run      ← 自解压安装包

${TORCH_DELEGATE_HOME}/output/third_party/  ← 三方库（nlohmann_json + GTest + makeself）
${TORCHAIR_ROOT}/build/stubs/             ← torchair stubs（converter CPU 运行所需）
```

## 常见问题

### Q1: ASCEND_HOME_PATH 未设置

**原因**：未 source CANN 环境脚本。

**解决方案**：

```bash
source /home/developer/Ascend/cann/set_env.sh
```

或由环境变量指定：`export CANN_SET_ENV=/path/to/Ascend/cann/set_env.sh`。

### Q2: stubs 目录缺失或 aclrtSetDevice → 107001

**原因**：converter（① env）需 torchair 编译产出的 stubs，缺失时会用真实 CANN lib64 触发 `aclrtSetDevice → 107001`（无设备）。

**解决方案**：

```bash
bash scripts/build.sh deps   # 构建并安装 torchair（含 stubs）
```

stubs 必须用 **torchair 编译生成的** `${TORCHAIR_ROOT}/build/stubs`（ABI 匹配），不要用真实 CANN lib64 替换。

### Q3: torchair / torch_delegate / cann_lm 未安装

**原因**：P1 venv 缺上游依赖。

**解决方案**：

```bash
bash scripts/build.sh deps   # 幂等：已装则跳过
bash scripts/build.sh check  # 校验状态
```

### Q4: C++ 测试 ASAN 失败或内存错误

**原因**：未开 ASAN 构建，或 cache 残留。

**解决方案**：

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build
```

ASAN flags 含 `-static-libasan`，ctest 可裸跑无需 `LD_PRELOAD`。

### Q5: OM 编译退出非零但 OM 已产出

**原因**：`cann_lm_compile` 退出时 GE 静态注册器 teardown 触发 glibc "corrupted size vs. prev size" → Aborted，但 OM 在此前已正常产出。

**解决方案**：以 OM 文件是否存在判定成功（`run_compile.sh` 已处理此 case，用 `|| true` 不中断）。检查 `build/om_output/<name>/<name>.om` 是否存在。

### Q6: CANN 版本未找到（cann-9.1.0 / cann-9.2.0）

**原因**：脚本不再硬编码 CANN 版本默认值，统一走 `ASCEND_HOME_PATH`。

**解决方案**：

1. 检查 `/mnt/workspace/Ascend_tar` 是否有 CANN run 包
2. 没有则从[master 下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)下载
3. 安装并建立符号链接：`ln -s cann-9.2.0 cann`

## 环境变量速查

| 环境变量 | 说明 | 由谁设置 |
|---------|------|---------|
| `ASCEND_HOME_PATH` | CANN 安装目录 | `_env_common.sh` source CANN `set_env.sh` |
| `LD_LIBRARY_PATH` | 动态库搜索路径（① stubs 在前 / ②③ CANN lib64 在前） | `set_env_export.sh` / `set_env_cann.sh` |
| `PYTHONPATH` | 追加 `PROJECT_ROOT`（converter/NPU 都需） | `_env_common.sh` |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0`，阻止 torch autoload npu backend | `_env_common.sh` |
| `CONVERTER_PYTHON` | P1 venv python（① converter 用） | `set_env_export.sh` |
| `DOPT_PYTHON` | P2 系统 python（③ dopt/出字 subprocess 用） | `_env_common.sh` |
| `TORCH_DELEGATE_HOME` | torch-delegate 源码根（含 .venv + third_party） | `build.conf` |
| `TORCHAIR_ROOT` | torchair 源码根（含 stubs） | `build.conf` |
| `SOC_VERSION` | 编译目标 SOC（② 如 `Ascend910_9362`） | `set_env_cann.sh` |
| `GE_EXTERNAL_WEIGHT` | `2`，OM 权值分离（② 编译） | `set_env_cann.sh` |
| `GE_FUSION_SWITCH_FILE` | fusion 开关配置（② 关 AUB fusion） | `set_env_cann.sh` |
| `DEVICE` | `npu`（③ NPU verify-runtime） | `set_env_cann.sh --npu` |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | `0`，不走 fallback codegen（①） | `set_env_export.sh` |

## 相关文件

- `scripts/build.sh` — 顶层端到端构建入口（subcommand 调度）
- `scripts/build.conf` — 版本号 + 路径默认值集中点
- `scripts/_env_common.sh` — set_env 公共部分（CANN/PYTHONPATH/通用变量）
- `scripts/set_env_export.sh` — ① converter env（CPU venv + torchair stubs）
- `scripts/set_env_cann.sh` — ②③ real-CANN env（OM 编译 / NPU verify-runtime）
- `scripts/cann_lm_compile/build_cpp.sh` — C++ 统一构建（顶层 CMakeLists）
- `examples/test_converter/export_llm_model/run_compile.sh` — epair→OM 一键编译
- `scripts/cann_lm_compile/package_run.sh` — 打 `.run` 自解压包
- `scripts/cann_lm_compile/build.sh` — C++ 前置环境准备（校验 + env）
- `scripts/cann_lm_converter/build_wheel.sh` — cann_lm whl 打包（converter + models + dopt_lm）
- `scripts/cann_lm_engine/build_wheel.sh` — cann_lm_engine whl 打包
- `scripts/pull_models.sh` — 从 ModelScope/HuggingFace 拉取 HF 模型
- `CMakeLists.txt` — 顶层 CMake（add_subdirectory 串 csrc + tests）
- `cmake/modules/` — CMake 模块（DelegateCommonFlags + Find*.cmake + TorchDelegateCannCommon）

## 关联文档

- [ADR-0002: 环境与构建隔离](adr/0002-env-build-isolation.md) — 四套 env + 顶层统一 build 树的设计决策
- [CLI 设计](cann_lm_converter/cli/cli_design.md) — converter 与 torch_npu 的关系、stub 路径分析
- [导出设计](cann_lm_converter/export/export_design.md) — loader/saver/ExportWrapper/C++ 加载
- [量化设计](cann_lm_converter/quant/quant_design.md) — W4A8 方案 / dopt GPTQ / 算子规格
- [权值分离 OM 设计](cann_lm_converter/weight_separated_om_design.md) — externalWeight / OM 权值分离/共享
