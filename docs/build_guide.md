# CANN-LM 构建指南

本文档描述 CANN-LM 的端到端构建流程，包括环境准备、运行环境（converter+compile 统一 env + ③ NPU）、前置条件（CANN / torchair / torch-delegate）、构建命令、安装包使用和常见问题。

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

## 安装上游组件（前置条件）

CANN-LM **不本仓构建** torchair / torch-delegate，它们是**前置条件**（类似 CANN toolkit，用户预先安装）。`bash scripts/build.sh deps` 只构建 cann_lm 本身，前置校验只查「`CONVERTER_PYTHON` 可执行 + torchair/torch_delegate 已装 venv」——兼容下面两种安装方式。

| 组件 | 作用 | git 源（见 `scripts/build.conf`） |
|------|------|----------------------------------|
| torch-delegate | `torch_delegate` whl（epair 序列化 API）+ 通常顺带建 `.venv`（torch 2.9.0+cpu） | `gitcode.com/shengnan666/torch-delegate.git` |
| torchair | `torchair` whl（FX→GE IR 转换） | `gitcode.com/Ascend/torchair.git` |

> **⚠️ torch-delegate 当前未开源**：torch-delegate 代码仓（`gitcode.com/shengnan666/torch-delegate.git`）暂未公开，拉取代码需**额外申请访问权限**（联系仓库 owner）。torchair（`gitcode.com/Ascend/torchair.git`）为公开仓，可直接 clone。若已有预编 whl（方式二），则无需 clone torch-delegate 源码。

> **关键设计**：converter 用 P1 venv（CPU torch，**不装 torch_npu**）+ 真 CANN lib64（**无 stubs**，`lite_export_context` 绕过 `aclrtSetDevice`）。torchair 是含 C++ 扩展的包，**whl 须与 venv 的 torch ABI 匹配**（torch 2.9.0+cpu）。CANN-LM 只复用 venv + 两个 whl；torch-delegate 的 `.so`/`third_party`/`.run` 是副产（CANN-LM 自建三方库 + 自建 backend .so，不复用）。

### 方式一：从源码构建安装（clone + build）

适用：无预编 whl，或要本地改动。需联网 + 编译。

**构建顺序**：先 torch-delegate（产 `.venv`），再 torchair（`TARGET_PYTHON_PATH` 指向 P1 venv python，ABI 匹配）。

#### 安装 torch-delegate

torch-delegate 的 `build.sh` 一站式完成：环境检查 → 创建 `.venv`（torch 2.9.0+cpu）→ 编译三方库 → 编译 C++ delegate 库 → 构建 whl → 打包 `.run`。

```bash
source /home/developer/Ascend/cann/set_env.sh   # 前置：设 ASCEND_HOME_PATH
cd $PROJECT_ROOT          # PROJECT_ROOT

# clone 到默认 deps/ (或自定义路径, 用 TORCH_DELEGATE_HOME 覆盖)
git clone https://gitcode.com/shengnan666/torch-delegate.git deps/torch-delegate
cd deps/torch-delegate
./build.sh                  # 默认 release, 自动创建 .venv
# ./build.sh --pre_check    # 仅校验环境, 不构建
```

构建产物（CANN-LM 用到的）：

| 产物 | 路径 | CANN-LM 用途 |
|------|------|--------------|
| P1 venv（torch 2.9.0+cpu） | `.venv/` | converter 运行环境 + C++ 构建链期 torch cmake 来源 |
| Python whl | `output/package/torch_delegate-*.whl` | 装进 P1（`torch_delegate` 包，epair 序列化） |

将 torch_delegate whl 装进 P1 venv：

```bash
.venv/bin/pip install --force-reinstall --no-deps output/package/torch_delegate-*.whl
```

> torch-delegate 的 `.so`/`third_party`/`.run` 是其自建副产，CANN-LM 不复用（CANN-LM 自建 `output/third_party` + 自建 backend .so）。更多细节见 torch-delegate 仓库 `docs/build.md`。

#### 安装 torchair

torchair 须 `cd` 进其源码根构建（`configure.py` 写相对 CWD 的 `tools/PYTHON_BIN_PATH`），`TARGET_PYTHON_PATH` 指向 P1 venv python。torchair 用 `setup.py bdist_wheel`，须先给 venv 装 `wheel` 包。

```bash
source /home/developer/Ascend/cann/set_env.sh   # 前置：torchair cmake 需 CANN 头
cd $PROJECT_ROOT

git clone https://gitcode.com/Ascend/torchair.git deps/torchair
# torchair 用 setup.py bdist_wheel, P1 venv 默认无 wheel 包; numpy/protobuf 是 torchair 运行时依赖
deps/torch-delegate/.venv/bin/pip install wheel numpy protobuf

# cd 进 torchair 根构建 (TARGET_PYTHON=P1 venv, ABI 匹配; 用绝对路径避免 configure.py 写入相对路径致 make 从 build/ 子目录解析失败)
cd deps/torchair
TARGET_PYTHON_PATH="$(realpath ../torch-delegate/.venv/bin/python)" bash build.sh -c
```

将 torchair whl 装进 P1 venv：

```bash
../torch-delegate/.venv/bin/pip install --force-reinstall --no-deps output/torchair-*.whl
```

torchair `build.sh` 常用参数：`-c`（CI 编译）、`-i`（编译+安装）、`-j[n]`（线程数）、`-h`（帮助）。

### 方式二：用 whl 包安装（无源码编译）

适用：已有兼容 torch 2.9.0+cpu 的 venv + 预编 whl（来源：方式一源码构建产出，或发布包）。不 clone、不编译，直接 pip 装。**torchair whl 须与 venv 的 torch ABI 匹配**（torchair 含 C++ 扩展）。

```bash
# 1. 准备 venv (torch 2.9.0+cpu) —— 已有兼容 venv 可跳过
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 2. 装预编 whl (torch_delegate + torchair; 须与 torch 2.9.0+cpu ABI 匹配)
#    numpy/protobuf 是 torchair 运行时依赖 (ge._ge_graph import numpy, ge_ir_pb2 查 protobuf 版本)
pip install numpy protobuf
pip install torch_delegate-0.1.0-py3-none-any.whl
pip install torchair-0.1-py3-none-any.whl

# 3. 让 CANN-LM 用该 venv (设 CONVERTER_PYTHON 指向其 python)
export CONVERTER_PYTHON=/path/to/venv/bin/python
# (可选) 若 venv 命名符合 ${TORCH_DELEGATE_HOME}/.venv 结构, 设 TORCH_DELEGATE_HOME 即可, 无需设 CONVERTER_PYTHON
```

> 方式二不需 `TORCHAIR_ROOT` / `TORCH_DELEGATE_HOME` 源码根目录，只靠 `CONVERTER_PYTHON` 指向装好上游的 venv。`build.sh deps` 前置校验只查「`CONVERTER_PYTHON` 可执行 + torchair/torch_delegate 已装」，源码根 / whl 两种来源皆过。

### 校验前置就绪

```bash
source /home/developer/Ascend/cann/set_env.sh
bash scripts/build.sh check    # 校验 CANN/P1 venv/torchair/torch_delegate/cann_lm/third_party 全就绪
```

## 快速开始

### 1. 设置 CANN 环境

根据 CANN 安装路径执行对应的 setup 脚本（**须手动 source**，`_env_common.sh` 不自动 source，只校验 `ASCEND_HOME_PATH` 已设）：

```bash
# 普通用户默认路径（本机）
source /home/developer/Ascend/cann/set_env.sh

# root 用户默认路径
source /usr/local/Ascend/cann/set_env.sh

# 自定义安装路径（根据实际安装路径替换）
source ${install_path}/cann/set_env.sh
```

设置完成后，`ASCEND_HOME_PATH` 环境变量会自动指向 CANN 安装目录。路径默认值集中在 `scripts/build.conf`，可由 `CANN_SET_ENV` 环境变量覆盖。

### 2. 理解运行环境

CANN-LM 有两套 runtime env（公共部分 `scripts/_env_common.sh` 校验 `ASCEND_HOME_PATH` 外部预置、设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` / `DOPT_PYTHON`）：

```bash
source scripts/set_env.sh              # converter (①) + compile (②) 统一 env (真 CANN, 无 stubs/无 0715)
source scripts/set_env_cann.sh --npu   # ③ NPU verify-runtime (真 CANN + DEVICE=npu; CANN 轻量化工具 GPTQ / NPU 出字 / OM 验证)
# set_env_export.sh 为 back-compat 薄 wrapper (→ set_env.sh)
```

| env | LD_LIBRARY_PATH | python/torch | 用途 |
|-----|-----------------|--------------|------|
| `set_env.sh`（converter+compile 统一） | 真 CANN lib64（无 stubs，幂等） | P1 venv（torch 2.9.0+cpu，无 torch_npu） | ① `torch.export`+torchair GE 转换（走真 libge_runner，本机有 NPU 设备）；② `build_cpp.sh`+`cann_lm_compile`（epair→OM，带 SOC/GE_FUSION/GE_EXTERNAL flag） |
| `set_env_cann.sh --npu`（③ NPU） | 真 CANN lib64（无 stubs） | **P2 系统 Python（独立于 P1 venv，自带 torch_npu）** | **轻量化工具 GPTQ 量化（需 NPU 加速时）** / NPU 出字 / OM 运行验证 |

> converter+compile **复用一套 env**（真 CANN，去 stubs、去 0715 pin）：converter 用 P1 venv（CPU torch，不装 torch_npu）+ 真 CANN lib64，torchair GE 转换走真 `libge_runner`（本机有 910 设备）。`set_env.sh` 还反查 cann_lm 装位置自定位注入 PYTHONPATH（不硬编码 repo 根）。
>
> **torch_npu（③ P2 系统 python）是可选的**：仅当**量化（轻量化工具 GPTQ）需要 NPU 加速**时才需要。converter 导出（①）+ OM 编译（②）**不依赖 torch_npu**——它们用 P1 venv（CPU torch，无 torch_npu）+ 真 CANN lib64。无 NPU 设备或仅做导出/编译时无需装 torch_npu。
>
> **⚠️ torch_npu 与 converter 的 P1 venv 冲突，必须装在独立的 python 环境**：torch_npu 不能装进 converter 的 CPU-torch venv（torch_npu 需配对带 NPU 支持的 torch，且其 import 会触发 NPU 设备初始化 `aclrtSetDevice`，与 converter 的 CPU torch + CPU 运行路径互斥）。故 CANN-LM 用**两个独立 python 环境**：P1 venv（`CONVERTER_PYTHON`，CPU torch 2.9.0+cpu，无 torch_npu，跑 converter）+ P2 系统 python（`DOPT_PYTHON`，装 torch_npu，跑轻量化工具 GPTQ 量化 NPU 加速）。轻量化工具以 subprocess 形式经 `DOPT_PYTHON` 调起，与 converter 隔离。

`build.sh` 各 subcommand 自动 source `set_env.sh`（converter+compile 用），无需手动 source。

### 3. 一键端到端构建

```bash
bash scripts/build.sh all
```

`all` 串联 `deps → third_party → cpp → wheel`（torchair+torch-delegate 须前置就绪），完成从 cann_lm 安装到 C++ 库 + 发布 whl 的全流程。OM 编译不在 `all`（属测试/示例，见下文）。

### 4. 校验基底（不构建）

```bash
bash scripts/build.sh check
```

校验基底：**必需项**（CANN / P1 venv / torch+torchair+torch_delegate+cann_lm / third_party）缺则报"缺 N 项必需项"；**可选项**（P2 系统 python + torch_npu）仅量化 NPU 加速时需要，converter/compile 不依赖，✗ 不计入失败。

## 构建命令

### build.sh — 顶层端到端构建入口

| subcommand | env | 说明 |
|-----------|-----|------|
| `check` | set_env | 校验基底（必需：CANN / P1 venv / torchair / torch_delegate / cann_lm / third_party；**可选**：P2 python + torch_npu，仅量化 NPU 加速需），不构建 |
| `deps` | set_env | 装 cann_lm/engine wheels 进 P1 venv（torchair+torch-delegate 为前置条件，见上节；幂等：已装则跳） |
| `third_party` | 无 | 自带 C++ 三方库（nlohmann_json + GTest + makeself）→ `output/third_party` |
| `cpp [flags]` | set_env | delegate `build_cpp.sh`（`libtorch_delegate_backend.so` + `cann_lm_compile`，**不含测试**） |
| `tests [flags]` | set_env | delegate `build_test.sh`（ASAN，独立 `build_tests/` 树 + 构建 + ctest；兑现 ASAN 强制） |
| `wheel` | 无 | delegate `build_wheel.sh`（P1 venv pip 纯打包，不需 CANN） |
| `package` | set_env | delegate `package_run.sh`（打 `.run` 自解压包） |
| `all` | set_env | deps → third_party → cpp → wheel |
| `-h\|--help\|help` | — | 帮助信息 |

env 映射：`check`/`deps`/`cpp`/`tests`/`package` 均 source `set_env.sh`（converter+compile 统一 env，需先 source CANN set_env.sh 设 `ASCEND_HOME_PATH`）；`wheel`/`third_party` 不需 env。③ NPU 运行用 `source scripts/set_env_cann.sh --npu`。

> **OM 编译（epair→OM）属测试/示例，不在 build.sh 构建流程**：用 `cpp` 产出的 `cann_lm_compile` 工具编译具体模型。直接跑 `examples/test_converter/export_llm_model/run_compile.sh`（自带 source set_env.sh + 构建+运行+存日志）。

示例：

```bash
# 校验基底
bash scripts/build.sh check

# 装 cann_lm wheels 进 P1 venv（幂等；torchair+torch-delegate 须前置就绪）
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

# 路径默认值（可被环境变量覆盖; PROJECT_ROOT 自定位, 上游仓默认 deps/）
# PROJECT_ROOT: 自定位 (build.conf 从 scripts 父目录推, 不硬编码; source env 脚本后 $PROJECT_ROOT 可用)
DEPS_DIR=${PROJECT_ROOT}/deps
TORCHAIR_ROOT=${DEPS_DIR}/torchair                  # 前置条件, 见「安装上游组件」
TORCH_DELEGATE_HOME=${DEPS_DIR}/torch-delegate      # 前置条件, 含 .venv (P1)
CANN_SET_ENV=/home/developer/Ascend/cann/set_env.sh # hint (CANN 外部预置)
CONVERTER_PYTHON=${TORCH_DELEGATE_HOME}/.venv/bin/python   # P1 venv
DOPT_PYTHON=                                   # P2 系统 Python (③ NPU 用); 由 _env_common.sh 自动探测, 不硬编码绝对路径
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

${TORCH_DELEGATE_HOME}/.venv/                 ← P1 venv（torch 2.9.0+cpu + torchair + torch_delegate + cann_lm），前置条件产
output/third_party/                         ← CANN-LM 自建三方库（nlohmann_json + GTest + makeself）
```

## 常见问题

### Q1: ASCEND_HOME_PATH 未设置

**原因**：未 source CANN 环境脚本。

**解决方案**：

```bash
source /home/developer/Ascend/cann/set_env.sh
```

或由环境变量指定：`export CANN_SET_ENV=/path/to/Ascend/cann/set_env.sh`。

### Q2: ASCEND_HOME_PATH 未设 / aclrtSetDevice 报错

**原因**：未先 source CANN set_env.sh（`set_env.sh` 不自动 source CANN，要求外部预置 `ASCEND_HOME_PATH`）。

**解决方案**：

```bash
source /home/developer/Ascend/cann/set_env.sh   # 设 ASCEND_HOME_PATH
source scripts/set_env.sh                       # 再 source 统一 env
```

> converter+compile 现复用一套 env（真 CANN，无 stubs）。本机有 NPU 设备时 torchair GE 转换走真 `libge_runner`；若纯 CPU 无设备，需另行装 torch_npu 或恢复 stubs 方案（见 git 历史）。

### Q3: torchair / torch_delegate 未安装

**原因**：前置条件未就绪（torchair+torch-delegate 不由 build.sh 构建，须手动安装）。

**解决方案**：见上文「安装上游组件」节，按顺序装 torch-delegate（产 `.venv`）→ torchair，然后：

```bash
bash scripts/build.sh deps    # 装 cann_lm wheels 进 P1 venv (幂等)
bash scripts/build.sh check   # 校验全基底
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

### Q7: 换机器/换 python 版本后 DOPT_PYTHON 要改吗

**不用**。`DOPT_PYTHON` 已由 `scripts/_env_common.sh` 的 `_resolve_dopt_python` 自动探测，不再硬编码绝对路径。

**探测优先级**：

1. 已 `export DOPT_PYTHON` 且能 `import torch_npu` → 直接用（用户覆盖最高优先）
2. 候选路径 + `PATH` 里的 `python3`，取第一个能 `import torch_npu` 的：`/opt/buildtools/python-3.12.9/bin/python3`、`/opt/buildtools/Python-3.11.4/bin/python3`、`/usr/local/bin/python3`、`/usr/bin/python3`、`$(command -v python3)`
3. 都找不到 → 打印警告（仅 ③ NPU 阶段需，①② 不阻塞）

**换机器**：新机有装 `torch_npu` 的 python 在候选路径或 `PATH` 即自动命中；非标路径 `export DOPT_PYTHON=/path/to/python` 覆盖。converter P1 venv 的 python 无 `torch_npu`，会被自动跳过不会误选。

> 探测时 LD 前置候选 python 的 `<prefix>/lib` + CANN `lib64`（原因见 Q8），探测成功后 `<prefix>/lib` 持久 prepend 到 `LD_LIBRARY_PATH`，使后续 `$DOPT_PYTHON` 调用也能正常加载 `libpython`。

### Q8: `$DOPT_PYTHON` 报 `No module named 'torch_npu'`（实际装了）/ `_ctypes` undefined symbol

**现象**：`$DOPT_PYTHON -c "import torch_npu"` 报 `ModuleNotFoundError: No module named 'torch_npu'`，但 `torch_npu` 确实装在该 python 的 `site-packages`；或报 `undefined symbol: _PyErr_SetLocaleString` 等 `_ctypes` 错误。

**原因**：buildtools 的 `python3` 动态依赖 `libpython3.12.so.1.0`（在 `<prefix>/lib`，如 `/opt/buildtools/python-3.12.9/lib`）。当 `LD_LIBRARY_PATH` 不含 `<prefix>/lib`（典型：干净 shell `unset LD_LIBRARY_PATH` 后 `source CANN set_env.sh`，LD 只含 CANN `lib64`），`python3` 加载到系统旧版 `libpython3.12.so.1.0` → prefix/site-packages 计算错 → `sys.path` 丢失 buildtools `site-packages` → `import torch_npu` 失败。`ldd <python>` 可见 `libpython3.12.so.1.0` 依赖。

**解决方案**：`LD_LIBRARY_PATH` 前置该 python 的 `<prefix>/lib`：

```bash
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH:-}
```

`_env_common.sh` 的 `_resolve_dopt_python` 已从候选 python 路径自动派生 `<prefix>/lib`（`$(dirname $(dirname <python>))/lib`）前置 + 探测成功后持久 prepend，故 `source scripts/set_env_cann.sh --npu` 流程无需手动处理。仅当脱离 env 脚本手动跑 `$DOPT_PYTHON` 时需手 prepend（路径按实际 python prefix 替换）。

### Q9: 轻量化工具量化报 `torchaudio: undefined symbol: torch_library_impl` / `torchaudio.__spec__ is None`

**现象**：`dopt_quantization_st` 跑真实量化（或 `import torchaudio`）报：

```
OSError: torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
```

**原因**：系统 python 3.12.9 的 `torchaudio 2.11.0+cpu` 是为**更新的 torch** 编译的（其 C 扩展 `_torchaudio.abi3.so` 引用 `torch_library_impl` 符号），但运行时 torch `2.7.1+cpu` 的 `libtorch.so` **没有**该符号 → ABI 不兼容，`import torchaudio` 即崩。轻量化工具加载模型时 `transformers` 的 `is_torchaudio_available()` → `importlib.util.find_spec("torchaudio")` 触发 torchaudio 包查找 → 撞坏的 .so。

**与 torch_npu 无关**：torch 2.7.1+cpu + torch_npu 2.7.1.post4 自身 .so 配对匹配（`import torch/torch_npu` 正常），崩的只是第三方 torchaudio build。轻量化工具量化文本模型（Qwen/gemma4）不需音频。

**解决方案**：用 `sitecustomize.py` 设 `sys.modules['torchaudio']=None`（Python 显式屏蔽机制）：

```python
# /tmp/opencode/sitecustomize.py (或任意 PYTHONPATH 前置目录)
import sys
sys.modules["torchaudio"] = None   # import→ImportError; find_spec→None(不raise); is_torchaudio_available→False
```

```bash
export PYTHONPATH=/tmp/opencode:${PYTHONPATH:-}   # 仅 ③ NPU 量化场景临时用, 不永久改系统
bash examples/test_converter/dopt_quantization_st/run.sh run \
    --models qwen3_5 --qwen3-5-model <model_dir> --profile demo --device npu
```

**屏蔽效果**：

- `import torchaudio` → `ImportError`（None 表示显式不可用）
- `importlib.util.find_spec("torchaudio")` → 返回 `None`（不 raise，不触发真包查找）
- `transformers.is_torchaudio_available()` → `False` → transformers 跳过 torchaudio

> ⚠️ 不要用 `types.ModuleType("torchaudio")` 假模块——它的 `__spec__` 是 None，`find_spec` 会抛 `ValueError: torchaudio.__spec__ is None`。必须用 `sys.modules["torchaudio"]=None`（CPython `find_spec` 对 None 值返回 None 而非 raise）。

**根治**：重装与 torch 2.7.1 ABI 匹配的 torchaudio（但 AGENTS.md 约定不改系统环境，故用屏蔽 shim）。

## 参数说明

### build.conf 配置项（`scripts/build.conf`，单一来源，环境变量优先覆盖）

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROJECT_ROOT` | 自定位（scripts 父目录） | CANN-LM 仓根；build.conf 自定位注入，任何 source 它的脚本都拿到 |
| `TORCH_DELEGATE_VERSION` | `0.1.0` | torch-delegate 版本（信息性，对齐校验） |
| `TORCH_VERSION` | `2.9.0` | torch 版本（信息性） |
| `TORCHAIR_VERSION` | `0.1` | torchair 版本 |
| `CANN_VERSION` | `9.2.0` | CANN 版本（信息性） |
| `TORCHAIR_GIT_URL` | `gitcode.com/Ascend/torchair.git` | torchair git 源（前置安装参考，见「安装上游组件」） |
| `TORCHAIR_GIT_TAG` | 空 | torchair git tag（空=默认分支） |
| `TORCH_DELEGATE_GIT_URL` | `gitcode.com/shengnan666/torch-delegate.git` | torch-delegate git 源 |
| `TORCH_DELEGATE_GIT_TAG` | 空 | torch-delegate git tag |
| `NLOHMANN_JSON_VERSION` / `GTEST_VERSION` / `MAKESELF_VERSION` | `3.11.3` / `1.14.0` / `2.5.0-patch1` | 自建三方库版本（build_third_party.sh 下载编译） |
| `DEPS_DIR` | `${PROJECT_ROOT}/deps` | 上游仓 clone + 构建根（gitignored） |
| `TORCHAIR_ROOT` | `${DEPS_DIR}/torchair` | torchair 根（前置条件，源码装） |
| `TORCH_DELEGATE_HOME` | `${DEPS_DIR}/torch-delegate` | torch-delegate 根（前置条件，含 `.venv`） |
| `CONVERTER_PYTHON` | `${TORCH_DELEGATE_HOME}/.venv/bin/python` | P1 venv python（converter 用；尊重外部预置，whl 模式可指向自建 venv） |
| `SOC_VERSION` | `Ascend910_9362` | OM 编译目标 SOC。**NPU 机用 910_9362；CPU-only + 950DT CANN 机用 `Ascend950DT_9575` 交叉编译**（env 覆盖） |
| `CANN_SET_ENV` | `/home/developer/Ascend/cann/set_env.sh` | CANN set_env.sh 路径（hint；用户须先 source 它设 `ASCEND_HOME_PATH`） |
| `DOPT_PYTHON` | _(自动探测)_ | P2 系统 python（③ 量化 NPU 加速用，**独立于 P1 venv**，torch_npu 与 converter CPU-torch 冲突须隔离；`_env_common.sh` 探测候选路径+PATH 中能 `import torch_npu` 的 python，换机器无需改） |

### 运行时环境变量（set_env.sh / _env_common.sh 导出 + 外部预置）

| 环境变量 | 说明 | 由谁设置 |
|---------|------|---------|
| `ASCEND_HOME_PATH` | CANN 安装目录（**外部预置**，不自动 source） | 用户 source CANN `set_env.sh` |
| `ASCEND_OPP_PATH` | CANN opp 目录（GE 找 ops/kernel 用） | CANN `set_env.sh` |
| `LD_LIBRARY_PATH` | 动态库搜索路径（真 CANN lib64 + devlib [libascend_hal.so]，无 stubs，幂等 prepend） | `set_env.sh` / `set_env_cann.sh` |
| `PYTHONPATH` | cann_lm 装位置（反查 venv site-packages 自定位，不硬编码 repo 根） | `set_env.sh` |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0`，阻止 torch autoload npu backend | `_env_common.sh` |
| `CONVERTER_PYTHON` | P1 venv python（converter 用；尊重外部预置） | `set_env.sh` |
| `DOPT_PYTHON` | P2 系统 python（独立于 P1 venv；装 torch_npu，跑轻量化工具 GPTQ 量化 NPU 加速；**自动探测**装 torch_npu 的 python，换机器无需改） | `_env_common.sh`（自动探测） |
| `TORCH_DELEGATE_HOME` | torch-delegate 根（前置条件，含 `.venv`） | `build.conf` / `set_env.sh` |
| `TORCHAIR_ROOT` | torchair 根（前置条件） | `build.conf` |
| `SOC_VERSION` | OM 编译目标 SOC（`Ascend910_9362` 默认；950DT 机用 `Ascend950DT_9575`，env 覆盖） | `set_env.sh` |
| `GE_EXTERNAL_WEIGHT` | `2`，OM 权值分离（编译期，权值落 `<output>/weight/*.bin`） | `set_env.sh` |
| `GE_FUSION_SWITCH_FILE` | fusion 开关配置（关 AUB + AddRmsNorm/InplaceAddRmsNorm fusion） | `set_env.sh` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `1`，GE 详细日志到 stdout（定位 not registered 等） | `set_env.sh` |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | `0`，不走 fallback codegen | `set_env.sh` |
| `DEVICE` | `npu`（③ NPU verify-runtime） | `set_env_cann.sh --npu` |

### build.sh subcommand flags

| subcommand | flags | 说明 |
|-----------|-------|------|
| `cpp` | `--debug` / `--release`(默认) / `--asan` / `--coverage` / `--tests` / `--no-tests`(默认) / `--no-compile-tool`(`--no-om` 别名) / `--no-backend` / `--target <name>` / `-h` | 构建 `libtorch_delegate_backend.so` + `cann_lm_compile`（默认不含测试，测试用 `tests`）；`--coverage` 需独立 build 目录，与 `--asan` 互斥 |
| `tests` | `--no-asan` / `--skip-run` / `-h` | ASAN 构建（独立 `build_tests/` 树）+ ctest；默认 ASAN ON + 跑测试；`--skip-run` 仅构建不跑 |
| `third_party` | `--third_party_path=PATH` / `--build_type=TYPE` / `--jobs=N` / `--help` | 自建 nlohmann_json + GTest + makeself → `output/third_party` |
| `deps` | （无） | 装 cann_lm/engine wheels 进 P1 venv（幂等；torchair+torch-delegate 须前置就绪） |
| `wheel` | （无） | cann_lm + cann_lm_engine whl → `dist/`（P1 venv pip 纯打包，不需 CANN） |
| `package` | （透传 `package_run.sh`） | 打 `cann_lm_compile-*.run` 自解压包 |
| `check` | （无） | 校验基底（必需项 + 可选项 P2/torch_npu），不构建 |
| `all` | （透传） | deps → third_party → cpp → wheel |
| e2e `run_e2e_pipeline.sh` | `--hf-path <dir>` / `--model-type qwen3_5\|gemma4` / `--seq-len N` / `--prompt <text>` / `--output-dir <dir>` / `--skip-export` / `--skip-om` / `--skip-npu` / `--check-env` | 端到端 ① 导出 epair + ② OM 编译 + ③ NPU 出字。gemma4 需 `--seq-len 128`；无 NPU 用 `--skip-npu` |

## 用例：HF → epair → OM 端到端（gemma4）

以 `gemma-4-E2B-it` 为例，串通 ① HF 模型导出 epair + ② epair 编译 OM。前置：CANN + torchair/torch-delegate（P1 venv）已就绪（见「安装上游组件」），`bash scripts/build.sh cpp` 已产出 `build/cann_lm_compile`。

### 0. 准备 env

```bash
source /home/developer/Ascend/cann/set_env.sh        # 设 ASCEND_HOME_PATH + 完整 CANN LD (含 plugins/opp)
source scripts/set_env.sh                            # converter+compile 统一 env (真 CANN, 无 stubs)
# ② OM 编译目标 SOC: NPU 机用默认 Ascend910_9362; CPU-only + 950DT CANN 机交叉编译:
export SOC_VERSION=Ascend950DT_9575                  # 仅 950DT CANN 机需; 910 NPU 机省略
```

### ① HF → epair（cli_export 导出 prefill+decode）

converter 把 HF 模型经 `torch.export` + torchair GE 转换，产出 prefill/decode 两段 epair（含 ascend_ir 注解的 EP JSON + 权重索引）。

```bash
CONVERTER="$PWD/cann_lm/converter/cann_lm_converter.py"
OUT="$PWD/examples/test_converter/export_llm_model/output/e2e_gemma4"
mkdir -p "$OUT"

# --mode prefill-decode: 同时导出 prefill + decode 两段; gemma4 需 --seq-len 128 (run_gemma4_gen.py PREFILL_SEQ=128)
"$CONVERTER_PYTHON" "$CONVERTER" \
    --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
    --output-dir "$OUT" \
    --mode prefill-decode --target-soc Ascend910B
```

产物（`$OUT/`）：

| 文件 | 说明 |
|------|------|
| `prefill_ep_json_with_ascend_ir.json` | prefill 段 EP JSON（含 ascend_ir 注解，GE opdef） |
| `decode_ep_json_with_ascend_ir.json` | decode 段 EP JSON |
| `weight_info.json` | 权重索引（cann_lm_compile 加载权重用） |
| `model_config.json` / `tokenizer.json` | 模型配置 + tokenizer（③ NPU 出字用） |

> converter 后台跑会在 GE finalize 处挂起（已知），产物此前已落盘；`run_e2e_pipeline.sh` 用后台+轮询+kill 绕过。

### ② epair → OM（cann_lm_compile 编译 prefill+decode）

`cann_lm_compile` 加载 epair（ep_json + weight_info）→ 构建 GE Graph → `aclgrphBuildModel` 编译 OM（权值分离时落 `<output>/weight/*.bin`）。各段独立子目录以隔离权值。

```bash
BIN="$PWD/build/cann_lm_compile"
cd "$OUT"
for st in prefill decode; do
  mkdir -p "$st"
  "$BIN" "${st}_ep_json_with_ascend_ir.json" weight_info.json "${st}/${st}"
  # → ${st}/${st}.om (+ ${st}/weight/*.bin 权值分离)
done
```

产物：

| 文件 | 说明 |
|------|------|
| `prefill/prefill.om` | prefill 段 OM（编译后模型，部署到 NPU 跑） |
| `decode/decode.om` | decode 段 OM |
| `prefill|decode/weight/*.bin` | 权值分离时外置权值（`GE_EXTERNAL_WEIGHT=2`） |

> cann_lm_compile 退出时 GE REG_OP 静态注册器 teardown 会触发 glibc abort（非零退出），OM 在此前已正常产出——以 `.om` 文件存在判定成功（`run_compile.sh` 已用 `|| true` 处理）。

### ③（可选）NPU 出字验证

需 NPU 设备 + torch_npu（P2 系统 python，独立 env）：

```bash
source scripts/set_env_cann.sh --npu
"$DOPT_PYTHON" examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_gen.py "Hello, my name is"
```

无 NPU/未装 torch_npu 则跳过 ③（①② 仍完成）。

### 一键跑（推荐）

`examples/test_converter/export_llm_model/run_e2e_pipeline.sh` 串通 ①②③，自动处理后台挂起/GE teardown/SOC env：

```bash
source /home/developer/Ascend/cann/set_env.sh
export SOC_VERSION=Ascend950DT_9575   # 950DT 机; 910 NPU 机省略
bash examples/test_converter/export_llm_model/run_e2e_pipeline.sh \
    --model-type gemma4 --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
    --seq-len 128 --skip-npu          # 无 NPU 跳过 ③
# 复用已有 epair 只跑 ②: 加 --skip-export
# 只验 env 不跑: --check-env
```

qwen3.5 同理（`--model-type qwen3_5 --hf-path <Qwen3.5-0.8B> --seq-len 8`）；注意 0828 CANN 上 qwen3.5 的 ChunkGatedDeltaRule op_info 回归（gemma4 不受影响）。

## 相关文件

- `scripts/build.sh` — 顶层端到端构建入口（subcommand 调度）
- `scripts/build.conf` — 版本号 + git 源 + 自定位派生路径（PROJECT_ROOT 自定位）
- `scripts/_env_common.sh` — set_env 公共部分（校验 ASCEND_HOME_PATH / 通用变量）
- `scripts/set_env.sh` — converter+compile 统一 env（真 CANN，无 stubs/无 0715）
- `scripts/set_env_cann.sh` — ③ NPU verify-runtime env
- `scripts/set_env_export.sh` — back-compat 薄 wrapper（→ `set_env.sh`）
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
- [量化设计](cann_lm_converter/quant/quant_design.md) — W4A8 方案 / 轻量化工具 GPTQ / 算子规格
