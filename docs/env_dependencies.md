# cann-device-lm 环境依赖梳理

> **构建依赖** = 产 `.whl` / `.run` 分发物所需（编 `cann_lm_compile` / `.so` + 打包）。
> **运行依赖** = 产 epair / omc / 出字所需（用分发物跑实际工作）。
>
> 仅"构建分发物"（`bash build.sh --pkg` / `scripts/build.sh wheel|package|cpp`）只装 Part 1；要"导出 epair / 编 omc / 量化 / 出字"才需 Part 2。
> 学 GE 反复明示的「编译 vs 运行边界」——GE `quick_install.md` 明确「驱动固件是运行态依赖，若仅编译源码可不安」。
>
> 版本来自 `scripts/build.conf` + `build_guide.md` + 2026-10-21 本机核验；torch_npu 配套来自[官方 COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。与 `scripts/build.sh check` 同源（§3.2）。

---

## 总览：任务 → 构建? / 运行?

| 任务 | Part 1 构建? | Part 2 运行? | 备注 |
|------|:----------:|:----------:|------|
| 编 `cann_lm_compile` + backend `.so`（`scripts/build.sh cpp`） | ✓ | — | C++ 编译，产 `build/` 中间产物 |
| 打 `cann_lm-*.whl` / `cann_lm_engine-*.whl`（`wheel`） | ✓ | — | pip 纯源码打包 |
| 打 `cann_lm_compile-*.run` / `engine_ohos-*.run`（`package` / `--pkg`） | ✓ | — | makeself 自解压 |
| ③a engine `.so` 鸿蒙交叉编译（→ `engine_ohos.run`） | ✓ | — | OHOS SDK（含 SDK clang）+ 签名 |
| UT 构建（`bash build.sh -u` / `tests`） | ✓ | — | GTest + ASAN（可选） |
| ① 导出 epair（converter） | — | ✓ | P1 venv + CANN |
| ② kirinx90 omc 编译（run `cann_lm_compile`） | — | ✓ | P1 venv（TBE/tvm）+ CANN |
| 量化 Step A（dopt GPTQ） | — | ✓ | P2 torch_npu/NPU（或 GPU） |
| 量化 Step B（converter `--quant-path`） | — | ✓ | P1 venv（同 ①） |
| ③ NPU 出字（数值验证） | — | ✓ | CANN + P2 torch_npu + NPU |
| ③b/c Kirin 设备出字 | — | ✓（设备侧） | 设备自带 musl + libneural_network_core.so |

> ✓ = 必需；— = 不需要。**仅构建分发物**只装 Part 1；无 NPU/Kirin 设备可只构建 + 跑 Part 2 的 ①②（CPU，不需 NPU）。

---

# Part 1: 构建依赖（产 `.whl` / `.run` 分发物）

## 1.1 系统级基础工具（所有构建必需）

| 工具 | 版本要求 | 本机实测 | 获取方式 | 用途 |
|------|---------|---------|---------|------|
| **gcc / g++** | >= 8.0 且 < 14.0 | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) | `apt install build-essential` | C++ 编译（`-Wall -Wextra -Werror` 零警告） |
| **cmake** | >= 3.18 | 3.20.5 | `apt install cmake` | C++ 统一构建（顶层 CMakeLists） |
| **make** | GNU Make | 4.3 | `apt install build-essential`（自带） | cmake 构建执行 |
| **Python** | >= 3.10 | 3.12.9 | `apt install python3 python3-venv python3-pip` | whl 打包（pip） |
| **patchelf** | 任意（缺失则跳过 rpath） | 0.18.0 | `apt install patchelf` | 打 `.run` 包时设 .so rpath（缺失不阻塞，靠 `set_env.sh` 的 LD_LIBRARY_PATH） |
| **makeself** | >= 2.5.0 | `/usr/bin/makeself`（已装） | 由三方库构建产出，或 `apt install makeself` | 打 `.run` 自解压包 |
| **git-lfs** | 任意 | 3.4.1 | `apt install git-lfs` | `pull_models.sh` 拉取 HF 模型权重 |
| **pip** | 任意 | 随 python3-pip | `apt install python3-pip` | converter whl 打包（`--no-deps` 纯源码） |
| **java (JRE)** | 任意 | openjdk 21.0.10 ✓ | `apt install default-jre` | 仅 `--pkg` [3/4] OHOS 签名需要；日常 `cpp`/`wheel` 不需要 |

一键安装（Ubuntu/Debian）：
```bash
apt install -y cmake build-essential python3 python3-venv python3-pip patchelf git-lfs default-jre
```

> **C++ 三方库（nlohmann_json / GTest / makeself）**：非预装依赖，由 `bash scripts/build.sh third_party` 自动下载+编译到 `output/third_party`，属构建动作非环境依赖。

---

## 1.2 CANN（编译 `cann_lm_compile` C++ 需 CANN headers/libs）

CANN 是**外部预置**（用户先 source 其 `set_env.sh` 设 `ASCEND_HOME_PATH`），本仓不构建。

| 组件 | 版本 | 获取 |
|------|------|------|
| CANN toolkit | >= 9.3.0 | [master 下载](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) 或官网发布版，详见 `build_guide.md`「安装 CANN 包」 |
| CANN ops | 按 SOC | Ascend910 → `Ascend910-ops`；kirinx90 → `ops-nn`/`ops-math`/`ops-transformer`；950DT → `950-ops` |

**版本配套**（`build.conf` 信息性）：
```
CANN_VERSION=9.3.0
SOC_VERSION=Ascend910_9362      # 默认; kirinx90 用 KirinX90; 950DT 用 Ascend950DT_9575
```

**安装三步**：
1. toolkit：`./Ascend-cann-toolkit_9.3.0_*.run --install --install-path=<path>`
2. ops（按 SOC）：`./Ascend-cann-<soc>-ops_*.run --install --install-path=同上`
3. 符号链接：`ln -s cann-9.3.0/cann cann`（让 GE 找到 platform_config，无版本号路径解耦版本升级）

> CANN 同时是运行依赖（Part 2 §2.2 + ② omc 编译）：运行 `cann_lm_compile` 编 omc + NPU 出字都需 CANN lib64。多套 CANN 并存的切换管理（unset 防污染等）属运维细节，见 `build_guide.md` 与 `AGENTS.md`。

---

## 1.3 OHOS 交叉编译套件（③a engine `.so`，仅根 `build.sh --pkg` 的 [3/4] 阶段）

> ⚠️ **OHOS SDK 需人工浏览器下载**：[command-line-tools](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos) 必须登录华为开发者账号后通过浏览器下载，**无法脚本自动下载**（API 返回 403 需登录）。下载 **Command Line Tools for Linux(ARM)**，得到 zip（如 `commandline-tools-linux-arm64-26.0.0.851.zip`）。无 OHOS SDK 时根 `build.sh --pkg` 在 [3/4] engine 阶段中断——日常 `scripts/build.sh cpp/wheel` 不受影响。安装步骤见 [build_guide.md §1.3](build_guide.md)。

`build_engine()` 逐项硬检查，缺任一即中断 `--pkg`。**日常开发（`scripts/build.sh`）不需要**。

> **为什么需要 OHOS 交叉编译套件**：`libcann_lm_engine.so` 要部署到 Kirin 手机（HarmonyOS）上跑，鸿蒙用 musl libc（与本机 glibc 不兼容），本机 gcc/clang 编出的 .so 鸿蒙加载不了。command-line-tools 提供鸿蒙专用的 clang + sysroot（musl 头文件/库），是交叉编译鸿蒙 .so 的唯一合法工具链。

| # | 项 | 检查逻辑 | 本机实测 | 获取方式 |
|---|----|---------|---------|---------|
| 1 | **编译器** | SDK clang（`${OHOS_SDK_NATIVE}/llvm/bin/clang`） | ✓ clang 15.0.4 | SDK 自带（command-line-tools zip 内 `llvm/bin/clang`），本机直接运行，无需额外安装 |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` 已设且含 `sysroot/` | ✓ | `export OHOS_SDK_NATIVE=<解压目录>/command-line-tools/sdk/default/openharmony/native` |
| 3 | **java + 签名 jar** | `java` 在 PATH；`tools/binary-sign-tool.jar` 或 `cann_lm/engine/ohos/binary-sign-tool.jar` 存在 | java 21.0.10 ✓；jar ✓ | `apt install default-jre`；jar 在 SDK zip 内 `command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar`，拷到仓内 `tools/`（`mkdir -p tools/ && cp ... tools/`） |
| 4 | **cmake / make** | 同 §1.1 | 3.20.5 / 4.3 ✓ | 同 §1.1 |

> **为什么需要 java + 签名 jar**：鸿蒙要求所有二进制（`.so` + 可执行）必须**签名**才能在设备上运行（安全机制），`binary-sign-tool.jar` 是签名工具（用 `java` 跑），交叉编译产出的 `libcann_lm_engine.so` + UT 可执行经它签名（selfSign 开发签名）后才能部署到设备。

> **GTest（UT，可选）**：OHOS 部署包含 UT/bbit/smoke 可执行需 GTest 编（`build_ohos.sh` 硬检查 `GTEST_TAR`，默认 `/mnt/workspace/gtest-1.14.0.tar.gz`）；仓内已有 `output/third_party/gtest/googletest-1.14.0.tar.gz`（`scripts/build.sh third_party` 产出），`export GTEST_TAR=$REPO/output/third_party/gtest/googletest-1.14.0.tar.gz` 指向即可。仅产 `.so` 不需 GTest。GTest 分类见 §1.1。

---

## 1.4 ASAN（UT 构建，开发期可选）

| 项 | 要求 | 本机实测 | 用途 |
|----|------|---------|------|
| **libasan** | 与 gcc 主版本匹配 | gcc 13 → `libasan8`（apt 随 gcc 装） | `build_cpp.sh --asan` / `scripts/build.sh tests` |

ASAN flags 含 `-static-libasan`，ctest 可裸跑无需 `LD_PRELOAD`。**提交前 C++ 改动必须 ASAN 构建全过**（AGENTS.md「ASAN 强制原则」）。

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build
```

---

# Part 2: 运行依赖（产 epair / omc / 出字）

## 2.1 P1 converter venv（① 导出 epair + 量化 Step B 消费）

**CPU-only venv，不装 torch_npu（stub 路径）**。converter 设备无关（不需 NPU），运行 `cann_lm_converter.py`（torch.export 执行模型 + torchair GE 转换）产 epair artifact。

> **用户约束（P1/P2 环境隔离根因）**：converter 设计为可脱离 NPU 运行（CPU-only 机器也能导出），为此对 torchair 与 torch_npu **解耦**——torchair GE 转换不依赖 torch_npu。解耦导致 P1（torch 2.9.0，无 torch_npu）与 P2（torch_npu）torch 版本不兼容，**不能共用 python**。量化（dopt GPTQ）也跑 NPU/GPU，同样不能与 converter 共用 P1。因此 P1 用独立 venv（`CONVERTER_PYTHON`），与 P2 系统 python（`DOPT_PYTHON`）隔离。

| 组件 | 版本 | 本机实测 | 来源 |
|------|------|---------|------|
| **torch** | 2.9.0+cpu | 2.9.0+cpu ✓ | torch-delegate `.venv`（`pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu`） |
| **torchair** | 0.1 | OK ✓ | 源码编译（`gitcode.com/Ascend/torchair.git`，公开仓） |
| **torch_delegate** | 0.1.0 | OK ✓ | 源码编译（`gitcode.com/shengnan666/torch-delegate.git`，⚠️ 当前未开源，需申请权限） |
| **torch_npu** | **不装** | absent ✓（stub path OK） | 若装需 **2.9.0.post8**（TorchNPU 26.1.1，配 CANN 9.1.X，见 §2.2）——但 converter 走 stub 路径**刻意不装**（设计决策，见 `cli_design.md`「converter 与 torch_npu 的关系」） |

> 完整包列表见 [`scripts/requirements-p1.txt`](../scripts/requirements-p1.txt)（torchair/torch-delegate 非 pip 包，需源码安装）。
| numpy / protobuf | 任意 | 随 torchair 装入 | torchair 运行时依赖（`ge._ge_graph` import numpy，`ge_ir_pb2` 查 protobuf 版本） |
| wheel | 任意 | 随 venv | torchair `setup.py bdist_wheel` 需要 |

**venv 位置**（`build.conf` 派生）：`${PROJECT_ROOT}/deps/torch-delegate/.venv`（`CONVERTER_PYTHON` = 该 venv 的 python）

**两种安装方式**（详见 `build_guide.md`「安装上游组件」）：
- **方式一源码构建**：clone torch-delegate → `./build.sh`（产 .venv）→ 装 torch_delegate whl → clone torchair → `TARGET_PYTHON_PATH=venv bash build.sh -c` → 装 torchair whl
- **方式二 whl 直装**：自建 venv + `pip install torch==2.9.0+cpu` + 装 torch_delegate/torchair 预编 whl + numpy/protobuf；设 `export CONVERTER_PYTHON=/path/to/venv/bin/python`

**关键校验**（`scripts/build.sh check` 检查项）：
- `CONVERTER_PYTHON` 可执行
- `import torch` / `import torchair` / `import torch_delegate` / `import cann_lm` 全通过
- `import torch_npu` **必须失败**（stub path OK）

**TBE/tvm 依赖包（供 ② omc 编译用）**：`cann_lm_compile`（C++ 二进制）跑 `aclgrphBuildModel` 编算子时回调 Python TBE/tvm，需以下包。这些包**不在 CANN .run 包里**，但大部分是 `te`（pip 包）或 `torch` 的自动依赖，随 `pip install te` / `pip install torch` 自动拉入：

| 包 | 来源 | 备注 |
|----|------|------|
| attrs / decorator / numpy / psutil / scipy | `te` 的 pip 依赖 | `pip install te` 自动拉入 |
| sympy / networkx | `torch` 的 pip 依赖 | `pip install torch` 自动拉入 |
| **requests / pyyaml / cffi** | **无人声明依赖** | **需手动装**（`pip install requests pyyaml cffi`） |
| tqdm | 无依赖 | 可选，缺失非阻塞 |

> TBE/tvm 是**运行时**依赖（② 运行 `cann_lm_compile` 编 omc 时调），非构建依赖（编 `cann_lm_compile` C++ 二进制不需）。

---

## 2.2 CANN 运行 + P2 系统 python + torch_npu/NPU（或 GPU）— ③ NPU 出字 + 量化 Step A

本节是**使用 torch_npu/NPU 的运行任务**的总归组：③ NPU 出字（数值验证）+ 量化 Step A（dopt GPTQ）共用 P2 系统 python + torch_npu + NPU。**独立于 P1 venv**（torch_npu 与 CPU-torch 互斥，不可同环境，见下方绑定表）。仅这些运行任务需要。

> **量化 Step A 可选 GPU**：dopt GPTQ 除 NPU（torch_npu）外，**也可用 GPU（CUDA torch）加速**——此时用 GPU env（CUDA torch，无 torch_npu、不需 NPU），不撞下方三重绑定。**③ NPU 出字必须 NPU**（跑 omc 经 `acl.mdl.execute`），不可用 GPU。下表默认按 NPU 路径；GPU 路径把 `torch_npu` 行换成 `torch (CUDA)` + GPU 设备即可。

> 完整包列表见 [`scripts/requirements-p2.txt`](../scripts/requirements-p2.txt)（GPU/NPU 通用）+ [`scripts/requirements-p2-npu.txt`](../scripts/requirements-p2-npu.txt)（NPU 附加 torch_npu）。

**torch_npu 三重版本绑定**（[权威配套表 COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)）：

| 绑定 | 规则 | 不可错配原因 |
|------|------|-------------|
| **① torch_npu ↔ torch（1:1）** | torch_npu 安装包版本号 = torch 版本号（`{torch版本}` 或 `{torch版本}.postN`） | C++ 扩展 .so 链接 libtorch.so，错配 → undefined symbol（同 torchaudio 版本不匹配，下） |
| **② torch_npu ↔ CANN** | 每个 torch_npu 版本标称配套固定 CANN 版本 | torch_npu .so 依赖 CANN lib64 符号 |
| **③ torch ↔ Python（间接）** | torch 版本决定支持的 python 范围 | torch C 扩展依赖 libpython |

**配套表节选**（影响本机的行）：

| TorchNPU 产品版本 | torch_npu 安装包 | torch | 标称 CANN | 备注 |
|------------------|----------------|-------|-----------|------|
| 26.1.1 | 2.9.0.post8 | 2.9.0 | CANN 9.1.X | 最新；P1 §2.1 若装 torch_npu 用此 |
| 26.1.1 | 2.7.1.post10 | 2.7.1 | CANN 9.1.X | 含 `npu_chunk_gated_delta_rule` |
| 26.1.0 | 2.9.0.post6 | 2.9.0 | CANN 9.1.0 | |
| 26.1.0 | 2.7.1.post8 | 2.7.1 | CANN 9.1.0 | `npu_chunk_gated_delta_rule` ≥ 此版本才有 |
| 26.0.0 | 2.9.0.post2 | 2.9.0 | CANN 9.0.0 | |
| **26.0.0** | **2.7.1.post4** | **2.7.1** | **CANN 9.0.0** | **← 本机 P2 实际用此行** |

> **关键规则**：同一 TorchNPU 产品版本（如 26.1.1）可配多个 torch 版本，但每个 torch 版本需对应的安装包版本——`2.7.1.post4` 只能配 torch 2.7.1，**不能**配 torch 2.9.0。`.postN` 是同一 torch 版本下的发布序号，**不能换算为产品版本**（2.7.1.post4 ≠ 2.9.0.post4）。
>
> **P1/P2 分两环境根因**：P1 §2.1 用 torch **2.9.0**（converter 需 torch.export 新特性），本节 P2 用 torch **2.7.1** + torch_npu **2.7.1.post4**（TorchNPU 26.0.0）；torch 版本不同 → torch_npu 版本也不同 → 不可同一 python 环境（torch ABI 不兼容）。

**CANN 运行用**：同 Part 1 §1.2（同一安装，NPU 出字 / 量化 Step A / ② omc 编译复用，无需另装）。torch_npu .so 依赖 CANN lib64（见上方绑定表）。

**P2 系统 python + torch_npu + NPU 设备**：

| 组件 | 版本 | 本机实测 | 来源 |
|------|------|---------|------|
| **python** | 3.12.9 | 3.12.9 ✓ | `/opt/buildtools/python-3.12.9/bin/python3` |
| **torch** | 2.7.1+cpu | 2.7.1+cpu ✓ | 系统预装 |
| **torch_npu** | 2.7.1.post4 | OK ✓（npu True, 2 devices） | 系统预装；属 TorchNPU **26.0.0**，标称配 CANN **9.0.0**（见上方配套表） |
| torchvision | 0.22.1 | — | 系统预装（可选） |
| torchaudio | 2.7.1 | OK ✓ | `pip install torchaudio==2.7.1 --force-reinstall --no-deps --user`（匹配 torch 2.7.1，装到 `~/.local/` shadow 系统旧版） |

**NPU 设备**：本机 2 张 Ascend910（64GB HBM），`npu-smi` 25.5.5

**驱动与固件**：NPU 运行依赖（③ NPU 出字 / 设备出字需 NPU 设备 + 驱动固件）；仅构建（①②）不需安装。安装见《CANN 软件安装指南》"安装 NPU 驱动和固件"章节。

**`DOPT_PYTHON` 自动探测**（`_env_common.sh` 的 `_resolve_dopt_python`）：
- 优先级：已 export 的 DOPT_PYTHON → 候选路径（`/opt/buildtools/python-3.12.9/bin/python3` / `/opt/buildtools/Python-3.11.4/bin/python3` / `/usr/local/bin/python3` / `/usr/bin/python3`）+ PATH 中能 `import torch_npu` 的 → 都找不到则警告（仅 ③ 需，Part 1 不阻塞）
- 探测时 LD 前置候选 python 的 `<prefix>/lib` + CANN lib64（防加载旧 libpython 致 site-packages 计算错）
- 换机器无需改：新机有装 torch_npu 的 python 在候选路径或 PATH 即自动命中

**⚠️ P2 三个坑**（源自上方三重绑定 + 实测）：

1. **CANN 版本错位**：torch_npu 2.7.1.post4 标称配套 **CANN 9.0.0**，本机 runtime 用 **9.3.0**。本机能跑（npu True 2，向后兼容），但严格按配套表应配 9.0.0；若遇 torch_npu 报符号缺失/segfault，优先排查此错位。
2. **torch_npu 较老缺算子**：2.7.1.post4（TorchNPU 26.0.0）缺 `npu_chunk_gated_delta_rule`（需 ≥2.7.1.**post8** / TorchNPU 26.1.0+）。影响 qwen3.5 量化 NPU 侧 dopt GPTQ（converter 侧靠 `cann_lm::` 自定义算子绕开，不撞）。**建议**：升级到 torch_npu 2.7.1.post10（TorchNPU 26.1.1，含该算子，配 CANN 9.1.X），但需同步升级 CANN 到 9.1.X 配套。
3. **torchaudio 版本匹配**：torchaudio 须与 torch 同版本（torchaudio 2.7.1 ↔ torch 2.7.1）。若版本不匹配（如系统装了 torchaudio 2.11.0 + torch 2.7.1），`import torchaudio` 报 `undefined symbol: torch_library_impl`。修复：

```bash
pip install torchaudio==2.7.1 --force-reinstall --no-deps --user   # 装到 ~/.local/，shadow 系统旧版
```

**另一个 P2 环境坑**（`build_guide.md` Q8）：
- **`_ctypes` undefined symbol / `No module named 'torch_npu'`（实际装了）**：干净 shell source CANN 后 LD 不含 python `<prefix>/lib` → 加载系统旧 libpython。处理：`LD_LIBRARY_PATH` 前置 `/opt/buildtools/python-3.12.9/lib`（`_env_common.sh` 已自动派生并持久 prepend）

---

## 2.3 Kirin 设备侧（③b/c 设备出字，设备自带，host 不装）

③b/c 在 Kirin/HarmonyOS 设备上跑 `cann_lm.generate`（经 `libcann_lm_engine.so` + hiai ACL），**host 不需装任何依赖**——设备自带：

| 组件 | 来源 | 用途 |
|------|------|------|
| musl libc | 鸿蒙系统提供 | C 运行时（musl，非 glibc） |
| `libneural_network_core.so` | 鸿蒙系统库（Hiai/NNRT） | NPU 推理（hiai ACL 底层） |
| `libc++_shared.so` | 随 `deploy_ohos/` 包提供 | C++ 运行时（与 engine .so 同目录） |
| `libcann_lm_engine.so` | host 交叉编译产（Part 1 §1.3）→ 部署到设备 | engine SHARED 库（签名版） |

> host 侧只需完成 Part 1 构建（产 `engine_ohos.run`），产物部署到设备后由设备侧运行时承载。

---

# Part 3: 速查

## 3.1 env → 依赖映射速查表

从"任务"查"需 Part 1 哪节 + Part 2 哪节"：

| 我想做 | source 哪个 env | Part 1 构建依赖 | Part 2 运行依赖 | 需 NPU/GPU? |
|--------|----------------|---------------|---------------|------------|
| 编 cann_lm_compile + 打 whl/.run | `source scripts/set_env.sh` | §1.1 + §1.2 CANN | — | 否 |
| ③a engine .so（鸿蒙交叉） | 根 `bash build.sh --pkg` | §1.1 + §1.3 OHOS 套件 | — | 否 |
| UT 构建 + 运行 | `bash build.sh -u` | §1.1 + §1.2 + §1.4 ASAN | — | 否 |
| ① 导出 epair（converter） | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | 否 |
| ② kirinx90 omc 编译 | source CANN set_env.sh + source scripts/set_env.sh（SOC_VERSION=KirinX90） | — | §1.2 CANN + §2.1 P1 venv（TBE/tvm） | 否 |
| 量化 Step A（dopt GPTQ） | `source scripts/set_env_cann.sh --npu`（或 GPU env） | — | §2.2 P2（NPU 或 GPU） | NPU 或 GPU |
| 量化 Step B（converter 消费） | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | 否 |
| ③ NPU 出字（数值验证） | `source scripts/set_env_cann.sh --npu` | — | §1.2 CANN + §2.2 P2 | NPU |
| ③b/c Kirin 设备出字 | 设备上 `cann_lm.generate` | — | §2.3 设备侧 | Kirin 设备 |
| 发版全量打包 | 根 `bash build.sh --pkg` | Part 1 全装（§1.1-1.4） | — | 否 |

> **编译 vs 运行边界**：仅构建分发物（`.whl`/`.run`）只装 Part 1；产 epair/omc/出字才需 Part 2。无 NPU/Kirin 设备可只做 Part 1 构建 + Part 2 的 ①②（CPU，不需 NPU）。

---

## 3.2 校验命令

```bash
# 一键校验基底（必需项缺则报"缺 N 项必需项"；P2/torch_npu 标可选）
source /home/developer/Ascend/cann/set_env.sh
bash scripts/build.sh check

# 期望输出:
#   ✓ CANN /home/developer/Ascend/cann-9.3.0
#   ✓ P1 venv .../deps/torch-delegate/.venv/bin/python
#   ✓ torch (2.9.0+cpu)
#   ✓ torchair / torch_delegate / cann_lm
#   ✓ third_party json
#   可选 (③ NPU 出字 / 量化加速, Part 2; 量化可 NPU 或 GPU, 见 §2.2):
#   ✓ P2 .../python-3.12.9 (③ NPU 出字 / 量化 NPU 加速用)
#   ✓ torch_npu (P2, NPU 路径; GPU 路径不经此 check)
#   ✓ 基底齐 (构建 ①② 必需项就绪; ③ NPU 出字 / 量化加速为可选)
```

> 本表与 `scripts/build.sh check` 检查项同源；env/版本变更须同步更新本表与 `build.conf`（doc-脚本同源）。
