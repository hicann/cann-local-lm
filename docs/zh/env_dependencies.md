# cann-local-lm环境依赖梳理

简体中文 | [English](../en/env_dependencies.md)

> **构建依赖** = 产 `.whl` / `.run` 分发物所需（编 `cann_lm_compile` / `.so` + 打包）。
> **运行依赖** = 产epair / omc / 出字所需（用分发物跑实际工作）。
>
> 仅"构建分发物"（`bash build.sh --pkg` / `scripts/build.sh wheel|package|cpp`）只装Part 1；要"导出epair / 编omc / 量化 / 出字"才需Part 2。
> 学GE反复明示的「编译vs运行边界」——GE `quick_install.md` 明确「驱动固件是运行态依赖，若仅编译源码可不安」。
>
> 版本来自 `scripts/build.conf` + `build_guide.md` + 2026-10-21本机核验；torch_npu配套来自[官方COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。与 `scripts/build.sh check` 同源（§3.2）。

---

## 总览：任务 → 构建? / 运行?

| 任务 | Part 1构建? | Part 2运行? | 备注 |
|------|:----------:|:----------:|------|
| 编 `cann_lm_compile` + backend `.so`（`scripts/build.sh cpp`） | ✓ | — | C++ 编译，产 `build/` 中间产物 |
| 打 `cann_local_lm-*.whl` / `cann_local_lm_engine-*.whl`（`wheel`） | ✓ | — | pip纯源码打包 |
| 打 `cann_lm_compile-*.run`（`package` / `--pkg`）+ 签名 engine wheel（`cann_local_lm_engine-*+signed*.whl`） | ✓ | — | makeself `.run`（编译工具）+ pip wheel（engine .so 内嵌） |
| ③a engine `.so` 鸿蒙交叉编译（→ 签名 wheel `cann_local_lm_engine-*+signed*.whl`） | ✓ | — | OHOS SDK（含SDK clang）+ 签名 |
| UT构建（`bash build.sh -u` / `tests`） | ✓ | — | GTest + ASAN（可选） |
| ① 导出epair（converter） | — | ✓ | P1 venv + CANN |
| ② kirinx90 omc编译（run `cann_lm_compile`） | — | ✓ | P1 venv（TBE/tvm）+ CANN |
| 量化Step A（dopt GPTQ） | — | ✓ | P2 torch_npu/NPU（或GPU） |
| 量化Step B（converter `--quant-path`） | — | ✓ | P1 venv（同 ①） |
| ③ NPU出字（数值验证） | — | ✓ | CANN + P2 torch_npu + NPU |
| ③b/c Kirin设备出字 | — | ✓（设备侧） | 设备自带musl + libneural_network_core.so |

> ✓ = 必需；— = 不需要。**仅构建分发物**只装Part 1；无NPU/Kirin设备可只构建 + 跑Part 2的 ①②（CPU，不需NPU）。

---

# Part 1: 构建依赖（产 `.whl` / `.run` 分发物）

## 系统级基础工具（所有构建必需）

| 工具 | 版本要求 | 本机实测 | 获取方式 | 用途 |
|------|---------|---------|---------|------|
| **gcc / g++** | >= 8.0且 < 14.0 | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) | `apt install build-essential` | C++ 编译（`-Wall -Wextra -Werror` 零警告） |
| **cmake** | >= 3.18 | 3.20.5 | `apt install cmake` | C++ 统一构建（顶层CMakeLists） |
| **make** | GNU Make | 4.3 | `apt install build-essential`（自带） | cmake构建执行 |
| **Python** | >= 3.10 | 3.12.9 | `apt install python3 python3-venv python3-pip` | whl打包（pip） |
| **patchelf** | 任意（缺失则跳过rpath） | 0.18.0 | `apt install patchelf` | 打 `.run` 包时设 .so rpath（缺失不阻塞，靠 `set_env.sh` 的LD_LIBRARY_PATH） |
| **makeself** | >= 2.5.0 | `/usr/bin/makeself`（已装） | 由三方库构建产出，或 `apt install makeself` | 打 `.run` 自解压包 |
| **git-lfs** | 任意 | 3.4.1 | `apt install git-lfs` | `pull_models.sh` 拉取HF模型权重 |
| **pip** | 任意 | 随python3-pip | `apt install python3-pip` | converter whl打包（`--no-deps` 纯源码） |
| **java (JRE)** | 任意 | openjdk 21.0.10 ✓ | `apt install default-jre` | 仅 `--pkg` [3/4] OHOS签名需要；日常 `cpp`/`wheel` 不需要 |

一键安装（Ubuntu/Debian）：
```bash
apt install -y cmake build-essential python3 python3-venv python3-pip patchelf git-lfs default-jre
```

> **C++ 三方库（nlohmann_json / GTest / makeself）**：非预装依赖，由 `bash scripts/build.sh third_party` 自动下载+编译到 `output/third_party`，属构建动作非环境依赖。

---

## CANN（编译 `cann_lm_compile` C++ 需CANN headers/libs）

CANN是**外部预置**（用户先source其 `set_env.sh` 设 `ASCEND_HOME_PATH`），本仓不构建。

| 组件 | 版本 | 获取 |
|------|------|------|
| CANN toolkit | >= 9.3.0 | [master下载](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)或官网发布版，详见 `build_guide.md`「安装CANN包」 |
| CANN ops | 按SOC | Ascend910 → `Ascend910-ops`；kirinx90 → `ops-nn`/`ops-math`/`ops-transformer`；950DT → `950-ops` |

**版本配套**（`build.conf` 信息性）：
```
CANN_VERSION=9.3.0
SOC_VERSION=KirinX90      # kirinx90; 910 用 Ascend910_9362; 950DT 用 Ascend950DT_9575
```

**安装三步**：
1. toolkit：`./Ascend-cann-toolkit_9.3.0_*.run --install --install-path=<path>`
2. ops（按SOC）：`./Ascend-cann-<soc>-ops_*.run --install --install-path=同上`
3. 符号链接：`ln -s cann-9.3.0/cann cann`（让GE找到platform_config，无版本号路径解耦版本升级）

> CANN同时是运行依赖（Part 2 §2.2 + ② omc编译）：运行 `cann_lm_compile` 编omc + NPU出字都需CANN lib64。多套CANN并存的切换管理（unset防污染等）属运维细节，见 `build_guide.md` 与 `AGENTS.md`。

---

## OHOS交叉编译套件（③a engine `.so`，仅根 `build.sh --pkg` 的 [3/4]阶段）

> ⚠️ **OHOS SDK需人工浏览器下载**：[command-line-tools](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)必须登录华为开发者账号后通过浏览器下载，**无法脚本自动下载**（API返回403需登录）。下载 **Command Line Tools for Linux(ARM)**，得到zip（如 `commandline-tools-linux-arm64-26.0.0.851.zip`）。无OHOS SDK时根 `build.sh --pkg` 在 [3/4] engine阶段中断——日常 `scripts/build.sh cpp/wheel` 不受影响。安装步骤见 [build_guide.md §1.3](build_guide.md)。

`build_engine()` 逐项硬检查，缺任一即中断 `--pkg`。**日常开发（`scripts/build.sh`）不需要**。

> **为什么需要OHOS交叉编译套件**：`libcann_lm_engine.so` 要部署到Kirin手机（HarmonyOS）上跑，鸿蒙用musl libc（与本机glibc不兼容），本机gcc/clang编出的 .so鸿蒙加载不了。command-line-tools提供鸿蒙专用的clang + sysroot（musl头文件/库），是交叉编译鸿蒙 .so的唯一合法工具链。

| # | 项 | 检查逻辑 | 本机实测 | 获取方式 |
|---|----|---------|---------|---------|
| 1 | **编译器** | SDK clang（`${OHOS_SDK_NATIVE}/llvm/bin/clang`） | ✓ clang 15.0.4 | SDK自带（command-line-tools zip内 `llvm/bin/clang`），本机直接运行，无需额外安装 |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` 已设且含 `sysroot/` | ✓ | `export OHOS_SDK_NATIVE=<解压目录>/command-line-tools/sdk/default/openharmony/native` |
| 3 | **java + 签名jar** | `java` 在PATH；`tools/binary-sign-tool.jar` 或 `cann_local_lm/engine/ohos/binary-sign-tool.jar` 存在 | java 21.0.10 ✓；jar ✓ | `apt install default-jre`；jar在SDK zip内 `command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar`，拷到仓内 `tools/`（`mkdir -p tools/ && cp ... tools/`） |
| 4 | **cmake / make** | 同 §1.1 | 3.20.5 / 4.3 ✓ | 同 §1.1 |

> **为什么需要java + 签名jar**：鸿蒙要求所有二进制（`.so` + 可执行）必须**签名**才能在设备上运行（安全机制），`binary-sign-tool.jar` 是签名工具（用 `java` 跑），交叉编译产出的 `libcann_lm_engine.so` + UT可执行经它签名（selfSign开发签名）后才能部署到设备。

> **GTest（UT，可选）**：OHOS部署包含UT/bbit/smoke可执行需GTest编（`build_ohos.sh` 硬检查 `GTEST_TAR`，默认 `/mnt/workspace/gtest-1.14.0.tar.gz`）；仓内已有 `output/third_party/gtest/googletest-1.14.0.tar.gz`（`scripts/build.sh third_party` 产出），`export GTEST_TAR=$REPO/output/third_party/gtest/googletest-1.14.0.tar.gz` 指向即可。仅产 `.so` 不需GTest。GTest分类见 §1.1。

---

## ASAN（UT构建，开发期可选）

| 项 | 要求 | 本机实测 | 用途 |
|----|------|---------|------|
| **libasan** | 与gcc主版本匹配 | gcc 13 → `libasan8`（apt随gcc装） | `build_cpp.sh --asan` / `scripts/build.sh tests` |

ASAN flags含 `-static-libasan`，ctest可裸跑无需 `LD_PRELOAD`。**提交前C++ 改动必须ASAN构建全过**（AGENTS.md「ASAN强制原则」）。

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build
```

---

# Part 2: 运行依赖（产epair / omc / 出字）

## P1 converter venv（① 导出epair + 量化Step B消费）

**CPU-only venv，不装torch_npu（stub路径）**。converter设备无关（不需NPU），运行 `cann_lm_converter.py`（torch.export执行模型 + torchair GE转换）产epair artifact。

> **用户约束（P1/P2环境隔离根因）**：converter设计为可脱离NPU运行（CPU-only机器也能导出），为此对torchair与torch_npu **解耦**——torchair GE转换不依赖torch_npu。P1刻意不装torch_npu（stub路径，设计决策），P2装torch_npu跑NPU；两者虽同为torch 2.9.0但 **P1 venv与P2系统python隔离**（venv vs系统python，依赖不同）。量化（dopt GPTQ）也跑NPU/GPU，同样不能与converter共用P1。因此P1用独立venv（`CONVERTER_PYTHON`），与P2系统python（`DOPT_PYTHON`）隔离。

| 组件 | 版本 | 本机实测 | 来源 |
|------|------|---------|------|
| **torch** | 2.9.0+cpu | 2.9.0+cpu ✓ | torch-delegate `.venv`（`pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu`） |
| **torchair** | 0.1 | OK ✓ | 源码编译（`gitcode.com/Ascend/torchair.git`，公开仓） |
| **torch_delegate** | 0.1.0 | OK ✓ | 已合入cann-local-lm仓 `torch_delegate/` 目录（`gitcode.com/cann/cann-local-lm`） |
| **torch_npu** | **不装** | absent ✓（stub path OK） | 若装需 **2.9.0.post8**（TorchNPU 26.1.1，配CANN 9.3.0，见 §2.2）——但converter走stub路径**刻意不装**（设计决策，见 `cli_design.md`「converter与torch_npu的关系」） |

> 完整包列表见 [`scripts/requirements-p1.txt`](../../scripts/requirements-p1.txt)（torchair/torch-delegate非pip包，需源码安装）。
| numpy / protobuf | 任意 | 随torchair装入 | torchair运行时依赖（`ge._ge_graph` import numpy，`ge_ir_pb2` 查protobuf版本） |
| wheel | 任意 | 随venv | torchair `setup.py bdist_wheel` 需要 |

**venv位置**（`build.conf` 派生）：`${PROJECT_ROOT}/deps/torch-delegate/.venv`（`CONVERTER_PYTHON` = 该venv的python）

**两种安装方式**（详见 `build_guide.md`「安装上游组件」）：
- **方式一源码构建**：clone torch-delegate → `./build.sh`（产 .venv）→ 装torch_delegate whl → clone torchair → `PYTHON_BIN_PATH=venv bash build.sh -c` → 装torchair whl + `packaging`/`cloudpickle`
- **方式二whl直装**：自建venv + `pip install torch==2.9.0+cpu` + 装torch_delegate/torchair预编whl + numpy/protobuf；设 `export CONVERTER_PYTHON=/path/to/venv/bin/python`

**关键校验**（`scripts/build.sh check` 检查项）：
- `CONVERTER_PYTHON` 可执行
- `import torch` / `import torchair` / `import torch_delegate` / `import cann_local_lm` 全通过
- `import torch_npu` **必须失败**（stub path OK）

**TBE/tvm依赖包（供 ② omc编译用）**：`cann_lm_compile`（C++ 二进制）跑 `aclgrphBuildModel` 编算子时回调Python TBE/tvm，需以下包。这些包**不在CANN .run包里**，但大部分是 `te`（pip包）或 `torch` 的自动依赖，随 `pip install te` / `pip install torch` 自动拉入：

| 包 | 来源 | 备注 |
|----|------|------|
| attrs / decorator / numpy / psutil / scipy | `te` 的pip依赖 | `pip install te` 自动拉入 |
| sympy / networkx | `torch` 的pip依赖 | `pip install torch` 自动拉入 |
| **requests / pyyaml / cffi** | **无人声明依赖** | **需手动装**（`pip install requests pyyaml cffi`） |
| tqdm | 无依赖 | 可选，缺失非阻塞 |

> TBE/tvm是**运行时**依赖（② 运行 `cann_lm_compile` 编omc时调），非构建依赖（编 `cann_lm_compile` C++ 二进制不需）。

---

## CANN运行 + P2系统python + torch_npu/NPU（或GPU）— ③ NPU出字 + 量化Step A

> **PyTorch与TorchNPU安装顺序与约束**：
> 1. **先装CANN toolkit，再装TorchNPU**（TorchNPU .so依赖CANN lib64符号，未装CANN会报undefined symbol）。
> 2. **按实际Python版本单独安装**（PyTorch + TorchNPU必须装在同一个Python环境下，版本配套见[《TorchNPU软件安装》](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)）。
> 3. **Python版本必须一致**：PyTorch、TorchNPU、运行Python三者的Python版本须一致（torch C扩展依赖libpython，不一致会报ABI不兼容）。
> 4. **Kirin处理器系列产品仅依赖PyTorch，不需安装TorchNPU**（Kirin走hiai ACL，不经过torch_npu NPU backend）。TorchNPU仅用于Ascend NPU（③ NPU出字 / 量化加速）。

本节是**使用torch_npu/NPU的运行任务**的总归组：③ NPU出字（数值验证）+ 量化Step A（dopt GPTQ）共用P2系统python + torch_npu + NPU。**独立于P1 venv**（P1 venv刻意不装torch_npu走stub路径，P2系统python装真torch_npu跑NPU，两者隔离，见下方绑定表）。仅这些运行任务需要。

> **量化Step A可选GPU**：dopt GPTQ除NPU（torch_npu）外，**也可用GPU（CUDA torch）加速**——此时用GPU env（CUDA torch，无torch_npu、不需NPU），不撞下方三重绑定。**③ NPU出字必须NPU**（跑omc经 `acl.mdl.execute`），不可用GPU。下表默认按NPU路径；GPU路径把 `torch_npu` 行换成 `torch (CUDA)` + GPU设备即可。

> 完整包列表见 [`scripts/requirements-p2.txt`](../../scripts/requirements-p2.txt)（GPU/NPU通用）+ [`scripts/requirements-p2-npu.txt`](../../scripts/requirements-p2-npu.txt)（NPU附加torch_npu）。

**torch_npu三重版本绑定**（[权威配套表COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)）：

| 绑定 | 规则 | 不可错配原因 |
|------|------|-------------|
| **① torch_npu ↔ torch（1:1）** | torch_npu安装包版本号 = torch版本号（`{torch版本}` 或 `{torch版本}.postN`） | C++ 扩展 .so链接libtorch.so，错配 → undefined symbol（同torchaudio版本不匹配，下） |
| **② torch_npu ↔ CANN** | 每个torch_npu版本标称配套固定CANN版本 | torch_npu .so依赖CANN lib64符号 |
| **③ torch ↔ Python（间接）** | torch版本决定支持的python范围 | torch C扩展依赖libpython |

**配套表节选**（影响本机的行）：

| TorchNPU产品版本 | torch_npu安装包 | torch | 标称CANN | 备注 |
|------------------|----------------|-------|-----------|------|
| **26.1.1** | **2.9.0.post8** | **2.9.0** | **CANN 9.3.0** | **← 本机P2实际用此行**；最新；P1 §2.1若装torch_npu用此 |
| 26.1.1 | 2.7.1.post10 | 2.7.1 | CANN 9.1.X | 含 `npu_chunk_gated_delta_rule` |
| 26.1.0 | 2.9.0.post6 | 2.9.0 | CANN 9.1.0 | |
| 26.1.0 | 2.7.1.post8 | 2.7.1 | CANN 9.1.0 | `npu_chunk_gated_delta_rule` ≥ 此版本才有 |
| 26.0.0 | 2.9.0.post2 | 2.9.0 | CANN 9.0.0 | |

> **关键规则**：同一TorchNPU产品版本（如26.1.1）可配多个torch版本，但每个torch版本需对应的安装包版本——`2.9.0.post8` 只能配torch 2.9.0，**不能**配torch 2.7.1。`.postN` 是同一torch版本下的发布序号，**不能换算为产品版本**（2.7.1.post4 ≠ 2.9.0.post4）。
>
> **P1/P2隔离根因**：P1 §2.1用torch **2.9.0+cpu**（converter需torch.export，venv隔离），本节P2用torch **2.9.0** + torch_npu **2.9.0.post8**（TorchNPU 26.1.1）；P1/P2同为torch 2.9.0，但 **P1 venv刻意不装torch_npu**（stub路径，设计决策），P2系统python装torch_npu跑NPU——venv与系统python隔离（依赖不同、环境不同），不能共用。

**CANN运行用**：同Part 1 §1.2（同一安装，NPU出字 / 量化Step A / ② omc编译复用，无需另装）。torch_npu .so依赖CANN lib64（见上方绑定表）。

**P2系统python + torch_npu + NPU设备**：

| 组件 | 版本 | 本机实测 | 来源 |
|------|------|---------|------|
| **python** | 3.12.9 | 3.12.9 ✓ | `/opt/buildtools/python-3.12.9/bin/python3` |
| **torch** | 2.9.0+cpu | 待验证 | 系统预装（当前2.7.1+cpu，需升级） |
| **torch_npu** | 2.9.0.post8 | 待验证 | 当前2.7.1.post4（TorchNPU 26.0.0），需升级到2.9.0.post8（TorchNPU 26.1.1，配CANN 9.3.0，见上方配套表） |
| torchvision | 0.22.1 | — | 系统预装（可选） |
| torchaudio | 2.9.0 | 待验证 | `pip install torchaudio==2.9.0 --force-reinstall --no-deps --user`（匹配torch 2.9.0，装到 `~/.local/` shadow系统旧版） |

**NPU设备**：本机2张Ascend910（64GB HBM），`npu-smi` 25.5.5

**驱动与固件**：NPU运行依赖（③ NPU出字 / 设备出字需NPU设备 + 驱动固件）；仅构建（①②）不需安装。安装见《CANN软件安装指南》"安装NPU驱动和固件"章节。

**`DOPT_PYTHON` 自动探测**（`_env_common.sh` 的 `_resolve_dopt_python`）：
- 优先级：已export的DOPT_PYTHON → 候选路径（`/opt/buildtools/python-3.12.9/bin/python3` / `/opt/buildtools/Python-3.11.4/bin/python3` / `/usr/local/bin/python3` / `/usr/bin/python3`）+ PATH中能 `import torch_npu` 的 → 都找不到则警告（仅 ③ 需，Part 1不阻塞）
- 探测时LD前置候选python的 `<prefix>/lib` + CANN lib64（防加载旧libpython致site-packages计算错）
- 换机器无需改：新机有装torch_npu的python在候选路径或PATH即自动命中

**⚠️ P2三个坑**（源自上方三重绑定 + 实测）：

1. **CANN版本配套**：torch_npu 2.9.0.post8（TorchNPU 26.1.1）标称配套 **CANN 9.3.0**，与本机runtime CANN 9.3.0一致，无版本错位风险。
2. **torch_npu版本**：2.9.0.post8（TorchNPU 26.1.1）含 `npu_chunk_gated_delta_rule`（≥2.7.1.post8 / TorchNPU 26.1.0+ 即有），不缺算子。
3. **torchaudio版本匹配**：torchaudio须与torch同版本（torchaudio 2.9.0 ↔ torch 2.9.0）。若版本不匹配（如系统装了torchaudio 2.11.0 + torch 2.9.0），`import torchaudio` 报 `undefined symbol: torch_library_impl`。修复：

```bash
pip install torchaudio==2.9.0 --force-reinstall --no-deps --user   # 装到 ~/.local/，shadow 系统旧版
```

**另一个P2环境坑**（`build_guide.md` Q8）：
- **`_ctypes` undefined symbol / `No module named 'torch_npu'`（实际装了）**：干净shell source CANN后LD不含python `<prefix>/lib` → 加载系统旧libpython。处理：`LD_LIBRARY_PATH` 前置 `/opt/buildtools/python-3.12.9/lib`（`_env_common.sh` 已自动派生并持久prepend）

---

## Kirin设备侧（③b/c设备出字，设备自带，host不装）

③b/c在Kirin/HarmonyOS设备上跑 `cann_local_lm.generate`（经 `libcann_lm_engine.so` + hiai ACL），**host不需装任何依赖**——设备自带：

| 组件 | 来源 | 用途 |
|------|------|------|
| musl libc | 鸿蒙系统提供 | C运行时（musl，非glibc） |
| `libneural_network_core.so` | 鸿蒙系统库（Hiai/NNRT） | NPU推理（hiai ACL底层） |
| `libc++_shared.so` | 内嵌于签名 engine wheel（`cann_local_lm/lib/`） | C++ 运行时（与engine .so同目录） |
| `libcann_lm_engine.so` | host交叉编译产（Part 1 §1.3）→ 部署到设备 | engine SHARED库（签名版） |

> host侧只需完成Part 1构建（产签名 engine wheel），产物部署到设备后由设备侧运行时承载。

---

# Part 3: 速查

## env → 依赖映射速查表

从"任务"查"需Part 1哪节 + Part 2哪节"：

| 我想做 | source哪个env | Part 1构建依赖 | Part 2运行依赖 | 需NPU/GPU? |
|--------|----------------|---------------|---------------|------------|
| 编cann_lm_compile + 打whl/.run | `source scripts/set_env.sh` | §1.1 + §1.2 CANN | — | 否 |
| ③a engine .so（鸿蒙交叉） | 根 `bash build.sh --pkg` | §1.1 + §1.3 OHOS套件 | — | 否 |
| UT构建 + 运行 | `bash build.sh -u` | §1.1 + §1.2 + §1.4 ASAN | — | 否 |
| ① 导出epair（converter） | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | 否 |
| ② kirinx90 omc编译 | source CANN set_env.sh + source scripts/set_env.sh（SOC_VERSION=KirinX90） | — | §1.2 CANN + §2.1 P1 venv（TBE/tvm） | 否 |
| 量化Step A（dopt GPTQ） | `source scripts/set_env_cann.sh --npu`（或GPU env） | — | §2.2 P2（NPU或GPU） | NPU或GPU |
| 量化Step B（converter消费） | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | 否 |
| ③ NPU出字（数值验证） | `source scripts/set_env_cann.sh --npu` | — | §1.2 CANN + §2.2 P2 | NPU |
| ③b/c Kirin设备出字 | 设备上 `cann_local_lm.generate` | — | §2.3设备侧 | Kirin设备 |
| 发版全量打包 | 根 `bash build.sh --pkg` | Part 1全装（§1.1-1.4） | — | 否 |

> **编译vs运行边界**：仅构建分发物（`.whl`/`.run`）只装Part 1；产epair/omc/出字才需Part 2。无NPU/Kirin设备可只做Part 1构建 + Part 2的 ①②（CPU，不需NPU）。

---

## 校验命令

```bash
# 一键校验基底（必需项缺则报"缺 N 项必需项"；P2/torch_npu 标可选）
source /home/developer/Ascend/cann/set_env.sh
bash scripts/build.sh check

# 期望输出:
#   ✓ CANN /home/developer/Ascend/cann-9.3.0
#   ✓ P1 venv .../deps/torch-delegate/.venv/bin/python
#   ✓ torch (2.9.0+cpu)
#   ✓ torchair / torch_delegate / cann_local_lm
#   ✓ third_party json
#   可选 (③ NPU 出字 / 量化加速, Part 2; 量化可 NPU 或 GPU, 见 §2.2):
#   ✓ P2 .../python-3.12.9 (③ NPU 出字 / 量化 NPU 加速用)
#   ✓ torch_npu (P2, NPU 路径; GPU 路径不经此 check)
#   ✓ 基底齐 (构建 ①② 必需项就绪; ③ NPU 出字 / 量化加速为可选)
```

> 本表与 `scripts/build.sh check` 检查项同源；env/版本变更须同步更新本表与 `build.conf`（doc-脚本同源）。
