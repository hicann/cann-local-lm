# CANN-LM — Agent 协作指南

## 项目结构

```
CANN-LM_5883/
├── cann_lm/      # 核心源码：模型定义 (gemma4/qwen3/qwen3_5) + converter 导出工具链 + C++ 后端
├── ci/           # CI/CD 配置：流水线定义、构建脚本、镜像配置
├── cmake/        # CMake 公共模块：DELEGATE_COMMON_FLAGS 等共享构建变量定义
├── docs/         # 设计文档：converter 设计、模型统一方案 (短期/长期)
├── examples/     # 端到端用例：export_and_load_epair (SimpleNetwork) + export_llm_model (qwen3 LLM) + 生成产物
├── generate/     # 代码生成器：基于模板自动生成模型骨架或配置文件
├── scripts/      # 辅助脚本：环境初始化、批量构建、部署工具
└── tests/        # 单元测试：Python 测试 (tests/python/) + C++ 测试 (tests/c++/)
```

## 架构文档加载
重要： 在探索仓库/项目、回答问题、修改代码、输出设计文档、需求Spec或做代码检视时，根据下表加载对应文档。每个文档只需加载一次，匹配触发词、涉及目录或场景中任意一条即加载。同时，修改了代码后，要更新docs/和docs/下的对应文档。
| 文档 | 触发词 | 涉及目录 |
|------|----------------|----------|
|[`export_design.md`](docs/cann_lm_converter/cann_lm_conveter_export/export_design.md)|导出和加载epair模型|cann_lm/converter/export/|
|[`converter_design.md`](docs/cann_lm_converter/converter_design.md)|大模型导出全流程|cann_lm/converter/|

## 开发约定
### 通用
- 环境安装全部在.venv环境中进行，不要修改系统环境
- Python 版本：`>= 3.10`（使用 `dict[str, Any]` 等内置泛型语法，不引入 `from __future__ import annotations`）
- 依赖：`torch`（必需）+ `torchair`（Ascend 后端必需，用于 IR 转换）;不要依赖torch_npu,如果需要引入torch_npu,设置stub,参考下方运行环境设置
- **运行环境设置（converter，已验证）**：本机为 CPU-only（无 NPU），converter 只用 torchair 的 converter 模块（IR 转换）。方案是 **torchair 源码编译（NO_ASCEND_SDK=1）+ torchair 自带的 stubs**——把 `libtorch_npu`/`libge_runner`/`libhccl` 等 CANN 重库在 lib 层打桩，torchair 的 GE 引擎调用被 stub 接住（`[STUB] Session/GEFinalize`），CPU 上能跑且 decomposition/IR 行为与生产一致。组件版本：
  - `torch` 2.9.0+cpu（venv: `/mnt/workspace/torch-delegate-harmony/.venv`）
  - `torch-delegate` 0.1.0（whl，按 torch 2.9.0 + CANN 9.2.0 编，源码 `/mnt/workspace/torch-delegate-harmony`）
  - `torchair`（源码编译，`NO_ASCEND_SDK=1`，源码 `/mnt/workspace/torchair`，Python 包手动复制到 venv site-packages）
  - `torch_npu` 打桩（lib 层 stub，`/mnt/workspace/torchair/build/stubs`，含 `libtorch_npu.so`/`libge_local_stub.so`/`libge_runner.so`/`libhccl_stub.so` 等，非真实安装）
  - CANN 9.2.0 weekly 20260715（toolkit + 950-ops，提供 lib64 + platform_config + TBE/tvm；装到 `/home/developer/Ascend/cann-9.2.0/`；set_env.sh 实际路径 `/home/developer/Ascend/cann/set_env.sh`）。**符号链接**：`/home/developer/Ascend/cann` -> `cann-9.2.0/`，让 `Ascend950DT_950x.ini` 等 platform_config 可被 GE 找到。**Python 依赖**：TBE/tvm 需要 `decorator`/`scipy`/`numpy`/`attrs`/`sympy`/`psutil`/`networkx`/`tqdm`/`requests`/`pyyaml`/`cffi`（已装到 torchair_lazy_venv）

  运行 converter / 任何依赖 torchair 的脚本前，
**必须先 source 这套环境变量**：
```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0          # 阻止 torch autoload npu backend
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0 # 不走 fallback codegen
```
  - **关键**：stubs 必须用 **torchair 编译生成的** `/mnt/workspace/torchair/build/stubs`（ABI 匹配）。不要用真实 CANN lib64——会 `initialize_graph_engine → aclrtSetDevice → 107001`（无设备）。只有"torchair + 其自带 stubs"这套组合在 CPU 上可用。
  - converter 本身不用 transformers；若跑 `examples/` 下用 transformers 的 run/verify 脚本，先 `import hf_cpu`（禁用 transformers 的 torch_npu probe，shim 在 `examples/export_llm_model/llm_engine_on_cpu/hf_cpu.py`）。
  - 跑 `CANN-LM_5883` 仓库脚本统一加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
- **NPU 环境（torch_npu，已验证）**：本机有 2 张 Ascend910（64GB HBM），torch_npu 已安装在系统 Python 中，可直接用于 NPU 推理/验证。组件版本：
  - Python 3.11.4（`/opt/buildtools/Python-3.11.4/bin/python3`）
  - `torch` 2.7.1+cpu
  - `torch_npu` 2.7.1.post4
  - `torchvision` 0.22.1 / `torchaudio` 2.11.0+cpu
  - CANN npu-smi 25.5.5，芯片 Ascend910 × 2

  进入 NPU 环境：
  ```bash
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  ```
  验证 NPU 可用：
  ```bash
  python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
  ```
  - 注意：该环境是系统 Python（非 venv），与上方 converter 的 venv 环境独立，互不影响。
  - 有一个 `libop_plugin_atb.so` 权限警告（owner 不匹配），不影响功能，可忽略。
- 测试：`pytest`，测试文件放置于 `tests/python/`；单文件运行 `pytest tests/python/test_api.py`，单用例 `pytest tests/python/test_api.py::test_name`
- **端到端测试原则**：所有端到端测试用例及其生成产物（模型文件、权重、OM 等）统一放置于 `examples/` 路径下，按用例名分目录组织；禁止散落到 `cann_lm/` 或其他路径；在进行测试的时候先不要修改主仓代码，先修改测试代码，如果无法通过进行交确认
- **测试时禁改 cann_lm/**：跑测试用例时**不要修改 `cann_lm/` 下任何代码**（含 converter/models/export 等）；要改 `cann_lm/` 前必须先找确认。优先改测试代码（`tests/`、`examples/`），无法通过再确认改主仓。
- **组件规模原则**：单函数实现代码最好控制在 50 行以内；若明显超过，应优先按单一职责拆分辅助函数，但不要为了拆分而制造无意义跳转，优先聚合而非拆分；实现复杂度上升后再重构拆分
- **函数参数原则**：函数/方法参数个数默认不超过 5 个；若超过，优先通过结构体、context 对象或明确的数据聚合来收敛参数，而不是继续横向追加位置参数
- skills读取：可以使用./examples/export_llm_model/llm_engine_on_cpu/skills/路径下的skills


### C++ 模块

- 编码风格遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 函数名、方法名使用大驼峰（UpperCamelCase），如 `Partition()`、`BuildGraph()`
- 成员变量加 `_` 前缀，如 `_archive_path`
- 类名使用大驼峰，如 `EpairPartitioner`、`GraphPartitioner`
- **零警告原则**：C++ 代码必须在 `-Wall -Wextra -Werror` 下无警告编译；CMake 已配置 `-Werror`，PR 中不允许引入新的编译器警告
- **公共 Flag 原则**：所有 C++ target（库、可执行文件）必须通过 `DELEGATE_COMMON_FLAGS` 统一管理编译和链接选项；禁止在单个 target 上直接硬编码 `-Wall`/`-Wextra`/`-Werror` 等公共 flags，新增 target 时须显式调用 `target_compile_options` 和 `target_link_options` 并引用此变量。
- **ASAN 强制原则**：开发期间运行 C++ 测试时**必须**开启 ASAN（`./build_test.sh` 默认启用；手动 cmake 时传 `-DENABLE_ASAN=ON`）；提交前须确认测试在 ASAN 构建下全部通过且无内存错误报告。
- **C++ 函数约束**：C++ 函数/方法同样遵循“参数不超过 5 个、单函数最好不超过 50 行”的通用原则；复杂逻辑优先提取为私有辅助方法，但保持调用链可读，避免过度切碎。


### 环境问题
- **如果出现环境没有cann-9.1.0/cann-9.2.0**: 1.查看本地/mnt/workspace/Ascend_tar是否有cann包 2.没有的话从https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/ 下载 3.按照run包提供的能力进行安装