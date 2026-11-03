# ADR-0001: converter 主进程 load_model 写死 CPU，轻量化工具 subprocess 支持设备选择

简体中文 | [English](../../en/adr/0001-load-model-cpu-quant-subprocess-device.md)

> 状态: Accepted
> 日期: 2026-08-25
> 相关设计 .md: [cli_design.md](../cann_lm_converter/cli/cli_design.md) "运行环境支持" / "converter 与 torch_npu 的关系"

## 背景 (Context)

converter 全流程三阶段，各阶段计算域需求不同：

| 阶段 | 做什么 | 计算域需求 |
|------|--------|-----------|
| 加载 `load_model` + swap | 加载 cann_local_lm 模型、物化 int4、setattr 替换 nn.Linear | 只需 tensor 操作（CPU 可行） |
| 量化 GPTQ（`--quant-method gptq`） | CANN 轻量化工具跑 Hessian 矩阵累积 `X^T X` + Cholesky 求逆 + 逐列更新 | 需真 forward + backward，必须计算设备（NPU/GPU） |
| 导出 epair（`torch.export` + torchair GE 转换） | trace 模型 → FX 图 → GE IR | 只需 fake-meta（trace 用）+ torchair stubs（C 符号） |

关键约束：

1. **cann_local_lm 融合算子只在 CPU 实现**：`cann_local_lm/models/fused_ops/` 的算子（`npu_quant_matmul` / `npu_dynamic_quant` / `fused_infer_attention_score` 等）由 `_library.py` 提供 schema + fake-meta + CPU PoC impl，**不支持真 NPU 执行**（只支持 torch.export trace）。故 `load_model` 加载的 cann_local_lm 模型无法在 NPU 上 forward。
2. **converter venv 是 CPU-only**：`torch 2.9.0+cpu` + `torchair`（源码编译 `NO_ASCEND_SDK=1`）+ lib stubs（`libtorch_npu.so` 等打桩，给 torchair GE 引擎提供 C 符号，避免真 NPU `aclrtSetDevice→107001`）。**不装 torch_npu python 包**。
3. **GPTQ 需真算子 eager**：dopt_lm 用 `transformers.AutoModelForCausalLM` + 真 torch_npu/torch.cuda 算子 eager 跑 GPTQ（Hessian 需 batch forward 累积，Cholesky 需算力）。dopt_lm `SUPPORTED_DEVICE_TYPES = {"cpu","cuda","npu"}` **含 CPU**（CPU 走 `dispatch_dtype=float32`），但 7B+ 模型 Hessian 数 GB，CPU 极慢（仅小模型/调试适用）。
4. **环境冲突无解**：若 converter 主进程装真 torch_npu，会撞 `_LazyTorchair` 代理覆盖源码 torchair（`torch_npu/dynamo/__init__.py` 无条件装 `sys.modules['torchair']=_LazyTorchair()`，劫持到 bundled wrapper，wrapper 的 `fx2ge_converter.parse_input` 有 AssertionError 在 `input_ids` placeholder 上失败）+ torch_npu 版本依赖（`npu_chunk_gated_delta_rule` schema 仅 ≥2.7.1.post8）+ `_get_deterministic_level` API 缺失。详见 cli_design.md "converter 与 torch_npu 的关系"。

## 决策 (Decision)

**主进程（load_model + swap + export）写死 CPU venv；轻量化工具子程序（`--quant-method gptq`）用系统 Python + 真计算设备（NPU/GPU），通过 subprocess 隔离。**

- `load_model(hf_path, dtype, device="cpu")` 写死 `device="cpu"`，不暴露 `--device` CLI 参数。
- `--quant-device` 仅控制轻量化工具 subprocess 设备（`npu`/`cuda`/`cpu`），通过 `os.environ["DEVICE"]` 传递给 subprocess；`None` 则 `_detect_device_type` spawn 检测或继承父进程 `DEVICE`，无 NPU/GPU 时回落 `cpu`（dopt_lm 支持 CPU float32，7B+ 极慢）。
- subprocess 用独立 Python（`DOPT_PYTHON`，由 `_env_common.sh` 自动探测装 torch_npu 的系统 python，见 build_guide.md Q7）+ 真 torch_npu/torch.cuda + NPU/GPU env（`LD_LIBRARY_PATH` 去 stubs）。

> **状态变更（2026-08-31）**：`--quant-device` 已从 converter CLI 删除。subprocess 设备改由 `auto_quant._detect_device_type` 自动检测（读 `os.environ["DEVICE"]`，外部 env 如 `set_env_npu.sh` 可设；未设则 spawn 检测，无加速器回落 `cpu`），`_run_dopt_subprocess` 的 `env.setdefault("DEVICE", device)` 机制保留。本条所述 CLI `os.environ["DEVICE"]` 触发路径不再存在，但 `_detect_device_type` + subprocess env 机制保留。

理由：导出只需 fake-meta trace（`_library.py` 自提供，不需 torch_npu）；GPTQ 需真算子 eager（必须计算设备）。两者计算域不同，且环境冲突（torch_npu 装了会破坏 GE 转换）无解，故用 subprocess 隔离两套环境。

## 后果 (Consequences)

- **正面**：
  - 规避 `_LazyTorchair` 代理覆盖坑 + torch_npu 版本依赖（converter 正确性不依赖任何 torch_npu 版本）
  - converter venv 与 NPU env 独立，互不污染
  - 纯 CPU 机可用（`--quant-path` 消费离线轻量化工具产物 + export stubs）
- **负面/代价**：
  - subprocess 有磁盘桥接 IO（临时盘 `tempfile.mkdtemp`，swap 成功后 `shutil.rmtree` 清理，失败保留供调试）
  - ~~`auto_quant.py` 与 `dopt_wrapper.py` 各有一份 `_build_layer_strategy`（重复，计划合并到单文件消除，尚未落地）~~ **已落地（2026-08-31）**：`auto_quant._build_layer_strategy` 删除，`dopt_wrapper._build_layer_strategy` 改名 `build_layer_strategy`，单一实现
  - 场景 2 内存峰值 2×（subprocess 内轻量化工具 2 份 HF 模型 + 主进程 cann_local_lm，分进程；非真一步化）
- **引入的约束**：
  - `load_model` 永远 CPU（cann_local_lm 融合算子只在 CPU 实现，不计划移植到 NPU）
  - `--quant-device` 只影响 subprocess，不影响主进程
  - 纯 CPU 环境（无 NPU/GPU）`--quant-method gptq` 回落 `cpu`（dopt_lm 支持，7B+ 极慢）；或用 `--quant-path` 消费离线产物
  - cann_local_lm 模型层名必须与 transformers 层名一致（轻量化工具量化的 HF 模型层名要能 port 到 cann_local_lm 模型，靠命名一致）

## 替代方案 (Alternatives)

### 替代方案 A: 主进程也用计算设备（装真 torch_npu）

converter venv 装真 torch_npu（如 2.9.0.post6）+ 重命名 `torch_npu/dynamo/torchair` → `torchair.disabled`（让 `dynamo/__init__.py` 的 `if not exists` 命中 → 不装代理 → 源码 torchair 直接用）。

否决理由：
1. 撞 torch_npu 版本依赖（`npu_chunk_gated_delta_rule` schema 仅 ≥2.7.1.post8，须匹配 torch 版本）
2. 需一次性重命名 wrapper，复杂度高
3. converter 不需要真 torch_npu 算子（fake-meta 由 `_library.py` 自提供即可），装了反而引入代理坑
4. 即便重命名 workaround 能跑，仍需主进程同时满足 CPU stubs（export）+ NPU（GPTQ），环境配置矛盾

### 替代方案 B: CPU 跑 GPTQ

轻量化工具在 CPU 上跑 GPTQ（Hessian + Cholesky 全 CPU）。

> **状态变更（2026-08-26）**：dopt_lm 已支持 CPU（`SUPPORTED_DEVICE_TYPES` 含 `cpu`，`train.py`/`modelutils.py` 有 CPU 分支 `dispatch_dtype=float32`）。`--quant-device cpu` 或无加速器自动回落已**落地**（`auto_quant._detect_device_type` 不再对纯 CPU 报错）。故本方案从"否决"转为"**已支持，性能受限**"。

性能受限（仍建议 NPU/GPU 或 `--quant-path` 兜底）：
1. 7B+ 模型 Hessian 矩阵数 GB，CPU 极慢（耗时不可接受，仅小模型/调试适用）
2. GPTQ 校准需 batch forward 累积激活，CPU 吞吐不足

> 注：CPU GPTQ 仍走 **subprocess 隔离**（`DEVICE=cpu` 传给 subprocess，系统 Python + CPU torch），不违背本 ADR 主决策（load_model 写死 CPU + subprocess 隔离两套环境）。

### 替代方案 C: 内存桥接（进程内，不 spawn subprocess）

converter 主进程内 import dopt_lm 跑 GPTQ，省 subprocess。

否决理由：
1. 环境冲突无解（converter venv CPU stubs 与轻量化工具需真 NPU 矛盾，同进程无法同时满足）
2. dopt_lm 的 `args` 全局单例 + `ptq_loop` 重入风险（进程内重入会出错）
3. 内存峰值 3-4×（cann_local_lm + 轻量化工具 2 份 HF + state_dict）

## 关联

- 详见 cli_design.md "运行环境支持（NPU / GPU / 纯 CPU）" 三环境矩阵 + "converter 与 torch_npu 的关系" 三个耦合问题
- quant_design.md "一键式量化方案选型" 三方案对比（A 改轻量化工具 / B 内存 / C subprocess，选 C 即本 ADR）
- 若未来 cann_local_lm 融合算子移植到 NPU（支持真执行）：本 ADR 的"load_model 写死 CPU"约束可松动，但 subprocess 隔离仍可能保留（规避环境冲突）
