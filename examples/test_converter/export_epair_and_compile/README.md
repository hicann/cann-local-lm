# export_epair_and_compile — epair 导出 + OM 编译 e2e 验证

cann_lm 最小端到端示例：把一个小网络导出为 epair（`torch.export` → `lite_export_context` + torchair GE IR），再用 `cann_lm_compile` 编译 OM，验证整条「PyTorch → epair → OM」链路通。是接入新模型/新算子前的最小回归用例。

> LLM 级（gemma4/qwen3/qwen3.5）导出+OM 编译+出字见 [`../export_llm_model/`](../export_llm_model)（`om_npu_verify/`）。

## 目录结构

```
export_epair_and_compile/
├── simple_network.py                 # SimpleNetwork 模型 (rmsnorm+matmul+mul+add 残差)
├── export_and_save_epair.py         # 导出 SimpleNetwork → epair_model/ (lite_export_context 模式)
├── compare_single_op_vs_delegate.py # [用例3] ep 回环 (ep.module() CPU eager 跑)
├── run.sh                            # 一键脚本: 环境检查 + 导出 epair + cann_lm_compile 编译 OM
├── epair_model/                      # 导出产物 (ep_json + weight_info + simple.om)
└── export_addrmsnorm_model/         # [用例2] AddRmsNorm 单算子 demo
```

## 用例

### 用例1 — SimpleNetwork（`bash run.sh simple`，默认）

```bash
bash run.sh simple
# run.sh 自动: [1]环境检查 → [2]导出 epair → [3]cann_lm_compile 编译 OM
```

### 用例2 — AddRmsNorm 单算子（`bash run.sh addrmsnorm`）

```bash
bash run.sh addrmsnorm   # export+compile 两步 + cann_lm_compile 编译 OM
```

### 用例3 — ep 回环对比（`compare_single_op_vs_delegate.py`，单独跑）

```bash
# 前置: 共享 env (check_env.sh)
source ../common/check_env.sh
python compare_single_op_vs_delegate.py
# [1/2] Python: epair_load(.pt2) → ep.module()(CPU eager) → 输出 shape+数据
# [2/2] C++:   cann_lm_compile 加载同一 epair 编译 OM
```

## 环境

**环境检查内置在 `common/check_env.sh`**（各用例 `source ../common/check_env.sh` 复用）：
- NPU 探测（`npu-smi`）→ `NPU_MODE`（1=真 CANN / 0=CPU lite 模式）
- CANN 自动探测+兜底 `source`（`ASCEND_HOME_PATH` 未设时遍历候选路径）
- 路径自定位派生（`PROJECT_ROOT` + `deps/torch-delegate/.venv` 等）
- `ensure_backend`（.so 缺失→`build_cpp.sh`）+ `ensure_third_party`（json/gtest/makeself 缺失→`build_third_party.sh`）
- LD: 真 CANN lib64+devlib（**无 stubs**，`lite_export_context` 绕过 `aclrtSetDevice`）
- SOC: NPU→`Ascend910_9362` / CPU-only→`Ascend950DT_9575`（950DT 交叉编译）

converter 导出走 **`lite_export_context`**（`LiteConfig(target_soc=...)`，Python 级 fake torch_npu + `_get_device_id=lambda:0`），不需要 C 级 stubs 也不需 NPU 设备。

## OM 编译

OM 编译统一走 **`cann_lm_compile`**（`csrc/tools/compile_main.cc`，由 `build.sh cpp` 构建）：
```
cann_lm_compile <ep_json> <weight_info> <output_name>  →  <output_name>.om
```
- Load epair → `EpairModelLoader` 构建 GE Graph → `aclgrphBuildModel` → `aclgrphSaveModel` → `.om`
- 读 `SOC_VERSION` / `GE_FUSION_SWITCH_FILE` / `GE_EXTERNAL_WEIGHT` 等 env
- **CPU 侧流程，不需 NPU 设备**；需算子在 CANN 中对目标 SOC 存在
- 退出时 GE teardown 可能 abort（`corrupted size vs. prev_size`），OM 在此前已正常产出——`run.sh` 用 `|| true` + `.om` 存在判定成功

## reg_register.h

`reg_register.h`（IR 校准版 ops）在 `cann_lm/converter/export/csrc/inc/external/reg_register.h`（与 `cann_lm_compile` 同源）。`cann_lm_compile` 经 `torch_delegate_backend` 的 PUBLIC include（`inc/external`）解析。改算子注册请改 csrc 那个 master 文件。
