# export_and_load_epair — epair 导出 + C++ 加载基础 e2e 验证

cann_lm 最小端到端示例：把一个小网络导出为 epair（torch.export → torchair GE IR），再用 C++ `EpairModelLoader` 加载、构建 GE Graph、dump ONNX，验证整条「PyTorch → epair → GE Graph」链路通。是接入新模型/新算子前的最小回归用例。

> LLM 级（gemma4/qwen3/qwen3.5）导出+OM 编译+出字见 [`../export_llm_model/`](../export_llm_model)（`om_npu_verify/`）。

## 目录结构

```
export_and_load_epair/
├── simple_network.py                 # SimpleNetwork 模型 (rmsnorm+matmul+mul+add 残差)
├── export_and_save_epair.py         # 导出 SimpleNetwork → epair_model/ (+ .pt2 回环用)
├── compare_single_op_vs_delegate.py # [用例3] ep 回环 (ep.module() 跑) + C++ load 验证
├── main.cpp                         # C++ epair→GE Graph→ONNX dump→(可选)OM 编译 (EpairModelLoader)
├── CMakeLists.txt                    # 构建 export_and_load_example (链 ge_compiler/ge_runner/acl_rt + torch_delegate_backend)
├── reg_register.h                    # → 软链 ../export_llm_model/reg_register.h (IR 校准版 81 ops, 单一源)
├── run.sh                            # 一键脚本: 环境检查 + 后端构建 + 导出 + C++ 编译 + C++ 加载 (simple|addrmsnorm)
├── epair_model/                      # SimpleNetwork 导出产物 (ep_json + weight_info + weights_*.bin)
├── epair_model.pt2 / input_0.bin    # Python 回环 / C++ 验证输入
├── export_addrmsnorm_model/         # [用例2] AddRmsNorm 单算子 demo (export/compile 子命令)
└── export_epair_with_customop/       # [用例4] 自定义算子 demo (fused_attn/rope/beam 的 schema+GE converter)
```

## 用例

### 用例1 — SimpleNetwork（`bash run.sh simple`，默认）

```bash
source /home/developer/Ascend/cann/set_env.sh
bash run.sh simple
# run.sh 自动: [1]环境检查 → [2]构建后端 lib → [3]导出 epair → [4]C++ 编译 → [5]C++ 加载 (load→GetGEGraph→ONNX dump)
```

### 用例2 — AddRmsNorm 单算子（`bash run.sh addrmsnorm`）

```bash
bash run.sh addrmsnorm   # export_addrmsnorm_model.py 的 export+compile 两步 + C++ 加载
```

### 用例3 — ep 回环对比（`compare_single_op_vs_delegate.py`，单独跑）

```bash
# 前置: converter stubs 环境 (run.sh 的 [3]导出 阶段同款 env)
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0; export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
VENV=/mnt/workspace/torch-delegate-harmony/.venv
$VENV/bin/python3 compare_single_op_vs_delegate.py
# [1/2] Python: epair_load(.pt2) → ep.module()(CPU eager) → 输出 shape+数据
# [2/2] C++:   export_and_load_example 加载同一 epair (load→GetGEGraph→ONNX dump)
```

### 用例4 — 自定义算子（`export_epair_with_customop/`，单独跑）

```bash
cd export_epair_with_customop
# (同用例3 的 stubs env)
$VENV/bin/python3 run_epair_customop.py export  --output generate/customop.pt2 --device cpu
$VENV/bin/python3 run_epair_customop.py compile --ep-path generate/customop.pt2 --epair-path generate/customop.epair
# 验证 fused_infer_attention_score / rope_in_place / beam_search 的 schema+fake+GE converter 在 stubs 下能导出 epair (.epair 归档)
```

## 环境

**环境检查内置在 `run.sh` 的 [1] 阶段**（不在此重复），覆盖 9 项：
CANN(`ASCEND_HOME_PATH`) / NPU(`npu-smi`) / cmake / venv python / torch / torchair / torch_delegate / stubs 目录 / backend lib(`libtorch_delegate_backend.so`) / `TORCH_PKG_DIR`。

- 跑用例1/2 前只需 `source /home/developer/Ascend/cann/set_env.sh`，其余 env（venv/stubs/PYTHONPATH）由 `run.sh` 注入。
- 跑用例3/4（不经过 run.sh）需手动 source 上面用例3 的 stubs env 块。
- backend lib 缺失时 `run.sh` 自动触发 `cann_lm/converter/export/csrc/build_csrc.sh` 构建。

## SKIP_OM_BUILD / OM 编译

`main.cpp` 的 OM 编译（`aclgrphBuildModel`）受 `SKIP_OM_BUILD` 控制（**值判断**）：
- **默认 `SKIP_OM_BUILD=1`**（`run.sh` 默认）→ 只做 **load + GetGEGraph + ONNX dump** 验证（本例主用途，快）。
- `SKIP_OM_BUILD=0`（或 unset）→ **编译 OM**（`.om` 生成）。

```bash
SKIP_OM_BUILD=0 bash run.sh simple   # 编出 simple.om
```

**OM 编译是 CPU 侧流程，不需 NPU 设备**；但**需要算子（op_info + kernel）在 CANN 中对目标 Soc 存在**。Soc 由 `SOC_VERSION` 指定（`run.sh` 默认 `Ascend910_9362`，本机芯片、算子齐全）。
- 用错误 Soc（如 `Ascend950DT_950x`）会因算子 kernel 缺失/编不过失败（`aclgrphBuildModel failed, status: 1343225857`，mat_mul_v3 `WaitFlagDevLocal` 未声明等）——是算子对那个 Soc 不存在，非 NPU 问题。
- NPU 仅在后续 **执行/出字** OM（`acl.mdl.execute`）时需要，本例不执行。

> LLM 级完整 OM 编译+出字见 [`../export_llm_model/om_npu_verify`](../export_llm_model/om_npu_verify)（同样 `SOC_VERSION=Ascend910_9362` + 校准 reg）。

## reg_register.h

本目录 `reg_register.h` 是指向 `../export_llm_model/reg_register.h`（IR 校准版 81 ops，单一源）的软链。改算子注册请改那个 master 文件，所有用例同步生效。

## 末尾 abort 说明

`main.cpp` 末尾 `exit -6` / `corrupted size vs. prev_size` 是 GE finalize 退出时的堆损坏（在 `aclgrphSaveModel`/DumpToFile 之后触发，产物已保存），不影响验证结果——属已知 GE quirk（`compare_single_op_vs_delegate.py` 的 `[2/2]` 对此容错为 soft-pass）。
