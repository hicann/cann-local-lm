# export_epair_and_compile — epair 导出 + OM 编译 e2e 验证

最小端到端示例：把一个小网络导出为 epair（`torch.export` → `lite_export_context` + torchair GE IR），再用 `cann_lm_compile` 编译 OM，验证整条「PyTorch → epair → OM」链路通。

> 前置：`.run` 包已安装（`/opt/cann-device-lm/`），`cann_lm` whl 已装进 P1 venv。
>
> LLM 级（gemma4/qwen3.5）导出+OM 编译+出字见 [`../export_llm_model/`](../export_llm_model)。

## 目录结构

```
export_epair_and_compile/
├── simple_network.py                 # SimpleNetwork 模型 (rmsnorm+matmul+mul+add 残差)
├── export_and_save_epair.py         # 导出 SimpleNetwork → epair_model/ (lite_export_context 模式)
├── run.sh                            # 一键脚本: 环境检查 + 导出 epair + cann_lm_compile 编译 OM
├── epair_model/                      # 导出产物 (ep_json + weight_info + simple.om)
└── export_addrmsnorm_model/         # [用例2] AddRmsNorm 单算子 demo
```

## 用例

### 用例1 — SimpleNetwork（`bash run.sh simple`，默认）

```bash
# 前置: source CANN + cann-device-lm env + 设 CONVERTER_PYTHON
source $HOME/Ascend/cann/set_env.sh
source /opt/cann-device-lm/set_env.sh
export CONVERTER_PYTHON=/path/to/venv/bin/python

bash run.sh simple
# run.sh 自动: [1]环境检查 → [2]导出 epair → [3]cann_lm_compile 编译 OM
```

### 用例2 — AddRmsNorm 单算子（`bash run.sh addrmsnorm`）

```bash
bash run.sh addrmsnorm   # export+compile 两步 + cann_lm_compile 编译 OM
```


## 环境

- **`.run` 包**：`source /opt/cann-device-lm/set_env.sh`（设 LD_LIBRARY_PATH + GE 配置）
- **CANN**：`source $HOME/Ascend/cann/set_env.sh`（提供 CANN lib64/driver/opp）
- **P1 venv**：`export CONVERTER_PYTHON=/path/to/venv/bin/python`（装了 torch + torchair + torch_delegate + cann_lm）
- **SOC**：默认 `Ascend910_9362`（`export SOC_VERSION=...` 覆盖）

converter 导出走 **`lite_export_context`**（`LiteConfig(target_soc=...)`，Python 级 fake torch_npu + `_get_device_id=lambda:0`），不需要 C 级 stubs 也不需 NPU 设备。

## OM 编译

OM 编译统一走 **`cann_lm_compile`**（`/opt/cann-device-lm/cann_lm_compile`）：
```
cann_lm_compile <ep_json> <weight_info> <output_name>  →  <output_name>.om
```
- Load epair → `EpairModelLoader` 构建 GE Graph → `aclgrphBuildModel` → `aclgrphSaveModel` → `.om`
- 读 `SOC_VERSION` / `GE_FUSION_SWITCH_FILE` / `GE_EXTERNAL_WEIGHT` 等 env
- **CPU 侧流程，不需 NPU 设备**；需算子在 CANN 中对目标 SOC 存在
- 退出时可能看到 `Aborted` 报错（GE 库已知问题，不影响结果），检查 `.om` 文件存在即成功
