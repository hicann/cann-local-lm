# examples/ — cann_lm 端到端用例集

cann_lm 从 **HuggingFace 模型 → CLI 导出 epair → C++ OM 编译 → NPU 运行/出字 → 验证 PASS** 的完整用例集。本文是总览 + 端到端步骤导航，各子目录有自己的 README 细节。

## 总览：端到端流水线

```
HF 模型 (safetensors)
    │  ① CLI 导出 (cann_lm_converter.py, converter venv + torchair stubs)
    ▼
epair (ep_json_with_ascend_ir.json + weights_*.bin + host_weights_*.bin)
    │  ② 产物校验 (verify.py) + IR 算子统计
    ▼
C++ OM 编译 (build/cann_lm_compile + csrc/inc/external/reg_register.h, 真 CANN, CPU 侧)
    │  ③ aclgrphBuildModel → .om
    ▼
OM 运行/出字 (系统 Python + torch_npu + acl + NPU)
    │  ④ acl.mdl.execute (prefill + decode) → 文本 / vs eager 数值
    ▼
验证 PASS (出字合理 + OM≈eager)
```

> 三套环境：① 导出=converter venv（CPU+stubs）；② OM 编译=真 CANN（CPU 侧，**不需 NPU**，需算子对 Soc 存在）；③ OM 运行/出字=系统 Python + torch_npu + NPU。

## 用例分组

| 组　　　　　　　　　 | 目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 覆盖阶段　　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------| -----------------------------------------------------------------------| ------------------| -------------------------------------------------------------------------------------------------------------------------|
| **基础 e2e**　　　　 | [`export_epair_and_compile/`](export_epair_and_compile/)　　　　　　　　　　| ①②③ 导出+C++加载 | 最小回归（SimpleNetwork 单 op）；自带 `run.sh`（含环境检查）+ README　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **LLM 主流程**　　　 | [`export_llm_model/`](export_llm_model/)　　　　　　　　　　　　　　　| 全链路　　　　　 | 两模型端到端（①→②③）；`run_e2e_pipeline.sh`（含 `--check-env` 验前置 + 真实执行记录）；产物 gitignore（`output/`）　　　　　 |
| ├ CLI 导出测试　　　 | `export_llm_model/cann_lm_cli_export/`　　　　　　　　　　　　　　　| ①②　　　　　　　 | `run.sh`(source ①基底) + `verify.py` + `run_pd6_cases.sh`；HF 路径必填 env（`*_HF_PATH`）　　　　　　　　　　　　　 |
| ├ OM 编译+出字　　　 | `export_llm_model/om_npu_verify/`　　　　　　　　　　　　　　　　　　 | ②③　　　　　　　 | `run_*_gen.py`/`run_*_om.py` + `check_env.sh`(source ①基底) + README；OM/HF 路径经 env 传入　　　　　　　　　　　　　|
| └ e2e 编排　　　　　 | `export_llm_model/run_e2e_pipeline.sh`　　　　　　　　　　　　　　　| ①②③　　　　　　 | 串 cli_export→OM 编译→NPU 出字；每步子 shell source 对应基底脚本（隔离 LD）；产物落 `output/`　　　　　　　　　 |
| **CPU 调试验证**　　 | [`llm_models_run_on_cpu/`](llm_models_run_on_cpu/)　　　　　　　　　　　　　| ①③(ep回环)　　　 | 脱离 epair 编译链；`check_env.sh`(source ①基底) + `debug_*/verify_*.py`（ep 回环、vs HF、chunk）+ skills；HF 路径必填 env　　　 |
| └ OM 编译器+算子注册 | `cann_lm/converter/export/csrc/{tools/compile_main.cc, inc/external/reg_register.h}` | ③　　　　　　　　| `compile_main.cc`=EpairModelLoader→OM（产 `build/cann_lm_compile`）；`reg_register.h`=IR 校准 81 ops（master，om_npu_verify/export_epair_and_compile 经 backend PUBLIC include 引用） |
| **CANN 轻量化工具量化 ST**　　 | [`dopt_quantization_st/`](dopt_quantization_st/)　　　　　　　　　　| GPTQ+产物校验　　 | gemma4/Qwen3.5 真实模型路径运行时注入，校准集内置；仓库不含真实权重和机器路径；支持无权重静态自检　　　　　　　　　　　　|

## 端到端流程（从 CLI 到验证 PASS）

以 gemma4 为例（qwen3.5 同理，换 `--hf-path`）。

### ① CLI 导出 epair（converter venv）
```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$TORCHAIR_ROOT/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_DELEGATE_HOME=$TORCH_DELEGATE_HOME
export TORCH_DEVICE_BACKEND_AUTOLOAD=0; export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
VENV=$TORCH_DELEGATE_HOME/.venv

$VENV/bin/python3 cann_lm/converter/cann_lm_converter.py \
  --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
  --output-dir examples/test_converter/export_llm_model/om_npu_verify/epair_gemma4_pd \
  --dtype bfloat16 --mode prefill-decode   # --use-embeddings 默认 True; seq_len 固定 64
```
- **PASS**：`Done. Output: epair_gemma4_pd` + 产物含 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json` + `weight_info.json` + `weights_0.bin` + `host_weights_*.bin`。
- 批量两模型 + 产物校验见 `cann_lm_cli_export/run.sh` + `verify.py`。

### ② 产物校验 + IR 算子统计
```bash
# 结构校验
$VENV/bin/python3 examples/test_converter/export_llm_model/cann_lm_cli_export/verify.py gemma4
# IR 算子清单（确认 FusedInferAttentionScore/RotaryMul 等出现）
python3 -c "import json; ..."   # 见 flow doc 阶段3 one-liner
```
- **PASS**：verify.py 全 `[OK]` + IR 含预期算子。

### ③ C++ OM 编译（真 CANN，CPU 侧，不需 NPU）
```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export SOC_VERSION=Ascend910_9362                 # 算子对该 Soc 存在；错误 Soc → 算子缺失失败
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg   # 关 AUB
# echo '{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}' > /tmp/fusion_autoub_off.cfg  # 一次性建
cd examples/test_converter/export_llm_model
TORCH_PKG_DIR=$TORCH_DELEGATE_HOME/.venv/lib/python3.12/site-packages/torch
cmake -S . -B build -DCANN_ROOT=$ASCEND_HOME_PATH -DTORCH_PKG_DIR=$TORCH_PKG_DIR   # 一次性
cmake --build build -j4
./build/cann_lm_compile om_npu_verify/epair_gemma4_pd/prefill_ep_json_with_ascend_ir.json \
  om_npu_verify/epair_gemma4_pd/weight_info.json /tmp/opencode/gemma4_pd/prefill
./build/cann_lm_compile om_npu_verify/epair_gemma4_pd/decode_ep_json_with_ascend_ir.json  \
  om_npu_verify/epair_gemma4_pd/weight_info.json /tmp/opencode/gemma4_pd/decode
ln -sf prefill_linux_aarch64.om prefill.om   # 在 /tmp/opencode/gemma4_pd/
```
- **PASS**：3 阶段（Load→GetGEGraph→Compile）全 `[PASS]` + `prefill.om`/`decode.om` 生成（~6GB 各）。
- 环境检查：`om_npu_verify/check_env.sh`（Flow 2 段）。

### ④ OM 运行/出字 + 数值验证（系统 Python + torch_npu + NPU）
```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:$PROJECT_ROOT:${PYTHONPATH:-}
cd examples/test_converter/export_llm_model/om_npu_verify

# 出字（prefill OM + decode OM）
$DOPT_PYTHON run_gemma4_gen.py "Hello, my name is"
# 数值（单 OM vs eager, maxdiff）
GEMMA4_OM=/tmp/opencode/gemma4_calib/gemma4_calib_noaub.om $DOPT_PYTHON run_gemma4_om.py
```
- **PASS**：出字合理（如 `"Hello! It's nice to meet you. What is your name? 😊"`）+ `run_gemma4_om.py` `maxdiff<1.0 → MATCH`。
- 环境检查：`om_npu_verify/check_env.sh`（Flow 3 段）。

## 环境速查

| 环境 | 用于 | 关键 |
|---|---|---|
| converter venv（CPU+stubs） | ①导出 | `torchair/build/stubs` 在 LD_LIBRARY_PATH 前；`TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 真 CANN（CPU 侧） | ③OM 编译 | 不带 stubs；`SOC_VERSION=Ascend910_9362`；`GE_FUSION_SWITCH_FILE` 关 AUB；不需 NPU |
| 系统 Python+torch_npu+NPU | ④OM 运行/出字 | `$DOPT_PYTHON`；PYTHONPATH 含 CANN `python/site-packages`；需 NPU |

- 环境检查脚本：`export_llm_model/om_npu_verify/check_env.sh`（三套全覆盖）、`export_epair_and_compile/run.sh`（基础 e2e，内置 phase 1）。

## 产物与 .gitignore

`examples/test_converter/**/epair_*/` 被 `.gitignore` 忽略（挡 `ep_json`/`model_config`/`tokenizer`/`host_weights_*` 等导出产物目录），`build/`/`*.om`/`*.pt2`/`generate/`/`*.bin`/`weight_info.json` 等也有具体规则。**源码（`*.py`/`*.cpp`/`*.h`/`*.sh`/`*.md`/`CMakeLists.txt`）不被忽略**，可正常 `git add` 跟踪（无需 `-f`）。二进制产物可重建，不入库。

## 详细文档

- 各子目录 README（`export_epair_and_compile/`、`export_llm_model/cann_lm_cli_export/`、`export_llm_model/om_npu_verify/`、`llm_models_run_on_cpu/`）。
- converter 设计：[`docs/zh/cann_lm_converter/`](../../docs/zh/cann_lm_converter/)（cli_design / export_design / quant_design）。
