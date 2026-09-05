# export_llm_model — LLM 端到端 (HF→epair→OM→NPU 出字)

三模型（gemma4 / qwen3 / qwen3.5）端到端流水线：HuggingFace 模型 → epair（torch.export + GE 转换）→ OM（aclgrphBuildModel）→ NPU 出字（acl.mdl.execute）。顶层入口 `run_e2e_pipeline.sh` 串通三阶段。

> 单阶段调试/验证脚本分别在子目录：`cann_lm_cli_export/`（① CLI 导出）、`om_npu_verify/`（②③ OM 编译 + NPU 出字）。CPU 侧 PyTorch/epair 层调试与 vs HF 数值对齐见 `../llm_models_run_on_cpu/`，模型定义单元测试见 `tests/llm_models/`。

## 三阶段 × 三套 env（互不叠加，每步子 shell source 对应基底 + unset LD 隔离）

| 阶段 | 做什么 | env 基底 | 关键产物 |
|---|---|---|---|
| ① 导出 epair | `cann_lm_converter.py --mode prefill-decode`（torch.export + torchair GE 转换） | `scripts/set_env_export.sh`（converter venv + torchair stubs，CPU，不装 torch_npu） | `prefill/decode_ep_json_with_ascend_ir.json` + `weight_info.json` + `weights_*.bin` |
| ② OM 编译 | `cann_lm_compile` 加载 epair → GE Graph → `aclgrphBuildModel` | `scripts/set_env_cann.sh`（真 CANN，CPU 侧，SOC + 关 AUB + 权值分离 flag） | `prefill/prefill.om` + `decode/decode.om` |
| ③ NPU 出字 | `run_<model>_gen.py` 用 acl 接口跑 OM（prefill+decode 逐 token） | `scripts/set_env_cann.sh --npu`（真 CANN + DEVICE=npu，系统 python + torch_npu + acl） | 生成的文本 |

> ① 与 ③ 的 LD 互斥（stubs-first ↔ 真 torch_npu），故每步在子 shell 内 `unset LD_LIBRARY_PATH` 再 source 对应基底，防 ① stubs 污染 ②③ 真 CANN。

## 前置检查（不跑全流程）

```bash
bash run_e2e_pipeline.sh --check-env --model-type gemma4 --hf-path <gemma4_model_dir>
```
验 ①②③ 三套 env 前置：① converter venv + torchair stubs + cann_lm_converter；② cmake + `build/cann_lm_compile` + backend lib + CANN；③ NPU（npu-smi）+ `$DOPT_PYTHON`（torch_npu + acl）+ runner + HF 模型。pass=无硬缺项即可跑对应阶段。

## 跑端到端

```bash
# gemma4 (run_gemma4_gen.py PREFILL_SEQ=128, 故导出需 --seq-len 128):
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <gemma4_model_dir> --seq-len 128 --prompt "Hello, my name is"

# qwen3.5 (PREFILL_SEQ=8, 用默认 seq):
bash run_e2e_pipeline.sh --model-type qwen3_5 --hf-path <qwen3.5_model_dir> --prompt "Hello, my name is"

# 跳过 ③ (无 NPU 机器, 仅 ①②):
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <dir> --seq-len 128 --skip-npu

# 复用已有 epair, 只跑 ②③:
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <dir> --skip-export --output-dir ./output/e2e_gemma4
```

gemma4 导出较慢（5B prefill-decode 两图），脚本后台跑 converter + 轮询产物 + kill 挂起的 GE finalize 尾；可用 `RUN_CONV_POLLS=240 RUN_CONV_SLEEP=10` 加大轮询窗口。

## 真实执行记录（2026-09-05，gemma4 E2B）

```bash
RUN_CONV_POLLS=240 RUN_CONV_SLEEP=10 \
  bash run_e2e_pipeline.sh --model-type gemma4 \
    --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it --seq-len 128 \
    --prompt "Hello, my name is"
```

| 阶段 | 结果 | 说明 |
|---|---|---|
| ① 导出 epair | ✓ | prefill+decode ep_json + weight_info 产出（后台 converter + 轮询 kill 挂起尾） |
| ② OM 编译 | ✓ | `prefill.om` + `decode.om`（均 77 in / 71 out）经 `run_compile.sh` |
| ③ NPU 出字 | ✓ | `run_gemma4_gen.py` 经 acl 跑 OM，prefill（seq=128，1 chunk）+ decode loop（16 步到 `<turn|>` EOS） |

**生成结果**：
```
Prompt:   'Hello, my name is'
Generated: "Hello! It's nice to meet you. What is your name? 😊"
```
逐 token decode：`Hello! It's nice to meet you. What is your name? 😊` → `<turn|>`（EOS）。出字连贯（chat-template 风格问候 + 反问），证明 ①→②→③ 全链路数值与机制正确。

产物落 `output/e2e_gemma4/`（`output/` 已 gitignore）：epair + `prefill/prefill.om` + `decode/decode.om`。

## 成功标准

- ①：`output/e2e_<model>/{prefill,decode}_ep_json_with_ascend_ir.json` + `weight_info.json` 存在（GE 转换成功，含 ascend_ir 标注）
- ②：`output/e2e_<model>/{prefill,decode}/*.om` 存在（`aclgrphBuildModel` 成功；末尾 `Aborted exit 134` 是 GE finalize 堆损坏，OM 已保存，不影响）
- ③：runner 输出连贯文本（prefill 首字 logits 合理 + decode 逐 token 生成到 EOS）

## 目录结构

```
export_llm_model/
├── run_e2e_pipeline.sh        # 顶层 e2e 编排 (①→②③); --check-env 验前置
├── cann_lm_cli_export/        # ① CLI 导出测试 (run.sh + run_pd6_cases.sh + verify.py; HF 路径必填 env)
├── om_npu_verify/             # ②③ OM 编译 + NPU 出字 (run_*_gen/om.py + check_env.sh; OM/HF 路径经 env)
└── output/                    # e2e 产物 (epair + OM, gitignore)
```
