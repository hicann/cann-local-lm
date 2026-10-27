# export_llm_model — LLM 端到端（HF→epair→OM→NPU 出字）

两模型（gemma4 / qwen3.5）端到端流水线：HuggingFace 模型 → epair（torch.export + GE 转换）→ OM（aclgrphBuildModel）→ NPU 出字（acl.mdl.execute）。顶层入口 `run_e2e_pipeline.sh` 串通三阶段。

> 本目录为**开发者工具**，使用源码仓环境。用户面向的端到端用例见 [快速入门](../../../docs/zh/QUICKSTART.md)。
>
> 子目录：`cann_lm_cli_export/`（① CLI 导出）、`om_npu_verify/`（②③ OM 编译 + NPU 出字）。CPU 侧调试见 `../llm_models_run_on_cpu/`，单元测试见 `tests/python/llm_models/`。

## 三阶段

| 阶段 | 做什么 | 产物 |
|---|---|---|
| ① 导出 epair | `cann_lm_converter.py --mode prefill-decode`（torch.export + torchair GE 转换） | `prefill/decode_ep_json_with_ascend_ir.json` + `weight_info.json` + `weights_*.bin` |
| ② OM 编译 | `cann_lm_compile` 加载 epair → GE Graph → `aclgrphBuildModel` | `prefill/prefill.om` + `decode/decode.om` |
| ③ NPU 出字 | `run_<model>_gen.py` 用 acl 接口跑 OM（prefill+decode 逐 token） | 生成的文本 |

## 前置检查（不跑全流程）

```bash
bash run_e2e_pipeline.sh --check-env --model-type gemma4 --hf-path <gemma4_model_dir>
```

## 跑端到端

```bash
# gemma4
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <gemma4_model_dir> --prompt "Hello, my name is"

# qwen3.5
bash run_e2e_pipeline.sh --model-type qwen3_5 --hf-path <qwen3.5_model_dir> --prompt "Hello, my name is"

# 跳过 ③（无 NPU，仅 ①②）
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <dir> --skip-npu

# 复用已有 epair，只跑 ②③
bash run_e2e_pipeline.sh --model-type gemma4 --hf-path <dir> --skip-export --output-dir ./output/e2e_gemma4
```

> converter 后台运行可能在 GE finalize 处挂起（已知），脚本用后台+轮询+kill 绕过。

## 成功标准

- ①：`output/e2e_<model>/{prefill,decode}_ep_json_with_ascend_ir.json` + `weight_info.json` 存在
- ②：`output/e2e_<model>/{prefill,decode}/*.om` 存在（编译末尾可能 `Aborted`，GE 库已知问题，OM 已保存不影响）
- ③：runner 输出连贯文本（prefill 首字 logits 合理 + decode 逐 token 生成到 EOS）

## 目录结构

```
export_llm_model/
├── run_e2e_pipeline.sh        # 顶层 e2e 编排 (①→②③); --check-env 验前置
├── cann_lm_cli_export/        # ① CLI 导出测试
├── om_npu_verify/             # ②③ OM 编译 + NPU 出字
└── output/                    # e2e 产物 (epair + OM, gitignore)
```
