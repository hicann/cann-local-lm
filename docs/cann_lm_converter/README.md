# CANN-LM Converter 文档导航

> 把 HuggingFace 模型转换为 epair 格式的 CLI 工具设计文档集。支持 gemma4 / qwen3 / qwen3_5（后续可扩展）。
>
> 文档按 **CLI / 模型导出 / 量化** 三模块组织，其余为参考文档。

## 文档布局

```
docs/cann_lm_converter/
├── README.md                              ← 本文件（顶层导航）
├── converter_design.md                    ← 总览入口（跨模块调用链 + 模型差异总览 + 精度汇总）
│
├── cli/                                   ← CLI 模块
│   ├── cli_design.md                      ← CLI 设计
│   └── INDEX.md                           ← CLI 模块索引
├── export/                                ← 模型导出模块
│   ├── export_design.md                   ← 导出设计
│   └── INDEX.md                           ← 导出模块索引
├── quant/                                 ← 量化模块
│   ├── quant_design.md                    ← 量化设计
│   ├── dopt_design.md                     ← dopt vendored GPTQ 框架
│   ├── embedding_quant_design.md          ← embedding 量化子主题
│   └── INDEX.md                           ← 量化模块索引
│
├── ─── 参考文档（顶层，reference_ 前缀） ───
├── reference_pipeline_walkthrough.md      ← gemma4 端到端 6 阶段流程
├── reference_python_to_om_self_test_flow.md ← 改源码到 OM 出字的自验证流程
├── reference_qwen3_5_export.md            ← qwen3.5 混合架构导出专题
├── reference_litert_lm_export.md          ← LiteRT-LM 外部参考
├── reference_npu_tuned_export_qwen3.md    ← npu_tuned_export 参考
└── reference_npu_op_elimination.md        ← NPU 算子消除技能
```

## 三模块入口

| 模块 | 设计文档 | 索引 | 范围 |
|------|---------|------|------|
| **CLI** | [cli_design.md](./cli/cli_design.md) | [INDEX.md](./cli/INDEX.md) | `cann_lm_converter.py` 入口 / `convert()` / 命令行参数 / 调用链 / 环境变量 |
| **模型导出** | [export_design.md](./export/export_design.md) | [INDEX.md](./export/INDEX.md) | `model_registry` / `loader` / `saver` / `ExportWrapper` / `epair_utils` / C++ 加载侧 / GE 算子注册 |
| **量化** | [quant_design.md](./quant/quant_design.md) | [INDEX.md](./quant/INDEX.md) | W4A8 s8s4 / dopt GPTQ / `quant_matmul_s8s4` / swap 接入 |

子主题：[embedding_quant_design.md](./quant/embedding_quant_design.md)（embed_tokens 的 W8 host gather 量化）

## 总览入口

[converter_design.md](./converter_design.md) — 跨模块调用链总览 + 模型差异总览（Config / forward 签名 / forward 返回 / 权重 key 重映射）+ 精度验证汇总 + 待实现与遗留索引。

## 参考文档

| 文档 | 主题 |
|------|------|
| [reference_pipeline_walkthrough.md](./reference_pipeline_walkthrough.md) | gemma4 端到端 6 阶段流程示例（模型构建→注册→CLI→加载→导出→C++ OM） |
| [reference_python_to_om_self_test_flow.md](./reference_python_to_om_self_test_flow.md) | 改 Python 源码到 OM 出字的 7 阶段自验证流程（fused_ops / modeling / ge_converter / reg_register.h） |
| [reference_qwen3_5_export.md](./reference_qwen3_5_export.md) | qwen3.5 混合架构导出专题（`ChunkGatedDeltaRule` 维度修复 / delta_states tensor 化 / ACL OM 调试 / npu-op-elimination 改造） |
| [reference_litert_lm_export.md](./reference_litert_lm_export.md) | LiteRT-LM + litert-torch 外部参考（prefill/decode 两签名 / KV cache pytree / wrapper 统一 I/O） |
| [reference_npu_tuned_export_qwen3.md](./reference_npu_tuned_export_qwen3.md) | npu_tuned_export qwen3 fork 改造参考（预分配 KV buffer + scatter-write + 预缓存 RoPE + host 4D mask） |
| [reference_npu_op_elimination.md](./reference_npu_op_elimination.md) | NPU 算子消除技能（host 预构 mask / repeat_kv concat / RoPE 预缓存 Gather） |

## 快速开始

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

# 导出 epair（single 模式）
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 8
```

> 完整使用示例（single / use-embeddings / prefill-decode / 量化）见 [cli_design.md](./cli/cli_design.md) "使用示例"。环境配置见 [AGENTS.md](../../AGENTS.md) "运行环境设置"。
