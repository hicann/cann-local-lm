# CANN-LM Converter 设计文档（总览）

> **顶层总览入口**。将 HuggingFace 模型转换为 epair 格式的 CLI 工具，支持 gemma4 / qwen3 / qwen3_5。
>
> 详细设计已按模块拆分：
> - **CLI 模块** → [cli_design.md](./cli/cli_design.md)（`cann-lm-converter.py` 入口 / `convert()` / 命令行参数 / 调用链 / 环境变量 / 使用示例）
> - **模型导出模块** → [export_design.md](./export/export_design.md)（`model_registry` / `loader` / `saver` / `ExportWrapper` / `epair_utils` / C++ 加载侧 / GE 算子注册 / prefill-decode / 新增模型接入 / 精度分析）
> - **量化模块** → [quant_design.md](./quant/quant_design.md)（W4A8 s8s4 / dopt GPTQ 框架 / `quant_matmul_s8s4` 算子 / swap 接入）
> - **模块索引** → [INDEX.md](./cli/INDEX.md) / [INDEX.md](./export/INDEX.md) / [INDEX.md](./quant/INDEX.md)
> - **参考文档** → 见 [模块导航](#模块导航)
>
> 最后更新: 2026-08-15

## 目录

- [端到端流程总览](#端到端流程总览)
- [模型差异总览](#模型差异总览)
- [精度验证汇总](#精度验证汇总)
- [模块导航](#模块导航)
- [待实现与遗留](#待实现与遗留)

---

## 端到端流程总览

converter 把 HF 模型导出为 epair，供 C++ 后端加载编译 OM。三阶段由 CLI 编排，各阶段归属模块如下：

```
cann-lm-converter.py (CLI)                                          ← cli_design.md
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --mode,
    │                    --use-embeddings, [--quant-path | --quant-method gptq]
    ▼
convert(hf_path, output_dir, dtype, ...)
    │
    ▼
loader.load_model(hf_path, dtype, device)                          ← export_design.md
    │   ① 读 config.json
    │   ② model_registry.detect_model_type → ModelEntry
    │   ③ build_config (字段过滤 + __post_init__)
    │   ④ entry.model_cls(config).to(dtype).eval()
    │   ⑤ entry.load_weights_fn (safetensors → key 重映射 → param.copy_)
    │   ⑥ collect_tokenizer_files
    │   → ModelArtifact
    ▼
(可选) 量化 swap                                                    ← quant_design.md
    │   swap_to_quant_linear_from_dopt(model, quant_path, ...)
    │   swap_embeddings_from_dopt(sd, model)   # embed 自动检测 (embed_quant 非空时)
    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)             ← export_design.md
    │   ① register_ge_converters
    │   ② wrapper = artifact.wrapper_cls(model, ...).eval()
    │   ③ wrapper.get_sample_inputs(seq_len, use_embeddings)
    │   ④ (use_embeddings) _extract_host_side_weights + detach_externalized_lookups
    │   ⑤ torch.export.export(wrapper, sample, strict=False)
    │   ⑥ torch_delegate.export_to_epair(ep) → epair
    │   ⑦ del ep, wrapper, model; gc.collect()       # OOM 防护
    │   ⑧ consolidated_weights_save (weights_*.bin + host_weights_*.bin + ep_json)
    │   ⑨ copy_tokenizer_files + save_model_config
    ▼
C++ EpairModelLoader → GetGEGraph → aclgrphBuildModel → .om        ← export_design.md "C++ 加载侧"
```

> 完整 CLI 参数与调用链细节见 [cli_design.md](./cli/cli_design.md)；加载/保存/导出每步的实现细节见 [export_design.md](./export/export_design.md)。

## 模型差异总览

三个模型接口不一致，converter 通过注册表 + adapter 抹平差异。各差异的详细处理见 [export_design.md](./export/export_design.md)。

### Config 加载差异

| 模型 | from_pretrained | from_json_file | to_dict | config_json_key |
|------|:-:|:-:|:-:|---|
| gemma4 | 无 | 无 | 有 | `text_config` |
| qwen3 | 有 | 有 | 有 | 顶层（无嵌套） |
| qwen3_5 | 有 | 有 | 无 | `text_config` |

> `build_config` 统一走字段过滤路径（按 `config_json_key` 取子字典 + dataclass 字段名过滤 + `__post_init__`），不再调用 `from_pretrained`/`from_json_file`。详见 [export_design.md](./export/export_design.md) "build_config"。

### forward 签名差异

| 参数 | gemma4 | qwen3 | qwen3_5 |
|------|:------:|:-----:|:-------:|
| input_ids | 有 | 有 | 有 |
| attention_mask | 有 | 有 | 有 |
| position_ids | 无（内部计算） | 有 | 无（内部计算） |
| past_key_values | `list` | `list` | `SimpleKVCache` |
| inputs_embeds | 有 | 有 | 有 |
| labels | 有 | 有 | 有 |
| output_hidden_states | 有 | 有 | 无 |
| use_cache | 无 | 无 | 有 |
| shared_kv_states | 有（独有） | 无 | 无 |
| delta_cache | 无 | 无 | 有（独有） |

### forward 返回差异

| 返回 key　　　　 | gemma4　　 | qwen3 | qwen3_5　　|
| ------------------| ------------| -------| ------------|
| loss　　　　　　 | 有　　　　 | 有　　| 有　　　　 |
| logits　　　　　 | 有　　　　 | 有　　| 有　　　　 |
| past_key_values　| 有　　　　 | 有　　| 有　　　　 |
| shared_kv_states | 有（独有） | 无　　| 无　　　　 |
| hidden_states　　| 有　　　　 | 有　　| 无　　　　 |
| delta_cache　　　| 无　　　　 | 无　　| 有（独有） |

### 权重 key 重映射差异

| 模型 | 重映射策略 | 说明 |
|------|-----------|------|
| gemma4 | `model.language_model.*` → `model.*` | 去掉 VLM 前缀 |
| qwen3 | 无需重映射 | HF key 直接匹配 |
| qwen3_5 | `model.language_model.*` → `model.*` | 通过 `_remap_key` 去掉 VLM 前缀 |

> 三模型的 `ExportWrapper` 子类差异（cache 类型 / forward kwargs / PLE 处理）见 [export_design.md](./export/export_design.md) "三个子类的差异"。

## 精度验证汇总

所有模型生成的 token 序列与 HF 完全一致，精度差异在 float16/bfloat16 正常范围内。

| 模型 | logits max_diff (vs HF) | tokens match | 层数 | 架构特点 |
|------|----------------------|-------------|------|---------|
| qwen3-0.6B | 0.023 | ✅ | 28 | 标准 GQA |
| qwen3.5-0.8B | 0.019 | ✅ | 24 | 混合 (6 full + 18 linear) |
| gemma4-E2B | 0.17 | ✅ | 35 | KV 共享 + sliding + per-layer embeddings |

> gemma4 误差积累分析（ClippableLinear / 175 RMSNorm / PLE 累积 / KV 共享不衰减）、scatter-write 下 bfloat16 attention 精度策略、tie-breaking 问题见 [export_design.md](./export/export_design.md) "精度差异分析"。qwen3.5 的 `ChunkGatedDeltaRule` 算子精度调试见 [reference_qwen3_5_export.md](./reference_qwen3_5_export.md)。

## 模块导航

### 设计文档（三模块）

| 文档 | 范围 | 关键内容 |
|------|------|---------|
| [cli_design.md](./cli/cli_design.md) | `cann-lm-converter.py` | `convert()` 程序化入口 / 命令行参数 / 完整调用链 / prefill-decode CLI / 环境变量 / 使用示例 |
| [export_design.md](./export/export_design.md) | `model_registry` / `loader` / `saver` / `export_wrapper` / `export/`（含 C++ `csrc/`） | `ModelArtifact` / `build_config` / `ExportWrapper` 基类 + 三子类 / 统一图 I/O 契约 / 纯 embeddings 模式 / `pack_weights` / `consolidated_weights_save` / `EpairModelLoader` / `GeWeightLoader` 懒加载 / `_assert_tensor_metadata` / GE 算子注册两层 / prefill-decode 分离 / 新增模型接入 / 精度分析 |
| [quant_design.md](./quant/quant_design.md) | `cann_lm/converter/quant/` | W4A8 s8s4 方案 / dopt GPTQ 框架 / `materialize_int4_with_scale` / `QuantLinear` / `swap_to_quant_linear_from_dopt` / 方案 A 自研 `quant_matmul_s8s4` 算子 / C++ 注册 / CANN V3/V4 源码分析附录 |
| [embedding_quant_design.md](./quant/embedding_quant_design.md) | embed_tokens 量子主题 | W8 host gather / `materialize_int8_per_channel` / `swap_embeddings_from_dopt` |

### 模块索引

| 索引 | 模块 |
|------|------|
| [INDEX.md](./cli/INDEX.md) | CLI 模块文档 + 代码索引 |
| [INDEX.md](./export/INDEX.md) | 导出模块文档 + 代码索引 |
| [INDEX.md](./quant/INDEX.md) | 量化模块文档 + 代码索引 |

### 参考文档

| 文档 | 主题 |
|------|------|
| [reference_pipeline_walkthrough.md](./reference_pipeline_walkthrough.md) | gemma4 端到端 6 阶段流程示例（模型构建→注册→CLI→加载→导出→C++ OM） |
| [reference_python_to_om_self_test_flow.md](./reference_python_to_om_self_test_flow.md) | 改 Python 源码到 OM 出字的 7 阶段自验证流程（fused_ops / modeling / ge_converter / reg_register.h） |
| [reference_qwen3_5_export.md](./reference_qwen3_5_export.md) | qwen3.5 混合架构导出专题（`ChunkGatedDeltaRule` 维度修复 / delta_states tensor 化 / ACL OM 调试 / npu-op-elimination 改造） |
| [reference_litert_lm_export.md](./reference_litert_lm_export.md) | LiteRT-LM + litert-torch 外部参考（prefill/decode 两签名 / KV cache pytree / wrapper 统一 I/O） |
| [reference_npu_tuned_export_qwen3.md](./reference_npu_tuned_export_qwen3.md) | npu_tuned_export qwen3 fork 改造参考（预分配 KV buffer + scatter-write + 预缓存 RoPE + host 4D mask） |
| [reference_npu_op_elimination.md](./reference_npu_op_elimination.md) | NPU 算子消除技能（host 预构 mask / repeat_kv concat / RoPE 预缓存 Gather） |

## 待实现与遗留

按模块分组，详情见各模块文档的遗留章节。

### 导出模块（详见 [export_design.md](./export/export_design.md) "待实现与遗留"）

- **统一 cache 签名（全量版）** — 轻量版已实现，全量版待做（各 attention 加 scatter-write 路径，一张静态图覆盖 prefill+decode）
- **generation_config.json 处理** — 需和 llm_engine 确认使用哪些字段后再实现
- **图的导出输入输出接口** — 需确定如何统一导出输入输出
- **qwen3.5 线性注意力导出** — 详见 [reference_qwen3_5_export.md](./reference_qwen3_5_export.md)（epair 层已解决，OM 编译+ACL 运行已通）
- **qwen3 / qwen3.5 scatter-write 改造** — 当前仅 gemma4 实现，详见 [reference_qwen3_5_export.md](./reference_qwen3_5_export.md) "scatter-write 改造"

### 量化模块（详见 [quant_design.md](./quant/quant_design.md) "已知问题与遗留"）

- **一步化流程未实现** — GPTQ 与导出仍是两步磁盘桥接，三处改动点待落地
- **dopt gemma4 多模态阻塞** — dopt GPTQ 循环写死文本-only `ForCausalLM` 结构
- **dopt_lm `heissen` 拼写错误** — 系统性 `Hessian` 拼写错误待重命名
- **dopt 函数超规** — `swap_to_quant_linear_from_dopt` 76 行 / `materialize_int4_with_scale` 6 参数，待按 dataclass 拆分收敛

### 已解决

- ~~prefill/decode 分离~~ — 已实现（gemma4 Python 导出 + C++ OM 编译全链路跑通）
- ~~新 ExportWrapper ep_json 格式与 C++ 解析器不兼容~~ — C++ `parse_named_arg` 已支持双格式
- ~~aten.index_copy.default GE converter 缺失~~ — 用 `ge.InplaceUpdate` + `ge.TransposeD` 实现
- ~~纯 embeddings 全链路~~ — qwen3-0.6B + gemma4-E2B 导出+OM 编译跑通
- ~~qwen3.5 线性注意力 export_to_epair segfault~~ — 用 `ChunkGatedDeltaRule` 融合算子替代 Python 循环展开
