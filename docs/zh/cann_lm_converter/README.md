# CANN-LM Converter 设计文档（总览）

简体中文 | [English](../../en/cann_lm_converter/README.md)

> 把 HuggingFace 模型转换为 epair 格式的 CLI 工具设计文档集。支持 gemma4 / qwen3_5（后续可扩展）。
>
> 文档按 **CLI / 模型导出 / 量化** 三模块组织，其余为参考文档。

## 目录

- [文档布局](#文档布局)
- [端到端流程总览](#端到端流程总览)
- [模型差异总览](#模型差异总览)
- [精度验证汇总](#精度验证汇总)
- [模块导航](#模块导航)
- [待实现与遗留](#待实现与遗留)
- [快速开始](#快速开始)

---

<a id="文档布局"></a>
## 文档布局

```
docs/cann_lm_converter/
├── README.md                              ← 本文件（顶层总览 + 导航）
│
├── cli/                                   ← CLI 模块
│   ├── cli_design.md                      ← CLI 设计
│   └── INDEX.md                           ← CLI 模块索引
├── export/                                ← 模型导出模块
│   ├── export_design.md                   ← 导出设计
│   └── INDEX.md                           ← 导出模块索引
├── quant/                                 ← 量化模块
│   ├── quant_design.md                    ← 量化设计
│   └── INDEX.md                           ← 量化模块索引
│
├── ─── 参考文档 ───
├── reference_litert_lm_export.md          ← LiteRT-LM 外部参考
├── reference_npu_op_elimination.md        ← NPU 算子消除技能
└── feature_dump_design.md                ← Dump 维测能力（FX node 级逐算子中间张量 dump）
```

<a id="端到端流程总览"></a>
## 端到端流程总览

converter 把 HF 模型导出为 epair，供 C++ 后端加载编译 OM。三阶段由 CLI 编排，各阶段归属模块如下：

```
cann_lm_converter.py (CLI)                                          ← cli_design.md
    │
    ├── parse_args()  ← --hf-path, --output-dir, --target-soc, --mode,
    │                    [--quant-path | --quant-method gptq]
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
    │   swap_from_dopt(model, quant_path)
    │   swap_embeddings_from_dopt(sd, model)   # embed 自动检测 (embed_quant 非空时)
    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)             ← export_design.md
    │   ① register_ge_converters
    │   ② wrapper = artifact.wrapper_cls(model, ...).eval()
    │   ③ wrapper.get_sample_inputs(use_embeddings)
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

<a id="模型差异总览"></a>
## 模型差异总览

两个模型接口不一致，converter 通过注册表 + adapter 抹平差异。各差异的详细处理见 [export_design.md](./export/export_design.md)。

### Config 加载差异

| 模型 | from_pretrained | from_json_file | to_dict | config_json_key |
|------|:-:|:-:|:-:|---|
| gemma4 | 无 | 无 | 有 | `text_config` |
| qwen3_5 | 有 | 有 | 无 | `text_config` |

> `build_config` 统一走字段过滤路径（按 `config_json_key` 取子字典 + dataclass 字段名过滤 + `__post_init__`），不再调用 `from_pretrained`/`from_json_file`。详见 [export_design.md](./export/export_design.md) "build_config"。

### forward 签名差异

| 参数 | gemma4 | qwen3_5 |
|------|:------:|:-------:|
| input_ids | 有 | 有 |
| attention_mask | 有 | 有 |
| position_ids | 无（内部计算） | 无（内部计算） |
| past_key_values | `list` | `SimpleKVCache` |
| inputs_embeds | 有 | 有 |
| labels | 有 | 有 |
| output_hidden_states | 有 | 无 |
| use_cache | 无 | 有 |
| shared_kv_states | 有（独有） | 无 |
| delta_cache | 无 | 有（独有） |

### forward 返回差异

| 返回 key　　　　 | gemma4　　 | qwen3_5　　|
| ------------------| ------------| ------------|
| loss　　　　　　 | 有　　　　 | 有　　　　 |
| logits　　　　　 | 有　　　　 | 有　　　　 |
| past_key_values　| 有　　　　 | 有　　　　 |
| shared_kv_states | 有（独有） | 无　　　　 |
| hidden_states　　| 有　　　　 | 无　　　　 |
| delta_cache　　　| 无　　　　 | 有（独有） |

### 权重 key 重映射差异

| 模型 | 重映射策略 | 说明 |
|------|-----------|------|
| gemma4 | `model.language_model.*` → `model.*` | 去掉 VLM 前缀 |
| qwen3_5 | `model.language_model.*` → `model.*` | 通过 `_remap_key` 去掉 VLM 前缀 |

> 两模型的 `ExportWrapper` 子类差异（cache 类型 / forward kwargs / PLE 处理）见 [export_design.md](./export/export_design.md) "两个子类的差异"。

<a id="精度验证汇总"></a>
## 精度验证汇总

所有模型生成的 token 序列与 HF 完全一致，精度差异在 float16/bfloat16 正常范围内（qwen3.5-0.8B max_diff=0.019 / gemma4-E2B max_diff=0.17）。

> 误差积累分析（ClippableLinear / 175 RMSNorm / PLE 累积 / KV 共享不衰减）、scatter-write 下 bfloat16 attention 精度策略、tie-breaking 问题详见 [export_design.md](./export/export_design.md) "精度差异分析"。

<a id="模块导航"></a>
## 模块导航

### 设计文档（三模块）

| 文档 | 范围 | 关键内容 |
|------|------|---------|
| [cli_design.md](./cli/cli_design.md) | `cann_lm_converter.py` | `convert()` 程序化入口 / 命令行参数 / 完整调用链 / prefill-decode CLI / 环境变量 / 使用示例 |
| [export_design.md](./export/export_design.md) | `model_registry` / `loader` / `saver` / `export_wrapper` / `export/`（含 C++ `csrc/`） | `ModelArtifact` / `build_config` / `ExportWrapper` 基类 + 两子类 / 统一图 I/O 契约 / 纯 embeddings 模式 / `pack_weights` / `consolidated_weights_save` / `EpairModelLoader` / `GeWeightLoader` 懒加载 / `_assert_tensor_metadata` / GE 算子注册两层 / prefill-decode 分离 / 新增模型接入 / 精度分析 |
| [quant_design.md](./quant/quant_design.md) | `cann_local_lm/converter/quant/` | W4A8 s8s4 方案 / dopt GPTQ 框架 / `materialize_int4_per_group_symmetric` / `QuantLinear` / `swap_from_dopt` / `npu::npu_quant_matmul` 算子 / C++ 注册 / CANN V3/V4 源码分析附录 |
| `embedding_quant_design.md` | embed_tokens 量子主题 | W8 host gather / `materialize_int8_per_channel_symmetric` / `swap_embeddings_from_dopt` |

### 模块索引

| 索引 | 模块 |
|------|------|
| [INDEX.md](./cli/INDEX.md) | CLI 模块文档 + 代码索引 |
| [INDEX.md](./export/INDEX.md) | 导出模块文档 + 代码索引 |
| [INDEX.md](./quant/INDEX.md) | 量化模块文档 + 代码索引 |

### 参考文档

| 文档 | 主题 |
|------|------|
| [reference_litert_lm_export.md](./reference_litert_lm_export.md) | LiteRT-LM + litert-torch 外部参考（prefill/decode 两签名 / KV cache pytree / wrapper 统一 I/O） |
| [reference_npu_op_elimination.md](./reference_npu_op_elimination.md) | NPU 算子消除技能（host 预构 mask / repeat_kv concat / RoPE 预缓存 Gather） |
| [feature_dump_design.md](./feature_dump_design.md) | Dump 维测能力（FX node 级逐算子中间张量 dump / `gen_golden_with_dump.py` 工具 / 能力对比 torch hook / 产物布局 / 应用场景） |

<a id="待实现与遗留"></a>
## 待实现与遗留

按模块分组，详情见各模块文档的遗留章节。

- **导出模块** — 统一 cache 签名（全量版）/ generation_config.json 处理 / 图导出 I/O 接口 / qwen3.5 线性注意力 / qwen3.5 scatter-write 改造。详见 [export_design.md](./export/export_design.md) "待实现与遗留"
- **量化模块** — dopt 函数超规待拆分。详见 [quant_design.md](./quant/quant_design.md) "已知问题与遗留"

### 已解决

详见各模块文档"已解决"条目：[export_design.md](./export/export_design.md) / [quant_design.md](./quant/quant_design.md)。

<a id="快速开始"></a>
## 快速开始

详见 [快速入门](../QUICKSTART.md)（安装 / 导出 epair / 量化导出 / omc 编译 / 部署到鸿蒙设备）和 [构建指南](../build_guide.md)（构建前置 / 一键构建 / FAQ）。

> CLI 参数速查见 [cli/INDEX.md](./cli/INDEX.md)「命令行参数速查」，完整设计见 [cli_design.md](./cli/cli_design.md)。
