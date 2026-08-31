# 模型导出模块索引

> `model_registry` / `loader` / `saver` / `export_wrapper` / `export/`（含 C++ `csrc/`）— 把 HF 模型导出为 epair 格式。

## 模块文档

| 文档 | 内容 |
|------|------|
| [export_design.md](./export_design.md) | `model_registry` / `loader` / `saver` / `ExportWrapper` 基类 + 三子类 / 统一图 I/O 契约 / 纯 embeddings 模式 / `epair_utils` / C++ `EpairModelLoader` / `GeWeightLoader` 懒加载 / `_assert_tensor_metadata` / GE 算子注册两层 / prefill-decode 分离 / 新增模型接入 / 精度差异分析 |
| [weight_format_design.md](./weight_format_design.md) | 权重 NZ 格式转换（Stage 1 设计阶段，代码未落地）：`weight_format/` 包（`nd2nz` / `resolve_format`）+ `weight_format_rules.json` 配置（op_type + format）+ `epair_utils` 集成 + C++ `MapGeLayoutString` 加 NZ case / Stage 1 fp16 路径 / Stage 2 量化 int4-NZ 研究门 / 后续 quant converter 迁 export |

## 代码文件

### Python

| 文件 | 职责 |
|------|------|
| `cann_lm/models/model_registry.py` | `ModelEntry` + `REGISTRY` + `detect_model_type` + `available_models` |
| `cann_lm/converter/loader.py` | `load_model` + `build_config` + `collect_tokenizer_files` |
| `cann_lm/converter/saver.py` | `save_epair` / `save_epair_prefill_decode` / `save_model_config` / `copy_tokenizer_files` |
| `cann_lm/models/export_wrapper.py` | `ExportWrapper` 基类（`forward` + `get_sample_inputs` + `_drive_layers`） |
| `cann_lm/models/{gemma4,qwen3,qwen3_5}/export_wrapper.py` | 各模型 `ExportWrapper` 子类 |
| `cann_lm/converter/export/epair_utils.py` | `pack_weights` + `consolidated_weights_save` |
| `cann_lm/converter/export/weight_format/`（规划中，未实现） | 权重格式转换包：`transforms.nd2nz`（ND→FractNZ）+ `rules.resolve_format`（配置解析，按 op_type 分流）+ `load_rules`（详见 [weight_format_design.md](./weight_format_design.md) Stage 1） |
| `cann_lm/converter/export/weight_format_rules.json`（规划中，未实现） | NZ 转换规则 shipped 默认配置（`name_pattern` + `op_type` + `format` + `pack`） |
| `cann_lm/converter/export/quant_swap/` | 量化物化内部包（2026-08-31 从 `quant/` 迁入）：`materialize.py`（int4 物化 + `transdata_scale`/`pack_int4`）/ `quant_linear.py`（QuantLinear）；`__init__` re-export 6 物化符号。编排层 `swap.py` 已升至 `cann_lm/converter/swap.py`（converter 根）。详见 [quant_design.md](../quant/quant_design.md) |
| `cann_lm/converter/export/ge_converters/ge_converter.py` | `register_ge_converters`（cann_lm.* + aten.* → GE 算子映射） |
| `cann_lm/converter/export/ge_converters/_quant_matmul_converter.py` | `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4` GE converter（详见 [INDEX.md](../quant/INDEX.md)） |

### C++（`cann_lm/converter/export/csrc/`）

| 文件 | 职责 |
|------|------|
| `inc/external/torch_delegate/epair_model_loader.h` | C++ 公共 API：`EpairModelLoader` |
| `backend/epair_model_loader.cc` | `EpairModelLoader::Impl` 实现（初始化、Graph 构建） |
| `backend/ge_graph_builder.h/.cc` | `GeGraphBuilder`：解析 model JSON + 构建 GE Graph |
| `backend/ge_weight_loader.h/.cc` | `GeWeightLoader`：解析 `weight_info.json`、懒加载 bin |
| `backend/epair_model_parser.h/.cc` | `EpairModelParser`：解析 `ep_json_with_ascend_ir.json` |

### 示例（`examples/test_converter/export_llm_model/`）

| 文件　　　　　　 | 职责　　　　　　　　　　　　　　　　 |
| ------------------| --------------------------------------|
| `main.cpp`　　　 | C++ 加载 epair + 编译 OM 的示例 main |
| `reg_register.h` | GE 图编译期算子注册（`REG_OP`）　　　|
| `om_npu_verify/` | OM 编译 + NPU 推理验证工具集　　　　 |

## 关键函数 / 类

| 符号 | 作用 |
|------|------|
| `load_model(hf_path, *, dtype, device) -> ModelArtifact` | 加载 HF 模型（自动探测类型 + config + weights + tokenizer） |
| `build_config(hf_config, entry)` | 统一字段过滤构造 config dataclass |
| `save_epair(artifact, output_dir, ...)` | 保存 epair（single 模式） |
| `save_epair_prefill_decode(artifact, output_dir, ...)` | 保存 prefill + decode 两图（共享权重） |
| `ExportWrapper` | 统一 I/O 契约基类（KV cache in/out + output_pos 选位） |
| `pack_weights(contents, *, max_file_size) -> ConsolidatedWeights` | 权重打包为多个 bin |
| `consolidated_weights_save(...)` | 写入磁盘（weights_*.bin + weight_info.json + ep_json） |
| `nd2nz(tensor, block=16, pad=True)`（规划中，未实现） | `[M,K]` ND 行主序 → FractNZ 物理块 `[ceil(M/b),ceil(K/b),b,b]`（见 [weight_format_design.md](./weight_format_design.md)） |
| `resolve_format(name, is_quantized)`（规划中，未实现） | 按 (name_pattern, op_type) 匹配 `weight_format_rules.json`，返回 format/block/pad/pack 规则 |
| `inline_weight_layout(ep_json_path, names_nz)`（规划中，未实现） | ep_json 后处理：命中张量 `layout` 字段 `ND`→`NZ`（仿 `inline_weight_scale_dtype`） |
| `td::EpairModelLoader` | C++ 加载 epair → 构建 GE Graph |
| `GeWeightLoader` | C++ 懒加载权重 bin |
| `register_ge_converters()` | 注册 cann_lm.* + aten.* → GE 算子映射 |

## 产物布局

```
output_dir/
├── ep_json_with_ascend_ir.json     # (single) 或 prefill_*.json + decode_*.json (prefill-decode)
├── weight_info.json                # 权重索引
├── weights_0.bin                   # 权重数据
├── model_config.json               # config + hf_config + use_embeddings 标记
├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
└── host_weights_*.bin + host_weights_info.json   # (仅 use_embeddings=True) host 查表权重
```

> 完整说明见 [export_design.md](./export_design.md) "输出文件布局"。

## 关联模块

- **CLI**：[INDEX.md](../cli/INDEX.md) — `cann_lm_converter.py` 入口 / `convert()` 编排
- **量化**：[INDEX.md](../quant/INDEX.md) — `QuantLinear` 替换 `nn.Linear` / `npu_quant_matmul` 算子
- **参考**：[reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md) / [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) / [reference_litert_lm_export.md](../reference_litert_lm_export.md) / [reference_npu_tuned_export_qwen3.md](../reference_npu_tuned_export_qwen3.md)
