# CLI 模块索引

> `cann_lm_converter.py` — 对外唯一入口，模型类型从 `config.json` 自动探测。

## 模块文档

| 文档 | 内容 |
|------|------|
| [cli_design.md](./cli_design.md) | `convert()` 程序化入口 / 命令行参数 / 完整调用链 / prefill-decode CLI / 环境变量 / 使用示例 |

## 代码文件

| 文件 | 职责 |
|------|------|
| `cann_lm/converter/cann_lm_converter.py` | CLI 入口（`parse_args` + `convert`）+ 程序化 `convert()` 函数 |

## 关键函数

| 函数　　　　　　　　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------------------------------| ------------------------------------------------------------------------|
| `convert(hf_path, output_dir, dtype, ...)` | 程序化入口：HF 模型 → epair（编排 load_model → save_epair → 验证产物） |
| `parse_args()`　　　　　　　　　　　　　　 | 命令行参数解析　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

## 命令行参数速查

| 参数　　　　　　　　 | 默认　　　　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ----------------------| ------------------| --------------------------------------------------------------------------------|
| `--hf-path`　　　　　| (必填)　　　　　 | HF 模型本地目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--output-dir`　　　 | (必填)　　　　　 | epair 输出目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--dtype`　　　　　　| `float16`　　　　| `float32` / `float16` / `bfloat16`　　　　　　　　　　　　　　　　　　　　　　 |
| `--max-file-size`　　| 50GB　　　　　　 | 单 bin 最大字节数（默认 50GB）　　　　　　　　　　　　　　　　　　　　　　　 |
| `--kv-cache-max-len` | 2048　　　　　　 | KV buffer 第一维　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--seq-len`　　　　　| 128　　　　　　　| single=图 seq_len / prefill-decode=prefill 图 seq_len (decode=1)　　　　　　　 |
| `--use-embeddings`　 | True　　　　　　 | 纯 embeddings（默认）; --no-use-embeddings 走图内查表　　　　　　　　　　　　　|
| `--mode`　　　　　　 | `prefill-decode` | `single` / `prefill-decode`　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--num-layers`　　　 | —　　　　　　　　| 仅导出前 N 层（切片/调试，None=全部）　　　　　　　　　　　　　　　　　　　　 |
| `--quant-path`　　　 | —　　　　　　　　| dopt GPTQ 产物路径（量化接入，详见 [INDEX.md](../quant/INDEX.md)）　　　　　　 |
| `--quant-method`　　 | `gptq`　　　　　 | 量化算法（当前仅 gptq；自动 spawn dopt，场景 2，已实现；与 --quant-path 互斥）|
| `--quant-bits`　　　 | `w4a8`　　　　　 | 格式/位宽（w4a8，默认 w4a8；--quant-method 时生效）　　　　　　　　　　　　　　|
| `--quant-device`　　 | —　　　　　　　　| 量化子进程设备（npu/cuda/cpu），跳过环境探测　　　　　　　　　　　　　　　　　|

> 完整参数说明见 [cli_design.md](./cli_design.md) "命令行参数"。

## 调用链

```
parse_args → convert → loader.load_model (export_index) → [可选 quant swap (quant_index)]
           → saver.save_epair / save_epair_prefill_decode (export_index) → verify output
```

## 关联模块

- **导出**：[INDEX.md](../export/INDEX.md) — `load_model` / `save_epair` / `ExportWrapper` / 产物布局
- **量化**：[INDEX.md](../quant/INDEX.md) — `--quant-path` / `--quant-method` / `--quant-bits` 接入
