# CLI 模块设计文档

> 对外唯一入口 `cann-lm-converter.py`：把 HuggingFace 模型转换为 epair 格式。
>
> 模型类型从 `config.json` 的 `model_type` 字段自动探测，用户无需指定 `--model-type`。同时提供程序化入口 `convert()` 供脚本/测试复用。
>
> 最后更新: 2026-08-15

## 目录

- [概述](#概述)
- [convert 程序化入口](#convert-程序化入口)
- [命令行参数](#命令行参数)
- [完整调用链](#完整调用链)
- [prefill/decode 分离模式 CLI](#prefilldecode-分离模式-cli)
- [环境变量](#环境变量)
- [使用示例](#使用示例)
- [文档关联](#文档关联)

---

## 概述

`cann-lm-converter.py` 是 converter 对外暴露的唯一入口，职责是编排三个阶段：

1. **加载**：`loader.load_model(hf_path)` → `ModelArtifact`（详见 [export_design.md](../export/export_design.md) "loader"）
2. **保存**：`saver.save_epair` / `save_epair_prefill_decode` → epair 目录（详见 [export_design.md](../export/export_design.md) "saver"）
3. **验证产物**：检查输出文件是否齐全

CLI 只做编排，不实现加载/保存逻辑（二者在 [export_design.md](../export/export_design.md)）。量化接入由 `--quant-path` / `--quant-embeddings` 触发，调用 [quant 模块](../quant/quant_design.md) 的 swap 函数（详见 [quant_design.md](../quant/quant_design.md)）。

## convert 程序化入口

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device: str = "cpu",
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    sample_seq_len: int = 128,
    use_embeddings: bool = False,
    mode: str = "single",            # "single" | "prefill-decode"
    prefill_seq_len: int = 128,
) -> None:
    """程序化入口：HF 模型 → epair

    mode:
      - "single": 一张形状特化图 (seq_len=sample_seq_len)
      - "prefill-decode": prefill (seq_len=prefill_seq_len) + decode (seq_len=1) 两图, 共享权重
    """
    print(f"[1/3] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device=device)
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")
    print(f"      Wrapper: {artifact.wrapper_cls.__name__}")
    print(f"      Embedding mode: {'external' if use_embeddings else 'internal'}")
    print(f"      Export mode: {mode}")

    print(f"[2/3] Saving epair to {output_dir}")
    if mode == "prefill-decode":
        save_epair_prefill_decode(
            artifact, output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            prefill_seq_len=prefill_seq_len,
            use_embeddings=use_embeddings,
        )
    else:
        save_epair(
            artifact, output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            sample_seq_len=sample_seq_len,
            use_embeddings=use_embeddings,
        )

    print(f"[3/3] Verifying output")
    out = Path(output_dir)
    if mode == "prefill-decode":
        expected = ["prefill_ep_json_with_ascend_ir.json",
                    "decode_ep_json_with_ascend_ir.json",
                    "model_config.json", "weight_info.json"]
    else:
        expected = ["ep_json_with_ascend_ir.json",
                    "model_config.json", "weight_info.json"]
    for name in expected:
        fp = out / name
        print(f"      [{'OK' if fp.exists() else 'MISSING'}] {name}")
    print(f"      Weight files: {len(list(out.glob('weights_*.bin')))}")
    print(f"\nDone. Output: {output_dir}")
```

## 命令行参数

```python
def parse_args():
    parser = argparse.ArgumentParser(description="CANN-LM Converter: HF model → epair format")
    parser.add_argument("--hf-path", required=True, help="HuggingFace 模型路径 (本地目录)")
    parser.add_argument("--output-dir", required=True, help="epair 输出目录")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-file-size", type=int, default=50*1024*1024*1024)
    parser.add_argument("--kv-cache-max-len", type=int, default=2048)
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="single 模式的 seq_len")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use external embeddings (no embed_tokens in graph)")
    parser.add_argument("--mode", default="single", choices=["single", "prefill-decode"],
                        help="single: 一张形状特化图; prefill-decode: prefill+decode 两图 + 共享权重")
    parser.add_argument("--prefill-seq-len", type=int, default=128,
                        help="prefill-decode 模式下 prefill 图的 seq_len (decode 固定 1)")
    # 量化参数: 见 quant_design.md (--quant-path / --quant-embeddings)
    return parser.parse_args()
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hf-path` | (必填) | HuggingFace 模型本地目录 |
| `--output-dir` | (必填) | epair 输出目录 |
| `--dtype` | `float16` | `float32` / `float16` / `bfloat16` |
| `--device` | `cpu` | 加载设备 |
| `--max-file-size` | 50GB | 单个 `weights_*.bin` 最大字节数 |
| `--kv-cache-max-len` | 2048 | KV buffer 第一维 |
| `--sample-seq-len` | 128 | single 模式 seq_len |
| `--use-embeddings` | False | 纯 embeddings 模式（图外查表，详见 [export_design.md](../export/export_design.md) "纯 embeddings 模式"） |
| `--mode` | `single` | `single` / `prefill-decode` |
| `--prefill-seq-len` | 128 | prefill-decode 模式 prefill 图 seq_len（decode 固定 1） |

> 量化参数 `--quant-path` / `--quant-embeddings` 见 [quant_design.md](../quant/quant_design.md) "端到端流程"。

## 完整调用链

```
cann-lm-converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --device, --max-file-size,
    │                    --kv-cache-max-len, --sample-seq-len, --use-embeddings,
    │                    --mode, --prefill-seq-len, --quant-path*
    │
    ▼
convert(hf_path, output_dir, dtype, device, ...)
    │
    ▼
loader.load_model(hf_path, dtype, device)            # ① 加载（详见 export_design.md "loader"）
    │
    ├── 读 config.json
    ├── model_registry.detect_model_type(hf_config)
    │     → hf_config["model_type"] → get_entry → ModelEntry(config_cls, model_cls, ...)
    ├── build_config(hf_config, entry)
    ├── entry.model_cls(config).to(dtype).eval()
    ├── entry.load_weights_fn(model, hf_path, ...)   # safetensors → key 重映射 → param.copy_()
    └── collect_tokenizer_files(hf_path)

    ▼
(可选) 量化 swap                                       # ② 量化接入（详见 quant_design.md）
    │   swap_to_quant_linear_from_dopt(model, quant_path, ...)
    │   swap_embeddings_from_dopt(sd, model)           # --quant-embeddings 时

    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)  # ③ 保存（详见 export_design.md "saver"）
    │
    ├── register_ge_converters()
    ├── wrapper = artifact.wrapper_cls(model, ...).eval()
    ├── wrapper.get_sample_inputs(seq_len, use_embeddings)
    ├── (use_embeddings) _extract_host_side_weights + detach_externalized_lookups
    ├── torch.export.export(wrapper, sample, strict=False)
    ├── torch_delegate.export_to_epair(ep) → epair
    ├── del ep, wrapper, model; gc.collect()           # OOM 防护
    ├── consolidated_weights_save(epair, output_dir, ...)
    ├── copy_tokenizer_files(...)
    └── save_model_config(artifact, output_dir, use_embeddings)

    ▼
verify output                                          # ④ 检查产物文件齐全
```

## prefill/decode 分离模式 CLI

实际 LLM 推理需要 prefill（处理 prompt，seq_len 可变）+ decode（逐 token 生成，seq_len=1）两个阶段。`torch.export` 形状特化后单张图无法覆盖两个阶段，故 CLI 支持 `--mode prefill-decode` 导出两张图。

### CLI 用法

```bash
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/gemma-4-E2B \
    --output-dir /path/to/output \
    --mode prefill-decode \
    --prefill-seq-len 128 \
    --dtype bfloat16
```

### 接口（saver 侧，详见 export_design.md）

```python
from cann_lm.converter.saver import save_epair_prefill_decode

save_epair_prefill_decode(
    artifact, output_dir,
    *,
    max_file_size=50*1024**3,      # 单 bin 最大字节数
    kv_cache_max_len=2048,         # KV buffer 第一维
    prefill_seq_len=128,           # prefill 图 seq_len (可配)
    use_embeddings=False,          # True: 图外查表
    save_weights=True,             # 是否写 weights_*.bin (False 只写 ep_json)
)
```

### 产物布局

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # prefill 图 (seq_len=prefill_seq_len)
├── decode_ep_json_with_ascend_ir.json    # decode 图 (seq_len=1)
├── weight_info.json                       # 共享权重索引 (一份)
├── weights_0.bin                          # 共享权重 (一份)
├── model_config.json
└── tokenizer files
```

> 设计细节（prefill/decode 两图共享权重、`save_epair_prefill_decode` 内部流程、C++ 加载两图）见 [export_design.md](../export/export_design.md) "prefill/decode 分离导出"。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装路径 | (需设置) |
| `PYTHONPATH` | 项目根目录 | (需设置) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload 开关 (设为 0 禁用) | - |
| `TORCH_DELEGATE_HOME` | torch-delegate 安装路径 | - |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | fallback codegen 开关 (设为 0 禁用) | - |
| `LD_LIBRARY_PATH` | stubs + CANN lib64（converter CPU 环境需 stubs 在前） | - |

> 环境配置完整说明见 [AGENTS.md](../../../AGENTS.md) "运行环境设置（converter，已验证）"。

## 使用示例

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

# 模式 A: 图内查表 (默认, single 图)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --sample-seq-len 8

# 模式 B: 图外查表 (embedding 量化用, 图内不含 embed_tokens; input_ids 不入图)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --sample-seq-len 8 \
    --use-embeddings

# 模式 B 验证: ep 回环 (host 用 embed_tokens 查表喂 embeddings) vs in-graph wrapper vs HF
python examples/export_llm_model/llm_engine_on_cpu/debug_qwen3/verify_qwen3_epair.py \
    --use-embeddings --num-steps 10

# 模式 B C++ OM 编译: 纯 embeddings epair → .om (与模式 A 同一 main)
./examples/export_llm_model/build/export_llm_model \
    output/ep_json_with_ascend_ir.json output/weight_info.json output

# 模式 C: prefill/decode 分离 (推理用, prefill+decode 两图 + 共享权重)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/gemma-4-E2B \
    --output-dir /path/to/output \
    --mode prefill-decode \
    --prefill-seq-len 128 \
    --dtype bfloat16
```

> 模式 B 的纯 embeddings 设计（`detach_externalized_lookups` / host_weights / `model_config.json` 的 `use_embeddings` 字段）详见 [export_design.md](../export/export_design.md) "纯 embeddings 模式"。

## 文档关联

| 文档 | 关联内容 |
|------|---------|
| [export_design.md](../export/export_design.md) | `loader.load_model` / `saver.save_epair` / `ExportWrapper` / `model_registry` / 产物布局 / C++ 加载侧 |
| [quant_design.md](../quant/quant_design.md) | `--quant-path` / `--quant-embeddings` 量化接入 / `swap_to_quant_linear_from_dopt` |
| [converter_design.md](../converter_design.md) | 跨模块调用链总览 / 模型差异总览 / 精度汇总 |
| [reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md) | gemma4 端到端 6 阶段流程示例（参考） |
| [reference_python_to_om_self_test_flow.md](../reference_python_to_om_self_test_flow.md) | 改源码到 OM 出字的自验证流程（参考） |
