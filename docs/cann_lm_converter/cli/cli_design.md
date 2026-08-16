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
- [运行环境支持（NPU / GPU / 纯 CPU）](#运行环境支持npu--gpu--纯-cpu)
- [文档关联](#文档关联)

---

## 概述

`cann-lm-converter.py` 是 converter 对外暴露的唯一入口，职责是编排三个阶段：

1. **加载**：`loader.load_model(hf_path)` → `ModelArtifact`（详见 [export_design.md](../export/export_design.md) "loader"）
2. **保存**：`saver.save_epair` / `save_epair_prefill_decode` → epair 目录（详见 [export_design.md](../export/export_design.md) "saver"）
3. **验证产物**：检查输出文件是否齐全

CLI 只做编排，不实现加载/保存逻辑（二者在 [export_design.md](../export/export_design.md)）。量化接入由 `--quant-path` / `--quant-method` / `--quant-bits` 触发，调用 [quant 模块](../quant/quant_design.md) 的 swap 函数（详见 [quant_design.md](../quant/quant_design.md)）。

## convert 程序化入口

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device: str = "cpu",
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = True,   # 默认纯 embeddings (图外查表); False 走图内查表
    mode: str = "prefill-decode",   # "single" | "prefill-decode"
) -> None:
    """程序化入口：HF 模型 → epair

    mode:
      - "single": 一张形状特化图 (seq_len 由参数指定)
      - "prefill-decode": prefill (seq_len 由参数指定) + decode (seq_len=1) 两图, 共享权重
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
            seq_len=seq_len,
            use_embeddings=use_embeddings,
        )
    else:
        save_epair(
            artifact, output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            seq_len=seq_len,
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
    parser.add_argument("--seq-len", type=int, default=128,
                        help="single 模式 = 图 seq_len; prefill-decode 模式 = prefill 图 seq_len (decode 固定 1)")
    parser.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Use external embeddings (no embed_tokens in graph). 默认 True; --no-use-embeddings 走图内查表")
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="single: 一张形状特化图; prefill-decode: prefill+decode 两图 + 共享权重 (默认)")
    # 量化参数: 见 quant_design.md (--quant-path / --quant-method / --quant-bits)
    return parser.parse_args()
```

| 参数　　　　　　　　 | 默认值　　　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------| ------------------| ---------------------------------------------------------------------------------------------------------------------------------------------|
| `--hf-path`　　　　　| (必填)　　　　　 | HuggingFace 模型本地目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--output-dir`　　　 | (必填)　　　　　 | epair 输出目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--dtype`　　　　　　| `float16`　　　　| `float32` / `float16` / `bfloat16`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--device`　　　　　 | `cpu`　　　　　　| 加载设备　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--max-file-size`　　| 50GB　　　　　　 | 单个 `weights_*.bin` 最大字节数　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--kv-cache-max-len` | 2048　　　　　　 | KV buffer 第一维　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--seq-len`　　　　　| 128　　　　　　　| single 模式 = 图 seq_len; prefill-decode 模式 = prefill 图 seq_len（decode 固定 1）　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--use-embeddings`　 | True　　　　　　 | 纯 embeddings（图外查表，默认；`--no-use-embeddings` 走图内查表，详见 [export_design.md](../export/export_design.md) "纯 embeddings 模式"） |
| `--mode`　　　　　　 | `prefill-decode` | `single` / `prefill-decode`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--quant-path`　　　 | `None`　　　　　 | dopt GPTQ 产物路径（场景 1，自动推断 group_size/embed；与 `--quant-method gptq` 互斥）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--quant-method`　 | `gptq`　 | 量化算法（当前仅 gptq；自动 spawn dopt，场景 2，待实现；与 `--quant-path` 互斥）　|
| `--quant-bits`　　 | `w4a8`　 | 格式/位宽（w4a8/w4a16，默认 w4a8；`--quant-method` 时生效）　|

> 量化参数 `--quant-path` / `--quant-method` / `--quant-bits` 互斥，详见 [quant_design.md](../quant/quant_design.md) "量化工具链分工设计"。精细调整走 dopt 分步（[dopt_design.md](../quant/dopt_design.md) "使用方法"）。

## 完整调用链

```
cann-lm-converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --device, --max-file-size,
    │                    --kv-cache-max-len, --seq-len, --use-embeddings,
    │                    --mode, --seq-len, [--quant-path | --quant-method gptq]
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
    │   swap_embeddings_from_dopt(sd, model)           # embed 自动检测 (embed_quant 非空时)

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
    --seq-len 128 \
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
    seq_len=128,           # prefill 图 seq_len (可配)
    use_embeddings=True,          # 默认 True (纯 embeddings); False 走图内查表
    save_weights=True,             # 是否写 weights_*.bin (False 只写 ep_json)
)
```

### 产物布局

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # prefill 图 (seq_len 由参数指定)
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

# 默认: 纯 embeddings (use_embeddings=True 默认, embed_tokens 查表外置到 host)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --seq-len 8

# 图内查表 (需显式 --no-use-embeddings 关闭纯 embeddings, 图内含 embed_tokens)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 \
    --seq-len 8 \
    --no-use-embeddings

# 数值验证: OM runner (ep 回环的超集, OM vs eager/HF)
python3 examples/export_llm_model/om_npu_verify/run_gemma4_om.py   # gemma4: OM vs eager, maxdiff<1.0
python3 examples/export_llm_model/om_npu_verify/run_qwen3_om.py     # qwen3: OM vs eager
python3 examples/export_llm_model/om_npu_verify/run_qwen3_5_om.py  # qwen3.5: OM vs EP/HF

# C++ OM 编译: 纯 embeddings epair → .om
./examples/export_llm_model/build/export_llm_model \
    output/ep_json_with_ascend_ir.json output/weight_info.json output

# prefill/decode 分离 (推理用, prefill+decode 两图 + 共享权重; 默认模式)
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/gemma-4-E2B \
    --output-dir /path/to/output \
    --mode prefill-decode \
    --seq-len 128 \
    --dtype bfloat16
```

> 纯 embeddings 设计（`detach_externalized_lookups` / host_weights / `model_config.json` 的 `use_embeddings` 字段）详见 [export_design.md](../export/export_design.md) "纯 embeddings 模式"。

## 运行环境支持（NPU / GPU / 纯 CPU）

converter 全流程三阶段：**加载**（torch CPU）→ **量化**（GPTQ 需计算设备 + swap CPU）→ **导出**（torchair IR 转换）。

### 环境矩阵

| 阶段 | NPU（Ascend910 + CANN + torch_npu） | GPU（CUDA + torch） | 纯 CPU |
|------|------|------|------|
| 加载 `load_model` | ✅ torch CPU | ✅ | ✅ |
| swap（`--quant-path` 消费 dopt 产物） | ✅ CPU | ✅ | ✅ |
| GPTQ（`--quant-method gptq`） | ✅ 方案 C subprocess | ⚠️ dopt `DEVICE=cuda` 适配待验证 | ❌ 需计算设备 |
| 导出 epair（torch.export + export_to_epair） | ✅ stubs torchair | ✅ stubs torchair（CPU） | ✅ stubs torchair |

### 当前各环境可用性

**NPU**（已验证）：
- `--quant-path`（消费产物 + swap + export stubs）✅ 全流程
- `--quant-method gptq` 方案 C（subprocess 系统 Python torch_npu + stubs export）✅ 全流程

**GPU**（未验证）：
- `--quant-path` ✅（swap + export stubs，CPU）
- `--quant-method gptq`：dopt GPTQ 需 GPU（torch CUDA）。**dopt 的 `DEVICE=cuda` 分支未验证**（dopt 有 `if DEVICE==npu` NPU 特定逻辑，GPU 走非 npu 分支，Hessian cholesky 等是否 GPU 兼容待验证）
- 导出 ✅（converter venv stubs torchair，CPU，不需 GPU）

**纯 CPU**：
- `--quant-path` ✅（swap + export stubs）
- `--quant-method gptq` ❌（GPTQ 需 NPU/GPU，纯 CPU 跑不了）
- 导出 ✅（stubs torchair）

### 还缺什么

**通用（`auto_quant.py`）**：
1. ✅ **DEVICE 检测**：`_detect_device_type` 自动检测 NPU→`npu` / GPU→`cuda` / 纯 CPU→报错
2. ✅ **纯 CPU 报错**：无 NPU/GPU → 提示用 `--quant-path`

**NPU**：方案 C 已可用（subprocess + stubs export），无缺失（方案 A 已放弃，torchair 带 CANN SDK 不完整）

**GPU**：
1. ⏳ **dopt GPU 适配验证**：dopt 的 `if DEVICE==npu` 分支（NPU 特定：Hessian 搬 CPU cholesky）。GPU（`DEVICE=cuda`）走非 npu 分支，验证 dopt GPU 跑 GPTQ

**纯 CPU**：✅ `--quant-method gptq` 报错检测已实现

### 优先级

| 缺什么 | 优先级 | 状态 |
|--------|--------|------|
| 纯 CPU `--quant-method gptq` 报错检测 | 高（防误用） | ✅ 完成 |
| auto_quant DEVICE 检测（NPU/GPU/CPU） | 高（三环境通用） | ✅ 完成 |
| GPU dopt 适配验证 | 中（GPU 支持） | ⏳ 待做 |

## 文档关联

| 文档　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 关联内容　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| -----------------------------------------------------------------------------------------| -------------------------------------------------------------------------------------------------------|
| [export_design.md](../export/export_design.md)　　　　　　　　　　　　　　　　　　　　　| `loader.load_model` / `saver.save_epair` / `ExportWrapper` / `model_registry` / 产物布局 / C++ 加载侧 |
| [quant_design.md](../quant/quant_design.md)　　　　　　　　　　　　　　　　　　　　　　 | `--quant-path` / `--quant-method` / `--quant-bits` 量化接入 / `swap_to_quant_linear_from_dopt`　　　　　　　　　　　　　　　　 |
| [converter_design.md](../converter_design.md)　　　　　　　　　　　　　　　　　　　　　 | 跨模块调用链总览 / 模型差异总览 / 精度汇总　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| [reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md)　　　　　　　 | gemma4 端到端 6 阶段流程示例（参考）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| [reference_python_to_om_self_test_flow.md](../reference_python_to_om_self_test_flow.md) | 改源码到 OM 出字的自验证流程（参考）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
