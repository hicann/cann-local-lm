# CLI 模块设计文档

> 对外唯一入口 `cann_lm_converter.py`：把 HuggingFace 模型转换为 epair 格式。
>
> 模型类型从 `config.json` 的 `model_type` 字段自动探测，用户无需指定 `--model-type`。同时提供程序化入口 `convert()` 供脚本/测试复用。
>
> 状态: 当前采用 | 最后核对: 2026-08-25 | ADR: [ADR-0001](../../adr/0001-load-model-cpu-quant-subprocess-device.md)（load_model CPU vs quant subprocess 设备）

## 目录

- [概述](#概述)
- [convert 程序化入口](#convert-程序化入口)
- [命令行参数](#命令行参数)
- [完整调用链](#完整调用链)
- [prefill/decode 分离模式 CLI](#prefilldecode-分离模式-cli)
- [环境变量](#环境变量)
- [使用示例](#使用示例)
- [运行环境支持（NPU / GPU / 纯 CPU）](#运行环境支持npu--gpu--纯-cpu)
- [converter 与 torch_npu 的关系（设计决策：不依赖真实 torch_npu）](#converter-与-torch_npu-的关系设计决策不依赖真实-torch_npu)
- [文档关联](#文档关联)

---

## 概述

`cann_lm_converter.py` 是 converter 对外暴露的唯一入口，职责是编排三个阶段：

1. **加载**：`loader.load_model(hf_path)` → `ModelArtifact`（详见 [export_design.md](../export/export_design.md) "loader"）
2. **保存**：`saver.save_epair` / `save_epair_prefill_decode` → epair 目录（详见 [export_design.md](../export/export_design.md) "saver"）
3. **验证产物**：检查输出文件是否齐全

CLI 只做编排，不实现加载/保存逻辑（二者在 [export_design.md](../export/export_design.md)）。量化接入由 `--quant-path` / `--quant-method` 触发，调用 [quant 模块](../quant/quant_design.md) 的 swap 函数（详见 [quant_design.md](../quant/quant_design.md)）。

## convert 程序化入口

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = True,   # 默认纯 embeddings (图外查表); False 走图内查表
    mode: str = "prefill-decode",   # "single" | "prefill-decode"
    num_layers: int | None = None,  # 截层调试; None = 全部
    quant_path: str | None = None,  # 场景 1: dopt GPTQ 产物; 与 quant_method 互斥
    quant_method: str | None = None,  # 场景 2: "gptq" 自动 spawn dopt; 与 quant_path 互斥
) -> None:
    """程序化入口：HF 模型 → epair

    load_model 写死 CPU (融合算子 cann_lm/models/fused_ops/ 只在 CPU 实现),
    不接受 device 参数. NPU 跑 GPTQ 通过 quant_method + subprocess 隔离.

    mode:
      - "single": 一张形状特化图 (seq_len 由参数指定)
      - "prefill-decode": prefill (seq_len 由参数指定) + decode (seq_len=1) 两图, 共享权重
    """
    # 场景 2 (--quant-method gptq): 先 spawn dopt subprocess 跑 GPTQ (主进程不预 load, 避免 2x 内存)
    # 场景 1 (--quant-path): dopt 已离线跑完, 直接消费产物
    # 无量化: dopt_out_path = None
    dopt_out_path = None
    if quant_method == "gptq":
        from cann_lm.converter.quant_entry import run_auto_quant
        dopt_out_path = run_auto_quant(hf_path)
    elif quant_path:
        dopt_out_path = quant_path

    print(f"[1/2] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device="cpu")  # 场景 2 时在 subprocess 之后
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")
    print(f"      Wrapper: {artifact.wrapper_cls.__name__}")
    print(f"      Embedding mode: {'external' if use_embeddings else 'internal'}")
    print(f"      Export mode: {mode}")

    if num_layers is not None:
        artifact.model.model.layers = artifact.model.model.layers[:num_layers]
        print(f"      Truncated to first {num_layers} layers")

    # 量化 swap: 场景 1 + 场景 2 共用统一路径 (一次 load → Linear swap + Embedding 抽取)
    embed_quant = None
    if dopt_out_path:
        from cann_lm.converter.quant_entry import swap_from_dopt
        info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (group_size={info['group_size']})")

    print(f"[2/2] Saving epair to {output_dir}")
    if mode == "prefill-decode":
        save_epair_prefill_decode(
            artifact, output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            seq_len=seq_len,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
        )
    else:
        save_epair(
            artifact, output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            seq_len=seq_len,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
        )

    print(f"\nDone. Output: {output_dir}")
```

## 命令行参数

```python
def parse_args():
    parser = argparse.ArgumentParser(description="CANN-LM Converter: HF model → epair format")
    parser.add_argument("--hf-path", required=True, help="HuggingFace 模型路径 (本地目录)")
    parser.add_argument("--output-dir", required=True, help="epair 输出目录")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-file-size", type=int, default=50*1024*1024*1024)
    parser.add_argument("--kv-cache-max-len", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128,
                        help="single 模式 = 图 seq_len; prefill-decode 模式 = prefill 图 seq_len (decode 固定 1)")
    parser.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Use external embeddings (no embed_tokens in graph). 默认 True; --no-use-embeddings 走图内查表")
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="single: 一张形状特化图; prefill-decode: prefill+decode 两图 + 共享权重 (默认)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="只导出前 N 层 (截层调试). None = 全部. 截 model.model.layers, config.num_hidden_layers 不变")
    # 量化: --quant-path (场景 1) 与 --quant-method (场景 2) 互斥
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="dopt GPTQ 产物路径 (场景 1). 与 --quant-method 互斥")
    quant_group.add_argument("--quant-method", default=None, choices=["gptq"],
                             help="量化算法 (场景 2, 自动 spawn dopt, NPU/GPU/CPU). 与 --quant-path 互斥")
    return parser.parse_args()
```

| 参数　　　　　　　　 | 默认值　　　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------| ------------------| ---------------------------------------------------------------------------------------------------------------------------------------------|
| `--hf-path`　　　　　| (必填)　　　　　 | HuggingFace 模型本地目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--output-dir`　　　 | (必填)　　　　　 | epair 输出目录　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--dtype`　　　　　　| `float16`　　　　| `float32` / `float16` / `bfloat16`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--max-file-size`　　| 50GB　　　　　　 | 单个 `weights_*.bin` 最大字节数　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--kv-cache-max-len` | 2048　　　　　　 | KV buffer 第一维　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `--seq-len`　　　　　| 128　　　　　　　| single 模式 = 图 seq_len; prefill-decode 模式 = prefill 图 seq_len（decode 固定 1）　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--use-embeddings`　 | True　　　　　　 | 纯 embeddings（图外查表，默认；`--no-use-embeddings` 走图内查表，详见 [export_design.md](../export/export_design.md) "纯 embeddings 模式"） |
| `--mode`　　　　　　 | `prefill-decode` | `single` / `prefill-decode`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--num-layers`　　　 | `None`　　　　　 | 只导出前 N 层（截层调试）；`None` = 全部；截 `model.model.layers`，`config.num_hidden_layers` 不变 |
| `--quant-path`　　　 | `None`　　　　　 | dopt GPTQ 产物路径（场景 1，自动推断 group_size/embed；与 `--quant-method gptq` 互斥）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `--quant-method`　 | `gptq`　 | 量化算法（当前仅 gptq；自动 spawn dopt subprocess 跑 GPTQ，场景 2；与 `--quant-path` 互斥）　|

> **load_model 写死 CPU**（融合算子只在 CPU 实现），故**不暴露 `--device` CLI 参数**。GPTQ 跑在 NPU/GPU/CPU subprocess（dopt_lm 支持 CPU float32），通过 `--quant-method gptq` + subprocess 隔离（subprocess 设备由 `_detect_device_type` 自动检测）。

> **量化两流程**：`--quant-path <dopt产物>`（场景 1，手动先跑 dopt GPTQ 产物到 `--output-dir` 目录，converter 消费）或 `--quant-method gptq`（场景 2，converter 自动 spawn dopt subprocess 在 NPU/GPU/CPU 跑 GPTQ → swap → export，一键式）。互斥。精细调整走 dopt 分步（[dopt_design.md](../quant/dopt_design.md)）。详见 [quant_design.md](../quant/quant_design.md)。

## 完整调用链

> **改造(2026-08-18)**:场景 2 (`--quant-method gptq`) 时 `load_model` 推到 `run_auto_quant` 之后,避免主进程 cann_lm 模型与 subprocess transformers 模型同时持有(2× 内存峰值)。策略赋值逻辑在 subprocess 内用 transformers 模型生成(`dopt_wrapper.build_layer_strategy`),swap 移回主进程与场景 1 共用统一路径。**当前 subprocess 入口为独立脚本 `quant/dopt_wrapper.py`**(`python dopt_wrapper.py ...`,系统 Python + torch_npu + NPU env)。

```
cann_lm_converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --max-file-size,
    │                    --kv-cache-max-len, --seq-len, --use-embeddings,
    │                    --mode, --seq-len, [--quant-path | --quant-method gptq]
    │
    ▼
convert(hf_path, output_dir, dtype, ...)   # device 写死 CPU, 不暴露 CLI
    │
    ▼
(可选) 场景 2 量化前置: spawn dopt subprocess      # ① GPTQ (详见 quant_design.md "方案 C")
    │   场景2 (--quant-method gptq, 主进程不预 load cann_lm):
     │     auto_quant.run_auto_quant(hf_path) → 返回 dopt_out_dir
     │       └─ spawn 'python dopt_wrapper.py --model-path <hf> --optimize-config config.yaml --output-dir <临时盘> --hf-type-save' (系统 python + torch_npu + NPU)
     │            内部 dopt_wrapper.main() 运行 (parse_args 从命令行读 --model-path / --optimize-config / --output-dir / --hf-type-save)
    │            ├─ model_define(hf_path) 加载 transformers 模型
     │            ├─ build_layer_strategy(model) → dopt_config.json (带策略)
    │            │   Linear（含 lm_head）→Quant_INT8-4_ECO,
    │            │   Embedding→Quant_Embed_MinMax
    │            ├─ del model 释放
    │            └─ weight_quant_pipline(...) (dopt 内部加载 2 次,不改 dopt 的代价)
    │               → 产物 → 临时盘 model.safetensors + dopt_config.json
    │   场景1 (--quant-path): dopt_out_path = quant_path (离线产物, 无 subprocess)
    │   无量化: dopt_out_path = None
    │
    ▼
loader.load_model(hf_path, dtype, device)            # ② 加载 cann_lm 模型 (场景 2 时在 subprocess 之后)
    │
    ├── 读 config.json
    ├── model_registry.detect_model_type(hf_config)
    │     → hf_config["model_type"] → get_entry → ModelEntry(config_cls, model_cls, ...)
    ├── build_config(hf_config, entry)
    ├── entry.model_cls(config).to(dtype).eval()
    ├── entry.load_weights_fn(model, hf_path, ...)   # safetensors → key 重映射 → param.copy_()
    └── collect_tokenizer_files(hf_path)

    ▼
(可选) 量化接入 (场景 1 + 场景 2 共用统一 swap)        # ③ 量化 swap (详见 quant_design.md)
    │   if dopt_out_path (场景1 quant_path 或 场景2 临时盘):
    │     info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path)  # 一次 load → (quant_info, embed_quant)
    │       ├─ _swap_linear_from_sd → QuantLinear 替换
    │       └─ swap_embeddings_from_dopt → embed 自动检测

    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)  # ④ 保存 (详见 export_design.md "saver")
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
```

## prefill/decode 分离模式 CLI

实际 LLM 推理需要 prefill（处理 prompt，seq_len 可变）+ decode（逐 token 生成，seq_len=1）两个阶段。`torch.export` 形状特化后单张图无法覆盖两个阶段，故 CLI 支持 `--mode prefill-decode` 导出两张图。

### CLI 用法

```bash
python cann_lm/converter/cann_lm_converter.py \
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
    use_embeddings=True,          # saver 默认 True (纯 embeddings); CLI 默认 True 传入; False 走图内查表
    save_weights=True,             # 是否写 weights_*.bin (False 只写 ep_json)
    whitelist_decompositions=None, # 限定 decomposition 白名单 (None = 全跑)
    embed_quant=None,              # 量化 embed int8+scale (swap_from_dopt 返回; None 无量化 embed)
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

> **各 env 直接 source 具名脚本**：`scripts/set_env_<env>.sh`（export/compile/generate/npu 四套，互不叠加；公共部分 `_env_common.sh`）。路径默认值可由环境变量覆盖(`TORCH_DELEGATE_HOME` / `TORCHAIR_ROOT` / `DOPT_PYTHON` / `PROJECT_ROOT` / `CANN_SET_ENV` / `SOC_VERSION` / `GE_FUSION_SWITCH_FILE`)。下面表格列变量与默认值。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装路径 (由 CANN `set_env.sh` 设置) | (需 source CANN set_env.sh) |
| `PYTHONPATH` | 项目根目录 (`$PROJECT_ROOT`) | (需设置) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload 开关 (设为 0 禁用) | `0` |
| `TORCH_DELEGATE_HOME` | torch-delegate 安装路径 | `$TORCH_DELEGATE_HOME` |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | fallback codegen 开关 (设为 0 禁用) | `0` (converter 模式) |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | GE 详细日志 (定位 not registered 等) | `1` (converter 模式) |
| `LD_LIBRARY_PATH` | converter 模式: stubs 在 CANN lib64 前;NPU/OM 模式: 仅 CANN lib64 | (按模式) |
| `DOPT_PYTHON` | dopt subprocess 用的系统 Python (`--quant-method gptq` 触发) | _(自动探测；`_env_common.sh` 找能 `import torch_npu` 的 python，换机器无需改)_ |
| `CONVERTER_PYTHON` | converter venv python (export 设置后供开发者用) | `${TORCH_DELEGATE_HOME}/.venv/bin/python` |
| `SOC_VERSION` | OM 编译目标 SOC (om 模式) | `Ascend910_9362` |
| `GE_FUSION_SWITCH_FILE` | OM 编译 fusion switch 配置 (关 AUB fusion) | `/tmp/fusion_autoub_off.cfg` |

> 环境配置完整说明见 [AGENTS.md](../../../AGENTS.md) "运行环境设置（converter，已验证）"。

## 使用示例

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=$TORCH_DELEGATE_HOME
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

# 默认: 纯 embeddings (use_embeddings=True 默认, embed_tokens 查表外置到 host)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3.5-0.8B \
    --output-dir /path/to/output \
    --dtype float16 \
    --seq-len 8

# 图内查表 (需显式 --no-use-embeddings 关闭纯 embeddings, 图内含 embed_tokens)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3.5-0.8B \
    --output-dir /path/to/output \
    --dtype float16 \
    --seq-len 8 \
    --no-use-embeddings

# 数值验证: OM runner (ep 回环的超集, OM vs eager/HF)
python3 examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_om.py   # gemma4: OM vs eager, maxdiff<1.0
python3 examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_om.py  # qwen3.5: OM vs EP/HF

# C++ OM 编译: 纯 embeddings epair → .om
./examples/test_converter/export_llm_model/build/cann_lm_compile \
    output/ep_json_with_ascend_ir.json output/weight_info.json output

# prefill/decode 分离 (推理用, prefill+decode 两图 + 共享权重; 默认模式)
python cann_lm/converter/cann_lm_converter.py \
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

| 阶段　　　　　　　　　　　　　　　　　　　　 | NPU（Ascend910 + CANN + torch_npu） | GPU（CUDA + torch）　　　　　　 | 纯 CPU　　　　　　　　　　　　　　　　　　　　　　 |
| ----------------------------------------------| -------------------------------------| ---------------------------------| ----------------------------------------------------|
| 加载 `load_model`　　　　　　　　　　　　　　| ✅ torch CPU　　　　　　　　　　　　 | ✅　　　　　　　　　　　　　　　 | ✅　　　　　　　　　　　　　　　　　　　　　　　　　|
| swap（`--quant-path` 消费 dopt 产物）　　　　| ✅ CPU　　　　　　　　　　　　　　　 | ✅　　　　　　　　　　　　　　　 | ✅　　　　　　　　　　　　　　　　　　　　　　　　　|
| GPTQ（`--quant-method gptq`）　　　　　　　　| ✅ 方案 C subprocess　　　　　　　　 | ⚠️ dopt `DEVICE=cuda` 适配待验证 | ✅ 回落 `cpu`（dopt_lm 支持 CPU float32，7B+ 极慢） |
| 导出 epair（torch.export + export_to_epair） | ✅ stubs torchair　　　　　　　　　　| ✅ stubs torchair（CPU）　　　　 | ✅ stubs torchair　　　　　　　　　　　　　　　　　 |

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
- `--quant-method gptq` ✅（**回落 `cpu`**，dopt_lm `SUPPORTED_DEVICE_TYPES` 含 `cpu`，走 `dispatch_dtype=float32`）。**代价**：7B+ Hessian 极慢，仅小模型/调试适用；生产建议 NPU/GPU 或 `--quant-path`
- 导出 ✅（stubs torchair）

### 还缺什么

**通用（`auto_quant.py`）**：
1. ✅ **DEVICE 检测**：`_detect_device_type` 自动检测 NPU→`npu` / GPU→`cuda` / 无加速器→**回落 `cpu`**（不再报错；dopt_lm 支持 CPU float32）
2. ✅ **CPU 回落**：无 NPU/GPU → `_detect_device_type` 自动回落 `cpu`，warn 7B+ 极慢

**NPU**：方案 C 已可用（subprocess + stubs export），无缺失（方案 A 已放弃，torchair 带 CANN SDK 不完整）

**GPU**：
1. ⏳ **dopt GPU 适配验证**：dopt 的 `if DEVICE==npu` 分支（NPU 特定：Hessian 搬 CPU cholesky）。GPU（`DEVICE=cuda`）走非 npu 分支，验证 dopt GPU 跑 GPTQ

**纯 CPU**：✅ `--quant-method gptq` 回落 `cpu` 已实现（warn 慢）

### 优先级

| 缺什么 | 优先级 | 状态 |
|--------|--------|------|
| auto_quant DEVICE 检测（NPU/GPU/CPU 回落） | 高（三环境通用） | ✅ 完成 |
| CPU GPTQ 回落 + warn | 高（CPU 可用） | ✅ 完成 |
| GPU dopt 适配验证 | 中（GPU 支持） | ⏳ 待做 |

## converter 与 torch_npu 的关系（设计决策：不依赖真实 torch_npu）

> converter（① 导出 epair）**不依赖真实 torch_npu python 包**。走 stub 路径：converter venv 不装 `torch_npu`，`cann_lm/models/fused_ops/_library.py` 自定义 `npu::` 算子的 schema+fake-meta（含 `npu_chunk_gated_delta_rule`/`npu_dynamic_quant`/`npu_quant_matmul` 等），torch.export 能 trace；`cann_lm::` 算子本就由 `_library.py` 自定义。lib 层 stub（`.so` 在 `LD_LIBRARY_PATH`）给 torchair GE 引擎提供 C 符号。

### 为什么不依赖真实 torch_npu（背景与踩坑）

converter venv（CPU-only）组件：`torch 2.9.0+cpu` + `torchair`（源码编译 `NO_ASCEND_SDK=1`，CPU+stubs 可用）+ lib stubs（`/mnt/workspace/torchair/build/stubs` 的 `libtorch_npu.so` 等，给 torchair GE 引擎提供 C 符号，避免真 NPU `aclrtSetDevice→107001`）。

`_library.py` 的顶层 `try: import torch_npu`：导入成功→真实路径（`npu::` 由 torch_npu 提供）；`ImportError`→stub 路径（`_library.py` 自定义 `npu::` schema+fake）。两条路都能让 torch.export trace `npu::` 算子。

若装真实 torch_npu，会撞三个耦合问题：

1. **`npu_chunk_gated_delta_rule` 的 python 版本**：CANN kernel 8.5.0+ 支持（本机 9.2.0 满足），但 torch_npu **python schema/fake-meta 从 ≥2.7.1.post8 起才有**；2.7.1.post4 / 2.9.0 / 2.9.1 等非 post 版**无此算子**（`torch.ops.npu.npu_chunk_gated_delta_rule` AttributeError）。2.9.0.post6（匹配 torch 2.9.0、不升 torch）才有。
2. **`_LazyTorchair` 代理覆盖源码 torchair（核心坑）**：`torch_npu/dynamo/__init__.py:113`（外部依赖，torch_npu 2.9.0）在 `import torch_npu` 时**无条件**装 `sys.modules['torchair'] = _LazyTorchair()`，把 `import torchair` 劫持到 torch_npu 自带的 bundled wrapper（`torch_npu/dynamo/torchair/`），**忽略** venv 里的源码 torchair。而 wrapper 的 `_ge_concrete_graph/fx2ge_converter.py:1336`（外部依赖，torchair 源码）的 `parse_input` 有 `AssertionError("meta_outputs must be instance of torch.Tensor")`（在 `input_ids` placeholder 上失败），**源码 torchair 已修该处**。即"装真 torch_npu → 代理屏蔽源码 torchair（已修）→ 用 wrapper（未修）→ GE 转换挂"。
3. **`_get_deterministic_level` API**：源码 torchair 的 `_normalize_ge_option`（`fx2ge_converter.py:1901`，外部依赖，torchair 源码）调 `sys.modules['torch_npu'].npu._get_deterministic_level()`；torch_npu 2.9.0 无此 API（2.9.0.post6/2.9.1 有）。源码 torchair 有 `if 'torch_npu' not in sys.modules: → "0"` 兜底，但只要 torch_npu 在 sys.modules 就走 else 必调。

### 关键结论：bug 只在 ① 导出 epair 触发

上述 bug（wrapper 的 `fx2ge_converter.parse_input` AssertionError）只在 **FX 图 → GE IR 转换**路径触发，即 ① `torch_delegate.export_to_epair → torchair._optimize_and_convert → npu_fx_compiler`。其他阶段不跑 FX→GE 转换，不撞：

| 阶段 | `import torch_npu`? | 跑 FX→GE 转换? | 撞 bug? |
|------|---|---|---|
| ① converter 导出 epair | ✓ | ✓（`export_to_epair`） | ✓（`input_ids` placeholder 触发 `parse_input`） |
| Step A dopt GPTQ | ✓（系统 python，真 NPU） | ✗（transformers eager 跑 GPTQ） | ✗ |
| ③ OM 编译 | ✗（C++ 二进制，无 python） | ✗（读 epair 里已转好的 `ascend_ir`） | ✗ |
| ④ NPU 出字 | ✓（系统 python，acl 跑 OM） | ✗（`acl.mdl.execute`；CPU 端 load 模型只取 mask/kv 是 eager） | ✗ |

代理在 `import torch_npu` 时就装，但代理本身不报错；只有真正调 torchair 做 GE 转换（`fx2ge_converter` 处理 placeholder）时，wrapper 的 bug 才暴露。dopt / 出字 / OM 编译虽也 import 了 torch_npu（代理也装），但不跑 FX→GE 转换 → `fx2ge_converter.parse_input` 不被调 → 不撞 bug。

### 设计决策：converter 走 stub 路径（不装 torch_npu）

converter 只需要 `npu::` 算子的 **fake-meta**（torch.export trace 用），`_library.py` 能自提供，**不需要真实 torch_npu 的算子**。故：

- converter venv **不装 `torch_npu` python 包** → `import torch_npu` → `ImportError` → `_HAS_TORCH_NPU=False`（stub 路径）→ `_library.py` 自定义 `npu::` schema+fake（含 `npu_chunk_gated_delta_rule`/`npu_dynamic_quant`/`npu_quant_matmul` 等）。
- 无 torch_npu → `_LazyTorchair` 代理不安装 → **源码 torchair 直接用**（GE 转换能过的版本）；`_normalize_ge_option` 走 `torch_npu not in sys.modules → "0"` 兜底，不调 `_get_deterministic_level`。
- lib 层 stub 保留（`LD_LIBRARY_PATH`），给 torchair GE 引擎提供 C 符号。

好处：绕开 P1（torch_npu 版本依赖）/P2（代理覆盖）/P3（API 缺失）三个问题；无 torch_npu 版本漂移；无需重命名 wrapper。**converter 的正确性不依赖任何特定 torch_npu 版本**。

副作用：converter 导出产出有效产物后进程会在 GE finalize/stub 收尾**挂起不退出**（产物在挂起前已落盘）→ 用"后台跑 + 轮询产物 + kill"绕过。

### 替代方案（不采用，仅记录）

- **真 torch_npu 2.9.0.post6 + 重命名 `dynamo/torchair`**：装 torch_npu 2.9.0.post6（含 `npu_chunk_gated_delta_rule`，真实路径，torch_npu 提供算子 fake）+ 重命名 venv 的 `torch_npu/dynamo/torchair` → `torchair.disabled`（让 `dynamo/__init__.py` 的 `if not exists(...)` 命中（外部依赖，torch_npu 2.9.0.post6）→ 不装代理 → 源码 torchair 直接用）。能跑（用官方算子 + 源码 torchair），但依赖 torch_npu 版本（须 ≥2.7.1.post8 且匹配 torch）+ 需一次性重命名 wrapper，复杂度高。设计上不采用——converter 不需要真实 torch_npu 算子（fake-meta 由 `_library.py` 自提供即可）。



| 文档　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 关联内容　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| -----------------------------------------------------------------------------------------| -------------------------------------------------------------------------------------------------------|
| [export_design.md](../export/export_design.md)　　　　　　　　　　　　　　　　　　　　　| `loader.load_model` / `saver.save_epair` / `ExportWrapper` / `model_registry` / 产物布局 / C++ 加载侧 |
| [quant_design.md](../quant/quant_design.md)　　　　　　　　　　　　　　　　　　　　　　 | `--quant-path` / `--quant-method` 量化接入 / `swap_from_dopt`（主路径）　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| [converter_design.md](../converter_design.md)　　　　　　　　　　　　　　　　　　　　　 | 跨模块调用链总览 / 模型差异总览 / 精度汇总　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| [reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md)　　　　　　　 | gemma4 端到端 6 阶段流程示例（参考）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| [reference_python_to_om_self_test_flow.md](../reference_python_to_om_self_test_flow.md) | 改源码到 OM 出字的自验证流程（参考）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
