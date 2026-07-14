# CANN-LM Converter 设计文档

> 将 HuggingFace 模型转换为 epair 格式的转换工具
>
> 最后更新: 2026-07-14

## 能力边界

`cann_lm/converter/` 是一个 **CLI 工具**，能力如下：

- **输入**：HuggingFace 模型目录（`config.json` + safetensors 权重 + tokenizer 文件）
- **输出**：epair 模型目录（`ep_json_with_ascend_ir.json` + 权重 bin + tokenizer + `model_config.json`），供 C++ 后端加载
- **量化**：作为内置扩展能力预留，具体方案 TBD（`quant/` 目录预留）

对外只暴露一个入口 `cann-lm-converter.py`，模型类型从 HF `config.json` 自动探测，用户无需指定 `--model-type`。

## 目录结构

```
cann_lm/converter/
├── __init__.py
├── cann-lm-converter.py          # CLI 入口 (pipeline) + 程序化 convert() 函数
├── loader.py                     # 模型加载 (config + weights + tokenizer, 自动探测模型类型)
├── saver.py                      # epair 保存 (含 tokenizer/config 复制)
├── model_registry.py             # 模型注册表 + detect_model_type
├── export/
│   └── epair_utils.py            # 已有: pack_weights, consolidated_weights_save
└── quant/                        # 预留扩展能力 (量化, TBD)
```

## model_registry.py — 模型注册表

```python
@dataclass
class ModelEntry:
    """一个模型在注册表中的条目"""
    model_type: str               # "gemma4_text" / "qwen3_5_text"
    config_cls: type              # Gemma4TextConfig / Qwen3_5TextConfig
    model_cls: type               # Gemma4ForCausalLM / Qwen3_5ForCausalLM
    load_weights_fn: callable     # load_gemma4_weights / load_qwen3_5_weights
    config_json_key: str          # config.json 中 text 配置的 key (如 "text_config")


REGISTRY: dict[str, ModelEntry] = {
    "gemma4": ModelEntry(
        model_type="gemma4_text",
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",
    ),
    "qwen3": ModelEntry(
        model_type="qwen3",
        config_cls=Qwen3TextConfig,
        model_cls=Qwen3ForCausalLM,
        load_weights_fn=load_qwen3_weights,
        config_json_key="text_config",
    ),
    "qwen3_5": ModelEntry(
        model_type="qwen3_5_text",
        config_cls=Qwen3_5TextConfig,
        model_cls=Qwen3_5ForCausalLM,
        load_weights_fn=load_qwen3_5_weights,
        config_json_key="text_config",
    ),
}

def get_entry(name: str) -> ModelEntry:
    """按名称查找模型条目，支持别名"""
    aliases = {
        "gemma4": "gemma4", "gemma-4": "gemma4",
        "qwen3": "qwen3", "qwen-3": "qwen3",
        "qwen3.5": "qwen3_5", "qwen3_5": "qwen3_5",
    }
    key = aliases.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(REGISTRY)}")
    return REGISTRY[key]


def detect_model_type(hf_config: dict) -> ModelEntry:
    """从 HF config.json 自动探测模型类型。

    读 config.json 顶层 `model_type` 字段（如 "gemma4" / "qwen3" / "qwen3_5"），
    走 get_entry 的 alias 表映射到 registry key，返回对应 ModelEntry。用户无需
    在 CLI 指定 --model-type。未注册则报错并列出已支持模型。
    """
    model_type = hf_config.get("model_type")
    if not model_type:
        raise ValueError(
            f"config.json 缺少 'model_type' 字段，无法自动探测；已支持: {list(REGISTRY)}"
        )
    return get_entry(model_type)
```

## loader.py — 模型加载

### ModelArtifact 数据结构

```python
@dataclass
class ModelArtifact:
    """加载产物"""
    model: nn.Module              # 自定义模型实例 (权重已加载, eval 模式)
    config: object                # 自定义 config dataclass 实例
    hf_path: str                  # 原始 HF 模型路径
    hf_config: dict               # 原始 config.json 原始字典 (保留全部字段)
    tokenizer_files: list[Path]   # tokenizer 相关文件路径列表
    model_type: str               # 模型类型标识
```

### load_model 接口

```python
def load_model(
    hf_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ModelArtifact:
    """
    从 HF 路径加载模型 — 使用自定义模型定义，模型类型自动探测

    流程:
      1. 读取 HF config.json
      2. 从 config.json 的 model_type 字段自动探测模型类型 (detect_model_type)
      3. 构造自定义 config dataclass
      4. 实例化自定义模型
      5. 加载 HF safetensors 权重 (调用各模型自带的 load_*_weights)
      6. 收集 tokenizer 文件列表
    """
    # 1. 读取原始 config.json
    config_json_path = os.path.join(hf_path, "config.json")
    with open(config_json_path) as f:
        hf_config = json.load(f)

    # 2. 自动探测模型类型
    entry = detect_model_type(hf_config)

    # 3. 构造自定义 config
    config = build_config(hf_config, entry)

    # 4. 实例化模型
    model = entry.model_cls(config)
    model = model.to(dtype=dtype).to(device=device)
    model.eval()

    # 5. 加载权重 — 调用各模型自带的 load_*_weights
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 6. 收集 tokenizer 文件 (不加载, 只记录路径)
    tokenizer_files = collect_tokenizer_files(hf_path)

    return ModelArtifact(
        model=model,
        config=config,
        hf_path=hf_path,
        hf_config=hf_config,
        tokenizer_files=tokenizer_files,
        model_type=entry.model_type,
    )
```

### build_config — 从 HF config 构造自定义 config

```python
def build_config(hf_config: dict, entry: ModelEntry):
    """
    从 HF config.json 构造自定义 config dataclass

    策略:
      - 若 config_json_key 存在 (如 "text_config"), 从嵌套字段取值
      - 否则用顶层字段
      - 按 dataclass 字段名过滤, 忽略多余字段
    """
    if entry.config_json_key in hf_config:
        raw = hf_config[entry.config_json_key]
    else:
        raw = hf_config

    # 取 dataclass 定义的字段名
    valid_fields = {f.name for f in dataclasses.fields(entry.config_cls)}

    # 过滤: 只保留 dataclass 认识的字段
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    config = entry.config_cls(**filtered)
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config
```

### collect_tokenizer_files — 收集 tokenizer 文件

```python
def collect_tokenizer_files(hf_path: str) -> list[Path]:
    """
    收集 HF 目录中的 tokenizer 相关文件

    匹配文件名:
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      tokenizer.model
      generation_config.json
      vocab.json
      merges.txt
    """
    patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]
    files = []
    for p in patterns:
        fp = Path(hf_path) / p
        if fp.exists():
            files.append(fp)
    return files
```

## saver.py — epair 保存

### save_epair 接口

```python
def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,  # 默认 50GB
    sample_seq_len: int = 128,                    # torch.export 的示例输入长度
    save_weights: bool = True,
) -> None:
    """
    完整保存流程

    output_dir/
    ├── ep_json_with_ascend_ir.json     # torch.export + ascend_ir
    ├── weight_info.json                # 权重索引
    ├── weights_0.bin                   # 权重数据 (按 max_file_size 切分)
    ├── model_config.json               # 自定义 config 序列化 + HF 原始信息
    ├── generation_config.json          # 生成配置 (从 HF 复制)
    ├── tokenizer.json                  # tokenizer 文件 (从 HF 复制)
    ├── tokenizer_config.json
    └── ...
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建示例输入
    sample_inputs = create_sample_inputs(artifact, sample_seq_len)

    # 2. torch.export 导出模型
    ep = torch.export.export(artifact.model, sample_inputs, strict=False)

    # 3. export_to_epair
    epair = torch_delegate.export_to_epair(ep)

    # 4. 保存权重 + ep_json (复用已有 consolidated_weights_save)
    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
    )

    # 5. 复制 tokenizer 和 generation_config 文件
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)

    # 6. 保存 model_config.json
    save_model_config(artifact, output_dir)
```

### create_sample_inputs — 创建示例输入

```python
def create_sample_inputs(artifact: ModelArtifact, seq_len: int = 128) -> tuple:
    """
    根据 config 创建 torch.export 所需的示例输入

    返回: (input_ids,) — shape=(1, seq_len), dtype=int64
    """
    config = artifact.config
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else 128000

    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64)
    return (input_ids,)
```

### save_model_config — 保存模型配置

```python
def save_model_config(artifact: ModelArtifact, output_dir: Path):
    """
    保存 model_config.json — 包含自定义 config 和 HF 元信息

    格式:
    {
      "model_type": "gemma4_text",
      "config": { ...config dataclass 的 to_dict()... },
      "hf_config": { ...原始 HF config.json... }
    }
    """
    config_data = {
        "model_type": artifact.model_type,
        "config": artifact.config.to_dict() if hasattr(artifact.config, "to_dict")
                   else dataclasses.asdict(artifact.config),
        "hf_config": artifact.hf_config,
    }
    path = output_dir / "model_config.json"
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved: {path}")
```

### copy_tokenizer_files — 复制 tokenizer 文件

```python
def copy_tokenizer_files(files: list[Path], output_dir: Path):
    """复制 tokenizer 相关文件到输出目录"""
    import shutil
    for fp in files:
        dest = output_dir / fp.name
        shutil.copy2(fp, dest)
        print(f"Saved: {dest}")
```

## cann-lm-converter.py — CLI pipeline

对外只暴露 `cann-lm-converter.py`，模型类型从 `config.json` 自动探测，无需 `--model-type`。同时提供程序化入口 `convert()` 供脚本/测试复用，CLI 只是 argparse 薄包一层调它。

### convert 函数 — 程序化入口

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device: str = "cpu",
    max_file_size: int = 50 * 1024 * 1024 * 1024,  # 默认 50GB
    sample_seq_len: int = 128,
) -> None:
    """程序化入口：HF 模型 → epair，返回 None。

    模型类型由 load_model 内部自动探测（从 config.json 的 model_type），
    调用方无需传入。量化参数后续在此签名上扩展。
    """
    # 1. 加载模型（自动探测模型类型）
    print(f"[1/3] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device=device)
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")

    # 2. 保存为 epair
    print(f"[2/3] Saving epair to {output_dir}")
    save_epair(
        artifact, output_dir,
        max_file_size=max_file_size,
        sample_seq_len=sample_seq_len,
    )

    # 3. 验证输出
    print(f"[3/3] Verifying output")
    out = Path(output_dir)
    for name in ["ep_json_with_ascend_ir.json", "model_config.json", "weight_info.json"]:
        fp = out / name
        print(f"      [{'OK' if fp.exists() else 'MISSING'}] {name}")
    print(f"      Weight files: {len(list(out.glob('weights_*.bin')))}")
    print(f"\nDone. Output: {output_dir}")
```

### 命令行参数

```python
#!/usr/bin/env python3
"""CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式

模型类型从 config.json 自动探测，无需 --model-type。

用法:
  python cann-lm-converter.py \
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --dtype float16 \
      --device cpu \
      --max-file-size 53687091200 \
      --sample-seq-len 128
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="CANN-LM Converter: HF model → epair format",
    )
    parser.add_argument("--hf-path", required=True,
                        help="HuggingFace 模型路径 (本地目录)")
    parser.add_argument("--output-dir", required=True,
                        help="epair 输出目录")
    parser.add_argument("--dtype", default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="权重精度 (默认 float16)")
    parser.add_argument("--device", default="cpu",
                        help="加载设备 (默认 cpu)")
    parser.add_argument("--max-file-size", type=int, default=50*1024*1024*1024,
                        help="单个 bin 文件最大字节数 (默认 50GB)")
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="torch.export 示例输入长度 (默认 128)")
    # 量化参数: TBD
    return parser.parse_args()
```

### main 函数

```python
def main():
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    convert(
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        dtype=dtype_map[args.dtype],
        device=args.device,
        max_file_size=args.max_file_size,
        sample_seq_len=args.sample_seq_len,
    )


if __name__ == "__main__":
    main()
```

## 完整调用链

```
cann-lm-converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --dtype, --device, --max-file-size, --sample-seq-len
    │
    ▼
convert(hf_path, output_dir, dtype, device, max_file_size, sample_seq_len)   # 程序化入口
    │
    ▼
loader.load_model(hf_path, dtype, device)   # 不再接收 model_name
    │
    ├── 读 config.json
    ├── model_registry.detect_model_type(hf_config)
    │     → hf_config["model_type"] → get_entry → ModelEntry(config_cls, model_cls, load_weights_fn, ...)
    │
    ├── build_config(hf_config_json, entry)
    │     → 读 config.json → 取 text_config → 过滤字段 → Gemma4TextConfig(**kwargs)
    │
    ├── entry.model_cls(config)  → Gemma4ForCausalLM(config)
    ├── model.to(dtype).eval()
    ├── entry.load_weights_fn(model, hf_path, ...)  → load_gemma4_weights(...)
    │     → safetensors 分片读取 → key 重映射 → param.copy_()
    │
    └── collect_tokenizer_files(hf_path)
          → [tokenizer.json, tokenizer_config.json, ...]

    ▼
saver.save_epair(artifact, output_dir, max_file_size)
    │
    ├── create_sample_inputs(artifact)
    │     → torch.randint(0, vocab_size, (1, seq_len))
    │
    ├── torch.export.export(artifact.model, sample_inputs, strict=False)
    │     → ExportedProgram
    │
    ├── torch_delegate.export_to_epair(ep)
    │     → EpairContents (ep_json + state_dict + graph_signature)
    │
    ├── epair_utils.consolidated_weights_save(epair, output_dir, max_file_size)
    │     → pack_weights(max_file_size) → weights_0.bin, weights_1.bin, ...
    │     → weight_info.json
    │     → ep_json_with_ascend_ir.json
    │
    ├── copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    │     → tokenizer.json, tokenizer_config.json, ...
    │
    └── save_model_config(artifact, output_dir)
          → model_config.json (含自定义 config + HF 原始 config)
```

## 输出文件布局

```
output_dir/
├── ep_json_with_ascend_ir.json    # torch.export 序列化 + ascend_ir 标注
├── weight_info.json               # 权重索引 {bin文件名: [{name, offset, size}]}
├── weights_0.bin                  # 权重数据 (按 max_file_size 切分, 默认 50GB)
├── weights_1.bin
├── model_config.json              # 自定义 config + HF 元信息
├── generation_config.json         # 从 HF 复制
├── tokenizer.json                 # 从 HF 复制
├── tokenizer_config.json          # 从 HF 复制
└── special_tokens_map.json        # 从 HF 复制
```

## model_config.json 格式

```json
{
  "model_type": "gemma4_text",
  "config": {
    "model_type": "gemma4_text",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "vocab_size": 256000,
    "tie_word_embeddings": true,
    "...": "..."
  },
  "hf_config": {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForCausalLM"],
    "text_config": { ... },
    "vision_config": { ... },
    "...": "..."
  }
}
```

## 与已有代码的关系

```
已有代码 (不修改):
  cann_lm/models/gemma4/
    ├── configuration_gemma4.py    ← config dataclass
    └── modeling_gemma4.py        ← model class + load_gemma4_weights()
  cann_lm/models/qwen3/
    ├── configuration_qwen3.py     ← config dataclass
    └── modeling_qwen3.py         ← model class + load_qwen3_weights()
  cann_lm/models/qwen3_5/
    ├── configuration_qwen3_5.py
    └── modeling_qwen3_5.py        ← model class + load_qwen3_5_weights()
  cann_lm/converter/export/epair_utils.py  ← pack_weights, consolidated_weights_save
  torch_delegate                           ← export_to_epair

新增代码:
  cann_lm/converter/model_registry.py   ← 模型注册表
  cann_lm/converter/loader.py           ← 加载 (config + weights + tokenizer)
  cann_lm/converter/saver.py            ← 保存 (epair + tokenizer + config)
  cann_lm/converter/cann-lm-converter.py ← CLI pipeline
```

核心设计原则: 新增代码不修改已有模型定义和权重加载函数，通过注册表模式将 CLI → loader → saver 串联，各模型只需在注册表中注册一次即可接入。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装路径 | (需设置) |
| `PYTHONPATH` | 项目根目录 | (需设置) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload 开关 (设为 0 禁用) | - |

## 使用示例

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 模型类型从 config.json 自动探测，无需 --model-type
python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/gemma-4-e2b-it \
    --output-dir /path/to/output \
    --dtype float16 \
    --device cpu \
    --max-file-size 53687091200 \
    --sample-seq-len 128
```

## 目标导出方案：统一 cache 签名（轻量版）

参考 `npu_tuned_export`（gitee: `ye-xiangqazxsw/cannkit_samplecode_lm_engine_cpp` → `CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/`），把 prefill/decode 导出统一成"KV cache 永远作为输入也作为输出"的图签名，消除 `if past_key_values is None` 的控制流分叉。

> **阶段定位**：当前先做**轻量版**——先立起统一**对外 I/O 契约**和 wrapper 脚手架（基类 + 每模型子类），子类内部仍调各模型原生 `forward`。此时导出的图仍按 sample 走一条分支、且按 sample 形状特化（**不是**一张图覆盖 prefill+decode）。**全量版**（预分配 buffer + `new_kv_cache_pos` scatter 写、一张静态图覆盖 prefill+decode）作为各模型 attention 的后续逐步改造。

### 统一图 I/O 契约（目标）

```
输入:
  input_ids          [batch, seq_len] int64
  attention_mask     [batch, 1, seq_len, kv_cache_max_len]
  position_ids       [batch, seq_len] int64
  past_key_in{i}     [kv_cache_max_len, kv_head, batch, head_dim]   (每层一对)
  past_value_in{i}   同上
  new_kv_cache_pos   [seq_len] int64           # 新 K/V 写进 buffer 的位置 (scatter 索引)
  output_pos         [1] int32 (可选)          # 只算哪个位置的 logits
输出:
  lm_logits          [batch, n_pos, vocab]     # output_pos 选位后
  past_key{i}        更新后的 cache buffer (每层一对)
  past_value{i}
```

三个动态索引张量表达"写到哪 / 读到哪 / 位置是几"：`new_kv_cache_pos`（scatter 写位）、`attention_mask`（屏蔽未填的 cache slot）、`position_ids`（绝对位置，RoPE 索引 `cos[position_ids]`）。`output_pos` 在 norm + lm_head 前选位，decode 只算 1 个 logit。核心机制（来自参考工程 `npu_tuned_model/qwen3/modeling_qwen3.py`）：

```python
# 预分配 buffer，scatter 写入新 K/V，再读整个 buffer（attention_mask 屏蔽空槽）
past_key_value[0][new_kv_cache_pos] = key_states.transpose(0, 2)
past_key_value[1][new_kv_cache_pos] = value_states.transpose(0, 2)
key_states = past_key_value[0].transpose(0, 2)
```

### 结构：ExportWrapper 基类 + 每模型子类

三个模型的 forward 签名、cache 数据结构、特殊特性都不同，无法用一个 wrapper body 通吃：

| | gemma4 | qwen3 | qwen3_5 |
|---|---|---|---|
| 原生 cache | `list[(k,v)]` + `shared_kv_states` dict | `list[(k,v)]` | `SimpleKVCache` + `SimpleDeltaNetCache` |
| 特殊特性 | KV 共享、sliding window、softcap、per-layer rope | 无 | 线性注意力 recurrent state、Q Gate、MRoPE |

采用基类 + 子类：

```python
class ExportWrapper(nn.Module):
    """统一对外 I/O：cache in/out + output_pos 选位。子类实现 _drive_layers。"""
    def __init__(self, model, *, kv_cache_max_len: int, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_values, new_kv_cache_pos, output_pos=None):
        logits, past_kv_flat = self._drive_layers(
            input_ids, attention_mask, position_ids,
            past_key_values, new_kv_cache_pos,
        )
        if output_pos is not None:
            logits = logits[:, output_pos, :]
        return logits, *past_kv_flat

    def _drive_layers(self, *args, **kwargs):
        raise NotImplementedError  # 子类实现：flat tensor ↔ 原生 cache 互转 + 调原生 forward
```

- **基类 `ExportWrapper`**：定义统一签名 + 共用逻辑（`output_pos` 选位、embedding 分离、lm_head）。抽象 `_drive_layers`。
- **`{Gemma4,Qwen3,Qwen3_5}ExportWrapper` 子类**：实现 `_drive_layers`——flat tensor list ↔ 原生 cache 对象互转、调用原生 `forward`、返回 `(logits, flat_past_kv)`。差异（gemma4 的 `shared_kv_states`、qwen3_5 的 `delta_cache`）隔离在子类。

### 轻量 vs 全量

| | 轻量（当前里程碑） | 全量（后续） |
|---|---|---|
| 对外 I/O | 统一（基类） | 统一（基类） |
| 子类内部 | 调原生 `forward`（带 `if past is None` 分支） | 各 attention 加 scatter-write 路径 |
| 导出图 | 按 sample 走一条分支、形状特化 | 一张静态图覆盖 prefill+decode |
| 工作量 | 小（基类 + 3 子类脚手架） | 大（逐模型改 attention） |
| 难点 | flat ↔ native cache 互转 | gemma4 KV 共享/sliding、qwen3_5 线性 attn recurrent state |

### 与现有 saver 的关系（后续落地）

轻量版落地后，`saver.save_epair` 的 `torch.export.export(artifact.model, ...)` 改为 `torch.export.export(wrapper, export_sample)`，sample 构造从 `create_sample_inputs` 升级为 `create_export_sample`（带 past_kv buffer + `new_kv_cache_pos` + `output_pos`）。多 `seq_len` 多图由 `convert()` 循环驱动（后续，参考工程的 `for seq_len in config["seq_len"]`）。

### 各模型的全量难点（后续）

- **gemma4**：KV 共享（后 n 层无自己的 buffer，复用别的层）→ 不 fit "每层一个 buffer"；sliding window 的局部 mask；per-layer head_dim/rope。
- **qwen3_5**：线性注意力（GatedDeltaNet）的 cache 是 `[num_v_heads, k_head_dim, v_head_dim]` 的 **recurrent state**（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——参考工程未覆盖，需单独设计 recurrent buffer 的增量协议。
- **qwen3**：最接近参考工程（标准 GQA + KV cache），全量改造阻力最小，建议先做 qwen3 的 scatter-write 打样。

## 待实现功能

1. **分区模式** — 按 transformer layer 分区保存，支持多卡加载（TBD，公开接口暂不暴露）
2. **量化模块** (`cann_lm/converter/quant/`) — 权重量化（方案 TBD，后续在 `convert()` 签名上扩展量化参数）

## 待办事项

- ~~HuggingFace 模型路径的 tokenizer 和 LLM config 处理~~ — **已完成**：`loader.collect_tokenizer_files` 收集 tokenizer 文件、`saver.copy_tokenizer_files` 复制、`saver.save_model_config` 落盘 `model_config.json`（含自定义 config + HF 原始 config）。
