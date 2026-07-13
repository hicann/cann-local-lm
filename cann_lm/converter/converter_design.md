# CANN-LM Converter 设计文档

> 将 HuggingFace 模型转换为 epair 格式的转换工具
>
> 最后更新: 2026-07-11

## 目录结构

```
cann_lm/converter/
├── __init__.py
├── cann-lm-converter.py          # CLI 入口 (pipeline)
├── loader.py                     # 模型加载 (config + weights + tokenizer)
├── saver.py                      # epair 保存 (含 tokenizer/config 复制)
├── model_registry.py             # 模型注册表
├── export/
│   └── epair_utils.py            # 已有: pack_weights, consolidated_weights_save
└── quant/                        # 预留: 量化
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
        "qwen3.5": "qwen3_5", "qwen3_5": "qwen3_5",
    }
    key = aliases.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(REGISTRY)}")
    return REGISTRY[key]
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
    model_name: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ModelArtifact:
    """
    从 HF 路径加载模型 — 使用自定义模型定义

    流程:
      1. 读取 HF config.json
      2. 构造自定义 config dataclass
      3. 实例化自定义模型
      4. 加载 HF safetensors 权重 (调用已有的 load_*_weights)
      5. 收集 tokenizer 文件列表
    """
    entry = get_entry(model_name)

    # 1. 读取原始 config.json
    config_json_path = os.path.join(hf_path, "config.json")
    with open(config_json_path) as f:
        hf_config = json.load(f)

    # 2. 构造自定义 config
    config = build_config(hf_config, entry)

    # 3. 实例化模型
    model = entry.model_cls(config)
    model = model.to(dtype=dtype).to(device=device)
    model.eval()

    # 4. 加载权重 — 调用各模型自带的 load_*_weights
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 5. 收集 tokenizer 文件 (不加载, 只记录路径)
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
    max_file_size: int = 512 * 1024 * 1024,  # 默认 512MB
    sample_seq_len: int = 128,               # torch.export 的示例输入长度
    save_weights: bool = True,
) -> None:
    """
    完整保存流程

    output_dir/
    ├── ep_json_with_ascend_ir.json     # torch.export + ascend_ir
    ├── weight_info.json                # 权重索引
    ├── weights_0.bin                   # 权重数据 (支持切分)
    ├── weights_1.bin
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

    # 2. torch.export
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

### 命令行参数

```python
#!/usr/bin/env python3
"""
CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式

用法:
  python cann-lm-converter.py \
      --hf-path /path/to/gemma-4-e2b-it \
      --model-type gemma4 \
      --output-dir /path/to/output \
      --dtype float16 \
      --max-file-size 536870912

  # 分区模式 (后续实现)
  python cann-lm-converter.py \
      --hf-path /path/to/qwen3.5 \
      --model-type qwen3_5 \
      --output-dir /path/to/output \
      --partition-layers 8
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="CANN-LM Converter: HF model → epair format",
    )
    parser.add_argument("--hf-path", required=True,
                        help="HuggingFace 模型路径 (本地目录)")
    parser.add_argument("--model-type", required=True,
                        choices=["gemma4", "qwen3_5"],
                        help="模型类型")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--dtype", default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="权重精度 (默认 float16)")
    parser.add_argument("--device", default="cpu",
                        help="加载设备 (默认 cpu)")
    parser.add_argument("--max-file-size", type=int, default=512*1024*1024,
                        help="单个 bin 文件最大字节数 (默认 512MB)")
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="torch.export 示例输入长度 (默认 128)")
    parser.add_argument("--no-weights", action="store_true",
                        help="只导出 ep_json, 不保存权重")
    parser.add_argument("--partition-layers", type=int, default=None,
                        help="按层分区: 每个 partition 包含的 transformer layer 数")
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

    # 1. 加载模型
    print(f"[1/3] Loading model: {args.model_type} from {args.hf_path}")
    artifact = load_model(
        args.hf_path,
        args.model_type,
        dtype=dtype_map[args.dtype],
        device=args.device,
    )
    print(f"      Model loaded: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")

    # 2. 保存为 epair
    print(f"[2/3] Saving epair to {args.output_dir}")
    if args.partition_layers:
        # 分区模式 (后续实现)
        print(f"      Partition mode: {args.partition_layers} layers/partition")
        raise NotImplementedError("Partition mode not yet implemented")
    else:
        save_epair(
            artifact,
            args.output_dir,
            max_file_size=args.max_file_size,
            sample_seq_len=args.sample_seq_len,
            save_weights=not args.no_weights,
        )

    # 3. 验证输出
    print(f"[3/3] Verifying output")
    output_dir = Path(args.output_dir)
    expected = ["ep_json_with_ascend_ir.json", "model_config.json"]
    if not args.no_weights:
        expected.append("weight_info.json")
    for name in expected:
        fp = output_dir / name
        if fp.exists():
            print(f"      [OK] {name}")
        else:
            print(f"      [MISSING] {name}")

    bin_files = list(output_dir.glob("weights_*.bin"))
    print(f"      Weight files: {len(bin_files)}")
    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
```

## 完整调用链

```
cann-lm-converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --model-type, --output-dir, --dtype, --max-file-size
    │
    ▼
loader.load_model(hf_path, model_name, dtype, device)
    │
    ├── model_registry.get_entry("gemma4")
    │     → ModelEntry(config_cls, model_cls, load_weights_fn, ...)
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
    ├── torch.export.export(model, sample_inputs, strict=False)
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
├── weights_0.bin                  # 权重数据 (512MB 切分)
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
| `TORCH_DELEGATE_HOME` | torch-delegate-harmony 根目录 | (需设置) |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | fallback codegen 开关 | 启用 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload 开关 | - |
| `ASCEND_HOME_PATH` | CANN 安装路径 | (需设置) |
| `PYTHONPATH` | 项目根目录 | (需设置) |

## 使用示例

```bash
source /home/developer/Ascend/cann/set_env.sh
export TORCH_DELEGATE_HOME="/mnt/workspace/torch-delegate-harmony"
export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python cann_lm/converter/cann-lm-converter.py \
    --hf-path /path/to/gemma-4-e2b-it \
    --model-type gemma4 \
    --output-dir /path/to/output \
    --dtype float16 \
    --max-file-size 536870912
```

## 待实现功能

1. **分区模式** (`--partition-layers`) — 按 transformer layer 分区保存，支持多卡加载
2. **量化模块** (`cann_lm/converter/quant/`) — 权重量化 (W8A16 / W4A16)
3. **pt2 格式 `model_constants_config.json` 解析** — lifted tensor constants 元信息
