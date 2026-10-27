# 短期设计：Converter 层 Adapter 方案

简体中文 | [English](../../../en/cann_lm_converter/models/short_term_design.md)

> 不修改模型代码，通过注册表 + adapter 抹平 gemma4/qwen3_5 的接口差异
>
> 状态: 待核对 | 最后核对: 2026-07-14

## 背景

两个模型（gemma4, qwen3_5）有相同的骨架但接口不一致：

- Config 加载方式不同（gemma4 无 from_pretrained，qwen3_5 无 to_dict）
- forward 签名不同（position_ids 有/无，delta_cache 有/无）
- forward 返回 dict 但 key 不同
- load_weights 签名不同（默认 dtype 不同，导出位置不同）

## 设计原则

- **不改模型代码** — 所有适配在 converter 层完成
- **注册表驱动** — 每个模型只需注册一次，自动接入 pipeline
- **torch.export 优先** — wrapper 统一返回 logits，过滤 assert 节点

## 目录结构

```
cann_lm/
├── models/                          # 不修改
│   ├── gemma4/
│   └── qwen3_5/
├── converter/
│   ├── model_registry.py            # 模型注册表 (差异记录)
│   ├── loader.py                    # 加载适配 (config + weights + tokenizer)
│   ├── saver.py                     # 保存适配 (export + epair + filter)
│   ├── cann_lm_converter.py         # CLI 入口
│   └── export/
│       └── epair_utils.py           # 权重打包 (已有)
```

## model_registry.py — 差异记录

每个模型的接口差异通过 `ModelEntry` 注册：

```python
@dataclass
class ModelEntry:
    config_cls: type                 # Gemma4TextConfig / Qwen3_5TextConfig / ...
    model_cls: type                  # Gemma4ForCausalLM / Qwen3_5ForCausalLM / ...
    load_weights_fn: callable        # load_gemma4_weights / load_qwen3_5_weights / ...
    config_json_key: str             # "text_config" 或 None (扁平 config)
    forward_kwargs: dict             # torch.export 时的 None kwargs
    needs_wrapper: bool = True        # 是否需要 ExportWrapper (dict → logits)

REGISTRY = {
    "gemma4": ModelEntry(
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",     # 嵌套 config
        forward_kwargs={
            "attention_mask": None,
            "past_key_values": None,
            "inputs_embeds": None,
            "labels": None,
            "shared_kv_states": None,      # gemma4 特有
            "output_hidden_states": False,
        },
    ),
    "qwen3_5": ModelEntry(
        config_cls=Qwen3_5TextConfig,
        model_cls=Qwen3_5ForCausalLM,
        load_weights_fn=load_qwen3_5_weights,
        config_json_key="text_config",
        forward_kwargs={
            "attention_mask": None,
            "past_key_values": None,
            "delta_cache": None,           # qwen3_5 特有
            "inputs_embeds": None,
            "labels": None,
            "use_cache": False,            # qwen3_5 特有
        },
    ),
}
```

## loader.py — 加载适配

```python
def load_model(hf_path, model_name, *, dtype=torch.float16, device="cpu"):
    entry = get_entry(model_name)

    # 1. Config 加载 — 适配 from_pretrained 有/无
    if hasattr(entry.config_cls, "from_pretrained"):
        config = entry.config_cls.from_pretrained(hf_path)
    elif hasattr(entry.config_cls, "from_json_file"):
        config = entry.config_cls.from_json_file(f"{hf_path}/config.json")
    else:
        # gemma4: 手动读取 + 字段过滤
        with open(f"{hf_path}/config.json") as f:
            hf_cfg = json.load(f)
        raw = hf_cfg.get(entry.config_json_key, hf_cfg) if entry.config_json_key else hf_cfg
        valid_fields = {f.name for f in dataclasses.fields(entry.config_cls)}
        config = entry.config_cls(**{k: v for k, v in raw.items() if k in valid_fields})

    # 2. 实例化 + 加载权重
    model = entry.model_cls(config).to(dtype=dtype).to(device).eval()
    model = entry.load_weights_fn(model, hf_path, device=device, dtype=dtype)

    # 3. 收集 tokenizer 文件
    tokenizer_files = collect_tokenizer_files(hf_path)

    return ModelArtifact(model, config, hf_path, ..., model_type=entry.model_type)
```

## saver.py — 保存适配

```python
class ExportWrapper(nn.Module):
    """统一 forward 返回 — 只取 logits"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        out = self.model(input_ids=input_ids)
        return out["logits"] if isinstance(out, dict) else out


def save_epair(artifact, output_dir, *, max_file_size=512*1024*1024):
    # 1. torch.export (wrapper 统一返回 logits)
    wrapper = ExportWrapper(artifact.model).eval()
    sample_inputs = (torch.randint(0, artifact.config.vocab_size, (1, 8), dtype=torch.long),)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 2. export_to_epair
    epair = torch_delegate.export_to_epair(ep)

    # 3. 过滤 _assert_tensor_metadata 节点 (run_decompositions 会重新插入)
    nodes = epair.ep_json["graph_module"]["graph"]["nodes"]
    epair.ep_json["graph_module"]["graph"]["nodes"] = [
        n for n in nodes if "assert" not in n.get("target", "").lower()
    ]

    # 4. 保存权重 + ep_json
    consolidated_weights_save(epair, output_dir, max_file_size=max_file_size)

    # 5. 复制 tokenizer + 保存 model_config
    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir)
```

## 处理的差异清单

| 差异 | adapter 方案 | 影响文件 |
|------|-------------|---------|
| gemma4 无 from_pretrained | loader 检测 hasattr，fallback 到手动构造 | loader.py |
| qwen3_5 无 to_dict | saver 用 dataclasses.asdict fallback | saver.py |
| forward 返回 dict | ExportWrapper 只返回 logits | saver.py |
| forward kwargs 不同 | ModelEntry.forward_kwargs 记录 | model_registry.py |
| config_json_key 不同 | ModelEntry 记录，loader 适配 | model_registry.py |
| load_weights 默认 dtype 不同 | 调用方统一传 dtype=torch.float16 | loader.py |
| _assert_tensor_metadata | ep_json 层过滤 target 含 "assert" | saver.py |
| load_gemma4_weights 未导出 | model_registry 直接 import modeling 模块 | model_registry.py |

## 局限性

- 每新增模型需手动在注册表添加条目
- forward kwargs 必须手工填写（无法自动推导）
- ExportWrapper 假设返回 dict 含 "logits" key
- 无统一 Cache 接口（list / SimpleKVCache / SimpleDeltaNetCache 不兼容）
- 无统一 generate（各模型独立实现，推理行为不一致）
