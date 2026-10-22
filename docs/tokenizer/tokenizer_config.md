该文件指定tokenizer模块的配置参数输入结构：

example:

```json
{
    "tokenizer_config": {
        "tokenizer_file": "tokenizer.json",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "special_tokens": {
            "[unused9]": 45887,
            "[unused10]": 45892,
            "[unused18]": 45986
        }
    }
}
```

字段说明：

| 字段 | 类型 | 必填 | 语义 |
|------|------|------|------|
| `tokenizer_file` | string | 否 | 词表产物文件路径（HF tokenizer.json；BPE/Qwen 算法消费）。空/缺省 = tokenizer 不加载（未就绪）。**本字段由 SR20260903001（2026-09-04 用户拍板）扩展加入**。**2026-09-07 增补**：`EngineOptions.tokenizer_path`（`CANN_LMEngine_ExecutorCreate` 传入）非空时**优先于**本字段；算法选择 likewise 经 `EngineOptions.tokenizer_type`（"BPE"/"Qwen"/"SPM"，空 = BPE），不再硬编码 |
| `bos_token_id` | int | 否 | bos 的 token id（-1/缺省 = 未配置）。**2026-09-07 起**：`add_bos=true` 时 inject 真实注入此 id |
| `eos_token_id` | int | 否 | eos 的 token id。**注意：不参与生成终止**——终止只认 `RequestOptions` 的 stop 集（与 eos 无关，2026-09-07 裁决） |
| `add_bos` | bool | 否 | **2026-09-07 加入（B2 通道来源）**：true → encode 尾段注入 bos_token_id（id 未配置则 WARN 跳过）；缺省 false = 零注入 |
| `add_eos` | bool | 否 | **2026-09-07 加入**：同上，注入 eos_token_id |
| `bos_token` | string | 否 | bos 字面量（承载） |
| `eos_token` | string | 否 | eos 字面量（承载） |
| `special_tokens` | object | 否 | 附加 token 字面量 → id 映射 |

注：`pad_token_id` 不在本段——跨段取自 `model_config` 段（见 docs/model_executor/model_config.md）。
