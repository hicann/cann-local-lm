简体中文 | [English](../../../en/cann_lm_engine/tokenizer/tokenizer_config.md)

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
| `eos_token_id` | int | 否 | eos 的 token id（-1/缺省 = 未配置）。**2026-10-29 裁决（推翻 2026-09-07）**：命中采样输出即终止生成（`FinishReason::StopToken`，归因次序在请求级 `stop_token_ids` 之后）；口径：`token_ids` 含 eos、`text` 不含 eos 字面量、**不入 PromptCache 台账**（下轮含 eos 的 token 级重建按"末 token 强制重算"回退 1，至多损 1 token 复用） |
| `add_bos` | bool | 否 | **2026-09-07 加入（B2 通道来源）**：true → encode 尾段注入 bos_token_id（id 未配置则 WARN 跳过）；缺省 false = 零注入 |
| `add_eos` | bool | 否 | **2026-09-07 加入**：同上，注入 eos_token_id |
| `bos_token` | string | 否 | bos 字面量（承载） |
| `eos_token` | string | 否 | eos 字面量（承载） |
| `special_tokens` | object | 否 | 附加 token 字面量 → id 映射。**2026-10-29 起被消费**（此前仅记日志）：encode 时字面量命中映射为目标特殊 id（先于 BPE 切分，与 added token 同窗）；decode 时命中 id 反查字面量原样输出（词表外 id 出字面量的唯一通道）；这些 id 视为 special（`skip_special_tokens` 时一并剥离）。冲突语义：与词表 added/字面量冲突 → **meta 优先 + WARN**（显式配置权威，可覆盖产物）。id < 0 条目 WARN 跳过。注：本映射存于配置层（SpecialTokenHandler），不进 VocabStore——`.bincache` 缓存键只覆盖 tokenizer.json，meta 变更需可即时生效 |

注：`pad_token_id` 不在本段——跨段取自 `model_config` 段（见 docs/model_executor/model_config.md）。
