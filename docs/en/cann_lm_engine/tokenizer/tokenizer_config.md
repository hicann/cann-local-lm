English | [简体中文](../../../zh/cann_lm_engine/tokenizer/tokenizer_config.md)

This file specifies the configuration parameter input structure of the tokenizer module:

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

Field descriptions:

| Field | Type | Required | Semantics |
|------|------|------|------|
| `tokenizer_file` | string | No | Vocabulary artifact file path (HF tokenizer.json; consumed by BPE/Qwen algorithm). Empty/default = tokenizer not loaded (not ready). **This field was added by SR20260903001 (user decision on 2026-09-04).** **2026-09-07 supplement**: When `EngineOptions.tokenizer_path` (passed in by `CANN_LMEngine_ExecutorCreate`) is non-empty, it takes **priority over** this field; the algorithm selection likewise goes through `EngineOptions.tokenizer_type` ("BPE"/"Qwen"/"SPM", empty = BPE), no longer hardcoded |
| `bos_token_id` | int | No | bos token id (-1/default = not configured). **From 2026-09-07**: when `add_bos=true`, this id is actually injected |
| `eos_token_id` | int | No | eos token id. **Note: does not participate in generation termination** — termination only recognizes the stop set of `RequestOptions` (unrelated to eos, decided on 2026-09-07) |
| `add_bos` | bool | No | **Added on 2026-09-07 (B2 channel source)**: true → encode injects bos_token_id at the tail (if id is not configured, WARN and skip); default false = no injection |
| `add_eos` | bool | No | **Added on 2026-09-07**: same as above, injects eos_token_id |
| `bos_token` | string | No | bos literal (carrier) |
| `eos_token` | string | No | eos literal (carrier) |
| `special_tokens` | object | No | Additional token literal → id mapping |

Note: `pad_token_id` is not in this section — it is taken from the `model_config` section across sections (see docs/model_executor/model_config.md).
