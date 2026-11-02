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
| `eos_token_id` | int | No | eos token id (-1/default = not configured). **Ruling on 2026-10-29 (reverses 2026-09-07)**: when a sampled output token hits this id, generation terminates (`FinishReason::StopToken`; attribution order is after the request-level `stop_token_ids`). Output contract: `token_ids` includes the eos, `text` excludes the eos literal, and the eos is **not appended to the PromptCache ledger** (a next-round token-level rebuild containing the eos rolls back 1 via the "last-token forced recompute" rule, costing at most 1 token of reuse). |
| `add_bos` | bool | No | **Added on 2026-09-07 (B2 channel source)**: true → encode injects bos_token_id at the tail (if id is not configured, WARN and skip); default false = no injection |
| `add_eos` | bool | No | **Added on 2026-09-07**: same as above, injects eos_token_id |
| `bos_token` | string | No | bos literal (carrier) |
| `eos_token` | string | No | eos literal (carrier) |
| `special_tokens` | object | No | Additional token literal → id mapping. **Consumed since 2026-10-29** (previously only logged): on encode, a literal matching this mapping maps to the target special id (before BPE splitting, same window as added tokens); on decode, a hit id demaps to the literal emitted verbatim (the only channel for out-of-vocabulary ids to yield a literal); these ids are treated as special (stripped along with the rest when `skip_special_tokens`). Conflict semantics: on conflict with vocab added/literal entries, **meta wins + WARN** (explicit configuration is authoritative and may override the artifact). Entries with id < 0 are skipped with a WARN. Note: the mapping lives in the configuration layer (SpecialTokenHandler), not in VocabStore — the `.bincache` key only covers tokenizer.json, so meta changes must take effect immediately. |

Note: `pad_token_id` is not in this section — it is taken from the `model_config` section across sections (see docs/model_executor/model_config.md).
