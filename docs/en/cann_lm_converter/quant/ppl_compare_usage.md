# ppl_compare Usage (Float vs GPTQ Quantized Artifact PPL Comparison)

English | [简体中文](../../../zh/cann_lm_converter/quant/ppl_compare_usage.md)

> `cann_lm/converter/quant/ppl_compare.py` — Compare the perplexity (PPL) of a float HF model and a CANN lightweight tool GPTQ quantized artifact to evaluate W4A8 quantization degradation. Single-file script, no `dopt_lm`/converter runtime dependency.
>
> Status: Measured (Qwen3-0.6B, 2026-10-10) | Last reviewed: 2026-10-10

## Principle

GPTQ artifacts are saved by `--hf-type-save` in **plain HF weight format** (W4 pseudo-quantized values already written back to `weight`; you can verify by spot-checking that each group(128) has only ≤16 discrete values). So you can directly `AutoModelForCausalLM.from_pretrained` to load, no `dopt_lm` runtime needed. The quantizer param keys (scale/offset, etc.) in the artifact state_dict are ignored by HF and produce an unexpected-keys warning; the script degrades that log, which is expected behavior.

Comparison fairness: both sides use **same dtype / same data / same sliding window** (seq 2048, non-overlapping windows, logits at position i predict token i+1); the PPL delta is purely quantization degradation.

## Usage

Prerequisite: any python with `torch` + `transformers` (NPU machines need `torch_npu`).

```bash
cd cann_lm/converter/quant

# CPU (no NPU needed, dtype auto float32)
python ppl_compare.py \
    --float-model /path/to/Qwen3-0.6B \
    --quant-model /path/to/quant_qwen3_0dot6_w4a8

# NPU (dtype auto float16)
DEVICE=npu python ppl_compare.py \
    --float-model ... --quant-model ...

# Swap held-out data / adjust token budget
python ppl_compare.py ... --data /path/to/any_text.jsonl --max-tokens 65536
```

## Parameters

| Param | Default | Description |
|------|------|------|
| `--float-model` | (required) | Float HF model dir (also provides the tokenizer; the artifact dir has no tokenizer) |
| `--quant-model` | (required) | Lightweight tool GPTQ artifact dir (`--hf-type-save` format) |
| `--data` | `calibrate_sets/calibrate_coig_cqia.jsonl` | Evaluation jsonl, each line must contain a `text` field |
| `--max-tokens` | `16384` | Token budget cap for PPL computation |
| `--seq-len` | `2048` | Sliding window length |
| `--stride` | `= seq-len` | Window step (default non-overlapping) |
| `--dtype` | `auto` | `auto`: cpu→float32 / others→float16; can be explicit float32/float16/bfloat16 |
| Env var `DEVICE` | `cpu` | `cpu` / `npu` (npu needs torch_npu), consistent with lightweight tool convention |

## Output Example

```
[ppl] device=cpu dtype=float32 seq=2048 stride=2048
[ppl] data=.../calibrate_coig_cqia.jsonl tokens=65536
[ppl] float /path/to/Qwen3-0.6B
      ppl=21.2377 (n_pred=65504)
[ppl] quant /path/to/quant_qwen3_0dot6_w4a8
      ppl=25.1932 (n_pred=65504)
[ppl] quantization impact: 21.2377 -> 25.1932  (+18.62%)
```

## Measured Reference (Qwen3-0.6B, W4A8 GPTQ, CPU fp32)

| Dataset | tokens | float PPL | quant PPL | Degradation |
|--------|--------|-----------|-----------|------|
| coig_cqia (calibration set, in-set) | 65536 | 21.24 | 25.19 | **+18.62%** |
| wikitext-2 test (held-out English) | 65536 | 20.84 | 27.62 | +32.56% |

## Notes

1. **The default data is the GPTQ calibration set** (the same file as `config.yaml`'s `train_files`); the quantized model has seen this data, so results are optimistic. For formal evaluation, specify held-out jsonl via `--data`.
2. **Distribution shift amplifies degradation**: the calibration set is Chinese instruction data; English wiki (wikitext) testing stacks language + domain dual shift, so +32.6% cannot be fully attributed to quantization itself. Choose same-distribution held-out data close to the deployment scenario.
3. Increase `--max-tokens` to reduce variance (coig is stable at both 16k/65k budgets: +18.5% / +18.6%).
4. The artifact has no tokenizer; the script uniformly uses the `--float-model` dir's tokenizer (both sides are same-source).
