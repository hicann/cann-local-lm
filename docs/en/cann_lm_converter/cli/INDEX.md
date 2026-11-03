# CLI Module Index

English | [简体中文](../../../zh/cann_lm_converter/cli/INDEX.md)

> `cann_lm_converter.py` — the sole external entry point; model type is auto-detected from `config.json`.

## Module Documents

| Document | Content |
|------|------|
| [cli_design.md](./cli_design.md) | `convert()` programmatic entry / command-line parameters / complete call chain / prefill-decode CLI / environment variables / usage examples |

## Code Files

| File | Responsibility |
|------|------|
| `cann_local_lm/converter/cann_lm_converter.py` | CLI entry (`parse_args` + `convert`) + programmatic `convert()` function |

## Key Functions

| Function | Purpose |
|------|------|
| `convert(hf_path, output_dir, dtype, ...)` | Programmatic entry: HF model → epair (orchestrates load_model → save_epair → verify artifacts) |
| `parse_args()` | Command-line argument parsing |

## Command-Line Parameter Quick Reference

| Parameter | Default | Description |
|------|------|------|
| `--hf-path` | (required) | HF model local directory |
| `--output-dir` | (required) | epair output directory |
| `--target-soc` | `KirinX90` | Target SoC (determines dtype derivation + swap packing format; Ascend→bfloat16, Kirin→float16) |
| `--max-file-size` | 50GB | Max bytes per single bin file (default 50GB) |
| `--mode` | `prefill-decode` | `single` / `prefill-decode` |
| `--num-layers` | — | Export only the first N layers (slicing/debugging, None=all) |
| `--quant-path` | — | CANN lightweight tool GPTQ output path (quantization integration, see [INDEX.md](../quant/INDEX.md)) |
| `--quant-method` | — | Quantization algorithm (currently only gptq; auto-spawns lightweight tool, scenario 2, implemented; mutually exclusive with --quant-path) |

> For full parameter descriptions, see [cli_design.md](./cli_design.md) "Command-Line Parameters".

## Call Chain

```
parse_args → convert → loader.load_model (export_index) → [optional quant swap (quant_index)]
           → saver.save_epair / save_epair_prefill_decode (export_index)
```

## Command Reference

### Entry

| Method | Command |
|------|------|
| console_scripts (direct call after pip install) | `cann-lm-converter --hf-path ... --output-dir ...` |
| Script direct call | `python cann_local_lm/converter/cann_lm_converter.py --hf-path ... --output-dir ...` |
| Module entry | `python -m cann_local_lm.converter.cann_lm_converter --hf-path ... --output-dir ...` |

> console_scripts registration: `scripts/cann_lm_converter/pyproject.toml:20-21` `cann-lm-converter = "cann_local_lm.converter.cann_lm_converter:main"`

### Usage Examples

```bash
# Default: prefill-decode + pure embeddings (dtype derived from --target-soc)
cann-lm-converter --hf-path /path/to/qwen3.5-0.8B --output-dir /path/to/out

# Quantization scenario 1: consume lightweight tool GPTQ output (--quant-path)
cann-lm-converter --hf-path /path/to/model --output-dir /path/to/out --quant-path /path/to/dopt_output

# Quantization scenario 2: one-click auto_quant (--quant-method gptq, NPU/GPU/CPU subprocess)
cann-lm-converter --hf-path /path/to/model --output-dir /path/to/out --quant-method gptq

# Layer truncation debugging (--num-layers N)
cann-lm-converter --hf-path /path/to/model --output-dir /path/to/out --num-layers 4
```

> For full usage examples, see [cli_design.md](./cli_design.md) "Usage Examples"; for parameter descriptions, see "Command-Line Parameter Quick Reference" above or [cli_design.md](./cli_design.md) "Command-Line Parameters".

## Related Modules

- **Export**: [INDEX.md](../export/INDEX.md) — `load_model` / `save_epair` / `ExportWrapper` / artifact layout
- **Quantization**: [INDEX.md](../quant/INDEX.md) — `--quant-path` / `--quant-method` integration
