## ADDED Requirements

### Requirement: CLI entry point with auto-detection

The system shall provide a CLI `cann-lm-converter.py` that accepts `--hf-path`, `--output-dir`, `--dtype`, `--device`, `--max-file-size`, `--sample-seq-len` — with model type auto-detected from config.json (no `--model-type` flag needed).

#### Scenario: Run CLI with qwen3-0.6B
- **WHEN** `python cann-lm-converter.py --hf-path /path/to/qwen3-0.6B --output-dir /path/to/output --dtype float16` is run
- **THEN** it auto-detects model_type="qwen3", loads model, exports epair, and verifies output

### Requirement: Programmatic convert() function

The system shall provide a `convert(hf_path, output_dir, dtype, device, max_file_size, sample_seq_len)` function for script/test reuse.

#### Scenario: Call convert() programmatically
- **WHEN** `convert(hf_path="/path/to/model", output_dir="/out", dtype=torch.float16)` is called
- **THEN** it runs load_model → save_epair → verify, printing progress at each stage

### Requirement: Output verification

The system shall verify output after export: check that ep_json_with_ascend_ir.json, model_config.json, weight_info.json exist, and count weight bin files.

#### Scenario: Verify successful export
- **WHEN** export completes successfully
- **THEN** it prints `[OK]` for each expected file and counts `weights_*.bin` files
