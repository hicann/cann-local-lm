## ADDED Requirements

### Requirement: Load model from HF path with auto-detection

The system shall provide `load_model(hf_path, *, dtype, device)` that reads `config.json`, auto-detects model type, constructs config, instantiates model, loads weights, and collects tokenizer files — all in one call.

#### Scenario: Load qwen3-0.6B
- **WHEN** `load_model("/path/to/qwen3-0.6B", dtype=torch.float16)` is called
- **THEN** it returns a `ModelArtifact` with the model loaded, config constructed, and tokenizer files collected

### Requirement: Config loading with 3-level fallback

The system shall try `from_pretrained` first, then `from_json_file`, then manual field filtering — to handle gemma4 (no from_pretrained), qwen3 (has both), and qwen3_5 (has from_json_file but no to_dict).

#### Scenario: gemma4 config (no from_pretrained)
- **WHEN** `build_config` is called for gemma4
- **THEN** it falls back to manual field filtering with `config_json_key="text_config"`

#### Scenario: qwen3 config (has from_pretrained)
- **WHEN** `build_config` is called for qwen3
- **THEN** it uses `Qwen3TextConfig.from_pretrained(hf_path)`

### Requirement: ModelArtifact data structure

The system shall return a `ModelArtifact` containing: `model`, `config`, `hf_path`, `hf_config`, `tokenizer_files`, `model_type`.

#### Scenario: Access loaded model
- **WHEN** `load_model` returns
- **THEN** `artifact.model` is an `nn.Module` with weights loaded and `.eval()` called
- **AND** `artifact.config` is the config dataclass instance
- **AND** `artifact.tokenizer_files` is a list of existing file Paths

### Requirement: Tokenizer file collection

The system shall collect known tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json, etc.) from the HF model directory.

#### Scenario: Collect tokenizer files
- **WHEN** `collect_tokenizer_files(hf_path)` is called
- **THEN** it returns a list of Path objects for files that exist in the HF directory
