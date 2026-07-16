## ADDED Requirements

### Requirement: Model registration via ModelEntry dataclass

The system shall provide a `ModelEntry` dataclass that records per-model metadata: `model_type`, `config_cls`, `model_cls`, `load_weights_fn`, `config_json_key` (str or None for flat config).

#### Scenario: Register a new model
- **WHEN** a new model is added to `REGISTRY` dict with a `ModelEntry`
- **THEN** it becomes available via `get_entry(name)` and `detect_model_type(hf_config)`

### Requirement: Alias-based name lookup

The system shall support alias lookup so that "gemma-4" maps to "gemma4", "qwen3.5" maps to "qwen3_5", etc.

#### Scenario: Look up model by alias
- **WHEN** `get_entry("gemma-4")` is called
- **THEN** it returns the `ModelEntry` for "gemma4"

### Requirement: Automatic model type detection from config.json

The system shall read `model_type` field from HF `config.json` and automatically resolve to the correct `ModelEntry` without user specifying `--model-type`.

#### Scenario: Auto-detect qwen3 from config.json
- **WHEN** `detect_model_type({"model_type": "qwen3"})` is called
- **THEN** it returns the `ModelEntry` with `config_cls=Qwen3TextConfig, model_cls=Qwen3ForCausalLM`

#### Scenario: Unknown model type
- **WHEN** `detect_model_type({"model_type": "unknown"})` is called
- **THEN** it raises `ValueError` listing all available models

### Requirement: Three models registered by default

The system shall have `gemma4`, `qwen3`, `qwen3_5` pre-registered in `REGISTRY`.

#### Scenario: List available models
- **WHEN** `available_models()` is called
- **THEN** it returns `["gemma4", "qwen3", "qwen3_5"]`
