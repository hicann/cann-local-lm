## ADDED Requirements

### Requirement: ExportWrapper for unified logits output

The system shall provide an `ExportWrapper` nn.Module that wraps a CausalLM model so forward returns only the logits tensor, solving the dict-return issue where torch.export creates multiple output_specs including as_none entries.

#### Scenario: Wrap qwen3 model
- **WHEN** `ExportWrapper(model)` is created and `forward(input_ids)` is called
- **THEN** it calls `model(input_ids=input_ids)` and returns `out["logits"]` (not the full dict)

### Requirement: register_ge_converters before export_to_epair

The system shall call `register_ge_converters()` before `torch_delegate.export_to_epair()` to ensure ascend_ir annotations are populated.

#### Scenario: Export without register_ge_converters
- **WHEN** `export_to_epair` is called without prior `register_ge_converters()`
- **THEN** ascend_ir.source is empty for all nodes → C++ reports UNMAPPED_OPERATOR

### Requirement: bfloat16 serialization in pack_weights

The system shall handle bfloat16 tensors in `pack_weights` by viewing as int16 before numpy serialization, since numpy does not support bfloat16.

#### Scenario: Pack bfloat16 weight
- **WHEN** a bfloat16 tensor is encountered in pack_weights
- **THEN** it is viewed as int16 before `numpy().tobytes()` — preserving raw bytes

### Requirement: Non-persistent buffer inclusion in pack_weights

The system shall supplement `state_dict()` with non-persistent buffers from `ep.module().named_buffers()`, since `persistent=False` buffers (e.g. rope.sin_cache) are not in state_dict.

#### Scenario: Pack with non-persistent buffers
- **WHEN** a model has `persistent=False` buffers like `rope.sin_cache`
- **THEN** `pack_weights` includes them from `named_buffers()` → C++ finds matching records in weight_info.json

### Requirement: Save epair output with full file set

The system shall produce: ep_json_with_ascend_ir.json, weight_info.json, weights_*.bin, model_config.json, tokenizer files, .pt2, input_*.bin.

#### Scenario: Save qwen3-0.6B
- **WHEN** `save_epair(artifact, output_dir)` completes
- **THEN** output_dir contains ep_json + weight_info + weights_*.bin + model_config.json + tokenizer files
- **AND** output_dir.pt2 exists for Python-side verification
- **AND** input_0.bin exists for C++ verification

### Requirement: model_config.json with to_dict/asdict fallback

The system shall serialize config to dict using `to_dict()` if available, else `dataclasses.asdict()`.

#### Scenario: gemma4 config (has to_dict)
- **WHEN** `save_model_config` is called for gemma4
- **THEN** it uses `config.to_dict()`

#### Scenario: qwen3_5 config (no to_dict)
- **WHEN** `save_model_config` is called for qwen3_5
- **THEN** it uses `dataclasses.asdict(config)`
