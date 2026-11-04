# Manifest Field Description

English | [简体中文](README.md)

`manifest-v1.schema.json` is automatically generated from the typed Python Manifest. The
Manifest is written into the PTE together with `processed_bytes`; the OM resides in the PTE
named data, and the external weights reside in the PTD named data.

| Field | Purpose |
|---|---|
| `runtime_protocol.name` | Fixed to `AscendBackend`, identifying the current backend protocol |
| `runtime_protocol.version` | runtime protocol version, currently `1` |
| `runtime_protocol.artifact_abi` | artifact ABI version, currently `1` |
| `method` | ExecuTorch method name |
| `delegate_index` | delegate index within the method |
| `role` | `inference`, `prefill`, or `decode` |
| `acl_soc_name` | Validated device name, also used as the CANN `ge.socVersion` and the runtime matching target |
| `model.named_data_key` | key of the OM in the PTE named data |
| `model.sha256` | OM content digest |
| `model.nbytes` | OM logical length |
| `weight_groups[].weight_group_key` | PTD weight group key |
| `weight_groups[].sha256` | weight group content digest |
| `weight_groups[].total_nbytes` | weight group logical length |
| `file_constant_bindings[].binding_id` | FileConstant binding ID |
| `file_constant_bindings[].file_name` | file name on the OM/GE side |
| `file_constant_bindings[].weight_group_key` | the weight group it belongs to |
| `file_constant_bindings[].offset` | offset within the weight group |
| `file_constant_bindings[].nbytes` | length of the bound range |
| `file_constant_bindings[].shape` | logical shape of the bound range |
| `file_constant_bindings[].alias_of` | optional, shares an existing bound range |
| `io_contract.inputs` | list of input tensor bindings |
| `io_contract.outputs` | list of output tensor bindings |
| `io_contract.*[].et_arg_index` | ExecuTorch argument index |
| `io_contract.*[].direction` | `input` or `output` |
| `io_contract.*[].ascend_name` | ACL tensor name |
| `io_contract.*[].dtype` | tensor data type |
| `io_contract.*[].shape` | tensor shape |
| `io_contract.*[].shape_range` | optional dynamic shape range |
| `io_contract.*[].semantic_role` | `data`, `state_current`, or `state_next` |
| `state.bindings` | list of state bindings, may be empty |
| `state.bindings[].state_id` | state ID |
| `state.bindings[].current_input_arg` | argument index of the current-state input |
| `state.bindings[].next_output_arg` | argument index of the next-state output |
| `state.bindings[].dtype` | state data type |
| `state.bindings[].shape` | state shape |
| `state.bindings[].reset_policy` | `on_prefill`, `explicit`, or `never` |
| `state.bindings[].reset_owner` | whether it is responsible for resetting the state |
| `state.bindings[].host_visibility` | `device_only` or `synchronized` |

State inputs and outputs always use separate current/next double buffers, and the new state
is committed after successful execution. The `buffer_mode` field in single-buffer
experimental artifacts is no longer supported; they must be re-exported with the current code.

Fixed storage locations, artifact types, capability declarations, alignment parameters,
storage types, layouts, and extension fields are no longer written into the Manifest; they
are handled by the producer/runtime using fixed conventions.

The Python field model and the JSON/ASCD output are located in
`python/executorch_npu/artifacts/manifest.py`. Graph mapping, weight registration, and
shared-state checks are performed in the corresponding construction steps under
`artifacts/`; runtime parsing and validation are located in `runtime/manifest.cpp`. The main
compilation path keeps the model objects through to the final output and does not
deserialize the manifest.


## Update and Check

After modifying fields, run from the repository root:

```bash
python scripts/generate_manifest_schema.py
python scripts/generate_manifest_schema.py --check
```

The generated file describes field types, values, and structural constraints; cross-field and
cross-delegate business constraints are still validated by the artifact builders and the C++
runtime. Tests also verify that the generated file is not out of date, and that the actual
serialized artifacts (including single/double state) conform to the JSON Schema.
