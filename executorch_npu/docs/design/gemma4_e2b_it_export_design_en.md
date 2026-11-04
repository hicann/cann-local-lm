# Gemma4 E2B-IT Export Approaches: Static Shape and Dynamic Shape

> **Note**: This document describes two export approaches for Gemma4 E2B-IT:
> - **Static shape approach** (Sections 1-13): uses fixed prefill/decode methods and explicitly manages state tensors
> - **Dynamic shape approach** (Section 14): uses a single method and supports variable sequence lengths through dynamic shape
>
> Both approaches have been validated in MMLU evaluation and can be chosen according to actual requirements.

## Purpose

This document defines two export and execution approaches for Gemma4 E2B-IT text-only. Model cache semantics exist
only in the Python adapter; the generic backend/runtime manages only named, device-resident state tensors.

The goal is not to rewrite the runtime's K/V management, but to make Gemma4's cache explicit at the export layer as
ExecuTorch method boundaries, reusing the generic EPAIR, GE AOT, PTE/PTD, and
device-resident state flows.

## Model and State Facts

The target model is the text-only path of Gemma4 E2B-IT, whose decoder contains 35 layers:

```text
35 decoder layers
31 DynamicCache layers (sliding_attention and full_attention)
2 shared attention layers (sliding_attention and full_attention)
hidden_size=1536
num_attention_heads=8
num_key_value_heads=1
head_dim=256 (sliding_attention)
global_head_dim=512 (full_attention)
sliding_window=512
num_kv_shared_layers=20
```

DynamicCache layers use the standard K/V cache; shared attention layers share the K/V cache of
sliding_attention and full_attention across the last 20 layers. The Transformers cache object is an export-time
implementation detail and cannot serve as a cross-method interface of the ExecuTorch runtime.

## Approach Selection

### Adopted: Explicit State Adapter at the Export Layer

`Gemma4StaticStateAdapter` receives all state tensors, rebuilds the Transformers `DynamicCache` and the shared KV dictionary that are valid only within the current forward, and returns logits together with the updated state tensors.
The generic backend marks them as `kv_cache` through `state_io_contract`.

This approach preserves the Transformers weights and forward semantics, while the runtime only takes on device tensor allocation,
namespace isolation, and inout lifecycle management.

### Not Adopted: Runtime Awareness of the Gemma4 Cache

No Gemma4-, Transformers-, K/V-, or shared-attention-specific branches are introduced in the C++ runtime. This avoids
coupling the runtime with the Python model implementation and keeps the existing state contract reusable for other models.

### Not Adopted: Directly Exporting the Transformers StaticCache

`StaticCache` can assist eager/export, but its internal state does not stably become an ExecuTorch user input/output.
It may be lifted as a constant or fall into internal copy paths, and therefore cannot satisfy the requirements for backend-owned mutable state,
multi-session isolation, and per-capacity validation.

## State Contract

The adapter explicitly defines all Gemma4 cache tensors as method inputs and outputs:

```text
cache.layer_<n>.key/value              kv_cache
shared.sliding_attention.key/value     kv_cache
shared.full_attention.key/value        kv_cache
```

prefill resets the state namespace. decode reads the previous round's state from the same namespace and writes the next round's
state. The runtime must not depend on the Gemma4 or Transformers cache class.

### DynamicCache K/V State

Each DynamicCache layer produces:

```text
cache.layer_<layer_idx>.key
cache.layer_<layer_idx>.value
```

with shape:

```text
[batch, num_key_value_heads, max_seq_len, head_dim]
```

where `head_dim` is chosen according to the layer type:
- sliding_attention: `head_dim=256`
- full_attention: `global_head_dim=512`

The role is `kv_cache`. prefill writes the K/V of the prompt; decode reads the existing prefix and writes the current token
position. The physical shape of the state always uses `max_seq_len`, rather than the number of tokens of the current run round.

### Shared Attention K/V State

The last 20 layers share the K/V cache of sliding_attention and full_attention, producing:

```text
shared.sliding_attention.key
shared.sliding_attention.value
shared.full_attention.key
shared.full_attention.value
```

with shape:

```text
[batch, num_key_value_heads, max_seq_len, head_dim]
```

where sliding_attention uses `head_dim=256` and full_attention uses `global_head_dim=512`.

The role is `kv_cache`. prefill writes the K/V of the entire prompt; decode reads the existing prefix and writes the current
token position.

### State Update Strategy

The decode phase of Gemma4 has two kinds of state update patterns:

1. **DynamicCache layers**: each layer independently updates its own K/V cache.
2. **Shared attention layers**: the last 20 layers share the K/V cache of sliding/full attention,
   so only the layer that updates the cache last writes the shared state.

In `_static_states_from_outputs`, the adapter distinguishes `cache` from `shared` according to the `kind` field of `state_specs`, and extracts the corresponding K/V tensors from `past_key_values` and `shared_kv_states`.

## Method Boundaries and Runtime Placement

The artifact contains a fixed prefill method and a single-token decode method:

```text
prefill(input_ids, input_pos, states...) -> logits, next_states...
decode(input_ids, input_pos, states...) -> logits, next_states...
```

Currently `decode_seq_len` is fixed to 1. The caller first executes `prefill`, then calls the same `decode` in each round,
writing the current absolute position into `input_pos`. The full/sliding attention state uses a fixed capacity, and the mask
shields the unwritten region according to the runtime position, so there is no need to generate a separate method for each cache length.

The Transformers cache update of Gemma4 requires the position to be passed explicitly. In runtime decode mode, the adapter
obtains the current cache length from `input_pos` and functionally updates the fixed-capacity cache and the shared K/V state,
ensuring that the causal mask, the K/V prefix, the position IDs, and the actual generation length remain consistent.

## Export Compatibility and Copy

### ExecuTorch Compatibility Configuration

The export entry point uses `aten_edge_compile_config()`:

```text
Older ExecuTorch: _use_edge_ops=False
Newer ExecuTorch: _skip_dim_order=True, followed by EdgeOpOverload -> ATen
Unknown API: fails before export
```

Before entering EPAIR conversion, the system restores known EdgeOpOverloads through the underlying ATen overload.
The identity `_clone_dim_order` is rewritten as `aten.clone`; non-identity dim-order layouts are explicitly rejected.

### Text-Path Model Trimming

The Gemma4 checkpoint contains both a vision tower and an audio tower. A pure text export does not invoke either of these inputs, so the export
entry point sets `vision_tower` and `audio_tower` to null after the model is loaded, reducing the host memory peak. This operation does not
change the language model weights or the text computation path.

```python
mm = getattr(model, "model", model)
for name in ("vision_tower", "audio_tower"):
    if hasattr(mm, name) and getattr(mm, name) is not None:
        setattr(mm, name, None)
```

### Eager Attention

Gemma4 uses `scaled_dot_product_attention` (SDPA) by default, but this may cause numerical issues on the NPU.
The adapter specifies `attn_implementation="eager"` when loading the model, using a manual matmul + softmax implementation:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
```

### Copy Semantics

The Gemma4 state update may generate `aten.copy.default`. The system must preserve this operator and have it converted by the supported
TorchAir and TorchDelegate copy paths. Rewriting it as `aten.alias` would lose the storage
semantics of the copy and is therefore incorrect.

## Data Flow

Export phase:

```text
Gemma4 E2B-IT checkpoint
  -> Release unused vision/audio tower (text-only path)
  -> Gemma4StaticStateAdapter (attn_implementation="eager")
  -> torch.export(prefill/decode)
  -> ExecuTorch transform/decomposition
  -> EdgeOpOverload -> ATen compatibility lowering
  -> AscendBackend.preprocess_multimethod
  -> EPAIR
  -> GE Offline_compile
  -> model.pte + ascend.weights.package.ge_aot.ptd
```

Execution phase:

```text
prefill input_ids/input_pos
  -> runtime resolves reset state namespace
  -> GE execute
  -> logits copied to host; next state remains device-resident
  -> greedy token
  -> decode with the same namespace and runtime input_pos
```

An `executorch_npu_stateful_runner session` must complete prefill and the subsequent decode within the same process and the same Module, releasing the state namespace before exit.

## Artifact Contract

The core backend artifacts are:

```text
model.pte
ascend.weights.package.ge_aot.ptd
```

`generation_config.json` is an application-layer sidecar. Its schema version 1 records
`prefill_seq_len`, `max_seq_len`, and `max_new_tokens`, and is not part of the backend artifact ABI.

## Implementation Boundaries

The current implementation only covers the Gemma4 E2B-IT text-only static-state path, including:

1. State spec and zero-state construction.
2. Real checkpoint eager prefill/decode.
3. Fixed-shape `torch.export`.
4. Multimethod prefill/decode packaging.
5. EPAIR, GE AOT, PTE/PTD, and stateful runner flows.
6. Text-path tower trimming and eager attention configuration.

Currently not included:

1. Vision tokens or image inputs.
2. Dynamic shape.
3. Chunked decode with `decode_seq_len > 1`.
4. Gemma4-specific runtime C++ logic.
5. Accommodating incorrect prompt lengths with padding or truncation.

## Testing and Acceptance

### Unit Tests

Tests cover at least:

1. DynamicCache/shared attention layers each generate the correct state spec.
2. State names, roles, dtypes, and shapes are stable.
3. Under different `max_seq_len` values, the fixed-capacity state and the runtime position contract are consistent.
4. `generation_config.json` schema, capacity, and prompt length validation.
5. ExecuTorch legacy/current compatibility configuration and rejection of unknown APIs.
6. Text-path tower trimming does not affect language model output.

### Real Checkpoint Acceptance

Real model acceptance is executed in layers:

1. The tokenizer prompt token count matches `prefill_seq_len` exactly.
2. CPU eager prefill/decode logits are finite, and greedy tokens can be generated.
3. After `torch.export`, all states reside at the method input/output boundary and cannot be hidden as lifted constants.
4. Full EPAIR conversion, GE Offline_compile, and PTE/PTD serialization succeed.
5. The NPU session completes `prefill -> decode` in the same namespace and updates `input_pos` round by round.
6. NPU token IDs match the CPU greedy baseline of the same adapter.
7. NPU output contains no NaN/Inf.

## Risks and Known Performance Issues

### Changes in Transformers Internal Implementation

The adapter must concentrate its dependencies on Transformers internal cache fields, layer types, and mask hooks in
`examples/gemma4/export/adapter.py`, and constrain them with state-spec tests.

### State Memory Scale

Both the DynamicCache K/V state and the shared attention K/V state grow with the number of layers and `max_seq_len`.
Export and validation should first use a smaller `max_seq_len` and then gradually expand it; the success of a small-capacity artifact must not be extrapolated to long-context capability.

### ExecuTorch Decomposition Performance

The real Gemma4 static-state graph contains a large number of explicit state inputs (34 state tensors). It is currently observed that
`ExportedProgram.run_decompositions()` enters the PyTorch AOT Autograd
functionalization, alias metadata, and symbolic-shape interpreter paths. This phase may become an export
bottleneck, and EPAIR or GE has not yet been reached. Optimization must preserve the state input/output and copy semantics, and must not
replace `aten.copy.default` with `aten.alias`.

### SDPA Numerical Stability

Gemma4's default `scaled_dot_product_attention` may cause numerical issues on the NPU. The
`attn_implementation="eager"` configuration must be used, with a manual matmul + softmax implementation.

## Differences from Qwen3.5

### Cache Structure

| Feature | Qwen3.5 | Gemma4 |
|------|---------|--------|
| Cache type | DynamicCache + LinearCache | DynamicCache + Shared KV |
| State type | kv_cache + mutable_state | kv_cache only |
| Linear attention | Yes (conv_states + recurrent_states) | No |
| Shared attention | No | Yes (sliding + full) |
| Number of layers | 24 | 35 |

### State Binding Strategy

Qwen3.5 uses a simple state binding strategy, because all states are updated in the decode phase.

Gemma4 uses a 3-pass state binding strategy:

1. **Pass 1: Exclusive dependency matching**
   - Matches outputs that depend only on the current state input
   - Applicable to the prefill phase, where each state output depends only on the corresponding state input

2. **Pass 2: Fewest dependencies matching**
   - Matches outputs that depend on the current state input and depend on the fewest other state inputs
   - Applicable to the decode phase, where a state output may accumulate dependencies on multiple state inputs

3. **Pass 3: Unused state detection**
   - Detects whether a state input is used by any state output
   - If unused, sets `next_output_arg = -1` to indicate that the state is not updated
   - Applicable to the shared attention states in the decode phase

### Anchor Mechanism

There is an issue in the prefill phase of Gemma4: some state inputs (such as the shared attention states) are
completely replaced during prefill and never read the previous state. This may cause `torch.export` to optimize away these state
inputs.

The solution is to add an anchor on the state outputs:

```python
anchor = sum(s.flatten()[0] * 0 for s in states)
next_states = tuple(ns + anchor for ns in next_states)
```

This adds a dependency on all state inputs to the state outputs, but does not change the values (because anchor = 0).
`torch.export` retains all state inputs, ensuring that the delegated subgraph contains complete state I/O.

## Detailed 3-Pass State Binding Algorithm

### Problem Background

ExecuTorch's `to_edge_transform_and_lower` converts the original exported graph into a delegated subgraph,
in which the order of state outputs may differ from that of the original exported graph. We need to correctly identify
each state output in the delegated subgraph.

### Pass 1: Exclusive Dependency Matching

**Goal**: match outputs that depend only on the current state input.

**Algorithm**:

```python
exclusive_matches = [
    output_name
    for output_name in output_names
    if output_name not in used_output_names
    and _matches_state_metadata(nodes.get(output_name), binding)
    and source_input_name in dependency_cache[output_name]
    and not (dependency_cache[output_name] - {source_input_name}) & state_input_names
]
```

**Applicable scenarios**:
- Prefill phase: each state output depends only on the corresponding state input
- Simple models: no cross dependencies between state outputs

**Example**:

```text
states_0 (input) -> aten_cat_0 (output)  # depends only on states_0
states_1 (input) -> aten_cat_1 (output)  # depends only on states_1
```

### Pass 2: Fewest Dependencies Matching

**Goal**: match outputs that depend on the current state input and depend on the fewest other state inputs.

**Algorithm**:

```python
dep_matches = [
    output_name
    for output_name in output_names
    if output_name not in used_output_names
    and _matches_state_metadata(nodes.get(output_name), binding)
    and source_input_name in dependency_cache[output_name]
]
if dep_matches:
    output_name = min(
        dep_matches,
        key=lambda name: len(dependency_cache[name] & state_input_names),
    )
```

**Applicable scenarios**:
- Decode phase: a state output may accumulate dependencies on multiple state inputs
- Complex models: cross dependencies exist between state outputs

**Example**:

```text
states_0 (input) -> aten_cat_0 (output)  # depends on states_0, states_1
states_1 (input) -> aten_cat_1 (output)  # depends on states_0, states_1, states_2
```

Pass 1 cannot match (because both have cross dependencies), so Pass 2 selects the output with the fewest dependencies.

### Pass 3: Unused State Detection

**Goal**: detect whether a state input is used by any state output.

**Algorithm**:

```python
state_is_used = any(
    source_input_name in dependency_cache[output_name]
    and _matches_state_metadata(nodes.get(output_name), binding)
    for output_name in output_names
)
if not state_is_used:
    binding["next_output_arg"] = -1  # sentinel: no output
```

**Applicable scenarios**:
- Shared attention states in the decode phase: some states are not updated during decode

**Example**:

```text
states_30 (shared.sliding_attention.key) -> no output depends on it
states_31 (shared.sliding_attention.value) -> no output depends on it
```

Set `next_output_arg = -1`, and the runtime keeps the state unchanged.

### Why the 3-Pass Strategy Is Needed

1. **Pass 1 takes priority**: exclusive dependency is the most reliable matching method and avoids mismatches.
2. **Pass 2 supplements it**: handles the accumulated dependency problem in the decode phase.
3. **Pass 3 is the fallback**: handles states that are not updated, avoiding errors caused by forced mapping.

This layered strategy ensures that state outputs are correctly bound in different scenarios while maintaining backward compatibility.

## Dynamic Shape Approach

In addition to the static shape approach described above, Gemma4 E2B-IT also supports a dynamic shape export approach. This approach uses a single method
(rather than separate prefill/decode) and supports variable sequence lengths through `torch.export.Dim`.

### Approach Overview

The core ideas of the dynamic shape approach are:

1. **Single method**: export one unified `model` method that handles both prefill and decode
2. **Dynamic sequence length**: use `torch.export.Dim` to define the sequence length dimension, with the range `[2, MAX_SEQ_LEN]`
3. **Internal KV cache**: use `Gemma4CustomKVCache` to manage the KV cache, controlling prefill/decode behavior through `cumulative_length`
4. **No explicit state I/O**: the KV cache is internal model state and is not an input/output of the method

### Export Configuration

```python
class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, cache: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.cache = cache

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        for layer_cache in self.cache.layers:
            layer_cache.cumulative_length.copy_(cache_position[0])
        return self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        ).logits

# Dynamic shape definition
sequence_length = torch.export.Dim(
    "sequence_length", min=2, max=MAX_SEQ_LEN
)
dynamic_shapes = {
    "input_ids": {1: sequence_length},
    "cache_position": {0: sequence_length},
}
```

### Method Signature

The dynamic shape approach exports only one method:

```text
model(input_ids, cache_position) -> logits
```

- `input_ids`: shape `[batch, seq_len]`, where `seq_len` is a dynamic dimension
- `cache_position`: shape `[seq_len]`, consistent with the `seq_len` of `input_ids`
- `logits`: shape `[batch, seq_len, vocab_size]`

### Runtime Behavior

The runtime distinguishes prefill from decode through `cache_position`:

- **Prefill**: `cache_position` starts from 0, with a length equal to the number of prompt tokens
- **Decode**: `cache_position` starts from the prompt length and increases by 1 each time

The KV cache is managed internally by `Gemma4CustomKVCache`, which records the current sequence length through `cumulative_length`.
On each forward, `cache_position[0]` is copied into `cumulative_length`, and the model automatically reads and updates
the KV cache at the corresponding position.

### Comparison with the Static Shape Approach

| Feature | Static shape approach | Dynamic shape approach |
|------|-------------|-------------|
| Number of methods | 2 (prefill + decode) | 1 (model) |
| Sequence length | Fixed (prefill_seq_len, decode_seq_len=1) | Dynamic [2, MAX_SEQ_LEN] |
| State management | Explicit state I/O (34 tensors) | Internal KV cache |
| State binding | Requires the 3-pass algorithm | No binding required |
| Anchor mechanism | Required | Not required |
| Compilation time | Longer (two methods) | Shorter (one method) |
| Deployment complexity | Higher (two methods must be managed) | Lower (single method) |
| Performance | Slightly higher (fixed shape optimization) | Slightly lower (dynamic shape overhead) |

### Applicable Scenarios

**The static shape approach is suitable for**:
- Fine-grained control of the state lifecycle is needed
- Multi-session isolation is needed
- Per-capacity validation is needed
- Extremely high performance requirements

**The dynamic shape approach is suitable for**:
- Rapid prototype validation
- Simplified deployment flows
- Complex state management is not needed
- Sensitivity to compilation time

### MMLU Evaluation Validation

The dynamic shape approach has been validated in MMLU evaluation:

- **NPU accuracy**: 48.87% (6863/14042)
- **CPU accuracy**: 48.92% (6870/14042)
- **Difference between NPU and CPU**: 0.05% (highly consistent)

The evaluation uses `MAX_SEQ_LEN=4096`, which supports all sample lengths in the MMLU dataset.

### Implementation Details

Key implementation files of the dynamic shape approach:

- `examples/gemma4/export/pt2.py`: PT2 export script
- `examples/gemma4/export/pte.py`: PTE export script
- `examples/common/cache.py`: `Gemma4CustomKVCache` implementation

Export flow:

```text
Gemma4 E2B-IT checkpoint
  -> Gemma4CustomKVCache (manages the KV cache internally)
  -> torch.export with dynamic shapes
  -> ExecuTorch transform/decomposition
  -> AscendBackend.preprocess_multimethod
  -> EPAIR
  -> GE Offline_compile
  -> model.pte + ascend.weights.package.ge_aot.ptd
```

Execution flow:

```text
input_ids + cache_position
  -> runtime loads model
  -> GE execute with dynamic shape
  -> logits copied to host
  -> greedy token
  -> next iteration with updated cache_position
```

### Limitations and Considerations

1. **Sequence length range**: `min=2` is a requirement of torch.export and cannot be 1
2. **Maximum sequence length**: limited by NPU memory and compilation time; it is recommended not to exceed 4096
3. **KV cache memory**: grows linearly with `MAX_SEQ_LEN` and must be planned in advance
4. **Compilation time**: dynamic shape compilation is slower than fixed shape compilation; the first export may take 30-60 minutes
5. **Runtime performance**: dynamic shape introduces a small performance overhead, but its impact on MMLU evaluation is negligible
