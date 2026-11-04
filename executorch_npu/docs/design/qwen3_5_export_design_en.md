# Qwen3.5 Export Design: Dynamic Prefill and Fixed Decode

## Purpose

This document defines the export, state management, and runtime contract for the Qwen3.5 text-only model on ExecuTorch and the Ascend backend.
The design goal is to encapsulate prefill and decode into a single PTE, share one copy of the weight data,
and persist model state across multiple invocations via a state namespace.

The model is exported as two `ExportedProgram`s: `prefill` with a dynamic sequence length and `decode` with a fixed single token.
Both use the same model forward semantics; the differences are determined by the input shapes and the initial state at export time.

## Model and State Facts

The Qwen3.5 text-only decoder is a hybrid attention architecture, with each layer's type specified by `config.layer_types`:

```text
full_attention
linear_attention
```

The two layer types have different state semantics:

- Full Attention stores a fixed-capacity K/V cache and maintains the logical sequence length.
- Linear Attention stores a convolution state and a recurrent state.
- Physical state capacity is determined at export time; the logical length is determined by runtime positions.
- Batch size is currently fixed at 1.

The Transformers Cache object belongs only to the export-side model abstraction and is not exposed as a public Python interface across methods.
Persistent state across prefill/decode is carried by the ExecuTorch state contract and the Ascend state
namespace.

## Design Choices

### Adopted: Export-Side Cache Module with a Generic State Bridge

The Cache is registered as Module State of the export Wrapper. Functionalization promotes mutated
buffers to state inputs/outputs at the method boundary, and the backend then converts them into device-resident
state. This approach preserves the Transformers model forward semantics while keeping the runtime unaware of Qwen3.5's specific
Cache classes.

Prefill and decode use separate export-time states:

- Prefill starts from an empty Linear Attention state.
- Decode uses "previous state exists" as its initial export condition.
- At runtime, the same state namespace hands the state produced by prefill over to decode.

### Not Adopted: Qwen3.5 Cache Branches Built into the Runtime

The C++ runtime does not recognize Qwen3.5's layer types, K/V cache, convolution state, or recurrent
state. The runtime only handles generic state binding, lifecycle, namespace isolation, and device buffers.

### Not Adopted: Flattening the Cache into Application-Level Inputs and Outputs

The Cache is not a State Tuple that callers must maintain manually, nor is it exposed as an ordinary `EValue`. This keeps
the state on the device and avoids moving a large-capacity cache between host and device on every decode round.

## State Contract

### Full Attention State

Each Full Attention layer maintains:

```text
key/value:
  [batch, num_key_value_heads, max_context_length, head_dim]

cumulative_length:
  scalar int64
```

K/V uses a fixed physical capacity and is written into the corresponding region via position updates. `cumulative_length` represents the logical length
of the current request and does not change the physical shape of K/V.

### Linear Attention State

Each Linear Attention layer maintains:

```text
conv_state:
  [batch, conv_dim, linear_conv_kernel_dim]

recurrent_state:
  [batch, linear_num_value_heads,
   linear_key_head_dim, linear_value_head_dim]
```

`has_previous_state` is a control state determined at export time. The prefill graph pins it to the empty-state path, and the decode
graph pins it to the existing-state path, avoiding reliance on Python-level `S == 1` branches in the graph.

### State Owner and Lifecycle

Each Cache Tensor may have only one registered Module owner. Public references from Cache layers may point to that
owner, but the same underlying Tensor must not be registered more than once, to avoid ambiguous
alias relationships produced by AOT Functionalization.

The state lifecycle is as follows:

```text
state namespace created
  -> prefill reads the initial state and writes the next state
  -> current/next swap
  -> decode reads the previous round's state and writes the next state
  -> swap after each round of execution
  -> namespace released when the request ends
```

## Method Boundaries

### Prefill

```text
prefill(
  input_ids:      int64[1, S],
  cache_position: int64[S]
) -> logits[1, S, vocab]
```

The dynamic constraint is:

```text
4 <= S <= max_seq_len - 1
```

Prefill's `cache_position` runs from `0` to `S-1`. Full Attention uses its first position to update the
logical length; Linear Attention processes the entire prompt starting from an empty state.

### Decode

```text
decode(
  input_ids:      int64[1, 1],
  cache_position: int64[1]
) -> logits[1, 1, vocab]
```

Decode always processes exactly one token. The position of the first decode is `S`, and subsequent positions increment one by one. Decode
uses the same state namespace as prefill, but follows the state contract corresponding to the decode graph.

### Invocation Order

```text
prefill(prompt_ids, positions=0..S-1) -> token_0
decode(token_0, position=S)           -> token_1
decode(token_1, position=S+1)         -> token_2
...
```

Prefill and subsequent decodes must use the same namespace within the same execution context. Different requests must use
mutually isolated namespaces.

## Export Strategy

### Prefill Export

Prefill uses a symbolic sequence dimension. Example inputs are used only to determine the exported graph and state capacity; the example
length must not be mistaken for a fixed runtime length. The lower bound of the dynamic dimension is 4, which is the minimum valid sequence
length of the current Linear Attention convolution kernel.

The upper bound uses `max_seq_len - 1`, reserving boundary space for the maximum context position and preventing export constraints from
treating boundary values as out of range.

### Decode Export

Decode uses an input shape of `[1, 1]` and a position shape of `[1]`. The existing-state
path of Linear Attention is determined at export time; the runtime does not rely on Python conditional branches to select the state path.

### PTE Organization

Both methods use the same `acl_soc_name`, with roles `prefill` and `decode` respectively, and use
a consistent delegate identity:

```text
method_name
delegate_index
role
acl_soc_name
```

The runtime must verify that the manifest, the current method, and the delegate identity are consistent, preventing a wrong multi-method
artifact from being loaded into the wrong execution context.

## Export Compatibility and State Conversion

### Functionalization

Mutations of registered Cache Buffers appear in the graph signature as:

```text
InputKind.BUFFER
OutputKind.BUFFER_MUTATION
```

State matching is based on the identity in the ExportedProgram Graph Signature, not on FX names, Tensor shapes,
or post-lowering target names.

### CANN State Bridge

```text
Buffer Mutation
  -> delegated graph state input/output
  -> manifest state_current/state_next
  -> device-resident state namespace
  -> swap current/next on each execution
```

Auto-generated device-only states use:

```text
reset_policy = never
reset_owner = false
host_visibility = device_only
```

When the first execution has no explicit initializer, the runtime initializes the state to all zeros. For dynamic states, if the
input/output size queried from the ACL descriptor is 0, the runtime uses the actual capacity recorded in the state cache,
ensuring that GE execution never receives a zero-byte buffer.

### Dynamic Input Capacity

Device buffers for dynamic ordinary inputs are allocated with the capacity required by the model descriptor, but H2D copies only the data
length of the current actual Tensor. This satisfies GE's requirements for dynamic binding capacity while also avoiding reading
data beyond the actual input from the host.

## Data Flow

Export phase:

```text
Qwen3.5 text-only checkpoint
  -> Cache Module combined with the model forward
  -> torch.export(prefill/decode)
  -> ExecuTorch edge transform/decomposition
  -> Ascend partitioning
  -> GE offline compilation
  -> One PTE + one shared PTD
```

Runtime phase:

```text
prefill input_ids/cache_position
  -> state namespace resolution and initial state
  -> GE execute
  -> logits returned to host, Cache state remains device-resident
  -> decode input_ids/cache_position
  -> execution continues in the same namespace
```

## Artifact Contract

The core artifacts are:

```text
model.pte
ascend.weights.package.ge_aot.ptd
```

The PTE contains both the `prefill` and `decode` methods. The PTD stores shared weight data; duplicate weight copies must not be
generated for the two methods. The device SoC, method identity, state contract, and artifact manifest must
be consistent with one another.

## Implementation Scope

The current design covers:

1. text-only Qwen3.5;
2. Hybrid Cache of Full Attention and Linear Attention;
3. batch size of 1;
4. dynamic prefill sequence length;
5. single-token decode;
6. device-resident state and namespace isolation;
7. PTE and PTD shared by prefill/decode.

Currently not covered:

1. vision or multimodal inputs;
2. batch size greater than 1;
3. chunked decode with a decode sequence length greater than 1;
4. dynamically expanding physical Cache capacity;
5. MMLU evaluation adaptation and accuracy conclusions.

## Testing and Acceptance

### Structural Acceptance

At a minimum, the following should be verified:

1. Cache layer types are consistent with `layer_types`;
2. Each Cache Tensor has only one Module owner;
3. The Graph Signatures of prefill and decode retain the required state mutations;
4. The dynamic range of prefill is `[4, max_seq_len - 1]`;
5. Decode input shapes are `[1, 1]` and `[1]`;
6. The manifest identities of both methods are consistent with the compile spec;
7. The PTE references only one shared PTD weight set.

### Runtime Acceptance

The following should be covered:

1. Prefill lengths of 4, 8, 32, and 128;
2. Consecutive decodes within the same namespace;
3. State isolation between different namespaces;
4. Buffer fallback when the ACL size of a dynamic state is 0;
5. File and namespace isolation when different requests generate inputs concurrently;
6. Output logits are valid and generated tokens are parseable;
7. After the MMLU evaluation flow is completed, add loglikelihood consistency acceptance.

## Risks and Known Limitations

### Dynamic Shape Boundaries

A sequence length below 4 does not satisfy the constraints of the current Linear Attention convolution kernel. Reaching the physical
`max_seq_len` boundary may also trigger position update and export shape constraints, so the current design sets the dynamic upper bound to
`max_seq_len - 1`.

### State Capacity

The physical Cache capacity is fixed by the export configuration and grows with the number of layers, number of heads, head dimension, and maximum
context length. Success with a small-capacity artifact cannot be extrapolated to success with longer contexts.

### Multi-Method Identity

Prefill and decode must use the correct method/delegate identity. Any inconsistency among the manifest, compile
spec, or runtime method should cause loading to be rejected rather than attempting execution.

### Evaluation Semantics

The current decode path validates generation functionality and token output, which is not equivalent to MMLU option loglikelihood
evaluation. MMLU requires separately defined protocols for option scoring, request isolation, and result aggregation.

## Differences from Gemma4

| Feature | Qwen3.5 | Gemma4 |
| --- | --- | --- |
| Cache structure | Full Attention + Linear Attention | DynamicCache + Shared KV |
| Linear state | convolution state + recurrent state | None |
| Shared attention | None | Yes |
| Prefill | Dynamic, `4 <= S <= max_seq_len-1` | Defined separately by the Gemma4 solution |
| Decode | Fixed single token | Fixed single token |
| State form | Cache Module mutation, generic runtime bridge | Explicit static-state adapter |
| Physical Cache | Fixed capacity at export time | Fixed capacity at export time |
| MMLU | Not yet completed | Accepted separately by the Gemma4 solution |

Qwen3.5 does not need to introduce Gemma4's shared attention state, explicit state spec, or adapter-specific
logic into the runtime. What the two share is the generic ExecuTorch state bridge, PTE/PTD organization, and
state namespace lifecycle.
