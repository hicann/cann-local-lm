---
name: npu-op-elimination
description: |
  NPU op elimination skill: rewrites NPU-unsupported / inefficient ops in the HF model export graph, making the epair→OM compile pipeline pass and producing clean artifacts.
  Use when:
  - Export graph contains NPU-unsupported ops (ge.MaskedFill / ge.Where / ge.SelectV2 etc.)
  - Export graph contains inefficient ops to replace (BroadcastTo → ConcatV2D, Pow/Neg/Cos/Sin → Gather)
  - OM compile fails with "operator not supported" or a missing op kernel
  - Applying gitee npu_tuned mode (host pre-constructed mask / repeat_kv concat / RoPE pre-cached Gather) to cann_local_lm models
  - Graph rewriting before export for gemma4 / qwen3.5
version: 0.1.0-beta.1
metadata:
  author: assistant
  category: graph-rewrite
  last_updated: 2026-08-10
  demonstrated_on: gemma4 (e2b real config)
  applicable_to: [gemma4, qwen3_5]
allowedTools:
  - read
  - grep
  - glob
  - edit
  - write
  - bash
---

# NPU Op Elimination Skill

English | [简体中文](../../zh/cann_lm_converter/reference_npu_op_elimination.md)

Rewrites NPU-unsupported / inefficient ops in the HF model export graph into NPU-friendly form, making epair→OM compile pass and the op table clean.
Source: gitee `npu_tuned` mode.
First complete application to cann_local_lm gemma4 (e2b real config: 35 layers with 20 KV shared at the end, head_dim=256/global=512, PLE on, sliding=512, kv_max=2048).

## Overview: Three Rewrites

| # | Rewrite | Eliminated op (export graph) | Post-rewrite op | Location |
|---|---|---|---|---|
| 1 | host pre-construct mask + bypass forward | `ge.MaskedFill` (NPU unsupported) | mask as graph input (no in-graph construction) | `export_wrapper.py` |
| 2 | repeat_kv reshape+cat | `BroadcastTo` | `ConcatV2D` | `modeling_*.py` |
| 3 | RoPE pre-cached table + Gather | `Pow`/`Neg`/`Cos`/`Sin` | `Gather` | `modeling_*.py` |

Verification baseline: e2b gemma4 three OM compiles then dump op table confirms MaskedFill=0, BroadcastTo=0 (ConcatV2D=70), Pow/Neg/Cos/Sin=0; numerical maxdiff=0.00e+00.

## When to Apply

- OM compile reports an op unsupported (check CANN ops-info for that op definition)
- ATC dump shows `BroadcastTo` / `Pow` / `Neg` / `Cos` / `Sin` / `MaskedFill` / `Where` / `SelectV2` etc. to eliminate
- Graph cleanup before adapting a new model to cann_local_lm export pipeline

## Rewrite 1: host pre-construct mask (eliminate ge.MaskedFill)

### Symptom
gemma4 `_prepare_causal_mask` constructs attention mask (causal + sliding window) in-graph using `torch.full` + masked_fill. After export, `ge.MaskedFill` appears in the graph. NPU does not support this op (grep CANN ops-info confirms no such op definition); OM compile fails.

### Cause
The mask construction logic (causal triangle + sliding window) contains data-dependent control flow; torch.export can trace it but materializes it as MaskedFill in the graph. NPU op library has no MaskedFill kernel.

### Solution
**Move mask construction to host/pipeline side, pass as graph input; wrapper bypasses `Model.forward` to drive layers directly, statically distributing mask per layer by `layer_types`.** No mask construction in graph → no MaskedFill.

`forward` signature: a single `attention_mask` is split into `mask_full` + `mask_sliding` two graph inputs (host rebuilds per pipeline step):

```python
def forward(self, input_ids=None, mask_full=None, mask_sliding=None,
            position_ids=None, past_key_values=None, new_kv_cache_pos=None,
            embeddings=None, output_pos=None,
            per_layer_token_identity=None, embeddings_scale=None):
    ...
    logits, past_kv_flat = self._drive_layers(
        input_ids, mask_full, mask_sliding, position_ids,
        past_key_values, new_kv_cache_pos, embeddings, per_layer_token_identity)
    if output_pos is not None:
        logits = logits[:, output_pos, :]
    return logits, *past_kv_flat
```

`_drive_layers` layer loop statically distributes mask (no control flow, torch.export friendly):

```python
for i, layer in enumerate(tm.layers):
    layer_type = self._layer_types[i] if i < len(self._layer_types) else "full_attention"
    mask_i = mask_sliding if layer_type == "sliding_attention" else mask_full
    hidden_states = layer(hidden_states, per_layer_input,
                         shared_kv_states=shared_kv_states,
                         position_embeddings=rope_embs[layer_type],
                         attention_mask=mask_i,
                         past_key_values=kv_list,
                         new_kv_cache_pos=new_kv_cache_pos)
```

Host-side mask construction (`get_sample_inputs` calls, not in graph, can use any op):

```python
def _build_mask_full(self, seq_len, kv_max, dtype, new_kv_cache_pos):
    """causal-over-buffer: k<=q → 0, else MIN. q position = new_kv_cache_pos."""
    min_val = torch.finfo(dtype).min
    q = new_kv_cache_pos.to(dtype=dtype).unsqueeze(1)        # [seq, 1]
    k = torch.arange(kv_max, dtype=dtype).unsqueeze(0)       # [1, kv_max]
    mask = torch.where(k <= q, torch.tensor(0.0, dtype=dtype), min_val)
    return mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)

def _build_mask_sliding(self, seq_len, kv_max, dtype, new_kv_cache_pos):
    """causal + sliding window: k<=q AND q-k<=W-1 → 0, else MIN.
    Degrades to mask_full when kv_len <= sliding_window."""
    w = self._sliding_window
    if not w or w <= 0 or kv_max <= w:
        return self._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
    min_val = torch.finfo(dtype).min
    q = new_kv_cache_pos.to(dtype=dtype).unsqueeze(1)
    k = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
    diff = q - k
    attend = (diff >= 0) & (diff <= w - 1)
    mask = torch.where(attend, torch.tensor(0.0, dtype=dtype), min_val)
    return mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)
```

### Key Points
- mask not in graph → host can use `torch.where`/`torch.arange` (these would become ops to eliminate if in graph)
- per-layer distribution is **static** (indexed by `config.layer_types`, no if/else control flow) → torch.export traceable
- eager generate path unaffected (still goes `model.forward` with `_prepare_causal_mask`'s None/2D branch); only export path changed
- When PLE on, `per_layer_token_identity` similarly host-precomputed (embed_tokens_per_layer lookup externalized)

### File
- `cann_local_lm/models/gemma4/export_wrapper.py` (`Gemma4ExportWrapper.forward` / `_drive_layers` / `_build_mask_full` / `_build_mask_sliding`)

## Rewrite 2: repeat_kv reshape+cat (BroadcastTo → ConcatV2D)

### Symptom
GQA (grouped query attention)'s `repeat_kv` uses `expand` to copy KV heads. `BroadcastTo` appears in the export graph. This op is inefficient on NPU / may not be fused.

### Cause
`hidden_states.expand(batch, num_kv_heads*n_rep, slen, head_dim)` materializes as BroadcastTo in the export graph.

### Solution
**Replace expand with reshape+cat+view (npu_tuned approach), so the export graph has ConcatV2D instead of BroadcastTo.**

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (b, kv_heads, s, d) -> (b, heads, s, d).
    Use reshape+cat+view to replace expand, so export graph has ConcatV2D not BroadcastTo."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch * num_kv_heads, 1, slen, head_dim)
    hidden_states = torch.cat([hidden_states] * n_rep, dim=1)
    return hidden_states.view(batch, num_kv_heads * n_rep, slen, head_dim)
```

### Key Points
- **Head order must be kv_head-major**: first reshape to `[b*kv, 1, s, d]` (kv merged into dim0, dim1 left size-1 for cat), then `cat([x]*n_rep, dim=1)` materializes n_rep copies, finally view back to `[b, kv*n_rep, s, d]`. kv outer, n_rep inner, elementwise identical to expand+reshape
- **naive `cat([x]*n_rep, dim=1)` is rep-major which misaligns with q, unusable** — must reshape first to merge kv into dim0
- Zero numerical difference (maxdiff=0.00e+00), e2b OM dump measured ConcatV2D=70 (=35 layers × 2 (k,v)), BroadcastTo=0

### File
- `cann_local_lm/models/gemma4/modeling_gemma4.py` `repeat_kv()` (qwen3.5 same-named function same change)

## Rewrite 3: RoPE pre-cached table + Gather (Pow/Neg/Cos/Sin → Gather)

### Symptom
RoPE's `forward` computes cos/sin dynamically in-graph: `inv_freq = pow(base, -exponents)` → `freqs = outer(t, inv_freq)` → `emb = cat(freqs, freqs)` → `cos(emb)` / `sin(emb)`. The export graph shows a bunch of dynamic ops `Pow`/`Neg`/`Cos`/`Sin`.

### Cause
The cos/sin table is a **constant** (depends only on position and rope_theta, not on input), but the original implementation recomputes every forward, materializing as Pow/Neg/Cos/Sin in the graph.

### Solution
**`__init__` pre-caches cos/sin table (`register_buffer persistent=False`), `forward` only Gathers (`cos_cached[position_ids]`).** All init-computed (inv_freq + outer + cat + cos/sin), not in graph.

`__init__` pre-cache (per layer_type):

```python
for layer_type in self.layer_types:
    ...
    # Pre-cache cos/sin table (init computes: inv_freq + outer + cat + cos/sin, all in init, not in graph)
    # forward only Gathers (cos_cached[position_ids]), eliminating Pow/Neg/Cos/Sin dynamic ops
    inv_freq = self._compute_inv_freq(layer_type, dev, torch.float32)
    t = torch.arange(max_seq, device=dev, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer(f"cos_cached_{layer_type}", emb.cos(), persistent=False)
    self.register_buffer(f"sin_cached_{layer_type}", emb.sin(), persistent=False)
```

`forward` Gather:

```python
@torch.no_grad()
def forward(self, x, position_ids, layer_type="full_attention"):
    # Gather over pre-cached constant table. No inv_freq/cos/sin/pow/neg dynamic ops.
    cos_cached = getattr(self, f"cos_cached_{layer_type}")  # [max_seq, dim] constant
    sin_cached = getattr(self, f"sin_cached_{layer_type}")
    cos = cos_cached[position_ids]   # -> [batch, seq, dim]
    sin = sin_cached[position_ids]
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### Key Points
- `persistent=False` → not in state_dict (export weights exclude, avoid redundancy); but register_buffer lets it follow the model
- per layer_type caches one each (`cos_cached_full_attention` / `cos_cached_sliding_attention`), because gemma4 different layer_type has different head_dim / rope_theta
- Zero numerical difference (maxdiff=0.00e+00), e2b OM dump measured Pow/Neg/Cos/Sin=0
- `_compute_inv_freq` still called in init (proportional / default two rope_type branches), just not in graph

### File
- `cann_local_lm/models/gemma4/modeling_gemma4.py` `Gemma4TextRotaryEmbedding`

## Verification Method (must do after rewrite)

### 1. Numerical accuracy verification (eager)
wrapper single-step forward vs original model.forward, compare logits maxdiff:
- Expect maxdiff=0.00e+00 (all three rewrites are numerically identical transforms)
- Script: `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_bypass.py`

### 2. Op table verification (OM dump)
After epair → OM compile, dump op table to confirm target ops eliminated:

```bash
# ATC dump (add --dump=... at compile time)
# View atc_dump_*/ge_*.json's op_list, count op occurrences
python3 -c "
import json
ops = json.load(open('atc_dump.json'))['op_list']
from collections import Counter
c = Counter(o['type'] for o in ops)
for op in ['MaskedFill','BroadcastTo','ConcatV2D','Pow','Neg','Cos','Sin','Gather','Where','SelectV2']:
    print(f'{op:20s} {c.get(op,0)}')
"
```

Expected:
| Op | Before rewrite | After rewrite |
|---|---|---|
| MaskedFill | >0 | 0 |
| BroadcastTo | >0 | 0 |
| ConcatV2D | few | 70 (e2b, =35 layers × 2) |
| Pow/Neg/Cos/Sin | each >0 | each 0 |
| Gather | few | increased (rope) |

### 3. Disable AUB verification (optional)
If `AutomaticBufferFusionOp` interferes, use fusion_switch_file to disable AutomaticUbFusion:
```bash
# fusion_autoub_off.cfg content: {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}
export GE_FUSION_SWITCH_FILE=/path/to/fusion_autoub_off.cfg
```
After compile, dump to confirm AutomaticBufferFusionOp=0.

## Environment Preparation

### converter (CPU-only, export epair)
```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_HOME=$TORCH_DELEGATE_HOME
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
# venv: $TORCH_DELEGATE_HOME/.venv (torch 2.9.0+cpu + torchair source compile NO_ASCEND_SDK=1)
```

### NPU (OM compile + inference verification)
```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
# System Python 3.11.4 (torch 2.7.1+cpu, torch_npu 2.7.1.post4, Ascend910 ×2)
```

## Debugging Experience

### 1. host mask sliding degradation
In `_build_mask_sliding`, when `kv_max <= sliding_window` it must degrade to `mask_full` (causal is already implicit in the window), otherwise sliding layers will attend to tokens they shouldn't.

### 2. repeat_kv head order error
naive `torch.cat([x]*n_rep, dim=1)` without reshape yields rep-major (all n_rep copies contiguous), misaligning with query's kv_head-major → numerical explosion. Must `reshape(b*kv, 1, s, d)` first to merge kv into dim0.

### 3. RoPE register_buffer persistent
`persistent=False` is mandatory, otherwise cos/sin table enters state_dict, epair weights contain redundant constant table (max_seq × dim float32 table, tens of MB).

### 4. bypass forward must not break eager generate
wrapper's `_drive_layers` only goes the export path; `model.forward` (with `_prepare_causal_mask`'s None/2D branch) must be preserved for eager generate. The two paths are independent and do not affect each other.

### 5. PLE lookup externalization
gemma4 PLE's `embed_tokens_per_layer` lookup if left in-graph is a large embedding gather (vocab × num_layers × hidden). Host precomputes `per_layer_token_identity` to externalize, `detach_externalized_lookups()` deletes the embedding module, leaving only project in-graph. qwen3.5 has no PLE, skip.

## Cross-Model Applicability

| Rewrite | gemma4 | qwen3.5 | Notes |
|---|:---:|:---:|---|
| 1. host mask | ✓ Applied | Pending | Only models with sliding window need mask_sliding; pure causal uses mask_full |
| 2. repeat_kv concat | ✓ Applied | Pending | Universal for all GQA models (n_rep>1) |
| 3. RoPE pre-cache | ✓ Applied | Pending | Universal for all implementations that dynamically compute cos/sin |

> "Pending" = the rewrite logic is universal; when adapting a new model, align its `repeat_kv` / RoPE / export_wrapper with the code snippets here.

## Related Files

| File | Purpose |
|---|---|
| `cann_local_lm/models/gemma4/export_wrapper.py` | Rewrite 1 (bypass forward + host mask + PLE externalization) |
| `cann_local_lm/models/gemma4/modeling_gemma4.py` `repeat_kv()` | Rewrite 2 (reshape+cat) |
| `cann_local_lm/models/gemma4/modeling_gemma4.py` `Gemma4TextRotaryEmbedding` | Rewrite 3 (pre-cache + Gather) |
| `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_bypass.py` | Numerical verification (wrapper vs original) |
| `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_export.py` | Export verification |
| `examples/test_converter/export_llm_model/om_npu_verify/` | OM compile + inference tools |
| `docs/cann_lm_converter/README.md` | converter full-flow design |

## Reference Commit
- `a429d4a` feat(gemma4): bypass-forward wrapper + host mask + repeat_kv concat + rope precompute
