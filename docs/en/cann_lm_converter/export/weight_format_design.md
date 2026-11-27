# Weight Format Conversion Design Document

English | [简体中文](../../../zh/cann_lm_converter/export/weight_format_design.md)

> epair export converts weights from ND row-major to fractal block-major on disk (fp16 MatMul → FractNZ; quantized QuantBatchMatmulV4 weight_q → FractZ), avoiding NPU runtime/intra-kernel transdata.
>
> Covers `cann_local_lm/converter/export/weight_format/` (new package) + config file `weight_format_rules.json` + epair_utils integration + C++ format mapping.
>
> Related: export main flow see [export_design.md](./export_design.md), quantization see [quant_design.md](../quant/quant_design.md).
>
> Status: Stage 1/2 landed and verified (A3 NZ max_diff=0; KirinX90 Z board cos=0.999912); master switch deprecation design finalized (rule-level enable: MatMul off / QuantBatchMatmulV4 on) | Last verified: 2026-10-20

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Config File](#config-file)
- [Environment Variable Master Switch](#environment-variable-master-switch)
- [Master Switch Deprecation Design (Finalized 2026-10-20)](#master-switch-deprecation-design-finalized-2026-10-20)
- [Format Conversion Functions](#format-conversion-functions)
- [Packer Integration](#packer-integration)
- [ep_json layout annotation](#ep_json-layout-annotation)
- [ep_json Annotation Sync Mechanism (_sync_* added)](#ep_json-annotation-sync-mechanism-_sync_-added)
- [C++ format mapping](#c-format-mapping)
- [NZ shape semantics](#nz-shape-semantics)
- [Stage division](#stage-division)
- [Pending Verification](#pending-verification)
- [Follow-up Migration](#follow-up-migration)

---

<a id="overview"></a>
## Overview

When `cann_local_lm/converter/export/` writes weights to disk, it defaults to ND (row-major) for `weights_*.bin`. Ascend cube compute natively works in NZ (FractNZ, 16×16 block-major) layout; ND on disk introduces transdata at OM compile/runtime. This design converts weight tensors to NZ physical bytes before writing to disk, and synchronously annotates ep_json's `layout` field and C++'s `ge::Format`, letting weight Const nodes enter the GE graph directly as `FORMAT_FRACTAL_NZ`.

### Capability Boundary

- **Does (Stage 1, implemented)**: fp16/bf16 `MatMul` class Linear weights (`*_proj`) → NZ (standard 16×16 block conversion); A3 measured max_diff=0 (see [Pending Verification](#pending-verification) L2/L3)
- **Does (Stage 2, implemented)**: quantized `QuantBatchMatmulV4` weight_q → Z (`pack_int4_nz`, int4 nibble-aware FractZ `[K//64, N//16, 16, 8]`); KirinX90 board measured cos=0.999912 (``op_npu_verify/GUIDE_kirinx90_board.md``)
- **Keep ND**: embed_tokens / lm_head / norm / bias / KV cache / quantized weight_scale (int64 bit pattern, not a layout object, see [ep_json Annotation Sync Mechanism](#ep_json-annotation-sync-mechanism-_sync_-added))

<a id="directory-structure"></a>
## Directory Structure

```
cann_local_lm/converter/export/
├── weight_format/                 # ← New package: format conversion functions + rule parsing
│   ├── __init__.py                # re-export nd2nz / pack_int4_nz / resolve_format / load_rules
│   ├── layout.py              # nd2nz() (Stage 1, FractNZ) + pack_int4_nz() (Stage 2, FractZ, implemented)
│   └── rules.py                   # Config loading + resolve_format(name, is_quantized)
├── weight_format_rules.json       # ← shipped default config (export/ root)
├── epair_utils.py                 # Integration: _pack_weights_to_dir calls resolve_format + nd2nz
└── ...（existing files）
```

> **Responsibility Boundary**: `layout.py` contains `nd2nz` (Stage 1) and `pack_int4_nz` (Stage 2, implemented). `transdata_scale` (fp32→int64 bit pattern) semantically also belongs to "format conversion", currently in `cann_local_lm/converter/export/quant_swap/materialize.py` (`swap.py` at `cann_local_lm/converter/swap.py`, absolute imports it); it is **materialization bit-pattern packing** (not layout transformation), **not merged into** `weight_format/layout.py`, stays in `materialize.py` with `pack_int4` as the materialization layer (`epair_utils` imports from `quant_swap.materialize`).

<a id="config-file"></a>
## Config File

shipped default `cann_local_lm/converter/export/weight_format_rules.json`; converter always loads this default config (no CLI override exposed). When config is missing → all ND (equivalent to current state, NZ not enabled). There is also an environment variable master switch `CANN_LM_ENABLE_WEIGHT_TRANS` default off (see [Environment Variable Master Switch](#environment-variable-master-switch)) — even if config matches NZ, env not on means no conversion.

### schema

**Current shipped reality** ([`cann_local_lm/converter/export/weight_format_rules.json`](../../../../cann_local_lm/converter/export/weight_format_rules.json)):

```json
{
  "default": {"format": "ND"},
  "rules": [
    {
      "name_pattern": "*_proj.weight",
      "op_type": "MatMul",
      "format": "NZ",
      "block": 16,
      "pad": true,
      "pack": "nd2nz"
    },
    {
      "name_pattern": "*",
      "op_type": "QuantBatchMatmulV4",
      "format": "Z",
      "pack": "pack_int4_nz"
    }
  ]
}
```

<a id="wildcard-trap-and-code-guard-recorded-2026-10-20"></a>
### wildcard trap and code guard (recorded 2026-10-20)

The second rule above `"name_pattern": "*"` is **too broad**: it matches all 2D tensors in state_dict, including lifted tensor constants (rotary cos/sin cache etc.). During quantized export they would be wrongly Z-packed, causing GE format propagation infershape failure (Reshape_11 compile error). Currently held by a code-level guard — `_transform_weight_tensor` (`epair_utils.py:273`) additionally requires `name.endswith("weight_q")` **before** calling `resolve_format`.

Problems:

1. **Config lies**: rules.json claims "all QuantBatchMatmulV4 2D tensors are Z", but actually only `*.weight_q` is converted; readers of the config get wrong expectations.
2. **Two packers semantically inconsistent (existing hazard)**: v1 `_pack_weights_to_dir` (`epair_utils.py:748`) **lacks** this guard — if a quantized model goes through v1 path, `weight_scale` (2D int64) would be matched by `"*"` rule and sent directly to `pack_int4_nz`.

The master switch deprecation migration (see [Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20)) will tighten the pattern to `"*.weight_q"` and make both packers reuse the same transform, eliminating both points.

### Field Semantics

| Field | Meaning |
|------|------|
| `name_pattern` | Weight name glob (`fnmatch` semantics), e.g. `*_proj.weight` / `*.weight_q` |
| `op_type` | **Dispatch key**: GE op type consuming this weight. `MatMul` (fp16 path) / `QuantBatchMatmulV4` (quantized path) |
| `format` | Target format: `ND` / `NZ` (FractNZ) / `Z` (FractZ, quantized int4pack-specific) |
| `block` | NZ block size (default 16; Z path fractal constants hardcoded in `pack_int4_nz`, does not use this field) |
| `pad` | Whether to zero-pad non-block-multiple dimensions |
| `pack` | Transform function name: `nd2nz` (standard 16×16) / `pack_int4_nz` (int4 nibble-aware FractZ) |
| `enable` | **Rule-level switch** (added in 2026-10-20 migration design, current code does not read it yet): `false` → rule treated as nonexistent (continue matching subsequent rules / fall to default); omitted = `true`. Implementation see [Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20) |

> Currently the entire rule set is still gated by env master switch ([Environment Variable Master Switch](#environment-variable-master-switch)); migration design see [Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20) — after landing this table becomes the sole authority.

<a id="environment-variable-master-switch"></a>
## Environment Variable Master Switch

> **⚠️ The switch described in this section has been superseded by [Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20) (2026-10-20)**; text below retained as historical design and incident root-cause record.

NZ conversion is **off** by default (equivalent to current state, all ND, zero byte change). At export time set environment variable `CANN_LM_ENABLE_WEIGHT_TRANS=1` to enable the entire `weight_format` flow: `resolve_format` + `nd2nz` (write NZ bytes) + `weight_info.json` entry's `format` field (mark `"NZ"`). If unset or not `1` → packer skips `resolve_format` (incl. `is_quantized` probing), all weights ND + `format` all `"ND"`, C++ `CreateConstNode` goes `MapPt2Layout` (current ND path) — code dormant, zero risk.

**Design intent**: NZ is a performance optimization (avoid runtime transdata), not a correctness requirement (current ND quantized OM works, see [Stage Division](#stage-division)). Stage 1's MatMul NZ input acceptance unverified ([Pending Verification](#pending-verification)), so default off, env on, facilitating code-first dormant merge; after verification passes, decide whether to default on or deprecate switch.

**Scope**: Python export side (`epair_utils._pack_weights_to_dir` writes NZ bytes + `weight_info.json` marks `format`) + C++ loading side (`ge_graph_builder.cc:CreateConstNode` reads `weight_info.json`'s `format` field for dispatch, see [C++ format mapping](#c-format-mapping)). env has no C++ read — when env off `weight_info.format` all `"ND"`, C++ goes `MapPt2Layout` (ND); when env on `format` is `"NZ"`, C++ goes `MapGeLayoutString("NZ")` (TODO).

**Naming convention**: `CANN_LM_ENABLE_WEIGHT_TRANS` (`1` enable, other/unset = off). Follows `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN`'s `ENABLE` boolean switch style; `CANN_LM_` prefix because it only acts on cann_local_lm converter export side. Uses `TRANS` (not `NZ`) because master switch gates the entire `weight_format` flow — Stage 1 `nd2nz` + Stage 2 `pack_int4_nz` + future other format conversions, not just NZ.

---

<a id="master-switch-deprecation-design-finalized-2026-10-20"></a>
## Master Switch Deprecation Design (Finalized 2026-10-20)

### Motivation

The master switch's design intent was "dormant merge, decide after verification passes" (see design intent above). As of 2026-10-20, verification has all passed:

- Stage 1 (MatMul ND→NZ): A3 full-network L2/L3, NPU output NZ OM == ND OM, max_diff=0
- Stage 2 (QuantBatchMatmulV4 weight_q Z-pack): KirinX90 board q_proj single-op cos=0.999912 (omc→MRT→golden full chain)

The reason to keep the env switch has disappeared, and its cost has been empirically proven — **asymmetric gating incident** (see [ep_json Annotation Sync Mechanism](#ep_json-annotation-sync-mechanism-_sync_-added)): packing executes unconditionally, annotation sync is env-gated; env off produces self-contradictory epair, error delayed to compile-time size mismatch, two hops from root cause. Control should be handed to `weight_format_rules.json` — config is versionable, reviewable, and the "clear rules to all ND" escape hatch already exists (config missing → all ND = current state).

### Gating Surface Inventory: 4 classes of actions env actually gates

Before removing env, must first unpack what it gates — **not every class should go into rules.json**:

| # | Action | Location | Current | Destination |
|---|---|---|---|---|
| ① | weight_q Z-pack (`pack_int4_nz`) and fp16 `*_proj` NZ (`nd2nz`) | `epair_utils.py:273` `_transform_weight_tensor` | env gate + code `endswith("weight_q")` fallback | **rules.json** (layout selection is config's job) |
| ② | ep_json annotation sync (`_sync_input_desc_layout` / `_sync_weight_scale_dtype` / converter-side `x2.desc.layout="Z"`) | `epair_utils.py:210/:483`, `ge_converters/_quant_matmul_converter.py:103` | env gate | **Not in config** — unconditionally follows ①; packer hits Z/NZ then sync |
| ③ | weight_scale transdata_scale (fp32→int64 bit pattern) | `epair_utils.py:260` | **Already unconditional** | **Not in config** — kernel semantic contract (arch31 s8s4 requires x2_scale uint64 ND), not optional |
| ④ | `is_quantized` probing + rules loading | `epair_utils.py:303` `_load_weight_trans_rules` | env gate | Merged with ①: rules always loaded, probing always done (json missing → all ND fallback already exists) |

**②③ not in config principle**: rules.json field semantics locked to "layout/format selection" layer. dtype re-annotation, materialization bit pattern are "necessary accompanying actions after hitting a format" — expressing mandatory actions as optional actions is exactly the last incident's pattern. If weight_scale handling must be made explicit, at most add a `"materialize": "transdata_scale` declarative field to the QuantBatchMatmulV4 rule (always effective, self-documenting only), not a switch field.

### rules.json Target Form (revised 2026-10-20: per-rule independent `enable`, MatMul off / QuantBatchMatmulV4 on)

```json
{
  "default": {"format": "ND"},
  "rules": [
    {
      "name_pattern": "*_proj.weight",
      "op_type": "MatMul",
      "format": "NZ",
      "block": 16,
      "pad": true,
      "pack": "nd2nz",
      "enable": false
    },
    {
      "name_pattern": "*.weight_q",
      "op_type": "QuantBatchMatmulV4",
      "format": "Z",
      "pack": "pack_int4_nz",
      "enable": true
    }
  ]
}
```

Four changes:

1. **Added per-rule `enable` field** (replaces env master switch's granularity unit): `enable: false` → rule treated as nonexistent (continue matching subsequent rules, eventually fall to default ND); omitted = `true` (listing in rules means intent to enable). Control granularity refined from "one env variable flips all" to "each conversion path declared separately" — this is the complete form of "control into config".
2. **MatMul rule default off (no conversion)**: fp16 NZ is performance optimization not correctness requirement (`ND_ND` kernel variant is correct anyway, see [Stage Division](#stage-division)), and enabling changes **all** fp16 Linear exported graph structures (C++ side needs Transpose skip + `transpose_x2` machinery) — default off makes migration **zero behavior change, zero regression surface** for fp16 export; when A3 L3 performance needed, explicitly flip `enable: true`. Quantized export is dispatched by `op_type` and only goes QuantBatchMatmulV4 rule set; this rule only affects fp16 path (no effect on current q_proj quantization flow).
3. **QuantBatchMatmulV4 rule default on (must convert)**: kirinx90 s8s4 kernel contract requires x2 = FractZ int4pack, x2_scale = uint64 — **correctness-required** path (if this rule doesn't take effect, quantized export directly produces broken epair), also the only active consumer of this capability currently.
4. pattern tightened from `"*"` to `"*.weight_q"` — wildcard false-positive on rotary cos/sin moves from code hack back to config-level explicit expression (see [wildcard trap](#wildcard-trap-and-code-guard-recorded-2026-10-20)).

Escape hatch semantics unchanged: rules.json missing or cleared `rules` → all ND. **`enable` must land in same PR as code support (`resolve_format` reads this field)** — currently `resolve_format` ignores unknown fields; changing json before code would let `enable: false` silently not take effect (config lies, replaying "data and annotation not co-located" incident pattern).

### Dual-Source Consistency

After removing env, one glue point must be caught: packer (writes Z bytes + weight_info format) and converter (`_quant_matmul_converter` sets `x2.desc.layout="Z"`) are **two independent decisions**, currently glued by "same env value". Post-migration both must **read the same rules** — converter side also `load_rules()` (shipped path, read fresh each time, no cache), decide whether to set layout based on `resolve_format(name, is_quantized).format == "Z"`, eliminating two-source drift.

### Change List

All `os.environ.get("CANN_LM_ENABLE_WEIGHT_TRANS")` read points (8 on python side; C++ side has no env read, see scope above):

| File:line | Change |
|---|---|
| `epair_utils.py:303` `_load_weight_trans_rules` | Remove env check; rules always loaded (missing → all ND); `is_quantized` always probed |
| `weight_format/rules.py` `resolve_format` | Read `enable` field: `false` rule skipped (treated as nonexistent, continue scanning), omitted=true — the landing point of "config and behavior same source" |
| `weight_format_rules.json` | Update to "rules.json target form": MatMul rule `enable: false` (no conversion), QuantBatchMatmulV4 rule `*.weight_q` + `enable: true` — **same PR** as rules.py support (prevent config before code) |
| `epair_utils.py:210` / `:483` (consolidated/PD two outlets) | `_sync_input_desc_layout` + `_sync_weight_scale_dtype` calls **unconditional** (as soon as save_weights) — this item is historical debt [P1 fix], auto-resolved after switch removal |
| `epair_utils.py:273` (v2 transform) | `trans_enabled` parameter deleted; guard fully moved into name_pattern (`*.weight_q`) |
| `epair_utils.py:725-748` (v1 `_pack_weights_to_dir`) | No longer inline transform separately, **reuse `_transform_weight_tensor`** — incidentally fixes v1's missing guard `"*"` false-positive bug (see wildcard trap point 2) |
| `ge_converters/_quant_matmul_converter.py:103` | Remove env; when `is_s8s4` and `resolve_format(...).format=="Z"` set `x2.desc.layout` (same-source decision with packer) |
| `tests/python/converter/test_weight_format.py` | `env-off` two cases → "empty rules json" cases; `env-on` cases remove monkeypatch (ND/NZ byte assertions unchanged); add `enable: false` rule skip cases (incl. subsequent rule matching order semantics) |
| Docs | This doc (this section + [ep_json Annotation Sync Mechanism](#ep_json-annotation-sync-mechanism-_sync_-added)), `TODO_notes.md` env master switch note, `op_npu_verify/GUIDE_kirinx90_board.md` §1.2 and quick troubleshooting #2 (mark "fixed upstream"); `export_design.md` checked has no env mention, no change needed |

### Regression Acceptance

Per `regression anchor`, criteria from strong to weak:

1. **fp16 zero behavior change** (MatMul `enable:false`): non-quantized model new-chain export vs pre-migration env=0 old-chain — weights bin + ep_json **bit-identical** (new chain with MatMul rule off → all ND, fully equivalent to "no conversion").
2. **Single-op bit reproduction** (QuantBatchMatmulV4 `enable:true`): q_proj re-export without env (no `CANN_LM_ENABLE_WEIGHT_TRANS`) → five artifacts md5 **bit-identical** to 2026-10-20 env=1 baseline (omc `65be769d…`, SubGraph_0.weight `2366b036…`, input `4f6b5f3…`) → board run output_0 md5 reproduces `b1344d8e…` (no cos recompute needed).
3. **e2e full-model reconciliation**: same input dual-path (env=1 old chain vs no-env new chain) **quantized model** export, weights bin + ep_json diff empty.
4. **Unit tests**: `pytest tests/python/converter/test_weight_format.py` all green ("empty rules → all ND" escape hatch, `enable:false` skip two families of cases).

### Decision Point Record

- **Rule-level switch (revised 2026-10-20)**: per-rule independent `enable`, replacing initial "global default on" stance — **MatMul (fp16 NZ) default off** (performance optimization not required, enabling changes all fp16 graph structure, explicit opt-in), **QuantBatchMatmulV4 (quantized Z) default on** (kirinx90 kernel correctness contract, must convert). Escape hatch = clear rules; rule-level granularity is the complete replacement form after env master switch deprecation.
- **Deprecation timing**: migration PR removes all read points at once, no dual-track env/config coexistence period (dual-track = breeding ground for two-end drift, the starting point for second opportunistic gating). Old env silently ignored (no env read points), noted in CHANGELOG/release notes.

<a id="format-conversion-functions"></a>
## Format Conversion Functions

`weight_format/layout.py`:

### nd2nz (Stage 1)

```python
def nd2nz(tensor: torch.Tensor, block: int = 16, pad: bool = True) -> torch.Tensor:
    """[M, K] ND row-major → FractNZ physical block [ceil(M/b), ceil(K/b), b, b] block-major.
    Non-b-multiple dims internally zero-padded (partial block); returns same dtype contiguous for tobytes()."""
```

Logic: `M, K = shape` → `m, k = ceil(M/b), ceil(K/b)` → `F.pad(..., (0, k*b-K, 0, m*b-M))` → `reshape(m, b, k, b)` → `permute(0, 2, 1, 3)` → `.contiguous()`. bf16 first int16 view (follows `epair_utils` existing convention).

### pack_int4_nz (Stage 2, implemented)

int4pack int32 `[K, N//8]` (1 int32 packs 8 int4) → FractZ physical block `[K//64, N//16, 16, 8]`. **Cannot reuse `nd2nz`** (int4 values cross int32 byte boundaries, element-wise 16×16 conversion would break nibble pairing and int32 carrier structure). Padding digested internally (nibble plane domain padded); fractal constants `64/16/8` are consistent with `epair_utils._sync_input_desc_layout`'s `_B_FRAC_K/_B_FRAC_N/_INT4_PER_INT32` (see [ep_json Annotation Sync Mechanism](#ep_json-annotation-sync-mechanism-_sync_-added) — two sources must be same-origin). Byte layout empirically verified on KirinX90 board (cos=0.999912, see `GUIDE_kirinx90_board.md` regression anchor).

<a id="packer-integration"></a>
## Packer Integration

Among the three packers in `epair_utils` (v1 `_pack_weights_to_dir` / v2 `_pack_weights_v2` / PD `_pack_weights_pd_dedup`), **v2 and PD execute conversion via shared `_transform_weight_tensor` (:246)**; core structure: **dtype view and layout conversion are orthogonal steps** (bfloat16 first `view(int16)` doesn't change bytes, only dtype interpretation; layout then reorders physical blocks), not stuffed into same `elif` chain — otherwise bfloat16 weights always hit first branch, layout branch unreachable. v1 is currently **independent inline duplicate implementation** (legacy, guard semantics inconsistent with v2 — see wildcard trap), post-migration unified reuse of `_transform_weight_tensor`:

```python
# _transform_weight_tensor current skeleton (epair_utils.py:246; final form of migration target see "Master Switch Deprecation Design")
if raw.dtype == torch.bfloat16:
    raw = raw.view(torch.int16)
elif "weight_scale" in name and raw.dtype == torch.float32:
    raw = transdata_scale(raw)      # Unconditional: kernel semantic contract
entry_format = "ND"
if trans_enabled and raw.dim() == 2 and "weight_scale" not in name and name.endswith("weight_q"):
    rule = resolve_format(name, is_quantized, rules=rules)
    if rule.get("format") == "NZ":
        raw = nd2nz(raw, block=rule["block"], pad=rule["pad"]); entry_format = "NZ"
    elif rule.get("format") == "Z":
        raw = pack_int4_nz(raw); entry_format = "Z"
# Note: current shipped rules second rule name_pattern="*", held by in-function endswith("weight_q") fallback to prevent false-positives
# (wildcard trap, see "Config File" section); post-migration tighten to "*.weight_q", guard moves to config layer.
```

### op_type dispatch (is_quantized probing)

A single export is either pure fp16 or pure quantized (when quantized all Linear become QuantLinear, no mixing). Packer probes whether state_dict contains `*_weight_scale` tensor → judges `is_quantized` → selects `MatMul` or `QuantBatchMatmulV4` rule set. **No per-weight graph traversal to find consuming op** (avoids adding graph traversal burden to packer). prefill-decode packs once shared, NZ/Z converted once (naturally correct).

> The "skip resolve_format + is_quantized" behavior when env off is removed along with master switch deprecation (see [Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20) action ④) — rules always loaded, probing always done; when json missing `resolve_format` goes default ND, behavior degrades equivalently with no env-dependent two-state.

<a id="ep_json-layout-annotation"></a>
## ep_json layout annotation

> **Mechanism correction (discovered 2026-09-05): original design assumed NZ modifies ep_json via `inline_weight_layout`, this path is blocked**:
> - `tensor_values[p_*].layout` is a **pt2 enum** (7=kLayoutStrided=ND), pt2 has no NZ enum → cannot write NZ
> - Weights are `tensor_values` Data nodes (`p_*`), **not ascend_ir Const ops** (ascend_ir's Const are all small inline constants, max 4 elements) → no `ascend_ir.desc.layout` to modify
> - Consuming op `inputDesc[i].layout` (string "ND") can be changed to "NZ", but weight Const still ND → triggers TransData instead

**NZ signal now goes through `weight_info.json`'s `"format":"NZ"` string** (`_pack_weights_to_dir` already writes to disk), read by C++ `CreateConstNode`/`GeWeightLoader` → `MapGeLayoutString` (see [C++ format mapping](#c-format-mapping)). `inline_weight_layout` no longer implemented.

<a id="ep_json-annotation-sync-mechanism-_sync_-added"></a>
## ep_json Annotation Sync Mechanism (_sync_* added)

> Added 2026-10-20: two ep_json post-processing functions that exist in code but were previously undocumented. They are key links for Stage 2 (Z) path to compile, and the root cause of the 2026-10-20 board incident.

Weight byte conversion happens in packer (`_transform_weight_tensor`), but the ep_json generated by torchair lite export still carries **original ND-semantic annotations**. Before writing to disk, annotations must be synced to post-conversion semantics, two patches:

### `_sync_input_desc_layout` (`epair_utils.py:491`)

Reads weight_info's blob `format` field, patches ep_json three points for non-ND tensors:

1. **inputDesc layout**: `"ND"` → blob format (e.g. `"Z"`)
2. **inputDesc shape (Z only)**: 2D `[K, N_packed]` → 4D `[K//64, N//16, 16, 8]` (N = N_packed×8, 8 int4 packed into 1 int32)
3. **tensor_values sizes (Z only)**: 2D → 4D, matching Z fractal shape

Z fractal constants (`_B_FRAC_K=64, _B_FRAC_N=16, _INT4_PER_INT32=8`) must be consistent with `layout.py`'s `pack_int4_nz`. Not patching #2/#3 consequences: GE infershape propagates 2D shape through Z layout, producing wrong StridedSlice output dims (e.g. `[1,1,16,8]` instead of `[2,1]`), causing Reshape compile failure. Tensor name mapping: `op.weight_q` → inputDesc `x2` (QuantBatchMatmulV4).

### `_sync_weight_scale_dtype` (`epair_utils.py:615`)

`transdata_scale` materializes weight_scale data from fp32 to uint64 bit pattern; ep_json dtype annotation syncs two patches:

1. **tensor_values**: dtype 7 (float32) → 100 (synthesized uint64) — not patching causes C++ side `WEIGHT_SIZE_MISMATCH`
2. **inputDesc**: `x2_scale`'s `DT_FLOAT` → `DT_UINT64` — not patching causes arch31 tiling rejection (arch31 s8s4 requires x2_scale uint64 ND)

Self-check signals: export logs show `Synced N inputDesc shape(s) 2D → 4D for Z layout` / `Synced N tensor_values sizes 2D → 4D for Z layout` / `Synced weight_scale dtype: N tensor_values, N inputDesc` three lines.

### 2026-10-20 Asymmetric Gating Incident (root cause record)

**Symptom**: single-op export (env not on) → compile-time `PreparePlaceholders failed: load tensor data error, placeholder=b_op_weight_scale`.

**Root cause**: same weight_scale, two processing paths with inconsistent gating —
| Processing | Execution condition |
|---|---|
| `transdata_scale` packing (fp32→uint64 bit pattern, 8B) + `pack_int4_nz` | **Unconditional** (`_transform_weight_tensor` inline executes) |
| `_sync_weight_scale_dtype` / `_sync_input_desc_layout` annotation sync | **Only `CANN_LM_ENABLE_WEIGHT_TRANS=1`** (`epair_utils.py:210/:483` if gate) |

env off → produces self-contradictory epair (weights_1.bin 8B×40960=327680B data, but ep_json marks 4B float32) → C++ `BuildTensorHolder` expected_bytes=163840 ≠ actual 327680 → compile-time size mismatch, **two hops from root cause** (error doesn't point to export side).

**Lesson (consolidated as core argument for master switch deprecation design)**: expressing "necessary accompanying actions" as "optional actions" inevitably creates two-end drift space. Data and its annotations must be co-located co-sourced — migration plan ([Master Switch Deprecation Design](#master-switch-deprecation-design-finalized-2026-10-20) action ②) makes them unconditional, mechanically eliminating this class of contradiction.

<a id="c-format-mapping"></a>
## C++ format mapping

> **Mechanism correction (discovered 2026-09-05): original design wrote "weight Const format driven by `ascend_ir.desc.layout` (`MapGeLayoutString`)" — wrong function**. Actual `ge_graph_builder.cc`:

**Weight Const path = `CreateConstNode`** (`ge_graph_builder.cc:473-509`, called at `:1213-1221` for Parameter/Buffer/TensorConstant placeholder):
- `:478` `format = mapper.MapPt2Layout(holder.format, ...)` — `holder` from `GeWeightLoader::ReadTensorData`, `holder.format` is **pt2 enum** (from ep_json `tensor_values[p_*].layout`=7)
- `:487` `ge::TensorDesc desc(ge::Shape(holder.dims), format, dtype)` → Const node (`--external_weight` → GE compile-time `ConvertConstToFileConst`→FileConstant)

**`MapPt2Layout`** (`aten_tensor_desc_mapper.cc:67`) only `kLayoutStrided→FORMAT_ND`, **no NZ**; pt2 has no NZ enum → weight Const currently cannot express NZ.

**`MapGeLayoutString`** (`aten_tensor_desc_mapper.cc:119`) only used by `ValidateConstOutputDesc` (`:517`, inline Const validation) + `RestoreInlineConstTensor` (inline Const's pt2-layout-string conversion) — **not the weight Const path**.

### C++ format changes (landed, NZ + Z two cases)

| File | Function | Change | Status |
|---|---|---|---|
| `aten_tensor_desc_mapper.cc:119` | `MapGeLayoutString` | `"NZ"` → `ge::FORMAT_FRACTAL_NZ` (:127 comment is this source); Z path goes through `QuantBatchMatmulV4` inputDesc's `"Z"` via `MapAscendIrDesc` | ✅ |
| `ge_weight_loader.cc` (`TensorRecordInfo`/`TensorHolder`) + `ge_graph_builder.cc:478` (`CreateConstNode`) | Read `weight_info.json`'s `"format"` string + `CreateConstNode` dispatch on it | `holder`/`TensorRecordInfo` carries `format`; hit `"NZ"` → `MapGeLayoutString("NZ")→FORMAT_FRACTAL_NZ`, else `MapPt2Layout(holder.format)` (ND) | ✅ |

Note: `ge_graph_builder.cc:401 MakeTensorDesc` (`:407 SetFormat(info.format) ← MapAscendIrDesc`) is the **ascend_ir op desc path** (inline Const / normal op inputDesc/outputDesc), **not weight Const**; weight Const format is constructed directly at `CreateConstNode:487` `TensorDesc(format, dtype)`, does not go through MakeTensorDesc.

## Z shape semantics (Stage 2 quantized path)

Parallel to NZ section's FractZ contract (`pack_int4_nz` product, consumed by kirinx90 QuantBatchMatmulV4 s8s4 kernel):

- inputDecl 2D `[K, N_packed]` (int32, 8 int4/element) → Z physical 4D `[K//64, N//16, 16, 8]` (N = N_packed×8)
- ep_json shape annotation must sync to 4D (`_sync_input_desc_layout`, see [ep_json Annotation Sync Mechanism](#ep_json-annotation-sync-mechanism-_sync_-added)), otherwise GE infershape propagates 2D through Z producing wrong dims
- q_proj measured: `[2048, 320]` int32 → `[40, 128(=2048/16), 16, 8]`

<a id="nz-shape-semantics"></a>
## NZ shape semantics

Follows GE `FORMAT_FRACTAL_NZ` standard contract:

- **ep_json shape = logical [M, K]** (not padded)
- `nd2nz` internally zero-pads non-16-multiple dims (ensures byte completeness), writes physical NZ blocks
- GE kernel reads valid elements of partial block, ignores pad zeros

**Does not write padded shape** (`[m*16, k*16]`) — would make MatMul output dims padded_M, breaking downstream shape. LLM Linear dims are practically 16-aligned (hidden 1024/2048, intermediate 3072, head_dim 128×n_heads etc.), pad rarely triggers.

<a id="stage-division"></a>
## Stage Division

### Stage 1 — fp16 path (this delivery, low risk)

1. `weight_format/layout.py` add `nd2nz`
2. `weight_format/rules.py` config loading (`load_rules` reads shipped default config) + `resolve_format`
3. `epair_utils._pack_weights_to_dir` integration (env-gated, see [Environment Variable Master Switch](#environment-variable-master-switch)) — writes NZ bytes + `weight_info.json` marks `format` (`inline_weight_layout` deprecated, see [ep_json layout annotation](#ep_json-layout-annotation))
4. C++ `CreateConstNode`/`GeWeightLoader` reads `weight_info.format` + `MapGeLayoutString` adds NZ case (see [C++ format mapping](#c-format-mapping))
5. Doc sync (this doc + export_design.md / quant_design.md notes + INDEX.md)

### Stage 2 — quantized path (implemented and board-verified)

Quantized weight_q is int4pack int32 `[K, N//8]`, **standard `nd2nz` would break nibble pairing**, implemented as dedicated `pack_int4_nz` (FractZ, see [Format Conversion Functions](#format-conversion-functions)). Historical research gate record (reverse-engineering CANN kernel's exact byte arrangement) closed with KirinX90 board cos=0.999912; oracle decision method (constant input enumeration f-bit decomposition) consolidated in board guide.

**Key fact**: A3 measured `QuantBatchMatmulV4` has `ND_ND` and `ND_NZ` two parallel kernel variants — feeding ND weights selects `ND_ND` (internal transpose handling), feeding Z weights selects the variant skipping internal conversion; pre-converting layout is **performance optimization not correctness requirement**, doing it wrong breaks a working OM (this is why historically a research gate was set, and shipped rules once marked `"format": "ND"`).

<a id="pending-verification"></a>
## Pending Verification

> **Background correction**: ND→NZ conversion has two locations — ① HOST-side `TransData` op (visible in OM); ② **inside kernel** (no TransData in OM, but MatMul kernel does ND→NZ each execution). No host TransData ≠ no cost — may move into kernel (per-execution performance drop). CLI-015's goal is to avoid kernel-internal conversion, premise **holds**.
>
> **CANN official NZ path**: CANN provides `aclnnTransMatmulWeight` (device ND→NZ) + `aclnnMatmulWeightNz` (NZ matmul), A3 supports. NZ 4D = `[n1, k1, k0, n0]` (k0=16, n0=16). nn.Linear uses view/strides for transpose (no Transpose op). This is the **ACLNN runtime path**, not GE graph/OM compile path.
>
> **torchair innocent**: plain `aten.mm`/`aten.matmul` converter = `ge.MatMul(self, mat2, None)` — no Transpose inserted. Transpose_1 is a product of **torch.export decomposing `F.linear`** (nn.Linear → weight.T + mm), not torchair actively inserting. bare-matmul (`x @ weight[K,N]`, no nn.Linear) → ep_json only MatMul (no Transpose_1).

### L2 Verification Result (bare-matmul, A3, 2026-09-07)

**✅ Path works! NPU output NZ OM == ND OM, max_diff=0.000000e+00.**

Three-piece set (all required):

| # | Component | Content | Effect |
|---|------|------|------|
| ① | Python `nd2nz` | `[K,N]` → `reshape(k1,16,n1,16)` → **`permute(2,0,1,3)`** → `[n1,k1,k0,n0]` | CANN-correct NZ bytes |
| ② | C++ `CreateConstNode` | **2D shape `holder.dims`** + format `FRACTAL_NZ` (**no manual reshape 4D**) | GE itself `TransShapeToFracNz` computes 4D storage (no 6D hit) |
| ③ | C++ input desc NZ override | MatMul input[1] desc format → `FRACTAL_NZ` (when referencing NZ weight) | Const NZ + MatMul input NZ consistent → format refiner doesn't revert |

OM structure (env-on):
```
Const p_q_proj_weight  out_layout=FRACTAL_NZ  out_shape=[8,8,16,16]  ← GE-computed 4D (not 6D!)
MatMul MatMul          in_layout=['ND', 'FRACTAL_NZ']                 ← input[1]=NZ, not reverted
total=4  TransData=0
```

### Key Lessons (pitfalls during L2)

1. **Don't manually reshape to 4D**: setting 4D storage `[8,8,16,16]` → GE `TransShapeToFracNz` treats as NCHW `[8,8,16,16]` and converts again → 6D `[8,8,1,1,16,16]` (H=16,W=16 → ceil/16=1)→ format_transfer/InferShape/format refiner three mechanisms all reject. **Keep 2D origin + format FRACTAL_NZ** → GE computes correct 4D storage (no 6D hit).
2. **nd2nz permute order**: must match CANN doc `[n1,k1,k0,n0]`. `permute(0,2,1,3)` produces `[n1,k1,n0,k0]` (inner swapped) → NPU max_diff=41 (wrong). `permute(0,2,3,1)` produces `[k1,n1,n0,k0]` (all swapped) → max_diff=39 (wrong). **`permute(2,0,1,3)`** produces `[n1,k1,k0,n0]` (correct) → max_diff=0.
3. **input desc NZ override must be set simultaneously**: only Const NZ + MatMul input ND (no override) → format refiner reverts to ND (no-op). Const NZ + MatMul input NZ (override) → consistent → retained → compiles.
4. **GE FRACTAL_NZ ≠ CANN API NZ**: GE's `FORMAT_FRACTAL_NZ` is NCHW-based (`TransShapeToFracNz` treats any shape as NCHW); CANN API `aclnnMatmulWeightNz` uses matmul-specific 4D `[n1,k1,k0,n0]` (direct use). But **keep 2D origin + FRACTAL_NZ → GE computes 4D (consistent with CANN API)** — key is not to manually set 4D.
5. **torchair doesn't insert Transpose**: Transpose_1 comes from torch.export decomposing `F.linear` (nn.Linear → weight.T + mm). bare-matmul (`x @ weight[K,N]`) has no F.linear → no Transpose_1.

### L2 Failure Paths (excluded during L2)

| Attempt | Result | Root cause |
|------|------|------|
| 4D storage + FRACTAL_NZ (manual reshape) | ❌ Compile failure | GE treats as NCHW, converts again → 6D → IsShapeArgValid 4D-vs-6D |
| 2D + FRACTAL_NZ (no input override) | ❌ no-op | Const NZ vs MatMul input ND → format refiner reverts to ND |
| C++ Transpose perm override (①) | ❌ Irreconcilable | format_transfer uses 6D storage, InferShape uses 4D origin → perm cannot satisfy both |
| Lock (`ATTR_NAME_STORAGE_FORMAT`/`FORMAT_LOCKED`) | ❌ SDK infeasible | `ge_attr_define.h`/`AttrUtils` not in CANN 0715 public headers |
| `inline_weight_layout` (ep_json layout rewrite) | ❌ Blocked | `tensor_values.layout` is pt2 enum (no NZ), weights not ascend_ir Const ops |

### L2 Success Path (bare-matmul)

**bare-matmul** (`x[K] @ weight[K,N]`, no nn.Linear) → ep_json only `ge.MatMul` (no Transpose_1) → env-on export → cann_lm_compile → OM compile passes (Const FRACTAL_NZ + MatMul input NZ + 0 TransData) → NPU output vs ND OM **max_diff=0**.

### L2 Verification Result: nn.Linear scenario (Transpose_1, A3, 2026-09-07)

**✅ nn.Linear also works! NPU max_diff=0.000000e+00.**

nn.Linear's `F.linear` is decomposed by torch.export into `Transpose(weight) + mm` — Transpose op's 2D perm `[1,0]` hits `format_transfer_transpose` (4D-vs-2D) + `TransposeCommonInferShape` (origin 2D, perm value out of bounds) on NZ weight Const's 4D storage. **perm irreconcilable** (format_transfer wants 4D perm, InferShape wants 2D).

Solution: **skip Transpose + set MatMul `transpose_x2=True`** — GE MatMul native attribute, internal transpose, no Transpose op.

Four-piece set (nn.Linear scenario, all required):

| # | Component | Content | Effect |
|---|------|------|------|
| ① | Python `nd2nz` | `[K,N]` → reshape(k1,16,n1,16) → `permute(2,0,1,3)` → `[n1,k1,k0,n0]` | CANN-correct NZ bytes |
| ② | C++ `CreateConstNode` | 2D shape `holder.dims` + `FRACTAL_NZ` (no reshape) | GE itself `TransShapeToFracNz` computes 4D storage |
| ③ | C++ pre-pass + skip Transpose | Transpose on NZ weight → skip + register pass-through (`Transpose:0 → Const`) | Eliminate Transpose op (avoid 2D-perm-on-4D) |
| ④ | C++ MatMul `transpose_x2=True` + input NZ override | bind to Const + `transpose_x2=True` + input desc `FRACTAL_NZ` | GE internal transpose + Const NZ + MatMul input NZ consistent |

Key implementation details:
- `nz_transpose_passthrough` must be in `GraphBuildContext` (cross-node shared) — Transpose at FX node[0], MatMul at node[1], local variable doesn't cross nodes.
- MatMul's input reference may be an FX alias (e.g. `"t"`), need to `ResolveBoundaryInputSourceRef` to resolve to GE ref (e.g. `"Transpose:0"`) then check passthrough.
- `transpose_x2=True` added to attrs via `ge::AttrValue::SetAttrValue(true)`.

OM structure (env-on):
```
Const p_q_proj_weight  out_layout=FRACTAL_NZ  out_shape=[8,8,16,16]
MatMul MatMul          in_layout=['ND', 'FRACTAL_NZ']  ← transpose_x2=True, no Transpose op
total=4  TransData=0
```

### L2 Path Verification Summary

| Scenario | OM structure | NPU max_diff | Status |
|------|---------|-------------|------|
| bare-matmul (`x @ weight[K,N]`, no nn.Linear) | Const NZ + MatMul input NZ + 0 TransData | **0** | ✅ |
| nn.Linear (`F.linear`, has Transpose_1) | Const NZ + MatMul input NZ + 0 TransData + transpose_x2=True | **0** | ✅ |
| bare-matmul 130×130 (non-16-aligned) | — | — | ❌ GE limitation (see below) |

Both scenarios **NPU output NZ OM == ND OM, max_diff=0**. NZ weight pre-conversion fully passes on A3.

### Non-aligned shape limitation (2026-09-07)

**16-alignment is a hard constraint for NZ**. Non-aligned shapes (e.g. 130×130) are infeasible in the GE graph compile path (Const + ge.MatMul); root cause is GE's two contradictory checks:

| GE check | Expected shape | 130×130 result |
|---------|-----------|-------------|
| `CheckTensorIsValid` (`graph_prepare.cc:2270`) | `data_size == Product(shape) * elem_size` | 2D `[130,130]`: 33800 ≠ 41472 (NZ padded) → ❌ |
| `MatMul InferShape` (`matrix_calculation_matmul.cc:473`) | `dim[0] == input_k` (k-axis match) | 4D `[9,9,16,16]`: k=9 ≠ 130 (input k) → ❌ |

- **2D `[130,130]` + FRACTAL_NZ**: CheckTensorIsValid fails (NZ padded bytes 41472 ≠ logical 33800).
- **4D `[9,9,16,16]` + FRACTAL_NZ**: InferShape fails (GE sees dim[0]=9 as k-axis, doesn't recognize NZ block semantics).
- **padded 2D `[144,144]`**: InferShape fails (k=144 ≠ input k=130; input is runtime Data, can't pad).
- **128×128 passes**: because 16-aligned → logical size = NZ padded size (32768=32788), both checks satisfied.

**Root cause**: GE's `CheckTensorIsValid` doesn't understand NZ padding (checks `data_size == Product(logical_shape) * elem_size`, doesn't recognize NZ block padding). CANN official `aclnnMatmulWeightNz` goes through ACLNN runtime (4D NZ direct pass, doesn't go GE `CheckTensorIsValid`), so official path can handle non-aligned; GE graph compile path cannot.

**Current implementation**: 16-aligned shapes use NZ (verified); non-aligned auto-fallback ND (need shape alignment check in `_pack_weights_to_dir`). LLM actual weight dims mostly 16-aligned (hidden 1024/2048, intermediate 3072, head_dim 128×n_heads etc.), non-aligned rare.

### L3 Full Network (A3)

LLM `--mode single` + `CANN_LM_ENABLE_WEIGHT_TRANS=1` export → OM compile → `atc --mode=1` check `*_proj`-corresponding MatMul has no TransData → NPU text generation vs ND version consistent.

### L4 Kirin (2026-10-20 partially achieved)

KirinX90 **quantized Z path (Stage 2)** empirically打通: q_proj single-op (E4B int4pack W4A8) → epair (Z-packed weight_q + uint64 scale) → `cann_lm_compile` → skeleton omc + externalized SubGraph_0.weight → board MRT V2 → board output vs dopt fakequant golden **cos=0.999912** (cross-semantic criterion ≥0.99, same-semantic ≥0.999 both pass). Full-flow command template and regression anchor (md5 baseline) consolidated in ``examples/test_converter/dopt_op_benchmark/op_npu_verify/GUIDE_kirinx90_board.md``. fp16 MatMul NZ on Kirin L2/L3 re-verification still deferred (kirinx90 board doesn't support multi-stream, matmul ND/Z kernel behavior to be verified separately).

> Note Z ≠ NZ semantics: quantized weight_q goes through `pack_int4_nz` to **FractZ** (`[K//64,N//16,16,8]`, policy defined by QuantBatchMatmulV4 kernel), different layout from fp16 `*_proj`'s FractNZ (16×16). rules.json's `"format": "Z"` and `"NZ"` fields correspond to each respectively.

### Core Criterion (corrected)

NZ rearrange success ⟺ **NPU output NZ OM == ND OM** (max_diff=0) + OM weight Const `layout=FRACTAL_NZ` + 0 host TransData. TransData count is not the only metric (kernel-internal conversion has no TransData but has cost).

<a id="follow-up-migration"></a>
## Follow-up Migration

### Completed (2026-08-31): quant_swap migrated into export/ + swap promoted to converter root

`materialize.py` / `quant_linear.py` / `swap.py` migrated out of what was then `cann_local_lm/converter/quant/`:
- `materialize.py` / `quant_linear.py` → `cann_local_lm/converter/export/quant_swap/` (materialization internal, co-located with export pipeline)
- `swap.py` → `cann_local_lm/converter/swap.py` (converter root, orchestration layer peer of `loader.py` / `saver.py`, absolute imports materialize/quant_linear)

Old import paths cleanly cut: `swap_from_dopt` goes `cann_local_lm.converter.swap`, materialization symbols go `cann_local_lm.converter.export.quant_swap`. `quant/` now only retains CANN lightweight tool orchestration (`auto_quant` / `dopt_wrapper`) + vendored `dopt_lm` + calibration data. See [quant_design.md](../quant/quant_design.md) "quant package architecture".

### Decided (2026-09-05): transdata_scale / pack_int4 not merged into weight_format/

`transdata_scale` / `pack_int4` currently in `export/quant_swap/materialize.py`, semantically also "format conversion". **Not merged into** `weight_format/layout.py` — both are **materialization bit-pattern packing** (`transdata_scale`: fp32→int64 bit pattern; `pack_int4`: int4→nibble-packed int8), not layout transformation, orthogonal to `nd2nz` / `pack_int4_nz`'s **layout rearrangement** responsibility, stays in `export/quant_swap/materialize.py`. `epair_utils` continues to import `transdata_scale` from `quant_swap.materialize`, no import/test changes.

### Finalized (2026-10-20): CANN_LM_ENABLE_WEIGHT_TRANS deprecation migration

See [Master Switch Deprecation Design (Finalized 2026-10-20)](#master-switch-deprecation-design-finalized-2026-10-20). Key points: control handed to `weight_format_rules.json` (pattern tightened `*` → `*.weight_q`, default effective, clear rules to all ND escape hatch); annotation sync `_sync_*` unconditionalized; v1 packer reuses `_transform_weight_tensor` to fix wildcard false-positive; migration PR one-shot removes all 8 env reads, no dual-track period. **Not implemented** — this section is design finalization; when landing execute per that section's "Change List" + "Regression Acceptance".

## Document References

| Document | Related |
|------|------|
| [export_design.md](./export_design.md) | Export main flow / `epair_utils` / C++ `GeWeightLoader` / `inline_weight_scale_dtype` precedent |
| [quant_design.md](../quant/quant_design.md) | Quantization path / `transdata_scale` / `QuantBatchMatmulV4` / weight_scale on-disk |
| [INDEX.md](./INDEX.md) | Export module index (synced to add weight_format package entry) |
| ``op_npu_verify/GUIDE_kirinx90_board.md`` | User-level end-to-end guide: single-op export→omc compile→KirinX90 board run→compare; includes this capability's usage contract (export-time env, 3 Synced self-check) and regression anchor |
