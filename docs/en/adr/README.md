# Architecture Decision Records (ADR)

English | [简体中文](../../zh/adr/README.md)

> Records the context, choices, and consequences of major architectural decisions in the CANN-LM project, for future tracing of "why this was chosen".

## When to write an ADR

Write an ADR (rather than just in a design .md) when any of the following conditions is met:

- **Operator/technical path pivot**: e.g., self-developed op → reuse CANN native op (option A → path C' type)
- **Cross-module architecture selection**: e.g., quant vs export memory bridge vs integration, subprocess vs in-process
- **Irreversible or high-cost-to-rollback decisions**: e.g., registering ops, on-disk format, IR node structure
- **Selecting one option after weighing multiple**: record the rejected alternatives and rationale

Scenarios that do not need an ADR: function renames, line-number updates, bugfixes, local refactoring (these rely on design .md + code comments).

### Trigger checklist (any Yes → write ADR)

When you get a change, go through these 8 questions; if any answer is Yes, write an ADR; if all are No, just modify the design .md + code comments:

| # | Judgment question | Typical scenario when Yes |
|---|---------|-----------------|
| 1 | Are you choosing between two or more technical options? | Self-developed op vs reuse CANN native; V3 vs V4; memory bridge vs disk bridge vs subprocess |
| 2 | Are you introducing or removing a custom op / GE op registration? | `reg_register.h` adding/removing `REG_OP`; adding a full `cann_local_lm::` schema+fake+converter set |
| 3 | Is the cost of rolling back this change very high? | Requires modifying many files / re-exporting all epairs / recompiling OM / existing artifacts incompatible |
| 4 | Does it affect the link structure across modules? | quant↔export↔loader↔saver bridging method; prefill-decode dual-graph vs single graph |
| 5 | Does it change the on-disk format or memory layout of weights/scales? | weight_scale fp32→int64 bit pattern; pre-transposed vs natural orientation; nibble-packed vs unpack |
| 6 | Does it change environment dependencies? | Introducing/removing torch_npu (stub path vs real torch_npu); torchair source-compiled vs bundled |
| 7 | Does it change the deployment form? | host-embed vs in-graph embed; lm_head on NPU vs host; tie_word_embeddings untying |
| 8 | Looking back three months later, would you need to know "why this was chosen then"? | A certain option was rejected but the rationale is non-obvious; workaround trade-offs (e.g., dummy bias) |

### Typical scenarios → whether to write an ADR

| Scenario | Write ADR? | Corresponding checklist |
|------|---------|---------------|
| Option A `quant_matmul_s8s4` → path C' `npu_quant_matmul` | ✅ (already happened, not retroactively recorded, left as historical archive section) | 1,2,3,8 |
| weight_scale on-disk fp32→int64 (option A modify buffer vs B on-disk only) | ✅ | 3,5 |
| GPTQ one-click chose option C (subprocess) vs A (modify CANN lightweight tool) vs B (memory) | ✅ (already happened) | 1,4,6 |
| converter does not install torch_npu, goes through stub path | ✅ (already happened, recorded in cli_design.md) | 6,8 |
| ~~`swap_to_quant_linear_from_dopt` split into `_swap_linear_from_sd` + `swap_from_dopt`~~ | ✅ Landed (2026-08-31): `swap_to_quant_linear_from_dopt` deleted, `_swap_linear_from_sd` + `swap_from_dopt` as the only path  — |
| `materialize_int4_with_scale` renamed to `materialize_int4_per_group_symmetric` | ❌ rename | — |
| Function line-number offsets, signature params collapsed into a dataclass | ❌ | — |
| Adding `QuantLinear.bias=None` makes GE converter insert dummy zeros[N] | ⚠️ workaround trade-off (part of the path C' decision, not written separately) | 8 (already included in the path C' ADR) |

> Judgment criterion: **if the "why" of a decision cannot be seen just by looking at the code → write an ADR**. Things that are self-explanatory in code (renames/splits/line-numbers) are not written.

## Status field

| Status | Meaning |
|------|------|
| `Proposed` | Proposed, pending review/decision |
| `Accepted` | Adopted, currently in effect |
| `Superseded` | Superseded by a later ADR (note which one supersedes it) |
| `Deprecated` | Deprecated with no replacement |

The "Status" field at the top of the design .md references the corresponding ADR's Status. When the Status changes, **you must** synchronously update the status field at the top of the corresponding design .md.

## Naming convention

- `NNNN-kebab-case-title.md` (NNNN = 4-digit serial number, starting from 0001)
- `0000-template.md` = this directory's template (not an official ADR)
- Serial numbers are monotonically increasing and are not reused for Superseded ones

## Relationship with design .md

| Document | Content carried | Mutability |
|------|---------|--------|
| `docs/adr/*.md` | The "why" of decisions — context, trade-offs, rejected alternatives | Low (once a decision is written it basically does not change, only the Status changes) |
| `docs/cann_lm_*/*.md` | The "what" of design — function lists, signatures, flows, line-number anchors | High (evolves with code) |

Add `> Status: currently adopted | Last reviewed: YYYY-MM-DD` at the top of the design .md (with a link when there is a corresponding ADR). The ADR is the source of truth; the design .md reflects the current implementation state.

## Index

| ADR | Title | Status |
|-----|------|--------|
| [ADR-0001](./0001-load-model-cpu-quant-subprocess-device.md) | converter main process load_model hard-coded to CPU; lightweight tool subprocess supports device selection | Accepted |

> Historical pivots that have already happened (e.g., option A `quant_matmul_s8s4` → path C' `npu_quant_matmul`) are not retroactively recorded as ADRs, but are retained in the corresponding design .md's "historical archive" section for tracing.
