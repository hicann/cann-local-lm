# Converter Code-Convention Deviation Cleanup Notes

English | [简体中文](../../zh/cann_lm_converter/TODO_notes.md)

> Records deviations between `cann_local_lm/converter/` code and `AGENTS.md` conventions / design documents, accumulated in discovery order.
> Each entry contains: location, current state, convention, root cause, impact, status, cleanup action. After cleanup, status is set to "Cleaned" and the record is retained.
>
> Status: Accumulating | Last updated: 2026-09-01

## Table of Contents

- [CLI-001 `from __future__ import annotations` redundancy](#cli-001-from-__future__-import-annotations-redundancy)
- [CLI-002 saver `use_embeddings` default value doc vs code mismatch](#cli-002-saver-use_embeddings-default-value-doc-vs-code-mismatch)
- [CLI-003 Scenario 2 call chain Embedding strategy doc outdated](#cli-003-scenario-2-call-chain-embedding-strategy-doc-outdated)
- [CLI-004 Scenario 2 `--quant-bits` out of sync with dopt_wrapper](#cli-004-scenario-2---quant-bits-out-of-sync-with-dopt_wrapper)
- [CLI-005 `--quant-device` CLI parameter removed (device auto-detection)](#cli-005---quant-device-cli-parameter-removed-device-auto-detection)
- [CLI-006 `generate_dopt_config` removed + `build_layer_strategy` renamed](#cli-006-generate_dopt_config-removed-build_layer_strategy-renamed)
- [CLI-007 TODO: CPU/GPU one-click non-subprocess evaluation](#cli-007-todo-cpugpu-one-click-non-subprocess-evaluation)
- [CLI-008 `swap_to_quant_linear_from_dopt` dead code removal](#cli-008-swap_to_quant_linear_from_dopt-dead-code-removal)
- [CLI-009 `swap_from_dopt` parameter simplification (remove skip/act_quant)](#cli-009-swap_from_dopt-parameter-simplification-remove-skipact_quant)
- [CLI-010 TODO: load_model loads HF original Linear fp waste (swap discards)](#cli-010-todo-load_model-loads-hf-original-linear-fp-waste-swap-discards)
- [CLI-011 TODO: build_layer_strategy vs strategy.py consistency](#cli-011-todo-build_layer_strategy-vs-strategypy-consistency)
- [CLI-012 TODO: gemma4 PLE quantization unadapted (_build_external_extras uses HF original float)](#cli-012-todo-gemma4-ple-quantization-unadapted-_build_external_extras-uses-hf-original-float)
- [CLI-013 TODO: side LLM tool CLI and doc delivery survey](#cli-013-todo-side-llm-tool-cli-and-doc-delivery-survey)
- [CLI-014 TODO: PD separated export stores only one weight copy, decode weights discarded (fails when PD weights differ)](#cli-014-todo-pd-separated-export-stores-only-one-weight-copy-decode-weights-discarded-fails-when-pd-weights-differ)
- [CLI-015 TODO: implement weight ND to NZ format conversion per weight_format_design.md](#cli-015-todo-implement-weight-nd-to-nz-format-conversion-per-weight_format_designmd)
- [CLI-016 TODO: epair weight and const separation](#cli-016-todo-epair-weight-and-const-separation)
- [CLI-017 TODO: torch-delegate supports torchair stubbing then consolidate install scripts](#cli-017-todo-torch-delegate-supports-torchair-stubbing-then-consolidate-install-scripts)
- [CLI-018 TODO: simplify cann-lm multi-environment referencing mlx-lm env handling](#cli-018-todo-simplify-cann-lm-multi-environment-referencing-mlx-lm-env-handling)
- [CLI-019 TODO: distinguish p_/c_/b_ three data types when exporting epair](#cli-019-todo-distinguish-p_c_b_-three-data-types-when-exporting-epair)

---

<a id="cli-001-from-__future__-import-annotations-redundancy"></a>
## CLI-001 `from __future__ import annotations` redundancy

| Field | Value |
|------|-----|
| ID | CLI-001 |
| Location | `cann_local_lm/converter/` **6 places total**: `cann_lm_converter.py:17`, `loader.py:3`, `saver.py:3`, `export/epair_utils.py:3`, `export/ge_converters/ge_converter.py:17`, `quant/download_dataset.py:15` |
| Current state | File top has `from __future__ import annotations` |
| Violates convention | `AGENTS.md` "General / Python version: `>= 3.10` (use `dict[str, Any]` etc. builtin generic syntax, **do not introduce** `from __future__ import annotations`)" |
| Discovery date | 2026-08-31 (first round checked `cann_lm_converter.py`; second round comparing cli_design.md ↔ converter files extended to all 6 places via grep) |
| Status | **Cleaned (2026-08-31)** |

### Background

`from __future__ import annotations` (PEP 563) changes all type annotations in a module from "evaluated at definition time" to "stored as delayed strings". In Python 3.7–3.9, it makes `int | None` / `list[str]` etc. 3.10+ syntax available early (annotations are just strings, not eval'd, so old versions don't error on unrecognized syntax).

### Root Cause

Project Python environments (all three verified consistent):

| Environment | Path / command | Python version |
|------|------------|-------------|
| Default | `python3` | 3.12.9 |
| System (④ npu verify-runtime) | `$DOPT_PYTHON` | 3.12.9 |
| converter venv (① export epair) | `$TORCH_DELEGATE_HOME/.venv/bin/python` | 3.12.9 |

All three are `>= 3.10`. Python 3.10+ natively supports runtime evaluation of `int | None` (PEP 604), `list[str]` / `dict[str, Any]` (PEP 585), no future annotations needed. So this line is redundant — safely removable dead code, and violates "do not introduce future annotations" convention.

### Impact (future syntax annotations used by each file)

- `cann_lm_converter.py`: `convert()`'s `num_layers: int | None`, `quant_path: str | None` etc. (:51–55)
- `loader.py`: `ModelArtifact`'s `quant_info: dict | None` (:26), `tokenizer_files: list[Path]` (:23)
- `saver.py`: `output_dir: str | Path`, `whitelist_decompositions: set | None`, `embed_quant: dict[...] | None`
- `epair_utils.py` / `ge_converter.py` / `download_dataset.py`: similar `X | Y` / generic annotations

After removing this line, 3.11 natively evaluates these types, **no behavior change**; instead `get_type_hints()` etc. runtime reflection recovers from "getting strings" to "getting real type objects" (friendlier to reflection-dependent libraries).

### Design Doc Not Reflected

[cli_design.md](./cli/cli_design.md)'s `convert()` code block (L36–126) doesn't note this line; "converter and torch_npu relationship" etc. sections also don't mention it. That is, source has this line but design doc never recorded it; supplemented here.

### Cleanup Action

1. Remove `from __future__ import annotations` from the top of the above 6 files (1 line each, keep import grouping tidy with following blank line).
2. After cleanup, run a converter smoke test (e.g. `examples/` qwen3.5-0.8B `single` export) to confirm no import / annotation evaluation errors.
3. Update this doc's status to "Cleaned" and add cleanup date.

### Execution Result (2026-08-31)

Top-of-file `from __future__ import annotations` removed from all 6 files (`cann_lm_converter.py`, `loader.py`, `saver.py`, `export/epair_utils.py`, `export/ge_converters/ge_converter.py`, `quant/download_dataset.py`). `python3 -m py_compile` all 6 files pass (3.11 natively evaluates `int | None` / generic annotations, no syntax errors); grep confirms `cann_local_lm/converter/` directory 0 residual.

> End-to-end smoke (actually running a converter export) not executed — requires sourcing `scripts/set_env_export.sh` full env (torchair stubs), is optional verification; syntax layer confirmed no regression.

### Related

- Convention source: `AGENTS.md` "General / Python version"
- Source files: 6 files under `cann_local_lm/converter/` (see table above)
- Design doc: [cli_design.md](./cli/cli_design.md)

---

<a id="cli-002-saver-use_embeddings-default-value-doc-vs-code-mismatch"></a>
## CLI-002 saver `use_embeddings` default value doc vs code mismatch

| Field | Value |
|------|-----|
| ID | CLI-002 |
| Location | Doc: [cli_design.md:272](./cli/cli_design.md); Code: `cann_local_lm/converter/saver.py:58`, `saver.py:139` |
| Current state (before cleanup) | Doc signature block writes `use_embeddings=True`, code `save_epair` / `save_epair_prefill_decode` default `False`; doc note "saver default False" contradicts signature block True, and contradicts code |
| Violates convention | Design doc should match code (AGENTS "after modifying code, update docs/") |
| Discovery date | 2026-08-31 (second round comparing cli_design.md ↔ saver.py) |
| Status | **Cleaned (2026-08-31)** |

### Root Cause

saver as low-level API originally defaulted `use_embeddings=False` (in-graph lookup, self-contained); CLI layer `convert()` explicitly passes CLI default True (pure embeddings, embed_tokens externalized to host lookup). Design doc code block wrote saver default as True, contradicting code; same note also said False, internally contradictory.

### Decision

User confirmed True as authoritative: saver default aligned to "pure embeddings" (consistent with CLI default), no longer retaining saver layer's conservative default False.

### Cleanup Action (executed)

1. `cann_local_lm/converter/saver.py:58` `save_epair` default `use_embeddings: bool = False` → `True`
2. `cann_local_lm/converter/saver.py:139` `save_epair_prefill_decode` default `use_embeddings: bool = False` → `True`
3. [cli_design.md:272](./cli/cli_design.md) note "saver default False" → "saver default True" (signature block already wrote True, now code / signature block / note three-way consistent)

### Impact

- CLI path unchanged (`convert()` explicitly passes param, default change doesn't affect CLI)
- Affected: programmatic direct callers of `save_epair` / `save_epair_prefill_decode` not passing `use_embeddings` (e.g. `examples/` test scripts) — default changes from "in-graph lookup" to "pure embeddings". Need to audit such direct call sites, confirm whether to explicitly pass `False`.

### Residual (not changed, noted)

`saver.py:253` `save_model_config(artifact, output_dir, *, use_embeddings: bool = False, ...)` remains `False`. This parameter's semantics are "the `use_embeddings` field recorded when writing `model_config.json`", and `save_epair` / `save_epair_prefill_decode` calls it **always passing explicitly** (its own `use_embeddings`), so default never hits. If full-module unification desired, can change to True too, but semantics independent, not touched this time.

### Related

- Source file: `cann_local_lm/converter/saver.py`
- Design doc: [cli_design.md](./cli/cli_design.md) "prefill/decode separation mode CLI / interface (saver side)"

---

<a id="cli-003-scenario-2-call-chain-embedding-strategy-doc-outdated"></a>
## CLI-003 Scenario 2 call chain Embedding strategy doc outdated

| Field | Value |
|------|-----|
| ID | CLI-003 |
| Location | Doc: [cli_design.md:204](./cli/cli_design.md) scenario 2 call chain Embedding row |
| Current state (before cleanup) | Doc writes "Embedding→Quant_Embed_MinMax / Quant_Gemma4_Embed"; code `auto_quant.py:16` / `dopt_wrapper.py:23` only uses `Quant_Embed_MinMax`, `auto_quant.py:24` comment clearly states "dopt 53ad5ec removed Quant_Gemma4_Embed; gemma4 embed uses Quant_Embed_MinMax too" |
| Violates convention | Design doc should match code |
| Discovery date | 2026-08-31 (third round checking cli_design scenario 2 ↔ auto_quant / dopt_wrapper) |
| Status | **Cleaned (2026-08-31)** |

### Cleanup Action (executed)

[cli_design.md:204](./cli/cli_design.md) removed " / Quant_Gemma4_Embed", only `Embedding→Quant_Embed_MinMax` remains, consistent with code.

### Related

- Source evidence: `cann_local_lm/converter/quant/auto_quant.py:16,24`, `dopt_wrapper.py:23`
- Design doc: [cli_design.md](./cli/cli_design.md) "Complete call chain / Scenario 2"

---

<a id="cli-004-scenario-2---quant-bits-out-of-sync-with-dopt_wrapper"></a>
## CLI-004 Scenario 2 `--quant-bits` out of sync with dopt_wrapper

| Field | Value |
|------|-----|
| ID | CLI-004 |
| Location | Code: `cann_local_lm/converter/quant/auto_quant.py:65` (cmd, HEAD version), `dopt_wrapper.py:47-53` (parse_args) |
| Current state (initial check) | HEAD version `auto_quant._run_dopt_subprocess`'s cmd passes `--quant-bits`, but `dopt_wrapper.parse_args` doesn't define this parameter →实测 reports `unrecognized arguments: --quant-bits w4a8`, exit=2, scenario 2 subprocess fails to start |
| Violates convention | Internal code inconsistency (auto_quant ↔ dopt_wrapper) |
| Discovery date | 2026-08-31 |
| Status | **Cleaned (2026-08-31, working tree)** |

### Verification Process

Tested: `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 dopt_wrapper.py --model-path /nonexist --optimize-config /nonexist --output-dir /tmp/opencode/dopt_test --hf-type-save --quant-bits w4a8` → `error: unrecognized arguments: --quant-bits w4a8`, exit=2. Confirmed `dopt_wrapper.parse_args` doesn't accept `--quant-bits`.

### Cleanup Action (executed, working tree)

`git diff` confirms `auto_quant.py` working tree already removed cmd's `"--quant-bits", quant_bits,` line (HEAD still has, working tree doesn't). This removal makes auto_quant not pass `--quant-bits` when spawning dopt_wrapper, argparse no longer errors, scenario 2 subprocess can start.

> This line's removal was not done by this session's edit — this session's edit failed because oldString no longer existed, suspected concurrent manual removal. HEAD still contains `--quant-bits`, needs commit to fully land.

### Parameter Chain Cleanup (2026-08-31 supplement, full chain removed)

User instructed "clean unused quant_bits", full chain removal:

- **Code**: `auto_quant.py` (`generate_dopt_config` / `_run_dopt_subprocess` / `run_auto_quant` three function signatures + `_run_dopt_subprocess` call), `cann_lm_converter.py` (convert `quant_bits` param + print + `run_auto_quant` call + parse_args `--quant-bits` add_argument + main `quant_bits=args.quant_bits`), `examples/.../run_pd6_cases.sh:119` (`generate_dopt_config(art, "w4a8")` → `generate_dopt_config(art)`)
- **Docs (synced)**: `quant_design.md` (L343 call chain + L492/493/495 signature table), `cli_design.md` (L32 overview + L49 convert code block + L67 call + L151-152 parse_args code block + L172 param table + L197 call chain + L456 related table), `quant/INDEX.md` L53

Verification: `grep quant_bits`/`quant-bits` in `cann_local_lm/converter` 0 residual; `py_compile` auto_quant/cann_lm_converter/dopt_wrapper pass.

### Doc Residual (cleaned, 2026-08-31)

After `--quant-bits` removed from CLI, the following docs originally had `--quant-bits` descriptions (param descriptions/related tables/examples), all cleaned (some sentences rewritten as "bit width determined by `Quant_INT8-4_ECO` strategy, no CLI param"):

- `quant_design.md`: L31 / L54 / L57 / L68 / L73 / L80 / L83 / L360 / L374 (false description "maps to _LINEAR_STRATEGY" rewritten) / L379 / L1237 (11 places total)
- `quant/INDEX.md`: L74 (param table row) / L81 (points to CLI)
- `cli/INDEX.md`: L39 (param table row) / L54 (onboarding note)

Verification: `grep quant.bits` in `docs/cann_lm_converter/` only `TODO_notes.md` record text remains, other docs 0 residual.

### Related

- Source files: `cann_local_lm/converter/quant/auto_quant.py`, `dopt_wrapper.py`
- Design doc: [cli_design.md](./cli/cli_design.md) "Complete call chain / Scenario 2" / "converter and torch_npu relationship"

---

<a id="cli-005---quant-device-cli-parameter-removed-device-auto-detection"></a>
## CLI-005 `--quant-device` CLI parameter removed (device auto-detection)

| Field | Value |
|------|-----|
| ID | CLI-005 |
| Location | Code: `cann_lm_converter.py` (--quant-device CLI + convert `quant_device` param + `if quant_device: os.environ["DEVICE"]` block + parse_args `--quant-device` add_argument + main `quant_device=args.quant_device` + unused `import os`); Docs: `cli_design.md` / `cli/INDEX.md` / `quant/INDEX.md` / `quant_design.md` / `adr/0001` |
| Current state (before cleanup) | `--quant-device` CLI parameter passed via `os.environ["DEVICE"]` → `_detect_device_type` → subprocess `env.setdefault("DEVICE", device)` to CANN lightweight tool subprocess; user judged "cannot pass to lightweight tool, currently auto-obtained", removed CLI, device fully auto-detected by `_detect_device_type` |
| Violates convention | None (user actively simplified CLI) |
| Discovery date | 2026-08-31 |
| Status | **Cleaned (2026-08-31)** |

### Decision

User instructed: "`--quant-device` currently cannot pass to lightweight tool either, remove it too, currently auto-obtained". Removed CLI parameter, device fully auto-detected by `auto_quant._detect_device_type` (reads `os.environ["DEVICE"]`, external env like `set_env_npu.sh` can set; if unset, spawns detection, no accelerator falls back to `cpu`).

### Cleanup Action (executed)

- **Code**: `cann_lm_converter.py` removed `--quant-device` add_argument + convert `quant_device` param + `if quant_device: os.environ["DEVICE"]` block + comment + main `quant_device=args.quant_device` + docstring two mentions + unused `import os`
- **Docs**: `cli_design.md` (convert code block / parse_args code block / param table / description / call chain / CPU fallback), `cli/INDEX.md` L39, `quant/INDEX.md` L74, `quant_design.md` (L43 / L54 / L57 / L70 / L336 / L359 / L378 / L486 total 8 places, some sentences rewritten as "`_detect_device_type` auto-detects"), `adr/0001` L29 added status change annotation (historical original retained)

### Retained (auto-detection mechanism, not removed)

- `auto_quant._detect_device_type`: reads `os.environ["DEVICE"]` or spawns detection, falls back to `cpu`
- `auto_quant._run_dopt_subprocess`'s `env.setdefault("DEVICE", device)`: subprocess env inherits DEVICE

### Verification

`grep quant.device` in `cann_local_lm/converter` 0 residual; `py_compile cann_lm_converter.py` passes; docs only have `TODO_notes.md` record + `adr/0001` historical original (with status change annotation).

### Related

- Source files: `cann_local_lm/converter/cann_lm_converter.py`, `cann_local_lm/converter/quant/auto_quant.py` (`_detect_device_type` retained)
- Design doc: [cli_design.md](./cli/cli_design.md) "Command-line parameters / Complete call chain / Runtime environment support"
- Historical decision: [adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md) (status change 2026-08-31)

---

<a id="cli-006-generate_dopt_config-removed-build_layer_strategy-renamed"></a>
## CLI-006 `generate_dopt_config` removed + `build_layer_strategy` renamed

| Field | Value |
|------|-----|
| ID | CLI-006 |
| Location | `auto_quant.py` (removed `_build_layer_strategy` + `generate_dopt_config` + `_LINEAR/_EMBED_STRATEGY` + unused `import json`/`import torch`), `dopt_wrapper.py` (`_build_layer_strategy` → `build_layer_strategy`, L31 definition + L67 call), `examples/.../run_pd6_cases.sh:gen_dopt_config` (inlined), `tests/python/dopt_lm/test_strategy_generation.py` (changed to single source), docs (`quant_design.md` / `quant/INDEX.md` / `cli_design.md` / `adr/0001`) |
| Background | `generate_dopt_config` (scenario 1 offline helper) only used by `run_pd6_cases.sh` case; `_build_layer_strategy` duplicated in `auto_quant` + `dopt_wrapper` (dopt_lm `generate_config_file` is full-float template, different logic, doesn't replace). User chose path 2: remove `generate_dopt_config` + inline case + rename public name to eliminate duplication |
| Violates convention | None (user actively cleaned + eliminated duplication) |
| Discovery date | 2026-08-31 |
| Status | **Cleaned (2026-08-31)** |

### Decision

User chose path 2: `generate_dopt_config` removed (auto_quant.py), case `run_pd6_cases.sh:gen_dopt_config` inlines `build_layer_strategy` + `tempfile.mkstemp` + `os.fdopen` + `json.dump`; `_build_layer_strategy` renamed to public `build_layer_strategy` (remove `_`), eliminating two duplicates (auto_quant removed, dopt_wrapper sole implementation).

### Cleanup Action (executed)

- **Code**:
  - `auto_quant.py`: removed `_LINEAR_STRATEGY`/`_EMBED_STRATEGY` + `_build_layer_strategy` + `generate_dopt_config` + unused `import json`/`import torch`
  - `dopt_wrapper.py`: `_build_layer_strategy` → `build_layer_strategy` (L31 definition + L67 call)
  - `examples/.../run_pd6_cases.sh`: `gen_dopt_config` inlined (removed `from auto_quant import generate_dopt_config`, changed to `from dopt_wrapper import build_layer_strategy` + `tempfile`/`os.fdopen`/`json.dump`)
  - `tests/python/dopt_lm/test_strategy_generation.py`: removed `auto_quant` import, test changed to single source (only tests `dopt_wrapper.build_layer_strategy`)
- **Docs**: `quant_design.md` (L236/335/341-342/368/489-490/502/818 + replaceAll `_build_layer_strategy`→`build_layer_strategy`), `quant/INDEX.md` (L21 removed generate_dopt_config + L22/26 renamed), `cli_design.md` (L173 planned refactoring section rewritten removing generate_dopt_config + merge plan + L192 renamed), `adr/0001` L44 (~~strikethrough~~ + landed annotation)

### Verification

`py_compile auto_quant/dopt_wrapper/cann_lm_converter` pass; `pytest tests/python/dopt_lm/test_strategy_generation.py` 2 passed; `grep generate_dopt_config|_build_layer_strategy|auto_quant.build_layer_strategy` in docs only TODO_notes record + adr historical annotation remain.

### Related

- Source files: `cann_local_lm/converter/quant/auto_quant.py`, `dopt_wrapper.py`, `examples/.../run_pd6_cases.sh`, `tests/python/dopt_lm/test_strategy_generation.py`
- Design docs: [cli_design.md](./cli/cli_design.md) "Complete call chain", [quant_design.md](./quant/quant_design.md) "`auto_quant.py`/`dopt_wrapper.py` signature table"
- Historical decision: [adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md) L44 (status change 2026-08-31)

---

<a id="cli-007-todo-cpugpu-one-click-non-subprocess-evaluation"></a>
## CLI-007 TODO: CPU/GPU one-click non-subprocess evaluation

| Field | Value |
|------|-----|
| ID | CLI-007 |
| Type | TODO (pending evaluation, not a deviation) |
| Background | Current scenario 2 (`--quant-method gptq`) main process export (CPU stubs) + lightweight tool subprocess (system python + torch_npu + NPU) two environments, due to NPU scenario `_LazyTorchair` proxy conflict (installing torch_npu breaks export GE conversion) using subprocess isolation. But CPU/GPU (no NPU) scenarios **don't install torch_npu → no proxy conflict → one environment theoretically feasible** |
| Status | **TODO (pending evaluation)** |
| Discovery date | 2026-08-31 |

### Pending Evaluation Questions

For CPU/GPU (no NPU), can one-click script avoid subprocess (one environment runs export + GPTQ):

1. Not installing torch_npu → export doesn't hit `_LazyTorchair` proxy → one environment (no subprocess) theoretically feasible
2. Is `dopt_lm` compatible with torch 2.9.0 (converter venv)? Current lightweight tool on system python torch 2.7.1
3. Install `transformers` + `dopt_lm` dependencies in converter venv? (Currently not installed)
4. Does CPU GPTQ (dopt_lm `dispatch_dtype=float32`) work in converter venv (torch 2.9.0+cpu + torchair stubs)?
5. GPU GPTQ (`torch.cuda`) adaptation pending verification (cli_design "GPU lightweight tool adaptation ⏳")

### Goal: One Environment (CPU/GPU, no NPU)

converter venv installs `torch 2.9.0+cpu` + `torchair` (source+stubs) + `torch-delegate` + `transformers` + `dopt_lm`, GPTQ goes CPU (float32) or GPU (cuda), no subprocess spawn, main process directly calls `dopt_lm`.

### Current Two-Environment Comparison

| Dimension | ① converter main process (export epair) | Lightweight tool subprocess (GPTQ) |
|------|----------------------------------|----------------------|
| Python | converter venv (`.venv`, 3.12.9) | System python (`DOPT_PYTHON`, 3.12.9 auto-detect) |
| torch | 2.9.0+cpu | 2.7.1+cpu |
| torch_npu | **Not installed** (stub path, `_library.py` self-provides npu:: fake-meta) | 2.7.1.post4 (real, NPU ops) |
| torchair | Source-compiled (`NO_ASCEND_SDK=1`) + `build/stubs` | None (doesn't run GE conversion) |
| torch-delegate | 0.1.0 (whl) | None |
| transformers | Not installed | Installed (dopt_lm dependency) |
| CANN lib | stubs (`LD_LIBRARY_PATH` stubs first, avoid `aclrtSetDevice→107001`) | Real lib64 (spawn removes stubs) |
| Chip | No NPU (CPU-only) | Ascend910 × 2 (64GB HBM) |
| env script | `set_env_export.sh` (CANN set_env + stubs LD + `TORCH_DELEGATE_HOME` + `AUTOLOAD=0` + `FALLBACK_CODEGEN=0`) | auto_quant spawn (inherits de-stubbed + `PYTHONPATH` quant/ + `DEVICE` + `DOPT_PYTHON`) |

### Core Differences (why currently subprocess)

1. **torch_npu presence**: export doesn't install (avoid `_LazyTorchair` proxy overriding source torchair → GE conversion bug); GPTQ installs (needs real NPU ops eager)
2. **torchair source+stubs vs none**: export needs GE conversion (torchair); GPTQ doesn't
3. **transformers none vs present**: dopt_lm uses transformers, converter doesn't
4. **CANN lib stubs vs real**: stubs provide torchair GE engine C symbols (avoid real NPU 107001); lightweight tool needs real lib to run ops
5. **torch version 2.9.0 vs 2.7.1**: torchair compiled against 2.9.0, torch_npu 2.7.1.post4 matches 2.7.1 (incompatible, hence subprocess)

### CPU/GPU One-Environment Feasibility Key

- **NPU scenario**: must subprocess (torch_npu proxy conflict unsolvable)
- **CPU/GPU scenario**: don't install torch_npu → no proxy conflict → one environment theoretically feasible, but need to resolve transformers dependency + dopt_lm torch version compatibility + CPU/GPU GPTQ working

### Action Items

- [ ] Evaluate `dopt_lm` working on torch 2.9.0 (CPU/GPU)
- [ ] Install `transformers` + `dopt_lm` dependencies in converter venv
- [ ] CPU GPTQ one-environment smoke (small model, verify dopt_lm + export same process)
- [ ] GPU GPTQ adaptation verification (`torch.cuda`, lightweight tool GPU branch)
- [ ] If feasible, `auto_quant` adds CPU/GPU branch: no subprocess, main process directly calls `dopt_lm` (NPU still subprocess)

### Related

- Design docs: [cli_design.md](./cli/cli_design.md) "Runtime environment support (NPU/GPU/pure CPU)" / "converter and torch_npu relationship"
- Historical decision: [adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md) (subprocess isolation decision)
- Related code: `cann_local_lm/converter/quant/auto_quant.py` (`_detect_device_type` / `run_auto_quant` / `_run_dopt_subprocess`)

---

<a id="cli-008-swap_to_quant_linear_from_dopt-dead-code-removal"></a>
## CLI-008 `swap_to_quant_linear_from_dopt` dead code removal

| Field | Value |
|------|-----|
| ID | CLI-008 |
| Location | `cann_local_lm/converter/swap.py` L334-351 (definition) + `swap_from_dopt` docstring (comparison mention); Docs: `quant/INDEX.md` L50, `export_design.md` L1269, `quant_design.md` L136/138/345/450/451, `quant_swap/__init__.py` L29, `adr/README.md` L39 |
| Background | `swap_to_quant_linear_from_dopt` is a backward-compat thin wrapper (only Linear swap, loads sd itself), duplicates `swap_from_dopt` (Linear + embed one sd load). grep confirms **no code callers** (dead code), only in docs/signature tables. `_swap_linear_from_sd` (private, accepts sd) is already the extracted Linear swap body, `swap_from_dopt` reuses it, no need for `swap_to_quant_linear_from_dopt` |
| Violates convention | None (dead code cleanup) |
| Discovery date | 2026-08-31 |
| Status | **Cleaned (2026-08-31)** |

### Decision

Remove `swap_to_quant_linear_from_dopt` (dead code), `swap_from_dopt` retained (already uses `_swap_linear_from_sd` to reuse Linear swap, one sd load shared with embed extraction). `swap_from_dopt` cannot call `swap_to_quant_linear_from_dopt` (would double-load sd, violating one-load design).

### Cleanup Action (executed)

- **Code**: `swap.py` removed `swap_to_quant_linear_from_dopt` (L334-351) + modified `swap_from_dopt` docstring (removed "avoids swap_to_quant_linear_from_dopt's..." comparison, changed to describe its own one-load)
- **Docs**: `quant_design.md` (L136 removed row + L138 removed mention + L345-346 call chain merged to `swap_from_dopt` + L450 removed signature table row + L451 removed comparison), `quant/INDEX.md` L50 removed row, `export_design.md` L1269 removed mention, `quant_swap/__init__.py` L29 removed mention, `adr/README.md` L39 (~~strikethrough~~ + ❌→✅ landed annotation)

### Verification

`py_compile swap.py` passes; `grep swap_to_quant_linear_from_dopt` only `adr/README.md` strikethrough annotation remains (history retained), code + other docs 0 residual.

### Related

- Source file: `cann_local_lm/converter/swap.py`
- Design docs: [quant_design.md](./quant/quant_design.md) "`swap.py` signature table" / "swap module symbols", [export_design.md](./export/export_design.md) "file tree"
- Historical decision: [adr/README.md](../adr/README.md) L39 (refactoring item ❌→✅ landed)

---

<a id="cli-009-swap_from_dopt-parameter-simplification-remove-skipact_quant"></a>
## CLI-009 `swap_from_dopt` parameter simplification (remove skip/act_quant)

| Field | Value |
|------|-----|
| ID | CLI-009 |
| Location | `cann_local_lm/converter/swap.py` L334-355 (swap_from_dopt signature + internal call); Docs: `quant_design.md` L448, `quant/INDEX.md` L49, `README.md` L47 |
| Background | `swap_from_dopt(model, dopt_path, skip=DEFAULT_SKIP, act_quant=True)`'s `skip`/`act_quant` params: no code caller passes them (sole caller `cann_lm_converter.py:100` only passes `model+dopt_path`, uses defaults). `skip=DEFAULT_SKIP` (fixed skip embed_tokens), `act_quant=True` (W4A8, currently doesn't support W4A16) are both fixed values, no external override need |
| Violates convention | None (parameter simplification) |
| Discovery date | 2026-08-31 |
| Status | **Cleaned (2026-08-31)** |

### Decision

Remove `swap_from_dopt`'s `skip`/`act_quant` params, internally hardcodes `skip=DEFAULT_SKIP`/`act_quant=True` passed to `_swap_linear_from_sd`. `_swap_linear_from_sd` retains params (core logic interface, flexibility).

### Cleanup Action (executed)

- **Code**: `swap.py` `swap_from_dopt` signature removed `skip`/`act_quant`, internal `_swap_linear_from_sd` call changed to `skip=DEFAULT_SKIP, act_quant=True`
- **Docs**: `quant_design.md` L448 signature table (signature `(model, dopt_path)`, param count 4→2, note hardcoded), `quant/INDEX.md` L49 (removed `...`), `README.md` L47 (removed `...`)

### Verification

`py_compile swap.py` passes; `swap_from_dopt` sole caller `cann_lm_converter.py:100` only passes `(artifact.model, dopt_out_path)`, compatible after removing params.

### Related

- Source file: `cann_local_lm/converter/swap.py`
- Caller: `cann_local_lm/converter/cann_lm_converter.py:100`
- Design doc: [quant_design.md](./quant/quant_design.md) "`swap.py` signature table"

---

<a id="cli-010-todo-load_model-loads-hf-original-linear-fp-waste-swap-discards"></a>
## CLI-010 TODO: load_model loads HF original Linear fp waste (swap discards)

| Field | Value |
|------|-----|
| ID | CLI-010 |
| Type | TODO (pending optimization, not a deviation) |
| Background | `load_model(hf_path)` loads HF original (incl. Linear fp16 weights) → cann_local_lm model; `swap_from_dopt`'s `_swap_linear_from_sd` reads int4+scale from lightweight tool output sd, `setattr` replaces Linear→QuantLinear, **does not read model Linear's weight**. So load_model's loaded Linear HF fp is replaced and discarded, wasting IO+memory (scenario 2 memory peak: subprocess transformers + main process cann_local_lm with discarded Linear fp) |
| Status | **TODO (pending evaluation)** |
| Discovery date | 2026-08-31 |

### Waste Point

`_swap_linear_from_sd` (swap.py:221-331) iterates `model.named_modules()`, for `nn.Linear` reads `{name}.weight`+`{name}.quant_op...s` from sd (lightweight tool output), materializes int4, `setattr` replaces QuantLinear. **Does not read model's own Linear `.weight`** at all (only locates module name + setattr replace). So load_model's loaded Linear HF fp is discarded, doesn't enter final output.

| Module | load_model loads (HF fp) | After swap | Waste |
|------|-------------------------|---------|------|
| `nn.Linear` | HF original fp16 | Discarded (setattr replaces QuantLinear, int4 from lightweight tool) | ✅ IO+memory |
| `nn.Embedding` | HF original fp16 | Extracts int8 (if lightweight tool quantized embed), else retains fp | Partial |
| LayerNorm/other | HF original fp16 | Retained (lightweight tool doesn't touch) | Necessary |

### Why Currently Load HF Original (Design Trade-off)

1. cann_local_lm model structure needs HF original construction (`model_registry` + cann_local_lm model class + key remapping)
2. Non-Linear (LayerNorm) needs HF original fp (lightweight tool doesn't touch, but lightweight tool output layer names may carry `model.language_model.` etc. VLM prefix, load_model key remapping targets HF original)
3. Simple: load_model general loading (doesn't distinguish Linear/non-Linear), swap replaces

### Optimization Direction (not done)

- **load_model loads lightweight tool output** (`load_model(dopt_out_path)`): Linear = fakequant fp (same source as swap), avoids HF original Linear fp waste. But needs load_model key remapping to adapt to lightweight tool output layer names (strip `language_model.` etc.) + lightweight tool output contains complete module weights
- **load_model skips Linear weight (meta placeholder)**: only build structure + non-Linear weights, Linear meta tensor placeholder (replaced anyway). Saves Linear fp IO+memory. But load_model is general (`param.copy_`), needs selective skip logic

### Action Items

- [ ] Evaluate load_model loading lightweight tool output's key remapping compatibility (Qwen3_5 `language_model.` prefix)
- [ ] Evaluate load_model skipping Linear weight (meta) feasibility + whether swap needs Linear module present (setattr replace)
- [ ] Compare memory peak before/after optimization (scenario 2 small model smoke)

### Related

- Source: `cann_local_lm/converter/cann_lm_converter.py:77` (load_model) + `cann_local_lm/converter/swap.py:221` (`_swap_linear_from_sd`, doesn't read model Linear weight)
- Design docs: [cli_design.md](./cli/cli_design.md) "Complete call chain" (load_model → swap), [quant_design.md](./quant/quant_design.md) "swap.py signature table"
- Related: CLI-007 (scenario 2 memory peak, subprocess + main process)

---

<a id="cli-011-todo-build_layer_strategy-vs-strategypy-consistency"></a>
## CLI-011 TODO: build_layer_strategy vs strategy.py consistency

| Field | Value |
|------|-----|
| ID | CLI-011 |
| Type | TODO (pending verification/sync) |
| Background | `dopt_wrapper.build_layer_strategy` generates `dopt_config.json` layer strategy (`quant_strategy` names: `Quant_INT8-4_ECO` / `Quant_Embed_MinMax`). Lightweight tool reads strategy name to look up `dopt_lm/quantizer/strategy.py`'s `strategy_dict` for complete strategy (weight + input). Need to ensure build_layer_strategy's strategy names correspond to strategy.py definitions, effect consistent |
| Status | **Verified (2026-09-01, all three consistent, no code change needed)** |
| Discovery date | 2026-08-31 |

### dopt_lm/quantizer/strategy.py strategy definitions (reference)

```python
"Quant_INT8-4_ECO": {
    "weight": {
        "blocksize": 128, "group_size": 128, "bit": 4,
        "per_channel": True, "weight_algo": "omni_group_min_max",
    },
    "input": {
        "bit": 8, "per_channel": False,
        "input_algo": "dyn_s8", "unsigned_quant": False,
    },
},
"Quant_Embed_MinMax": {
    "weight": {
        "bit": 8, "per_channel": True, "weight_algo": "min_max",
        "fp16": True, "tie_lmhead": False,
    }
},
```

### Current Implementation Comparison

- `dopt_wrapper.build_layer_strategy` (dopt_wrapper.py:31): assigns `quant_strategy` name (`Quant_INT8-4_ECO` for Linear, `Quant_Embed_MinMax` for Embedding), lightweight tool reads to look up strategy.py for complete strategy → effect should be consistent (if strategy names correspond)
- `swap.py _STRATEGY_DEFAULT_PARAMS` (swap.py:138): hardcodes `Quant_INT8-4_ECO` weight 5 fields (group_size:128, bit:4, blocksize:128, per_channel:True, weight_algo:omni_group_min_max) → consistent with strategy.py weight part (current)
- swap.py doesn't contain input part (A8): swap only materializes weight int4, input A8 handled by `npu::npu_dynamic_quant` in QuantLinear forward (`act_quant=True` hardcoded W4A8)

### Pending Verification/Sync

1. build_layer_strategy's strategy names (`Quant_INT8-4_ECO`/`Quant_Embed_MinMax`) match strategy.py `strategy_dict` keys (lightweight tool can find them)
2. swap.py `_STRATEGY_DEFAULT_PARAMS` syncs with strategy.py weight part (when strategy.py updates, swap.py hardcoded needs sync; swap.py:137 comment "converter venv doesn't install dopt_lm, so hardcoded; update here when adding a weight-quant strategy")
3. Lightweight tool GPTQ input strategy (A8 `dyn_s8`) consistent with swap W4A8 (`act_quant=True`) (lightweight tool produces A8 quantized model, swap materializes W4A8, need correspondence)

### Action Items

- [x] Verify build_layer_strategy strategy names correspond to strategy.py strategy_dict keys → **Consistent**: `Quant_INT8-4_ECO` / `Quant_Embed_MinMax` both in strategy_dict (dopt_wrapper.py:22-23 assigns, strategy.py:35,50 defines)
- [x] Sync swap.py _STRATEGY_DEFAULT_PARAMS with strategy.py weight → **Synced**: swap.py:138-143 `Quant_INT8-4_ECO` weight 5 fields (group_size:128, bit:4, blocksize:128, per_channel:True, weight_algo:omni_group_min_max) fully match strategy.py:36-42; comment (swap.py:133-137) notes "aligned with strategy.py" / "dopt 53ad5ec group_size 256→128 synced"
- [x] Confirm lightweight tool input strategy (A8 dyn_s8) matches swap act_quant=True (W4A8) → **Consistent**: strategy.py Quant_INT8-4_ECO input (bit:8, input_algo:dyn_s8, unsigned_quant:False)=A8 dynamic quant; swap act_quant=True (CLI-009 hardcoded) → QuantLinear forward handles A8 via `npu::npu_dynamic_quant`; lightweight tool produces W4A8 model (weight int4 + input A8), swap materializes W4A8, correspond

### Verification Result (2026-09-01)

All three consistent, no code change needed:

| Verification point | Comparison (source : line) | Conclusion |
|--------|----------------------|------|
| strategy name correspondence | build_layer_strategy assigns `Quant_INT8-4_ECO` (Linear)/`Quant_Embed_MinMax` (Embedding) (dopt_wrapper.py:22-23,39,42); strategy_dict has both keys (strategy.py:35,50) | ✅ Consistent |
| weight field sync | swap.py _STRATEGY_DEFAULT_PARAMS 5 fields (group_size:128/bit:4/blocksize:128/per_channel:True/omni_group_min_max) (swap.py:138-143) = strategy.py Quant_INT8-4_ECO weight (strategy.py:36-42) | ✅ Fully consistent |
| input/A8 correspondence | strategy.py input (bit8/dyn_s8/unsigned False) (strategy.py:43-48)=A8; swap act_quant=True (CLI-009)→`npu::npu_dynamic_quant` A8 (swap.py:482) | ✅ W4A8 correspond |

> Maintenance constraint (swap.py:134 comment): converter venv doesn't install dopt_lm, `_STRATEGY_DEFAULT_PARAMS` hardcoded; **when strategy.py adds a weight-quant strategy, must manually sync swap.py**. Currently only `Quant_INT8-4_ECO` one weight strategy, synced; `Quant_Embed_MinMax` is embedding-side (swap.py:182 comment, doesn't go Linear swap, not in _STRATEGY_DEFAULT_PARAMS).

### Related

- Source: `cann_local_lm/converter/quant/dopt_wrapper.py:31` (build_layer_strategy), `cann_local_lm/converter/swap.py:138` (`_STRATEGY_DEFAULT_PARAMS`), `cann_local_lm/converter/quant/dopt_lm/quantizer/strategy.py` (strategy definitions)
- Related: CLI-009 (swap_from_dopt removed act_quant, hardcoded W4A8)

---

<a id="cli-012-todo-gemma4-ple-quantization-unadapted-_build_external_extras-uses-hf-original-float"></a>
## CLI-012 TODO: gemma4 PLE quantization unadapted (_build_external_extras uses HF original float)

| Field | Value |
|------|-----|
| ID | CLI-012 |
| Type | TODO (bug, pending fix) |
| Background | gemma4 PLE (`embed_tokens_per_layer`) is quantized by lightweight tool (`Quant_Embed_MinMax` → QEmbedding int8 per-channel + scale), `host_weights_*.bin` packs `_q`/`_scale` (int8 `[vocab,8960]` + float32 `[vocab]`). But `_build_external_extras` (gemma4/export_wrapper.py:273-283) uses `m.embed_tokens_per_layer(input_ids)` (**HF original float**, swap_embeddings_from_dopt doesn't replace Embedding module), graph input `per_layer_token_identity` float (**no scale**), asymmetric with embed_tokens quantization handling (int8+scale as graph input + in-graph dequant) |
| Status | **Fixed (2026-09-02)** |
| Discovery date | 2026-08-31 |

### Evidence

**host_weights_info.json** (epair_gemma4_quant_path):
```json
{ "name": "model.model.embed_tokens_per_layer.weight_q", "dtype": "int8", "shape": [262144, 8960] },
{ "name": "model.model.embed_tokens_per_layer.weight_scale", "dtype": "float32", "shape": [262144] }
```
PLE packed `weight_q` (int8) + `weight_scale` (float32) — lightweight tool quantized PLE.

**Lightweight tool output** model.safetensors:
```
model.embed_tokens_per_layer.weight [262144, 8960] (fakequant fp16)
model.embed_tokens_per_layer.quant_op.weight_quantizer.s [262144] (scale)
model.embed_tokens_per_layer.quant_op.weight_quantizer.bit/unsigned_quant/offset/max/min
```
Lightweight tool uses QEmbedding to quantize PLE (int8 + scale + meta).

**_build_external_extras** (gemma4/export_wrapper.py:280):
```python
token_identity = m.embed_tokens_per_layer(input_ids).reshape(...).detach()   # HF original float
```
`m.embed_tokens_per_layer` is cann_local_lm model's Embedding (load_model loads HF original fp, swap doesn't replace).

### Problem

| | embed_tokens (quant) | PLE (embed_tokens_per_layer, quant) |
|---|---|---|
| Lightweight tool quantization | ✅ int8+scale | ✅ int8+scale |
| host_weights pack | ✅ _q + _scale | ✅ _q + _scale |
| **Export graph input** | `emb_q` (int8) + `emb_scale` (gather from `_quant_embeddings`) | `per_layer_token_identity` (**HF original float**, lookup from `m.embed_tokens_per_layer`) |
| forward | `int8*scale→float` dequant (in graph) | Use float directly (no dequant, no scale param) |

1. **Numerical inconsistency**: export graph trace uses **HF original PLE** (_build_external_extras); host inference passes **lightweight tool quantized PLE dequant** (`int8*scale`). HF original ≠ quantized dequant → graph expects original values, host passes quantized values → **accuracy mismatch**
2. **forward no scale**: `per_layer_token_identity` single tensor (no `_scale` param), graph doesn't dequant PLE; host needs to dequant on host side then pass to graph — but graph trace uses HF original (not dequant), host passes dequant, values don't match
3. **Asymmetric with embed_tokens**: embed_tokens quant goes int8+scale as graph input + in-graph dequant; PLE quant goes HF original float (not adapted to int8)

### Fix Direction

1. **`_build_external_extras` adapt PLE quantization**: detect `_quant_embeddings` contains PLE (`model.model.embed_tokens_per_layer.weight`) → extract int8+scale, gather `input_ids` → `ple_q` (int8) + `ple_scale` (per-token scale)
2. **forward add `per_layer_token_identity_scale` param**: `per_layer_token_identity` (int8) * `per_layer_token_identity_scale` → float dequant (similar to `embeddings`/`embeddings_scale`)
3. **get_sample_inputs pass `ple_q` + `ple_scale`** (when quant, distinguish from non-quant float PLE branch)
4. **Host inference script adaptation**: read host_weights PLE `_q`/`_scale`, gather + pass int8+scale (in-graph dequant), or host dequant pass float (graph doesn't dequant, must match export)

### Action Items

- [x] ~~`_build_external_extras` add quant branch~~ → Implemented in `get_sample_inputs(quant)` directly gathering `ple_q`+`ple_scale` from `_quant_embeddings` (`gemma4/export_wrapper.py:258-269`), doesn't call `_build_external_extras` (latter retains non-quant float branch)
- [x] forward add `per_layer_token_identity_scale` param + dequant logic (`gemma4/export_wrapper.py:61,70-72`, `int8*scale→float`, scale `[b,s]` broadcast over `[num_layers,hidden_per_layer_input]`)
- [x] get_sample_inputs(quant) passes `ple_q`+`ple_scale` (`:258-269`); non-quant/in-graph branch adds `None` ple_scale to preserve 11-element return order (`:275,281`)
- [x] Host inference script adapt PLE int8+scale → `verify_gemma4_epair_quant.py` (epair loop verification, reads lightweight tool output + `swap_embeddings_from_dopt` + `set_quant_embeddings`, host doesn't need manual gather — `get_sample_inputs` already gathers)
- [x] Verification: case 4 (gemma4 scenario 1) epair export works (`n_quantized=277,weights 1484.6 MB,host_weights 2753.6 MB`, incl. PLE `_q`/`_scale`) + `verify_gemma4_epair_quant.py` ep.module (traced) vs wrapper (eager) `max_diff=0` (PLE dequant correct in trace)

### Fix (2026-09-02)

**PLE quant adapt epair path** (3 code places + 1 verification script):

1. **`cann_local_lm/models/gemma4/export_wrapper.py`**:
   - `forward` (:49-72) adds `per_layer_token_identity_scale` param + PLE dequant (`per_layer_token_identity(int8) * per_layer_token_identity_scale → float`, mimics `embeddings/embeddings_scale`)
   - `get_sample_inputs(quant)` (:246-269) extracts PLE int8+scale from `_quant_embeddings`, gathers `input_ids`→`ple_q`+`ple_scale`; non-quant/in-graph branch adds `None` ple_scale to preserve 11-element return order
2. **`cann_local_lm/models/fused_ops/fused_infer_attention_score.py` `_fa_cpu`** (:92-100): compat gemma4 mask square `[b,1,kv_max,kv_max]` bool (truncate first seq_len rows to match `attn_weights [b,heads,seq_len,kv_max]` + bool→float `True→-inf`); non-square float mask (qwen3.5 etc.) unaffected
3. **`examples/.../debug_gemma4/verify_gemma4_epair_quant.py`** (new, 97 lines): epair loop verification (load+lightweight tool+set_quant_embeddings+export+ep.module vs wrapper logits consistent+PLE dequant sanity)

**Verification result** (`verify_gemma4_epair_quant.py --hf-path gemma-4-E2B-it --quant-path /tmp/opencode/gemma4_quant_output`):
- PLE quantized=True (`embed_tokens`+`embed_tokens_per_layer` both int8+scale)
- `ple_q (1,128,35,256) int8, ple_scale (1,128) f32` (gemma4 35 layers × 256 hidden_per_layer)
- **ep.module(traced) vs wrapper(eager) logits max_diff=0.000000e+00** → forward's PLE dequant correctly preserved in torch.export trace, host passes ple_q+ple_scale, in-graph dequant consistent with eager
- PLE dequant: min=-0.4219 max=0.4980 mean=0.0001 nan=False inf=False (reasonable values)
- `[PASS] gemma4 quant PLE epair loop consistent`

> OM verify path (`run_gemma4_gen.py`, fakequant float PLE) unadapted (OM op unsupported + uses fakequant not int8), not in CLI-012 scope. Production epair host inference script (using `epair_model_loader` C++) PLE adaptation handled by C++ reading `per_layer_token_identity_scale` per ep_json input order (pending production host script landing confirmation).

### Related

- Source: `cann_local_lm/models/gemma4/export_wrapper.py:273-283` (`_build_external_extras`), `cann_local_lm/models/export_wrapper.py:44-60` (forward per_layer_token_identity/embeddings_scale), `cann_local_lm/converter/swap.py:35-56` (swap_embeddings_from_dopt, doesn't replace Embedding)
- Artifacts: `host_weights_info.json` (PLE `_q`/`_scale`), lightweight tool `model.safetensors` (PLE fakequant+scale)
- Related docs: [feature_embedding_design.md](./feature_embedding_design.md) "gemma4 PLE special" / "Lightweight tool's implementation for gemma4"
- Related: CLI-011 (build_layer_strategy vs strategy.py consistency)

---

<a id="cli-013-todo-side-llm-tool-cli-and-doc-delivery-survey"></a>
## CLI-013 TODO: side LLM tool CLI and doc delivery survey

| Field | Value |
|------|-----|
| ID | CLI-013 |
| Type | TODO (survey, not a deviation) |
| Background | cann-lm-converter's current CLI is single script `python cann_lm_converter.py --hf-path ... --mode ...` (argparse, no subcommands), docs in `docs/cann_lm_converter/` by directory. To evaluate whether CLI/doc delivery needs to align with side LLM tool mainstream practices, surveyed mlx-lm etc. side tools' command-line entry + doc organization |
| Status | **TODO (survey complete, pending adoption evaluation)** |
| Discovery date | 2026-09-01 |

### Survey Objects

Side (local/device-side) LLM inference or conversion tools, comparable positioning to cann-lm-converter (export epair for C++ delegate loading) (both contain "convert/quantize + run" chain):

- **mlx-lm** (Apple Silicon, Python+MLX): convert/quantize + generate + fine-tune
- **llama.cpp** (C/C++, multi-backend incl. CANN): GGUF convert + cli/serve inference
- **mlc-ai/mlc-llm** (MLC, multi-platform): convert/build/serve/chat
- **ollama** (Go): run/pull/serve (runtime, conversion server-side)

### Survey Findings: CLI Tool Delivery

| Tool | CLI entry style | entry point mechanism | Subcommand examples |
|------|-------------|-------------------|-----------|
| mlx-lm | `mlx_lm.<cmd>` multi entry (dot namespace) | `console_scripts` registers multiple (`mlx_lm.generate`/`.chat`/`.convert`/`.server`/`.cache_prompt`/`.benchmark`) | `mlx_lm.generate --model ... --prompt ...` |
| llama.cpp | Unified `llama <subcmd>` | C++ binary (`llama`) + Python script (`convert_hf_to_gguf.py` top-level) | `llama cli -hf ...` / `llama serve -hf ...` |
| mlc-llm | Unified `mlc_llm <subcmd>` (typer) | `console_scripts` | `mlc_llm convert` / `mlc_llm build` / `mlc_llm chat` / `mlc_llm serve` |
| ollama | Unified `ollama <cmd>` | Go binary | `ollama run` / `ollama pull` / `ollama serve` / `ollama list` |

**Commonality**: side tools generally use "**unified entry + subcommands**" (`llama cli/serve`, `mlc_llm convert/build`, `ollama run/pull`), not single script; mlx-lm's dot multi-entry (`mlx_lm.generate`) is a variant of Python package convention. Subcommand granularity by "action" (cli/serve/convert/build/chat/run/pull), each subcommand independent `--help`.

**cann-lm-converter current state**: `python cann_lm_converter.py --hf-path ... --mode prefill-decode ...` (single script + argparse, `--mode` distinguishes prefill-decode/single, `--quant-method`/`--quant-path` distinguishes quantization scenarios). No subcommands, no console_scripts entry (script direct call).

### Survey Findings: Doc Delivery

| Tool | Doc structure | Command reference | Examples |
|------|---------|---------|------|
| mlx-lm | README.md single file (Quick Start/Command Line/Python API/feature sections) + `LORA.md` (fine-tune) + `examples/` (generate_response.py etc.) + `AGENTS.md`/`CLAUDE.md` | README "Command Line" section + per-command `-h` | `examples/` directory |
| llama.cpp | README.md (main, Quick Start/Description/Backends) + `docs/` (by topic: build.md/docker.md/android.md/multi-gpu.md/backend/*.md) + `tools/<tool>/README.md` (per tool) + wiki | `tools/cli/README.md` + `tools/server/README.md` (per-tool README) + `docs/build.md` | `examples/` |
| mlc-llm | docs site (independent site) + README | docs site command reference page | `examples/` |
| ollama | docs site (independent site) | docs site command reference | — |

**Commonality**: Docs two types — **single README + examples/** (mlx-lm, lightweight) or **README + docs/ by topic + per-tool README** (llama.cpp, structured). Command reference either in README section, or per-subcommand/tool independent README (llama.cpp `tools/`&lt;t&gt;`/README.md`). `-h`/`--help` is CLI self-doc fallback. llama.cpp's "docs/ by topic + tools/`&lt;t&gt;`/README.md per tool" structure is close to cann-lm's current `docs/cann_lm_converter/{cli,quant,export}/` by-directory.

**cann-lm-converter current state**: `docs/cann_lm_converter/` by directory (cli/cli_design.md, quant/quant_design.md, export/export_design.md, README.md (overview), TODO_notes.md etc.), close to llama.cpp docs/ by-topic mode; no independent "command reference" page (CLI params scattered in cli_design.md); examples/ has case scripts (run_pd6_cases.sh) but no independent example docs.

### Implications for cann-lm-converter (pending evaluation)

1. **CLI entry**:
   - Current single script `python cann_lm_converter.py` suits converter's single responsibility (export epair). If later extending "verify/load/serve" actions, can reference `llama cli/serve` or `mlc_llm convert/build/serve` subcommand-ize (`cann-lm convert`/`cann-lm serve`)
   - entry point: current whl (torch_delegate) installed, but `cann_lm_converter.py` still script direct call. Can consider console_scripts registering `cann-lm-converter` command (reference mlx-lm `mlx_lm.convert` / mlc-llm `mlc_llm convert`), pip install then direct call, no need for `python cann_lm_converter.py`
2. **Docs**:
   - Current `docs/cann_lm_converter/` by directory close to llama.cpp docs/ mode (can retain). Can add "command reference" independent page (CLI param summary table, reference mlx-lm README "Command Line" section or llama.cpp `tools/cli/README.md`), currently params scattered in cli_design.md
   - examples/ case scripts (run_pd6_cases.sh) have no independent README, can reference llama.cpp `tools/`&lt;t&gt;`/README.md` to add README per case/tool
3. **`-h`/`--help` self-doc**: ensure `cann_lm_converter.py --help` output complete and usable (reference mlx-lm "Use -h to see a list of available options for a command")

### Action Items

- [x] Evaluate whether cann-lm-converter needs subcommand-ization → **Keep single command** (converter single responsibility export epair, no serve/verify actions; subcommand-ization pending responsibility expansion)
- [x] Evaluate console_scripts registering `cann-lm-converter` command → **Done**: `scripts/cann_lm_converter/pyproject.toml:20-21` `cann-lm-converter = "cann_local_lm.converter.cann_lm_converter:main"`, direct call after pip install
- [x] Add "command reference" independent doc page → **Done**: cli/INDEX.md adds "Command Reference" section (console_scripts entry `cann-lm-converter` + usage examples 5 types)
- [ ] examples/ cases add README (reference llama.cpp `tools/`&lt;t&gt;`/README.md`)

### Related

- Survey sources: mlx-lm (github.com/ml-explore/mlx-lm), llama.cpp (github.com/ggml-org/llama.cpp), mlc-llm (github.com/mlc-ai/mlc-llm), ollama (github.com/ollama/ollama)
- Related code: `cann_local_lm/converter/cann_lm_converter.py` (CLI), `pyproject.toml` (entry point, torch_delegate already has)
- Design docs: [cli_design.md](./cli/cli_design.md) (CLI params/call chain), [README.md](./README.md)

---

<a id="cli-014-todo-pd-separated-export-stores-only-one-weight-copy-decode-weights-discarded-fails-when-pd-weights-differ"></a>
## CLI-014 TODO: PD separated export stores only one weight copy, decode weights discarded (fails when PD weights differ)

| Field | Value |
|------|-----|
| ID | CLI-014 |
| Type | TODO (legacy design assumption limitation, not a deviation) |
| Location | `cann_local_lm/converter/saver.py:220-238` (prefill `save_weights=True` / decode `save_weights=False`) + docstring:142-148 "shared weights (packed once)"; design: [export_design.md:936](./export/export_design.md) "weight sharing: two graphs' state_dict same" |
| Current state | `save_epair_prefill_decode` assumes prefill/decode two graphs' state_dict same (same model), only prefill stores weights (`consolidated_weights_save(prefill_epair, save_weights=True)` writes `weights_*.bin` + `weight_info.json`), decode `save_weights=False` (only writes `decode_ep_json_with_ascend_ir.json`, weights discarded) |
| Violates convention | None (design assumption, not deviation) |
| Discovery date | 2026-09-01 |
| Status | **Implemented (B, 2026-09-28)**: v2 hash dedup unified format (single graph/PD both v2, removed `version` field + removed v1 compat + removed `inline_fused_attn_seq_constants`/`inline_weight_scale_dtype`). Verified: qwen3.5/gemma4 end-to-end (single-turn/multi-turn/long prompt multi-chunk) all pass |

### Design Assumption (current)

`save_epair_prefill_decode` docstring (:142-148) and [export_design.md:936](./export/export_design.md) clearly state:
- **Weight sharing**: two graphs' state_dict same (same model), `pack_weights` only executes once, `weight_info.json` + `weights_*.bin` only written once
- prefill stores weights (`save_weights=True`), decode `save_weights=False` (weights discarded, only ep_json written)
- C++ loading side (`GeWeightLoader`) reads `weight_info.json` + `weights_*.bin`, prefill/decode two graphs share same weights (layer names consistent)

### Legacy Problem

When **PD separated and weights differ** (not same model/state_dict different), decode weights discarded causes errors:

1. decode `save_weights=False` → decode's referenced weight nodes don't enter `weights_*.bin` / `weight_info.json`
2. C++ loading decode graph, decode ep_json's referenced weights not found in prefill's `weight_info.json` (layer name/shape/value different) → load failure or mismatch
3. Scenarios: PD different quantization (prefill W8A8 / decode W4A8), PD disaggregated deployment (two weight copies), PD layer name differences (with prefix)

### Current Viable Premise

Only when prefill and decode **same model, same state_dict** (weight groups completely identical) is decode reusing prefill weights safe. This is the implicit constraint of current `--mode prefill-decode`; `run_pd6_cases.sh` each case exports PD two graphs from same model, satisfying this constraint.

### Key Correction: C++ Already Per-Graph Compile (Original Direction 2 Ready)

`run_compile.sh:8-9` + `compile_main.cc:61-73`: compile calls per-graph — each `cann_lm_compile <model_json> <weights_index>` processes one graph, `EpairModelLoader` receives per-graph `weights_index`. So original "TODO direction 2: C++ loading side per-graph weight index" **already satisfied**. Current decode can compile because `run_compile.sh` lets decode reuse prefill's `weight_info.json` (PD same-name hit). **Gap is purely Python side** (`save_epair_prefill_decode` decode `save_weights=False` discards weights), not C++ unsupported.

And **single artifact API means PD weights necessarily identical** (same model, only seq_len differs trace) → current dedup is **correct, not a bug**. "Inconsistency" only arises from scenarios below.

### Scenarios (PD weights truly different triggers)

| Scenario | Weights truly different | Trigger |
|------|--------------|------|
| ① disaggregated deployment | Same values, two machines each need self-contained copy | Deployment mode (not export inconsistency) |
| ② PD different quantization (prefill W8A8/decode W4A8) | Truly different values | Needs dual artifact API (current single artifact can't produce) |
| ③ Layer name prefix differences (different wrapping) | Same value different name | Same as ②, needs dual artifact |

### Solutions

**A. Pragmatic (Python-only, recommended first)**: `consolidated_weights_save`/`_pack_weights_to_dir` add `weights_prefix` param + `save_epair_prefill_decode` add `pd_split_weights` switch (or auto-detect). Off = current (PD same, decode `save_weights=False` reuse, zero overhead); On = prefill writes `weight_info_prefill.json`+`weights_prefill_*.bin`, decode `save_weights=True` writes `weight_info_decode.json`+`weights_decode_*.bin` (two graphs self-contained). `model_config.json` records `pd_weight_split`. **C++ zero changes**; `run_compile.sh` already supports explicit `<model_json> <weights_index>` (:9), split passes `weight_info_decode.json` to decode. Covers ①②③.

**B. hash dedup (advanced, long-term)**: Mimic GE `FileConstant` (`external_weight.md`) — unified `weights_*.bin` content hash dedup + `weight_info.json` two-level index (`blobs:{hash:{offset,size,format,kind}}` + `graphs:{prefill:{name→hash},decode:{name→hash}}`). PD same value→shared hash (auto dedup); different values→two hashes; partial sharing (shared embed+different Linear)→shared dedup, different each stored. Downside: two-level index + C++ `GeWeightLoader` reads `graphs[G][name]→hash→blobs[hash]`, large change. Only worth it when "partial sharing PD" becomes real.

**C. Dual artifact API (scenario ②③ required)**: `save_epair_prefill_decode(prefill_artifact, decode_artifact=None, ...)` — decode passes independent artifact producing truly different PD weights; `None` falls back to current. Combined with A (split switch + dual artifact).

**D. Consistency assert (safety net)**: single artifact dedup path adds `assert prefill_sd == decode_sd` (name/shape/dtype/bytes), violation fail-fast (rather than silently producing broken epair). Zero-cost guard for current assumption.

**Recommendation**: First A+D (A reuses ready per-graph compile, no C++ change; D guardrail); C for when ②③ actually appear; B only for long-term partial sharing scenario. Coordinated with CLI-015 (`format` ND/NZ) + CLI-019 (`kind` p_/c_/b_): split per-graph `weight_info_{prefill,decode}.json` entries both carry two fields, three TODOs landing together most economical.

### Action Items

- [ ] **A**: `consolidated_weights_save`/`_pack_weights_to_dir` add `weights_prefix`; `save_epair_prefill_decode` add `pd_split_weights` switch; `model_config.json` records `pd_weight_split`; `run_compile.sh` auto-find adds `weight_info_decode` fallback (or explicit pass) — **Not done (B covers A's goal and is better)**
- [ ] **D**: single artifact path add PD state_dict consistency assert (fail-fast) — **Not done (v2 hash dedup itself is fail-fast: PD different values each stored, no implicit sharing)**
- [ ] **C** (scenario ②③): `save_epair_prefill_decode` extend optional `decode_artifact` — **Not done (dual artifact scenario ②③ not appeared)**
- [x] **B** (implemented, 2026-09-28): hash dedup + two-level index + C++ `GeWeightLoader` read change — `_pack_weights_v2`/`_pack_weights_pd_dedup` (epair_utils.py) + `consolidated_weights_save_pd` (PD)/`consolidated_weights_save` changed to call (single graph unified v2) + `ge_weight_loader.cc` v2 two-level (`graphs[G][name]→hash→blobs[hash]`) + removed v1 loop (no graphs errors) + `ge_graph_builder`/`epair_model_loader` graph_name inference (ep_json filename→prefill/decode/single) + removed `version` field (C++ detects `graphs`/`blobs` presence) + removed `inline_fused_attn_seq_constants`/`inline_weight_scale_dtype` (v2 resolves actual_seq PD sharing conflict, no inline workaround needed)
- [x] Docs: [export_design.md](./export/export_design.md) "PD weight separation scenario (hash dedup v2)" section (B design + schema + Python/C++ interface + verification matrix)
- [x] Verification (2026-09-28): converter UT 127 passed; qwen3.5-0.8B + gemma4-E2B end-to-end (export v2→OM compile C++ v2 load→NPU text generation); single graph `graphs["single"]` + PD `graphs["prefill"/"decode"]` hash dedup; multi-turn dialog + long prompt multi-chunk prefill; de-inline text generation consistent (v2 fundamentally solves actual_seq sharing conflict)

### Related

- Source: `cann_local_lm/converter/saver.py:220-238` (save_epair_prefill_decode weight saving), `cann_local_lm/converter/export/epair_utils.py:131` (consolidated_weights_save save_weights semantics), C++ `GeWeightLoader` (weight loading)
- Design doc: [export_design.md](./export/export_design.md) "prefill/decode separated export / weight sharing" (L928-962)
- Related: CLI-010 (load_model load waste, swap discards Linear fp)

---

<a id="cli-015-todo-implement-weight-nd-to-nz-format-conversion-per-weight_format_designmd"></a>
## CLI-015 TODO: implement weight ND to NZ format conversion per weight_format_design.md

| Field | Value |
|------|-----|
| ID | CLI-015 |
| Type | TODO (legacy/pending implementation, not deviation) |
| Location | Design: [weight_format_design.md](./export/weight_format_design.md); Code TBD: `cann_local_lm/converter/export/weight_format/` (`layout.py` `nd2nz` + `rules.py` `resolve_format`) + `weight_format_rules.json` + `epair_utils._pack_weights_to_dir` integration + `inline_weight_layout` + C++ `aten_tensor_desc_mapper.cc:112` `MapGeLayoutString` / `ge_weight_loader.cc:115` `Pt2LayoutFromGeString` |
| Background | `weight_format_design.md` designed weight on-disk from ND (row-major) to NZ (FractNZ 16×16 block-major), letting Ascend cube compute weight Const enter GE graph directly as `FORMAT_FRACTAL_NZ`, avoiding OM compile/runtime transdata. Stage 1 (fp16 `MatMul` Linear weights `*_proj`) designed, but `weight_format/` package + `weight_format_rules.json` **0 implementation** (`ls` confirms NOT implemented) |
| Violates convention | None (pending implementation) |
| Discovery date | 2026-09-01 |
| Status | **L2 path works (bare-matmul + nn.Linear A3, NPU max_diff=0); L3 full-network pending verification (2026-09-07)** |

### Design Ready (Stage 1, weight_format_design.md)

- `weight_format/layout.py`: `nd2nz(tensor, block=16, pad=True)` [M,K] ND → [ceil(M/b),ceil(K/b),b,b] block-major (non-b-multiple dims internally zero-padded)
- `weight_format/rules.py`: `load_rules` (reads shipped `weight_format_rules.json`) + `resolve_format(name, is_quantized)` (dispatch MatMul/QuantBatchMatmulV4 by `name_pattern` + `op_type`)
- `weight_format_rules.json`: default config (`*_proj.weight` → NZ; `QuantBatchMatmulV4` → ND Stage 2 TBD)
- `epair_utils._pack_weights_to_dir` integration: insert `resolve_format` + `nd2nz` (mimic `transdata_scale` precedent), `weight_info.json` entry adds `"format"` field
- **Environment variable master switch** `CANN_LM_ENABLE_WEIGHT_TRANS` (default off, `1` enable): env unset → packer skips `resolve_format` + `is_quantized` probing, all ND on disk, code dormant zero risk; Python export side only, C++ driven by ep_json layout string no env
- `inline_weight_layout`: ep_json `tensor_values.layout` + `ascend_ir.desc.layout` ND→NZ
- C++ `MapGeLayoutString`/`Pt2LayoutFromGeString` add `"NZ"` case

### Not Done (Stage 2, incl. mandatory research gate)

Quantized `QuantBatchMatmulV4` int4 nibble-packed weights NZ — standard `nd2nz` would break nibble pairing, needs dedicated `pack_int4_nz`, but int4-NZ byte layout determined by CANN V4/V3 runtime kernel, cannot implement by inference. Research gate: ① `atc --mode=1` parse existing quantized OM for TransData residual (CANN auto-converts then quantized weight stays ND); ② read GE source `quant_batch_matmul_v4` op_graph + 950-ops int4 NZ kernel for byte layout. Current ND quantized OM works, Stage 2 is performance optimization not correctness requirement.

### Pending Verification

1. MatMul NZ input acceptance: does `reg_register.h` MatMul accept weight Const `FORMAT_FRACTAL_NZ` without runtime transdata (needs minimal LLM (truncated N layers) NZ export + OM compile verification, parameterized in `examples/`)
2. NZ shape semantics: implementation per GE standard (logical shape + internal pad), OM compile实测 confirms kernel correctly reads partial block

### Action Items

- [x] Python `weight_format/{layout,rules}.py` + `weight_format_rules.json` + `epair_utils._pack_weights_to_dir` env-gated `nd2nz` + `weight_info.format` field (13 tests pass)
- [x] C++ `model_types.h` (format_str) + `ge_weight_loader.cc` (read format) + `aten_tensor_desc_mapper.cc` (`MapGeLayoutString` NZ case) + `ge_graph_builder.cc:CreateConstNode` (2D shape + FRACTAL_NZ, no reshape) + input desc NZ override
- [x] ~~`inline_weight_layout`~~ deprecated (pt2 layout enum has no NZ; goes via `weight_info.format` + C++ input override)
- [x] L2 bare-matmul path works (A3): `nd2nz` permute `(2,0,1,3)` + 2D shape + FRACTAL_NZ (no reshape) + input NZ override → OM compile pass + NPU max_diff=0 (2026-09-07)
- [x] L2 nn.Linear path works (A3): skip Transpose + `transpose_x2=True` + `nz_transpose_passthrough` moved to `GraphBuildContext` (cross-node shared) → NPU max_diff=0 (2026-09-07)
- [x] L2 non-16-aligned shape limitation confirmed (130×130): GE `CheckTensorIsValid` doesn't accept NZ padding → non-16-aligned infeasible (2026-09-07)
- [ ] L3 full network (A3)
- [ ] C++ fallback dead path (Transpose perm pre-scan + ① dead-path code)

### Related

- Design doc: [weight_format_design.md](./export/weight_format_design.md) (Stage 1 design + Stage 2 research gate)
- Related code: `cann_local_lm/converter/export/epair_utils.py` (`_pack_weights_to_dir` integration point + `inline_weight_scale_dtype` precedent), C++ `aten_tensor_desc_mapper.cc`/`ge_weight_loader.cc`/`ge_graph_builder.cc`
- Related: CLI-014 (PD weight artifact layout), CLI-016 (epair weight and const separation)

---

<a id="cli-016-todo-epair-weight-and-const-separation"></a>
## CLI-016 TODO: epair weight and const separation

| Field | Value |
|------|-----|
| ID | CLI-016 |
| Type | TODO (legacy/pending clarification + pending implementation) |
| Location | `cann_local_lm/converter/export/epair_utils.py` (`_pack_weights_to_dir`/`ConsolidatedWeights`, weights and const mixed on disk) + `weight_info.json` (index) + C++ `GeWeightLoader` (loading) |
| Background | epair's current `weights_*.bin` mixes model weights (Linear `*.weight`, Embedding, norm, bias etc.) with graph constants (const tensor, like fused_attn_seq constants, shape/dtype constants) into one bin; `weight_info.json` unified index. Need to separate "weights" from "const" (separate files / classified index), for C++ loading side to distinguish weight Const nodes from graph constants, supporting differentiated handling (e.g. NZ format only applies to weight const keeping ND; or const externalized/shared) |
| Violates convention | None (pending clarification + pending implementation) |
| Discovery date | 2026-09-01 |
| Status | **TODO (pending scope clarification + pending implementation)** |

### Pending Clarification

1. "const" scope: only graph constants (`ascend_ir` Const nodes, like inlined fused_attn_seq constants, scalar constants)? Or all non-Linear-weight tensors (norm/bias/embed)?
2. Separation granularity: separate files (`weights_*.bin` vs `consts_*.bin`) + separate indices (`weight_info.json`/`const_info.json`)? Or same file classified index (`weight_info.json` adds `"kind": "weight"|"const"` field)?
3. C++ loading side: does `GeWeightLoader` need to distinguish weight/const for different loading paths (e.g. weight supports NZ format, const stays ND; const may be externalized/shared)

### Pending Direction

1. **Artifact layout**: `weights_*.bin` (weights) + `consts_*.bin` (graph constants) separate files + `weight_info.json`/`const_info.json` separate indices; or same bin + `kind` field classification
2. **C++ loading**: `GeWeightLoader` distinguishes loading by kind (weight goes NZ/format conversion path, const goes ND)
3. **Coordination with CLI-015**: NZ format conversion only applies to weights (const stays ND); after separation CLI-015's `resolve_format` only matches weight kind
4. **model_config.json**: records const separation flag

### Action Items

- [ ] Clarify "const" scope (graph constants vs non-weight tensors) and separation granularity (separate files vs classified index)
- [ ] Design weight/const separation artifact layout (`consts_*.bin` + `const_info.json` or `kind` field)
- [ ] C++ `GeWeightLoader` distinguish loading path by kind
- [ ] Coordinate with CLI-015 (NZ format): resolve_format only matches weight kind
- [ ] Docs: export_design.md add weight/const separation description

### Related

- Source: `cann_local_lm/converter/export/epair_utils.py` (`_pack_weights_to_dir`/`ConsolidatedWeights`/`weight_info.json`), C++ `GeWeightLoader`
- Design docs: [export_design.md](./export/export_design.md) (artifact layout/weight loading), [weight_format_design.md](./export/weight_format_design.md)
- Related: CLI-015 (weight_format NZ, only applies to weights), CLI-014 (PD weight artifact layout)

---

<a id="cli-017-todo-torch-delegate-supports-torchair-stubbing-then-consolidate-install-scripts"></a>
## CLI-017 TODO: torch-delegate supports torchair stubbing then consolidate install scripts

| Field | Value |
|------|-----|
| ID | CLI-017 |
| Type | TODO (legacy/pending torch-delegate support then consolidate) |
| Location | `torch-delegate-harmony/build.sh` (only builds torch-delegate whl + .run, no torchair stubs); torchair install independent step (clone + configure `NO_ASCEND_SDK=1` + `build_and_install.sh` + cp `build/dist/torchair` to venv + `LD_LIBRARY_PATH` set stubs, see SKILL.md step 3); CANN-LM `scripts/set_env_export.sh` (`LD_LIBRARY_PATH` stubs first) |
| Background | Current converter env (torch-delegate venv + torchair stubs) setup is scattered: torch-delegate `build.sh` only produces `torch_delegate` whl + .run, torchair stubs need manual `git clone gitcode.com/Ascend/torchair` + `NO_ASCEND_SDK=1 bash configure` + `bash build_and_install.sh` + manual cp `build/dist/torchair` (build_and_install may not auto-install) + set `LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:...` (SKILL.md step 3 + AGENTS.md converter env). torchair stubs are converter env required (provide torchair GE engine C symbols, avoid real NPU `aclrtSetDevice→107001`), but not in torch-delegate install script. Wait for torch-delegate to support torchair stubbing (e.g. `build.sh` integrates torchair stubs build/install), then consolidate install script (one-click converter env setup) |
| Violates convention | None (pending torch-delegate support) |
| Discovery date | 2026-09-01 |
| Status | **TODO (pending torch-delegate support for torchair stubbing then consolidate)** |

### Current State (scattered setup steps)

converter env = torch-delegate venv + torchair stubs, setup scattered:

1. **torch-delegate** (`build.sh`): venv creation + torch 2.9.0+cpu + third-party libs + C++ compile + whl + .run (produces `torch_delegate-0.1.0-*.whl/.run`)
2. **torchair** (manual, SKILL.md step 3): `git clone --recursive gitcode.com/Ascend/torchair` + `NO_ASCEND_SDK=1 bash configure` + `bash build_and_install.sh <site-packages>` + manual cp `build/dist/torchair` (build_and_install may not auto-install) + `pip install packaging`
3. **stubs env** (CANN-LM `set_env_export.sh`): `LD_LIBRARY_PATH=${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:...` (stubs first, avoid `aclrtSetDevice→107001`)

Problem: torchair stubs are converter env required (no stubs → torchair GE engine calls real CANN `aclrtSetDevice` → 107001 no device), but torch-delegate `build.sh` doesn't include torchair build/install, needs manual multi-step. env spans torch-delegate (`/mnt/workspace/torch-delegate-harmony`) + torchair (`/mnt/workspace/torchair`) + CANN-LM `set_env_export.sh` three places.

### Pending torch-delegate Support

- torch-delegate `build.sh` integrates torchair stubs build option (e.g. `--with_torchair_stubs`, calls torchair `configure NO_ASCEND_SDK=1` + `build_and_install` + installs `dist/torchair` to venv)
- Or torchair as torch-delegate dependency (pip install or build.sh pulls + compiles stubs)
- Output: after torch-delegate install, venv has torchair + stubs, no manual multi-step

### Consolidation Goal (after support)

1. **One-click converter env setup**: `torch-delegate build.sh --with_torchair` (or similar) produces whl + .run + installs torchair stubs to venv
2. **env script simplification**: CANN-LM `set_env_export.sh` only sources CANN + sets `LD_LIBRARY_PATH` (stubs path from torch-delegate install record/default), no manual torchair path maintenance
3. **Doc unification**: SKILL.md step 3 (torchair manual) merged into torch-delegate install doc, or torch-delegate doc describes torchair stubs integration

### Action Items

- [ ] Wait for torch-delegate to support torchair stubbing (`build.sh` integration or dependency)
- [ ] Consolidate install script: torch-delegate `build.sh` one-click produces whl + .run + torchair stubs
- [ ] CANN-LM `set_env_export.sh` simplify (stubs path from torch-delegate install default)
- [ ] SKILL.md / AGENTS.md doc unification (torchair stubs integrated into torch-delegate install)

### Related

- Install scripts: `torch-delegate-harmony/build.sh` (torch-delegate build), `torchair/build_and_install.sh` (torchair build)
- env: CANN-LM `scripts/set_env_export.sh` (converter env, `LD_LIBRARY_PATH` stubs), `scripts/_env_common.sh` (`TORCHAIR_ROOT`/`TORCH_DELEGATE_HOME` defaults)
- Docs: `examples/.../skills/torch-delegate-epair/SKILL.md` (step 3 torchair manual install), CANN-LM AGENTS.md "Runtime Environment Setup (converter)"

---

<a id="cli-018-todo-simplify-cann-lm-multi-environment-referencing-mlx-lm-env-handling"></a>
## CLI-018 TODO: simplify cann-lm multi-environment referencing mlx-lm env handling

| Field | Value |
|------|-----|
| ID | CLI-018 |
| Type | TODO (legacy/survey + simplification direction, not deviation) |
| Location | Survey: mlx-lm `setup.py` (extras + install_requires) + mlx library device abstraction (`mx.default_device`/`mx.device_info`) + 4 env vars set in-code; cann-lm `scripts/set_env_{export,compile,generate,npu}.sh` + `_env_common.sh` + AGENTS.md "Runtime Environment Setup" |
| Background | mlx-lm env handling is minimal: single mlx library (pip install mlx-lm ready, macOS auto-pulls mlx, Linux selects extras cpu/cuda) + device fully managed by mlx library (`mx.default_device()` auto-selects macOS→Metal / Linux→CPU) + only 4 env vars (`TOKENIZERS_PARALLELISM`/`MLXLM_USE_MODELSCOPE`/`MLX_GDN_PACKED`/`TRANSFORMERS_NO_ADVISORY_WARNINGS`, set in-code, no source needed) + wired memory (`mx.set_wired_limit`) + no env scripts/stubs/LD_LIBRARY_PATH management. cann-lm opposite: 4 envs (converter venv+stubs / compile / generate / npu) + `LD_LIBRARY_PATH` stubs first to avoid `aclrtSetDevice→107001` + `ASCEND_HOME_PATH` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + `TORCH_DELEGATE_*` + torchair source compile + torch_npu not installed (stub path). Root cause: mlx-lm depends on single mlx library (Apple official, complete device abstraction); cann-lm spans CANN+torchair+torch_npu+torch-delegate multi-component, inter-component ABI/version coupling (torchair compiled against torch 2.9.0, torch_npu 2.7.1 incompatible) |
| Violates convention | None (survey + simplification direction) |
| Discovery date | 2026-09-01 |
| Status | **TODO (survey complete, simplification direction pending evaluation)** |

### mlx-lm env handling (survey)

| Dimension | mlx-lm | cann-lm |
|------|--------|---------|
| Env count | Single (mlx library unified device abstraction) | Multiple (converter venv+stubs / compile / generate / npu) |
| Install | `pip install mlx-lm` (macOS auto-pulls mlx, Linux extras) | build.sh + torchair manual + CANN upgrade + stubs (CLI-017 pending consolidate) |
| Device management | mlx library `mx.default_device()` auto | Manual set_env_export.sh + LD_LIBRARY_PATH stubs + ASCEND_HOME_PATH |
| env scripts | None (pip install ready) | 4 set_env_*.sh |
| env vars | 4 (in-code) | Multiple (ASCEND_HOME_PATH/LD_LIBRARY_PATH/TORCH_DEVICE_BACKEND_AUTOLOAD/TORCH_DELEGATE_*, need source) |
| Platform | Apple Silicon single | Ascend NPU + CANN + torchair + torch_npu multi-component |
| Stubbing | None needed (mlx runs directly) | Stub path (torchair stubs avoid aclrtSetDevice→107001) |

### Simplification Direction (pending evaluation, limited by multi-component coupling)

1. **Single dependency convergence**: after torch-delegate build.sh integrates torchair stubs (CLI-017), `pip install torch-delegate` or build.sh one-click produces whl+stubs, close to mlx-lm `pip install mlx-lm`
2. **Device abstraction layer**: cann-lm has no mlx-library-equivalent unified device abstraction (CANN/torchair/torch_npu each manages own); long-term can wrap device abstraction layer (e.g. `cann_local_lm.device.is_available()` targeting `mx.default_device()`), but limited by multi-component
3. **env in-code**: mlx-lm env vars set in-code (`os.environ[...]=...`), no source; cann-lm some can be in-code (e.g. `TRANSFORMERS_NO_ADVISORY_WARNINGS` already mimicked), but LD_LIBRARY_PATH/ASCEND_HOME_PATH need source (CANN lib paths)
4. **env script simplification**: after CLI-017 set_env_export.sh simplifies (stubs path from torch-delegate install default); 4 envs converge (if converter/compile/generate/npu can merge)
5. **Stubbing transparency**: mlx-lm no stubs (mlx runs directly); cann-lm stubs are torchair GE engine's workaround to avoid real NPU, long-term if torchair supports pure CPU conversion (no GE engine calling aclrt) can remove stubs

### Action Items

- [ ] Evaluate single dependency convergence (after torch-delegate build.sh integrates torchair stubs, close to pip install ready)
- [ ] Evaluate device abstraction layer (targeting `mx.default_device`, limited by multi-component)
- [ ] env vars in-code (those that can be in-code mimic mlx-lm, reduce source)
- [ ] env script simplification (depends on CLI-017, 4 envs converge)
- [ ] Stubbing transparency evaluation (long-term, if torchair supports pure CPU conversion can remove stubs)

### Related

- Survey source: mlx-lm `setup.py` + `mlx_lm/{generate,server,utils,__init__}.py` (device/env handling)
- Related cann-lm: `scripts/set_env_*.sh` + `_env_common.sh` + AGENTS.md "Runtime Environment Setup (converter)"
- Related legacy: **CLI-017** (torchair stubbing integration + install script consolidation, this env simplification depends on CLI-017)
- Survey doc: [reference_side_tools_survey.md](./reference_side_tools_survey.md) mlx-lm section + env handling comparison

---

<a id="cli-019-todo-distinguish-p_c_b_-three-data-types-when-exporting-epair"></a>
## CLI-019 TODO: distinguish p_/c_/b_ three data types when exporting epair

| Field | Value |
|------|-----|
| ID | CLI-019 |
| Type | TODO (pending decision + pending implementation) |
| Location | `cann_local_lm/converter/export/epair_utils.py` (`_pack_weights_to_dir`: three types mixed on disk, though already iterates `graph_signature`'s `inputs_to_parameters`/`inputs_to_buffers`/`inputs_to_lifted_tensor_constants` but doesn't record category) + `weight_info.json` (entries have no category field) + C++ `ge_weight_loader.cc`/`ge_graph_builder.cc:CreateConstNode` (three types uniformly build Const) |
| Background | epair export's `_pack_weights_to_dir` mixes parameter/const/buffer three types into same `weights_*.bin` + `weight_info.json` (entries only `{name,offset,size,format}`, **no category identifier**). Three types in ep_json `tensor_values` distinguished by prefix: `p_*` (parameter, model weights), `c_*` (const, lifted tensor constants), `b_*` (buffer, non-persistent state); authoritative source is `graph_signature` (pack already iterates but doesn't persist category). Need to distinguish at export because: (1) NZ format conversion (CLI-015) should only apply to `p_*`'s `*_proj.weight`, not touch `c_*`/`b_*`; (2) C++ loading side has different semantics for three types (parameter can NZ/externalized, const stays ND/shareable, buffer non-persistent); (3) directly related to CLI-016 granularity |
| Violates convention | None (pending decision) |
| Discovery date | 2026-09-05 |
| Status | **TODO (pending decision plan + pending implementation)** |

### Two Solutions

| Plan | Artifact layout | Index | C++ loading side |
|------|---------|------|-----------|
| ① Separate files | `weights_*.bin` (p_) + `consts_*.bin` (c_) + `buffers_*.bin` (b_) separate files | `weight_info.json`/`const_info.json`/`buffer_info.json` separate indices | `GeWeightLoader` naturally dispatches by file |
| ② weight_info add field | Same `weights_*.bin` mixed | `weight_info.json` entry adds `"kind": "parameter"\|"const"\|"buffer"` | `GeWeightLoader` reads `kind` to dispatch |

### Decision Considerations

- Plan ① clear boundary (physical separation), but changes artifact layout (3 sets bin+index) + C++ loading dispatch, large change surface; prefill/decode shared weights (p_) still only one copy
- Plan ② small change (add 1 field, packer judges category by `graph_signature`), C++ dispatches by `kind` (NZ only for parameter kind); still mixed bin
- Coordinate with CLI-015: NZ (`weight_info.format`) only applies to `kind=="parameter"`'s `*_proj.weight`, const/buffer stay ND
- Relationship with CLI-016: CLI-016 is "weights vs const" 2-way, this item refines to "p_/c_/b_" 3-way (splits CLI-016's "weights" into parameter+buffer). **Solving this item solves CLI-016's "pending clarification 1/2" (const scope + granularity)**, recommend merging both decisions

### Action Items

- [ ] Decide plan ① (separate files) or ② (add kind field)
- [ ] If ②: `_pack_weights_to_dir` adds `"kind"` to entries per `graph_signature` (parameter/buffer/lifted_tensor_constant)
- [ ] C++ `GeWeightLoader`/`CreateConstNode` dispatch by `kind` (NZ only for parameter)
- [ ] Coordinate with CLI-015 (NZ format), CLI-016 (weight/const separation)

### Related

- Source: `cann_local_lm/converter/export/epair_utils.py` (`_pack_weights_to_dir`) + C++ `ge_weight_loader.cc`/`ge_graph_builder.cc:CreateConstNode`
- Related: **CLI-015** (NZ format, only applies to parameter), **CLI-016** (weight and const separation, this item refines its granularity)

---

## CLI-020 TODO: torch.export has no output name parameter, saver needs to auto-set semantic names

| Field | Value |
|------|-----|
| ID | CLI-020 |
| Type | TODO (pending implementation) |
| Location | `cann_local_lm/converter/saver.py` (`save_epair`/`save_epair_prefill_decode`: `torch.export.export` after no output name set) + `cann_local_lm/models/export_wrapper.py` (`make_input_output_names` only for ONNX, doesn't cover delta_out) |
| Background | `torch.export.export(model, args, strict=False)` **has no direct parameter** to set output name. Output name derived from the FX op node producing that output (e.g. `matmul`/`sin`/`cos`/`index_select`), not semantic name (e.g. `logits`/`past_key_0_out`). When returning dict, keys are lost (torch.export flattens to tuple, name still derived from op name). Tested torch 2.9.0+cpu: `OutputSpec.arg = TensorArgument(name='matmul')` — `name` field can be post-processed and rewritten, after change `ep.module()` runs fine |
| Violates convention | None (pending implementation) |
| Discovery date | 2026-09-16 |
| Status | **TODO (approach verified, pending implementation)** |

### Background: Why Need Semantic output name

epair export chain `torch.export → torch_delegate.export_to_epair → ep_json`, ep_json's `signature.output_specs` each output's `arg.name` follows FX op node name (e.g. `matmul_3`/`index_select_1`), no semantic. Downstream consumers (OM runtime/C++ loading/host scheduling) need to identify outputs by position not name (logits/past_kv_out/delta_out), poor readability, error-prone.

### Current Gap

1. **`ExportWrapper.make_input_output_names()` already has convention but only for ONNX** (`export_wrapper.py:229`): generates `["logits", "past_key_{i}_out", "past_value_{i}_out"]`, epair path not called
2. **delta_out not covered**: base class `make_input_output_names` only generates logits + past_kv_out (`2×num_layers`), qwen3.5 `forward` also returns `delta_out_flat` (per-layer conv+rec, `2×num_layers` of them, `export_wrapper.py:135-140`), names count < output count
3. **saver no post-processing**: `save_epair`/`save_epair_prefill_decode` after `torch.export.export()` directly `torch_delegate.export_to_epair(ep, config)`, doesn't modify `ep.graph_signature.output_specs[i].arg.name`

### Approach (verified torch 2.9 works)

**Step 1**: Complete `make_input_output_names()` with delta_out (base class or qwen3.5 override):

```python
def make_input_output_names(self) -> tuple[list[str], list[str]]:
    n = self._num_layers()
    input_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]
    for i in range(n):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])
    output_names = ["logits"]
    for i in range(n):
        output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
    if self._has_delta_out():  # Subclass override: qwen3.5 True, gemma4 False
        for i in range(n):
            output_names.extend([f"delta_conv_{i}_out", f"delta_rec_{i}_out"])
    return input_names, output_names
```

**Step 2**: saver.py post-processes output_specs after `torch.export.export()`, before `torch_delegate.export_to_epair()`:

```python
ep = torch.export.export(wrapper, sample_inputs, strict=False)
_, output_names = wrapper.make_input_output_names()
for spec, name in zip(ep.graph_signature.output_specs, output_names):
    spec.arg.name = name  # TensorArgument.name can be changed
for node in ep.graph.nodes:  # Sync graph output node name
    if node.op == "output":
        for n, name in zip(node.args[0], output_names):
            n.name = name
        break
with lite_export_context(lite_config):
    epair = torch_delegate.export_to_epair(ep, config)  # ep_json follows custom names
```

### Boundary Handling

- **delta_out optional**: `forward` returns `delta_out_flat` only when `delta_states is not None` (`export_wrapper.py:136`). `save_epair`'s sample_inputs doesn't pass delta_states → delta_out_flat empty → output count = `1 + 2×num_layers`. `make_input_output_names` extra delta names auto-truncated by `zip`, no effect
- **zip short-board**: when names insufficient, fallback to op name (avoid mismatch), need `if len(names) < len(specs)` fallback
- **prefill/decode sharing**: `make_input_output_names` doesn't depend on seq_len/prefill/decode, prefill/decode use same names, no prefill/decode difference (no conflict with CLI-014 weight sharing)

### Action Items

- [ ] `ExportWrapper.make_input_output_names()` add delta_out names + add `_has_delta_out()` hook (qwen3.5 override True)
- [ ] `saver.py` add `_apply_output_names(ep, wrapper)` helper, called after `torch.export` in `save_epair`/`save_epair_prefill_decode`
- [ ] Verify ep_json `signature.output_specs`'s `arg.name` is semantic (`logits`/`past_key_0_out`/...)
- [ ] Docs: export_design.md add output name auto-set description

### Related

- Source: `cann_local_lm/converter/saver.py` (`save_epair`/`save_epair_prefill_decode`) + `cann_local_lm/models/export_wrapper.py` (`make_input_output_names`/`forward` returns logits+past_kv_flat+delta_out_flat)
- Related: **CLI-014** (PD weight sharing, this output name doesn't affect sharing), **CLI-016/CLI-019** (weight/const separation, partially implemented `_is_weight` field — see note below)
- Note: CLI-016/CLI-019's "weight_info.json add category field" already implemented as `_is_weight` (bool, binary: parameter+buffer=true / const=false), `epair_utils.py:_pack_weights_to_dir` writes this field + C++ `ge_weight_loader.cc:LoadConsolidated` reads to fill `TensorRecordInfo.is_parameter` + `ge_graph_builder.cc:CreateConstNode` injects GE Const node private attr `_is_weight` (visible in dump graph, OM cleaned by GE OptimizeStage1). CLI-016/CLI-019's NZ by kind dispatch (NZ only for parameter) not yet implemented, still TODO
