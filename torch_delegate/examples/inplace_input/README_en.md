# Inplace Input Verification Case

English | [简体中文](README.md)

Verifies that a model containing inplace operations behaves consistently under two export paths: `torch.export.save/load` and the `torch_delegate` epair.

---

## What Is Verified

| Item | Description |
|-------|------|
| Inplace input modification | Input tensors modified in place behave the same as in Eager mode |
| Output accuracy | Output values match Eager (error < 1e-5) |
| FX Graph capture | Inplace operators are captured correctly |

---

## Directory Layout

```text
inplace_input/
├── models.py                 # Model definitions (3 inplace models)
├── utils.py                  # Utility functions
├── 1_export_fx_graph.py      # Phase 1: export the FX graph with torch.export
├── 2_verify_export.py        # Phase 2: verify torch.export load (cross-process)
├── 3_aoti_compile.py         # Phase 3: AOTI compile (skipped by default, needs gcc11+)
├── 4_verify_aoti/            # Phase 4: C++ AOTI verification (skipped by default, needs gcc11+)
│   ├── main.cpp              # C++ AOTI loading and execution
│   ├── verify_aoti.py        # Python-side comparison script (invokes the C++ binary)
│   └── CMakeLists.txt        # Build configuration
├── 5_compile_fx_graph.py     # Phase 5: torch.compile FX graph comparison
├── 7_npu/                    # Phase 7: NPU epair export and verification (C++ + Python)
│   ├── export_epair.py       # epair export script
│   ├── main.cpp              # C++ GE loading and execution
│   └── README.md             # Detailed NPU verification guide
├── run.sh                    # One-shot run script
└── generated/                # Output directory
```

---

## Requirements

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair** (needed for the NPU phase)
3. **torch-delegate** built
4. **CANN 9.2.0+** (needed for the NPU phase)

---

## Quick Start

```bash
# Run all phases
bash run.sh
```

`run.sh` automatically:
1. exports and verifies the FX graph with torch.export (Phases 1-2)
2. compares the torch.compile FX graph (Phase 5)
3. exports the NPU epair and verifies it with C++ GE (Phase 7, requires an NPU environment)

> [!CAUTION] Note
> Phases 3-4 (AOTI compilation and verification) depend on gcc11+ and a newer GLIBCXX, so they are skipped by default. To verify manually on a gcc11+ environment, set `SKIP_AOTI` to `false` in `run.sh`.

---

## Test Models

| Model | Inplace operations |
|------|-------------|
| SimpleInplaceModel | `x.add_(y); x.mul_(scale)` |
| ChainInplaceModel | `x.add_(1); x.mul_(2); x.div_(4)` |
| ParamInplaceModel | `x.add_(bias); x.mul_(scale)` |

---

## Phases

| Phase | Device | Description | Default state |
|-------|------|------|---------|
| 1 | CPU | Export and save the FX graph with torch.export | Runs |
| 2 | CPU | Verify torch.export load across processes | Runs |
| 3 | CPU | AOTI compilation | Skipped (needs gcc11+) |
| 4 | CPU | Verify AOTI load across processes | Skipped (needs gcc11+) |
| 5 | CPU | torch.compile FX graph comparison | Runs |
| 7 | NPU | epair export and C++ GE verification (see `7_npu/`) | Requires an NPU environment |

---

## Artifact Naming

Format: `{method}_{content}_{model}.{ext}`

| Artifact | Description |
|------|------|
| `export_model_simple.pt2` | Model exported by torch.export |
| `export_fx_graph_simple.txt` | torch.export FX graph structure |
| `dynamo_fx_graph_simple.txt` | FX graph captured by Dynamo |
| `epair_model_simple.epair` | torch_delegate epair file |
| `epair_manifest_simple.json` | epair manifest information |

---

## FX Graph Comparison

| Path | Inplace representation |
|------|-------------|
| torch.export | `aten.add_.Tensor`, `aten.mul_.Tensor` |
| Dynamo | `call_method: add_`, `mul_` |

**Key finding**: torch.export and epair keep the original inplace operators, and the results match Eager mode.

---

## Pass Criteria

- **Inplace modification**: `max|input_modified - eager_input_modified| < 1e-5`
- **Output accuracy**: `max|output - eager_output| < 1e-5`
