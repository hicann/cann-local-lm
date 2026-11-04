# `executorch_npu` Python package

English | [简体中文](README.md)

This package serves as the CANN backend plugin for ExecuTorch and keeps three entry points at the top level.
The public Python API is exported uniformly by `__init__.py`; internal functions are categorized by compilation, graph rewriting,
artifact construction, and helper capabilities.

## Directory Layout

```text
executorch_npu/
├── __init__.py                  # Public Python API
├── backend.py                   # Backend integration, multi-method orchestration, and staging lifecycle
├── partitioner.py               # Partitioner integration
├── edge_compile_config.py       # Edge export configuration integration
├── compiler/
│   ├── compile_spec.py          # Compile options and CompileSpec encoding/decoding
│   ├── pipeline.py              # Single-partition compilation, returns OM / weight paths
│   ├── delegate_adapter.py      # ExportedProgram → EPAIR → GE, with compatibility adaptation
│   └── offline.py               # GE offline compilation lifecycle
├── passes/                     # Each pass in its own file, preserving graph rewrite order
│   ├── edge_ops_pass.py
│   ├── static_split_pass.py
│   └── edge_copy_pass.py
├── artifacts/
│   ├── manifest.py              # Pydantic protocol models and ASCD encoding
│   ├── builder.py               # named data, full Manifest, and artifact set checks
│   ├── io_contract.py           # Graph parameter layout and manifest I/O construction
│   └── state.py                 # State declarations, graph association, and single-graph/cross-method state rules
└── shared/
    ├── constants.py
    ├── symbolic_shape.py        # Symbolic dimension range parsing shared by compilation and manifest
    ├── diagnostics.py           # Edge, compilation-phase, and state-mapping diagnostics
    └── sdk.py                   # Resource location for the installed C++ SDK
```

## Main Flow

```text
ExecuTorch → AscendPartitioner → AscendBackend
                                    │
                                    ├─ compiler.compile_partition()
                                    │    passes → EPAIR → GE → OM + weight files
                                    │    artifacts.builder registers artifacts and returns the Manifest
                                    │
                                    ├─ Multi-method: check_manifest_set()
                                    │    delegate identity check + shared state check
                                    │
                                    └─ Manifest.to_bytes() → PreprocessResult
                                         ExecuTorch writes PTE / PTD
```

`backend.py` owns the staging lifecycle: the temporary directory is released only after compilation artifacts have been read by `builder.py` and registered as
named data. `compiler/` does not register named data, nor does it construct the Manifest.
The EPAIR loader is closed before offline compilation; the fallback-disabled scope covers GE graph extraction and offline compilation.

`artifacts/io_contract.py` computes the final parameter indices uniformly, and state mapping consumes the same layout.
`artifacts/state.py` distinguishes export-time state declarations with provenance information from the final state bindings after delegation.
Reading the `ExportedProgram` when constructing the manifest is the responsibility of artifact mapping and does not require an additional generic IR.

Pydantic types and field constraints are centralized in `manifest.py`; graph association, state policies, and artifact set rules
are checked by their respective construction functions. Encoding happens only after cross-method checks complete, preserving protocol v1 and existing artifact semantics.

## Public API

Importing from the package root is recommended; internal migrations do not change these import styles:

```python
from executorch_npu import (
    AscendBackend,
    AscendCompileOptions,
    AscendPartitioner,
    Manifest,
    aten_edge_compile_config,
    build_state_io_contract,
    sdk_root,
)
```

Direct references to the former `executorch_npu.compile_spec`, `executorch_npu.manifest`,
`executorch_npu.sdk`, and old `compiler` internal paths should be changed to the package-root public API
or the new module paths listed above.
