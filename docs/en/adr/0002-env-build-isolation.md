# ADR-0002: Environment and build isolation — four envs + unified top-level build tree

English | [简体中文](../../zh/adr/0002-env-build-isolation.md)

> Status: Accepted
> Date: 2026-08-27
> Related design: `AGENTS.md` "Runtime Environment Setup" + [export_design.md](../cann_lm_converter/export/export_design.md)

## Background (Context)

The repository has three types of development/build activities whose boundaries were originally blurred, scripts were hard-coded, and env logic was duplicated in three places:

1. **converter tool**: HF→epair export + quantization, requires a Linux CPU venv + torchair stubs (does not install torch_npu).
2. **Model conversion tool**: epair→compiled model (OM), requires real CANN + C++ compilation.
3. **generate**: Cross-compilation for HarmonyOS PCs (this repo's `generate/`).

Specific problems found during exploration (P1-P6):

- **P1**: `DELEGATE_COMMON_FLAGS` is only referenced but never defined anywhere in the repo (`TorchDelegateTestStubs.cmake:30-31` references it, but there is no `set()`), so it currently equals an empty string, violating AGENTS.md's "Common Flag Principle".
- **P2**: AGENTS.md states "ASAN mandatory", but none of the 4 CMakeLists implement the `ENABLE_ASAN` option; `build_and_test.sh` also does not pass `-DENABLE_ASAN=ON`. It exists in name only.
- **P3**: CANN version drift — `build_csrc.sh`/`build_and_test.sh`/`tests/.../CMakeLists.txt` default to `cann-9.1.0`, but this machine only has `cann-9.2.0` (`cann`→`cann-9.2.0` symlink), so the script defaults are directly unusable.
- **P4**: env logic duplicated in three places — `set_env.sh` + `run.sh` + `run_pd6_cases.sh` (`setup_converter_env`/`setup_dopt_env`) each copy a hard-coded path.
- **P5**: All 4 CMakeLists hard-code `-Wall -Wextra` at both compile+link stages (the linker does not recognize `-Wall` at the link stage, and it violates the Common Flag Principle).
- **P6**: cmake module path ambiguity — csrc `CMakeLists.txt` points to this repo's `cmake/modules`, but `build_csrc.sh` also uses `-DCMAKE_MODULE_PATH` to point to the sibling `torch-delegate-harmony/cmake`, so two same-named `Find*.cmake` files are both in the search path.

## Decision (Decision)

Establish **four mutually non-overlapping envs** + **a unified top-level CMake build tree**, isolated by three layers of "venv (runtime) + build/ (artifacts) + toolchain (CMake targets)":

- env: `converter`(①) / `compile`(②, renamed from `om` + borrows ① venv torch cmake) / `generate`(③, reserved) / `npu`(④, verify-runtime). `om` serves as a compatibility alias for `compile`.
- C++ build: create a new **top-level `CMakeLists.txt`** (`add_subdirectory` chaining csrc+examples+tests), single entry point `scripts/cann_lm_compile/build_cpp.sh`; **delete** `csrc/build_csrc.sh` (the original standalone csrc build script) and `tests/build_and_test.sh` (the original standalone cmake test runner) — the former's 6 call sites all fall back to `build/libtorch_delegate_backend.so` + `bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend`, the latter is replaced by `ctest --test-dir build` (the tests CMakeLists already provides `ENVIRONMENT LD_LIBRARY_PATH` for each test, so it can run bare). csrc CMakeLists sets `LIBRARY_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}` so the .so lands at the build root instead of nested by source tree (keeping fallback paths stable). ASAN flags use `-static-libasan`, so ASAN binaries can also run bare `ctest` (no LD_PRELOAD needed).
- Common Flag: create `cmake/modules/DelegateCommonFlags.cmake`, which **defines** `DELEGATE_COMMON_FLAGS=-Wall -Wextra -Werror` + the `ENABLE_ASAN` option + the `delegate_apply_common_flags()` helper function; all C++ targets go through this, hard-coding on individual targets is forbidden.
- ② compile borrows ① converter venv's CPU torch cmake (torch 2.9.0+cpu ABI is stable, the backend .so does not call NPU); no separate build venv is created.
- ③ generate is only reserved: `generate/` + commented-out OHOS toolchain stub, to be completed after the target artifact/NDK is determined.

## Consequences (Consequences)

- **Positive**:
  - env boundaries are clear, four mutually non-overlapping; directly `source scripts/set_env_<env>.sh` (export/compile/generate/npu each have their own named script, with common parts in `_env_common.sh`).
  - `-Werror` landed (zero-warning principle realized), ASAN option integrated (`build_cpp.sh --asan`).
  - CANN paths uniformly go through `ASCEND_HOME_PATH` (fixes P3 cann-9.1.0 bad default); venv/torchair paths go through overridable `TORCH_DELEGATE_HOME`.
  - Unified top-level build tree: a single `bash scripts/cann_lm_compile/build_cpp.sh` produces the backend .so + OM binary + tests.
  - env logic consolidated into `set_env_<env>.sh` + common `_env_common.sh`; `run.sh`/`run_pd6_cases.sh` no longer carry a copy (fixes P4).
- **Negative/Cost**:
  - After removing the `cann-9.1.0` default, running `build_cpp.sh` directly without sourcing CANN env will FATAL (previously it silently used the wrong path). This is intentional fail-fast.
  - ② compile borrows ① venv: torch 2.9.0+cpu is linked into the backend .so; if the .so later needs to coexist with a different torch ABI on NPU/HarmonyOS, whether to create a ② dedicated venv needs re-evaluation.
  - `-Werror` exposes pre-existing dead code `PlanFallbackNode` (ge_graph_builder.cc:1543, 0 references), silenced with `[[maybe_unused]]`; dead code cleanup left for later.
- **Constraints introduced**:
  - New C++ targets **must** integrate common Flags via `delegate_apply_common_flags()`, hard-coding `-Wall/-Wextra/-Werror` is forbidden.
  - Before running C++ tests **must** `bash scripts/cann_lm_compile/build_cpp.sh --asan && ctest --test-dir build` (ASAN flags include `-static-libasan`, bare ctest needs no LD_PRELOAD).
  - New build scripts **must not** hard-code `cann-9.x.0`/venv absolute paths, always use overridable variables like `ASCEND_HOME_PATH`/`TORCH_DELEGATE_HOME`.
  - New content under `generate/` must follow the "reserved stub → complete after toolchain is determined" process, and must not introduce usability assumptions during the stub phase.

## Alternatives (Alternatives)

### Alternative A: ② compile creates a dedicated build venv

Rejected: torch 2.9.0+cpu ABI is stable, the backend .so does not call NPU (only borrows torch cmake + CANN headers at link time), borrowing ① venv costs nothing and keeps the same torch version as converter; creating a ② venv means maintaining a duplicate torch, the cost outweighs the benefit. Retained as a fallback for future ABI conflicts.

### Alternative B: Top-level CMakeLists completely replaces build_csrc.sh (delete build_csrc.sh)

**Adopted** (the original "thin wrapper keeping csrc/" was transitional, ultimately deleted): `build_csrc.sh` has been deleted, and the 6 call sites (2 example CMakeLists + tests CMakeLists + `export_epair_and_compile/run.sh` + `export_quant_matmul_single_op/run.sh` + `om_npu_verify/check_env.sh`) all fall back to `build/libtorch_delegate_backend.so` + `bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend`. A single C++ build logic (build_cpp.sh), with no build scripts left in the cann_lm/ source tree.

### Alternative C: npu folded into converter

Rejected: converter venv **does not install torch_npu** (design decision, goes through the stub path); CANN lightweight tool GPTQ / NPU text generation / OM verification requires real torch_npu, so it cannot be folded. npu must remain independent as verify-runtime.

## Related

- Upstream: [ADR-0001](./0001-load-model-cpu-quant-subprocess-device.md) (CPU quantization + subprocess device)
- Implementation: `scripts/set_env_{export,compile,generate,npu}.sh` + `_env_common.sh`, `scripts/cann_lm_compile/build_cpp.sh`, `cmake/modules/DelegateCommonFlags.cmake`, top-level `CMakeLists.txt`, `generate/`
