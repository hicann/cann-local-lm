# Repository Layout Restructuring

English | [简体中文](../../zh/adr/0003-repo-layout-restructure.md)

The project directory was messy: Python bindings scattered in `cann_local_lm/python/`, docs not aligned with source code context, tests mixing ut/st/e2e, vendored source mixed with find modules, `generate/` reserved placeholder being redundant (the real OHOS build is already in `scripts/cann_lm_engine/`), and the root `build.sh` and `scripts/build.sh` being duplicate double entry points. The decision is to align source and docs by engine/converter context; `tests/` holds only ut, `examples/` holds st/end-to-end; vendored source is unified into `cmake/third_party/` (same root as find modules); build scripts are unified under `scripts/` (root `build.sh` serves as a thin entry point to preserve the `bash build.sh` habit); the `generate/` placeholder and `deploy_ohos/` tracked artifacts are deleted.

## Considered Options

- Maintain the status quo: directories inconsistent with context, duplicate entry points, blurred test boundaries — rejected.
- One-shot big refactor squash commit: hard to review, coarse single-point rollback granularity — rejected.
- **Step-by-step commits**: one commit per decision, easier to review and single-point rollback — adopted.

## Consequences

- The `third_party` include paths in `cann_local_lm/engine/CMakeLists.txt` (5 places) and `cann_local_lm/engine/ohos/CMakeLists.txt` (5 places) are changed to `cmake/third_party/nlohmann`.
- The root `build.sh`'s `--pkg`/`-u` logic is migrated into `scripts/build.sh` as `pkg`/`ut` subcommands; the 11 `bash build.sh` references inside docs/examples remain unchanged (thin entry preserves the habit).
- `scripts/cann_lm_engine/pyproject.toml` + `build_wheel.sh` references to `cann_local_lm/python/` are updated to `cann_local_lm/engine/python/`.
- Added `GLOSSARY.md` (ut/st/e2e/4 envs/vendored vs find module/context boundary terminology).

## Related

- `docs/adr/0002-env-build-isolation.md` (4 build envs; ③ generate is now carried by `scripts/cann_lm_engine/build_ohos.sh`).
