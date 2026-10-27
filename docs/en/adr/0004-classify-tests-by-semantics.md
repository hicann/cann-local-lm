# Classify tests by semantics, not by toolchain dependency (correcting C7's misjudgment of epair_archive)

English | [简体中文](../../zh/adr/0004-classify-tests-by-semantics.md)

## Background

C7 (348d28c, refactor(layout): tests/bbit + tests/c++/epair_archive → examples/) moved `tests/c++/epair_archive/` to `examples/epair_archive/`, with the commit message stating "converter C++ integration, needs CANN", and thus categorized it under `examples/` (per ADR 0003's principle: `tests/=ut only`, `examples/=st/end-to-end`).

The author of epair_archive (a colleague) raised an objection: **these are UTs**.

## Root cause of the misjudgment

"**Requires the CANN toolchain to compile**" was treated as "**NPU integration test**". The two are different:

- `examples/epair_archive/CMakeLists.txt` indeed does `FATAL_ERROR` if `ASCEND_HOME_PATH` is unset, calls `find_package(ascendcl/acl_rt/graph/...)`, and at runtime `LD_LIBRARY_PATH=CANN_LIB_DIRS` — **compiling/linking/loading requires CANN libraries**, this is true.
- But a full grep of the 7 `.cc` files for device/execution APIs (`aclInit`/`aclrtCreate`/`aclmdlLoad`/`aclmdlExecute`/`GeExecutor`/`RunGraph`/`CompileGraph`/`->Run`) returns **all empty**. The assertions are all host-side pure logic: `test_base64.cc` tests base64 decoding; `test_epair_model_parser.cc` tests `parsed_model.nodes.size()`/`parser.SchemaVersionValue()`/`GraphInputs().size()`/`PlaceholderKindOf`/`FindTensorValue`. **It does not touch the device, does not run the model on NPU.**

That is: epair_archive is a UT that "requires CANN libraries to compile and link, but does not invoke NPU execution at runtime", not an integration test. C7 categorized by "needs CANN", mistaking **build toolchain dependency** for **test semantics**.

## Decision

**Classify tests by semantics (what they assert, what they invoke at runtime), not by build toolchain dependency (what they link, what toolchain is needed to compile).**

Determination method: grep the test source for device/execution APIs (`aclrt*`/`aclmdl*`/`GeExecutor`/`RunGraph`/`CompileGraph`/`->Run`). If none are found, even if it requires CANN library linking, it is a UT and belongs in `tests/`. What toolchain is needed to compile does not enter this determination.

Accordingly: `epair_archive` is corrected back to `tests/cpp/epair_archive/` (3 levels deep, parallel to `tests/py/` as a per-language root, avoiding the shell-unfriendly `c++/`). ADR 0003's "tests/=ut only, examples/=integration" principle **was always correct itself**; it was C7 that **misapplied** it — only after correction is the principle correctly executed.

## Incidental fix

C7's move was a pure rename (git show shows 0 insert/0 delete), but `epair_archive/CMakeLists.txt:11`'s `set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")` was originally calibrated for 3-level-deep `tests/c++/epair_archive` (3 levels up = repo root); after moving to 2-level-deep `examples/epair_archive`, `../../..` points one level above the repo (wrong path). Because `BUILD_CONVERTER` defaults to OFF and requires CANN, it was never built for a long time, so the bug lay dormant. Moving it back to 3-level-deep `tests/cpp/epair_archive` restores correctness (`../../..` = repo root), with no need to modify the CMakeLists.

## How to avoid next time

Before classifying a test, first look at **what it tests** (assertions) and **what it invokes at runtime** (grep device/execution APIs), then look at **what it needs to compile**. The order of these three cannot be reversed — toolchain dependency is the most visible but the least appropriate as a classification basis. Needing CANN library linking ≠ integration test.

## Related

- `docs/adr/0003-repo-layout-restructure.md` (the tests/=ut, examples/=integration principle; this ADR is its correct application, not an overturning of it)
- `tests/README.md` §1 classification rules, `examples/epair_archive/UT_DESIGN.md` (moved back to `tests/cpp/epair_archive/` along with the directory)
