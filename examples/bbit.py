#!/usr/bin/env python3
"""bbit — Python mirror of examples/test_engine/bbit/bbit.cpp (Black-Box Integration Test).

A real user of the engine via the **cann_lm** Python bindings module: it calls
every lifecycle step in sequence. Mirrors the LATEST bbit.cpp (split weight
fields, fail-fast/degraded-run contract, async block commented out), check
counters, and exit codes (0=all pass, 1=a check failed, 2=config/usage/
unsupported).

Two requirements vs the raw-ctypes approach:
  1. USE the cann_lm Python module (Engine/Context/Result + build_* helpers;
     --check NULL defense + the degraded ContextCreate(NULL) use cann_lm._ffi,
     the loaded-lib submodule).
  2. FEATURE-CHECK the module before use ("判断头文件支不支持"): verify cann_lm
     exposes the needed public API; if not, report unsupported and exit 2.

Flow contract (mirrors bbit.cpp run_flow, 2026-09-22 fail-fast):
  - omc paths configured but absent on disk (host stub-era with device paths)
    → DEGRADED: ExecutorCreate must fail, executor null, ContextCreate(NULL)
    → INVALID_ARGUMENT; verify contract, return early.
  - omc on disk (real deployment) → full flow: create → context → per-sentence
    Generate (SUCCESS/FAILURE/INVALID_STATE all pass) → ContextCancel →
    ContextDestroy×2 → ExecutorDestroy. (Async + Request lifecycle are
    commented out in the latest bbit.cpp, so omitted here too.)

Usage:
  python bbit.py --check            NULL defense tests (C ABI boundary)
  python bbit.py --config <path>    full API flow driven by JSON config
  python bbit.py                    same; default-searches ./testcase.json
"""

import os
import sys

# Make `import cann_lm` resolve from this file's sibling package dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctypes as ct  # noqa: E402  (after sys.path setup)

import cann_lm


# ────────────────────────── feature-check ("判断头文件支不支持") ──────────────────────────

REQUIRED_API = (
    "Engine",
    "Context",
    "Request",
    "Result",
    "build_model_artifact",
    "build_engine_options",
    "build_request_options",
    "build_sampling_params",
    "is_library_available",
    "get_library",
    "Status",
    "RequestStatus",
    "FinishReason",
    "CannLmError",
    "CannLmStatusError",
)


def supports_bbit_flow(module):
    """True iff `module` exposes every public name bbit's flow needs (API gate)."""
    return all(hasattr(module, name) for name in REQUIRED_API)


# ────────────────────────── generate-status acceptance ──────────────────────────
# Mirror bbit.cpp check_generate_status: SUCCESS / FAILURE / INVALID_STATE pass.


def is_generate_status_ok(status):
    s = status if isinstance(status, cann_lm.Status) else cann_lm.Status(int(status))
    return s in (
        cann_lm.Status.SUCCESS,
        cann_lm.Status.FAILURE,
        cann_lm.Status.INVALID_STATE,
    )


# ────────────────────────── check counters ──────────────────────────

_g_total = 0
_g_pass = 0


def check(ok, msg):
    global _g_total, _g_pass
    _g_total += 1
    if ok:
        _g_pass += 1
    else:
        print(f"FAIL: {msg}", file=sys.stderr)


def check_generate_status(status, msg):
    global _g_total, _g_pass
    _g_total += 1
    if is_generate_status_ok(status):
        _g_pass += 1
    else:
        print(f"FAIL: {msg} (got {int(status)})", file=sys.stderr)


def _reset_counters():
    global _g_total, _g_pass
    _g_total = _g_pass = 0


# Config loader + file_exists + is_degraded_run live in cann_lm.config (shared
# with the cann_lm.generate CLI; pip-installed as a package module). Re-exported
# here so `from bbit import load_config` keeps working (run_check/run_flow/main
# are byte-identical — only the loader's source changed, not bbit's logic).
from cann_lm.config import file_exists, is_degraded_run, load_config  # noqa: F401


# ────────────────────────── --check: NULL defense (C ABI contract) ──────────────────────────
# Uses cann_lm._ffi (the loaded-lib submodule) to call the C functions directly
# with NULL, mirroring bbit.cpp run_check.


def run_check():
    _reset_counters()
    try:
        lib = cann_lm.get_library()
    except cann_lm.CannLmError as e:
        print(f"bbit --check: no engine library: {e}", file=sys.stderr)
        return 2

    INVALID = int(cann_lm.Status.INVALID_ARGUMENT)
    # Executor-level: NULL inputs → INVALID_ARGUMENT
    check(
        lib.CANN_LMEngine_ExecutorCreate(None, None, None) == INVALID,
        "ExecutorCreate NULL → INVALID_ARGUMENT",
    )
    check(
        lib.CANN_LMEngine_ExecutorGenerate(None, None, 0, None, None, None) == INVALID,
        "ExecutorGenerate NULL → INVALID_ARGUMENT",
    )
    check(
        lib.CANN_LMEngine_ContextCreate(None, None) == INVALID,
        "ContextCreate NULL → INVALID_ARGUMENT",
    )
    check(
        lib.CANN_LMEngine_ContextDestroy(None, 0) == INVALID,
        "ContextDestroy NULL → INVALID_ARGUMENT",
    )
    check(
        lib.CANN_LMEngine_ContextCancel(None, 0) == INVALID,
        "ContextCancel NULL → INVALID_ARGUMENT",
    )

    # Request-level: NULL inputs → safe defaults (not INVALID_ARGUMENT)
    check(
        lib.CANN_LMEngine_RequestGetStatus(None) == int(cann_lm.RequestStatus.QUEUED),
        "RequestGetStatus NULL → QUEUED",
    )
    check(lib.CANN_LMEngine_RequestIsDone(None) == 0, "RequestIsDone NULL → 0")
    check(
        lib.CANN_LMEngine_RequestCancel(None) == int(cann_lm.Status.SUCCESS),
        "RequestCancel NULL → SUCCESS",
    )
    check(
        lib.CANN_LMEngine_RequestWait(None, None) == INVALID,
        "RequestWait NULL → INVALID_ARGUMENT",
    )
    check(
        lib.CANN_LMEngine_RequestTryGetResult(None, None) == INVALID,
        "RequestTryGetResult NULL → INVALID_ARGUMENT",
    )

    # NULL-safe no-ops (no crash = pass)
    lib.CANN_LMEngine_ExecutorDestroy(None)
    check(True, "ExecutorDestroy NULL no-op")
    lib.CANN_LMEngine_RequestDestroy(None)
    check(True, "RequestDestroy NULL no-op")
    lib.CANN_LMEngine_ResultDestroy(None)
    check(True, "ResultDestroy NULL no-op")

    print(f"bbit --check: {_g_pass}/{_g_total} passed")
    return 0 if _g_pass == _g_total else 1


# ────────────────────────── --config: full flow via cann_lm wrappers ──────────────────────────


def _try(ok_msg, fn, err_msg=None):
    """Run a status-returning wrapper call; map raise→check(False), ok→check(True)."""
    try:
        r = fn()
        check(True, ok_msg)
        return r
    except (cann_lm.CannLmError, cann_lm.CannLmStatusError) as e:
        check(False, f"{err_msg or ok_msg} ({getattr(e, 'status', '?')})")
        return None


def run_flow(cfg):
    _reset_counters()
    print(
        f"bbit: using config: prefill={cfg['prefill_omc_path']} "
        f"decode={cfg['decode_omc_path']} "
        f"prefill_weight={cfg['prefill_weight']} "
        f"decode_weight={cfg['decode_weight']} "
        f"meta={cfg['meta_json_path']} ({len(cfg['sentences'])} sentence(s))"
    )
    print(
        f"bbit: tokenizer via EngineOptions: type={cfg['tokenizer_type']} "
        f"path={cfg['tokenizer_path']}"
    )

    # 1. ExecutorCreate — build artifact (split weights) + EngineOptions
    artifact = cann_lm.build_model_artifact(
        prefill_omc_path=cfg["prefill_omc_path"],
        decode_omc_path=cfg["decode_omc_path"],
        meta_json_path=cfg["meta_json_path"],
        prefill_weight=cfg["prefill_weight"],
        decode_weight=cfg["decode_weight"],
    )
    opts = cann_lm.build_engine_options(
        default_chunk_size=1024,
        max_concurrent_prefill=1,
        tokenizer_type=cfg["tokenizer_type"],
        tokenizer_path=cfg["tokenizer_path"],
    )
    try:
        eng = cann_lm.Engine(artifact=artifact, options=opts)
    except (cann_lm.CannLmError, cann_lm.CannLmStatusError):
        eng = None

    # 2026-09-22 fail-fast / degraded-run contract:
    # omc configured but absent on disk → create MUST fail; verify contract; stop.
    if is_degraded_run(cfg["prefill_omc_path"], cfg["decode_omc_path"]):
        print(
            f"bbit: model omc absent on this machine "
            f"({cfg['prefill_omc_path']} / {cfg['decode_omc_path']}) — "
            f"degraded run: engine aborts init, ExecutorCreate reports error"
        )
        check(eng is None, "ExecutorCreate (omc absent → engine init must fail)")
        check(
            eng is None,
            "failed ExecutorCreate → null executor (no half-initialized engine)",
        )
        # ContextCreate(NULL executor) → INVALID_ARGUMENT (raw lib, like --check)
        lib = cann_lm.get_library()
        no_cid = ct.c_uint64(0)
        check(
            lib.CANN_LMEngine_ContextCreate(None, ct.byref(no_cid))
            == int(cann_lm.Status.INVALID_ARGUMENT),
            "ContextCreate(NULL executor) rejected",
        )
        print(f"bbit flow (degraded, no model): {_g_pass}/{_g_total} passed")
        return 0 if _g_pass == _g_total else 1

    # Full flow: omc on disk (or not configured) → create must succeed.
    check(eng is not None, "ExecutorCreate")
    if eng is None:
        print("bbit: aborted (no executor)")
        return 1
    check(eng.handle is not None, "executor non-null")

    try:
        # 2. ContextCreate
        ctx = _try("ContextCreate", lambda: eng.create_context(), "ContextCreate")
        if ctx is None:
            return 1
        check(ctx.id != 0, "context id non-zero")

        # 3. Generate (sync) — one per sentence (SUCCESS/FAILURE/INVALID_STATE ok)
        for i, s in enumerate(cfg["sentences"]):
            sp = cann_lm.build_sampling_params(
                temperature=s["temperature"],
                top_k=s["top_k"],
                top_p=s["top_p"],
                seed=s["seed"],
                repetition_penalty=s["repetition_penalty"],
            )
            try:
                r = eng.generate(
                    s["prompt"],
                    ctx,
                    sampling_params=sp,
                    max_tokens=s["max_tokens"],
                    stop_strings=s["stop_strings"],
                )
                print(f'bbit: sentence[{i}] prompt="{s["prompt"]}" -> status SUCCESS')
                check_generate_status(
                    cann_lm.Status.SUCCESS,
                    "Generate sync (SUCCESS/FAILURE/INVALID_STATE)",
                )
                check(len(r.token_ids) > 0, "Generate sync produced tokens")
                print(f"bbit: sentence[{i}] generated: {r.text}")
            except cann_lm.CannLmStatusError as e:
                print(
                    f'bbit: sentence[{i}] prompt="{s["prompt"]}" '
                    f"-> status {e.status.name}"
                )
                check_generate_status(
                    e.status, "Generate sync (SUCCESS/FAILURE/INVALID_STATE)"
                )
                if e.result is not None and e.result.text:
                    print(f"bbit: sentence[{i}] partial: {e.result.text}")

        # NOTE: steps 4-6 (GenerateAsync + Request lifecycle) are commented out
        # in the latest bbit.cpp; omitted here in lockstep.

        # 7. ContextCancel (on context with no active generate)
        _try(
            "ContextCancel (no active generate)",
            lambda: eng.cancel_context(ctx.id),
            "ContextCancel",
        )
        # 8. ContextDestroy
        _try("ContextDestroy", lambda: eng.destroy_context(ctx.id), "ContextDestroy")
        # 9. ContextDestroy (idempotent)
        _try(
            "ContextDestroy (idempotent)",
            lambda: eng.destroy_context(ctx.id),
            "ContextDestroy (idempotent)",
        )
    finally:
        # 10. ExecutorDestroy
        eng.close()
        check(True, "ExecutorDestroy")

    print(f"bbit flow: {_g_pass}/{_g_total} passed")
    return 0 if _g_pass == _g_total else 1


# ────────────────────────── main ──────────────────────────


def _print_usage(prog):
    print(
        f"Usage:\n"
        f"  {prog} --check          NULL defense tests (no config needed)\n"
        f"  {prog} --config <path>  full API flow driven by JSON config\n"
        f"  {prog}                  same as --config; default-searches\n"
        f"                         ./testcase.json then <prog dir>/testcase.json",
        file=sys.stderr,
    )


def _default_config_path(argv0):
    if os.path.isfile("testcase.json"):
        return "testcase.json"
    side = os.path.join(os.path.dirname(os.path.abspath(argv0)), "testcase.json")
    if os.path.isfile(side):
        return side
    return None


def main(argv):
    # Feature-check: does the cann_lm module support the bbit flow?
    if not supports_bbit_flow(cann_lm):
        missing = [n for n in REQUIRED_API if not hasattr(cann_lm, n)]
        print(
            f"bbit: cann_lm module does not support bbit flow "
            f"(missing: {', '.join(missing)})",
            file=sys.stderr,
        )
        return 2

    if len(argv) >= 2 and argv[1] == "--check":
        return run_check()

    config_path = None
    i = 1
    while i < len(argv):
        if argv[i] == "--config":
            if i + 1 >= len(argv):
                print("bbit: missing value for --config", file=sys.stderr)
                _print_usage(argv[0])
                return 2
            config_path = argv[i + 1]
            i += 2
        else:
            print(f"bbit: unknown argument: {argv[i]}", file=sys.stderr)
            _print_usage(argv[0])
            return 2

    if config_path is None:
        config_path = _default_config_path(argv[0])
        if config_path is None:
            print(
                "bbit: config not found (looked for ./testcase.json "
                "and testcase.json next to the script)",
                file=sys.stderr,
            )
            _print_usage(argv[0])
            return 2

    cfg = load_config(config_path)
    if cfg is None:
        return 2

    # .so gate for the flow (module API supported, but need the engine lib)
    if not cann_lm.is_library_available():
        print(
            "bbit: no loadable libcann_lm_engine.so "
            "(set CANN_LM_LIB or run on a device with a matching-arch .so)",
            file=sys.stderr,
        )
        return 2

    return run_flow(cfg)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
