#!/usr/bin/env python3
"""CANN-LM Python bindings — usage example.

Demonstrates the full public API of the `cann_lm` ctypes passthrough package
(mirrors cann_lm/engine/api/c/c_api.h 1:1):

  0. Library resolution            — get_library / is_library_available
  1. Engine lifecycle              — Engine (no-model form + real model load)
  2. Context + synchronous generate — create_context / generate / Result
  3. Async generate + streaming    — generate_async / Request.wait / callback
  4. Error handling                — CannLmStatusError (+ partial .result)

Run anywhere; the actual inference flow executes ONLY where a matching-arch
libcann_lm_engine.so can be loaded (OHOS device, or a host with a built .so).
Without one it prints discovery guidance and exits cleanly.

    python cann_lm/python/example.py
    python cann_lm/python/example.py \
        --prefill prefill.omc --decode decode.omc --meta meta.json \
        --prefill-weight weights/prefill --decode-weight weights/decode
"""
import os
import sys

# Make `import cann_lm` resolve from this file's sibling package dir
# (cann_lm/python/cann_lm/ <- cann_lm/python/ on sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cann_lm
from cann_lm import (
    Engine,
    CannLmError,
    CannLmStatusError,
    build_model_artifact,
    build_sampling_params,
    is_library_available,
)


def _print_result(label, r):
    """r is a cann_lm.Result: token_ids (list[int]), text (str), finish_reason."""
    print(f"  [{label}] finish={r.finish_reason.name}  "
          f"tokens={len(r.token_ids)}  text={r.text!r}")


# ────────────────────────── section 2: sync generate ──────────────────────────

def section_sync(eng):
    print("\n=== 2. Context + synchronous generate ===")
    ctx = eng.create_context()
    try:
        # SamplingParams: use build_sampling_params() (NOT a zero-init struct —
        # that would give 0.0 defaults instead of the C++ 1.0 defaults).
        params = build_sampling_params(
            temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)

        # RequestOptions (max_tokens / stop_token_ids / stop_strings) are passed
        # as kwargs to generate(); they are marshaled to the C struct internally.
        r = eng.generate(
            "你好，请介绍一下你自己。",
            ctx,
            sampling_params=params,
            max_tokens=128,
            stop_strings=["</s>", "EOS"],
        )
        _print_result("sync", r)
    except CannLmStatusError as e:
        # Non-SUCCESS: FAILURE still fills a (partial) Result, attached to the
        # exception; other errors (INVALID_ARGUMENT, NOT_FOUND, ...) do not.
        print(f"  [sync] {e.func_name} -> {e.status.name}  "
              f"(partial result attached: {e.result is not None})")
        if e.result is not None:
            _print_result("sync.partial", e.result)
    finally:
        eng.destroy_context(ctx.id)


# ────────────────────────── section 3: async + streaming ──────────────────────────

def section_async(eng):
    print("\n=== 3. Async generate + streaming callback ===")
    ctx = eng.create_context()
    streamed = []

    def on_token(token_id, text):
        # Fires on the pipeline internal thread — must not block or call Engine
        # APIs from here.
        streamed.append(text)
        print(f"    >> token {token_id}: {text!r}")

    try:
        req = eng.generate_async(
            "写一句关于秋天的诗。",
            ctx,
            callback=on_token,
            max_tokens=64,
        )
        print(f"  status={req.status.name}  done={req.is_done}")
        r = req.wait()            # block until done
        _print_result("async", r)
        print(f"  streamed {len(streamed)} token(s) via callback")
    except CannLmStatusError as e:
        print(f"  [async] {e.func_name} -> {e.status.name}")
    finally:
        eng.destroy_context(ctx.id)


# ────────────────────────── main ──────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="cann_lm Python bindings usage example")
    ap.add_argument("--prefill", help="prefill graph path (.omc)")
    ap.add_argument("--decode", help="decode graph path (.omc)")
    ap.add_argument("--meta", help="meta.json path")
    ap.add_argument("--prefill-weight", help="prefill weights directory")
    ap.add_argument("--decode-weight", help="decode weights directory")
    ap.add_argument("--lib", help="explicit libcann_lm_engine.so path "
                                  "(else auto-discovered)")
    args = ap.parse_args()

    # ── 0. Library ──
    print("=== 0. Library ===")
    if not is_library_available(args.lib):
        print("  No loadable libcann_lm_engine.so found.")
        print("  Set CANN_LM_LIB or pass --lib=<path> to a matching-arch .so")
        print("  (e.g. deploy_ohos/libcann_lm_engine.so on the OHOS device).")
        return
    print(f"  loaded: {cann_lm.get_library(args.lib).cann_lm_path}")

    # ── 1. Engine ──
    print("\n=== 1. Engine ===")
    if args.prefill or args.decode or args.meta:
        # Real model: build a ModelArtifact (paths + optional in-memory buffers).
        # The C side stores buffer data as a VIEW (no copy), so the keepalive
        # must outlive the model-load phase — Engine holds it for its lifetime.
        artifact = build_model_artifact(
            prefill_omc_path=args.prefill,
            decode_omc_path=args.decode,
            meta_json_path=args.meta,
            prefill_weight=args.prefill_weight,
            decode_weight=args.decode_weight,
        )
        print("  loading real model from provided paths")
    else:
        # No-model form (c_api.h:155): all-empty artifact -> construct succeeds,
        # lifecycle API usable, Generate reports FAILURE.
        artifact = None
        print("  no-model form (Generate will report FAILURE; lifecycle OK)")

    try:
        # Engine is a context manager: __exit__ -> ExecutorDestroy (frees the
        # C handle). __del__ also closes, so it's safe even without `with`.
        with Engine(artifact=artifact, lib_path=args.lib) as eng:
            print(f"  executor handle: {eng.handle:#x}")
            section_sync(eng)
            section_async(eng)
    except CannLmStatusError as e:
        # ExecutorCreate init failure (e.g. NPU quota exhausted during load —
        # c_api.h:173: *out stays NULL, failure code surfaced here).
        print(f"  [engine] init failed: {e.status.name}  ({e})")
    except CannLmError as e:
        # Library-side error (e.g. .so disappeared between probe and use).
        print(f"  [engine] library error: {e}")


if __name__ == "__main__":
    main()
