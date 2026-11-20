#!/usr/bin/env python3
"""cann_local_lm.generate — mlx_lm-style LLM CLI on HarmonyOS.

A thin user-facing layer ON TOP of the existing cann_local_lm Python bindings module
(+ bbit's testcase.json config loader, reused as-is — bbit is not modified).
Mirrors `mlx_lm.generate`: one command, config-driven model + a prompt → print
the response.

    cann_local_lm.generate --config testcase.json --prompt "你好"
    cann_local_lm.generate --config testcase.json --prompt xxx --max-tokens 128 --stop '</s>'

--config provides model paths + tokenizer + sampling defaults (from
sentences[0], same contract as bbit); --prompt / --max-tokens / --stop override
the prompt + termination. Sync generate + final print (bbit.cpp's async block is
commented out → sync is the reliable path).

Invocation:
  - device shim:  cann_local_lm.generate --config ... --prompt ...   (cann_local_lm.generate.sh)
  - module mode:  python -m cann_local_lm.generate --config ... --prompt ...
  - direct:       python cann_local_lm/engine/python/cann_local_lm/generate.py --config ... --prompt ...

Exit codes: 0 = success (response printed); 1 = engine/generate failure;
2 = config/usage/no-engine-library.
"""

import argparse
import os
import sys

# Make `import cann_local_lm` resolve when run directly (python cann_local_lm/generate.py)
# or via `python -m cann_local_lm.generate` without pip install. cann_local_lm/engine/python/
# is the parent of this package (C2 后由 cann_local_lm/python/ 迁入)。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cann_local_lm.config import load_config, print_resolved_paths  # noqa: E402  (package module; self-contained, pip-installed)

import cann_local_lm  # noqa: E402


def resolve_request(cfg, *, prompt=None, max_tokens=None, stop_strings=None):
    """Resolve effective prompt + termination from config (sentences[0]) + CLI
    overrides. Returns (prompt, max_tokens, stop_strings)."""
    s0 = cfg["sentences"][0]
    prompt = prompt if prompt is not None else s0["prompt"]
    max_tokens = max_tokens if max_tokens is not None else s0["max_tokens"]
    stop = list(stop_strings) if stop_strings is not None else list(s0["stop_strings"])
    return prompt, max_tokens, stop


def add_args(p):
    """Register the generate flags on parser `p` (shared by the standalone
    `cann_local_lm.generate` command and the `cann_local_lm generate` subcommand)."""
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="testcase.json (model paths + tokenizer + sampling defaults)",
    )
    p.add_argument(
        "--prompt", metavar="TEXT", help="prompt (default: config sentences[0].prompt)"
    )
    p.add_argument(
        "--max-tokens", type=int, metavar="N", help="override config maxGenTokens"
    )
    p.add_argument(
        "--stop",
        action="append",
        dest="stop_strings",
        metavar="STR",
        help="stop string (repeatable; default: config stopSeq)",
    )
    p.add_argument(
        "--lib",
        metavar="PATH",
        help="explicit libcann_lm_engine.so path (else auto-discovered)",
    )
    p.add_argument(
        "--show-paths",
        action="store_true",
        help="print the config's resolved paths + on-disk existence, then exit (no .so, no Engine)",
    )


def run(args):
    """The generate flow: load config → Engine → sync generate → print. Exit code."""
    # 1. config
    cfg = load_config(args.config)
    if cfg is None:
        return 2

    # 1b. --show-paths: dry-run — print resolved paths + existence, exit (no .so)
    if getattr(args, "show_paths", False):
        missing = print_resolved_paths(cfg)
        return 1 if missing else 0

    # 2. engine-library gate — surface the real load failure (missing .so, or
    #    .so found but its runtime deps absent from LD_LIBRARY_PATH, or wrong arch)
    try:
        cann_local_lm.get_library(args.lib)
    except cann_local_lm.CannLmError as e:
        print(
            f"cann_local_lm generate: cannot load libcann_lm_engine.so:\n{e}",
            file=sys.stderr,
        )
        return 2

    prompt, max_tokens, stop_strings = resolve_request(
        cfg,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        stop_strings=args.stop_strings,
    )
    s0 = cfg["sentences"][0]
    sp = cann_local_lm.build_sampling_params(
        temperature=s0["temperature"],
        top_k=s0["top_k"],
        top_p=s0["top_p"],
        seed=s0["seed"],
        repetition_penalty=s0["repetition_penalty"],
    )
    artifact = cann_local_lm.build_model_artifact(
        prefill_omc_path=cfg["prefill_omc_path"],
        decode_omc_path=cfg["decode_omc_path"],
        meta_json_path=cfg["meta_json_path"],
        prefill_weight=cfg["prefill_weight"],
        decode_weight=cfg["decode_weight"],
    )
    opts = cann_local_lm.build_engine_options(
        default_chunk_size=1024,
        max_concurrent_prefill=1,
        tokenizer_type=cfg["tokenizer_type"],
        tokenizer_path=cfg["tokenizer_path"],
    )

    # 3. Engine (loads model from the config paths; init fails fast if absent)
    try:
        eng = cann_local_lm.Engine(artifact=artifact, options=opts, lib_path=args.lib)
    except (cann_local_lm.CannLmError, cann_local_lm.CannLmStatusError) as e:
        print(f"cann_local_lm generate: engine init failed: {e}", file=sys.stderr)
        return 1

    # 4. generate (sync) + print
    try:
        ctx = eng.create_context()
        try:
            eng.set_context(ctx)  # ADR-0002: activate the context before generate
            r = eng.generate(
                prompt,
                sampling_params=sp,
                max_tokens=max_tokens,
                stop_strings=stop_strings,
            )
            print(r.text)
            return 0
        except cann_local_lm.CannLmStatusError as e:
            partial = e.result.text if (e.result is not None) else ""
            print(
                f"cann_local_lm generate: generate {e.status.name}: {e}",
                file=sys.stderr,
            )
            if partial:
                print(partial)  # print any partial text to stdout
            return 1
        finally:
            eng.destroy_context(ctx.id)
    finally:
        eng.close()


def main(argv=None):
    """Standalone `cann_local_lm.generate` command entry (also registered in pyproject)."""
    ap = argparse.ArgumentParser(
        prog="cann_local_lm.generate",
        description="CANN-LM generate CLI (standalone). "
        "Also available as: cann_local_lm generate ...",
    )
    add_args(ap)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
