#!/usr/bin/env python3
"""cann_lm.chat — interactive multi-turn chat CLI (mlx_lm-style).

Mirrors `cann_lm.generate`: config-driven model + a REPL loop. Each turn appends
the user's input + the model's reply to a running history; build_prompt() flattens
that history into a single prompt string ending with 'Assistant: ' so the engine
continues the assistant's reply. Round cap (20) + cumulative prompt token budget
(16k, conservative estimate) guard the loop.

    cann_lm chat --config testcase.json
    cann_lm.generate chat --config testcase.json --max-tokens 128

--config provides model paths + tokenizer + sampling defaults (from
sentences[0], same contract as generate/bbit); --max-tokens overrides the
per-round generation cap. Sync generate per turn + print the reply.

Exit codes: 0 = session ended cleanly (EOF/quit/round-cap); 1 = engine/generate
failure; 2 = config/usage/no-engine-library.
"""

import argparse
import os
import sys
import threading

# Make `import cann_lm` resolve when run directly (python cann_lm/chat.py)
# or via `python -m cann_lm.chat` without pip install. cann_lm/python/ is
# the parent of this package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cann_lm.config import load_config, print_resolved_paths  # noqa: E402  (package module; shared with generate, pip-installed)
import cann_lm  # noqa: E402  (lazy .so — get_library/Engine gate the real load)
from cann_lm._token_count import count_tokens  # noqa: E402  (stdlib-only budget guard)


def add_args(p):
    """Register the chat flags on parser `p` (shared by the standalone
    `cann_lm.chat` command and the `cann_lm chat` subcommand). Mirrors
    generate.add_args minus --prompt (chat is interactive: stdin per turn)."""
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="testcase.json (model paths + tokenizer + sampling defaults)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="override per-round config maxGenTokens",
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


# animated spinner shown while the engine generates (see _ThinkingIndicator);
# cycles thinking. -> thinking.. -> thinking... so a live generate is visibly
# distinct from a frozen hang.
_THINKING_FRAMES = ("thinking.", "thinking..", "thinking...")


class _ThinkingIndicator:
    """Animated spinner — 'thinking.' -> 'thinking..' -> 'thinking...' cycled on
    stdout (\\r + frame, flush) in a daemon thread while eng.generate() blocks.
    Gives the user live feedback that the model is working: a frozen static
    'thinking...' looks identical to a hang. stop() joins the thread and clears
    the line so the reply / error starts at column 0.

    No third-party deps — stdlib threading only."""

    def __init__(self, frames=_THINKING_FRAMES, interval=0.2):
        self._frames = frames
        self._width = max(len(f) for f in frames)
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        out = sys.stdout
        i = 0
        while not self._stop.is_set():
            out.write("\r" + self._frames[i % len(self._frames)])
            out.flush()
            i += 1
            self._stop.wait(self._interval)  # responsive to stop()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # overwrite the spinner with blanks, cursor back to column 0
        sys.stdout.write("\r" + " " * self._width + "\r")
        sys.stdout.flush()


def build_prompt(history):
    """Flatten chat history into a plain-text prompt ending with 'Assistant: '.

    Each turn -> '{Role}: {content}'; joined by '\\n'; a trailing 'Assistant: '
    (no content) invites the model to continue the assistant's next reply.
    """
    parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in history]
    parts.append("Assistant: ")
    return "\n".join(parts)


# user spec: 输入轮次小于20; 总累计 prompt 不超过 16k tokens
MAX_ROUNDS = 20
MAX_PROMPT_TOKENS = 16384


def check_round_limit(rounds_completed, max_rounds=MAX_ROUNDS):
    """True if the chat loop should stop before accepting another round.

    Allows up to `max_rounds` completed rounds; the (max_rounds+1)-th is blocked.
    """
    return rounds_completed >= max_rounds


def enforce_budget(history, count_fn, max_tokens=MAX_PROMPT_TOKENS):
    """FIFO-drop oldest turns from the front until count_fn(build_prompt(h)) <=
    max_tokens. Keeps at least 1 turn. Returns (trimmed_history, dropped_count);
    (None, 0) when even a single turn exceeds the budget (caller rejects input).

    count_fn(prompt:str)->int is injected so the budget check is testable without
    a real tokenizer (production wires it to _token_count.count_tokens).
    """
    h = list(history)
    dropped = 0
    while count_fn(build_prompt(h)) > max_tokens and len(h) > 1:
        h.pop(0)
        dropped += 1
    if count_fn(build_prompt(h)) > max_tokens:
        return None, 0
    return h, dropped


def run(args):
    """The chat flow: load config → (--show-paths dry-run | engine gate + REPL
    loop). Exit code."""
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
        cann_lm.get_library(args.lib)
    except cann_lm.CannLmError as e:
        print(f"cann_lm chat: cannot load libcann_lm_engine.so:\n{e}", file=sys.stderr)
        return 2

    s0 = cfg["sentences"][0]
    sp = cann_lm.build_sampling_params(
        temperature=s0["temperature"],
        top_k=s0["top_k"],
        top_p=s0["top_p"],
        seed=s0["seed"],
        repetition_penalty=s0["repetition_penalty"],
    )
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
    max_tokens = args.max_tokens if args.max_tokens is not None else s0["max_tokens"]
    stop = (
        list(args.stop_strings)
        if args.stop_strings is not None
        else list(s0["stop_strings"])
    )
    tok_path = cfg["tokenizer_path"]

    # 3. Engine (one per session; init fails fast if model paths absent)
    try:
        eng = cann_lm.Engine(artifact=artifact, options=opts, lib_path=args.lib)
    except (cann_lm.CannLmError, cann_lm.CannLmStatusError) as e:
        print(f"cann_lm chat: engine init failed: {e}", file=sys.stderr)
        return 1

    # 4. REPL loop: one Context for the session; /quit / empty / EOF / Ctrl-C exit
    history = []
    rounds = 0
    try:
        ctx = eng.create_context()
        try:
            while True:
                if check_round_limit(rounds, max_rounds=MAX_ROUNDS):
                    print(f"cann_lm chat: round limit ({MAX_ROUNDS}) reached, exiting.")
                    break
                try:
                    line = input("user> ")
                except (EOFError, KeyboardInterrupt):
                    break
                stripped = line.strip()
                if not stripped or stripped in ("/quit", "/exit"):
                    break
                history.append({"role": "user", "content": stripped})
                trimmed, dropped = enforce_budget(
                    history,
                    lambda p: count_tokens(p, tok_path),
                    max_tokens=MAX_PROMPT_TOKENS,
                )
                if trimmed is None:
                    print(
                        f"cann_lm chat: this input alone exceeds the "
                        f"{MAX_PROMPT_TOKENS}-token budget; not sent. Try "
                        f"shorter input.",
                        file=sys.stderr,
                    )
                    history.pop()  # reject the un-sendable user turn
                    continue
                if dropped:
                    print(
                        f"cann_lm chat: dropped {dropped} oldest turn(s) to "
                        f"fit the {MAX_PROMPT_TOKENS}-token budget.",
                        file=sys.stderr,
                    )
                    history = trimmed
                prompt = build_prompt(history)
                ind = _ThinkingIndicator()
                ind.start()
                try:
                    r = eng.generate(
                        prompt,
                        ctx,
                        sampling_params=sp,
                        max_tokens=max_tokens,
                        stop_strings=stop,
                    )
                except cann_lm.CannLmStatusError as e:
                    ind.stop()  # clear spinner before surfacing the error
                    partial = e.result.text if (e.result is not None) else ""
                    print(
                        f"cann_lm chat: generate {e.status.name}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if partial:
                        print(partial, flush=True)
                    history.pop()  # failed round: drop user turn, retry
                    continue
                ind.stop()  # clear spinner; reply starts at column 0
                print(f"assistant> {r.text}", flush=True)
                history.append({"role": "assistant", "content": r.text})
                rounds += 1
        finally:
            eng.destroy_context(ctx.id)
    finally:
        eng.close()
    return 0


def main(argv=None):
    """Standalone `cann_lm.chat` command entry (also registered in pyproject).
    Also available as: cann_lm chat ..."""
    ap = argparse.ArgumentParser(
        prog="cann_lm.chat",
        description="CANN-LM interactive chat CLI (standalone). "
        "Also available as: cann_lm chat ...",
    )
    add_args(ap)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
