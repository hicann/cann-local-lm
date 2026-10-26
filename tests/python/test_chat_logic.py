"""Tests for the cann_lm.chat CLI's pure logic + arg wiring + cli dispatch +
the interactive run() loop (exercised with a mocked Engine + faked stdin, no .so).

Pure-logic helpers (build_prompt / enforce_budget / check_round_limit / add_args)
and the cli dispatch are tested without any engine. The run() loop is exercised
end-to-end via a FakeEngine + monkeypatched builtins.input.
"""

import argparse

from cann_lm import chat
from cann_lm import cli
from cann_lm.chat import add_args, build_prompt, check_round_limit, enforce_budget, run


class TestBuildPrompt:
    """build_prompt: history turns -> plain-text prompt ending with 'Assistant: '
    so the model continues the assistant's reply."""

    def test_single_user_turn_appends_trailing_assistant(self):
        history = [{"role": "user", "content": "hi"}]
        assert build_prompt(history) == "User: hi\nAssistant: "

    def test_user_assistant_user_pair(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ]
        assert build_prompt(history) == (
            "User: hi\nAssistant: hello\nUser: how are you\nAssistant: "
        )

    def test_empty_history_yields_trailing_assistant_only(self):
        # no turns yet -> just the continuation marker
        assert build_prompt([]) == "Assistant: "

    def test_role_capitalized(self):
        history = [{"role": "user", "content": "q"}]
        assert build_prompt(history).startswith("User: ")


class TestCheckRoundLimit:
    """check_round_limit(n, max_rounds) -> True means 'stop, cap reached'.
    The loop allows up to max_rounds completed rounds; the next is blocked."""

    def test_under_cap_ok(self):
        assert check_round_limit(5) is False
        assert check_round_limit(19) is False

    def test_at_cap_stops(self):
        assert check_round_limit(20) is True

    def test_over_cap_stops(self):
        assert check_round_limit(25) is True

    def test_custom_max(self):
        assert check_round_limit(3, max_rounds=3) is True
        assert check_round_limit(2, max_rounds=3) is False


class TestEnforceBudget:
    """enforce_budget: FIFO-drop oldest turns from the front until the prompt
    built from history fits max_tokens. Keeps at least 1 turn; returns
    (None, 0) when even a single turn exceeds the budget."""

    def test_under_budget_unchanged(self):
        hist = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        # count_fn always reports 10 (well under 16k) -> nothing dropped
        out, dropped = enforce_budget(hist, lambda p: 10)
        assert out == hist
        assert dropped == 0

    def test_drops_oldest_until_under(self):
        hist = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        # count_fn = 100 per newline in the built prompt + 100. build_prompt of
        # N history turns yields N newlines (turn lines + trailing Assistant:).
        # full(5)=600>350; drop u1->500>350; drop a1->400>350; drop u2->300<=350.
        # kept=[a2,u3], dropped=3.
        out, dropped = enforce_budget(
            hist, lambda p: 100 * p.count("\n") + 100, max_tokens=350
        )
        assert [t["content"] for t in out] == ["a2", "u3"]
        assert dropped == 3

    def test_single_turn_over_budget_returns_none(self):
        hist = [{"role": "user", "content": "huge"}]
        out, dropped = enforce_budget(hist, lambda p: 99999, max_tokens=250)
        assert out is None
        assert dropped == 0

    def test_never_drops_to_empty(self):
        # even if over budget, we keep >= 1 turn; if 1 still over -> None.
        # 2 turns -> 300 over 250; 1 turn -> 200 under 250 -> keep 1.
        hist = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        out, dropped = enforce_budget(
            hist, lambda p: 100 * p.count("\n") + 100, max_tokens=250
        )
        assert out is not None
        assert len(out) == 1
        assert dropped == 1


class TestChatAddArgs:
    """chat.add_args wires --config/--max-tokens/--stop/--lib/--show-paths onto
    any parser (shared by the standalone command + the subcommand). No --prompt:
    chat is interactive (prompts come from stdin)."""

    def _parse(self, *argv):
        p = argparse.ArgumentParser()
        add_args(p)
        return p.parse_args(list(argv))

    def test_required_config_and_optional_defaults(self):
        a = self._parse("--config", "tc.json")
        assert a.config == "tc.json"
        assert a.max_tokens is None  # optional, not given
        assert a.stop_strings is None
        assert a.lib is None
        assert a.show_paths is False

    def test_max_tokens_and_stop(self):
        a = self._parse(
            "--config",
            "tc.json",
            "--max-tokens",
            "128",
            "--stop",
            "</s>",
            "--stop",
            "EOS",
        )
        assert a.max_tokens == 128
        assert a.stop_strings == ["</s>", "EOS"]  # repeatable --stop

    def test_show_paths_flag(self):
        a = self._parse("--config", "x", "--show-paths")
        assert a.show_paths is True

    def test_missing_config_errors(self):
        import pytest

        with pytest.raises(SystemExit):
            self._parse()  # --config required


class TestCliChatDispatch:
    """cann_lm chat --config ... routes to chat.run (the dispatcher wires the
    chat subcommand into the reserved slot in cli.build_parser)."""

    def test_chat_subcommand_routes_to_chat_run(self):
        args = cli.build_parser().parse_args(["chat", "--config", "tc.json"])
        assert args.command == "chat"
        assert args.func is chat.run
        assert args.config == "tc.json"

    def test_chat_subcommand_args_match_add_args(self):
        args = cli.build_parser().parse_args(
            ["chat", "--config", "x", "--max-tokens", "5", "--stop", "END"]
        )
        assert args.max_tokens == 5
        assert args.stop_strings == ["END"]


class TestChatRunShowPaths:
    """chat.run's --show-paths branch: same dry-run as generate — print resolved
    config paths + on-disk existence, exit before any .so/Engine. No .so needed."""

    def _cfg(self, tmp_path):
        f = tmp_path / "tok.json"
        f.write_text("")
        d = tmp_path / "weights"
        d.mkdir()
        return {
            "tokenizer_type": "BPE",
            "tokenizer_path": str(f),
            "prefill_omc_path": str(d),
            "decode_omc_path": str(tmp_path / "decode.omc"),
            "meta_json_path": str(tmp_path / "model.json"),
            "prefill_weight": str(tmp_path / "w_prefill"),
            "decode_weight": str(tmp_path / "w_decode"),
            "sentences": [{}],
        }

    def test_show_paths_flag_exits_before_so_gate(self, monkeypatch, tmp_path, capsys):
        # --show-paths must NOT touch the .so: monkeypatch get_library to explode
        # if it's ever called; if the flag works, run() returns before the gate.
        import cann_lm

        def boom(*a, **k):
            raise AssertionError("--show-paths must not call get_library")

        monkeypatch.setattr(cann_lm, "get_library", boom)
        cfg = self._cfg(tmp_path)
        monkeypatch.setattr("cann_lm.chat.load_config", lambda c: cfg)
        args = argparse.Namespace(
            config="ignored.json",
            max_tokens=None,
            stop_strings=None,
            lib=None,
            show_paths=True,
        )
        rc = run(args)
        out = capsys.readouterr().out
        assert "tokenizer_path" in out
        assert "MISSING" in out
        assert rc == 1  # missing > 0 → non-zero (actionable)

    def test_show_paths_all_present_returns_zero(self, monkeypatch, tmp_path):
        files = {
            n: (tmp_path / n)
            for n in ("tok.json", "p.omc", "d.omc", "m.json", "wp", "wd")
        }
        for f in files.values():
            f.write_text("")
        cfg = {
            "tokenizer_type": "BPE",
            "tokenizer_path": str(files["tok.json"]),
            "prefill_omc_path": str(files["p.omc"]),
            "decode_omc_path": str(files["d.omc"]),
            "meta_json_path": str(files["m.json"]),
            "prefill_weight": str(files["wp"]),
            "decode_weight": str(files["wd"]),
            "sentences": [{}],
        }
        monkeypatch.setattr("cann_lm.chat.load_config", lambda c: cfg)
        args = argparse.Namespace(
            config="x", max_tokens=None, stop_strings=None, lib=None, show_paths=True
        )
        rc = run(args)
        assert rc == 0

    def test_bad_config_returns_2(self, monkeypatch):
        # load_config returns None (parse error / missing file) -> run returns 2
        monkeypatch.setattr("cann_lm.chat.load_config", lambda c: None)
        args = argparse.Namespace(
            config="bad.json",
            max_tokens=None,
            stop_strings=None,
            lib=None,
            show_paths=False,
        )
        rc = run(args)
        assert rc == 2


# ── run() interactive loop: FakeEngine + faked stdin ──────────────────────


class _FakeResult:
    def __init__(self, text):
        self.text = text
        self.token_ids = []
        self.finish_reason = None


class _FakeContext:
    id = 1


class _FakeEngine:
    """Drop-in for cann_lm.Engine. Records every generate() prompt; optionally
    raises CannLmStatusError on the first N calls to exercise error-continues."""

    fail_first_n = 0  # class-level: fail the first N generate() calls

    def __init__(self, *a, **k):
        self.prompts = []
        self._calls = 0

    def create_context(self):
        return _FakeContext()

    def generate(
        self, prompt, ctx, sampling_params=None, max_tokens=0, stop_strings=None
    ):
        self._calls += 1
        if self._calls <= _FakeEngine.fail_first_n:
            import cann_lm

            raise cann_lm.CannLmStatusError(cann_lm.Status.FAILURE, "ExecutorGenerate")
        self.prompts.append(prompt)
        return _FakeResult(f"reply:{prompt[:8]}")

    def destroy_context(self, cid):
        pass

    def close(self):
        pass


class _SlowFakeEngine(_FakeEngine):
    """_FakeEngine whose generate() blocks briefly — lets the thinking
    indicator's animation thread render multiple frames during the call."""

    delay = 0.3

    def generate(self, *a, **k):
        import time

        time.sleep(self.delay)
        return super().generate(*a, **k)


def _chat_cfg(tmp_path):
    """Minimal valid cfg; tokenizer_path -> empty JSON object (unsupported layout)
    so count_tokens falls back to len(text) — no fixture dependency for the loop."""
    tok = tmp_path / "tok.json"
    tok.write_text("{}")
    return {
        "tokenizer_type": "BPE",
        "tokenizer_path": str(tok),
        "prefill_omc_path": "",
        "decode_omc_path": "",
        "meta_json_path": "",
        "prefill_weight": "",
        "decode_weight": "",
        "sentences": [
            {
                "prompt": "",
                "max_tokens": 16,
                "top_k": 0,
                "top_p": 1.0,
                "temperature": 1.0,
                "seed": 0,
                "repetition_penalty": 1.0,
                "stop_strings": [],
            }
        ],
    }


def _wire_chat_env(monkeypatch, tmp_path, cfg, inputs):
    """Monkeypatch cann_lm.get_library/Engine + load_config + builtins.input so
    run() executes without a .so. Returns nothing; assertions read stdout/stderr."""
    import cann_lm

    monkeypatch.setattr(cann_lm, "get_library", lambda lib=None: object())
    monkeypatch.setattr(cann_lm, "Engine", _FakeEngine)
    monkeypatch.setattr("cann_lm.chat.load_config", lambda c: cfg)
    monkeypatch.setattr(_FakeEngine, "fail_first_n", 0)
    it = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


def _args(**over):
    base = dict(
        config="tc.json", max_tokens=None, stop_strings=None, lib=None, show_paths=False
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestChatRunLoop:
    """run()'s REPL: one Engine+Context for the session, /quit/empty/EOF exit,
    round cap (only successful rounds count), single-round CannLmStatusError
    prints partial+error and continues (doesn't kill the session)."""

    def test_quit_after_one_round(self, monkeypatch, tmp_path, capsys):
        _wire_chat_env(monkeypatch, tmp_path, _chat_cfg(tmp_path), ["hi", "/quit"])
        rc = run(_args())
        out = capsys.readouterr().out
        assert rc == 0
        assert "reply:" in out  # the assistant reply was printed
        assert "User: hi" in out  # prompt fed to generate includes the user turn

    def test_empty_input_exits_cleanly(self, monkeypatch, tmp_path, capsys):
        _wire_chat_env(monkeypatch, tmp_path, _chat_cfg(tmp_path), [""])
        rc = run(_args())
        assert rc == 0
        assert "reply:" not in capsys.readouterr().out  # no round happened

    def test_eof_exits_cleanly(self, monkeypatch, tmp_path, capsys):
        _wire_chat_env(monkeypatch, tmp_path, _chat_cfg(tmp_path), [])
        rc = run(_args())
        assert rc == 0
        assert "reply:" not in capsys.readouterr().out

    def test_round_cap_exits_after_max(self, monkeypatch, tmp_path, capsys):
        # cap at 2 successful rounds; 3rd input never read (check fires before input)
        monkeypatch.setattr(chat, "MAX_ROUNDS", 2)
        _wire_chat_env(monkeypatch, tmp_path, _chat_cfg(tmp_path), ["a", "b"])
        rc = run(_args())
        out = capsys.readouterr().out
        assert rc == 0
        assert out.count("reply:") == 2  # exactly 2 successful rounds
        assert "round limit" in out.lower()

    def test_generate_error_continues_session(self, monkeypatch, tmp_path, capsys):
        # first generate() raises CannLmStatusError; session survives, 2nd round ok
        _wire_chat_env(
            monkeypatch, tmp_path, _chat_cfg(tmp_path), ["bad", "good", "/quit"]
        )
        monkeypatch.setattr(_FakeEngine, "fail_first_n", 1)
        rc = run(_args())
        out, err = capsys.readouterr()
        assert rc == 0
        assert out.count("reply:") == 1  # only "good" produced a reply
        assert "FAILURE" in err or "ExecutorGenerate" in err  # error surfaced to stderr

    def test_thinking_indicator_runs_during_generate(
        self, monkeypatch, tmp_path, capsys
    ):
        # while generate() blocks, an animated 'thinking.' spinner must render to
        # stdout so the user sees the model is working (not a silent hang)
        _wire_chat_env(monkeypatch, tmp_path, _chat_cfg(tmp_path), ["hi", "/quit"])
        import cann_lm

        monkeypatch.setattr(cann_lm, "Engine", _SlowFakeEngine)
        rc = run(_args())
        out = capsys.readouterr().out
        assert rc == 0
        assert "thinking" in out.lower()  # spinner rendered
        assert out.lower().index("thinking") < out.index("reply:")  # before the reply


class TestThinkingIndicator:
    """_ThinkingIndicator: animated spinner cycling thinking. -> thinking.. ->
    thinking... in a daemon thread. stop() joins + clears the line. Proves the
    indicator advances (not a frozen 'thinking...')."""

    def test_cycles_through_multiple_frames(self, capsys):
        import time
        from cann_lm.chat import _ThinkingIndicator

        ind = _ThinkingIndicator(interval=0.04)
        ind.start()
        time.sleep(0.25)  # ~6 frames at 0.04s — robust margin
        ind.stop()
        out = capsys.readouterr().out
        frames = [f for f in out.split("\r") if f.startswith("thinking")]
        assert len(set(frames)) >= 2, (
            f"expected >=2 distinct thinking frames (animation), got {frames}"
        )

    def test_stop_clears_the_line(self, capsys):
        # after stop(), the spinner text must be overwritten (cleared) so the
        # reply starts at column 0, not appended after 'thinking...'
        from cann_lm.chat import _ThinkingIndicator

        ind = _ThinkingIndicator(interval=0.5)
        ind.start()
        ind.stop()
        out = capsys.readouterr().out
        # stop() writes a clear sequence: \r + spaces + \r (cursor back to col 0)
        assert out.endswith("\r")
