"""Tests for the cann_local_lm.generate CLI's pure logic + arg wiring + cli dispatch.

The actual generate call needs the engine .so (run on device via
`cann_local_lm generate --config ... --prompt ...` or the standalone `cann_local_lm.generate`).
Here we test resolve_request + add_args + the cann_local_lm subcommand dispatch —
pure, no .so.
"""

import argparse
import os

from bbit import load_config
from cann_local_lm.generate import add_args, resolve_request, run
from cann_local_lm.config import print_resolved_paths

# bbit 测试用例随 9a4640c 布局重构从 tests/bbit/ 迁至 examples/bbit/;后续加 test_engine 层区分 engine 用例。
BBIT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "test_engine", "bbit"
    )
)
TESTCASE = os.path.join(BBIT_DIR, "testcase.json")


def _cfg():
    return load_config(TESTCASE)


class TestResolveRequest:
    def test_prompt_override(self):
        prompt, mt, stop = resolve_request(_cfg(), prompt="hello world")
        assert prompt == "hello world"
        s0 = _cfg()["sentences"][0]
        assert mt == s0["max_tokens"]  # not overridden
        assert stop == s0["stop_strings"]  # not overridden

    def test_prompt_defaults_to_config(self):
        prompt, _, _ = resolve_request(_cfg())
        assert prompt == _cfg()["sentences"][0]["prompt"]  # "你好"

    def test_max_tokens_override(self):
        _, mt, _ = resolve_request(_cfg(), max_tokens=999)
        assert mt == 999

    def test_stop_override(self):
        _, _, stop = resolve_request(_cfg(), stop_strings=["END", "STOP"])
        assert stop == ["END", "STOP"]

    def test_all_defaults_match_config(self):
        s0 = _cfg()["sentences"][0]
        prompt, mt, stop = resolve_request(_cfg())
        assert (prompt, mt, stop) == (
            s0["prompt"],
            s0["max_tokens"],
            s0["stop_strings"],
        )


class TestAddArgs:
    """generate.add_args wires the --config/--prompt/--max-tokens/--stop/--lib
    flags onto any parser (shared by the standalone command + the subcommand)."""

    def _parse(self, *argv):
        p = argparse.ArgumentParser()
        add_args(p)
        return p.parse_args(list(argv))

    def test_required_and_optional(self):
        a = self._parse("--config", "tc.json", "--prompt", "hi")
        assert a.config == "tc.json"
        assert a.prompt == "hi"
        assert a.max_tokens is None  # optional, not given
        assert a.stop_strings is None  # optional, not given
        assert a.lib is None

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

    def test_missing_config_errors(self):
        # --config is required → argparse raises SystemExit
        p = argparse.ArgumentParser()
        add_args(p)
        import pytest

        with pytest.raises(SystemExit):
            p.parse_args([])


class TestGateSurfacesLoadError:
    """When the .so exists but won't load (missing runtime deps / wrong arch),
    the gate must print the real OSError reason — not a generic 'no loadable .so'
    that hides the fix."""

    def test_missing_dep_reason_surfaced(self, monkeypatch, capsys):
        import cann_local_lm

        # pretend load_config succeeds (cfg content irrelevant — gate fires next)
        monkeypatch.setattr(
            "cann_local_lm.generate.load_config", lambda c: {"sentences": [{}]}
        )

        # pretend the .so is found but its deps are missing (the device failure)
        def boom(lib_path=None):
            raise cann_local_lm.CannLmError(
                "libcann_lm_engine.so could not be loaded.\n"
                "Found but CDLL failed (likely missing runtime deps or wrong arch):\n"
                "  /storage/x/libcann_lm_engine.so: libhiai_ascendcl.so: cannot open shared object file"
            )

        monkeypatch.setattr(cann_local_lm, "get_library", boom)
        args = argparse.Namespace(
            config="ignored.json",
            prompt="hi",
            max_tokens=None,
            stop_strings=None,
            lib=None,
        )
        rc = run(args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "libhiai_ascendcl.so" in err  # real reason surfaced, not swallowed
        assert "cannot load" in err


class TestShowPaths:
    """`cann_local_lm generate --config x --show-paths` prints each resolved config
    path + its on-disk existence, then exits (no .so, no Engine). For the
    'I set the paths but the engine says meta_json unreadable' class of mismatch."""

    def _cfg(self, tmp_path):
        f = tmp_path / "tok.json"  # exists (file)
        f.write_text("")
        d = tmp_path / "weights"  # exists (dir)
        d.mkdir()
        return {
            "tokenizer_type": "BPE",
            "tokenizer_path": str(f),  # file ✓
            "prefill_omc_path": str(d),  # dir ✓
            "decode_omc_path": str(tmp_path / "decode.omc"),  # MISSING
            "meta_json_path": str(tmp_path / "model.json"),  # MISSING
            "prefill_weight": str(tmp_path / "w_prefill"),  # MISSING
            "decode_weight": str(tmp_path / "w_decode"),  # MISSING
            "sentences": [{}],
        }

    def test_print_resolved_paths_marks_existing_and_missing(self, tmp_path, capsys):
        cfg = self._cfg(tmp_path)
        missing = print_resolved_paths(cfg)
        out = capsys.readouterr().out
        assert "tokenizer_path" in out
        assert "meta_json_path" in out
        assert "prefill_weight" in out
        assert "file" in out  # the existing file marker
        assert "dir" in out  # the existing dir marker
        assert "MISSING" in out  # the missing marker
        assert missing == 4  # decode_omc, meta_json, prefill_weight, decode_weight

    def test_show_paths_flag_exits_before_so_gate(self, monkeypatch, tmp_path, capsys):
        # --show-paths must NOT touch the .so: monkeypatch get_library to explode
        # if it's ever called; if the flag works, run() returns before the gate.
        import cann_local_lm

        def boom(*a, **k):
            raise AssertionError("--show-paths must not call get_library")

        monkeypatch.setattr(cann_local_lm, "get_library", boom)
        cfg = self._cfg(tmp_path)
        monkeypatch.setattr("cann_local_lm.generate.load_config", lambda c: cfg)
        args = argparse.Namespace(
            config="ignored.json",
            prompt=None,
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

    def test_show_paths_all_present_returns_zero(self, monkeypatch, tmp_path, capsys):
        # all 6 paths exist → exit 0
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
        monkeypatch.setattr("cann_local_lm.generate.load_config", lambda c: cfg)
        args = argparse.Namespace(
            config="x",
            prompt=None,
            max_tokens=None,
            stop_strings=None,
            lib=None,
            show_paths=True,
        )
        rc = run(args)
        assert rc == 0
