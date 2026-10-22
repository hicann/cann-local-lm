"""Tests for _ffi._candidate_paths — .so path resolution.

Pure: only the candidate-path list is generated; no .so is actually loaded.
Covers the on-device failure: CANN_LM_LIB set to a `~/dir` path that is (a) not
expanduser'd and (b) a directory rather than the .so file itself.
"""
import os

from cann_lm._ffi import _candidate_paths


class TestCandidatePaths:
    def test_explicit_arg_passed_through(self):
        cs = _candidate_paths("/opt/lib/libcann_lm_engine.so")
        assert "/opt/lib/libcann_lm_engine.so" in cs

    def test_env_var_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("CANN_LM_LIB", "/opt/eng/libfoo.so")
        cs = _candidate_paths(None)
        assert "/opt/eng/libfoo.so" in cs

    def test_explicit_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("CANN_LM_LIB", "/from/env.so")
        cs = _candidate_paths("/from/arg.so")
        assert cs[0] == "/from/arg.so"      # explicit first, env second

    def test_tilde_is_expanded(self, monkeypatch):
        # the on-device bug: `~/x` must become <home>/x, not the literal "~/x"
        monkeypatch.setenv("CANN_LM_LIB", "~/mylib.so")
        cs = _candidate_paths(None)
        expected = os.path.expanduser("~/mylib.so")
        assert expected in cs
        assert not any(c.startswith("~") for c in cs)   # literal ~ never reaches candidates

    def test_explicit_tilde_is_expanded(self):
        cs = _candidate_paths("~/explicit.so")
        assert os.path.expanduser("~/explicit.so") in cs

    def test_directory_probes_so_inside(self, monkeypatch, tmp_path):
        # CANN_LM_LIB points at a directory → probe libcann_lm_engine.so in it
        monkeypatch.setenv("CANN_LM_LIB", str(tmp_path))
        cs = _candidate_paths(None)
        assert os.path.join(str(tmp_path), "libcann_lm_engine.so") in cs

    def test_directory_with_actual_so_file(self, monkeypatch, tmp_path):
        # real dir + real .so inside → candidate present
        so = tmp_path / "libcann_lm_engine.so"
        so.write_text("")
        monkeypatch.setenv("CANN_LM_LIB", str(tmp_path))
        cs = _candidate_paths(None)
        assert str(so) in cs

    def test_repo_root_fallbacks_always_present(self, monkeypatch):
        # without explicit/env, the repo deploy/build trees are still probed
        monkeypatch.delenv("CANN_LM_LIB", raising=False)
        cs = _candidate_paths(None)
        assert any("deploy_ohos" in c and c.endswith("libcann_lm_engine.so") for c in cs)
        assert any("build_ohos" in c for c in cs)

    def test_no_duplicates(self, monkeypatch, tmp_path):
        # dir containing libcann_lm_engine.so → listed once, not twice
        so = tmp_path / "libcann_lm_engine.so"
        so.write_text("")
        monkeypatch.setenv("CANN_LM_LIB", str(tmp_path))
        cs = _candidate_paths(None)
        assert cs.count(str(so)) == 1
