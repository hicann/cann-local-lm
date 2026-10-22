"""Tests for the pure helpers in bbit.py (the Python mirror of tests/bbit/bbit.cpp).

The C-call flow (--check NULL defense, --config lifecycle) needs the engine .so
and is exercised by running `python bbit.py` on a device. These tests cover the
logic that runs WITHOUT the .so: the module feature-check, the config loader
(testcase.json split-weight contract), the generate-status acceptance set, and
the degraded-run decision.
"""
import os
import types

import cann_lm
from cann_lm import Status

from bbit import (
    file_exists,
    is_degraded_run,
    is_generate_status_ok,
    load_config,
    supports_bbit_flow,
)

BBIT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "bbit"))
TESTCASE = os.path.join(BBIT_DIR, "testcase.json")
TESTCASE_HOST = os.path.join(BBIT_DIR, "testcase_host.json")


# ────────────────────────── supports_bbit_flow (feature-check) ──────────────────────────

class TestSupportsBbitFlow:
    def test_cann_lm_supports_bbit_flow(self):
        assert supports_bbit_flow(cann_lm) is True

    def test_module_missing_a_name_is_unsupported(self):
        stub = types.SimpleNamespace(**{n: object() for n in (
            "Engine", "Context", "Request", "Result",
            "build_model_artifact", "build_engine_options",
            "build_request_options", "build_sampling_params",
            "is_library_available", "get_library",
            "Status", "RequestStatus", "CannLmStatusError", "FinishReason",
        )})
        del stub.Engine
        assert supports_bbit_flow(stub) is False


# ────────────────────────── load_config (split-weight contract) ──────────────────────────

class TestLoadConfig:
    def test_parses_real_testcase(self):
        cfg = load_config(TESTCASE)
        assert cfg is not None
        assert cfg["tokenizer_type"] == "BPE"
        assert cfg["prefill_omc_path"].endswith("model.omc")
        assert cfg["decode_omc_path"].endswith("model.omc")
        assert cfg["meta_json_path"].endswith("model.json")
        # split weight fields → prefill_weight / decode_weight
        assert cfg["prefill_weight"].endswith("SubGraph_0.weight")
        assert cfg["decode_weight"].endswith("SubGraph_0.weight")
        s = cfg["sentences"][0]
        assert s["prompt"] == "你好"
        assert s["max_tokens"] == 50
        assert s["top_k"] == 1
        assert s["seed"] == 99
        assert s["stop_strings"] == ["[unused10]"]

    def test_parses_host_testcase(self):
        cfg = load_config(TESTCASE_HOST)
        assert cfg is not None
        assert cfg["prefill_omc_path"] == "stub.omc"
        assert cfg["decode_omc_path"] == "stub.omc"
        assert cfg["prefill_weight"] == "stub_weight"
        assert cfg["decode_weight"] == "stub_weight"
        assert cfg["sentences"][0]["prompt"] == "hello"

    def test_missing_file_returns_none(self, tmp_path):
        assert load_config(str(tmp_path / "nope.json")) is None

    def test_missing_weight_field_returns_none(self, tmp_path):
        # split weight fields required; omitting one → None
        p = tmp_path / "bad.json"
        p.write_text(
            '{"tokenizer_type":"BPE","tokenizer_path":"t",'
            '"model_path_prefill":"a","model_path_decode":"b",'
            '"model_meta_config":"c",'
            '"model_weight_dir_prefill":"w",'
            '"sentences":[{"prompt":"x","maxGenTokens":1,"topK":1,"topP":1.0,'
            '"temperature":1.0,"seed":1,"repetitionPenalty":1.0,"stopSeq":[]}]}',
            encoding="utf-8")  # missing model_weight_dir_decode
        assert load_config(str(p)) is None

    def test_empty_sentences_returns_none(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(
            '{"tokenizer_type":"BPE","tokenizer_path":"t","model_path_prefill":"a",'
            '"model_path_decode":"b","model_meta_config":"c",'
            '"model_weight_dir_prefill":"w","model_weight_dir_decode":"w",'
            '"sentences":[]}', encoding="utf-8")
        assert load_config(str(p)) is None


# ────────────────────────── generate-status acceptance ──────────────────────────

class TestGenerateStatusOk:
    def test_success_failure_invalid_state_all_pass(self):
        assert is_generate_status_ok(Status.SUCCESS) is True
        assert is_generate_status_ok(Status.FAILURE) is True
        assert is_generate_status_ok(Status.INVALID_STATE) is True

    def test_other_statuses_fail(self):
        assert is_generate_status_ok(Status.INVALID_ARGUMENT) is False
        assert is_generate_status_ok(Status.NOT_FOUND) is False
        assert is_generate_status_ok(Status.CANCELED) is False


# ────────────────────────── file_exists / is_degraded_run ──────────────────────────

class TestFileExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "x"
        f.write_text("data", encoding="utf-8")
        assert file_exists(str(f)) is True

    def test_nonexistent(self):
        assert file_exists("/no/such/path/here") is False

    def test_empty_path(self):
        assert file_exists("") is False


class TestIsDegradedRun:
    def test_both_absent_is_degraded(self):
        assert is_degraded_run("/no/such/prefill", "/no/such/decode") is True

    def test_both_present_is_not_degraded(self, tmp_path):
        a = tmp_path / "p.omc"; a.write_text("", encoding="utf-8")
        b = tmp_path / "d.omc"; b.write_text("", encoding="utf-8")
        assert is_degraded_run(str(a), str(b)) is False

    def test_empty_paths_not_degraded(self):
        # not configured → not a degraded run (full-flow path, no-model form)
        assert is_degraded_run("", "") is False
