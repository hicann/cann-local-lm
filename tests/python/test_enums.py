"""Enum value tests — the IntEnums must match c_api.h macro/enum values
exactly so status codes and finish reasons cross the boundary correctly.

Sources in c_api.h:
  Status          : lines 38-43 (#define CANN_LM_ENGINE_*)
  FinishReason    : lines 53-58 (enum)
  RequestStatus   : lines 61-67 (enum)
"""

import pytest

from cann_lm._enums import FinishReason, RequestStatus, Status


class TestStatus:
    @pytest.mark.parametrize(
        "name,value",
        [
            ("SUCCESS", 0),
            ("FAILURE", 1),
            ("INVALID_ARGUMENT", 2),
            ("INVALID_STATE", 3),
            ("NOT_FOUND", 4),
            ("CANCELED", 5),
        ],
    )
    def test_values_match_header(self, name, value):
        assert int(getattr(Status, name)) == value

    def test_ok_true_for_success(self):
        assert Status.SUCCESS.ok() is True

    def test_ok_false_for_failures(self):
        for name in (
            "FAILURE",
            "INVALID_ARGUMENT",
            "INVALID_STATE",
            "NOT_FOUND",
            "CANCELED",
        ):
            assert getattr(Status, name).ok() is False


class TestFinishReason:
    @pytest.mark.parametrize(
        "name,value",
        [
            ("MAX_TOKENS", 0),
            ("STOP_TOKEN", 1),
            ("STOP_STRING", 2),
            ("CANCELED", 3),
            ("ERROR", 4),
        ],
    )
    def test_values_match_header(self, name, value):
        assert int(getattr(FinishReason, name)) == value


class TestRequestStatus:
    @pytest.mark.parametrize(
        "name,value",
        [
            ("QUEUED", 0),
            ("PREFILLING", 1),
            ("DECODING", 2),
            ("COMPLETED", 3),
            ("CANCELED", 4),
            ("FAILED", 5),
        ],
    )
    def test_values_match_header(self, name, value):
        assert int(getattr(RequestStatus, name)) == value
