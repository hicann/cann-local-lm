"""Exception + status->exception tests.

raise_for_status is the faithful translation of "the C call returned a non-zero
status": Python cannot silently ignore failure, so a non-SUCCESS status must
raise CannLmStatusError carrying the code + the C function name (so the error
points back at the exact boundary call that failed).
"""

import pytest

from cann_lm._enums import Status
from cann_lm.exceptions import CannLmError, CannLmStatusError, raise_for_status


class TestRaiseForStatus:
    def test_success_does_not_raise(self):
        # no exception
        raise_for_status(Status.SUCCESS, "ExecutorCreate")
        raise_for_status(0, "ContextCreate")  # raw int 0 also accepted

    @pytest.mark.parametrize(
        "status",
        [
            Status.FAILURE,
            Status.INVALID_ARGUMENT,
            Status.INVALID_STATE,
            Status.NOT_FOUND,
            Status.CANCELED,
        ],
    )
    def test_non_success_raises(self, status):
        with pytest.raises(CannLmStatusError) as exc:
            raise_for_status(status, "ExecutorGenerate")
        assert exc.value.status == status

    def test_raw_int_status_accepted(self):
        with pytest.raises(CannLmStatusError) as exc:
            raise_for_status(2, "ExecutorGenerate")  # INVALID_ARGUMENT
        assert exc.value.status == Status.INVALID_ARGUMENT

    def test_error_carries_func_name(self):
        with pytest.raises(CannLmStatusError) as exc:
            raise_for_status(Status.NOT_FOUND, "ExecutorGenerate")
        assert exc.value.func_name == "ExecutorGenerate"

    def test_strerror_in_message(self):
        with pytest.raises(CannLmStatusError) as exc:
            raise_for_status(Status.FAILURE, "ExecutorCreate")
        msg = str(exc.value)
        assert "ExecutorCreate" in msg
        assert "FAILURE" in msg


class TestExceptionHierarchy:
    def test_status_error_is_cann_lm_error(self):
        assert issubclass(CannLmStatusError, CannLmError)
