"""Exceptions for the CANN-LM Python bindings.

CannLmError        — base for all binding errors (incl. library-not-found).
CannLmStatusError  — raised by raise_for_status when a C function returns a
                     non-SUCCESS Status; carries the Status enum + the C
                     function name so the failure points back at the exact
                     boundary call.
raise_for_status   — faithful translation of a non-zero C Status into a
                     Python exception (Python cannot ignore a failed call).
"""

from ._enums import Status

__all__ = ["CannLmError", "CannLmStatusError", "raise_for_status"]


class CannLmError(Exception):
    """Base for all CANN-LM binding errors."""


class CannLmStatusError(CannLmError):
    """A C ABI call returned a non-SUCCESS Status."""

    def __init__(self, status, func_name, message=None, result=None):
        self.status = status if isinstance(status, Status) else Status(int(status))
        self.func_name = func_name
        self.message = message
        # Partial Result the C side filled on FAILURE (c_api.cpp fills Result
        # iff status in {SUCCESS, FAILURE}); None when Result was not filled.
        self.result = result
        detail = f": {message}" if message else ""
        super().__init__(
            f"{func_name} failed: {self.status.name}({int(self.status)}){detail}"
        )


def raise_for_status(status, func_name, message=None):
    """Raise CannLmStatusError if status != SUCCESS; no-op on SUCCESS.

    Accepts a Status enum or a raw int (1:1 with C status codes).
    """
    st = status if isinstance(status, Status) else Status(int(status))
    if not st.ok():
        raise CannLmStatusError(st, func_name, message)
