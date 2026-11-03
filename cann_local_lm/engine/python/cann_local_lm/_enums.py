"""IntEnums mirroring c_api.h status/enum values 1:1.

Status          — c_api.h:38-43 (the 6 #define status codes; 1:1 with
                  C++ infra/status.h).
FinishReason    — c_api.h:53-58.
RequestStatus   — c_api.h:61-67.
"""

from enum import IntEnum

__all__ = ["Status", "FinishReason", "RequestStatus"]


class Status(IntEnum):
    """Unified status code. int values 1:1 with C++ cann_local_lm::engine::Status."""

    SUCCESS = 0
    FAILURE = 1
    INVALID_ARGUMENT = 2
    INVALID_STATE = 3
    NOT_FOUND = 4
    CANCELED = 5

    def ok(self) -> bool:
        """True iff SUCCESS (mirrors C++ Status::ok())."""
        return self == Status.SUCCESS


class FinishReason(IntEnum):
    """Result completion semantics. c_api.h:53-58."""

    MAX_TOKENS = 0
    STOP_TOKEN = 1
    STOP_STRING = 2
    CANCELED = 3
    ERROR = 4


class RequestStatus(IntEnum):
    """Async request lifecycle. c_api.h:61-67."""

    QUEUED = 0
    PREFILLING = 1
    DECODING = 2
    COMPLETED = 3
    CANCELED = 4
    FAILED = 5
