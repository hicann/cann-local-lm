"""CANN-LM Python bindings — ctypes passthrough over the C engine ABI.

Public surface (mirrors cann_lm/engine/api/c/c_api.h 1:1):
  Engine / Context / Request / Result   — handle wrappers (context-managed)
  build_model_artifact / build_engine_options / build_sampling_params /
  build_request_options / build_buffer  — marshaling helpers
  Status / FinishReason / RequestStatus — IntEnums (1:1 with c_api.h values)
  CannLmError / CannLmStatusError      — status -> exception
  get_library / is_library_available   — .so resolution

No runtime dependencies beyond the Python stdlib (ctypes). The engine .so is
loaded lazily on first Engine construction (or via get_library); point it at a
libcann_lm_engine.so whose architecture matches the running Python, e.g.:

    import cann_lm
    with cann_lm.Engine() as eng:        # no-model form (c_api.h:155)
        ctx = eng.create_context()
        r = eng.generate("hello", ctx)   # Result on SUCCESS; CannLmStatusError
                                        # (with .result) on FAILURE
"""
from ._enums import FinishReason, RequestStatus, Status
from ._ffi import get_library, is_library_available
from .engine import (
    Context,
    Engine,
    Request,
    Result,
    build_buffer,
    build_engine_options,
    build_model_artifact,
    build_request_options,
    build_sampling_params,
)
from .exceptions import CannLmError, CannLmStatusError, raise_for_status

__all__ = [
    # wrappers
    "Engine", "Context", "Request", "Result",
    # marshaling helpers
    "build_model_artifact", "build_engine_options", "build_sampling_params",
    "build_request_options", "build_buffer",
    # enums
    "Status", "FinishReason", "RequestStatus",
    # errors
    "CannLmError", "CannLmStatusError", "raise_for_status",
    # library
    "get_library", "is_library_available",
]
