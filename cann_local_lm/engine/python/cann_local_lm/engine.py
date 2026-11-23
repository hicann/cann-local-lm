"""Pythonic wrapper over the CANN-LM C ABI.

Hosts (a) marshaling helpers turning Python inputs into the ctypes
arrays/buffers the C functions expect (FFI mechanics — no extra logic), and
(b) thin wrapper classes Engine/Context/Request/Result holding opaque handles,
forwarding each call 1:1 to the corresponding C function, and releasing
C-allocated memory on close.

Library load is LAZY: nothing here touches the .so at import time, so the
marshaling helpers are testable without a loadable engine library. The .so is
resolved only when an Engine is constructed (via cann_local_lm._ffi.get_library()).

Status contract (from c_api.cpp): Result is filled iff status == SUCCESS or
status == FAILURE. So on FAILURE the (partial) Result is still valid; we read
+ free it in either case and, on non-SUCCESS, attach it to the raised
CannLmStatusError so partial output is never lost. On other errors the Result
was zeroed (ResultDestroy is a safe no-op on a zeroed Result).
"""

import codecs
import ctypes as ct

from ._enums import FinishReason, RequestStatus, Status
from ._ffi import (
    CANN_LMEngine_Buffer,
    CANN_LMEngine_EngineOptions,
    CANN_LMEngine_ModelArtifact,
    CANN_LMEngine_RequestOptions,
    CANN_LMEngine_Result,
    CANN_LMEngine_SamplingParams,
    CANN_LMEngine_StreamCallback,
    get_library,
)
from .exceptions import CannLmStatusError, raise_for_status

__all__ = [
    "Engine",
    "Context",
    "Request",
    "Result",
    "build_request_options",
    "build_buffer",
    "build_model_artifact",
    "build_sampling_params",
    "build_engine_options",
]


# ────────────────────────── marshaling helpers ──────────────────────────
# Each (where it owns backing storage) returns (struct, keepalive_list). The
# caller MUST retain the keepalive for as long as the C side holds the pointers
# (a single synchronous call, or the lifetime of an async Request).


def build_request_options(max_tokens=0, stop_token_ids=None, stop_strings=None):
    """Build a CANN_LMEngine_RequestOptions from Python inputs.

    stop_token_ids : list[int] | None  -> (uint32* ptr, count); None/[] -> NULL.
    stop_strings   : list[str] | None   -> (char** ptr, count); None/[] -> NULL.
    """
    keepalive = []
    opts = CANN_LMEngine_RequestOptions()
    opts.max_tokens = int(max_tokens)

    if stop_token_ids:
        ids = [int(t) for t in stop_token_ids]
        arr = (ct.c_uint32 * len(ids))(*ids)
        keepalive.append(arr)
        opts.stop_token_ids = ct.cast(arr, ct.c_void_p).value
        opts.stop_token_ids_count = len(ids)

    if stop_strings:
        encoded = [s.encode("utf-8") for s in stop_strings]
        keepalive.extend(encoded)
        arr = (ct.c_char_p * len(encoded))(*encoded)
        keepalive.append(arr)
        opts.stop_strings = ct.cast(arr, ct.c_void_p).value
        opts.stop_strings_count = len(encoded)

    return opts, keepalive


def build_buffer(data):
    """Build a CANN_LMEngine_Buffer from bytes/bytearray/memoryview/None.

    None or empty -> NULL data, size 0 (absent per C contract: data==NULL).
    Otherwise -> (data ptr, size) with the backing bytearray kept alive.
    """
    if data is None:
        return CANN_LMEngine_Buffer(), []
    ba = bytearray(data)  # writable copy we own
    n = len(ba)
    if n == 0:
        return CANN_LMEngine_Buffer(), []
    arr = (ct.c_char * n).from_buffer(ba)
    keepalive = [ba, arr]
    buf = CANN_LMEngine_Buffer()
    buf.data = ct.cast(arr, ct.c_void_p).value
    buf.size = n
    return buf, keepalive


def _set_path_field(struct, field, path, keepalive):
    """Encode a str/bytes path into a c_void_p field; keep the bytes alive.
    str.encode -> CPython bytes carry a trailing NUL internally, so the pointer
    is a valid NUL-terminated C string as long as `b` is alive (in keepalive)."""
    b = path.encode("utf-8") if isinstance(path, str) else bytes(path)
    keepalive.append(b)
    setattr(struct, field, ct.cast(ct.c_char_p(b), ct.c_void_p).value)


def build_model_artifact(
    prefill_omc_path=None,
    prefill_omc_data=None,
    decode_omc_path=None,
    decode_omc_data=None,
    prefill_weight=None,
    decode_weight=None,
    meta_json_path=None,
    meta_json_data=None,
    is_enable_shared_weight=False,
):
    """Build a CANN_LMEngine_ModelArtifact (1:1 with c_api.h:87).

    Path args (str/bytes) -> const char* (NULL when None). Buffer args
    (bytes-like) -> CANN_LMEngine_Buffer (NULL/size 0 when None). Returns
    (artifact, keepalive). The C side stores buffer data as a *view* (no copy),
    so keepalive must outlive the model-load phase — keep it on the Engine.

    is_enable_shared_weight: enable prefill/decode share-weights binding
    (isEnableSharedWeight). False = decode keeps share_weights_model_id=0.
    """
    keepalive = []
    art = CANN_LMEngine_ModelArtifact()

    if prefill_omc_path is not None:
        _set_path_field(art, "prefill_omc_path", prefill_omc_path, keepalive)
    if decode_omc_path is not None:
        _set_path_field(art, "decode_omc_path", decode_omc_path, keepalive)
    if prefill_weight is not None:
        _set_path_field(art, "prefill_weight", prefill_weight, keepalive)
    if decode_weight is not None:
        _set_path_field(art, "decode_weight", decode_weight, keepalive)
    if meta_json_path is not None:
        _set_path_field(art, "meta_json_path", meta_json_path, keepalive)

    if prefill_omc_data is not None:
        art.prefill_omc_buf, ka = build_buffer(prefill_omc_data)
        keepalive.extend(ka)
    if decode_omc_data is not None:
        art.decode_omc_buf, ka = build_buffer(decode_omc_data)
        keepalive.extend(ka)
    if meta_json_data is not None:
        art.meta_json_buf, ka = build_buffer(meta_json_data)
        keepalive.extend(ka)

    art.isEnableSharedWeight = bool(is_enable_shared_weight)

    return art, keepalive


def build_sampling_params(
    temperature=1.0, top_k=0, top_p=1.0, seed=0, repetition_penalty=1.0
):
    """Build a CANN_LMEngine_SamplingParams with C++-matching defaults
    (temperature=1.0, top_p=1.0, repetition_penalty=1.0 — NOT the 0.0 a
    zero-initialised struct would give). Pass None for sampling_params in
    generate() to use the C NULL path (same C++ defaults)."""
    return CANN_LMEngine_SamplingParams(
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        seed=int(seed),
        repetition_penalty=float(repetition_penalty),
    )


def build_engine_options(
    default_chunk_size=0,
    max_concurrent_prefill=0,
    max_batch_size_override=0,
    tokenizer_type=None,
    tokenizer_path=None,
):
    """Build a CANN_LMEngine_EngineOptions. tokenizer_type/tokenizer_path
    are const char* (NULL when None -> C++ defaults)."""
    keepalive = []
    opts = CANN_LMEngine_EngineOptions()
    opts.default_chunk_size = int(default_chunk_size)
    opts.max_concurrent_prefill = int(max_concurrent_prefill)
    opts.max_batch_size_override = int(max_batch_size_override)
    if tokenizer_type is not None:
        _set_path_field(opts, "tokenizer_type", tokenizer_type, keepalive)
    if tokenizer_path is not None:
        _set_path_field(opts, "tokenizer_path", tokenizer_path, keepalive)
    return opts, keepalive


# ────────────────────────── Result wrapper ──────────────────────────


class Result:
    """Inference output. Python-owned copies; the C-allocated token_ids/text
    have already been freed (ResultDestroy) before this object is returned."""

    __slots__ = ("token_ids", "text", "finish_reason")

    def __init__(self, token_ids=(), text="", finish_reason=FinishReason.MAX_TOKENS):
        self.token_ids = list(token_ids)
        self.text = text
        self.finish_reason = FinishReason(finish_reason)

    def __repr__(self):
        return (
            f"Result(token_ids={self.token_ids!r}, "
            f"text={self.text!r}, finish_reason={self.finish_reason!r})"
        )


def _wrap_result(lib, c_result):
    """Copy CANN_LMEngine_Result out to a Python Result and free the C memory
    (ResultDestroy). Safe on a zeroed/empty Result (NULL ptrs -> empty)."""
    n = c_result.token_ids_count
    if n and c_result.token_ids:
        arr = ct.cast(c_result.token_ids, ct.POINTER(ct.c_uint32))
        ids = [int(arr[i]) for i in range(n)]
    else:
        ids = []
    # text is NUL-terminated (fill_c_result allocs size+1 with trailing NUL)
    if c_result.text:
        raw = ct.cast(c_result.text, ct.c_char_p).value
        text = raw.decode("utf-8", "replace") if raw is not None else ""
    else:
        text = ""
    fr = FinishReason(c_result.finish_reason)
    lib.CANN_LMEngine_ResultDestroy(ct.byref(c_result))
    return Result(ids, text, fr)


# ────────────────────────── Stream callback ──────────────────────────


def _make_callback(py_cb):
    """Wrap a Python callable (token_id: int, text: str) -> None into a
    CANN_LMEngine_StreamCallback. Returns (c_callback, keepalive_list). The
    keepalive MUST outlive the async request (callback fires from the pipeline
    internal thread). None -> (None, []).

    Text is decoded across token boundaries. A token containing only the
    beginning of a UTF-8 character emits empty text; a later token completes
    it. The final Result remains authoritative for an unfinished byte sequence.
    """
    if py_cb is None:
        return None, []

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _trampoline(token, text_ptr, text_len, _user_data):
        # Never let a Python exception propagate across the C boundary
        # (the callback runs on the pipeline thread).
        try:
            if text_ptr and text_len:
                raw = ct.string_at(text_ptr, text_len)
                text = decoder.decode(raw)
            else:
                text = ""
            py_cb(int(token), text)
        except Exception:  # noqa: BLE001 — boundary safety
            pass

    c_cb = CANN_LMEngine_StreamCallback(_trampoline)
    return c_cb, [c_cb]


# ────────────────────────── Engine ──────────────────────────


class Engine:
    """Wraps a CANN_LMEngine_Executor* opaque handle.

    artifact: None (no-model form, c_api.h:155), a ModelArtifact struct, or the
              (struct, keepalive) tuple returned by build_model_artifact().
    options:  None (NULL -> C++ defaults), an EngineOptions struct, or the
              (struct, keepalive) tuple from build_engine_options().
    lib_path: explicit .so path; else discovered via get_library().

    Generate flow (ADR-0002): create_context() -> set_context() ->
    generate()/generate_async() — requests belong to the Executor's active
    context; generating with no active context raises INVALID_STATE.
    """

    def __init__(self, artifact=None, options=None, lib_path=None):
        self._lib = get_library(lib_path)
        self._closed = False
        self._handle = None

        # ── artifact (must be non-NULL pointer; no-model form = empty struct)
        art_keepalive = []
        if artifact is None:
            art = CANN_LMEngine_ModelArtifact()
        elif isinstance(artifact, CANN_LMEngine_ModelArtifact):
            art = artifact
        elif isinstance(artifact, tuple) and len(artifact) == 2:
            art, art_keepalive = artifact
        else:
            raise TypeError(
                "artifact must be None, a ModelArtifact, or (struct, keepalive)"
            )
        self._art = art
        # buffer data is a *view* on the C side — keep backing storage alive
        # for the engine's lifetime (model load may be deferred).
        self._art_keepalive = art_keepalive

        # ── options (NULL = C++ defaults)
        opts_keepalive = []
        if options is None:
            opts = None
        elif isinstance(options, CANN_LMEngine_EngineOptions):
            opts = options
        elif isinstance(options, tuple) and len(options) == 2:
            opts, opts_keepalive = options
        else:
            raise TypeError(
                "options must be None, an EngineOptions, or (struct, keepalive)"
            )
        self._opts = opts
        self._opts_keepalive = opts_keepalive

        # ── create
        out = ct.c_void_p()
        opts_ptr = ct.byref(self._opts) if self._opts is not None else None
        status = self._lib.CANN_LMEngine_ExecutorCreate(
            ct.byref(self._art), opts_ptr, ct.byref(out)
        )
        if status != Status.SUCCESS:
            # c_api.h:173 — *out stays NULL on init failure. The coarse Status
            # code hides the specific cause (which file/field); the C++ side
            # logs it (hilog tag CANN_LM_OPEN on OHOS, stderr elsewhere).
            msg = None
            if status == Status.INVALID_ARGUMENT:
                msg = (
                    "a model/tokenizer/weight path in the config likely "
                    "doesn't exist on device, or a meta_json field is "
                    "invalid — see the C++ logs (hilog tag CANN_LM_OPEN "
                    "on HarmonyOS, stderr elsewhere) for the exact file/field"
                )
            raise CannLmStatusError(status, "ExecutorCreate", message=msg)
        self._handle = out.value

    @property
    def handle(self):
        """The opaque CANN_LMEngine_Executor* (int address), or None if closed."""
        return self._handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — finalizer must never raise
            pass

    def close(self):
        """Destroy the underlying executor (idempotent; safe after close)."""
        if not self._closed and self._handle:
            self._lib.CANN_LMEngine_ExecutorDestroy(self._handle)
            self._closed = True
            self._handle = None

    # ── context management ──

    def create_context(self):
        """Create a conversation context. Returns a Context (holds the id)."""
        out = ct.c_uint64(0)
        status = self._lib.CANN_LMEngine_ContextCreate(self._handle, ct.byref(out))
        raise_for_status(status, "ContextCreate")
        return Context(self, int(out.value))

    def set_context(self, context):
        """Activate `context` (a Context or raw id) as the Executor's active
        context (ADR-0002). generate()/generate_async() requests then belong
        to it until switched or destroyed. Unregistered id -> NOT_FOUND;
        in-flight requests -> INVALID_STATE; target == active -> idempotent
        success (c_api.h:177)."""
        ctx_id = context.id if isinstance(context, Context) else int(context)
        raise_for_status(
            self._lib.CANN_LMEngine_ExecutorSetContext(self._handle, ctx_id),
            "ExecutorSetContext",
        )

    def destroy_context(self, context_id):
        raise_for_status(
            self._lib.CANN_LMEngine_ContextDestroy(self._handle, int(context_id)),
            "ContextDestroy",
        )

    def cancel_context(self, context_id):
        raise_for_status(
            self._lib.CANN_LMEngine_ContextCancel(self._handle, int(context_id)),
            "ContextCancel",
        )

    # ── inference ──

    def generate(
        self,
        prompt,
        sampling_params=None,
        max_tokens=0,
        stop_token_ids=None,
        stop_strings=None,
    ):
        """Synchronous generate on the Executor's ACTIVE context (ADR-0002:
        set via set_context(); no active context -> CannLmStatusError
        INVALID_STATE). Returns Result on SUCCESS; raises CannLmStatusError
        (with .result) on non-SUCCESS (FAILURE still fills a partial Result,
        attached to the exception).

        sampling_params: None (NULL -> C++ defaults) or a SamplingParams struct
                         (use build_sampling_params() — a zero-init struct gives
                         0.0 values, NOT the 1.0 C++ defaults).
        """
        prompt_b = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
        params_ptr = (
            ct.byref(sampling_params)
            if isinstance(sampling_params, CANN_LMEngine_SamplingParams)
            else None
        )
        req_opts, req_keepalive = build_request_options(
            max_tokens, stop_token_ids, stop_strings
        )
        c_result = CANN_LMEngine_Result()
        status = self._lib.CANN_LMEngine_ExecutorGenerate(
            self._handle,
            prompt_b,
            params_ptr,
            ct.byref(req_opts),
            ct.byref(c_result),
        )
        result = _wrap_result(self._lib, c_result)
        if status != Status.SUCCESS:
            raise CannLmStatusError(status, "ExecutorGenerate", result=result)
        return result

    def generate_async(
        self,
        prompt,
        sampling_params=None,
        max_tokens=0,
        stop_token_ids=None,
        stop_strings=None,
        callback=None,
    ):
        """Asynchronous generate on the Executor's ACTIVE context (ADR-0002 —
        see generate()). Returns a Request. `callback(token_id, text)` fires
        per decoded token on the pipeline thread (must not block or call
        Engine APIs). callback=None -> no streaming (poll Request instead)."""
        prompt_b = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
        params_ptr = (
            ct.byref(sampling_params)
            if isinstance(sampling_params, CANN_LMEngine_SamplingParams)
            else None
        )
        req_opts, req_keepalive = build_request_options(
            max_tokens, stop_token_ids, stop_strings
        )
        c_cb, cb_keepalive = _make_callback(callback)
        out = ct.c_void_p()
        status = self._lib.CANN_LMEngine_ExecutorGenerateAsync(
            self._handle,
            prompt_b,
            params_ptr,
            ct.byref(req_opts),
            ct.byref(out),
            c_cb,
            None,
        )
        keepalive = req_keepalive + cb_keepalive
        if status != Status.SUCCESS:
            raise CannLmStatusError(status, "ExecutorGenerateAsync")
        return Request(self, out.value, keepalive)


# ────────────────────────── Context ──────────────────────────


class Context:
    """A conversation context (CANN_LMEngine_ContextId value). The id is owned
    by the Engine's context table; destroy() removes it there."""

    def __init__(self, engine, context_id):
        self._engine = engine
        self._id = int(context_id)

    @property
    def id(self):
        return self._id

    def cancel(self):
        self._engine.cancel_context(self._id)

    def activate(self):
        """Make this the Engine's active context (Engine.set_context,
        ADR-0002) — subsequent generate()/generate_async() calls belong to
        it until switched or destroyed."""
        self._engine.set_context(self._id)

    def destroy(self):
        if self._id is not None:
            self._engine.destroy_context(self._id)
            self._id = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


# ────────────────────────── Request ──────────────────────────


class Request:
    """Wraps a CANN_LMEngine_Request* opaque handle (async generate)."""

    def __init__(self, engine, handle, keepalive=None):
        self._engine = engine
        self._handle = handle
        self._keepalive = list(keepalive) if keepalive else []

    @property
    def status(self):
        """Lifecycle status (RequestStatus enum)."""
        return RequestStatus(
            self._engine._lib.CANN_LMEngine_RequestGetStatus(self._handle)
        )

    @property
    def is_done(self):
        return bool(self._engine._lib.CANN_LMEngine_RequestIsDone(self._handle))

    def cancel(self):
        raise_for_status(
            self._engine._lib.CANN_LMEngine_RequestCancel(self._handle), "RequestCancel"
        )

    def wait(self):
        """Block until done; returns Result (or raises CannLmStatusError with
        .result on non-SUCCESS)."""
        c_result = CANN_LMEngine_Result()
        status = self._engine._lib.CANN_LMEngine_RequestWait(
            self._handle, ct.byref(c_result)
        )
        result = _wrap_result(self._engine._lib, c_result)
        if status != Status.SUCCESS:
            raise CannLmStatusError(status, "RequestWait", result=result)
        return result

    def try_get_result(self):
        """Non-blocking: returns Result if completed, else raises
        CannLmStatusError (INVALID_STATE when not yet done)."""
        c_result = CANN_LMEngine_Result()
        status = self._engine._lib.CANN_LMEngine_RequestTryGetResult(
            self._handle, ct.byref(c_result)
        )
        result = _wrap_result(self._engine._lib, c_result)
        if status != Status.SUCCESS:
            raise CannLmStatusError(status, "RequestTryGetResult", result=result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — finalizer must never raise
            pass

    def close(self):
        """Destroy the underlying request handle (idempotent)."""
        if self._handle:
            self._engine._lib.CANN_LMEngine_RequestDestroy(self._handle)
            self._handle = None
