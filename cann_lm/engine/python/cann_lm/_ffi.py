"""Raw ctypes passthrough layer — 1:1 mirror of cann_lm/engine/api/c/c_api.h.

This module is the literal "透传": ctypes Structure definitions + (later) library
load + function prototypes matching the C header field-for-field. No Python
semantics live here; lifetime/encoding/status mapping belong to engine.py.

Pointer fields use c_void_p (not c_char_p) deliberately: c_char_p does not keep
the backing bytes alive, which is an FFI footgun. Lifetime is managed in the
wrapper layer. Layout (sizeof/offset) is identical either way on LP64.
"""

import ctypes as ct
import os

from .exceptions import CannLmError

__all__ = [
    "CANN_LMEngine_Buffer",
    "CANN_LMEngine_ModelArtifact",
    "CANN_LMEngine_EngineOptions",
    "CANN_LMEngine_SamplingParams",
    "CANN_LMEngine_RequestOptions",
    "CANN_LMEngine_Result",
    "CANN_LMEngine_StreamCallback",
    "get_library",
    "is_library_available",
]


# ────────────────────────── Stream Callback Type ──────────────────────────
# void (*)(TokenId token, const char* text, size_t text_len, void* user_data).
# text may NOT be NUL-terminated, so the wrapper reads text_len bytes — hence
# c_void_p (not c_char_p) for the text param.
CANN_LMEngine_StreamCallback = ct.CFUNCTYPE(
    None, ct.c_uint32, ct.c_void_p, ct.c_size_t, ct.c_void_p
)


# ────────────────────────── POD Structs (mirror c_api.h §POD Structs) ──────────────


class CANN_LMEngine_Buffer(ct.Structure):
    """{ const void* data; size_t size; } — c_api.h:78."""

    _fields_ = [
        ("data", ct.c_void_p),
        ("size", ct.c_size_t),
    ]


class CANN_LMEngine_ModelArtifact(ct.Structure):
    """SR20260903001 merged artifact: prefill/decode dual graph + dual weight
    paths + meta_json, dual-mode path(NULL=absent)/buffer(data==NULL=absent),
    buffer-takes-precedence. c_api.h:87."""

    _fields_ = [
        ("prefill_omc_path", ct.c_void_p),
        ("prefill_omc_buf", CANN_LMEngine_Buffer),
        ("decode_omc_path", ct.c_void_p),
        ("decode_omc_buf", CANN_LMEngine_Buffer),
        ("prefill_weight", ct.c_void_p),
        ("decode_weight", ct.c_void_p),
        ("meta_json_path", ct.c_void_p),
        ("meta_json_buf", CANN_LMEngine_Buffer),
        ("isEnableSharedWeight", ct.c_bool),  # tail-appended, offset 88 (c_api.h)
    ]


class CANN_LMEngine_EngineOptions(ct.Structure):
    """{ int default_chunk_size; int max_concurrent_prefill;
    int max_batch_size_override; const char* tokenizer_type;
    const char* tokenizer_path; } — c_api.h:98."""

    _fields_ = [
        ("default_chunk_size", ct.c_int),
        ("max_concurrent_prefill", ct.c_int),
        ("max_batch_size_override", ct.c_int),
        ("tokenizer_type", ct.c_void_p),
        ("tokenizer_path", ct.c_void_p),
    ]


class CANN_LMEngine_SamplingParams(ct.Structure):
    """Sampling knobs (mirror C++ SamplingParams). c_api.h:113."""

    _fields_ = [
        ("temperature", ct.c_float),
        ("top_k", ct.c_int),
        ("top_p", ct.c_float),
        ("seed", ct.c_uint64),
        ("repetition_penalty", ct.c_float),
    ]


class CANN_LMEngine_RequestOptions(ct.Structure):
    """Request-level termination (mirror C++ RequestOptions). c_api.h:123."""

    _fields_ = [
        ("max_tokens", ct.c_int),
        ("stop_token_ids", ct.c_void_p),  # const CANN_LMEngine_TokenId*
        ("stop_token_ids_count", ct.c_size_t),
        ("stop_strings", ct.c_void_p),  # const char* const*
        ("stop_strings_count", ct.c_size_t),
    ]


class CANN_LMEngine_Result(ct.Structure):
    """{ TokenId* token_ids; size_t token_ids_count; char* text;
    FinishReason finish_reason; } — library-malloc'd, caller frees via
    ResultDestroy. c_api.h:131."""

    _fields_ = [
        ("token_ids", ct.c_void_p),  # CANN_LMEngine_TokenId* (uint32*)
        ("token_ids_count", ct.c_size_t),
        ("text", ct.c_void_p),  # char*
        ("finish_reason", ct.c_int),  # CANN_LMEngine_FinishReason (enum=int)
    ]


# ────────────────────────── Library Loading ──────────────────────────
# LAZY: nothing loads at import time. get_library() resolves + caches on first
# call (e.g. when an Engine is constructed). Discovery order:
#   explicit `path` arg  ->  $CANN_LM_LIB  ->  repo-relative candidates:
#     deploy_ohos/libcann_lm_engine.so   (OHOS device deployment)
#     build_ohos/libcann_lm_engine.so    (OHOS cross-build output)
#     model_final/libcann_lm_engine.so    (signed deployment artifacts)
# The package is arch-agnostic; the caller must point at a .so whose
# architecture matches the running Python.

_lib = None  # cached CDLL after first successful load


def _repo_root():
    # _ffi.py lives at <repo>/cann_lm/engine/python/cann_lm/_ffi.py -> 5 dirnames up.
    # (C2: python/ → engine/python/ 多一层, 故 5 而非 4)
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )


def _candidate_paths(explicit, *, strict_explicit=False):
    """Resolve candidate .so paths. Leading ~ is expanded; if a candidate
    (explicit --lib or CANN_LM_LIB) is a directory, the conventional .so names
    inside it are also probed. Order: explicit > CANN_LM_LIB env > repo trees.

    strict_explicit: 当调用者显式传了 path 时置 True——只试该 path(目录则展开
    其 .so),失败即抛,不回退 CANN_LM_LIB env / repo 树。回退会让显式坏路径
    意外加载到别的 .so(违背 "用户明确指了路径" 的语义)。"""
    cands = []
    sources = (
        (explicit,) if strict_explicit else (explicit, os.environ.get("CANN_LM_LIB"))
    )
    for raw in sources:
        if not raw:
            continue
        p = os.path.expanduser(raw)
        if os.path.isdir(p):
            cands.append(os.path.join(p, "libcann_lm_engine.so"))
            cands.append(os.path.join(p, "libCannLmEngine.so"))
            try:
                for f in sorted(os.listdir(p)):
                    if f.endswith(".so"):
                        cands.append(os.path.join(p, f))
            except OSError:
                pass
        else:
            cands.append(p)
    if not strict_explicit:
        # package-internal: .so bundled in whl (cann_lm/lib/libcann_lm_engine.so)
        bundled = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lib", "libcann_lm_engine.so"
        )
        cands.append(bundled)
        # repo-relative candidates (dev mode)
        root = _repo_root()
        for sub in (
            "deploy_ohos/libcann_lm_engine.so",
            "build_ohos/libcann_lm_engine.so",
            "model_final/libcann_lm_engine.so",
        ):
            cands.append(os.path.join(root, sub))
    # dedup, preserve order
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _configure_prototypes(lib):
    """Set argtypes/restype on all 15 C functions — 1:1 with c_api.h
    (incl. the 2026-10-21 ADR-0002 changelog: ExecutorSetContext added,
    context_id removed from ExecutorGenerate/GenerateAsync)."""
    P = ct.POINTER
    MA = CANN_LMEngine_ModelArtifact
    EO = CANN_LMEngine_EngineOptions
    SP = CANN_LMEngine_SamplingParams
    RO = CANN_LMEngine_RequestOptions
    RS = CANN_LMEngine_Result

    # Executor lifecycle
    lib.CANN_LMEngine_ExecutorCreate.argtypes = [P(MA), P(EO), P(ct.c_void_p)]
    lib.CANN_LMEngine_ExecutorCreate.restype = ct.c_int
    lib.CANN_LMEngine_ExecutorDestroy.argtypes = [ct.c_void_p]
    lib.CANN_LMEngine_ExecutorDestroy.restype = None

    # Context management
    lib.CANN_LMEngine_ContextCreate.argtypes = [ct.c_void_p, P(ct.c_uint64)]
    lib.CANN_LMEngine_ContextCreate.restype = ct.c_int
    lib.CANN_LMEngine_ContextDestroy.argtypes = [ct.c_void_p, ct.c_uint64]
    lib.CANN_LMEngine_ContextDestroy.restype = ct.c_int
    # ADR-0002: explicit active-context switch (unregistered id -> NOT_FOUND;
    # target == active -> idempotent success)
    lib.CANN_LMEngine_ExecutorSetContext.argtypes = [ct.c_void_p, ct.c_uint64]
    lib.CANN_LMEngine_ExecutorSetContext.restype = ct.c_int
    lib.CANN_LMEngine_ContextCancel.argtypes = [ct.c_void_p, ct.c_uint64]
    lib.CANN_LMEngine_ContextCancel.restype = ct.c_int

    # Inference (prompt is c_char_p: caller passes UTF-8 bytes).
    # ADR-0002: context_id param REMOVED — requests belong to the Executor's
    # active context (set via ExecutorSetContext; no active -> INVALID_STATE).
    lib.CANN_LMEngine_ExecutorGenerate.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        P(SP),
        P(RO),
        P(RS),
    ]
    lib.CANN_LMEngine_ExecutorGenerate.restype = ct.c_int
    lib.CANN_LMEngine_ExecutorGenerateAsync.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        P(SP),
        P(RO),
        P(ct.c_void_p),
        CANN_LMEngine_StreamCallback,
        ct.c_void_p,
    ]
    lib.CANN_LMEngine_ExecutorGenerateAsync.restype = ct.c_int

    # Request handle
    lib.CANN_LMEngine_RequestGetStatus.argtypes = [ct.c_void_p]
    lib.CANN_LMEngine_RequestGetStatus.restype = ct.c_int
    lib.CANN_LMEngine_RequestIsDone.argtypes = [ct.c_void_p]
    lib.CANN_LMEngine_RequestIsDone.restype = ct.c_int
    lib.CANN_LMEngine_RequestCancel.argtypes = [ct.c_void_p]
    lib.CANN_LMEngine_RequestCancel.restype = ct.c_int
    lib.CANN_LMEngine_RequestWait.argtypes = [ct.c_void_p, P(RS)]
    lib.CANN_LMEngine_RequestWait.restype = ct.c_int
    lib.CANN_LMEngine_RequestTryGetResult.argtypes = [ct.c_void_p, P(RS)]
    lib.CANN_LMEngine_RequestTryGetResult.restype = ct.c_int
    lib.CANN_LMEngine_RequestDestroy.argtypes = [ct.c_void_p]
    lib.CANN_LMEngine_RequestDestroy.restype = None

    # Result memory management
    lib.CANN_LMEngine_ResultDestroy.argtypes = [P(RS)]
    lib.CANN_LMEngine_ResultDestroy.restype = None


def get_library(path=None):
    """Resolve and load libcann_lm_engine.so. Raises CannLmError if
    no candidate loads (not found, or wrong architecture, or missing runtime deps).

    An explicit `path` BYPASSES the module-global cache: the caller pointed at a
    specific .so, so returning some other previously-loaded .so (e.g. one the
    no-arg form cached at import via is_library_available()) would be wrong — a
    bad explicit path must raise, not silently fall back to the cached lib. The
    no-arg form reuses + populates the cache (the common case)."""
    global _lib
    if path is None and _lib is not None:
        return _lib
    skipped = []
    errors = []
    for cand in _candidate_paths(path, strict_explicit=(path is not None)):
        if not cand:
            continue
        if not os.path.isfile(cand):
            skipped.append(f"  {cand}  (does not exist)")
            continue
        try:
            # 确保同目录的依赖 .so（libc++_shared.so 等）能被找到
            _lib_dir = os.path.dirname(cand)
            if _lib_dir:
                os.environ["LD_LIBRARY_PATH"] = (
                    _lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                )
            lib = ct.CDLL(cand)
        except OSError as exc:
            errors.append(f"  {cand}: {exc}")
            continue
        _configure_prototypes(lib)
        lib.cann_lm_path = cand
        if path is None:  # only the no-arg form populates the cache
            _lib = lib
        return lib
    lines = ["libcann_lm_engine.so could not be loaded."]
    if errors:
        lines.append(
            "Found but CDLL failed (likely missing runtime deps or wrong arch):"
        )
        lines.extend(errors)
    if skipped:
        lines.append("Checked but not present:")
        lines.extend(skipped)
    if not errors and not skipped:
        lines.append("  (no candidate paths — set CANN_LM_LIB or pass --lib)")
    lines.append(
        "Fix: set CANN_LM_LIB to the .so path (or its directory). If the .so "
        "exists but won't load, put its runtime deps "
        "(libhiai_ascendcl.so / libneural_network_core.so / libhiai_foundation.so) "
        "on LD_LIBRARY_PATH."
    )
    raise CannLmError("\n".join(lines))


def is_library_available(path=None):
    """True iff get_library() can load a .so right now (for test skip-guards)."""
    if _lib is not None:
        return True
    try:
        get_library(path)
        return True
    except CannLmError:
        return False
