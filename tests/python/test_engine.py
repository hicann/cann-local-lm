"""Engine wrapper tests.

Tier A (no .so needed): class existence, the no-library error path, the
Result data shape, and build_model_artifact's no-model form. These run
everywhere.

Tier B (skip-guarded): actual C calls. These only run where a
matching-architecture libcann_lm_engine.so can be ctypes-loaded (OHOS device,
or a host with a built .so). On a host with only the OHOS aarch64 .so they
are skipped (wrong arch). Lifecycle is asserted to succeed per c_api.h:155
("图位全空构造成功，生命周期 API 可用"); generate's outcome is flexible
(SUCCESS->Result, or CannLmStatusError on a no-backend host-stub).
"""

import pytest

from cann_lm._enums import FinishReason, Status
from cann_lm._ffi import is_library_available
from cann_lm.engine import Engine, Request, Result, build_model_artifact
from cann_lm.exceptions import CannLmError, CannLmStatusError

HAS_LIB = is_library_available()
needs_lib = pytest.mark.skipif(not HAS_LIB, reason="no loadable libcann_lm_engine.so")


# ════════════════════════ Tier A: no .so needed ════════════════════════


class TestResultShape:
    def test_construct_and_attrs(self):
        r = Result(
            token_ids=[1, 2, 3], text="hello", finish_reason=FinishReason.MAX_TOKENS
        )
        assert r.token_ids == [1, 2, 3]
        assert r.text == "hello"
        assert r.finish_reason == FinishReason.MAX_TOKENS


class TestBuildModelArtifact:
    def test_no_model_form_all_paths_null(self):
        art, keepalive = build_model_artifact()
        # no-model form: all path pointers NULL, all buffer data NULL, sizes 0
        for f in (
            "prefill_omc_path",
            "decode_omc_path",
            "prefill_weight",
            "decode_weight",
            "meta_json_path",
        ):
            assert not getattr(art, f), f"{f} should be NULL"
        for b in ("prefill_omc_buf", "decode_omc_buf", "meta_json_buf"):
            assert not getattr(art, b).data
            assert getattr(art, b).size == 0

    def test_path_fields_set_and_kept_alive(self):
        art, keepalive = build_model_artifact(
            prefill_omc_path="/path/to/prefill.omc", meta_json_path="/path/to/meta.json"
        )
        assert art.prefill_omc_path  # non-NULL
        assert art.meta_json_path
        assert keepalive  # backing bytes retained

    def test_buffer_fields_from_bytes(self):
        art, _ = build_model_artifact(
            prefill_omc_data=b"GRAPH", meta_json_data=b'{"k":1}'
        )
        assert art.prefill_omc_buf.size == 5
        assert art.meta_json_buf.size == 7
        assert art.prefill_omc_buf.data
        assert art.meta_json_buf.data


class TestEngineNoLibrary:
    def test_engine_with_explicit_bad_path_raises_cannlm_error(self):
        with pytest.raises(CannLmError):
            Engine(lib_path="/nonexistent/libcann_lm_engine.so")

    @pytest.mark.skipif(
        HAS_LIB, reason="a lib is loadable here; the no-lib path can't be exercised"
    )
    def test_engine_with_no_loadable_lib_raises(self):
        with pytest.raises(CannLmError):
            Engine()


class TestExecutorCreateInvalidArgHint:
    """On INVALID_ARGUMENT the coarse Status hides which file/field is bad; the
    error message must point the user at the C++ logs (hilog tag CANN_LM_OPEN)
    where the specific cause was logged. Tier A: the lib is faked."""

    def test_invalid_argument_message_points_to_hilog(self, monkeypatch):
        import types
        from cann_lm.engine import Engine

        # fake lib: ExecutorCreate returns INVALID_ARGUMENT(2), *out stays NULL
        fake_lib = types.SimpleNamespace(
            CANN_LMEngine_ExecutorCreate=lambda *a: int(Status.INVALID_ARGUMENT)
        )
        monkeypatch.setattr("cann_lm.engine.get_library", lambda path=None: fake_lib)
        with pytest.raises(CannLmStatusError) as ei:
            Engine()
        assert ei.value.status == Status.INVALID_ARGUMENT
        msg = str(ei.value)
        assert "hilog" in msg.lower()
        assert "CANN_LM_OPEN" in msg


# ════════════════════════ Tier B: needs loadable .so ════════════════════════


@needs_lib
class TestEngineLifecycle:
    def test_no_model_form_constructs_and_closes(self):
        # c_api.h:155 — all-empty artifact constructs successfully ("no model
        # form"); lifecycle API usable; Generate reports error.
        with Engine() as eng:
            assert eng.handle is not None
        # after close, handle is released (no crash on re-close)

    def test_context_lifecycle(self):
        with Engine() as eng:
            ctx = eng.create_context()
            assert ctx.id >= 0
            eng.cancel_context(ctx.id)  # cancel a idle context is a no-op success
            eng.destroy_context(ctx.id)


@needs_lib
class TestSetContext:
    """ADR-0002 (bbit.cpp §2b): unregistered id -> NOT_FOUND (S3); created
    context activates (SUCCESS); re-setting the active one is idempotent (S5,
    c_api.h:177)."""

    def test_set_context_not_found_then_activate(self):
        with Engine() as eng:
            with pytest.raises(CannLmStatusError) as ei:
                eng.set_context(999)
            assert ei.value.status == Status.NOT_FOUND
            ctx = eng.create_context()
            eng.set_context(ctx)  # Context object form
            eng.set_context(ctx.id)  # raw-id form; already active -> idempotent


@needs_lib
class TestGenerate:
    def test_generate_without_active_context_rejected(self):
        # ADR-0002 S2 (bbit.cpp §2b): Generate before SetContext ->
        # INVALID_STATE. The pipeline checks this FIRST (before tokenize/
        # backend work), so it holds on the no-model host-stub too.
        with Engine() as eng:
            eng.create_context()  # created but NOT activated
            with pytest.raises(CannLmStatusError) as ei:
                eng.generate("hello")
            assert ei.value.status == Status.INVALID_STATE

    def test_generate_returns_result_or_raises_on_host_stub(self):
        with Engine() as eng:
            ctx = eng.create_context()
            eng.set_context(ctx)
            try:
                r = eng.generate("hello")
                assert isinstance(r, Result)
            except CannLmStatusError as e:
                # host-stub: no backend -> FAILURE; partial result may be attached
                assert e.status in (
                    Status.FAILURE,
                    Status.NOT_FOUND,
                    Status.INVALID_STATE,
                )


@needs_lib
class TestRequestAsync:
    def test_generate_async_returns_request_and_wait_resolves(self):
        # C contract (c_api.cpp:292, c_api.h:154): on non-SUCCESS,
        # ExecutorGenerateAsync returns the status SYNCHRONOUSLY and leaves
        # *out=NULL — no Request handle is created. So generate_async() raises
        # on immediate failure (e.g. no model loaded → prompt encode FAILURE)
        # rather than handing back a Request whose wait() would fail. Accept
        # both the synchronous-raise path and the deferred Request path.
        with Engine() as eng:
            ctx = eng.create_context()
            eng.set_context(ctx)
            seen = []
            # callback contract is (token_id, text) — two args; seen.append
            # alone would raise TypeError inside the trampoline (swallowed)
            # and never record, silently skipping the callback path.
            cb = lambda tok, text: seen.append((tok, text))  # noqa: E731
            try:
                req = eng.generate_async("hello", callback=cb)
            except CannLmStatusError as e:
                assert e.status in (
                    Status.FAILURE,
                    Status.NOT_FOUND,
                    Status.INVALID_STATE,
                )
                return
            assert isinstance(req, Request)
            try:
                r = req.wait()
                assert isinstance(r, Result)
            except CannLmStatusError as e:
                assert e.status in (
                    Status.FAILURE,
                    Status.NOT_FOUND,
                    Status.INVALID_STATE,
                )
            # status is a RequestStatus enum
            assert int(req.status) >= 0
