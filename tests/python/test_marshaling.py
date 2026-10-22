"""Marshaling tests — the helpers that turn Python-friendly inputs into the
ctypes arrays/buffers the C functions expect. These are FFI mechanics (not
"extra logic"): the C RequestOptions carries (ptr, count) pairs and the C
Buffer carries (data ptr, size); Python lists/bytes must become exactly that,
and the backing storage must stay alive for the duration of the C call.

These run without the engine .so — pure ctypes array/buffer work.
"""
import ctypes as ct

import pytest

from cann_lm.engine import build_buffer, build_request_options


# ────────────────────────── stop_token_ids ──────────────────────────

class TestStopTokenIds:
    def test_nonempty_array_content_and_count(self):
        opts, keepalive = build_request_options(
            max_tokens=8, stop_token_ids=[1, 2, 300])
        assert opts.stop_token_ids_count == 3
        assert opts.max_tokens == 8
        # read back through the stored pointer
        arr = ct.cast(opts.stop_token_ids, ct.POINTER(ct.c_uint32))
        assert [arr[i] for i in range(3)] == [1, 2, 300]
        assert keepalive  # backing array kept alive

    def test_empty_list_is_null_with_zero_count(self):
        opts, _ = build_request_options(stop_token_ids=[])
        assert opts.stop_token_ids_count == 0
        assert not opts.stop_token_ids  # NULL (c_void_p .value == 0)

    def test_none_is_null_with_zero_count(self):
        opts, _ = build_request_options(stop_token_ids=None)
        assert opts.stop_token_ids_count == 0
        assert not opts.stop_token_ids


# ────────────────────────── stop_strings ──────────────────────────

class TestStopStrings:
    def test_nonempty_ascii_and_unicode_roundtrip(self):
        opts, _ = build_request_options(stop_strings=["hello", "世界"])
        assert opts.stop_strings_count == 2
        arr = ct.cast(opts.stop_strings, ct.POINTER(ct.c_char_p))
        assert arr[0] == b"hello"
        assert arr[1] == "世界".encode("utf-8")

    def test_empty_list_is_null(self):
        opts, _ = build_request_options(stop_strings=[])
        assert opts.stop_strings_count == 0
        assert not opts.stop_strings


# ────────────────────────── Buffer ──────────────────────────

class TestBuildBuffer:
    def test_none_is_empty_buffer(self):
        buf, _ = build_buffer(None)
        assert not buf.data          # NULL
        assert buf.size == 0

    def test_bytes_content_and_size(self):
        buf, _ = build_buffer(b"abc")
        assert buf.size == 3
        assert buf.data              # non-NULL
        ptr = ct.cast(buf.data, ct.POINTER(ct.c_char))
        assert ptr[0:3] == b"abc"

    def test_null_bytes_preserved(self):
        data = b"\x00\x01\x02"
        buf, _ = build_buffer(data)
        assert buf.size == 3
        ptr = ct.cast(buf.data, ct.POINTER(ct.c_char))
        assert ptr[0:3] == data

    def test_bytearray_supported(self):
        buf, _ = build_buffer(bytearray(b"xyz"))
        assert buf.size == 3
        ptr = ct.cast(buf.data, ct.POINTER(ct.c_char))
        assert ptr[0:3] == b"xyz"

    def test_backing_storage_kept_alive(self):
        buf, keepalive = build_buffer(b"alive")
        assert keepalive            # caller must retain this during the C call
        ptr = ct.cast(buf.data, ct.POINTER(ct.c_char))
        assert ptr[0:5] == b"alive"
