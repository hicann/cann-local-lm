"""Layout tests — the ctypes Structures in cann_lm._ffi MUST mirror the C
structs in cann_lm/engine/api/c/c_api.h byte-for-byte (field order, types,
padding) on a 64-bit target. Wrong layout = memory corruption across the FFI
boundary, so these are the highest-value assertions in the whole suite.

Expected offsets computed by hand from c_api.h for an LP64 target (pointers &
size_t = 8 bytes, int/uint32/float/enum = 4 bytes, uint64 = 8 bytes):

  Buffer            : {void* data@0; size_t size@8}                        -> 16
  ModelArtifact     : 8 ptr/buffer fields, no internal padding             -> 88
  EngineOptions     : 3 ints + pad(4) + 2 ptrs                             -> 32
  SamplingParams    : 3x4 + pad(4) + uint64@16 + float@24 + pad(4)         -> 32
  RequestOptions    : int + pad(4) + ptr + size_t + ptr + size_t           -> 40
  Result            : ptr + size_t + ptr + enum(int)@24 + pad(4)           -> 32
"""
import ctypes as ct

import pytest

# RED: _ffi does not exist yet -> ImportError is the expected first failure.
from cann_lm._ffi import (
    CANN_LMEngine_Buffer,
    CANN_LMEngine_EngineOptions,
    CANN_LMEngine_ModelArtifact,
    CANN_LMEngine_RequestOptions,
    CANN_LMEngine_Result,
    CANN_LMEngine_SamplingParams,
)


def field_offset(cls, name):
    """Offset of a ctypes Structure field (CField descriptor exposes .offset)."""
    return getattr(cls, name).offset


def field_size(cls, name):
    return getattr(cls, name).size


# ────────────────────────── Buffer ──────────────────────────

class TestBuffer:
    def test_sizeof_is_16(self):
        assert ct.sizeof(CANN_LMEngine_Buffer) == 16

    def test_field_offsets(self):
        assert field_offset(CANN_LMEngine_Buffer, "data") == 0
        assert field_offset(CANN_LMEngine_Buffer, "size") == 8

    def test_field_types(self):
        assert field_size(CANN_LMEngine_Buffer, "data") == 8   # void*
        assert field_size(CANN_LMEngine_Buffer, "size") == 8   # size_t


# ────────────────────────── ModelArtifact ──────────────────────────

class TestModelArtifact:
    def test_sizeof_is_88(self):
        assert ct.sizeof(CANN_LMEngine_ModelArtifact) == 88

    def test_field_offsets(self):
        cls = CANN_LMEngine_ModelArtifact
        assert field_offset(cls, "prefill_omc_path") == 0
        assert field_offset(cls, "prefill_omc_buf") == 8
        assert field_offset(cls, "decode_omc_path") == 24
        assert field_offset(cls, "decode_omc_buf") == 32
        assert field_offset(cls, "prefill_weight") == 48
        assert field_offset(cls, "decode_weight") == 56
        assert field_offset(cls, "meta_json_path") == 64
        assert field_offset(cls, "meta_json_buf") == 72

    def test_pointer_fields_are_8_bytes(self):
        cls = CANN_LMEngine_ModelArtifact
        for name in ("prefill_omc_path", "decode_omc_path",
                     "prefill_weight", "decode_weight", "meta_json_path"):
            assert field_size(cls, name) == 8


# ────────────────────────── EngineOptions ──────────────────────────

class TestEngineOptions:
    def test_sizeof_is_32(self):
        assert ct.sizeof(CANN_LMEngine_EngineOptions) == 32

    def test_ints_packed_then_padding_before_pointers(self):
        cls = CANN_LMEngine_EngineOptions
        assert field_offset(cls, "default_chunk_size") == 0
        assert field_offset(cls, "max_concurrent_prefill") == 4
        assert field_offset(cls, "max_batch_size_override") == 8
        # pointer alignment: 4 bytes padding at offset 12, ptr at 16
        assert field_offset(cls, "tokenizer_type") == 16
        assert field_offset(cls, "tokenizer_path") == 24

    def test_int_fields_are_4_bytes(self):
        cls = CANN_LMEngine_EngineOptions
        for name in ("default_chunk_size", "max_concurrent_prefill",
                     "max_batch_size_override"):
            assert field_size(cls, name) == 4


# ────────────────────────── SamplingParams ──────────────────────────

class TestSamplingParams:
    def test_sizeof_is_32(self):
        assert ct.sizeof(CANN_LMEngine_SamplingParams) == 32

    def test_seed_at_16_after_padding(self):
        cls = CANN_LMEngine_SamplingParams
        assert field_offset(cls, "temperature") == 0      # float 4
        assert field_offset(cls, "top_k") == 4            # int 4
        assert field_offset(cls, "top_p") == 8            # float 4
        # uint64 needs 8-align: 4 bytes padding at 12, seed at 16
        assert field_offset(cls, "seed") == 16
        assert field_offset(cls, "repetition_penalty") == 24  # float 4
        # trailing 4 bytes pad to struct align 8

    def test_seed_is_8_bytes(self):
        assert field_size(CANN_LMEngine_SamplingParams, "seed") == 8


# ────────────────────────── RequestOptions ──────────────────────────

class TestRequestOptions:
    def test_sizeof_is_40(self):
        assert ct.sizeof(CANN_LMEngine_RequestOptions) == 40

    def test_int_then_padding_before_pointer_array_pair(self):
        cls = CANN_LMEngine_RequestOptions
        assert field_offset(cls, "max_tokens") == 0            # int 4
        # 4 bytes pad, pointer at 8
        assert field_offset(cls, "stop_token_ids") == 8        # TokenId* 8
        assert field_offset(cls, "stop_token_ids_count") == 16 # size_t 8
        assert field_offset(cls, "stop_strings") == 24         # char** 8
        assert field_offset(cls, "stop_strings_count") == 32   # size_t 8


# ────────────────────────── Result ──────────────────────────

class TestResult:
    def test_sizeof_is_32(self):
        assert ct.sizeof(CANN_LMEngine_Result) == 32

    def test_layout(self):
        cls = CANN_LMEngine_Result
        assert field_offset(cls, "token_ids") == 0          # TokenId* 8
        assert field_offset(cls, "token_ids_count") == 8    # size_t 8
        assert field_offset(cls, "text") == 16              # char* 8
        assert field_offset(cls, "finish_reason") == 24     # enum=int 4
        # trailing 4 bytes pad to struct align 8

    def test_finish_reason_is_4_byte_int(self):
        assert field_size(CANN_LMEngine_Result, "finish_reason") == 4
