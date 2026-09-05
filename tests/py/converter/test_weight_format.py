"""weight_format 包单测 — nd2nz 布局变换 + resolve_format 规则 + _pack_weights_to_dir env-gated 集成.

纯 tensor/logic ops (无 torchair/GE): nd2nz round-trip + dtype 保持 + 非 2D 拒绝;
resolve_format MatMul->NZ / QuantBatchMatmulV4->ND / 默认 ND; load_rules 读 shipped json;
_pack_weights_to_dir env off (全 ND, 字节零回归) + env on (*_proj.weight bf16->NZ, 字节 != ND).
"""
import torch

import pytest
from cann_lm.converter.export.epair_utils import _pack_weights_to_dir
from cann_lm.converter.export.weight_format import load_rules, nd2nz, resolve_format


# ── nd2nz ──

def test_nd2nz_roundtrip_aligned():
    t = torch.arange(128 * 128, dtype=torch.float16).reshape(128, 128)
    nz = nd2nz(t)
    assert nz.shape == (8, 8, 16, 16)
    back = nz.permute(0, 2, 1, 3).reshape(128, 128)
    assert torch.equal(back, t)


def test_nd2nz_roundtrip_unaligned():
    t = torch.arange(130 * 130, dtype=torch.float16).reshape(130, 130)
    nz = nd2nz(t, block=16, pad=True)
    assert nz.shape == (9, 9, 16, 16)  # ceil(130/16)=9
    back = nz.permute(0, 2, 1, 3).reshape(144, 144)
    assert torch.equal(back[:130, :130], t)
    assert torch.count_nonzero(back[130:, :]) == 0  # pad rows zero
    assert torch.count_nonzero(back[:, 130:]) == 0  # pad cols zero


def test_nd2nz_preserves_dtype():
    for dt in (torch.float16, torch.int8, torch.int32):
        t = torch.zeros(16, 16, dtype=dt)
        assert nd2nz(t).dtype == dt


def test_nd2nz_bf16_via_int16_view():
    t = torch.zeros(16, 16, dtype=torch.bfloat16)
    nz = nd2nz(t.view(torch.int16))
    assert nz.dtype == torch.int16
    assert nz.shape == (1, 1, 16, 16)


def test_nd2nz_rejects_non_2d():
    with pytest.raises(ValueError):
        nd2nz(torch.zeros(4, 4, 4))


def test_nd2nz_rejects_unaligned_no_pad():
    with pytest.raises(ValueError):
        nd2nz(torch.zeros(17, 17), pad=False)


# ── resolve_format / load_rules ──

def test_load_rules_default_has_matmul_nz():
    r = load_rules()
    assert r["default"]["format"] == "ND"
    matmul_rules = [x for x in r["rules"] if x["op_type"] == "MatMul"]
    assert any(x["format"] == "NZ" for x in matmul_rules)


def test_resolve_format_matmul_proj_nz():
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight", is_quantized=False)
    assert rule["format"] == "NZ"
    assert rule["block"] == 16


def test_resolve_format_quant_keeps_nd():
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight", is_quantized=True)
    assert rule["format"] == "ND"


def test_resolve_format_default_nd_for_non_proj():
    rule = resolve_format("model.norm.weight", is_quantized=False)
    assert rule["format"] == "ND"


# ── _pack_weights_to_dir env-gated 集成 ──

class _FakeGS:
    def __init__(self, names):
        self.inputs_to_parameters = {n: n for n in names}
        self.inputs_to_buffers = {}
        self.inputs_to_lifted_tensor_constants = {}


class _FakeEP:
    def __init__(self, sd):
        self._sd = dict(sd)
        self.graph_signature = _FakeGS(list(sd))
        self.constants = {}

    def state_dict(self):
        return self._sd

    def module(self):
        return torch.nn.Module()


class _FakeContents:
    def __init__(self, sd):
        self.ep = _FakeEP(sd)


def _proj_sd():
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            128 * 128, dtype=torch.bfloat16).reshape(128, 128),
        "model.norm.weight": torch.ones(128, dtype=torch.float16),
    }


def test_pack_env_off_all_nd(tmp_path, monkeypatch):
    monkeypatch.delenv("CANN_LM_ENABLE_WEIGHT_TRANS", raising=False)
    idx = _pack_weights_to_dir(_FakeContents(_proj_sd()), tmp_path, max_file_size=1 << 30)
    for entries in idx.values():
        for e in entries:
            assert e["format"] == "ND"


def test_pack_env_off_bytes_match_no_transform(tmp_path, monkeypatch):
    monkeypatch.delenv("CANN_LM_ENABLE_WEIGHT_TRANS", raising=False)
    sd = _proj_sd()
    _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
    bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
    expected = sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16).numpy().tobytes()
    expected += sd["model.norm.weight"].numpy().tobytes()
    assert bin_bytes == expected


def test_pack_env_on_proj_nz(tmp_path, monkeypatch):
    monkeypatch.setenv("CANN_LM_ENABLE_WEIGHT_TRANS", "1")
    sd = _proj_sd()
    idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
    by_name = {e["name"]: e for entries in idx.values() for e in entries}
    assert by_name["model.layers.0.self_attn.q_proj.weight"]["format"] == "NZ"
    assert by_name["model.norm.weight"]["format"] == "ND"
    bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
    expected_nd = sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16).numpy().tobytes()
    expected_nd += sd["model.norm.weight"].numpy().tobytes()
    assert bin_bytes != expected_nd  # q_proj part rearranged
    nz = nd2nz(sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16))
    assert bin_bytes.startswith(nz.numpy().tobytes())  # q_proj is first in bin
