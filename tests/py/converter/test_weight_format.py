"""weight_format 包单测 — nd2nz 布局变换 + resolve_format 规则 (含 enable 字段) + _pack_weights_to_dir 集成.

纯 tensor/logic ops (无 torchair/GE): nd2nz round-trip + dtype 保持 + 非 2D 拒绝 + NZ 布局
验证 [n1,k1,k0,n0]; resolve_format enable:false 跳过 + enable:true 命中 + default ND;
load_rules 读 shipped json; _pack_weights_to_dir 空 rules (全 ND) + shipped rules
(MatMul enable:false → 全 ND; QuantBatchMatmulV4 enable:true → weight_q Z-pack).
"""
import json
import pytest
import torch

from cann_lm.converter.export.epair_utils import _pack_weights_to_dir
from cann_lm.converter.export.weight_format import load_rules, nd2nz, resolve_format


# ── nd2nz ──

def test_nd2nz_shape_matches_cann_nz_spec():
    """NZ shape = [n1, k1, k0, n0] per CANN aclnnMatmulWeightNz doc (k0=n0=16)."""
    t = torch.zeros(128, 128, dtype=torch.float16)
    nz = nd2nz(t)
    assert nz.shape == (8, 8, 16, 16)  # [n1=8, k1=8, k0=16, n0=16]


def test_nd2nz_roundtrip_aligned():
    t = torch.arange(128 * 128, dtype=torch.float16).reshape(128, 128)
    nz = nd2nz(t)
    assert nz.shape == (8, 8, 16, 16)
    back = nz.permute(1, 2, 0, 3).reshape(128, 128)
    assert torch.equal(back, t)


def test_nd2nz_roundtrip_unaligned():
    t = torch.arange(130 * 130, dtype=torch.float16).reshape(130, 130)
    nz = nd2nz(t, block=16, pad=True)
    assert nz.shape == (9, 9, 16, 16)  # ceil(130/16)=9
    back = nz.permute(1, 2, 0, 3).reshape(144, 144)
    assert torch.equal(back[:130, :130], t)
    assert torch.count_nonzero(back[130:, :]) == 0
    assert torch.count_nonzero(back[:, 130:]) == 0


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


def test_nd2nz_byte_layout_matches_n1_k1_k0_n0():
    """Verify element ordering: NZ [n1,k1,k0,n0] — block-major, k0 before n0 (CANN spec)."""
    t = torch.arange(256, dtype=torch.float16).reshape(16, 16)  # [K=16, N=16]
    nz = nd2nz(t)
    assert nz.shape == (1, 1, 16, 16)
    for i in range(16):
        for j in range(16):
            assert nz[0, 0, i, j].item() == t[i, j].item()


def test_nd2nz_multi_block_layout():
    """32x32 weight → [2,2,16,16]; verify cross-block element mapping."""
    t = torch.arange(32 * 32, dtype=torch.float16).reshape(32, 32)  # [K=32, N=32]
    nz = nd2nz(t)
    assert nz.shape == (2, 2, 16, 16)
    for n1 in range(2):
        for k1 in range(2):
            for k0 in range(16):
                for n0 in range(16):
                    expected = t[k1 * 16 + k0, n1 * 16 + n0]
                    assert nz[n1, k1, k0, n0].item() == expected.item()


# ── resolve_format / load_rules ──

def test_load_rules_shipped_has_enable_field():
    """shipped rules.json has per-rule 'enable' field (env total switch deprecated)."""
    r = load_rules()
    assert r["default"]["format"] == "ND"
    for rule in r.get("rules", []):
        assert "enable" in rule, f"rule {rule.get('name_pattern')} missing 'enable' field"


def test_resolve_format_matmul_proj_nz_disabled():
    """MatMul *_proj.weight rule has enable:false → falls through to default ND."""
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight", is_quantized=False)
    assert rule["format"] == "ND"  # enable:false → skipped → default


def test_resolve_format_matmul_enable_true_overrides():
    """When MatMul rule is enabled, *_proj.weight → NZ."""
    rules = {
        "default": {"format": "ND"},
        "rules": [
            {"name_pattern": "*_proj.weight", "op_type": "MatMul",
             "format": "NZ", "block": 16, "pad": True, "pack": "nd2nz", "enable": True},
        ],
    }
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight", is_quantized=False, rules=rules)
    assert rule["format"] == "NZ"
    assert rule["block"] == 16


def test_resolve_format_matmul_enable_false_skips_and_continues():
    """enable:false rule is skipped; subsequent matching rules are still scanned."""
    rules = {
        "default": {"format": "ND"},
        "rules": [
            {"name_pattern": "*_proj.weight", "op_type": "MatMul",
             "format": "NZ", "enable": False},
            {"name_pattern": "*_proj.weight", "op_type": "MatMul",
             "format": "NZ", "enable": True, "block": 32},
        ],
    }
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight", is_quantized=False, rules=rules)
    assert rule["format"] == "NZ"
    assert rule["block"] == 32  # second rule matched (first skipped)


def test_resolve_format_quant_weight_q_z_enabled():
    """QuantBatchMatmulV4 *.weight_q rule has enable:true → Z format."""
    rule = resolve_format("model.layers.0.self_attn.q_proj.weight_q", is_quantized=True)
    assert rule["format"] == "Z"
    assert rule.get("pack") == "pack_int4_nz"


def test_resolve_format_quant_non_weight_q_falls_to_nd():
    """Non-*.weight_q tensors (e.g. rotary cos/sin) fall to default ND even in quant mode."""
    rule = resolve_format("model.model.rotary_emb.cos_cached", is_quantized=True)
    assert rule["format"] == "ND"


def test_resolve_format_default_nd_for_non_proj():
    rule = resolve_format("model.norm.weight", is_quantized=False)
    assert rule["format"] == "ND"


def test_resolve_format_omit_enable_defaults_true():
    """Omitted 'enable' field defaults to True (rule is active)."""
    rules = {
        "default": {"format": "ND"},
        "rules": [
            {"name_pattern": "*_proj.weight", "op_type": "MatMul", "format": "NZ"},
        ],
    }
    rule = resolve_format("model.layers.0.q_proj.weight", is_quantized=False, rules=rules)
    assert rule["format"] == "NZ"


# ── _pack_weights_to_dir 集成 ──

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


def test_pack_empty_rules_all_nd(tmp_path):
    """Empty rules json → all weights ND (escape hatch = clear rules)."""
    import cann_lm.converter.export.weight_format as wf_pkg
    original = wf_pkg.load_rules
    wf_pkg.load_rules = lambda path=None: {"default": {"format": "ND"}, "rules": []}
    try:
        idx = _pack_weights_to_dir(_FakeContents(_proj_sd()), tmp_path, max_file_size=1 << 30)
        for entries in idx.values():
            for e in entries:
                assert e["format"] == "ND"
    finally:
        wf_pkg.load_rules = original


def test_pack_empty_rules_bytes_match_no_transform(tmp_path):
    """Empty rules → bytes identical to raw state_dict (no layout transform)."""
    import cann_lm.converter.export.weight_format as wf_pkg
    original = wf_pkg.load_rules
    wf_pkg.load_rules = lambda path=None: {"default": {"format": "ND"}, "rules": []}
    try:
        sd = _proj_sd()
        _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
        bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
        expected = sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16).numpy().tobytes()
        expected += sd["model.norm.weight"].numpy().tobytes()
        assert bin_bytes == expected
    finally:
        wf_pkg.load_rules = original


def test_pack_shipped_rules_matmul_disabled_all_nd(tmp_path):
    """Shipped rules: MatMul enable:false → *_proj.weight stays ND (zero behavior change)."""
    sd = _proj_sd()
    idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
    by_name = {e["name"]: e for entries in idx.values() for e in entries}
    assert by_name["model.layers.0.self_attn.q_proj.weight"]["format"] == "ND"
    assert by_name["model.norm.weight"]["format"] == "ND"
    # Bytes identical to raw (no transform)
    bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
    expected = sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16).numpy().tobytes()
    expected += sd["model.norm.weight"].numpy().tobytes()
    assert bin_bytes == expected


def test_pack_shipped_rules_weight_q_z_pack(tmp_path):
    """Shipped rules: QuantBatchMatmulV4 *.weight_q enable:true → Z format + Z bytes."""
    from cann_lm.converter.export.weight_format import pack_int4_nz
    sd = {
        "model.layers.0.self_attn.q_proj.weight_q": torch.arange(
            128 * 16, dtype=torch.int32).reshape(128, 16),  # [K=128, N_packed=16] → N=128
        "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(128, 1, dtype=torch.float32),
        "model.norm.weight": torch.ones(128, dtype=torch.float16),
    }
    idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
    by_name = {e["name"]: e for entries in idx.values() for e in entries}
    assert by_name["model.layers.0.self_attn.q_proj.weight_q"]["format"] == "Z"
    assert by_name["model.norm.weight"]["format"] == "ND"
    # Verify Z-packed bytes match pack_int4_nz output
    bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
    expected_z = pack_int4_nz(sd["model.layers.0.self_attn.q_proj.weight_q"]).numpy().tobytes()
    assert bin_bytes.startswith(expected_z)  # weight_q is first in pack order


def test_pack_shipped_rules_rotary_not_packed(tmp_path):
    """Rotary cos/sin (lifted tensor constant) must NOT be Z-packed (wildcard guard)."""
    sd = {
        "model.model.rotary_emb.cos_cached_full_attention": torch.arange(
            128 * 128, dtype=torch.float32).reshape(128, 128),
    }
    idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
    by_name = {e["name"]: e for entries in idx.values() for e in entries}
    assert by_name["model.model.rotary_emb.cos_cached_full_attention"]["format"] == "ND"


def test_pack_matmul_enabled_nz(tmp_path):
    """When MatMul rule is enabled (override), *_proj.weight → NZ bytes."""
    import cann_lm.converter.export.weight_format as wf_pkg
    original = wf_pkg.load_rules
    wf_pkg.load_rules = lambda path=None: {
        "default": {"format": "ND"},
        "rules": [
            {"name_pattern": "*_proj.weight", "op_type": "MatMul",
             "format": "NZ", "block": 16, "pad": True, "pack": "nd2nz", "enable": True},
        ],
    }
    try:
        sd = _proj_sd()
        idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
        by_name = {e["name"]: e for entries in idx.values() for e in entries}
        assert by_name["model.layers.0.self_attn.q_proj.weight"]["format"] == "NZ"
        assert by_name["model.norm.weight"]["format"] == "ND"
        bin_bytes = (tmp_path / "weights_0.bin").read_bytes()
        expected_nz = nd2nz(sd["model.layers.0.self_attn.q_proj.weight"].view(torch.int16)).numpy().tobytes()
        expected_nd = sd["model.norm.weight"].numpy().tobytes()
        assert bin_bytes == expected_nz + expected_nd
    finally:
        wf_pkg.load_rules = original


def test_pack_non_aligned_falls_back_to_nd(tmp_path):
    """Non-16-aligned shapes fall back to ND (GE CheckTensorIsValid limitation)."""
    import cann_lm.converter.export.weight_format as wf_pkg
    original = wf_pkg.load_rules
    wf_pkg.load_rules = lambda path=None: {
        "default": {"format": "ND"},
        "rules": [
            {"name_pattern": "*_proj.weight", "op_type": "MatMul",
             "format": "NZ", "block": 16, "pad": True, "pack": "nd2nz", "enable": True},
        ],
    }
    try:
        sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                130 * 130, dtype=torch.bfloat16).reshape(130, 130),
            "model.norm.weight": torch.ones(128, dtype=torch.float16),
        }
        idx = _pack_weights_to_dir(_FakeContents(sd), tmp_path, max_file_size=1 << 30)
        by_name = {e["name"]: e for entries in idx.values() for e in entries}
        assert by_name["model.layers.0.self_attn.q_proj.weight"]["format"] == "ND"
        assert by_name["model.norm.weight"]["format"] == "ND"
    finally:
        wf_pkg.load_rules = original
