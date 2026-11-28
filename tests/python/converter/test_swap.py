"""swap.py 单元测试 — 纯逻辑函数 (group_size 推断 / parent 解析 / dopt 产物解析).

补充 tests/dopt_lm/test_strategy_generation.py 未覆盖部分:
- _infer_group_size (pergroup/perchannel/pertensor 拒绝/维度校验)
- _resolve_parent (nn.Module 树定位 parent+leaf name)
- load_dopt_state_dict (.pth 加载 / language_model. 前缀剥离 / dir 查找 model.safetensors)
- _read_dopt_quant_meta (bit/unsigned_quant/offset 读取 + 缺失 fallback)
- _read_quant_param_map 边缘 (None 路径 / 无 dopt_config.json / 未知 strategy / 非法 suffix)
- _swap_linear_from_sd (stub nn.Linear + fake state_dict: happy path / skip / param_map / errors)
- swap_embeddings_from_dopt (stub nn.Embedding + fake sd)
- swap_from_dopt (端到端 stub: .pth -> Linear swap + Embedding 提取)
跳过: 真实 dopt artifacts + 真实 model tree 全流程 (太重, 需 HF 模型权重).
"""

import json

import pytest
import torch
import torch.nn as nn

from cann_local_lm.converter.swap import (
    _infer_group_size,
    _read_dopt_quant_meta,
    _read_quant_param_map,
    _resolve_parent,
    load_dopt_state_dict,
    swap_embeddings_from_dopt,
    swap_from_dopt,
    _swap_linear_from_sd,
)
from cann_local_lm.converter.export.quant_swap.quant_linear import QuantLinear


# ── _resolve_parent ──


class _Sub(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Sub()


class _Tree(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


def test_resolve_parent_returns_parent_and_leaf():
    m = _Tree()
    parent, name = _resolve_parent(m, "model.layers.0.self_attn.q_proj")
    assert isinstance(parent, _Sub)
    assert name == "q_proj"
    assert parent is m.model.layers[0].self_attn


def test_resolve_parent_single_segment():
    m = nn.Module()
    m.proj = nn.Linear(2, 2)
    parent, name = _resolve_parent(m, "proj")
    assert parent is m
    assert name == "proj"


def test_resolve_parent_index_path():
    m = _Tree()
    parent, name = _resolve_parent(m, "model.layers.0")
    assert parent is m.model.layers  # ModuleList (parts[:-1]=["model","layers"])
    assert name == "0"


# ── _infer_group_size ──


def test_infer_group_size_pergroup():
    # out=4, in=8, group=4 → groups=2, numel=8
    s = torch.ones(8)
    assert _infer_group_size(s, 4, 8, "x") == (4, 2)


def test_infer_group_size_perchannel():
    # perchannel: numel=out → group_size=in, groups=1
    s = torch.ones(4)
    assert _infer_group_size(s, 4, 8, "x") == (8, 1)


def test_infer_group_size_rejects_pertensor():
    with pytest.raises(ValueError, match="pertensor"):
        _infer_group_size(torch.ones(1), 4, 8, "x")


def test_infer_group_size_rejects_non_1d():
    with pytest.raises(ValueError, match="1D flat scale"):
        _infer_group_size(torch.ones(2, 4), 4, 8, "x")


def test_infer_group_size_rejects_not_divisible_by_out():
    # numel=6, out=4 → 6 % 4 != 0
    with pytest.raises(ValueError, match="not divisible by out_f"):
        _infer_group_size(torch.ones(6), 4, 8, "x")


def test_infer_group_size_rejects_in_not_divisible_by_group():
    # out=2, in=5, numel=4 → groups=2, group_size=5//2=2, 5%2=1 !=0
    with pytest.raises(ValueError, match="not divisible by inferred group_size"):
        _infer_group_size(torch.ones(4), 2, 5, "x")


def test_infer_group_size_rejects_numel_below_out():
    # 空 tensor: numel=0, 0%out==0 → groups=0 → "< out_f"
    with pytest.raises(ValueError, match="< out_f"):
        _infer_group_size(torch.ones(0), 4, 8, "x")


# ── _read_dopt_quant_meta ──


def test_read_dopt_quant_meta_fallback_when_buffers_absent():
    # sd 不含 bit/unsigned_quant/offset 键 → (4, 0.0, None)
    bit, uq, offset = _read_dopt_quant_meta({}, "x", 4, 8, 4)
    assert bit == 4
    assert uq == 0.0
    assert offset is None


@pytest.mark.parametrize("missing", ["bit", "unsigned_quant", "offset"])
def test_read_dopt_quant_meta_rejects_partial_metadata(missing):
    prefix = "x.quant_op.weight_quantizer"
    state = {
        f"{prefix}.bit": torch.tensor(4),
        f"{prefix}.unsigned_quant": torch.tensor(1.0),
        f"{prefix}.offset": torch.zeros(8),
    }
    del state[f"{prefix}.{missing}"]

    with pytest.raises(ValueError, match="incomplete.*metadata"):
        _read_dopt_quant_meta(state, "x", 4, 8, 4)


def test_read_dopt_quant_meta_reads_buffers_and_reshapes_offset():
    name = "model.layers.0.self_attn.q_proj"
    out_f, in_f, group = 4, 8, 4
    groups = in_f // group  # 2
    pre = f"{name}.quant_op.weight_quantizer"
    sd = {
        f"{pre}.bit": torch.tensor(4, dtype=torch.int64),
        f"{pre}.unsigned_quant": torch.tensor(0.0, dtype=torch.float32),
        f"{pre}.offset": torch.zeros(out_f * groups, dtype=torch.float32),  # [8]
    }
    bit, uq, offset = _read_dopt_quant_meta(sd, name, out_f, in_f, group)
    assert bit == 4
    assert uq == 0.0
    assert offset.shape == (out_f, in_f // group)  # [4, 2]


# ── _read_quant_param_map (边缘: None / 无 config / 未知 strategy / 非法 suffix) ──
# 注: 标准路径 (Quant_INT8-4_ECO + ";64" suffix) 已在 tests/dopt_lm/test_strategy_generation.py 覆盖.


def test_read_quant_param_map_none_returns_empty():
    assert _read_quant_param_map(None) == {}


def test_read_quant_param_map_dir_without_config(tmp_path):
    (tmp_path / "model.safetensors").write_text("")  # 目录存在但无 dopt_config.json
    assert _read_quant_param_map(tmp_path) == {}


def test_read_quant_param_map_unknown_strategy_skipped(tmp_path):
    (tmp_path / "dopt_config.json").write_text(
        json.dumps({"layer_strategy": {"proj": {"quant_strategy": "Quant_Unknown"}}})
    )
    assert _read_quant_param_map(tmp_path) == {}  # 未知 strategy 无默认 → 跳过


def test_read_quant_param_map_invalid_suffix_skipped(tmp_path):
    (tmp_path / "dopt_config.json").write_text(
        json.dumps(
            {"layer_strategy": {"proj": {"quant_strategy": "Quant_INT8-4_ECO;abc"}}}
        )
    )
    assert _read_quant_param_map(tmp_path) == {}  # ";abc" 非法 int → 跳过


def test_read_quant_param_map_layer_field_overrides_default(tmp_path):
    (tmp_path / "dopt_config.json").write_text(
        json.dumps(
            {
                "layer_strategy": {
                    "proj": {
                        "quant_strategy": "Quant_INT8-4_ECO",
                        "bit": 8,  # 覆盖默认 bit=4
                        "blocksize": 64,
                    }
                }
            }
        )
    )
    params = _read_quant_param_map(tmp_path)["proj"]
    assert params["bit"] == 8
    assert params["blocksize"] == 64
    assert params["group_size"] == 128  # 默认 (无 suffix 覆盖)


# ── load_dopt_state_dict (.pth / language_model. 剥离 / dir / not found) ──


def test_load_dopt_state_dict_pth(tmp_path):
    sd = {
        "model.layers.0.weight": torch.zeros(2, 2),
        "model.layers.1.weight": torch.ones(2, 2),
    }
    p = tmp_path / "out.pth"
    torch.save(sd, p)
    loaded = load_dopt_state_dict(p)
    assert set(loaded) == set(sd)
    assert torch.equal(loaded["model.layers.1.weight"], sd["model.layers.1.weight"])


def test_load_dopt_state_dict_strips_language_model_prefix(tmp_path):
    # VLM 保存: model.language_model.layers.0.weight → 剥离 → model.layers.0.weight
    sd = {"model.language_model.layers.0.weight": torch.zeros(2, 2)}
    p = tmp_path / "vlm.pth"
    torch.save(sd, p)
    loaded = load_dopt_state_dict(p)
    assert list(loaded) == ["model.layers.0.weight"]


def test_load_dopt_state_dict_dir_resolves_model_safetensors(tmp_path):
    # 目录路径 → 查找 model.safetensors (用 safetensors 写一个最小文件)
    from safetensors.torch import save_file

    save_file({"a.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    loaded = load_dopt_state_dict(tmp_path)
    assert "a.weight" in loaded
    assert loaded["a.weight"].shape == (2, 2)


def test_load_dopt_state_dict_dir_missing_safetensors_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model.safetensors"):
        load_dopt_state_dict(tmp_path)


def test_load_dopt_state_dict_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dopt_state_dict(tmp_path / "nope.pth")


# ── _swap_linear_from_sd (stub nn.Linear + fake state_dict) ──


class _LinearModel(nn.Module):
    """Minimal model: one nn.Linear (in=8, out=4) named q_proj."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 4, bias=False)


def _make_q_proj_sd(out_f=4, in_f=8, group=4, *, unsigned_quant=0.0, offset_zero=True):
    """Build a fake dopt sd for q_proj: weight [out,in] + flat scale [out*groups]."""
    groups = in_f // group
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    s = torch.ones(out_f * groups, dtype=torch.float32) * 0.1
    sd = {"q_proj.weight": w, "q_proj.quant_op.weight_quantizer.s": s}
    pre = "q_proj.quant_op.weight_quantizer"
    sd[f"{pre}.bit"] = torch.tensor(4, dtype=torch.int64)
    sd[f"{pre}.unsigned_quant"] = torch.tensor(unsigned_quant, dtype=torch.float32)
    off = torch.zeros(out_f * groups, dtype=torch.float32)
    if not offset_zero:
        off.fill_(0.5)
    sd[f"{pre}.offset"] = off
    return sd


def test_swap_linear_from_sd_happy_path_swaps_quantlinear():
    model = _LinearModel()
    sd = _make_q_proj_sd()
    info = _swap_linear_from_sd(model, sd)

    assert isinstance(model.q_proj, QuantLinear)
    assert info["scheme"] == "W4A8-s8s4"
    assert info["n_quantized"] == 1
    assert info["n_skipped_no_dopt_key"] == 0
    assert info["group_size"] == 4
    assert info["bit"] == 4
    assert info["dopt_unsigned_quant"] == 0.0
    # weight_q pre-transposed to [k//2, n] (nibble-packed int4; CPU PoC kernel
    # doesn't unpack nibbles, so forward is export-only — verify buffers, not eager)
    assert model.q_proj.weight_q.shape == (8 // 2, 4)
    assert model.q_proj.weight_q.dtype == torch.int8
    assert model.q_proj.weight_scale.shape == (8 // 4, 4)
    assert model.q_proj.weight_scale.dtype == torch.float32
    assert model.q_proj.group_size == 4


@pytest.mark.skip(reason="pack_int4_int32 not yet implemented")
def test_swap_linear_from_sd_kirinx90_int4pack_int32():
    # KirinX90: int4pack int32 (packed-K, [N, K//8]), 区别于 Ascend910 int8 nibble ([K//2,N])
    # in(K) 须整除 8 (8 int4/int32 沿 K pack). commit 33df54c: materialize 去 transpose, 直接 pack q[N,K] 沿 K 维.
    class _KirinModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 16, bias=False)  # in=8(K), out=16(N), K%8==0

    model = _KirinModel()
    sd = _make_q_proj_sd(out_f=16, in_f=8, group=4)
    info = _swap_linear_from_sd(model, sd, target_soc="KirinX90")
    assert isinstance(model.q_proj, QuantLinear)
    assert model.q_proj.weight_q.dtype == torch.int32  # int4pack int32 (KirinX90)
    assert model.q_proj.weight_q.shape == (
        16,
        8 // 8,
    )  # [N, K//8] (packed-K, 不 transpose)
    assert model.q_proj.weight_scale.shape == (8 // 4, 16)  # [K//g, N]
    assert model.q_proj.weight_scale.dtype == torch.float32
    assert info["scheme"] == "W4A8-s8s4"


def test_swap_linear_from_sd_param_map_group_size():
    model = _LinearModel()
    sd = _make_q_proj_sd(group=4)
    param_map = {
        "q_proj": {
            "group_size": 4,
            "bit": 4,
            "blocksize": 128,
            "per_channel": True,
            "weight_algo": "omni_group_min_max",
        }
    }
    info = _swap_linear_from_sd(model, sd, param_map=param_map)

    assert isinstance(model.q_proj, QuantLinear)
    assert info["group_size"] == 4
    assert info["blocksize"] == 128
    assert info["per_channel"] is True
    assert info["weight_algo"] == "omni_group_min_max"


def test_swap_linear_from_sd_param_map_bad_div_raises():
    model = _LinearModel()
    sd = _make_q_proj_sd(group=4)
    param_map = {"q_proj": {"group_size": 3}}  # 8 % 3 != 0
    with pytest.raises(ValueError, match="not divisible by group_size"):
        _swap_linear_from_sd(model, sd, param_map=param_map)


def test_swap_linear_from_sd_param_map_scale_mismatch_raises():
    model = _LinearModel()
    # sd has group=4 -> groups=2 -> scale numel=8; param_map says group=2 -> groups=4 -> expect 16
    sd = _make_q_proj_sd(group=4)
    param_map = {"q_proj": {"group_size": 2}}
    with pytest.raises(ValueError, match="scale numel .* != out_f"):
        _swap_linear_from_sd(model, sd, param_map=param_map)


def test_swap_linear_from_sd_skips_missing_dopt_keys():
    class _Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 4, bias=False)  # has dopt keys
            self.o_proj = nn.Linear(4, 4, bias=False)  # no dopt keys -> skip

    model = _Two()
    sd = _make_q_proj_sd()
    info = _swap_linear_from_sd(model, sd)

    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.o_proj, nn.Linear)  # untouched
    assert info["n_quantized"] == 1
    assert info["n_skipped_no_dopt_key"] == 1


def test_swap_linear_from_sd_skips_embed_tokens_name():
    class _Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Linear(8, 4, bias=False)  # name matches DEFAULT_SKIP

    model = _Embed()
    sd = _make_q_proj_sd()  # keys are q_proj.* -> won't match embed_tokens anyway
    sd = {k.replace("q_proj", "embed_tokens"): v for k, v in sd.items()}
    info = _swap_linear_from_sd(model, sd)
    assert info["n_quantized"] == 0
    assert isinstance(model.embed_tokens, nn.Linear)


def test_swap_linear_from_sd_rejects_unsigned_quant():
    model = _LinearModel()
    sd = _make_q_proj_sd(unsigned_quant=1.0)
    with pytest.raises(ValueError, match="unsigned_quant=1.0"):
        _swap_linear_from_sd(model, sd)


def test_swap_linear_rejects_unsigned_partial_metadata_before_replacing_layer():
    model = _LinearModel()
    sd = _make_q_proj_sd(unsigned_quant=1.0)
    del sd["q_proj.quant_op.weight_quantizer.offset"]

    with pytest.raises(ValueError, match="incomplete.*metadata"):
        _swap_linear_from_sd(model, sd)

    assert isinstance(model.q_proj, nn.Linear)


def test_swap_linear_from_sd_rejects_nonzero_offset():
    model = _LinearModel()
    sd = _make_q_proj_sd(offset_zero=False)
    with pytest.raises(NotImplementedError, match="non-zero offset"):
        _swap_linear_from_sd(model, sd)


# ── swap_embeddings_from_dopt (stub nn.Embedding + fake sd) ──


class _EmbedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 8)


def test_swap_embeddings_from_dopt_materializes_int8():
    model = _EmbedModel()
    w = torch.randn(10, 8, dtype=torch.float32)
    s = torch.ones(10, dtype=torch.float32) * 0.2
    sd = {
        "embed_tokens.weight": w,
        "embed_tokens.quant_op.weight_quantizer.s": s,
    }
    out = swap_embeddings_from_dopt(sd, model)

    assert "model.embed_tokens.weight" in out
    int8, scale = out["model.embed_tokens.weight"]
    assert int8.shape == (10, 8)
    assert int8.dtype == torch.int8
    assert scale.shape == (10,)
    assert scale.dtype == torch.float32


def test_swap_embeddings_from_dopt_skips_missing_scale():
    model = _EmbedModel()
    sd = {"embed_tokens.weight": torch.randn(10, 8)}  # no scale key
    assert swap_embeddings_from_dopt(sd, model) == {}


def test_swap_embeddings_from_dopt_skips_non_embedding():
    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 8)
            self.q_proj = nn.Linear(8, 4, bias=False)  # not nn.Embedding

    model = _Mixed()
    sd = {
        "embed_tokens.weight": torch.randn(10, 8),
        "embed_tokens.quant_op.weight_quantizer.s": torch.ones(10),
    }
    out = swap_embeddings_from_dopt(sd, model)
    assert "model.embed_tokens.weight" in out
    assert "model.q_proj.weight" not in out


# ── swap_from_dopt (端到端 stub: .pth -> Linear swap + Embedding 提取) ──


class _FullStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 8)
        self.q_proj = nn.Linear(8, 4, bias=False)


def test_swap_from_dopt_end_to_end_stub(tmp_path):
    # realistic dopt output dir layout: model.safetensors + dopt_config.json
    from safetensors.torch import save_file

    model = _FullStub()
    # build sd with both q_proj (Linear) and embed_tokens (Embedding) dopt keys
    lin_sd = _make_q_proj_sd()
    emb_w = torch.randn(10, 8, dtype=torch.float32)
    emb_s = torch.ones(10, dtype=torch.float32) * 0.2
    full_sd = {k: v.clone() for k, v in lin_sd.items()}
    full_sd["embed_tokens.weight"] = emb_w
    full_sd["embed_tokens.quant_op.weight_quantizer.s"] = emb_s
    # safetensors requires contiguous + supported dtypes (int64/fp32 ok)
    save_file(full_sd, str(tmp_path / "model.safetensors"))
    # dopt_config.json with q_proj strategy + ;4 suffix override (group_size=4 matches in_f=8)
    (tmp_path / "dopt_config.json").write_text(
        json.dumps(
            {
                "layer_strategy": {
                    "q_proj": {
                        "quant_strategy": "Quant_INT8-4_ECO;4",
                        "bit": 4,
                        "blocksize": 128,
                    }
                }
            }
        )
    )

    info, embed_quant = swap_from_dopt(model, tmp_path)

    assert isinstance(model.q_proj, QuantLinear)
    assert info["n_quantized"] == 1
    assert info["group_size"] == 4  # suffix ";4" overrides default 128
    assert info["bit"] == 4
    assert info["blocksize"] == 128
    assert "model.embed_tokens.weight" in embed_quant
    int8, scale = embed_quant["model.embed_tokens.weight"]
    assert int8.shape == (10, 8)
    assert scale.shape == (10,)
