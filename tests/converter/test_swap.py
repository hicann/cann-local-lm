"""swap.py 单元测试 — 纯逻辑函数 (group_size 推断 / parent 解析 / dopt 产物解析).

补充 tests/dopt_lm/test_strategy_generation.py 未覆盖部分:
- _infer_group_size (pergroup/perchannel/pertensor 拒绝/维度校验)
- _resolve_parent (nn.Module 树定位 parent+leaf name)
- load_dopt_state_dict (.pth 加载 / language_model. 前缀剥离 / dir 查找 model.safetensors)
- _read_dopt_quant_meta (bit/unsigned_quant/offset 读取 + 缺失 fallback)
- _read_quant_param_map 边缘 (None 路径 / 无 dopt_config.json / 未知 strategy / 非法 suffix)
跳过: swap_from_dopt / _swap_linear_from_sd / swap_embeddings_from_dopt 全流程
(需完整 dopt artifacts + 真实 model tree, 太重).
"""
import json

import pytest
import torch
import torch.nn as nn

from cann_lm.converter.swap import (
    _infer_group_size,
    _read_dopt_quant_meta,
    _read_quant_param_map,
    _resolve_parent,
    load_dopt_state_dict,
)


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
    (tmp_path / "dopt_config.json").write_text(json.dumps(
        {"layer_strategy": {"proj": {"quant_strategy": "Quant_Unknown"}}}
    ))
    assert _read_quant_param_map(tmp_path) == {}  # 未知 strategy 无默认 → 跳过


def test_read_quant_param_map_invalid_suffix_skipped(tmp_path):
    (tmp_path / "dopt_config.json").write_text(json.dumps(
        {"layer_strategy": {"proj": {"quant_strategy": "Quant_INT8-4_ECO;abc"}}}
    ))
    assert _read_quant_param_map(tmp_path) == {}  # ";abc" 非法 int → 跳过


def test_read_quant_param_map_layer_field_overrides_default(tmp_path):
    (tmp_path / "dopt_config.json").write_text(json.dumps(
        {"layer_strategy": {"proj": {
            "quant_strategy": "Quant_INT8-4_ECO",
            "bit": 8,  # 覆盖默认 bit=4
            "blocksize": 64,
        }}}
    ))
    params = _read_quant_param_map(tmp_path)["proj"]
    assert params["bit"] == 8
    assert params["blocksize"] == 64
    assert params["group_size"] == 128  # 默认 (无 suffix 覆盖)


# ── load_dopt_state_dict (.pth / language_model. 剥离 / dir / not found) ──

def test_load_dopt_state_dict_pth(tmp_path):
    sd = {"model.layers.0.weight": torch.zeros(2, 2),
          "model.layers.1.weight": torch.ones(2, 2)}
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
