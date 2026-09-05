"""epair_utils.py 单元测试 — 权重打包/切分 + ep_json 原地改写 (纯文件/tensor ops).

测: ConsolidatedWeights dataclass 字段, _pack_weights_to_dir 按 max_file_size 切分逻辑
(小 max_file_size 触发多分片/bf16→int16/weight_scale→int64 transdata/单 tensor 超限占独立文件),
consolidated_weights_save 写 weights_*.bin + weight_info.json + ep_json, _strip_none_input_descs
清理 QuantBatchMatmulV4 的 None inputDesc, inline_fused_attn_seq_constants/
inline_weight_scale_dtype 原地改写 ep_json.
跳过: pack_weights (遗留函数: 写 cwd "." 且不返回 index, 已被 _pack_weights_to_dir 取代;
      仍用 monkeypatch.chdir 轻测落盘); 需真实 EpairContents/ExportedProgram/torchair 的路径.
"""
import json

import pytest
import torch
import torch.nn as nn

from cann_lm.converter.export.epair_utils import (
    ConsolidatedWeights,
    _pack_weights_to_dir,
    _strip_none_input_descs,
    consolidated_weights_save,
    inline_fused_attn_seq_constants,
    inline_weight_scale_dtype,
    pack_weights,
)


# ── fake EpairContents (无 ExportedProgram, 纯 dict/Module stub) ──

class _FakeGraphSignature:
    """graph_signature stub: 把每个 logical_name 映射到自身 (placeholder==logical_name)."""

    def __init__(self, names):
        self.inputs_to_parameters = {n: n for n in names}
        self.inputs_to_buffers = {}
        self.inputs_to_lifted_tensor_constants = {}


class _FakeEP:
    """ExportedProgram stub: state_dict() 返回 dict, module() 返回空 Module (无额外 buffer)."""

    def __init__(self, sd):
        self._sd = sd
        self.graph_signature = _FakeGraphSignature(list(sd))

    def state_dict(self):
        return self._sd

    def module(self):
        return nn.Module()


class _FakeContents:
    """EpairContents stub (ep + ep_json)."""

    def __init__(self, sd, ep_json=None):
        self.ep = _FakeEP(sd)
        self.ep_json = ep_json if ep_json is not None else {}


# ── ConsolidatedWeights dataclass ──

def test_consolidated_weights_dataclass_fields():
    cw = ConsolidatedWeights(bin_files=[b"\x00\x01"], weights_index={"weights_0.bin": []})
    assert cw.bin_files == [b"\x00\x01"]
    assert cw.weights_index == {"weights_0.bin": []}


# ── _pack_weights_to_dir: 单文件 ──

def test_pack_weights_to_dir_single_file(tmp_path):
    sd = {"a.weight": torch.zeros(4, 4, dtype=torch.float32),
          "b.weight": torch.ones(2, 2, dtype=torch.float32)}
    contents = _FakeContents(sd)

    index = _pack_weights_to_dir(contents, tmp_path, max_file_size=100_000)

    assert list(index) == ["weights_0.bin"]
    assert (tmp_path / "weights_0.bin").exists()
    # a.weight: 4*4*4=64 bytes @ offset 0; b.weight: 2*2*4=16 bytes @ offset 64
    assert index["weights_0.bin"] == [
        {"name": "a.weight", "offset": 0, "size": 64},
        {"name": "b.weight", "offset": 64, "size": 16},
    ]
    assert (tmp_path / "weights_0.bin").stat().st_size == 80


# ── _pack_weights_to_dir: 按 max_file_size 切分 ──

def test_pack_weights_to_dir_splits_on_max_file_size(tmp_path):
    # a.weight=64 bytes, b.weight=16 bytes, total 80; max=70 → a 进文件0, b 触发新文件1
    sd = {"a.weight": torch.zeros(4, 4, dtype=torch.float32),
          "b.weight": torch.ones(2, 2, dtype=torch.float32)}
    contents = _FakeContents(sd)

    index = _pack_weights_to_dir(contents, tmp_path, max_file_size=70)

    assert sorted(index) == ["weights_0.bin", "weights_1.bin"]
    assert (tmp_path / "weights_0.bin").stat().st_size == 64
    assert (tmp_path / "weights_1.bin").stat().st_size == 16
    assert index["weights_0.bin"][0] == {"name": "a.weight", "offset": 0, "size": 64}
    assert index["weights_1.bin"][0] == {"name": "b.weight", "offset": 0, "size": 16}


def test_pack_weights_to_dir_single_tensor_exceeds_limit_not_split(tmp_path):
    # 单个 tensor (400 bytes) 大于 max_file_size=70 → 占独立文件, 不跨文件切分
    sd = {"big.weight": torch.zeros(100, dtype=torch.float32)}
    contents = _FakeContents(sd)

    index = _pack_weights_to_dir(contents, tmp_path, max_file_size=70)

    assert list(index) == ["weights_0.bin"]
    assert index["weights_0.bin"][0]["size"] == 400
    assert (tmp_path / "weights_0.bin").stat().st_size == 400


def test_pack_weights_to_dir_bfloat16_viewed_as_int16(tmp_path):
    # bf16 tensor 通过 raw.view(int16) 落盘: 4 bf16 → 4*2=8 bytes
    sd = {"x.weight": torch.zeros(4, dtype=torch.bfloat16)}
    contents = _FakeContents(sd)

    index = _pack_weights_to_dir(contents, tmp_path, max_file_size=100_000)

    assert index["weights_0.bin"][0]["size"] == 8


def test_pack_weights_to_dir_weight_scale_transdata_to_int64(tmp_path):
    # "weight_scale" + fp32 → transdata_scale → int64 (8 bytes/elem); 4 fp32 → 32 bytes
    sd = {"layer.weight_scale": torch.ones(4, dtype=torch.float32)}
    contents = _FakeContents(sd)

    index = _pack_weights_to_dir(contents, tmp_path, max_file_size=100_000)

    assert index["weights_0.bin"][0]["size"] == 32  # 4 * int64(8)


# ── consolidated_weights_save: 落盘 weights_*.bin + weight_info.json + ep_json ──

def test_consolidated_weights_save_writes_all_files(tmp_path):
    sd = {"a.weight": torch.zeros(4, 4, dtype=torch.float32)}
    ep_json = {"graph_module": {"graph": {"nodes": []}}}
    contents = _FakeContents(sd, ep_json=ep_json)

    consolidated_weights_save(contents, tmp_path, max_file_size=100_000,
                             ep_json_filename="ep.json")

    assert (tmp_path / "weights_0.bin").exists()
    info = json.loads((tmp_path / "weight_info.json").read_text())
    assert info["weights_0.bin"][0]["name"] == "a.weight"
    assert (tmp_path / "ep.json").exists()
    assert json.loads((tmp_path / "ep.json").read_text()) == ep_json


def test_consolidated_weights_save_skip_weights_only_ep_json(tmp_path):
    contents = _FakeContents({}, ep_json={"graph_module": {"graph": {"nodes": []}}})

    consolidated_weights_save(contents, tmp_path, save_weights=False,
                             ep_json_filename="ep.json")

    assert not (tmp_path / "weights_0.bin").exists()
    assert not (tmp_path / "weight_info.json").exists()
    assert (tmp_path / "ep.json").exists()


def test_consolidated_weights_save_strips_none_input_descs_in_ep_json(tmp_path):
    # ep_json 含一个 QuantBatchMatmulV4 op, inputDesc 有 None dtype + 空 input ref
    ep_json = {
        "graph_module": {"graph": {"nodes": [
            {"ascend_ir": {"ops": [
                {"name": "QuantBatchMatmulV4", "type": "QuantBatchMatmulV4",
                 "inputDesc": [{"dtype": "DT_INT8"}, {"dtype": None}, {"dtype": None}],
                 "input": ["x", "", ""]},
            ]}},
        ]}},
    }
    contents = _FakeContents({}, ep_json=ep_json)

    consolidated_weights_save(contents, tmp_path, save_weights=False,
                             ep_json_filename="ep.json")

    op = (json.loads((tmp_path / "ep.json").read_text())
          ["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0])
    assert op["inputDesc"] == [{"dtype": "DT_INT8"}]
    assert op["input"] == ["x"]


# ── _strip_none_input_descs ──

def _qbm_op(input_desc, inp):
    return {"name": "QuantBatchMatmulV4", "type": "QuantBatchMatmulV4",
            "inputDesc": input_desc, "input": inp}


def test_strip_none_input_descs_removes_none_dtype_and_empty_ref():
    ep_json = {"graph_module": {"graph": {"nodes": [
        {"ascend_ir": {"ops": [_qbm_op(
            [{"dtype": "DT_INT8"}, {"dtype": None}, {"dtype": None}],
            ["x", "", ""],
        )]}},
    ]}}}
    _strip_none_input_descs(ep_json)
    op = ep_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]
    assert op["inputDesc"] == [{"dtype": "DT_INT8"}]
    assert op["input"] == ["x"]


def test_strip_none_input_descs_keeps_nonempty_ref_even_if_dtype_none():
    # keep 条件: not (dtype is None AND (ref is None or ref=="")) → dtype=None 但 ref="y" 保留
    ep_json = {"graph_module": {"graph": {"nodes": [
        {"ascend_ir": {"ops": [_qbm_op(
            [{"dtype": None}, {"dtype": None}],
            ["y", ""],
        )]}},
    ]}}}
    _strip_none_input_descs(ep_json)
    op = ep_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]
    assert op["inputDesc"] == [{"dtype": None}]
    assert op["input"] == ["y"]


def test_strip_none_input_descs_ignores_non_quant_ops():
    # name != "QuantBatchMatmulV4" 的 op 不处理 (Cast 等 inputDesc 必须保持原样)
    ep_json = {"graph_module": {"graph": {"nodes": [
        {"ascend_ir": {"ops": [
            {"name": "Cast", "type": "Cast",
             "inputDesc": [{"dtype": None}], "input": [""]},
        ]}},
    ]}}}
    _strip_none_input_descs(ep_json)
    op = ep_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]
    assert op["inputDesc"] == [{"dtype": None}]


def test_strip_none_input_descs_skips_when_length_mismatch():
    # len(inputDesc) != len(input) → 不修改
    ep_json = {"graph_module": {"graph": {"nodes": [
        {"ascend_ir": {"ops": [_qbm_op(
            [{"dtype": None}, {"dtype": None}],
            ["x"],  # 长度不一致
        )]}},
    ]}}}
    _strip_none_input_descs(ep_json)
    op = ep_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]
    assert len(op["inputDesc"]) == 2  # 原样未动


# ── pack_weights (遗留: 写 cwd, 不返回 index) ──

def test_pack_weights_writes_to_cwd_and_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    contents = _FakeContents({"a.weight": torch.zeros(4, 4, dtype=torch.float32)})

    result = pack_weights(contents, max_file_size=100_000)

    assert result is None  # 遗留函数不返回 ConsolidatedWeights
    assert (tmp_path / "weights_0.bin").exists()


def test_pack_weights_splits_on_max_file_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sd = {"a.weight": torch.zeros(4, 4, dtype=torch.float32),
          "b.weight": torch.ones(2, 2, dtype=torch.float32)}
    pack_weights(_FakeContents(sd), max_file_size=70)
    assert (tmp_path / "weights_0.bin").stat().st_size == 64
    assert (tmp_path / "weights_1.bin").stat().st_size == 16


# ── inline_fused_attn_seq_constants ──

def _fia_op(input_list):
    return {"type": "FusedInferAttentionScore", "input": input_list}


def _fia_ep_json(input_list):
    return {"graph_module": {"graph": {"nodes": [
        {"ascend_ir": {"ops": [_fia_op(input_list)]}},
    ]}}}


def test_inline_fused_attn_seq_constants_inlines_qlen_and_kvlen(tmp_path):
    p = tmp_path / "ep.json"
    # input[5]=qlen ref (truthy), input[6]=kvlen ref (truthy)
    inp = ["a", "b", "c", "d", "e", "qlen_ref", "kvlen_ref"]
    p.write_text(json.dumps(_fia_ep_json(inp)))

    n = inline_fused_attn_seq_constants(p, qlen_value=128, kvlen_value=256)

    assert n == 1
    doc = json.loads(p.read_text())
    ops = doc["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"]
    # 原 FIA op 的 input[5]/[6] 被改写为内联 Const 引用
    fia = ops[0]
    assert fia["input"][5].startswith("InlineSeqQLen")
    assert fia["input"][6].startswith("InlineSeqKvlen")
    # 追加了 2 个 Const op (qlen + kvlen)
    consts = [o for o in ops if o.get("type") == "Const"]
    assert len(consts) == 2
    assert consts[0]["outputDesc"][0]["dtype"] == "DT_INT64"


def test_inline_fused_attn_seq_constants_skips_empty_kvlen(tmp_path):
    p = tmp_path / "ep.json"
    inp = ["a", "b", "c", "d", "e", "qlen_ref", ""]  # input[6] 空 → 不内联 kvlen
    p.write_text(json.dumps(_fia_ep_json(inp)))

    n = inline_fused_attn_seq_constants(p, qlen_value=8)

    assert n == 1
    doc = json.loads(p.read_text())
    fia = doc["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]
    assert fia["input"][5].startswith("InlineSeqQLen")
    assert fia["input"][6] == ""  # kvlen 未改写
    consts = [o for o in doc["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"]
              if o.get("type") == "Const"]
    assert len(consts) == 1  # 只追加 qlen Const


def test_inline_fused_attn_seq_constants_skips_op_without_qlen(tmp_path):
    p = tmp_path / "ep.json"
    inp = ["a", "b", "c", "d", "e", "", "kvlen_ref"]  # input[5] 空 → 整体跳过
    p.write_text(json.dumps(_fia_ep_json(inp)))

    n = inline_fused_attn_seq_constants(p, qlen_value=8)

    assert n == 0
    doc = json.loads(p.read_text())
    ops = doc["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"]
    assert len(ops) == 1  # 无 Const 追加


def test_inline_fused_attn_seq_constants_kvlen_defaults_to_qlen(tmp_path):
    p = tmp_path / "ep.json"
    inp = ["a", "b", "c", "d", "e", "qlen_ref", "kvlen_ref"]
    p.write_text(json.dumps(_fia_ep_json(inp)))

    inline_fused_attn_seq_constants(p, qlen_value=64)  # kvlen_value=None → =qlen

    doc = json.loads(p.read_text())
    import base64
    import struct
    consts = [o for o in doc["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"]
              if o.get("type") == "Const"]
    qlen_val = struct.unpack("<q", base64.b64decode(consts[0]["attr"]["value"]["t"]["data"]))[0]
    kvlen_val = struct.unpack("<q", base64.b64decode(consts[1]["attr"]["value"]["t"]["data"]))[0]
    assert qlen_val == 64 and kvlen_val == 64


# ── inline_weight_scale_dtype ──

def test_inline_weight_scale_dtype_relabeled_int64(tmp_path):
    p = tmp_path / "ep.json"
    doc = {"graph_module": {"graph": {"tensor_values": {
        "model.layers.0.weight_scale": {"dtype": 7},   # 7=fp32 → 改 5=int64
        "model.layers.0.weight": {"dtype": 7},         # 无 "weight_scale" → 不改
        "model.layers.1.weight_scale": {"dtype": 5},    # 已是 int64 → 不改
    }}}}
    p.write_text(json.dumps(doc))

    n = inline_weight_scale_dtype(p)

    assert n == 1  # 只改了第一个 (dtype 7→5)
    tv = json.loads(p.read_text())["graph_module"]["graph"]["tensor_values"]
    assert tv["model.layers.0.weight_scale"]["dtype"] == 5
    assert tv["model.layers.0.weight"]["dtype"] == 7
    assert tv["model.layers.1.weight_scale"]["dtype"] == 5


def test_inline_weight_scale_dtype_no_tensor_values(tmp_path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps({"graph_module": {"graph": {}}}))
    assert inline_weight_scale_dtype(p) == 0
