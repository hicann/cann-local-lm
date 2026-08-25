"""Utilities for packing consolidated weights into binary files + index."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

DEFAULT_SPLIT_SIZE = 1024 * 1024 * 1024 * 50 # 50GB

@dataclasses.dataclass
class ConsolidatedWeights:
    """Packed consolidated weights in memory.

    All data is held in memory — use ``consolidated_weights_save()`` to write
    to disk when needed.

    Attributes:
        bin_files: List of packed weight binary data. Multiple files are
            created when tensor data exceeds ``max_file_size``.
        weights_index: Mapping from bin file name (e.g. ``"weights_0.bin"``)
            to a list of ``{"name": str, "offset": int, "size": int}`` entries
            describing each tensor's location within that file.
    """
    bin_files: list[bytes]
    weights_index: dict[str, list[dict]]


def pack_weights(contents, *, max_file_size: int = DEFAULT_SPLIT_SIZE) -> ConsolidatedWeights:
    """Pack weights from EpairContents into ConsolidatedWeights (in memory).

    Reads tensor data from ``contents.ep.state_dict()``, concatenates raw
    tensor blobs into one or more bin files, and builds a ``weights_index``
    mapping each bin file name to its tensor entries::

        {"weights_0.bin": [
            {"name": "<param_name>", "offset": <byte_offset>, "size": <bytes>}
        ]}

    Only tensors referenced in the ``ExportedProgram`` graph signature
    (parameters, buffers, and lifted tensor constants) are included.

    When the accumulated data in the current bin file would exceed
    ``max_file_size``, a new bin file is started. Individual tensors are
    never split across files — if a single tensor is larger than
    ``max_file_size``, it occupies its own file.

    Args:
        contents: EpairContents object containing model and weights.
        max_file_size: Maximum size (in bytes) of each bin file before
            splitting to the next file. Defaults to 5 KB for testing.

    Returns:
        ConsolidatedWeights with ``bin_files`` and ``weights_index``.
    """
    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)

    # Non-persistent buffers: add from ep.module().named_buffers() (uses module path names)
    for name, buf in contents.ep.module().named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    # Lifted tensor constants: add from ep.constants (uses ep_json tensor_constant_name)
    if hasattr(contents.ep, 'constants') and isinstance(contents.ep.constants, dict):
        for name, tensor in contents.ep.constants.items():
            if name not in state_dict and isinstance(tensor, torch.Tensor):
                state_dict[name] = tensor

    spec_names = set()
    gs = contents.ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for placeholder, logical_name in mapping.items():
            if logical_name in state_dict:
                spec_names.add(logical_name)

    names = [n for n in state_dict if n in spec_names]

    # Direct-to-disk writing (avoid accumulating all weights in memory)
    index: dict[str, list[dict]] = {}

    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    total_written = 0

    # Open first file for writing
    bin_path = None

    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()
        del raw  # Free tensor reference

        # If adding this tensor would exceed the limit, start a new file
        if current_offset > 0 and current_offset + len(data) > max_file_size:
            if bin_path is not None:
                bin_path.close()
            current_file_idx += 1
            current_file_name = f"weights_{current_file_idx}.bin"
            current_offset = 0

        # Open file if not yet open
        if current_offset == 0:
            import os
            bin_path = open(os.path.join(".", current_file_name), "wb")
            index[current_file_name] = []

        index[current_file_name].append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
        })

        bin_path.write(data)
        current_offset += len(data)
        total_written += len(data)
        del data  # Free bytes immediately

    # Close last file
    if bin_path is not None:
        bin_path.close()

    num_files = current_file_idx + 1
    print(f"Packed {len(index)} tensors into {num_files} file(s) ({total_written / 1024:.0f} KB)")


def consolidated_weights_save(contents, output_dir: str | Path, *,
                              save_weights: bool = True,
                              max_file_size: int = 5 * 1024,
                              ep_json_filename: str = "ep_json_with_ascend_ir.json",
                              host_side_weights: list[tuple[str, torch.Tensor]] | None = None) -> None:
    """Write epair contents to disk in consolidated weights format.

    Writes files directly under ``output_dir`` (no ``data/`` subdirectory):

    - ``weights_0.bin``, ``weights_1.bin``, ... — raw concatenated tensor bytes
    - ``weight_info.json`` — index describing each tensor and its bin file
    - ``<ep_json_filename>`` — annotated EP JSON (default ``ep_json_with_ascend_ir.json``)

    When ``host_side_weights`` is provided (pure-embeddings export: embedding
    lookups externalized from the graph), additionally writes:

    - ``host_weights_0.bin``, ... — host-side lookup weights (embed_tokens /
      gemma4 embed_tokens_per_layer), NOT graph inputs
    - ``host_weights_info.json`` — index with name/offset/size/dtype/shape

    These host-side files are independent of ``weight_info.json`` /
    ``weights_*.bin``; the C++ ``GeWeightLoader`` reads only the latter, so
    host-side packing has zero impact on the GE compile path.

    Weight files are written directly to disk (streaming, not in-memory)
    to support large models (>8B params) without OOM.

    Args:
        contents: EpairContents object containing model and weights.
        output_dir: Directory to write files into.
        save_weights: If False, skip writing ``weights_*.bin`` and
            ``weight_info.json``.
        max_file_size: Maximum size (in bytes) of each bin file before
            splitting. Defaults to 5 KB for testing.
        ep_json_filename: Filename for the annotated EP JSON. Defaults to
            ``ep_json_with_ascend_ir.json``. Use a custom name for
            prefill/decode split export (e.g.
            ``prefill_ep_json_with_ascend_ir.json``).
        host_side_weights: Optional list of ``(name, tensor)`` pairs for
            embedding-lookup weights externalized from the graph (pure-
            embeddings mode). Packed into separate ``host_weights_*.bin`` +
            ``host_weights_info.json`` for host-side lookup at inference.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_weights:
        weights_index = _pack_weights_to_dir(contents, output_dir, max_file_size)

        idx_path = output_dir / "weight_info.json"
        with open(idx_path, "w") as f:
            json.dump(weights_index, f, indent=2)
        print(f"Saved: {idx_path}")
    else:
        print("  [skip] weights_*.bin + weight_info.json (save_weights=False)")

    if host_side_weights:
        host_index = _pack_host_side_weights_to_dir(host_side_weights, output_dir, max_file_size)
        host_idx_path = output_dir / "host_weights_info.json"
        with open(host_idx_path, "w") as f:
            json.dump(host_index, f, indent=2)
        print(f"Saved: {host_idx_path}")

    ep_json_path = output_dir / ep_json_filename
    _strip_none_input_descs(contents.ep_json)
    with open(ep_json_path, "w") as f:
        json.dump(contents.ep_json, f, indent=2)
    print(f"Saved: {ep_json_path}")


def _strip_none_input_descs(ep_json: dict) -> None:
    """Remove inputDesc entries with dtype=None from QuantBatchMatmulV4 ascend_ir ops.

    QuantBatchMatmulV4's optional inputs (bias, y_scale, x1_offset, ...) are
    emitted as inputDesc with dtype=None when not provided. The C++ GE graph
    builder creates GE inputs for ALL inputDesc, including None-dtype ones.
    The InferShape pass then fails on the None bias (shape=[] → dimension mismatch).

    Only QuantBatchMatmulV4 is affected — other ops (Cast, Mul, etc.) must keep
    their inputDesc intact (stripping those breaks the C++ parser).
    """
    graph = ep_json.get("graph_module", {}).get("graph", {})
    for node in graph.get("nodes", []):
        ai = node.get("ascend_ir")
        if not isinstance(ai, dict):
            continue
        for op in ai.get("ops", []) or []:
            if not isinstance(op, dict):
                continue
            if op.get("name", "") != "QuantBatchMatmulV4":
                continue
            desc = op.get("inputDesc")
            inp = op.get("input")
            if isinstance(desc, list) and isinstance(inp, list) and len(desc) == len(inp):
                # Only strip entries that are BOTH dtype=None AND input ref is empty
                keep = [(d, r) for d, r in zip(desc, inp)
                        if not (d.get("dtype") is None and (r is None or r == ""))]
                op["inputDesc"] = [d for d, _ in keep]
                op["input"] = [r for _, r in keep]


def _pack_weights_to_dir(contents, output_dir: Path, max_file_size: int) -> dict[str, list[dict]]:
    """Pack weights from EpairContents directly to disk in output_dir.

    Streaming write — never accumulates more than one tensor in memory.
    """
    from cann_lm.converter.quant.materialize import transdata_scale

    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)

    # Non-persistent buffers: add from ep.module().named_buffers()
    for name, buf in contents.ep.module().named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    # Lifted tensor constants: add from ep.constants
    if hasattr(contents.ep, 'constants') and isinstance(contents.ep.constants, dict):
        for name, tensor in contents.ep.constants.items():
            if name not in state_dict and isinstance(tensor, torch.Tensor):
                state_dict[name] = tensor

    # Filter to only graph-signature-referenced tensors
    spec_names = set()
    gs = contents.ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for placeholder, logical_name in mapping.items():
            if logical_name in state_dict:
                spec_names.add(logical_name)

    names = [n for n in state_dict if n in spec_names]

    index: dict[str, list[dict]] = {}
    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    total_written = 0
    bin_fp = None

    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        elif "weight_scale" in name and raw.dtype == torch.float32:
            # 仅落盘时转 int64 (位模式 + & 0x00000000FFFFE000 掩码, 截断尾数低 13 位).
            # QuantLinear buffer 仍存 fp32 (计算用), ep_json dtype 由 inline_weight_scale_dtype
            # 后处理改标 DT_INT64, 保证 C++ 按 int64 读 (= uint64 位模式).
            raw = transdata_scale(raw)  # fp32 → int64
        data = raw.numpy().tobytes()
        del raw

        if current_offset > 0 and current_offset + len(data) > max_file_size:
            if bin_fp is not None:
                bin_fp.close()
            current_file_idx += 1
            current_file_name = f"weights_{current_file_idx}.bin"
            current_offset = 0

        if current_offset == 0:
            bin_fp = open(output_dir / current_file_name, "wb")
            index[current_file_name] = []

        index[current_file_name].append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
        })

        bin_fp.write(data)
        current_offset += len(data)
        total_written += len(data)
        del data

    if bin_fp is not None:
        bin_fp.close()

    print(f"Packed {len(index)} tensors into {current_file_idx + 1} file(s) "
          f"({total_written / 1024:.0f} KB)")
    return index


def _pack_host_side_weights_to_dir(
    host_side_weights: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    max_file_size: int,
) -> dict[str, list[dict]]:
    """Pack host-side lookup weights (externalized from the graph) to disk.

    Streaming + split-on-max_file_size logic mirrors _pack_weights_to_dir, but
    writes to ``host_weights_*.bin`` and records dtype+shape per entry (the host
    needs these to reconstruct the tensor, esp. bfloat16 stored as int16 bytes).

    These weights are NOT graph inputs (embed_tokens / gemma4 embed_tokens_per_layer
    lookups externalized in pure-embeddings mode); packed solely for host-side
    lookup at inference. Independent of weight_info.json / weights_*.bin.
    """
    output_dir = Path(output_dir)
    index: dict[str, list[dict]] = {}
    current_file_name = "host_weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    total_written = 0
    bin_fp = None

    for name, tensor in host_side_weights:
        shape = list(tensor.shape)
        dtype = tensor.dtype
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()
        del raw

        if current_offset > 0 and current_offset + len(data) > max_file_size:
            if bin_fp is not None:
                bin_fp.close()
            current_file_idx += 1
            current_file_name = f"host_weights_{current_file_idx}.bin"
            current_offset = 0

        if current_offset == 0:
            bin_fp = open(output_dir / current_file_name, "wb")
            index[current_file_name] = []

        index[current_file_name].append({
            "name": name,
            "offset": current_offset,
            "size": len(data),
            "dtype": str(dtype).split(".")[-1],
            "shape": shape,
        })

        bin_fp.write(data)
        current_offset += len(data)
        total_written += len(data)
        del data

    if bin_fp is not None:
        bin_fp.close()

    print(f"Packed {len(host_side_weights)} host-side weight(s) into "
          f"{current_file_idx + 1} file(s) ({total_written / 1024:.0f} KB)")
    return index


def inline_fused_attn_seq_constants(ep_json_path, qlen_value: int, kvlen_value: int | None = None) -> int:
    """Inline ``FusedInferAttentionScore`` actual_seq_qlen/kvlen as ``ge.Const`` in ep_json.

    prefill-decode 共享 ``weights_0.bin``: decode 的 actual_seq lifted 常量会按同名复用 prefill
    的值 (seq_len) 而非自己的, 且 Data 引用输入被 GE 标为 tilingDepend -> PartitionedCall
    (动态子图). 把 actual_seq_qlen/kvlen 内联成真正的 ``ge.Const`` 既修正逐图取值, 又让算子
    保持静态 (无 PartitionedCall).

    - ``qlen_value``: actual_seq_qlen = query 长度 (prefill=seq_len, decode=1).
    - ``kvlen_value``: actual_seq_kvlen = kv 有效长度. 默认 None -> 用 qlen_value (prefill
      无 past, kvlen=qlen=seq_len 正确). **decode 须显式传 kv_cache_max_len**: decode 的 valid
      kv = past_len+1 (随步动态, 静态 ge.Const 无法表达); 用 kv_max 让 kernel 读全 buffer,
      实际 attend 范围由显式 atten_mask 决定. (kvlen=1 + mask 仍会被静态 Const 截断到 1 个 kv
      丢上下文, 详见 docs/.../ops_replacement.md "FIA 回退" 节根因分析.)
    """
    import base64
    import struct

    if kvlen_value is None:
        kvlen_value = qlen_value

    dim = ["1"]

    def _make_const(name: str, value: int) -> dict:
        data = base64.b64encode(struct.pack("<q", int(value))).decode()
        readable = base64.b64encode(f"[{int(value)}]".encode()).decode()
        return {
            "name": name,
            "type": "Const",
            "attr": {
                "value": {"t": {"desc": {"dtype": "DT_INT64", "shape": {"dim": dim}, "layout": "ND"},
                                  "data": data}},
                "_readable_value": {"s": readable},
            },
            "outputDesc": [{"dtype": "DT_INT64", "shape": {"dim": dim}, "layout": "ND"}],
        }

    with open(ep_json_path) as f:
        doc = json.load(f)
    nodes = doc["graph_module"]["graph"]["nodes"]
    inlined = 0
    for n in nodes:
        ops = (n.get("ascend_ir") or {}).get("ops") or []
        for op in ops:
            if op.get("type") != "FusedInferAttentionScore":
                continue
            ins = op.get("input")
            if not isinstance(ins, list) or len(ins) < 7 or not ins[5]:
                continue  # qlen (ins[5]) 必须有; kvlen (ins[6]) 可空 (kvlen=None 新方案: op 读全 buffer, 由 atten_mask 控制有效区段)
            qlen_c = _make_const(f"InlineSeqQLen{inlined}", qlen_value)
            ops.append(qlen_c)
            ins[5] = f"{qlen_c['name']}:0"
            if ins[6]:  # kvlen 存在 (旧 kvlen=常量方案) 才内联; None/空 (新方案) 跳过
                kvlen_c = _make_const(f"InlineSeqKvlen{inlined}", kvlen_value)
                ops.append(kvlen_c)
                ins[6] = f"{kvlen_c['name']}:0"
            inlined += 1
    with open(ep_json_path, "w") as f:
        json.dump(doc, f)
    return inlined


def inline_weight_scale_dtype(ep_json_path) -> int:
    """把 weight_scale 的 tensor_values dtype 从 float32(7) 改标 int64(5) (落盘格式).

    QuantLinear buffer 存 fp32 (计算用, forward 直接传 npu_quant_matmul), 但 saver
    ``_pack_weights_to_dir`` 打包时调 transdata_scale 转 int64 (位模式 + 掩码).
    需同步改 ep_json 的 tensor_values dtype 为 DT_INT64 (pt2 kDtypeLong=5), 否则
    C++ 按 DT_FLOAT 读 int64 bytes 会 WEIGHT_SIZE_MISMATCH (4B/元素 vs 实际 8B).

    Returns: 改标的 weight_scale 数量.
    """
    with open(ep_json_path) as f:
        doc = json.load(f)
    tv = doc.get("graph_module", {}).get("graph", {}).get("tensor_values", {})
    inlined = 0
    for name, meta in tv.items():
        if "weight_scale" not in name:
            continue
        if isinstance(meta, dict) and meta.get("dtype") == 7:  # 7=float32 (kDtypeFloat)
            meta["dtype"] = 5  # 5=int64 (kDtypeLong)
            inlined += 1
    with open(ep_json_path, "w") as f:
        json.dump(doc, f)
    return inlined
