"""Utilities for packing consolidated weights into binary files + index."""

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

    # _is_weight: True for nn.Module state (parameters + buffers, shared by prefill/decode),
    # False for lifted tensor constants (trace-time graph constants). C++ GeWeightLoader
    # reads this to populate TensorRecordInfo.is_parameter; fallback to input_specs
    # derivation when the field is absent (back-compat with old weight_info.json).
    is_weight_of: dict[str, bool] = {}
    gs = contents.ep.graph_signature
    for logical_name in gs.inputs_to_parameters.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_buffers.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_lifted_tensor_constants.values():
        is_weight_of[logical_name] = False

    names = [n for n in state_dict if n in is_weight_of]

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
            "_is_weight": is_weight_of[name],
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
    - ``weight_info.json`` — index describing each tensor and its bin file.
      Each entry has: ``name``/``offset``/``size``/``format``/``_is_weight``.
      ``_is_weight`` is True for ``nn.Module`` state (parameters + buffers,
      shared across prefill/decode) and False for lifted tensor constants
      (trace-time graph constants). The C++ ``GeWeightLoader`` reads this to
      populate ``TensorRecordInfo.is_parameter``; falls back to
      ``input_specs`` derivation when absent (back-compat).
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
        weights_index = _pack_weights_v2(contents, output_dir, "single", max_file_size)

        idx_path = output_dir / "weight_info.json"
        with open(idx_path, "w") as f:
            json.dump(weights_index, f, indent=2)
        print(f"Saved: {idx_path} (v2)")
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


def _blob_hash(data: bytes) -> str:
    """Content hash of tensor bytes (sha256 truncated to 64-bit)."""
    import hashlib
    return "h_" + hashlib.sha256(data).hexdigest()[:16]


def _extract_state_dict(contents) -> tuple[dict, dict[str, bool]]:
    """Extract state_dict (params+buffers+lifted constants) + is_weight_of map from EpairContents."""
    raw_sd = contents.ep.state_dict() if callable(contents.ep.state_dict) else contents.ep.state_dict
    state_dict = dict(raw_sd)
    for name, buf in contents.ep.module().named_buffers():
        if name not in state_dict:
            state_dict[name] = buf
    if hasattr(contents.ep, 'constants') and isinstance(contents.ep.constants, dict):
        for name, tensor in contents.ep.constants.items():
            if name not in state_dict and isinstance(tensor, torch.Tensor):
                state_dict[name] = tensor
    is_weight_of: dict[str, bool] = {}
    gs = contents.ep.graph_signature
    for logical_name in gs.inputs_to_parameters.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_buffers.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_lifted_tensor_constants.values():
        is_weight_of[logical_name] = False
    return state_dict, is_weight_of


def _pack_weights_pd_dedup(
    prefill_contents, decode_contents, output_dir: Path, max_file_size: int,
) -> dict:
    """Pack PD two graphs' weights with content-hash dedup → v2 two-level index.

    Same bytes → same hash → shared blob (dedup); different bytes → different hash → each stored.
    Returns v2 index: {blobs:{hash:{bin,offset,size,format,_is_weight}}, graphs:{prefill:{name:hash}, decode:{name:hash}}} (无 version 字段, C++ 检测 graphs/blobs 存在).
    Bin write + split logic mirrors _pack_weights_to_dir (single tensor not split, cwd bin path).
    """
    import os
    output_dir = Path(output_dir).resolve()
    blobs: dict[str, dict] = {}
    graphs: dict[str, dict[str, str]] = {"prefill": {}, "decode": {}}
    seen: set[str] = set()

    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    bin_path = None

    for graph_name, contents in (("prefill", prefill_contents), ("decode", decode_contents)):
        state_dict, is_weight_of = _extract_state_dict(contents)
        names = [n for n in state_dict if n in is_weight_of]
        for name in names:
            tensor = state_dict[name]
            raw = tensor.detach().cpu().contiguous()
            if raw.dtype == torch.bfloat16:
                raw = raw.view(torch.int16)
            data = raw.numpy().tobytes()
            del raw
            h = _blob_hash(data)
            graphs[graph_name][name] = h
            if h not in seen:
                if current_offset > 0 and current_offset + len(data) > max_file_size:
                    if bin_path is not None:
                        bin_path.close()
                    current_file_idx += 1
                    current_file_name = f"weights_{current_file_idx}.bin"
                    current_offset = 0
                if current_offset == 0:
                    bin_path = open(os.path.join(str(output_dir), current_file_name), "wb")
                bin_path.write(data)
                blobs[h] = {
                    "bin": current_file_name,
                    "offset": current_offset,
                    "size": len(data),
                    "format": "nd",
                    "_is_weight": is_weight_of[name],
                }
                current_offset += len(data)
                seen.add(h)
            del data

    if bin_path is not None:
        bin_path.close()

    n_prefill = len(graphs["prefill"])
    n_decode = len(graphs["decode"])
    print(f"PD dedup packed: {len(blobs)} unique blobs "
          f"(prefill {n_prefill} / decode {n_decode} tensors, "
          f"{n_prefill + n_decode - len(blobs)} shared)")
    return {"blobs": blobs, "graphs": graphs}


def _pack_weights_v2(
    contents, output_dir: Path, graph_name: str = "single", max_file_size: int = 5 * 1024,
) -> dict:
    """Pack single-graph weights with content-hash dedup → v2 two-level index.

    Same logic as _pack_weights_pd_dedup but for single-graph export (consolidated_weights_save).
    Returns {blobs:{hash:{bin,offset,size,format,_is_weight}}, graphs:{graph_name:{name→hash}}}.
    """
    import os
    output_dir = Path(output_dir).resolve()
    blobs: dict[str, dict] = {}
    graphs: dict[str, dict[str, str]] = {graph_name: {}}
    seen: set[str] = set()
    current_file_name = "weights_0.bin"
    current_file_idx = 0
    current_offset = 0
    bin_path = None

    state_dict, is_weight_of = _extract_state_dict(contents)
    names = [n for n in state_dict if n in is_weight_of]
    for name in names:
        tensor = state_dict[name]
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        data = raw.numpy().tobytes()
        del raw
        h = _blob_hash(data)
        graphs[graph_name][name] = h
        if h not in seen:
            if current_offset > 0 and current_offset + len(data) > max_file_size:
                if bin_path is not None:
                    bin_path.close()
                current_file_idx += 1
                current_file_name = f"weights_{current_file_idx}.bin"
                current_offset = 0
            if current_offset == 0:
                bin_path = open(os.path.join(str(output_dir), current_file_name), "wb")
            bin_path.write(data)
            blobs[h] = {
                "bin": current_file_name,
                "offset": current_offset,
                "size": len(data),
                "format": "nd",
                "_is_weight": is_weight_of[name],
            }
            current_offset += len(data)
            seen.add(h)
        del data

    if bin_path is not None:
        bin_path.close()

    print(f"v2 packed: {len(blobs)} unique blobs, {len(graphs[graph_name])} tensors (graph={graph_name})")
    return {"blobs": blobs, "graphs": graphs}


def consolidated_weights_save_pd(
    prefill_epair, decode_epair, output_dir: str | Path, *,
    save_weights: bool = True,
    max_file_size: int = 5 * 1024,
    host_side_weights: list[tuple[str, torch.Tensor]] | None = None,
) -> None:
    """PD split export with content-hash dedup (CLI-014 解法 B).

    Writes v2 two-level ``weight_info.json`` (blobs + graphs) + ``weights_*.bin``
    (hash-deduped: same bytes shared across PD, different bytes each stored) +
    PD two ep_json + optional host_weights.

    v2 schema: ``{version:2, blobs:{hash:{bin,offset,size,format,_is_weight}},
    graphs:{prefill:{name:hash}, decode:{name:hash}}}``.
    C++ ``GeWeightLoader`` v2 path: ``graphs[graph][name]→hash→blobs[hash]``.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_weights:
        index = _pack_weights_pd_dedup(prefill_epair, decode_epair, output_dir, max_file_size)
        idx_path = output_dir / "weight_info.json"
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Saved: {idx_path} (v2 hash-dedup)")
    else:
        print("  [skip] weights_*.bin + weight_info.json (save_weights=False)")

    if host_side_weights:
        host_index = _pack_host_side_weights_to_dir(host_side_weights, output_dir, max_file_size)
        host_idx_path = output_dir / "host_weights_info.json"
        with open(host_idx_path, "w") as f:
            json.dump(host_index, f, indent=2)
        print(f"Saved: {host_idx_path}")

    for epair, ep_json_filename in (
        (prefill_epair, "prefill_ep_json_with_ascend_ir.json"),
        (decode_epair, "decode_ep_json_with_ascend_ir.json"),
    ):
        ep_json_path = output_dir / ep_json_filename
        _strip_none_input_descs(epair.ep_json)
        with open(ep_json_path, "w") as f:
            json.dump(epair.ep_json, f, indent=2)
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
    from cann_lm.converter.export.quant_swap.materialize import transdata_scale

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

    # _is_weight: True for nn.Module state (parameters + buffers, shared by prefill/decode),
    # False for lifted tensor constants (trace-time graph constants). C++ GeWeightLoader
    # reads this to populate TensorRecordInfo.is_parameter; fallback to input_specs
    # derivation when the field is absent (back-compat with old weight_info.json).
    is_weight_of: dict[str, bool] = {}
    gs = contents.ep.graph_signature
    for logical_name in gs.inputs_to_parameters.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_buffers.values():
        is_weight_of[logical_name] = True
    for logical_name in gs.inputs_to_lifted_tensor_constants.values():
        is_weight_of[logical_name] = False

    names = [n for n in state_dict if n in is_weight_of]

    import os
    trans_enabled = os.environ.get("CANN_LM_ENABLE_WEIGHT_TRANS") == "1"
    rules = None
    if trans_enabled:
        from cann_lm.converter.export.weight_format import load_rules, resolve_format, nd2nz
        rules = load_rules()
    is_quantized = bool(trans_enabled and any("weight_scale" in n for n in names))

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
            raw = transdata_scale(raw)  # fp32->int64; ep_json dtype relabeled to DT_INT64 by inline_weight_scale_dtype
        # NZ layout transform (orthogonal to the dtype view above; env off -> all ND)
        entry_format = "ND"
        if trans_enabled and raw.dim() == 2:
            rule = resolve_format(name, is_quantized, rules=rules)
            if rule.get("format") == "NZ":
                block = rule.get("block", 16)
                if raw.shape[0] % block != 0 or raw.shape[1] % block != 0:
                    # Non-aligned shapes can't use NZ in GE graph compilation path:
                    # CheckTensorIsValid checks data_size == Product(shape) * elem_size,
                    # which fails for NZ-padded bytes of non-aligned shapes. Skip NZ, keep ND.
                    pass
                else:
                    raw = nd2nz(raw, block=block, pad=rule.get("pad", True))
                    entry_format = "NZ"
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
            "format": entry_format,
            "_is_weight": is_weight_of[name],
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
    """Inline FusedInferAttentionScore actual_seq_qlen/kvlen as ge.Const in ep_json.

    prefill-decode shares weights_0.bin: lifted actual_seq constants would be reused by name
    across graphs and marked tilingDepend -> PartitionedCall. Inlining as ge.Const fixes
    per-graph values and keeps the op static.

    decode must pass kvlen=kv_cache_max_len (valid kv = past_len+1, dynamic; let the kernel
    read the full buffer, atten_mask decides the attend range). See ops_replacement.md "FIA fallback".
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
                continue  # qlen (ins[5]) must exist; kvlen (ins[6]) may be empty (kvlen=None new scheme: op reads full buffer, atten_mask controls the valid range)
            qlen_c = _make_const(f"InlineSeqQLen{inlined}", qlen_value)
            ops.append(qlen_c)
            ins[5] = f"{qlen_c['name']}:0"
            if ins[6]:  # inline kvlen only if present (old kvlen=const scheme); None/empty (new scheme) -> skip
                kvlen_c = _make_const(f"InlineSeqKvlen{inlined}", kvlen_value)
                ops.append(kvlen_c)
                ins[6] = f"{kvlen_c['name']}:0"
            inlined += 1
    with open(ep_json_path, "w") as f:
        json.dump(doc, f)
    return inlined


def inline_weight_scale_dtype(ep_json_path) -> int:
    """weight_scale tensor_values dtype float32(7) -> int64(5).

    saver packs int64 bytes; the ep_json dtype must be synced, else C++ reads as DT_FLOAT
    and hits WEIGHT_SIZE_MISMATCH.
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
