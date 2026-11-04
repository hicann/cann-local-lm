# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile Qwen3.5 PT2 methods to one Ascend PTE."""

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


import ctypes


_PROTO_HANDLES = []


def register_chunk_gdn_converter(proto_library: Path | None = None) -> None:
    import torch
    import torch_npu  # noqa: F401

    # _get_converter lazily imports this whole package for ops such as
    # scatter_update_. Import it first so it cannot later replace our adapter
    # with the installed unsupported ChunkGDR stub mid-graph.
    import torchair._ge_concrete_graph.ge_converter.custom  # noqa: F401
    from torchair import ge
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    # The caller opts in to this CANN 9.1 registration shim. Do not query
    # get_registered_ir_def here: its TorchAir session is initialized later,
    # during lowering. The converter's ge.custom_op validates the IR then.
    if proto_library is None:
        proto_library = (
            Path(__file__).resolve().parents[3]
            / "build/gdn_proto/libchunk_gdn_proto.so"
        )
    if not Path(proto_library).is_file():
        raise FileNotFoundError(
            "Build the CANN operator registration with examples/scripts/build_chunk_gdn_proto.sh"
        )
    library = Path(proto_library).resolve(strict=True)
    _PROTO_HANDLES.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))

    @register_fx_node_ge_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)
    def convert(
        query,
        key,
        value,
        *,
        beta=None,
        initial_state=None,
        actual_seq_lengths=None,
        scale=None,
        g=None,
        meta_outputs=None,
    ):
        if (
            beta is None
            or initial_state is None
            or actual_seq_lengths is None
            or scale is None
        ):
            raise ValueError(
                "ChunkGDR requires beta, initial_state, lengths and explicit scale"
            )
        return ge.custom_op(
            "ChunkGatedDeltaRule",
            inputs={
                "query": query,
                "key": key,
                "value": value,
                "beta": beta,
                "initial_state": initial_state,
                "actual_seq_lengths": actual_seq_lengths,
                "g": g,
            },
            outputs=["out", "final_state"],
            attrs={"scale_value": ge.attr.Float(scale)},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split",), default="split")
    parser.add_argument("--pt2-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--acl-soc-name", default=os.environ.get("ACL_SOC_NAME", "Ascend910B3")
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        default=os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") == "1",
        help="compile PTE without binding a physical NPU",
    )
    args = parser.parse_args()
    if args.host_only:
        os.environ["EXECUTORCH_NPU_HOST_ONLY"] = "1"

    from examples.common.export import compile_pte
    from examples.qwen3_5.model import modeling  # noqa: F401

    register_chunk_gdn_converter()

    if args.pt2_dir is None:
        methods = {
            "prefill": Path(
                os.environ.get("PREFILL_PT2_PATH", "/tmp/qwen3_5_prefill.pt2")
            ),
            "decode": Path(
                os.environ.get("DECODE_PT2_PATH", "/tmp/qwen3_5_decode.pt2")
            ),
        }
    else:
        methods = {
            "prefill": args.pt2_dir / "prefill.pt2",
            "decode": args.pt2_dir / "decode.pt2",
        }
    output_dir = args.output_dir or Path(
        os.environ.get("OUTPUT_DIR", "/tmp/qwen3_5_executorch_npu")
    )
    timings = {}
    compile_pte(
        methods,
        output_dir,
        acl_soc_name=args.acl_soc_name,
        roles={"prefill": "prefill", "decode": "decode"},
        precision_mode="origin",
        timings=timings,
    )

    (output_dir / "pte_timings.json").write_text(json.dumps(timings, indent=2) + "\n")


if __name__ == "__main__":
    main()
