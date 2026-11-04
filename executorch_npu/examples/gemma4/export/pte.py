# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile Gemma4 PT2 methods to one Ascend PTE."""

import argparse
import os
from pathlib import Path


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split", "unified"), default="split")
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
    from examples.gemma4.model import modeling  # noqa: F401

    if args.mode == "unified":
        methods = {"model": Path(os.environ.get("PT2_PATH", "/tmp/gemma4_model.pt2"))}
        if args.pt2_dir is not None:
            methods["model"] = args.pt2_dir / "model.pt2"
        output_dir = args.output_dir or Path(
            os.environ.get("OUTPUT_DIR", "/tmp/gemma4_executorch_npu")
        )
        roles = {"model": "inference"}
        precision_mode = None
    else:
        root = args.pt2_dir or Path(
            os.environ.get("OUT_ROOT", "/mnt/workspace/scratch_gemma4_pd")
        )
        methods = {"prefill": root / "prefill.pt2", "decode": root / "decode.pt2"}
        output_dir = args.output_dir or root / "pte"
        roles = {"prefill": "prefill", "decode": "decode"}
        precision_mode = "origin"

    compile_pte(
        methods,
        output_dir,
        acl_soc_name=args.acl_soc_name,
        roles=roles,
        precision_mode=precision_mode,
    )


if __name__ == "__main__":
    main()
