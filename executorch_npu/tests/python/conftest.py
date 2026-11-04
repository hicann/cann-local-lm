# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


def make_manifest() -> dict[str, Any]:
    om = b"fake-om"
    weight = b"fake-weight"
    om_hash = hashlib.sha256(om).hexdigest()
    weight_hash = hashlib.sha256(weight).hexdigest()
    return {
        "runtime_protocol": {
            "name": "AscendBackend",
            "version": 1,
            "artifact_abi": 1,
        },
        "method": "forward",
        "delegate_index": 0,
        "role": "inference",
        "acl_soc_name": "Ascend910_9362",
        "model": {
            "named_data_key": f"ascend.model.forward.d0.{om_hash}",
            "sha256": om_hash,
            "nbytes": len(om),
        },
        "weight_groups": [
            {
                "weight_group_key": f"ascend.weights.{weight_hash}",
                "sha256": weight_hash,
                "total_nbytes": len(weight),
            }
        ],
        "file_constant_bindings": [
            {
                "binding_id": "fc0",
                "file_name": "weight_0",
                "weight_group_key": f"ascend.weights.{weight_hash}",
                "offset": 0,
                "nbytes": len(weight),
                "shape": [len(weight)],
            }
        ],
        "io_contract": {
            "inputs": [
                {
                    "et_arg_index": 0,
                    "direction": "input",
                    "ascend_name": "x",
                    "dtype": "float32",
                    "shape": [1, 4],
                    "semantic_role": "data",
                }
            ],
            "outputs": [
                {
                    "et_arg_index": 1,
                    "direction": "output",
                    "ascend_name": "add",
                    "dtype": "float32",
                    "shape": [1, 4],
                    "semantic_role": "data",
                }
            ],
        },
        "state": {"bindings": []},
    }


def read_manifest_payload(processed_bytes: bytes) -> dict[str, Any]:
    """Inspect compiler output in tests; production compilation never decodes it."""
    magic, version, size = struct.unpack_from("<4sHI", processed_bytes)
    payload = processed_bytes[10:]
    assert magic == b"ASCD" and version == 1
    assert size == len(payload)
    return json.loads(payload)
