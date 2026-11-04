# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_manifest, read_manifest_payload
from jsonschema import Draft202012Validator, ValidationError

from executorch_npu import Manifest
from scripts.generate_manifest_schema import manifest_schema


@pytest.fixture
def schema():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/manifest_schema/manifest-v1.schema.json"
    )
    return json.loads(path.read_text())


def test_published_schema_matches_the_typed_contract(schema):
    assert schema == manifest_schema(), "Run python scripts/generate_manifest_schema.py"
    Draft202012Validator.check_schema(schema)


def test_published_schema_accepts_serialized_double_buffer_state(schema):
    data = make_manifest()
    data["state"]["bindings"] = [
        {
            "state_id": "kv",
            "current_input_arg": 0,
            "next_output_arg": 1,
            "dtype": "float32",
            "shape": [1, 4],
            "reset_policy": "never",
            "reset_owner": False,
            "host_visibility": "device_only",
        }
    ]
    payload = read_manifest_payload(Manifest.model_validate(data).to_bytes())
    assert "buffer_mode" not in payload["state"]["bindings"][0]
    Draft202012Validator(schema).validate(payload)
    for mode in ("single", "double"):
        payload["state"]["bindings"][0]["buffer_mode"] = mode
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "bad_weight_key",
        "bad_dtype",
        "bad_abi",
        "missing_protocol",
        "missing_version",
        "null_optional",
    ],
)
def test_published_schema_rejects_invalid_wire_fields(schema, mutation):
    payload = read_manifest_payload(Manifest.model_validate(make_manifest()).to_bytes())
    if mutation == "extra_field":
        payload["unexpected"] = True
    elif mutation == "bad_weight_key":
        payload["weight_groups"][0]["weight_group_key"] = "invalid"
    elif mutation == "bad_dtype":
        payload["io_contract"]["inputs"][0]["dtype"] = "float7"
    elif mutation == "bad_abi":
        payload["runtime_protocol"]["artifact_abi"] = 2
    elif mutation == "missing_protocol":
        del payload["runtime_protocol"]
    elif mutation == "missing_version":
        del payload["runtime_protocol"]["version"]
    else:
        payload["io_contract"]["inputs"][0]["shape_range"] = None
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(payload)
