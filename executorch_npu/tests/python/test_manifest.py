# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import copy
import os
import subprocess

import pytest
from conftest import make_manifest, read_manifest_payload

from executorch_npu import Manifest
from executorch_npu.artifacts.builder import check_manifest_set
from executorch_npu.artifacts.state import check_shared_states


@pytest.mark.parametrize("buffer_mode", ["single", "double", "guess"])
def test_manifest_rejects_retired_buffer_mode(buffer_mode) -> None:
    data = make_manifest()
    data["io_contract"]["inputs"][0]["semantic_role"] = "state_current"
    data["io_contract"]["outputs"][0]["semantic_role"] = "state_next"
    binding = dict(
        state_id="state",
        current_input_arg=0,
        next_output_arg=1,
        dtype="float32",
        shape=[1, 4],
        reset_policy="never",
        reset_owner=False,
        host_visibility="device_only",
    )
    data["state"]["bindings"] = [binding]
    payload = read_manifest_payload(Manifest.model_validate(data).to_bytes())
    assert "buffer_mode" not in payload["state"]["bindings"][0]
    binding["buffer_mode"] = buffer_mode
    with pytest.raises(ValueError, match="buffer_mode"):
        Manifest.model_validate(data)


def test_manifest_preserves_v1_json_and_deterministic_output() -> None:
    data = make_manifest()
    manifest = Manifest.model_validate(data)
    reversed_manifest = Manifest.model_validate(dict(reversed(list(data.items()))))
    assert manifest.to_bytes() == reversed_manifest.to_bytes()
    assert read_manifest_payload(manifest.to_bytes()) == data
    assert manifest.model_dump(exclude_none=True) == data


def test_manifest_set_rejects_empty_collection_and_duplicate_delegates() -> None:
    with pytest.raises(ValueError, match="manifest set must be non-empty"):
        check_manifest_set(iter(()))
    manifest = Manifest.model_validate(make_manifest())
    with pytest.raises(ValueError, match="duplicate delegate identity"):
        check_manifest_set(iter([manifest, manifest]))


def test_manifest_set_rejects_decode_reset_owner() -> None:
    data = make_manifest()
    data.update(method="decode", role="decode")
    data["state"]["bindings"] = [
        {
            "state_id": "cache",
            "current_input_arg": 0,
            "next_output_arg": 1,
            "dtype": "float32",
            "shape": [1, 4],
            "reset_policy": "on_prefill",
            "reset_owner": True,
            "host_visibility": "device_only",
        }
    ]
    with pytest.raises(ValueError, match="reset owner must be a prefill delegate"):
        check_manifest_set([Manifest.model_validate(data)])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runtime_protocol", "name"), "OtherBackend"),
        (("runtime_protocol", "version"), 2),
        (("delegate_index",), "0"),
        (("model", "location"), "file"),
        (("compatibility",), {"required_capabilities": []}),
        (("io_contract", "inputs", 0, "layout"), "contiguous"),
        (("state",), {"bindings": [], "reset_policy": "none"}),
    ],
)
def test_manifest_model_rejects_protocol_field_drift(path, value) -> None:
    data = make_manifest()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        Manifest.model_validate(data)


def test_manifest_encodes_utf8_byte_length_and_optional_fields() -> None:
    data = make_manifest()
    data["io_contract"]["inputs"][0]["ascend_name"] = "输入"
    binding = data["io_contract"]["inputs"][0]
    binding.update(shape=[-1, 4], shape_range=[[2, 4], [4, 4]])
    manifest = Manifest.model_validate(data)
    encoded = manifest.to_bytes()
    assert read_manifest_payload(encoded) == data
    assert b" " not in encoded[10:]
    assert b"null" not in encoded[10:]


def test_manifest_enforces_runtime_payload_size(monkeypatch) -> None:
    import executorch_npu.artifacts.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "MAX_MANIFEST_BYTES", 4)
    with pytest.raises(ValueError, match="protocol limit"):
        Manifest.model_validate(make_manifest()).to_bytes()


def test_manifest_set_validates_state_metadata_and_reset_owner() -> None:
    prefill = make_manifest()
    decode = make_manifest()
    prefill.update({"method": "prefill", "role": "prefill"})
    decode.update({"method": "decode", "role": "decode"})
    state = {
        "state_id": "kv",
        "current_input_arg": 0,
        "next_output_arg": 1,
        "dtype": "float32",
        "shape": [1],
        "reset_policy": "on_prefill",
        "reset_owner": True,
        "host_visibility": "synchronized",
    }
    for manifest, owner in ((prefill, True), (decode, False)):
        manifest["io_contract"] = {
            "inputs": [
                {
                    "et_arg_index": 0,
                    "direction": "input",
                    "ascend_name": "current",
                    "dtype": "float32",
                    "shape": [1],
                    "semantic_role": "state_current",
                }
            ],
            "outputs": [
                {
                    "et_arg_index": 1,
                    "direction": "output",
                    "ascend_name": "next",
                    "dtype": "float32",
                    "shape": [1],
                    "semantic_role": "state_next",
                }
            ],
        }
        binding = copy.deepcopy(state)
        binding["reset_owner"] = owner
        manifest["state"] = {"bindings": [binding]}

    check_shared_states(
        [Manifest.model_validate(prefill), Manifest.model_validate(decode)]
    )
    decode["state"]["bindings"][0]["shape"] = [2]
    decode["io_contract"]["inputs"][0]["shape"] = [2]
    decode["io_contract"]["outputs"][0]["shape"] = [2]
    with pytest.raises(ValueError, match="metadata differs"):
        check_shared_states(
            [Manifest.model_validate(prefill), Manifest.model_validate(decode)]
        )


@pytest.mark.parametrize("owners", [0, 2])
def test_shared_state_requires_one_reset_owner(owners) -> None:
    manifests = []
    for index in range(2):
        data = make_manifest()
        data.update(method=f"prefill_{index}", role="prefill")
        data["io_contract"]["inputs"][0]["semantic_role"] = "state_current"
        data["io_contract"]["outputs"][0]["semantic_role"] = "state_next"
        data["state"]["bindings"] = [
            {
                "state_id": "cache",
                "current_input_arg": 0,
                "next_output_arg": 1,
                "dtype": "float32",
                "shape": [1, 4],
                "reset_policy": "on_prefill",
                "reset_owner": index < owners,
                "host_visibility": "device_only",
            }
        ]
        manifests.append(Manifest.model_validate(data))
    with pytest.raises(ValueError, match="exactly one prefill reset owner"):
        check_shared_states(manifests)


def test_cpp_runtime_reads_pydantic_payloads(tmp_path) -> None:
    """Exercise the unchanged C++ parser with representative producer output."""
    binary = os.environ.get("CANN_MANIFEST_TEST_BINARY")
    if not binary:
        pytest.skip("set CANN_MANIFEST_TEST_BINARY to executorch_npu_manifest_test")
    variants = {"static": make_manifest()}
    dynamic = make_manifest()
    for binding in dynamic["io_contract"]["inputs"] + dynamic["io_contract"]["outputs"]:
        binding.update(shape=[-1, 4], shape_range=[[2, 4], [4, 4]])
    variants["dynamic"] = dynamic

    aliased = make_manifest()
    alias = dict(aliased["file_constant_bindings"][0])
    alias.update(binding_id="fc1", file_name="weight_alias", alias_of="fc0")
    aliased["file_constant_bindings"].append(alias)
    variants["alias"] = aliased

    for name, no_output in (("scalar_state", False), ("no_state_update", True)):
        data = make_manifest()
        data["io_contract"]["inputs"][0].update(
            semantic_role="state_current", dtype="int64", shape=[]
        )
        if not no_output:
            data["io_contract"]["outputs"][0].update(
                semantic_role="state_next", dtype="int64", shape=[]
            )
        data["state"]["bindings"] = [
            {
                "state_id": "counter",
                "current_input_arg": 0,
                "next_output_arg": -1 if no_output else 1,
                "dtype": "int64",
                "shape": [],
                "reset_policy": "never",
                "reset_owner": False,
                "host_visibility": "device_only",
            }
        ]
        variants[name] = data

    paths = []
    for name, data in variants.items():
        path = tmp_path / f"{name}.ascd"
        path.write_bytes(Manifest.model_validate(data).to_bytes())
        paths.append(str(path))
    subprocess.run([binary, *paths], check=True, capture_output=True, text=True)
