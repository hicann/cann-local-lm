# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import hashlib

import pytest
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch_npu.artifacts.builder import (
    collect_and_register_weights,
    register_model,
    weight_manifest_entries,
)
from executorch_npu.shared.constants import PTD_FILE_TAG


def test_model_is_internal_and_weight_is_external(tmp_path) -> None:
    store = NamedDataStore()
    model = register_model(store, b"om", "forward", 0)
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "weight_a").write_bytes(b"weight")
    weights = collect_and_register_weights(store, weight_dir, PTD_FILE_TAG)
    groups, bindings = weight_manifest_entries(weights)
    output = store.get_named_data_store_output()

    assert model.named_data_key in output.pte_data
    assert weights[0].named_data_key in output.external_data[PTD_FILE_TAG]
    assert groups[0].total_nbytes == 6
    assert bindings[0].file_name == "weight_a"


def test_equal_weights_dedupe_but_keep_alias_bindings(tmp_path) -> None:
    store = NamedDataStore()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "weight_a").write_bytes(b"same")
    (weight_dir / "weight_b").write_bytes(b"same")
    weights = collect_and_register_weights(store, weight_dir, "weights-tag")
    groups, bindings = weight_manifest_entries(weights)

    assert len(groups) == 1
    assert len(bindings) == 2
    assert bindings[1].alias_of == "fc0"
    assert len(store.buffers) == 1


def test_weight_key_uses_full_sha256(tmp_path) -> None:
    store = NamedDataStore()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    data = b"x"
    (weight_dir / "weight").write_bytes(data)
    artifact = collect_and_register_weights(store, weight_dir, "weights-tag")[0]
    assert artifact.named_data_key == (
        "ascend.weights." + hashlib.sha256(data).hexdigest()
    )


def test_empty_weight_directory_is_rejected(tmp_path) -> None:
    store = NamedDataStore()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    with pytest.raises(ValueError, match="no external weight"):
        collect_and_register_weights(store, weight_dir, "weights-tag")


def test_symlink_in_weight_tree_is_rejected(tmp_path) -> None:
    store = NamedDataStore()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (weight_dir / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        collect_and_register_weights(store, weight_dir, "weights-tag")


def test_weight_bindings_cover_registered_bytes_and_omit_unused_alias(tmp_path) -> None:
    store = NamedDataStore()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "weight_a").write_bytes(b"first")
    (weight_dir / "weight_b").write_bytes(b"second")
    groups, bindings = weight_manifest_entries(
        collect_and_register_weights(store, weight_dir, PTD_FILE_TAG)
    )
    group_sizes = {group.weight_group_key: group.total_nbytes for group in groups}
    for binding in bindings:
        assert binding.offset % 32 == 0
        assert binding.offset + binding.nbytes <= group_sizes[binding.weight_group_key]
        assert binding.shape == [binding.nbytes]
        assert "alias_of" not in binding.model_dump(exclude_none=True)
