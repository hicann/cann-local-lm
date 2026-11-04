# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from ..shared.constants import PTD_FILE_TAG, WEIGHT_ALIGNMENT
from .io_contract import build_io_contract
from .manifest import (
    FileConstantBinding,
    Manifest,
    ModelArtifact,
    WeightGroup,
)
from .state import check_shared_states, check_state_role


if TYPE_CHECKING:
    from executorch.exir._serialize._named_data_store import NamedDataStore
    from torch.export import ExportedProgram

    from ..compiler.compile_spec import AscendCompileOptions


@dataclass(frozen=True)
class ExternalWeightArtifact:
    """One external-weight file and the metadata used to reference it."""

    file_name: str
    named_data_key: str
    sha256: str
    nbytes: int
    data: bytes


def register_model(
    store: NamedDataStore,
    om_bytes: bytes,
    method_name: str,
    delegate_index: int,
) -> ModelArtifact:
    """Store an OM in PTE named data under a content-addressed model key."""
    if not om_bytes:
        raise ValueError("CANN produced an empty OM")
    digest = hashlib.sha256(om_bytes).hexdigest()
    key = f"ascend.model.{_key_part(method_name)}.d{delegate_index}.{digest}"
    store.add_named_data(key, om_bytes, alignment=1, external_tag=None)
    return ModelArtifact(named_data_key=key, sha256=digest, nbytes=len(om_bytes))


def collect_and_register_weights(
    store: NamedDataStore, weight_dir: Path, ptd_file_tag: str
) -> list[ExternalWeightArtifact]:
    """Validate, read, and register every CANN external-weight file.

    The staging tree is rejected if symlinks or duplicate basenames could make
    runtime lookup ambiguous. File bytes are captured before the temporary
    compilation directory is removed and registered under SHA-256-based keys
    so identical weights can be shared across partitions.
    """
    root = weight_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"external weight path is not a directory: {root}")
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"external weight staging tree contains a symlink: {path}")
    files = [path for path in entries if not path.is_dir()]
    if not files:
        raise ValueError(f"CANN produced no external weight files in {root}")

    basenames: set[str] = set()
    artifacts: list[ExternalWeightArtifact] = []
    for path in files:
        if not path.is_file():
            raise ValueError(f"external weight is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"external weight escapes its staging directory: {path}"
            ) from error
        if path.name in basenames:
            raise ValueError(f"duplicate external weight basename: {path.name}")
        basenames.add(path.name)
        data = path.read_bytes()
        if not data:
            raise ValueError(f"external weight is empty: {path}")
        digest = hashlib.sha256(data).hexdigest()
        key = f"ascend.weights.{digest}"
        store.add_named_data(
            key,
            data,
            alignment=WEIGHT_ALIGNMENT,
            external_tag=ptd_file_tag,
        )
        artifacts.append(
            ExternalWeightArtifact(
                file_name=path.name,
                named_data_key=key,
                sha256=digest,
                nbytes=len(data),
                data=data,
            )
        )
    return artifacts


def weight_manifest_entries(
    weights: list[ExternalWeightArtifact],
) -> tuple[list[WeightGroup], list[FileConstantBinding]]:
    """Deduplicate registered blobs while preserving each CANN file identity."""
    groups: dict[str, WeightGroup] = {}
    bindings: list[FileConstantBinding] = []
    canonical_by_range: dict[tuple[str, int, int], str] = {}
    for index, weight in enumerate(weights):
        if weight.named_data_key not in groups:
            groups[weight.named_data_key] = WeightGroup(
                weight_group_key=weight.named_data_key,
                sha256=weight.sha256,
                total_nbytes=weight.nbytes,
            )
        binding_id = f"fc{index}"
        range_key = (weight.named_data_key, 0, weight.nbytes)
        canonical = canonical_by_range.setdefault(range_key, binding_id)
        bindings.append(
            FileConstantBinding(
                binding_id=binding_id,
                file_name=weight.file_name,
                weight_group_key=weight.named_data_key,
                offset=0,
                nbytes=weight.nbytes,
                shape=[weight.nbytes],
                alias_of=None if canonical == binding_id else canonical,
            )
        )
    return list(groups.values()), bindings


def build_manifest(
    *,
    edge_program: ExportedProgram,
    specs: AscendCompileOptions,
    store: NamedDataStore,
    om_path: Path,
    weight_dir: Path,
) -> Manifest:
    """Register compiled artifacts and return their typed runtime contract."""
    io_contract, state = build_io_contract(edge_program, specs.state_io_contract)
    check_state_role(state, specs.role)
    model = register_model(
        store, om_path.read_bytes(), specs.method_name, specs.delegate_index
    )
    weights = collect_and_register_weights(store, weight_dir, PTD_FILE_TAG)
    weight_groups, bindings = weight_manifest_entries(weights)
    return Manifest(
        method=specs.method_name,
        delegate_index=specs.delegate_index,
        role=specs.role,
        acl_soc_name=specs.acl_soc_name,
        model=model,
        weight_groups=weight_groups,
        file_constant_bindings=bindings,
        io_contract=io_contract,
        state=state,
    )


def check_manifest_set(manifests: Iterable[Manifest]) -> None:
    """Validate delegate identities and shared state before final encoding."""
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("manifest set must be non-empty")
    identities: set[tuple[str, int]] = set()
    for manifest in manifests:
        identity = (manifest.method, manifest.delegate_index)
        if identity in identities:
            raise ValueError(
                f"duplicate delegate identity in manifest set: {identity[0]}[{identity[1]}]"
            )
        identities.add(identity)
    check_shared_states(manifests)


def _key_part(value: str) -> str:
    """Sanitize user-derived identifiers for stable named-data keys."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
