# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
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


_SPILL_CHUNK = 8 * 1024 * 1024


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_SPILL_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


_SPILL_ATEXIT_REGISTERED = False


def _spill_dir() -> Path:
    """Process-specific spill dir for file-backed weight blobs.

    Based on EXECUTORCH_NPU_STAGING (same base as the compilation staging
    tree) so the ownership move stays a cheap same-filesystem rename, and a
    disk-backed staging choice is not silently defeated by a tmpfs TMPDIR
    (tmpfs "files" would keep the bytes in cgroup memory, voiding the
    file-backed optimization).
    """
    base = os.environ.get("EXECUTORCH_NPU_STAGING") or tempfile.gettempdir()
    return Path(base) / f"executorch_npu_spill_{os.getpid()}"


def cleanup_spill_dir() -> None:
    """Best-effort removal of this process's spill dir once it is empty.

    FileBackedData finalizers remove the blobs themselves; this reclaims the
    directory. A SIGKILLed process may leave the directory (and any
    unreleased blobs) behind - a disk leak only, never a correctness issue.
    """
    try:
        os.rmdir(_spill_dir())
    except OSError:
        pass


def _file_backed_weight(path: Path, digest: str):
    """Move a weight file to the spill dir and hand ownership to the store.

    FileBackedData.move_from renames within the file's own directory, but the
    staging tree is removed right after the manifest is built, so the file is
    first moved into the process-specific spill dir (same filesystem base as
    staging, so a cheap rename). The store's finalizer removes the spilled
    file once the blob is released, and the PTD writer streams file-backed
    blobs in chunks (Cord.write_to_file), so external-weight bytes never
    reside in RAM. Duplicate blobs (identical digest across methods) are
    rejected by the store's dedup and their spill files are reclaimed by the
    finalizer.
    """
    global _SPILL_ATEXIT_REGISTERED
    from executorch.exir._serialize._cord import FileBackedData

    spill = _spill_dir()
    spill.mkdir(parents=True, exist_ok=True)
    if not _SPILL_ATEXIT_REGISTERED:
        # atexit is LIFO: blob finalizers created after this hook run before
        # it, so the directory is normally empty by the time we remove it.
        import atexit

        atexit.register(cleanup_spill_dir)
        _SPILL_ATEXIT_REGISTERED = True
    moved = spill / f"{digest[:16]}_{path.name}"
    shutil.move(str(path), str(moved))
    return FileBackedData.move_from(str(moved), sha256=bytes.fromhex(digest))


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
    """Validate and register every CANN external-weight file.

    The staging tree is rejected if symlinks or duplicate basenames could make
    runtime lookup ambiguous. Weight files are streamed-hashed, moved out of
    the temporary compilation directory into the process spill dir, and
    registered as file-backed blobs under SHA-256-based keys so identical
    weights are shared across partitions without residing in RAM.
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
        nbytes = path.stat().st_size
        if not nbytes:
            raise ValueError(f"external weight is empty: {path}")
        digest = _stream_sha256(path)
        key = f"ascend.weights.{digest}"
        store.add_named_data(
            key,
            _file_backed_weight(path, digest),
            alignment=WEIGHT_ALIGNMENT,
            external_tag=ptd_file_tag,
        )
        artifacts.append(
            ExternalWeightArtifact(
                file_name=path.name,
                named_data_key=key,
                sha256=digest,
                nbytes=nbytes,
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
