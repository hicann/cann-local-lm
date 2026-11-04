# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Typed v1 manifest and the final ASCD envelope written into PTE."""

from __future__ import annotations

import struct
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..shared.constants import FRAMING_MAGIC, FRAMING_VERSION, MAX_MANIFEST_BYTES


_HEADER = struct.Struct("<4sHI")
NonemptyString = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
WeightGroupKey = Annotated[str, Field(pattern=r"^ascend\.weights\.[0-9a-f]{64}$")]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Dtype = Literal[
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]
Direction = Literal["input", "output"]
SemanticRole = Literal["data", "state_current", "state_next"]


class ManifestModel(BaseModel):
    """Check fields once at construction; business checks live in artifact builders."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )


class RuntimeProtocol(ManifestModel):
    name: Literal["AscendBackend"] = "AscendBackend"
    """Always ``AscendBackend``; identifies the backend protocol."""
    version: Literal[1] = 1
    """Runtime protocol version; currently ``1``."""
    artifact_abi: Literal[1] = 1
    """Artifact ABI version; currently ``1``."""


class ModelArtifact(ManifestModel):
    named_data_key: NonemptyString
    """Key used to locate the OM in PTE named data."""
    sha256: Sha256
    """SHA-256 digest of the OM contents."""
    nbytes: PositiveInt
    """Logical size of the OM in bytes."""


class WeightGroup(ManifestModel):
    weight_group_key: WeightGroupKey
    """Key used to locate the weight group in PTD named data."""
    sha256: Sha256
    """SHA-256 digest of the weight-group contents."""
    total_nbytes: PositiveInt
    """Logical size of the weight group in bytes."""


class FileConstantBinding(ManifestModel):
    binding_id: NonemptyString
    """Unique identifier of the FileConstant binding."""
    file_name: Annotated[str, Field(min_length=1, pattern=r"^[^/\\]+$")]
    """File name exposed to the OM and GE runtime."""
    weight_group_key: WeightGroupKey
    """Key of the weight group that contains this binding."""
    offset: Annotated[int, Field(ge=0, multiple_of=32)]
    """Byte offset of the binding within its weight group."""
    nbytes: PositiveInt
    """Length of the bound byte range."""
    shape: Annotated[list[PositiveInt], Field(min_length=1, max_length=1)]
    """Logical shape of the bound byte range."""
    alias_of: NonemptyString | None = None
    """Optional identifier of an existing binding that shares this byte range."""


class IoBinding(ManifestModel):
    et_arg_index: NonnegativeInt
    """ExecuTorch argument index associated with this tensor."""
    direction: Direction
    """Tensor direction: ``input`` or ``output``."""
    ascend_name: NonemptyString
    """Tensor name used by the Ascend graph."""
    dtype: Dtype
    """Tensor data type."""
    shape: list[Annotated[int, Field(ge=-1)]]
    """Tensor shape, using ``-1`` for a dynamic dimension."""
    semantic_role: SemanticRole
    """Runtime role: data, current state, or next state."""
    shape_range: (
        list[Annotated[list[NonnegativeInt], Field(min_length=2, max_length=2)]] | None
    ) = None
    """Optional minimum and maximum values for each dynamic dimension."""


class IoContract(ManifestModel):
    inputs: list[IoBinding]
    """Input tensor bindings."""
    outputs: list[IoBinding]
    """Output tensor bindings."""


class StateBinding(ManifestModel):
    state_id: NonemptyString
    """Stable identifier of the runtime state."""
    current_input_arg: NonnegativeInt
    """Argument index of the tensor containing the current state."""
    next_output_arg: Annotated[int, Field(ge=-1)]
    """Argument index of the next state, or ``-1`` when it is not updated."""
    dtype: Dtype
    """State tensor data type."""
    shape: list[NonnegativeInt]
    """State tensor shape."""
    reset_policy: Literal["on_prefill", "explicit", "never"]
    """Policy controlling when the runtime resets this state."""
    reset_owner: bool
    """Whether this delegate owns the reset operation for the state."""
    host_visibility: Literal["device_only", "synchronized"]
    """Whether the state remains on device or is synchronized with the host."""


class StateContract(ManifestModel):
    bindings: list[StateBinding]
    """State bindings; empty for a stateless delegate."""


class Manifest(ManifestModel):
    runtime_protocol: RuntimeProtocol = Field(default_factory=RuntimeProtocol)
    """Backend runtime protocol and artifact ABI compatibility information."""
    method: NonemptyString
    """ExecuTorch method name."""
    delegate_index: NonnegativeInt
    """Index of this delegate within the ExecuTorch method."""
    role: Literal["inference", "prefill", "decode"]
    """Delegate role: inference, prefill, or decode."""
    acl_soc_name: NonemptyString
    """Validated device name used as both the GE target and runtime match."""
    model: ModelArtifact
    """Metadata used to locate and verify the compiled OM."""
    weight_groups: Annotated[list[WeightGroup], Field(min_length=1)]
    """External weight groups referenced by this delegate."""
    file_constant_bindings: Annotated[list[FileConstantBinding], Field(min_length=1)]
    """FileConstant bindings into the external weight groups."""
    io_contract: IoContract
    """Input and output tensor contract for the delegate."""
    state: StateContract
    """Persistent runtime state contract for the delegate."""

    def to_bytes(self) -> bytes:
        """Serialize once, retaining v1 fields, optional-key omission and framing.

        Pydantic emits fields in model declaration order, so output does not
        depend on the insertion order of dictionaries used at construction.
        No compiler path needs to decode this envelope again.
        """
        payload = self.model_dump_json(exclude_none=True).encode("utf-8")
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ValueError(
                f"manifest exceeds {MAX_MANIFEST_BYTES} byte protocol limit"
            )
        return _HEADER.pack(FRAMING_MAGIC, FRAMING_VERSION, len(payload)) + payload
