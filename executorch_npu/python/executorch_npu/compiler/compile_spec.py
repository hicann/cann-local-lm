# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from typing import Annotated, Any, Literal

from executorch.exir.backend.compile_spec_schema import CompileSpec
from pydantic import BaseModel, ConfigDict, Field


NonemptyString = Annotated[str, Field(min_length=1)]
NonnegativeInt = Annotated[int, Field(ge=0)]
DelegateRole = Literal["inference", "prefill", "decode"]
AclSocName = Literal[
    "Ascend910_9362",
    "Ascend910B3",
    "Ascend910B4",
    "Ascend310P3",
    "Ascend310B1",
]


class AscendCompileOptions(BaseModel):
    """Typed options serialized as one JSON ExecuTorch CompileSpec."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        use_attribute_docstrings=True,
    )

    method_name: NonemptyString
    """ExecuTorch method that owns this delegated partition."""
    delegate_index: NonnegativeInt
    """Zero-based index of the delegate within the method."""
    acl_soc_name: AclSocName
    """Verified CANN target used for offline compilation and runtime matching."""
    role: DelegateRole = "inference"
    """Delegate role used to derive state lifecycle behavior."""
    offline_build_options: dict[str, str] = Field(default_factory=dict)
    """Additional string options forwarded to GE offline compilation."""
    state_io_contract: dict[str, Any] = Field(default_factory=dict)
    """Optional state-binding metadata used to construct the runtime contract."""

    def to_compile_specs(self) -> list[CompileSpec]:
        """Encode all options as one Pydantic JSON payload."""
        return [
            CompileSpec(
                key="ascend_compile_options",
                value=self.model_dump_json().encode("utf-8"),
            )
        ]
