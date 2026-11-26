# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from typing import Annotated, Any, Final, Literal

from executorch.exir.backend.compile_spec_schema import CompileSpec
from pydantic import BaseModel, ConfigDict, Field, field_validator


NonemptyString = Annotated[str, Field(min_length=1)]
NonnegativeInt = Annotated[int, Field(ge=0)]
DelegateRole = Literal["inference", "prefill", "decode"]
# Offline compilation selects its GE target by SoC family, so a new variant
# inside a family needs no change here. CANN still rejects an unknown variant,
# and the runtime keeps matching the recorded name against aclrtGetSocName().
SUPPORTED_SOC_FAMILIES: Final[tuple[str, ...]] = (
    "Ascend910_93",
    "Ascend910B",
    "Ascend310P",
    "Ascend310B",
)
AclSocName = NonemptyString


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
    """CANN target of a supported SoC family, used for offline compilation.

    The runtime still requires an exact match against ``aclrtGetSocName()``.
    """
    role: DelegateRole = "inference"
    """Delegate role used to derive state lifecycle behavior."""
    offline_build_options: dict[str, str] = Field(default_factory=dict)
    """Additional string options forwarded to GE offline compilation."""
    state_io_contract: dict[str, Any] = Field(default_factory=dict)
    """Optional state-binding metadata used to construct the runtime contract."""

    @field_validator("acl_soc_name")
    @classmethod
    def _require_supported_soc_family(cls, value: str) -> str:
        """Accept every SoC of a supported family instead of a fixed variant list."""
        if not value.startswith(SUPPORTED_SOC_FAMILIES):
            families = ", ".join(f"{prefix}*" for prefix in SUPPORTED_SOC_FAMILIES)
            raise ValueError(
                f"unsupported acl_soc_name {value!r}; supported SoC families are "
                f"{families}. The runtime still requires an exact match against "
                "aclrtGetSocName()."
            )
        return value

    def to_compile_specs(self) -> list[CompileSpec]:
        """Encode all options as one Pydantic JSON payload."""
        return [
            CompileSpec(
                key="ascend_compile_options",
                value=self.model_dump_json().encode("utf-8"),
            )
        ]
