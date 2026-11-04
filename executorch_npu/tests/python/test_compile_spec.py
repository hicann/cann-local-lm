# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from executorch_npu.compiler.compile_spec import AscendCompileOptions


def test_compile_options_round_trip() -> None:
    options = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
        offline_build_options={"ge.foo": "bar"},
    )
    specs = options.to_compile_specs()
    parsed = AscendCompileOptions.model_validate_json(bytes(specs[0].value))
    assert specs[0].key == "ascend_compile_options"
    assert parsed == options


def test_acl_soc_name_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="acl_soc_name"):
        AscendCompileOptions(method_name="forward", delegate_index=0, acl_soc_name="")


def test_supported_acl_soc_names_are_accepted() -> None:
    for acl_soc_name in (
        "Ascend910_9362",
        "Ascend910B3",
        "Ascend910B4",
        "Ascend310P3",
        "Ascend310B1",
    ):
        options = AscendCompileOptions(
            method_name="forward",
            delegate_index=0,
            acl_soc_name=acl_soc_name,
        )
        assert options.acl_soc_name == acl_soc_name


def test_unsupported_acl_soc_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="acl_soc_name"):
        AscendCompileOptions(
            method_name="forward",
            delegate_index=0,
            acl_soc_name="FutureAscendTarget",
        )
