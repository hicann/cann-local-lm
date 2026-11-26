# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from torch.export import ExportedProgram


def resolve_symbolic_dimension_range(
    exported_program: ExportedProgram, dimension: Any
) -> tuple[int, int]:
    """Resolve one SymInt expression to finite non-negative bounds."""
    node = getattr(dimension, "node", None)
    expression = getattr(node, "expr", None)
    constraint = exported_program.range_constraints.get(expression)
    if constraint is None:
        shape_env = getattr(node, "shape_env", None)
        if shape_env is not None:
            constraint = shape_env.var_to_range.get(expression)
            if constraint is None and hasattr(shape_env, "bound_sympy"):
                constraint = shape_env.bound_sympy(expression)
    if constraint is None:
        raise ValueError(f"dynamic dimension {dimension!r} has no range constraint")
    _require_declared_bounds(dimension, constraint)
    lower = _finite_nonnegative_int(constraint.lower, "lower")
    upper = _finite_nonnegative_int(constraint.upper, "upper")
    if lower > upper:
        raise ValueError(
            f"dynamic dimension {dimension!r} has invalid range [{lower}, {upper}]"
        )
    return lower, upper


def _require_declared_bounds(dimension: Any, constraint: Any) -> None:
    """Reject dynamic dimensions exported without a finite bound.

    ``Dim("n")`` and ``Dim("n", min=k)`` both pass ``torch.export.export`` with
    an infinite upper bound, but CANN needs a finite interval to plan memory and
    to build the ``input_shape`` compile option. Report the missing declaration
    here instead of letting sympy's integer conversion raise an internal error.
    """
    for endpoint, value in (("lower", constraint.lower), ("upper", constraint.upper)):
        if _is_unbounded(value):
            raise ValueError(
                f"dynamic dimension {dimension!r} has no finite {endpoint} bound "
                f"(got {value!r}); declare both bounds explicitly, for example "
                "torch.export.Dim('name', min=1, max=128)"
            )


def _is_unbounded(value: Any) -> bool:
    """Detect every infinity sentinel that can appear in a range endpoint.

    ``sympy.oo`` reports ``is_infinite``, but the ``int_oo`` that torch uses for
    unbounded integer symbols reports ``is_finite``, so only a float conversion
    covers both spellings.
    """
    if getattr(value, "is_infinite", False):
        return True
    try:
        return not math.isfinite(float(value))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _finite_nonnegative_int(value: Any, endpoint: str) -> int:
    """Convert a symbolic range endpoint without accepting infinities."""
    try:
        result = int(value)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"dynamic shape {endpoint} bound must be a finite integer, got {value!r}"
        ) from error
    if result < 0 or value != result:
        raise ValueError(
            f"dynamic shape {endpoint} bound must be a non-negative integer, got {value!r}"
        )
    return result
