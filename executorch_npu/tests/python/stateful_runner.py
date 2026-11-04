# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


def build_test_state_io_contract(
    state_specs: Sequence[Any],
    *,
    current_input_start: int,
    next_output_start: int,
    reset_owner: bool,
) -> dict[str, list[dict[str, object]]]:
    return {
        "bindings": [
            {
                "state_id": spec.name,
                "current_input_arg": current_input_start + index,
                "next_output_arg": next_output_start + index,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "reset_policy": "on_prefill",
                "reset_owner": reset_owner,
                "host_visibility": "synchronized",
            }
            for index, spec in enumerate(state_specs)
        ]
    }


@dataclass(frozen=True)
class StatefulMethod:
    invoke: Callable[..., tuple[torch.Tensor, ...]]
    input_count: int
    data_input_args: tuple[int, ...]
    state_io_contract: Mapping[str, Any]
    reset_before_run: bool = False


class StatefulMethodRunner:
    def __init__(
        self,
        methods: Mapping[str, StatefulMethod],
        initial_states: Mapping[str, torch.Tensor],
    ) -> None:
        self._methods = dict(methods)
        self._initial_states = dict(initial_states)
        self._states = dict(initial_states)

    def reset(self) -> None:
        self._states = dict(self._initial_states)

    def run(
        self, method_name: str, data_inputs: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        method = self._methods[method_name]
        if method.reset_before_run:
            self.reset()
        args: list[torch.Tensor | None] = [None] * method.input_count
        for index, value in zip(method.data_input_args, data_inputs, strict=True):
            args[index] = value
        for binding in method.state_io_contract["bindings"]:
            args[int(binding["current_input_arg"])] = self._states[
                str(binding["state_id"])
            ]
        outputs = method.invoke(*args)
        state_outputs: set[int] = set()
        for binding in method.state_io_contract["bindings"]:
            output_index = int(binding["next_output_arg"]) - method.input_count
            self._states[str(binding["state_id"])] = outputs[output_index]
            state_outputs.add(output_index)
        return tuple(
            output for index, output in enumerate(outputs) if index not in state_outputs
        )
