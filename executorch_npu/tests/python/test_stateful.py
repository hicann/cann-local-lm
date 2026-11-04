# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

from dataclasses import dataclass

import torch
from stateful_runner import (
    build_test_state_io_contract,
    StatefulMethod,
    StatefulMethodRunner,
)

from executorch_npu import build_state_io_contract


@dataclass(frozen=True)
class _State:
    name: str = "counter"
    dtype: str = "float32"
    shape: tuple[int, ...] = (1,)


def test_stateful_runner_routes_states_and_resets() -> None:
    contract = build_test_state_io_contract(
        (_State(),),
        current_input_start=1,
        next_output_start=3,
        reset_owner=True,
    )

    def increment(value: torch.Tensor, state: torch.Tensor):
        next_state = state + value
        return next_state * 2, next_state

    runner = StatefulMethodRunner(
        {
            "prefill": StatefulMethod(increment, 2, (0,), contract, True),
            "decode": StatefulMethod(
                increment,
                2,
                (0,),
                build_test_state_io_contract(
                    (_State(),),
                    current_input_start=1,
                    next_output_start=3,
                    reset_owner=False,
                ),
            ),
        },
        {"counter": torch.tensor([0.0])},
    )

    assert torch.equal(
        runner.run("prefill", (torch.tensor([2.0]),))[0], torch.tensor([4.0])
    )
    assert torch.equal(
        runner.run("decode", (torch.tensor([3.0]),))[0], torch.tensor([10.0])
    )
    assert torch.equal(
        runner.run("prefill", (torch.tensor([1.0]),))[0], torch.tensor([2.0])
    )


def test_export_state_contract_records_source_mapping() -> None:
    @torch.no_grad()
    def make_exported():
        class Ordered(torch.nn.Module):
            def forward(self, value, first, second):
                first.add_(value.reshape(1))
                second.add_(value.repeat(2))
                return value, first, second

        return torch.export.export(
            Ordered(),
            (torch.tensor([2.0]), torch.zeros(1), torch.zeros(2)),
            strict=False,
        )

    contract = build_state_io_contract(
        make_exported(),
        (
            _State(name="first", shape=(1,)),
            _State(name="second", shape=(2,)),
        ),
        current_input_start=1,
        reset_owner=False,
    )

    assert [binding["next_output_arg"] for binding in contract["bindings"]] == [4, 5]
    assert [binding["source_input_name"] for binding in contract["bindings"]] == [
        "first",
        "second",
    ]
    assert [binding["source_output_arg"] for binding in contract["bindings"]] == [
        1,
        2,
    ]


def test_export_state_contract_maps_output_positions() -> None:
    @torch.no_grad()
    def make_exported():
        class Ordered(torch.nn.Module):
            def forward(self, value, first, second):
                first.add_(value.reshape(1))
                second.add_(value.repeat(2))
                return value, first, second

        exported = torch.export.export(
            Ordered(),
            (torch.tensor([2.0]), torch.zeros(1), torch.zeros(2)),
            strict=False,
        )
        for node in exported.graph_module.graph.nodes:
            if node.name in {"add_", "add__1"}:
                node.meta.pop("val", None)
        return exported

    contract = build_state_io_contract(
        make_exported(),
        (
            _State(name="first", shape=(1,)),
            _State(name="second", shape=(2,)),
        ),
        current_input_start=1,
        reset_owner=False,
    )

    assert [binding["source_output_arg"] for binding in contract["bindings"]] == [
        1,
        2,
    ]
