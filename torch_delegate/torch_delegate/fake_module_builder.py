# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Registry-driven Python fake modules used by Lite EPAIR export."""

from __future__ import annotations

import dataclasses
import enum
import sys
from types import ModuleType
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .lite_context import LiteConfig


class InterfaceKind(enum.Enum):
    FUNCTION = "function"
    VALUE = "value"
    CLASS = "class"
    ENUM = "enum"


@dataclasses.dataclass(frozen=True)
class FakeInterface:
    module: str
    name: str
    kind: InterfaceKind
    factory: Callable[["LiteConfig"], Any]


class FakeModuleBuilder:
    """Build and install a strict fake Python module tree."""

    def __init__(
        self,
        config: "LiteConfig",
        unsupported: Callable[[str, bool], Callable[..., Any]],
    ) -> None:
        self._config = config
        self._unsupported = unsupported
        self._modules: dict[str, ModuleType] = {}

    def build(self) -> dict[str, ModuleType]:
        for interface in get_registered_interfaces(self._unsupported):
            self._install(interface)
        self._link_parent_modules()
        for name, module in self._modules.items():
            module.__torch_delegate_lite_fake__ = True
            sys.modules[name] = module
        return dict(self._modules)

    def _install(self, interface: FakeInterface) -> None:
        module = self._get_or_create_module(interface.module)
        setattr(module, interface.name, interface.factory(self._config))

    def _get_or_create_module(self, name: str) -> ModuleType:
        if name not in self._modules:
            self._modules[name] = ModuleType(name)
        return self._modules[name]

    def _link_parent_modules(self) -> None:
        names = sorted(self._modules, key=lambda item: item.count("."))
        for name in names:
            module = self._modules[name]
            if any(other.startswith(name + ".") for other in self._modules):
                module.__path__ = []
            if "." not in name:
                continue
            parent_name, child_name = name.rsplit(".", 1)
            # torchair 是硬依赖且为真实命名空间包, 只 stub 叶子 _torchair (设计 5.2.2:
            # "TorchAir 可以继续导入真实 Python _backend.py"). 造假父 torchair.core 会用
            # __path__=[] 影子覆盖真实包, 致 from torchair.core import _backend 失败.
            # torch_npu 树无真实包, 仍需完整造假父.
            if parent_name == "torchair" or parent_name.startswith("torchair."):
                continue
            parent = self._get_or_create_module(parent_name)
            child = self._modules[name]
            setattr(parent, child_name, child)
            if parent_name not in sys.modules:
                sys.modules[parent_name] = parent


def get_registered_interfaces(
    unsupported: Callable[[str, bool], Callable[..., Any]],
) -> tuple[FakeInterface, ...]:
    """Return the explicitly supported Lite dependency surface."""
    return (
        FakeInterface(
            "torchair.core._torchair",
            "InitializeGraphEngine",
            InterfaceKind.FUNCTION,
            lambda _: lambda options=None: None,
        ),
        FakeInterface(
            "torchair.core._torchair",
            "FinalizeGraphEngine",
            InterfaceKind.FUNCTION,
            lambda _: lambda: None,
        ),
        FakeInterface(
            "torchair.core._torchair",
            "get_registered_ir_def",
            InterfaceKind.FUNCTION,
            lambda _: unsupported(
                "torchair.core._torchair.get_registered_ir_def", True
            ),
        ),
        FakeInterface(
            "torchair.core._torchair",
            "check_cann_compat",
            InterfaceKind.FUNCTION,
            lambda _: unsupported("torchair.core._torchair.check_cann_compat", True),
        ),
        FakeInterface(
            "torchair.core._torchair",
            "TorchNpuGraphBase",
            InterfaceKind.CLASS,
            lambda _: _build_graph_base(unsupported),
        ),
        FakeInterface(
            "torchair.core._torchair",
            "TorchDataType",
            InterfaceKind.ENUM,
            lambda _: _build_torch_data_type_namespace(),
        ),
        FakeInterface(
            "torch_npu.npu",
            "get_device_name",
            InterfaceKind.FUNCTION,
            lambda config: lambda: config.target_soc,
        ),
        FakeInterface(
            "torch_npu.npu",
            "current_device",
            InterfaceKind.FUNCTION,
            lambda _: unsupported("torch_npu.npu.current_device", True),
        ),
        FakeInterface(
            "torch_npu.npu",
            "_get_deterministic_level",
            InterfaceKind.FUNCTION,
            lambda _: lambda: 0,
        ),
        FakeInterface(
            "torch_npu.npu.matmul",
            "allow_hf32",
            InterfaceKind.VALUE,
            lambda config: config.allow_hf32,
        ),
        FakeInterface(
            "torch_npu.npu.conv",
            "allow_hf32",
            InterfaceKind.VALUE,
            lambda config: config.allow_hf32,
        ),
        FakeInterface(
            "torch_npu.npu.utils",
            "get_cann_version",
            InterfaceKind.FUNCTION,
            lambda _: unsupported("torch_npu.npu.utils.get_cann_version", True),
        ),
        FakeInterface(
            "torch_npu.npu.utils",
            "_is_gte_cann_version",
            InterfaceKind.FUNCTION,
            lambda _: unsupported("torch_npu.npu.utils._is_gte_cann_version", False),
        ),
    )


def _build_graph_base(unsupported):
    class BuildOnlyTorchNpuGraphBase:
        def __init__(self, name=""):
            self.name = str(name)

        def __getattr__(self, name):
            raise unsupported(
                f"torch_npu_graph.{name}", True
            )()  # Raise the factory-produced exception.

    return BuildOnlyTorchNpuGraphBase


def _build_torch_data_type_namespace():
    names = (
        "BOOL",
        "UINT8",
        "INT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
        "BF16",
        "FLOAT16",
        "FLOAT32",
        "FLOAT64",
        "COMPLEX64",
        "COMPLEX128",
    )
    return enum.Enum("TorchDataType", names)
