# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for Lite EPAIR context state and import isolation."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_does_not_load_torchair_or_torch_npu():
    result = _run_isolated(
        """
        import sys
        import torch_delegate
        assert "torchair" not in sys.modules
        assert "torch_npu" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_context_is_process_scoped_and_nested_same_config_is_allowed():
    result = _run_isolated(
        """
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import ProcessMode, process_mode

        config = LiteConfig("Ascend950", allow_hf32=True)
        assert process_mode() is ProcessMode.CLEAN
        with lite_export_context(config):
            assert process_mode() is ProcessMode.LITE_INITIALIZED
            with lite_export_context(config):
                pass
        assert process_mode() is ProcessMode.LITE_INITIALIZED
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_context_rejects_different_process_configuration():
    result = _run_isolated(
        """
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import LiteStateError

        with lite_export_context(LiteConfig("Ascend950")):
            pass
        try:
            with lite_export_context(LiteConfig("Ascend910B")):
                pass
        except LiteStateError as exc:
            assert "separate process" in str(exc)
        else:
            raise AssertionError("different Lite configuration was accepted")
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_context_rejects_another_thread_after_context_exit():
    result = _run_isolated(
        """
        import threading
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import LiteStateError

        with lite_export_context(LiteConfig("Ascend950")):
            pass
        errors = []
        def enter():
            try:
                with lite_export_context(LiteConfig("Ascend950")):
                    pass
            except LiteStateError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=enter)
        thread.start()
        thread.join()
        assert errors and "separate process" in errors[0]
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_context_rejects_preloaded_torch_npu():
    result = _run_isolated(
        """
        import torch_npu
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import LiteImportError

        try:
            with lite_export_context(LiteConfig("Ascend950")):
                pass
        except LiteImportError as exc:
            assert "torch_npu" in str(exc)
        else:
            raise AssertionError("preloaded torch_npu was not rejected")
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_context_disables_pattern_fusion():
    result = _run_isolated(
        """
        from torch_delegate import EpairConfig, LiteConfig, lite_export_context
        from torch_delegate.lite_context import prepare_lite_torchair

        with lite_export_context(LiteConfig("Ascend950")):
            prepare_lite_torchair()
            from torch_delegate.converter.torchair_converter import TorchAirGEConverter

            compiler_config = TorchAirGEConverter(
                EpairConfig(pattern_fusion_pass=True)
            )._create_compiler_config()
            assert compiler_config.experimental_config.pattern_fusion_pass.value == "0"
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_fake_device_name_selects_arch35():
    result = _run_isolated(
        """
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import prepare_lite_torchair

        with lite_export_context(LiteConfig("Ascend950")):
            prepare_lite_torchair()
            from torchair._utils.check_platform import is_arch35
            import torch_npu

            assert torch_npu.npu.get_device_name() == "Ascend950"
            assert is_arch35() is True
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_fake_device_name_selects_non_arch35():
    result = _run_isolated(
        """
        from torch_delegate import LiteConfig, lite_export_context
        from torch_delegate.lite_context import prepare_lite_torchair

        with lite_export_context(LiteConfig("Ascend910B")):
            prepare_lite_torchair()
            from torchair._utils.check_platform import is_arch35
            import torch_npu

            assert torch_npu.npu.get_device_name() == "Ascend910B"
            assert is_arch35() is False
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_add_export_save_and_load_roundtrip():
    result = _run_isolated(
        """
        import os
        import tempfile
        import torch
        from torch_delegate import (
            LiteConfig,
            lite_export_context,
            epair_load,
            epair_save,
            export_to_epair,
        )

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        ep = torch.export.export(Add(), (torch.randn(2, 3), torch.randn(2, 3)))
        with lite_export_context(LiteConfig("Ascend950")):
            contents = export_to_epair(ep)
        assert contents.manifest.converted_nodes == contents.manifest.total_nodes
        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        assert all(node["ascend_ir"]["source"] == "converter" for node in nodes)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as output:
            path = output.name
        try:
            epair_save(contents, path)
            loaded = epair_load(path)
            assert loaded.manifest.converted_nodes == loaded.manifest.total_nodes
        finally:
            os.unlink(path)
        """
    )
    assert result.returncode == 0, result.stderr


def test_lite_multi_aten_export_has_no_fallback():
    result = _run_isolated(
        """
        import torch
        from torch_delegate import LiteConfig, lite_export_context, export_to_epair

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.relu(torch.abs(torch.mm(x, y)))

        ep = torch.export.export(Model(), (torch.randn(2, 3), torch.randn(3, 4)))
        with lite_export_context(LiteConfig("Ascend950")):
            contents = export_to_epair(ep)
        assert contents.manifest.converted_nodes == 3
        assert contents.manifest.total_nodes == 3
        """
    )
    assert result.returncode == 0, result.stderr
