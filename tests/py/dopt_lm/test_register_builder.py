import importlib
import logging
import sys

import pytest
import torch

from dopt_lm import builder
from dopt_lm.common_tools.log import FileTxtLogger, Logger, set_record_mode, visualize_dist
from dopt_lm.common_tools.register import Register


def test_register_supports_named_and_default_registration():
    registry = Register("unit")

    @registry.register_func()
    def default_handler():
        return "default"

    @registry.register_func("alias")
    def named_handler():
        return "named"

    assert default_handler is registry.get("default_handler")
    assert named_handler is registry["alias"]
    assert "alias" in registry
    assert set(registry.get_keys()) == {"default_handler", "alias"}


def test_register_overwrites_duplicate_and_rejects_bad_values():
    registry = Register()

    @registry.register_func("same")
    def first():
        return 1

    @registry.register_func("same")
    def second():
        return 2

    assert registry.get("same")() == 2
    with pytest.raises(Exception, match="not supported missing"):
        registry.get("missing")
    with pytest.raises(Exception, match="should be callable"):
        registry.register_func()(object())


def test_file_import_imports_single_file_and_recursive_directory(tmp_path, monkeypatch):
    package = tmp_path / "ut_plugins"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (nested / "__init__.py").write_text("")
    (package / "one.py").write_text("VALUE = 1\n")
    (nested / "two.py").write_text("VALUE = 2\n")
    (nested / "skip.py").write_text("VALUE = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    Register.file_import(package / "one.py")
    Register.file_import(package, recursive=True, skip_libs_set=("skip.py",))

    assert importlib.import_module("ut_plugins.one").VALUE == 1
    assert importlib.import_module("ut_plugins.nested.two").VALUE == 2
    assert "ut_plugins.nested.skip" not in sys.modules


def test_file_import_logs_plugin_failure(tmp_path, monkeypatch):
    package = tmp_path / "broken_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    plugin = package / "bad.py"
    plugin.write_text("raise RuntimeError('plugin boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    warnings = []
    monkeypatch.setattr("dopt_lm.common_tools.register.logger.warning", warnings.append)

    Register.file_import(plugin)

    assert len(warnings) == 1
    assert "plugin boom" in str(warnings[0])


def test_init_builder_imports_builtins_and_absolute_plugins(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Register, "file_import", staticmethod(lambda path, recursive=False: calls.append((str(path), recursive))))
    (tmp_path / "plugin.py").write_text("")
    (tmp_path / "native.so").write_bytes(b"")
    (tmp_path / "ignored.txt").write_text("")

    builder.init_builder(str(tmp_path))

    assert any(path.endswith("/modules") and recursive for path, recursive in calls)
    assert any(path.endswith("/quantizer") and recursive for path, recursive in calls)
    assert (str(tmp_path / "plugin.py"), True) in calls
    assert (str(tmp_path / "native.so"), True) in calls
    assert all(not path.endswith("ignored.txt") for path, _ in calls)


def test_init_builder_rejects_relative_plugin_path():
    with pytest.raises(Exception, match="absolute path"):
        builder.init_builder("relative/plugins")


def test_file_text_logger_record_and_force(tmp_path):
    FileTxtLogger.output_path = str(tmp_path)
    FileTxtLogger.add_recoder("known", "nested/record")
    FileTxtLogger.record("known", "line-one")
    FileTxtLogger.record("missing", "ignored")
    FileTxtLogger.record("forced", "line-two", force=True)

    assert (tmp_path / "nested" / "record.txt").read_text() == "line-one\n"
    assert (tmp_path / "forced.txt").read_text() == "line-two\n"
    assert "missing" not in FileTxtLogger.loggers


def test_record_mode_and_visualize_distribution(tmp_path, monkeypatch):
    class NamedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(2, 2)
            self.block.layer_name = "block"

    FileTxtLogger.output_path = str(tmp_path)
    set_record_mode(NamedModule(), str(tmp_path))
    messages = []
    monkeypatch.setattr(Logger, "info", messages.append)
    ckpt = {
        "a.input_quantizer.s": torch.tensor(0.5),
        "a.input_quantizer.bit": torch.tensor(8),
        "a.input_quantizer.offset": torch.tensor(0.0),
        "b.quantizer.s": torch.tensor(0.25),
        "b.quantizer.bit": torch.tensor(4),
        "b.quantizer.offset": torch.tensor(1.0),
    }

    visualize_dist(ckpt)

    assert {"block", "block_out"}.issubset(FileTxtLogger.loggers)
    assert len(messages) == 2
    assert (tmp_path / "dist.txt").exists()
    assert (tmp_path / "dist_others.txt").exists()


def test_logger_suppresses_nonzero_local_rank(monkeypatch, caplog):
    monkeypatch.setenv("LOCAL_RANK", "1")
    with caplog.at_level(logging.INFO):
        Logger.info("not emitted")
    assert "not emitted" not in caplog.text
