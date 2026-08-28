import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOPT_PARENT = REPO_ROOT / "cann_lm" / "converter" / "quant"

# dopt_lm resolves its device and absolute imports at import time.
os.environ["DEVICE"] = "cpu"
if str(DOPT_PARENT) not in sys.path:
    sys.path.insert(0, str(DOPT_PARENT))


@pytest.fixture(autouse=True)
def _reset_dopt_global_state():
    """Keep mutable logger state from leaking between tests."""
    from dopt_lm.common_tools.log import FileTxtLogger

    old_loggers = FileTxtLogger.loggers
    old_files = FileTxtLogger.logger_files
    old_output = FileTxtLogger.output_path
    FileTxtLogger.loggers = {}
    FileTxtLogger.logger_files = {}
    FileTxtLogger.output_path = ""
    yield
    FileTxtLogger.loggers = old_loggers
    FileTxtLogger.logger_files = old_files
    FileTxtLogger.output_path = old_output
