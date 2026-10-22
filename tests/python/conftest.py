"""Pytest path bootstrap: make `import cann_lm` resolve to the bindings package
without requiring `pip install -e .` (keeps the test run dependency-free)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "cann_lm", "python"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
